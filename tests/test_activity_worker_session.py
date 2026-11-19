"""Check bounded native-window transport without loading models or media."""

from io import BytesIO
from threading import Event, Thread, enumerate as threads
from unittest import TestCase
from unittest.mock import patch

from voice_studio.vision.activity_worker_session import ActivityWorkerSession

TARGET = 'voice_studio.vision.activity_worker_session'


class FakeChild:
    """Release a blocked fake pipe when the owned child is killed."""

    def __init__(self, response=b'{"result": {"frames": []}}\n', *, blocked=None):
        self.stdin = BytesIO()
        self.stdout = BytesIO(response)
        self.entered, self.released = Event(), Event()
        self.killed = False
        if blocked:
            stream = self.stdin if blocked == 'write' else self.stdout
            original = stream.write if blocked == 'write' else stream.readline

            def wait_for_child(*arguments):
                """Simulate a native child that waits until termination closes its pipe."""
                self.entered.set()
                if not self.released.wait(2):
                    raise TimeoutError('Test fake pipe was not released')
                return original(*arguments)

            setattr(stream, 'write' if blocked == 'write' else 'readline', wait_for_child)

    def poll(self):
        """Report this fake child as live until its owner retires it."""
        return -9 if self.killed else None

    def kill(self):
        """Unblock test IO without signalling an operating-system process."""
        self.killed = True
        self.released.set()

    def wait(self, **_):
        """Return the completed fake child status."""
        return -9


class ActivityWorkerSessionTests(TestCase):
    def run_request(self, child, timeout=.04):
        """Connect a fake owned child while bypassing unrelated mod configuration."""
        with patch(TARGET + '.worker_environment', side_effect=lambda value: value), \
                patch(TARGET + '.subprocess.Popen', return_value=child):
            with ActivityWorkerSession('python', 'worker.py', []) as worker:
                return worker.request(['--window', '0'], timeout)

    def test_write_timeout_retires_child_and_joins_io(self):
        """A full stdin pipe consumes the same deadline as reading a reply."""
        child = FakeChild(blocked='write')
        finished = Event()
        errors = []

        def request():
            """Capture deadline delivery without letting a regression hang the suite."""
            try:
                self.run_request(child)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        caller = Thread(target=request)
        caller.start()
        self.assertTrue(child.entered.wait(1))
        within_deadline = finished.wait(.4)
        child.released.set()
        caller.join(2)
        self.assertFalse(caller.is_alive())
        self.assertTrue(within_deadline, 'stdin write bypassed the request deadline')
        self.assertIsInstance(errors[0], TimeoutError)
        self.assertTrue(child.killed)
        self.assertFalse(any(item.name.startswith('light-asd-window') for item in threads()))

    def test_read_timeout_retires_child_and_joins_io(self):
        """A stalled reply is stopped and no session reader thread survives exit."""
        child = FakeChild(blocked='read')
        with self.assertRaises(TimeoutError):
            self.run_request(child)
        self.assertTrue(child.killed)
        self.assertFalse(any(item.name.startswith('light-asd-window') for item in threads()))

    def test_protocol_failures_are_recoverable_and_retire_child(self):
        """Malformed envelopes and results cannot escape as unrelated Python errors."""
        for response in (b'null\n', b'[]\n', b'"result"\n', b'{}\n',
                         b'{"error": "failed"}\n', b'{"result": []}\n',
                         b'{"result": null}\n', b'{broken}\n', b'{}'):
            with self.subTest(response=response):
                child = FakeChild(response)
                with self.assertRaises(ValueError):
                    self.run_request(child)
                self.assertTrue(child.killed)

    def test_protocol_failure_replaces_child_for_next_window(self):
        """An invalid window cannot contaminate the next independently attempted one."""
        invalid, valid = FakeChild(b'{}\n'), FakeChild()
        with patch(TARGET + '.worker_environment', side_effect=lambda value: value), \
                patch(TARGET + '.subprocess.Popen', side_effect=[invalid, valid]) as spawn:
            with ActivityWorkerSession('python', 'worker.py', []) as worker:
                with self.assertRaises(ValueError):
                    worker.request([], 1)
                self.assertTrue(invalid.killed)
                self.assertEqual(worker.request([], 1), {'frames': []})
                self.assertEqual(spawn.call_count, 2)

    def test_success_reuses_one_child_and_returns_dict(self):
        """Two completed windows share a model process and keep original result shape."""
        child = FakeChild(b'{"result": {"window": 1}}\n{"result": {"window": 2}}\n')
        with patch(TARGET + '.worker_environment', side_effect=lambda value: value), \
                patch(TARGET + '.subprocess.Popen', return_value=child) as spawn:
            with ActivityWorkerSession('python', 'worker.py', []) as worker:
                self.assertEqual(worker.request([], 1), {'window': 1})
                self.assertEqual(worker.request([], 1), {'window': 2})
                self.assertEqual(spawn.call_count, 1)
        self.assertTrue(child.killed)

    def test_invalid_limits_are_rejected_before_spawn(self):
        """Invalid deadlines and oversized requests never allocate a native child."""
        with patch(TARGET + '.subprocess.Popen') as spawn:
            with ActivityWorkerSession('python', 'worker.py', []) as worker:
                for timeout in (0, -1, float('nan'), float('inf')):
                    with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                        worker.request([], timeout)
                with self.assertRaises(ValueError):
                    worker.request(['x' * (64 * 1024)], 1)
            spawn.assert_not_called()

    def test_flush_timeout_uses_the_same_deadline(self):
        """Buffered stdin flush cannot bypass the transport deadline."""
        child = FakeChild()

        def blocked_flush():
            """Wait until test child retirement releases its pipe."""
            child.entered.set()
            if not child.released.wait(2):
                raise TimeoutError('Test flush was not released')

        child.stdin.flush = blocked_flush
        with self.assertRaises(TimeoutError):
            self.run_request(child)
        self.assertTrue(child.killed)

    def test_oversized_and_exact_limit_responses(self):
        """The byte limit includes the newline and accepts exactly its boundary."""
        prefix, suffix = b'{"result":{"text":"', b'"}}\n'
        limit = 128
        boundary = prefix + b'x' * (limit - len(prefix) - len(suffix)) + suffix
        with patch(TARGET + '.MAX_RESPONSE_BYTES', limit):
            self.assertIsInstance(self.run_request(FakeChild(boundary)), dict)
            oversized = FakeChild(boundary[:-1] + b' \n')
            with self.assertRaises(ValueError):
                self.run_request(oversized)
            self.assertTrue(oversized.killed)

    def test_io_failure_and_cancellation_retire_only_owned_child(self):
        """Transport failures and interrupted callers cannot leave a reusable child."""
        for error in (BrokenPipeError('closed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                child = FakeChild()
                with patch.object(child.stdin, 'write', side_effect=error):
                    with self.assertRaises(type(error)):
                        self.run_request(child)
                self.assertTrue(child.killed)
                self.assertFalse(any(item.name.startswith('light-asd-window') for item in threads()))

    def test_broken_stdin_close_still_closes_stdout_and_joins_io(self):
        """A buffered pipe's close error must not interrupt owned cleanup."""
        child = FakeChild()
        with patch.object(child.stdin, 'close', side_effect=BrokenPipeError('buffered flush failed')):
            self.assertEqual(self.run_request(child), {'frames': []})
        self.assertTrue(child.stdout.closed)
        self.assertFalse(any(item.name.startswith('light-asd-window') for item in threads()))
