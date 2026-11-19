"""Exercise native-worker ownership and failure boundaries with real child processes."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from voice_studio.core.worker_process import worker_command
from voice_studio.speech.embedding_protocol import MAX_REQUEST_BYTES, MAX_STDERR_BYTES, file_signature
from voice_studio.speech.embedding_worker import EmbeddingWorker, EmbeddingWorkerError
from voice_studio.speech.embedding_worker_server import NativeEmbeddingSession


FAKE_CHILD = r'''
import json, os, sys, time
for line in sys.stdin.buffer:
    request = json.loads(line)
    action = request['wav'].rsplit('/', 1)[-1]
    if action == 'crash': os._exit(7)
    if action == 'hang': time.sleep(10)
    if action == 'delay': time.sleep(.35)
    if action == 'stderr':
        sys.stderr.buffer.write(b'x' * 100000)
        sys.stderr.buffer.flush()
    if action == 'badjson':
        print('{oops', flush=True)
        continue
    if action == 'oversize':
        sys.stdout.buffer.write(b'x' * (4 * 1024 * 1024 + 1))
        sys.stdout.buffer.flush()
        continue
    if action == 'wrongid': request['id'] += 1
    vectors = {str(k): [[float(window[0]), float(window[1])]] for k, windows in request['spans'].items() for window in windows[:1]}
    if action == 'nan': vectors = {'1': [[float('nan')]]}
    print(json.dumps({'id': request['id'], 'vectors': vectors, 'model_loads': 1}), flush=True)
'''


class EmbeddingWorkerTests(TestCase):
    def setUp(self):
        """Keep protocol tests independent of installed native weights."""
        self.worker = EmbeddingWorker([sys.executable, '-u', '-c', FAKE_CHILD], idle_seconds=5, queue_limit=1)
        self.spans = {1: [[10, 20]], 2: [[30, 40]]}

    def tearDown(self):
        self.worker.close()

    def extract(self, action='normal', **kwargs):
        """Send a real child request using the action encoded in its WAV basename."""
        return self.worker.extract(action, self.spans, '/unused/model', {'signature': []}, **kwargs)

    def wait_started(self):
        """Wait only for a process handle, without assuming native startup timing."""
        deadline = monotonic() + 2
        while not self.worker.stats()['pid']:
            self.assertLess(monotonic(), deadline)
            sleep(.005)

    def test_repeated_calls_share_process_and_preserve_integer_keys(self):
        self.assertEqual(self.extract(), {1: [[10., 20.]], 2: [[30., 40.]]})
        pid = self.worker.stats()['pid']
        self.assertEqual(self.extract(), {1: [[10., 20.]], 2: [[30., 40.]]})
        self.assertEqual(self.worker.stats()['pid'], pid)
        self.assertEqual(self.worker.stats()['starts'], 1)
        self.assertEqual(self.worker.stats()['requests'], 2)
        self.assertEqual(self.worker.stats()['model_loads'], 1)

    def test_concurrent_calls_are_correlated_and_serialized(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.extract, 'delay')
            self.wait_started()
            second = pool.submit(self.worker.extract, 'normal', {3: [[50, 60]]}, '/unused', {})
            self.assertEqual(first.result(timeout=2), {1: [[10., 20.]], 2: [[30., 40.]]})
            self.assertEqual(second.result(timeout=2), {3: [[50., 60.]]})
        self.assertEqual(self.worker.stats()['starts'], 1)

    def test_queue_wait_deadline_does_not_kill_active_request(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.extract, 'delay')
            self.wait_started()
            before = monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                self.extract(timeout=.04)
            self.assertLess(monotonic() - before, .25)
            self.assertEqual(first.result(timeout=2)[1], [[10., 20.]])
        self.assertEqual(self.worker.stats()['starts'], 1)

    def test_queue_admission_is_bounded(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.extract, 'delay')
            self.wait_started()
            queued = pool.submit(self.extract, 'normal')
            # The admitted waiter holds the only spare slot before it acquires the serial lock.
            deadline = monotonic() + 1
            while self.worker._admission._value != 0:
                self.assertLess(monotonic(), deadline)
                sleep(.002)
            with self.assertRaisesRegex(EmbeddingWorkerError, 'queue is full'):
                self.extract()
            first.result(timeout=2)
            queued.result(timeout=2)

    def test_failures_reap_process_and_restart_only_on_next_request(self):
        for action in ('crash', 'badjson', 'oversize', 'wrongid', 'nan', 'hang'):
            with self.subTest(action=action):
                starts = self.worker.stats()['starts']
                with self.assertRaises((EmbeddingWorkerError, ValueError, subprocess.TimeoutExpired)):
                    self.extract(action, timeout=.15 if action == 'hang' else 2)
                self.assertIsNone(self.worker.stats()['pid'])
                self.assertEqual(self.worker.stats()['starts'], starts + 1)
                self.assertEqual(self.extract()[1], [[10., 20.]])
                self.assertEqual(self.worker.stats()['starts'], starts + 2)
                self.worker.close()

    def test_idle_and_explicit_close_release_process(self):
        self.worker.idle_seconds = .05
        self.extract()
        child = self.worker._child.process
        deadline = monotonic() + 2
        while self.worker.stats()['pid'] is not None:
            self.assertLess(monotonic(), deadline)
            sleep(.01)
        self.assertIsNotNone(child.poll())
        self.extract()
        child = self.worker._child.process
        self.worker.close()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(self.worker.stats()['pid'])

    def test_close_interrupts_a_running_request(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.extract, 'hang', timeout=4)
            self.wait_started()
            child = self.worker._child.process
            self.worker.close()
            with self.assertRaises((EmbeddingWorkerError, OSError)):
                pending.result(timeout=2)
        self.assertIsNotNone(child.poll())

    def test_buffered_stdin_backpressure_is_interrupted_by_timeout_and_close(self):
        """A near-limit write to an unread pipe must not hold buffered stream locks forever."""
        for close_explicitly in (False, True):
            with self.subTest(close_explicitly=close_explicitly):
                worker = EmbeddingWorker([sys.executable, '-c', 'import time; time.sleep(10)'])
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending = pool.submit(worker.extract, 'unused', self.spans, '/unused',
                                              {'large': 'x' * (MAX_REQUEST_BYTES - 1024)}, timeout=.25)
                        deadline = monotonic() + 2
                        while worker.stats()['pid'] is None:
                            self.assertLess(monotonic(), deadline)
                            sleep(.002)
                        child = worker._child.process
                        if close_explicitly:
                            worker.close()
                        with self.assertRaises((EmbeddingWorkerError, subprocess.TimeoutExpired, OSError)):
                            pending.result(timeout=2)
                    self.assertIsNotNone(child.poll())
                    self.assertIsNone(worker.stats()['pid'])
                finally:
                    worker.close()

    def test_stderr_is_drained_and_tail_is_bounded(self):
        self.extract('stderr')
        self.worker.close()
        self.assertEqual(len(self.worker.stats()['stderr_tail']), MAX_STDERR_BYTES)

    def test_invalid_and_oversized_inputs_never_start_child(self):
        for spans in ({1: [[-1, 20]]}, {1: [[2, 1]]}, {1: [[0, 10]] * 73}, {i: [] for i in range(25)}):
            with self.assertRaises(ValueError):
                self.worker.extract('normal', spans, '/unused', {})
        with self.assertRaisesRegex(ValueError, 'exceeds limit'):
            self.worker.extract('normal', self.spans, '/unused', {'large': 'x' * MAX_REQUEST_BYTES})
        self.assertEqual(self.worker.stats()['starts'], 0)

    def test_nonfinite_intervals_and_noninteger_queue_limits_are_rejected(self):
        """Operator/direct-call limits must remain finite and bounded before a child exists."""
        for interval in (float('nan'), float('inf'), float('-inf'), 0):
            with self.subTest(interval=interval):
                with self.assertRaises(ValueError):
                    EmbeddingWorker(idle_seconds=interval)
                with self.assertRaises(ValueError):
                    self.extract(timeout=interval)
        for limit in (True, .5, float('inf'), -1):
            with self.subTest(queue_limit=limit), self.assertRaises(ValueError):
                EmbeddingWorker(queue_limit=limit)
        self.assertEqual(self.worker.stats()['starts'], 0)


class NativeEmbeddingSessionTests(TestCase):
    def test_real_server_process_reloads_model_identity(self):
        """Run the production server/cache in a real child with a lightweight native stand-in."""
        code = '''
from voice_studio.speech import embedding_worker_server as server
original = server.NativeEmbeddingSession
def factory(path):
    return object()
def infer(wav, spans, extractor):
    return {number: [[1., 0.]] for number in spans}
server.NativeEmbeddingSession = lambda: original(factory, infer)
server.serve()
'''
        with TemporaryDirectory() as temporary:
            weights = Path(temporary) / 'embedding.onnx'
            weights.write_bytes(b'first')
            worker = EmbeddingWorker([sys.executable, '-u', '-c', code])
            try:
                identity = {'signature': file_signature(weights), 'configuration': 'a'}
                for _ in range(2):
                    self.assertEqual(worker.extract('/unused', {1: [[0, 1000]]}, weights, identity), {1: [[1., 0.]]})
                self.assertEqual(worker.stats()['model_loads'], 1)
                identity['configuration'] = 'b'
                worker.extract('/unused', {1: [[0, 1000]]}, weights, identity)
                self.assertEqual(worker.stats()['model_loads'], 2)
                weights.write_bytes(b'replacement')
                identity['signature'] = file_signature(weights)
                worker.extract('/unused', {1: [[0, 1000]]}, weights, identity)
                self.assertEqual(worker.stats()['model_loads'], 3)
                self.assertEqual(worker.stats()['starts'], 1)
            finally:
                worker.close()

    def test_cache_reuses_then_reloads_for_config_path_and_file_signature(self):
        with TemporaryDirectory() as temporary:
            first = Path(temporary) / 'first.onnx'
            second = Path(temporary) / 'second.onnx'
            first.write_bytes(b'original')
            second.write_bytes(b'other')
            loaded = []

            def factory(path):
                """Record model loads while returning a distinct opaque extractor."""
                token = object()
                loaded.append((path, token))
                return token

            def infer(wav, spans, extractor):
                """Verify the injected extractor reaches the unchanged kernel boundary."""
                self.assertIs(extractor, loaded[-1][1])
                return {1: [[1., 0.]]}

            session = NativeEmbeddingSession(factory, infer)
            request = {'id': 1, 'wav': '/unused', 'spans': {1: [[0, 1000]]},
                       'model_path': str(first), 'identity': {'signature': file_signature(first), 'configuration': 'a'}}
            session.execute(request)
            session.execute(request)
            self.assertEqual(session.model_loads, 1)
            request['identity']['configuration'] = 'b'
            session.execute(request)
            self.assertEqual(session.model_loads, 2)
            first.write_bytes(b'changed-content')
            request['identity']['signature'] = file_signature(first)
            session.execute(request)
            self.assertEqual(session.model_loads, 3)
            request['model_path'] = str(second)
            request['identity']['signature'] = file_signature(second)
            session.execute(request)
            self.assertEqual(session.model_loads, 4)

    def test_resource_change_during_load_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.onnx'
            path.write_bytes(b'original')

            def factory(weights):
                """Simulate a concurrent installer replacing the selected weights."""
                weights.write_bytes(b'replaced')
                return object()

            session = NativeEmbeddingSession(factory, lambda *args, **kwargs: {})
            request = {'id': 1, 'wav': '/unused', 'spans': {}, 'model_path': str(path),
                       'identity': {'signature': file_signature(path)}}
            with self.assertRaisesRegex(ValueError, 'while loading'):
                session.execute(request)
            self.assertIsNone(session.extractor)

    def test_child_server_exits_on_eof_invalid_json_and_parent_loss(self):
        command = worker_command(sys.executable, 'voice_studio.speech.speaker_review', '--embedding-worker')
        with TemporaryDirectory() as cwd:
            # Native app launches need the packaged worker from an arbitrary directory.
            fake_package = Path(cwd) / "voice_studio"
            fake_package.mkdir()
            (fake_package / "__init__.py").write_text("raise RuntimeError('wrong worker package')\n")
            environment = dict(os.environ, PYTHONPATH=cwd)
            for wire in (b'', b'{invalid}\n', b'x' * (MAX_REQUEST_BYTES + 1)):
                process = subprocess.run(command, cwd=cwd, env=environment, input=wire, capture_output=True, timeout=3)
                self.assertEqual(process.returncode, 0, process.stderr)
                if wire:
                    self.assertIn('error', json.loads(process.stdout))
            environment['VOICE_EMBEDDING_PARENT_PID'] = '-1'
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(process.wait(timeout=3), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=2)

    def test_resident_resolves_resource_in_parent_without_mutating_snapshot(self):
        from voice_studio.speech import speaker_review
        with TemporaryDirectory() as temporary:
            weights = Path(temporary) / 'embedding.onnx'
            weights.write_bytes(b'test')
            environment = {'VOICE_SPEAKER_EMBEDDING_MODE': 'resident', 'VOICE_MOD_SNAPSHOT': 'unchanged'}
            with patch.dict(os.environ, environment), patch.object(speaker_review, 'model_resource', return_value=weights), \
                    patch.object(speaker_review, 'current_snapshot') as snapshot, \
                    patch('voice_studio.speech.embedding_worker.get_worker') as worker:
                snapshot.return_value.fingerprint.return_value = 'frozen-config'
                worker.return_value.extract.return_value = {1: [[1., 0.]]}
                self.assertEqual(speaker_review.extract_embeddings.__wrapped__('audio.wav', {1: [[0, 1000]]}), {1: [[1., 0.]]})
                arguments = worker.return_value.extract.call_args.args
                self.assertEqual(arguments[2], weights.resolve())
                self.assertEqual(arguments[3]['configuration'], 'frozen-config')
                self.assertEqual(os.environ['VOICE_MOD_SNAPSHOT'], 'unchanged')


class EmbeddingModCompatibilityTests(TestCase):
    def setUp(self):
        """Use a real approved provider package and isolated registry, without native weights."""
        from voice_studio import mod_runtime
        self.runtime = mod_runtime
        self.temporary = TemporaryDirectory()
        self.environment = patch.dict(os.environ, {'VOICE_MEDIA_MODS_DIR': str(Path(self.temporary.name) / 'mods'),
                                                  'VOICE_SPEAKER_EMBEDDING_MODE': 'resident'})
        self.environment.start()
        self.old_registry = mod_runtime.REGISTRY
        mod_runtime.REGISTRY = None
        mod_runtime.RESOURCE_CACHE.clear()
        self.registry = mod_runtime.registry()
        folder = self.registry.directory / 'test.embedding'
        folder.mkdir(parents=True)
        manifest = {'apiVersion': 1, 'id': 'test.embedding', 'name': 'Test embedding', 'version': '1.0.0',
                    'description': 'Test', 'capabilities': [{'id': 'speech.embedding', 'kind': 'provider', 'entrypoint': 'worker.py:run'}],
                    'settingsSchema': {'type': 'object', 'properties': {}}, 'settings': {}, 'license': 'Test',
                    'provenance': {'source': 'local test'}}
        (folder / 'mod.json').write_text(json.dumps(manifest))
        (folder / 'worker.py').write_text("def run(payload, settings):\n    return {str(number): [[1., 0.]] for number in payload['spans']}\n")
        self.fingerprint = next(row['fingerprint'] for row in self.registry.catalog()['mods'] if row['id'] == 'test.embedding')
        self.registry.enable('test.embedding', self.fingerprint)
        self.registry.select('speech.embedding', 'test.embedding')

    def tearDown(self):
        self.runtime.REGISTRY = self.old_registry
        self.runtime.RESOURCE_CACHE.clear()
        self.environment.stop()
        self.temporary.cleanup()

    def test_custom_embedding_provider_keeps_named_payload_and_integer_key_abi(self):
        from voice_studio.speech.speaker_review import extract_embeddings
        with patch('voice_studio.speech.embedding_worker.get_worker') as worker:
            self.assertEqual(extract_embeddings('unused.wav', {7: [[0, 1000]]}), {7: [[1., 0.]]})
        worker.assert_not_called()

    def test_disabled_pinned_provider_does_not_start_native_fallback(self):
        from voice_studio.mods import ModError
        from voice_studio.speech.speaker_review import extract_embeddings
        self.registry.enable('test.embedding', self.fingerprint, False)
        with patch('voice_studio.speech.embedding_worker.get_worker') as worker, self.assertRaises(ModError):
            extract_embeddings('unused.wav', {7: [[0, 1000]]})
        worker.assert_not_called()
