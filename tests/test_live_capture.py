"""Check bounded local session behavior for the browser capture pilot."""

import json
from pathlib import Path
from threading import Event
from time import monotonic
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch
from stat import S_IMODE

from voice_studio.app.live_capture import FaceProcess, LiveBusy, LiveCapture


class LiveCaptureTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.service = LiveCapture(self.root)
        self.addCleanup(self.service.close)

    def test_local_secret_is_stable_and_not_public(self):
        path = self.root / "live-token.txt"
        value = path.read_text().strip()
        self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)
        self.assertTrue(self.service.authorized(value))
        self.assertFalse(self.service.authorized("x" * len(value)))
        self.assertFalse(self.service.authorized("я"))
        self.assertEqual(LiveCapture(self.root).token, value)

    def test_one_request_and_monotonic_session(self):
        ident = "a" * 32
        with patch.object(self.service, "analyze", return_value={"segments": [], "faceFrames": []}) as analyze:
            with self.assertRaises(ValueError):
                self.service.process(self.root, "bad", 1000, "ru")
            self.service.process(self.root, ident, 3000, "ru")
            with self.assertRaises(LiveBusy):
                self.service.process(self.root, ident, 3000, "ru")
            self.service.lock.acquire()
            try:
                with self.assertRaises(LiveBusy):
                    self.service.process(self.root, ident, 6000, "ru")
            finally:
                self.service.lock.release()
            analyze.assert_called_once_with(self.root, ident, 3000, "ru")

    def test_later_fragment_keeps_whisper_offsets_on_session_clock(self):
        path = self.root / "capture.webm"
        raw = [{"text": "Продолжение", "offsets": {"from": 300, "to": 3500}}]
        with patch('voice_studio.app.live_capture.decode_fragment', return_value=(path.with_suffix(".wav"), 0)), \
                patch('voice_studio.app.live_capture.audio_info', return_value=(3000, True)), \
                patch('voice_studio.app.live_capture.recognize_file', return_value=raw):
            result = self.service.analyze(path, "a" * 32, 9000, "ru")
        self.assertEqual(result["segments"], [{"startMs": 9300, "endMs": 12000, "text": "Продолжение"}])
        self.assertEqual(result["durationMs"], 3000)

    def test_face_failure_does_not_delay_or_discard_speech(self):
        """A failing face stage and speech can overlap and return independently."""
        path = self.root / "capture.webm"
        faces_started, speech_started = Event(), Event()
        raw = [{"text": "Речь", "offsets": {"from": 100, "to": 1900}}]

        def fail_faces(*_):
            """Fail after confirming the speech stage has started."""
            faces_started.set()
            self.assertTrue(speech_started.wait(2), "speech did not run while faces were in flight")
            raise ValueError("face worker failed")

        def recognize(*_, **__):
            """Return speech after the face stage has started."""
            speech_started.set()
            self.assertTrue(faces_started.wait(2), "face worker did not run alongside speech")
            return raw

        with patch('voice_studio.app.live_capture.decode_fragment', return_value=(path.with_suffix(".wav"), 8)), \
                patch('voice_studio.app.live_capture.YUNET', Mock(is_file=lambda: True)), \
                patch('voice_studio.app.live_capture.PYTHON', Mock(is_file=lambda: True)), \
                patch('voice_studio.app.live_capture.audio_info', return_value=(2000, True)), \
                patch('voice_studio.app.live_capture.recognize_file', side_effect=recognize), \
                patch.object(self.service.faces, "run", side_effect=fail_faces):
            result = self.service.analyze(path, "a" * 32, 3000, "ru")

        self.assertEqual(result["segments"], [{"startMs": 3100, "endMs": 4900, "text": "Речь"}])
        self.assertEqual(result["faceFrames"], [])
        self.assertEqual(result["warnings"], ["Лица в этом фрагменте не обработаны."])

    def test_worker_restarts_without_reusing_visible_face_numbers(self):
        faces = FaceProcess()
        self.addCleanup(faces.close)
        before, after = Mock(), Mock()
        for worker, number in ((before, 4), (after, 5)):
            worker.poll.return_value = None
            worker.stdout.readline.return_value = json.dumps({"nextNumber": number})
        with patch('voice_studio.app.live_capture.subprocess.Popen', side_effect=(before, after)), \
                patch('voice_studio.app.live_capture.select.select', return_value=([before.stdout], [], [])):
            faces.run("a" * 32, self.root, 0, monotonic() + 10)
            faces.close()
            faces.run("a" * 32, self.root, 30000, monotonic() + 10)
        self.assertEqual(json.loads(before.stdin.write.call_args.args[0])["firstNumber"], 1)
        self.assertEqual(json.loads(after.stdin.write.call_args.args[0])["firstNumber"], 4)
