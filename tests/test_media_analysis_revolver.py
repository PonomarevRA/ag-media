"""Verify restart, dependency invalidation and publication of usable checkpoints."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_analysis_revolver import AnalysisRevolver, analysis_fingerprint
from voice_studio.graph.media_graph_ingest import SPEECH_WINDOW_MS, analyze_windows


class AnalysisRevolverTests(TestCase):
    def setUp(self):
        """Synthetic speech fixtures do not write PCM for the independent audio layer."""
        audio_layers = patch('voice_studio.graph.media_graph_ingest.analyze_audio_scene',
                             return_value={"audioLayers": [], "soundAnalysisStatus": "done"})
        audio_layers.start()
        self.addCleanup(audio_layers.stop)

    def test_last_window_failure_restarts_without_replaying_five_windows(self):
        """Simulate one hour of audio with a failure in the last ten-minute partition."""
        duration = 6 * SPEECH_WINDOW_MS
        metadata = {"kind": "audio", "durationMs": duration,
                    "streams": [{"kind": "audio", "index": 0, "startMs": 0}]}
        decoded = []
        attempt = 0

        def extract(source, work, start, end, audio):
            decoded.append(start)
            return work / str(start), 0, end - start

        def recognize(wav, *args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt == 6:
                raise RuntimeError("late failure")
            return {"segments": [{"startMs": 0, "endMs": 1000, "text": wav.name, "speaker": 1}]}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio.wav").write_bytes(b"checkpoint audio source fixture")
            with patch('voice_studio.graph.media_graph_ingest.extract_speech', side_effect=extract), \
                    patch('voice_studio.graph.media_graph_ingest.transcribe_wav', side_effect=recognize):
                full = [root, root / "audio.wav", metadata, "ru", False, False, None]
                with self.assertRaisesRegex(RuntimeError, "late failure"):
                    analyze_windows(*full, revolver=AnalysisRevolver(root / "cache"))
                decoded.clear()
                restarted = AnalysisRevolver(root / "cache")
                actual = analyze_windows(*full, revolver=restarted)
                self.assertEqual(decoded, [5 * SPEECH_WINDOW_MS])
                self.assertEqual(restarted.counts["reused"], 10)
                self.assertEqual(actual, analyze_windows(*full))

    def test_stage_namespace_changes_after_recovered_diarization(self):
        """A failed voice result must never overwrite newly computed speech via the face cache."""
        duration = 65_000
        metadata = {"kind": "video", "durationMs": duration,
                    "streams": [{"kind": "audio", "index": 0, "startMs": 0}]}
        old = {"segments": [{"startMs": 0, "endMs": 1000, "text": "before"}], "speakerWarning": "busy"}
        new = {"segments": [{"startMs": 0, "endMs": 1000, "text": "after", "speaker": 1}]}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").write_bytes(b"recovered diarization source fixture")
            with patch('voice_studio.graph.media_graph_ingest.extract_speech', return_value=(root / "mock.wav", 0, duration)), \
                    patch('voice_studio.graph.media_graph_ingest.transcribe_wav', side_effect=[old, new]) as asr, \
                    patch('voice_studio.graph.media_graph_ingest.visual_clip'), \
                    patch('voice_studio.graph.media_graph_ingest.collect_faces', return_value={"stages": {"light-asd": "ok"}}) as face, \
                    patch('voice_studio.graph.media_graph_ingest.scan_shots', return_value=[]):
                args = [root, root / "video.mp4", metadata, "ru", True, False, None]
                checkpoint = root / "cache"
                first = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                second = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                third = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                self.assertEqual(first["segments"][0]["text"], "before")
                self.assertEqual(second["segments"][0]["text"], "after")
                self.assertEqual(second, third)
                self.assertEqual((asr.call_count, face.call_count), (2, 2))
                coverage = second["coverage"]["face"][-1]
                self.assertEqual((coverage["startMs"], coverage["endMs"], coverage["status"]),
                                 (0, duration, "unavailable"))

    def test_failed_scene_retries_without_redecoding_speech_or_faces(self):
        """The nested vision group retries NanoDet where speech and face results survived."""
        duration = 65_000
        metadata = {"kind": "video", "durationMs": duration,
                    "streams": [{"kind": "audio", "index": 0, "startMs": 0}]}
        speech = {"segments": [{"startMs": 0, "endMs": 1000, "text": "cue", "speaker": 1}]}
        failed = {"status": "failed"}
        done = {"status": "ok", "frames": [], "stages": {"nanodet": "ok"}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").write_bytes(b"scene retry source fixture")
            with patch('voice_studio.graph.media_graph_ingest.extract_speech', return_value=(root / "mock.wav", 0, duration)) as audio, \
                    patch('voice_studio.graph.media_graph_ingest.transcribe_wav', return_value=speech) as asr, \
                    patch('voice_studio.graph.media_graph_ingest.visual_clip'), \
                    patch('voice_studio.graph.media_graph_ingest.collect_faces', return_value={"stages": {"light-asd": "ok"}}) as face, \
                    patch('voice_studio.graph.media_graph_ingest.scan_shots', return_value=[]), \
                    patch('voice_studio.graph.media_graph_ingest.review_scene', side_effect=[failed, done]) as scene:
                args = [root, root / "video.mp4", metadata, "ru", True, True, None]
                checkpoint = root / "cache"
                first = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                second = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                third = analyze_windows(*args, revolver=AnalysisRevolver(checkpoint))
                self.assertEqual(first["coverage"]["scene"][0]["status"], "failed")
                self.assertEqual(second["coverage"]["scene"][0]["status"], "sampled")
                self.assertEqual(second, third)
                self.assertEqual((audio.call_count, asr.call_count, face.call_count, scene.call_count), (1, 1, 1, 2))

    def test_invalid_checkpoint_recomputes_and_can_be_replaced(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.json").write_text('{"unrelated": true}')
            revolver = AnalysisRevolver(root)
            self.assertIsNone(revolver.load("scene", required=("frames",)))
            revolver.save("scene", {"frames": []})
            self.assertEqual(revolver.load("scene", required=("frames",)), {"frames": []})

    def test_fingerprint_changes_with_ocr_backend_and_weights(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "whisper.bin"
            model.write_bytes(b"v1")
            with patch('voice_studio.graph.media_analysis_revolver.MODEL', model), patch.dict("os.environ", {"VOICE_SCENE_OCR": "auto"}):
                first = analysis_fingerprint("same-source", {"scene": True})
                with patch.dict("os.environ", {"VOICE_SCENE_OCR": "tesseract"}):
                    self.assertNotEqual(first, analysis_fingerprint("same-source", {"scene": True}))
                model.write_bytes(b"version-two")
                self.assertNotEqual(first, analysis_fingerprint("same-source", {"scene": True}))

    def test_fingerprint_changes_when_nested_source_changes(self):
        """A stage cannot reuse a checkpoint from an older version of a packaged worker."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "voice_studio/audio/native/audio_scene.swift"
            helper.parent.mkdir(parents=True)
            helper.write_text("// Original helper")
            worker = root / "voice_studio/vision/tracking.py"
            worker.parent.mkdir(parents=True)
            worker.write_text("LIMIT = 1\n")
            with patch("voice_studio.graph.media_analysis_revolver.ROOT", root):
                first = analysis_fingerprint("same-source", {})
                worker.write_text("LIMIT = 2\n")
                updated_worker = analysis_fingerprint("same-source", {})
                self.assertNotEqual(first, updated_worker)
                helper.write_text("// Updated helper")
                self.assertNotEqual(updated_worker, analysis_fingerprint("same-source", {}))

    def test_fingerprint_changes_with_face_clothing_and_audio_modes(self):
        """Each stage mode must invalidate previous results when the source stays the same."""
        modes = {
            "VOICE_YUNET_SAMPLING": "dense",
            "VOICE_PERSON_FACE_PILOT": "1",
            "VOICE_CLOTHING_PROFILE": "accurate",
            "VOICE_AUDIO_CLASSIFIER": "zipformer",
        }
        with patch.dict("os.environ", dict.fromkeys(modes, "")):
            first = analysis_fingerprint("same-source", {})
            for key, mode in modes.items():
                with self.subTest(key=key), patch.dict("os.environ", {key: mode}):
                    self.assertNotEqual(first, analysis_fingerprint("same-source", {}))

    def test_people_count_switch_separates_checkpoint_keys(self):
        default = analysis_fingerprint("same-source", {"peopleCount": True})
        self.assertNotEqual(default, analysis_fingerprint("same-source", {"peopleCount": False}))
