"""Check progressive analysis ordering and backward-compatible NDJSON completion."""

from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import wave

from voice_studio.graph.media_graph_ingest import SPEECH_WINDOW_MS, analyze_windows
from voice_studio.speech.media_transcript import transcribe_media
from voice_studio.app.server import Handler


def pcm_fixture(path):
    """Write one second of PCM for a deterministic recognition stub."""
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))


class ProgressiveAnalysisTests(TestCase):
    def test_experimental_languages_are_routed_only_for_clip_uploads(self):
        """The full recording assembled by live capture keeps the existing language policy."""
        handler = Mock()
        handler.request_scope.return_value = nullcontext(True)
        handler.language.side_effect = lambda query, **options: Handler.language(handler, query, **options)
        handler.transcription_language.side_effect = (
            lambda query, fallback, **options: Handler.transcription_language(handler, query, fallback, **options)
        )
        handler.path = "/api/transcribe-media?lang=ru&speech_lang=ps"
        Handler.do_POST(handler)
        self.assertEqual(handler.receive_media.call_args.args, ("ps",))
        self.assertTrue(handler.receive_media.call_args.kwargs["clip_languages"])
        self.assertEqual(handler.receive_media.call_args.kwargs["object_sampling"], "small-detail")

        handler.receive_media.reset_mock()
        handler.path = "/api/transcribe-capture?lang=ru&speech_lang=ps"
        with self.assertRaisesRegex(ValueError, "русский или английский"):
            Handler.do_POST(handler)
        handler.receive_media.assert_not_called()

        handler.path = "/api/transcribe-capture?lang=ru&speech_lang=auto&speakers=1"
        Handler.do_POST(handler)
        self.assertEqual(handler.receive_media.call_args.args, ("auto",))
        self.assertFalse(handler.receive_media.call_args.kwargs["clip_languages"])
        self.assertTrue(handler.receive_media.call_args.kwargs["speakers"])

    def test_speech_audio_and_objects_are_published_in_order_with_one_decode(self):
        """Keep the ordinary final result and do not re-run ASR while publishing layers."""
        order = []
        published = {}
        audio_detail = {"audioLayers": [{"kind": "music", "startMs": 0, "endMs": 1000}]}
        cup = {"label": "cup", "box": [.1, .2, .1, .15], "score": .91}
        scene_detail = {"status": "ok", "frames": [{"timeMs": 0, "objects": [cup], "text": []}]}

        def publish(phase, report):
            order.append(phase)
            published[phase] = deepcopy(report)

        def decode(command, **kwargs):
            pcm_fixture(Path(command[-1]))

        def classify(*args):
            self.assertEqual(order, ["base"])
            self.assertIn("фраза", published["base"]["srt"])
            self.assertNotIn("audioLayers", published["base"])
            return audio_detail

        def scene(*args, **kwargs):
            self.assertEqual(order, ["base", "audio"])
            return {**scene_detail, "shotColors": [{"startMs": 0, "average": "#123456", "sampleCount": 1}]}

        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "media").write_bytes(b"source fixture")
            with patch('voice_studio.speech.media_transcript.audio_offset', return_value=0), \
                    patch('voice_studio.speech.media_transcript.ffmpeg', side_effect=decode) as ffmpeg, \
                    patch('voice_studio.speech.media_transcript.recognize_and_diarize',
                          return_value=([{"text": "фраза", "offsets": {"from": 0, "to": 500}}], "ru", [], None)
                          ) as recognize, \
                    patch('voice_studio.speech.media_transcript.analyze_audio_scene', side_effect=classify) as audio, \
                    patch('voice_studio.speech.media_transcript.review_scene', side_effect=scene) as video:
                final = transcribe_media(root, "ru", root / "media", scene=True, publish=publish)
        self.assertEqual(order, ["base", "audio", "scene"])
        self.assertEqual(published["audio"], audio_detail)
        self.assertEqual(published["scene"]["sceneEvidence"], scene_detail)
        self.assertNotIn("objectTimeline", published["base"])
        self.assertEqual([clip["label"] for clip in published["scene"]["objectTimeline"]], ["cup"])
        self.assertEqual(published["scene"]["objectTimeline"], final["objectTimeline"])
        self.assertEqual(final["segments"], published["base"]["segments"])
        self.assertEqual(final["audioLayers"], audio_detail["audioLayers"])
        for worker in (ffmpeg, recognize, audio, video):
            worker.assert_called_once()

    def test_ndjson_reports_base_before_error_and_preserves_source_identity(self):
        """A late error leaves a usable early transcript on the already-flushed stream."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "media"
            path.write_bytes(b"fixture")
            handler = Mock()
            handler.wfile = BytesIO()
            handler.uploaded_media.return_value = nullcontext(path)
            handler.server = SimpleNamespace(studio=SimpleNamespace(root=path.parent))
            handler.stream_analysis.side_effect = lambda analyze: Handler.stream_analysis(handler, analyze)

            def fail_after_base(*args, publish, **kwargs):
                publish("base", {"segments": [{"text": "early"}]})
                ready = [json.loads(row) for row in handler.wfile.getvalue().splitlines()]
                self.assertEqual(ready[-1]["partial"]["phase"], "base")
                publish("audio", {"audioLayers": []})
                raise ValueError("late object recognition error")

            with patch('voice_studio.app.server.transcribe_uploaded_media', side_effect=fail_after_base):
                Handler.receive_media(handler, "en", progress=True, context=True)

        events = [json.loads(row) for row in handler.wfile.getvalue().splitlines()]
        self.assertEqual(events, [
            {"stage": "upload", "state": "done"},
            {"partial": {"phase": "base", "report": {
                "segments": [{"text": "early"}], "sourceDigest": sha256(b"fixture").hexdigest(),
            }}},
            {"partial": {"phase": "detail", "layer": "audio", "report": {"audioLayers": []}}},
            {"error": "late object recognition error"},
        ])

    def test_ndjson_delivers_final_result_after_details(self):
        """Keep the established result envelope for progressive clients."""
        handler = Mock()
        handler.wfile = BytesIO()
        report = {"segments": [{"text": "final"}], "audioLayers": []}

        def analyze(*, emit):
            emit({"partial": {"phase": "base", "report": {"segments": report["segments"]}}})
            emit({"partial": {"phase": "detail", "layer": "audio", "report": {"audioLayers": []}}})
            return report

        Handler.stream_analysis(handler, analyze)
        events = [json.loads(row) for row in handler.wfile.getvalue().splitlines()]
        self.assertEqual([event["partial"]["phase"] for event in events[1:-1]], ["base", "detail"])
        self.assertEqual(events[-1], {"result": report})

    def test_json_api_still_returns_one_final_response(self):
        """Legacy API requests do not emit progressive records."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "media").write_bytes(b"JSON upload source fixture")
            handler = Mock()
            handler.uploaded_media.return_value = nullcontext(root / "media")
            handler.server = SimpleNamespace(studio=SimpleNamespace(root=root))
            final = {"segments": [{"text": "final"}], "audioLayers": []}
            with patch('voice_studio.app.server.transcribe_uploaded_media', return_value=final) as analyze:
                Handler.receive_media(handler, "en")
        handler.reply.assert_called_once_with(200, final)
        handler.stream_analysis.assert_not_called()
        self.assertIsNone(analyze.call_args.kwargs["publish"])

    def test_windowed_speech_and_audio_use_the_same_contiguous_media_clock(self):
        """Long sources expose an early window and only decode audio once per window."""
        duration = SPEECH_WINDOW_MS + 3500
        metadata = {"kind": "audio", "durationMs": duration,
                    "streams": [{"kind": "audio", "index": 0, "startMs": 0, "durationMs": duration}]}
        published = []

        def publish(phase, report):
            published.append((phase, deepcopy(report)))

        def decode(source, work, start, end, stream):
            return work / "audio.wav", 0, end - start

        speech = {"segments": [{"startMs": 200, "endMs": 800, "text": "sample", "speaker": 1}]}
        layer = {"soundAnalysisStatus": "done",
                 "audioLayers": [{"kind": "music", "label": "music", "startMs": 100, "endMs": 900}]}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "media").write_bytes(b"windowed progressive source fixture")
            with patch('voice_studio.graph.media_graph_ingest.extract_speech', side_effect=decode) as ffmpeg, \
                    patch('voice_studio.graph.media_graph_ingest.transcribe_wav', return_value=speech) as recognize, \
                    patch('voice_studio.graph.media_graph_ingest.analyze_audio_scene', return_value=layer) as audio, \
                    patch('voice_studio.graph.media_graph_ingest.window_profiles', return_value=[]):
                result = analyze_windows(root, root / "media", metadata, "en", False, False, None, publish=publish)
        self.assertEqual([phase for phase, _ in published], ["base", "audio", "base", "audio"])
        self.assertEqual([row["startMs"] for row in published[0][1]["segments"]], [200])
        self.assertEqual(published[0][1]["audioLayers"], [])
        self.assertEqual([row["startMs"] for row in published[1][1]["audioLayers"]], [100])
        self.assertEqual([row["startMs"] for row in published[2][1]["segments"]], [200, SPEECH_WINDOW_MS + 200])
        self.assertEqual([row["startMs"] for row in result["audioLayers"]], [100, SPEECH_WINDOW_MS + 100])
        self.assertEqual(published[-1][1]["audioLayers"], result["audioLayers"])
        for worker in (ffmpeg, recognize, audio):
            self.assertEqual(worker.call_count, 2)
