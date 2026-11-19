"""Check the common report format and backwards-compatible visual recovery."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_graph_ingest import restore_legacy_visual_track
from voice_studio.mcp.media_graph_mcp import import_report
from voice_studio.graph.media_graph_store import MediaGraphStore, identify
from voice_studio.mcp.voice_mcp import StudioClient, clothing_request


class MediaGraphImportTests(TestCase):
    def test_canonical_evidence_survives_import_without_changes(self):
        with TemporaryDirectory() as folder:
            report = {"sourceDigest": "digest", "language": "ru", "segments": [],
                      "videoEvidence": {"timeline": {"frames": [[]], "stepMs": 40},
                                        "futureVisualMeasurement": {"kept": True}},
                      "sceneEvidence": {"frames": []}}
            path = Path(folder) / "analysis.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(import_report(path, {"sha256": "digest"}, "ru"), report)
            with self.assertRaises(ValueError):
                import_report(path, {"sha256": "another-source"}, "ru")

    def test_masks_keep_accepting_old_sidecars_and_the_common_report(self):
        timeline = {"frames": [[]], "stepMs": 40}
        profile = [{"faceId": 2, "id": "profile-2", "number": 1}]
        current = {"sourceDigest": "exact", "videoEvidence": {"timeline": timeline}, "speakerProfileIndex": profile}
        previous = {"sourceDigest": "exact", "timeline": timeline, "speakerProfiles": profile}
        self.assertEqual(clothing_request(current, 2, 0, 40), clothing_request(previous, 2, 0, 40))

    @patch('voice_studio.graph.media_graph_ingest.attach_person_evidence')
    @patch('voice_studio.graph.media_graph_ingest.analyze_visual_track')
    @patch('voice_studio.graph.media_graph_ingest.extract_speech', return_value=(Path("audio.wav"), 0, 800))
    def test_legacy_uses_common_visual_stages_and_keeps_edits(self, extract, analyze, attach):
        evidence = {"timeline": {"frames": [[(7, .1, .2, .3, .4, 1, 8)]], "stepMs": 40},
                    "editorTimeline": {"shots": [{"startMs": 0}]}, "phrases": {}}

        def analyze_from_current_cues(source, wav, offset, work, *, people_count=None):
            self.assertEqual(work["segments"][0]["speakerNumber"], 2)
            self.assertNotIn("visualFace", work["segments"][0])
            work["videoEvidence"] = evidence

        analyze.side_effect = analyze_from_current_cues
        report = {"language": "ru", "segments": [{"startMs": 0, "endMs": 900, "text": "Реплика",
                                                 "speakerNumber": 2, "visualFace": 3}],
                  "videoEvidence": {"stages": {"light-asd": "ok"}, "personLayer": {"plan": {"gaze": True}}}}
        metadata = {"kind": "video", "hasAudio": True, "durationMs": 800, "streams": [{"kind": "audio"}]}
        with TemporaryDirectory() as folder:
            self.assertEqual(restore_legacy_visual_track(Path(folder), Path("source.mov"), metadata, report),
                             "restored")
        self.assertEqual(report["segments"][0]["endMs"], 900)
        self.assertEqual(report["segments"][0]["speakerNumber"], 2)
        self.assertEqual(report["videoEvidence"]["timeline"], evidence["timeline"])
        self.assertEqual(attach.call_args.kwargs, {"body": False, "gaze": True})


    @patch('voice_studio.mcp.voice_mcp.http.client.HTTPConnection')
    def test_voice_mcp_writes_report_accepted_by_graph_import(self, connection):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "video.mov"
            source.write_bytes(b"original")
            info = identify(source)
            report = {"sourceDigest": info["sha256"], "language": "ru", "segments": [], "srt": "",
                      "videoEvidence": {"timeline": {"frames": [[]], "stepMs": 40},
                                        "futureLayer": {"frames": []}}}
            response = BytesIO(json.dumps(report).encode())
            response.status = 200
            connection.return_value.getresponse.return_value = response
            result = StudioClient(8766, Path(folder)).transcript(str(source), "ru", visual=True, scene=True)
            self.assertEqual(import_report(result["analysisPath"], info, "ru"), report)
            self.assertIn("scene=1", connection.return_value.request.call_args.args[1])
            self.assertIn("object_sampling=small-detail", connection.return_value.request.call_args.args[1])

            response.seek(0)
            StudioClient(8766, Path(folder)).transcript(str(source), "ru")
            self.assertIn("scene=0", connection.return_value.request.call_args.args[1])

            response.seek(0)
            StudioClient(8766, Path(folder)).transcript(str(source), "ru", object_sampling="every-third")
            self.assertIn("scene=1", connection.return_value.request.call_args.args[1])

    def test_technical_frame_uses_only_the_person_track_visible_at_that_time(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "video.mov"
            source.write_bytes(b"original")
            store = MediaGraphStore(Path(folder) / "graphs")
            body = [.1, .2, .3, .5]
            sample = {"id": "person", "kind": "person_observation", "startMs": 250, "endMs": 251,
                      "label": "Лицо 8", "data": {"track": 7, "faceId": 8, "box": body,
                                                 "gaze": {"horizontal": "right"}}}
            face = {"track": 7, "faceId": 8, "box": [.2, .2, .1, .1], "active": False}
            graph = {"nodes": [sample], "edges": [], "faceSampleMs": 40,
                     "faceFrames": [(240, [face]), (280, [{**face, "track": 9}])]}
            key = store.publish(identify(source), {"kind": "video", "durationMs": 500}, graph, "ru", {})["snapshotId"]
            context = store.frame_context(key, 260)[2]
            self.assertEqual(context["boxes"][0]["label"], "Лицо 8 · Взгляд →")
            self.assertEqual(context["boxes"][1]["box"], body)
            self.assertEqual(store.frame_context(key, 280)[2]["people"], [])
