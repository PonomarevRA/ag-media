"""Verify early reads against actual SQLite snapshots with a controllable analyzer."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import Mock, patch

from voice_studio.mcp.media_graph_mcp import GraphBuildQueue, build_snapshot, import_report, intermediate_publisher
from voice_studio.graph.media_graph_progress import GraphPublication
from voice_studio.graph.media_graph_ingest import compact_graph_publisher
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


class ProgressiveGraphTests(TestCase):
    def test_preview_exposes_geometry_without_claiming_scene_coverage(self):
        """Query early geometric nodes, then publish recognized evidence as a separate revision."""
        publication = GraphPublication("job")
        media = {"kind": "video", "durationMs": 83000, "streams": [
            {"kind": "audio", "index": 1, "startMs": 0}, {"kind": "video", "index": 0, "startMs": 0},
        ]}
        changes = {"status": "ok", "framesCompared": 50, "rows": [
            {"startMs": 400, "endMs": 440, "regions": [{"box": [.1, .2, .15, .1], "changedPixels": 250}]},
        ]}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "preview.mp4"
            original.write_bytes(b"source fixture")
            source = identify(original)
            store = MediaGraphStore(root / "graphs")
            persist = intermediate_publisher(publication, store, source, media, "en")
            publish = compact_graph_publisher(media, False, True, persist)
            publish("base", {"language": "en", "durationMs": 83000, "segments": []})
            publish("scene-preview", {"sceneEvidence": {"status": "partial", "frameChanges": {"rows": []}}})
            self.assertEqual(publication.state()["phase"], "base")
            publish("scene-preview", {"sceneEvidence": {"status": "partial", "frameChanges": changes}})
            early = publication.state()["snapshot"]
            early_id = early["snapshotId"]
            self.assertEqual(early["provenance"]["progress"]["phase"], "scene-preview")
            self.assertEqual(early["coverage"]["scene"][0]["status"], "partial")
            self.assertEqual(early["coverage"]["scene"][0]["frameChanges"]["framesCompared"], 50)
            self.assertEqual(store.timeline(early_id, 0, 83000, "frame_change")["nodes"][0]["startMs"], 400)
            with self.assertRaisesRegex(ValueError, "нет измеренного покрытия"):
                store.scene_evidence(early_id, source["sha256"], "en")
            publish("scene", {"sceneEvidence": {"status": "ok", "frameChanges": changes, "frames": [
                {"timeMs": 400, "text": [{"text": "Menu", "box": [.1, .2, .15, .1]}], "objects": []},
            ]}})
            later = publication.state()["snapshot"]
            self.assertEqual(later["provenance"]["progress"]["previousSnapshotId"], early_id)
            self.assertEqual(later["coverage"]["scene"][0]["status"], "sampled")
            self.assertEqual(store.search(later["snapshotId"], "Menu")["nodes"][0]["label"], "Menu")
            self.assertEqual(store.search(early_id, "Menu")["nodes"], [])

    def test_worker_phase_differs_from_last_readable_snapshot(self):
        publication = GraphPublication('job')
        publication.activity({'stage': 'whisper', 'state': 'running'})
        publication.publish('base', lambda marker: {'snapshotId': 'readable'})
        self.assertEqual(publication.state()['phase'], 'whisper')
        self.assertEqual(publication.state()['snapshotPhase'], 'base')
        self.assertEqual(publication.state()['snapshot']['snapshotId'], 'readable')
        publication.activity({'stage': 'whisper', 'state': 'done'})
        publication.activity({'stage': 'light-asd', 'state': 'running'})
        self.assertEqual(publication.state()['phase'], 'light-asd')
        publication.publish('complete', lambda marker: {'snapshotId': 'final'})
        self.assertEqual(publication.state()['phase'], 'complete')
        self.assertNotIn('workingStages', publication.state())

    def test_graph_is_readable_while_enrichment_is_running(self):
        """Keep pagination pinned to an older immutable snapshot after publication advances."""
        self.exercise_build(fail=False)

    def test_late_failure_preserves_early_graph(self):
        """Surface an enrichment error without losing the available graph."""
        self.exercise_build(fail=True)

    def test_invalid_partial_graph_does_not_hide_the_next_revision(self):
        """Treat a malformed intermediate report as a recoverable publication error."""
        publication = GraphPublication("job")
        store = Mock()
        store.publish.return_value = {"snapshotId": "available"}
        with patch('voice_studio.graph.media_graph_model.build_graph', side_effect=[ValueError("partial"), {}]):
            publish = intermediate_publisher(publication, store, {}, {"kind": "audio", "durationMs": 2000}, "en")
            with self.assertLogs('voice_studio.mcp.media_graph_mcp', level="WARNING"):
                publish("base", {})
            self.assertEqual(publication.state(), {})
            publish("audio", {"audioLayers": [{}]})
        self.assertEqual((publication.state()["phase"], publication.state()["revision"]), ("audio", 1))

    def test_partial_browser_export_is_not_imported_as_a_complete_graph(self):
        """Older exports remain importable while explicit in-progress exports are rejected."""
        source = {"sha256": "source"}
        report = {"language": "en", "segments": [], "sourceDigest": source["sha256"]}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({**report, "analysisComplete": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "промежуточный"):
                import_report(path, source, "en")
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(import_report(path, source, "en"), report)

    def exercise_build(self, fail):
        """Block analysis after its first revision and inspect both views from the caller."""
        base_ready, resume = Event(), Event()
        streams = [{"kind": "audio", "index": 0}]
        media = {"kind": "audio", "durationMs": 2000, "streams": streams}
        report = {"language": "en", "mediaStreams": streams,
                  "segments": [{"startMs": 100, "endMs": 800, "text": "First reply"}]}

        def analyze(*_args, publish, **_kwargs):
            """Simulate a later sound classification while retaining speech."""
            publish("base", report)
            base_ready.set()
            if not resume.wait(5):
                raise TimeoutError("Test did not release the graph worker")
            if fail:
                raise ValueError("analysis failed")
            enriched = {**report, "audioLayers": [
                {"kind": "music", "label": "music", "startMs": 0, "endMs": 1500, "confidence": .8}]}
            publish("audio", enriched)
            publish("audio", enriched)
            return enriched

        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "media.wav"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(root / "graphs")
            queue = GraphBuildQueue()
            with patch('voice_studio.graph.media_graph_ingest.probe_source', return_value=media), \
                    patch('voice_studio.graph.media_graph_ingest.analyze_source', side_effect=analyze):
                job = queue.submit(build_snapshot, store, root / "studio", str(source), "en",
                                   None, False, False, None, progressive=True)["jobId"]
                try:
                    self.assertTrue(base_ready.wait(5))
                    early = queue.status(job)
                    self.assertEqual((early["status"], early["phase"], early["revision"]), ("running", "base", 1))
                    base_id = early["snapshot"]["snapshotId"]
                    self.assertEqual(store.search(base_id, "First")["nodes"][0]["label"], "First reply")
                    self.assertEqual(store.timeline(base_id, 0, 2000, "audio_music")["nodes"], [])
                finally:
                    resume.set()
                    queue.worker.shutdown(wait=True)
                late = queue.status(job)
                self.assertEqual(store.timeline(base_id, 0, 2000, "audio_music")["nodes"], [])
                if fail:
                    self.assertEqual((late["status"], late["error"], late["revision"]), ("failed", "analysis failed", 1))
                    self.assertEqual(late["snapshot"]["snapshotId"], base_id)
                else:
                    self.assertEqual((late["status"], late["phase"], late["revision"]), ("done", "complete", 3))
                    final = late["snapshot"]
                    previous = final["provenance"]["progress"]["previousSnapshotId"]
                    self.assertEqual(store.overview(previous)["provenance"]["progress"]["phase"], "audio")
                    self.assertEqual(store.timeline(final["snapshotId"], 0, 2000, "audio_music")["nodes"][0]["confidence"], .8)
                    self.assertEqual(len(store.list()["snapshots"]), 3)
