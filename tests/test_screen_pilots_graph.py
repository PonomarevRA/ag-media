"""Verify native controls and state OCR on the graph's exact source clock."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.graph.media_graph_ingest import apply_scene_evidence
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify

PANEL = {"basis": "frame-layout", "panelType": "mobile-ui", "box": [.6, .1, .38, .8]}
SMALL_CONTROL = {"role": "icon_candidate", "basis": "native-small-control-glyph",
                 "box": [.91, .20, .02, .03], "pixelBox": [910, 200, 20, 30], "confidence": .813}
STATE = {"recognizedAtMs": 400, "verifiedFromMs": 300, "verifiedUntilMs": 501, "sourceFrame": 12,
         "cropBox": [.6, .25, .38, .12], "kind": "stable", "supportCount": 3}
TEXT = {"role": "text_line", "text": "Keep this label", "basis": "ocr", "box": [.68, .27, .16, .06]}


def report_for(elements, *, offset=0, options=None):
    """Publish one analyzed scene window with the worker timestamps still local."""
    end = offset + 1200
    panel = {**PANEL, "elements": elements, "connections": [], **(options or {})}
    scene = {"status": "ok", "frames": [{"timeMs": 400, "sourceFrame": 12, "interfaces": [panel]}]}
    report = {"language": "en", "durationMs": end,
              "coverage": {"scene": [], "shots": [{"startMs": offset, "endMs": end, "status": "sampled"}]},
              "sceneEvidence": {"frames": []}}
    apply_scene_evidence(report, scene, offset, end)
    return report


class ScreenPilotsGraphTests(TestCase):
    def test_exact_observations_and_sqlite_restore_keep_both_optional_fields(self):
        """Nested source-time references survive the window offset, graph and SQLite."""
        prior = {"recognizedAtMs": 190, "sourceFrame": 6, "box": [.69, .27, .16, .06]}
        elements = [{**TEXT, "stateEvidence": {**STATE, "sameContentAs": prior}}, SMALL_CONTROL]
        original = deepcopy(elements)
        report = report_for(elements, offset=5000, options={
            "screenStateEndMs": 501, "screenStateElements": elements,
        })
        self.assertEqual(elements, original)
        expected_evidence = {**STATE, "recognizedAtMs": 5400, "verifiedFromMs": 5300, "verifiedUntilMs": 5501,
                             "sameContentAs": {**prior, "recognizedAtMs": 5190}}
        graph = build_graph(report, kind="video", duration_ms=report["durationMs"])
        observed = [node for node in graph["nodes"] if node["kind"] == "ui_element"]
        text, control = observed
        self.assertEqual((text["startMs"], text["endMs"]), (5400, 5401))
        self.assertEqual(text["data"]["basis"], "ocr")
        self.assertEqual(text["data"]["stateEvidence"], expected_evidence)
        self.assertEqual(control["data"]["pixelBox"], SMALL_CONTROL["pixelBox"])
        self.assertEqual(control["data"]["confidence"], SMALL_CONTROL["confidence"])
        self.assertIn((text["id"], "at_frame", "frame:5400"), graph["edges"])

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source_file = folder / "source.mp4"
            source_file.write_bytes(b"source reference fixture")
            source = identify(source_file)
            store = MediaGraphStore(folder / "graphs")
            media = {"kind": "video", "durationMs": report["durationMs"]}
            snapshot = store.publish(source, media, graph, "en", {})["snapshotId"]
            persisted = store.node(snapshot, text["id"])
            self.assertEqual(persisted["data"]["stateEvidence"], expected_evidence)
            _, sample, crop_context = store.frame_context(snapshot, node_id=text["id"])
            self.assertEqual((sample, crop_context["targetObservationId"]), (5400, text["id"]))
            self.assertEqual(crop_context["boxes"][0]["box"], TEXT["box"])
            restored, _ = store.scene_evidence(snapshot, source["sha256"], "en")
        panel, = next(row["interfaces"] for row in restored["frames"] if row["timeMs"] == 5400)
        expected = [{**TEXT, "stateEvidence": expected_evidence}, SMALL_CONTROL]
        self.assertEqual(panel["elements"], expected)
        self.assertEqual(panel["screenStateElements"], expected)

    def test_invalid_geometry_or_spans_do_not_become_pilot_evidence(self):
        """Accept only known bases and text support tied to the sampled frame."""
        mismatched_basis = {**SMALL_CONTROL, "basis": "ocr"}
        too_far = {**SMALL_CONTROL, "pixelBox": [0, 0, 0, 18]}
        for element in (mismatched_basis, too_far):
            with self.subTest(element=element):
                report = report_for([element])
                graph = build_graph(report, kind="video", duration_ms=1200)
                item, = [node["data"] for node in graph["nodes"] if node["kind"] == "ui_element"]
                self.assertNotIn("pixelBox", item)
                self.assertNotIn("confidence", item)

        invalid = (
            {"recognizedAtMs": 401},
            {"verifiedUntilMs": 1400},
            {"verifiedFromMs": 401},
            {"cropBox": [.02, .26, .2, .08]},
            {"sourceFrame": True},
            {"kind": "changed"},
            {"supportCount": 1},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                report = report_for([{**TEXT, "stateEvidence": {**STATE, **changes}}])
                graph = build_graph(report, kind="video", duration_ms=1200)
                item, = [node["data"] for node in graph["nodes"] if node["kind"] == "ui_element"]
                self.assertEqual(item["text"], TEXT["text"])
                self.assertNotIn("stateEvidence", item)

    def test_state_does_not_bridge_cut_and_singleton_keeps_instant_support(self):
        """A measured stable run stays within its shot and a singleton spans one millisecond."""
        report = report_for([{**TEXT, "stateEvidence": STATE}])
        report["videoEvidence"] = {"editorTimeline": {"shots": [{"startMs": 0}, {"startMs": 450}]}}
        graph = build_graph(report, kind="video", duration_ms=1200)
        item, = [node["data"] for node in graph["nodes"] if node["kind"] == "ui_element"]
        self.assertNotIn("stateEvidence", item)

        changed = {**STATE, "kind": "changed", "verifiedFromMs": 400, "verifiedUntilMs": 401, "supportCount": 1}
        report = report_for([{**TEXT, "basis": "stateful-crop", "stateEvidence": changed}])
        graph = build_graph(report, kind="video", duration_ms=1200)
        item, = [node["data"] for node in graph["nodes"] if node["kind"] == "ui_element"]
        self.assertEqual(item["stateEvidence"], changed)

    def test_prior_content_cannot_link_across_cut_forward_or_outside_panel(self):
        """Preserve the current OCR while rejecting incompatible earlier references."""
        reference = {"recognizedAtMs": 150, "sourceFrame": 3, "box": [.69, .27, .16, .06]}
        variants = (reference, {**reference, "recognizedAtMs": 400},
                    {**reference, "recognizedAtMs": 350, "box": [.1, .2, .2, .2]})
        for prior in variants:
            with self.subTest(prior=prior):
                report = report_for([{**TEXT, "stateEvidence": {**STATE, "sameContentAs": prior}}])
                report["videoEvidence"] = {"editorTimeline": {"shots": [{"startMs": 0}, {"startMs": 200}]}}
                graph = build_graph(report, kind="video", duration_ms=1200)
                item, = [node["data"] for node in graph["nodes"] if node["kind"] == "ui_element"]
                self.assertEqual(item["stateEvidence"], STATE)
