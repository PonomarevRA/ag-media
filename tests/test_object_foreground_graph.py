"""Check GrabCut comparison provenance and its isolation from object identity."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.graph.media_graph_ingest import apply_scene_evidence
from voice_studio.graph.media_graph_model import build_graph, object_foreground_matches, object_layers
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


SOURCE = {"label": "car", "box": [.12, .25, .27, .3], "score": .91}
OTHER = {"label": "car", "box": [.64, .24, .24, .31], "score": .90}


def comparison(observation, time=100):
    """Create a complete measured pair spanning the raw and masked comparisons."""
    return {"timeMs": time, "label": observation["label"], "box": observation["box"], "basis": "grabcut-foreground",
            "baselineObjectSimilarity": .74, "baselineEdgeSimilarity": None,
            "maskedObjectSimilarity": .99, "maskedEdgeSimilarity": .81,
            "sourceFraction": .33, "targetFraction": .46, "sourceStability": .92, "targetStability": .89,
            "maskOverlap": .79}


def source_report(scene, start=0):
    """Align one scene window while leaving the raw worker report available for comparison."""
    end = start + 1200
    coverage = {"scene": [], "shots": [{"startMs": start, "endMs": end, "status": "sampled"}]}
    result = {"language": "en", "durationMs": end, "coverage": coverage,
              "videoEvidence": {"editorTimeline": {"shots": [{"startMs": start}]}},
              "sceneEvidence": {"frames": []}}
    apply_scene_evidence(result, scene, start, end)
    return result


class ObjectForegroundGraphTests(TestCase):
    def test_window_and_sqlite_roundtrip_preserve_raw_and_exact_profile_pairs(self):
        """Keep cross-object measurements raw and preserve existing profile identities."""
        target = {**SOURCE, "box": [.13, .25, .27, .3]}
        scene = {"status": "ok", "frames": [
            {"timeMs": 100, "objects": [SOURCE, OTHER], "text": []},
            {"timeMs": 400, "objects": [target, OTHER], "text": []},
        ]}
        baseline = object_layers(source_report(scene, start=1200))
        scene["frames"][1]["objects"][0] = {**target, "objectForegroundMatches": [comparison(SOURCE), comparison(OTHER)]}
        raw = deepcopy(scene)
        report = source_report(scene, start=1200)
        self.assertEqual(scene, raw)
        layers = object_layers(report)
        self.assertEqual(baseline["objectTimeline"], layers["objectTimeline"])
        without_foreground = [{key: value for key, value in profile.items() if key != "foregroundComparisons"}
                              for profile in layers["objectProfiles"]]
        self.assertEqual(baseline["objectProfiles"], without_foreground)

        profile, = [item for item in layers["objectProfiles"] if "foregroundComparisons" in item]
        pair, = profile["foregroundComparisons"]
        self.assertEqual((pair["fromTimeMs"], pair["toTimeMs"]), (1300, 1600))
        self.assertEqual((pair["maskOverlap"], pair["maskedObjectSimilarity"], pair["baselineEdgeSimilarity"]),
                         (.79, .99, None))
        graph = build_graph(report, kind="video", duration_ms=report["durationMs"])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.mp4"
            path.write_bytes(b"local source fixture")
            source = identify(path)
            store = MediaGraphStore(root / "graphs")
            snapshot = store.publish(source, {"kind": "video", "durationMs": report["durationMs"]},
                                     graph, "en", {})["snapshotId"]
            previous = store.node(snapshot, pair["fromObservationId"])
            current = store.node(snapshot, pair["toObservationId"])
            self.assertEqual(previous["data"]["box"], SOURCE["box"])
            self.assertEqual(previous["data"]["identityId"], current["data"]["identityId"])
            self.assertEqual(current["data"]["objectForegroundMatches"],
                             [comparison(SOURCE, 1300), comparison(OTHER, 1300)])
            self.assertEqual(store.node(snapshot, profile["id"])["data"]["foregroundComparisons"], [pair])
            restored, coverage = store.scene_evidence(snapshot, source["sha256"], "en")
            replay = {**report, "sceneEvidence": restored, "coverage": {**report["coverage"], "scene": coverage}}
            self.assertEqual(object_layers(replay), layers)

    def test_raw_pair_validation_and_profile_budget(self):
        """Reject out-of-range measurements and cap valid source-paired evidence."""
        valid = comparison(SOURCE)
        bad = {**valid, "sourceStability": float("nan")}
        wrong_frame = {**valid, "timeMs": 999}
        record = {"objectForegroundMatches": [bad, valid, wrong_frame, valid]}
        self.assertEqual(object_foreground_matches(record, 400), [valid])
        no_masked_scores = {**valid, "maskedObjectSimilarity": None, "maskedEdgeSimilarity": None}
        same_frame = {**valid, "timeMs": 400}
        record["objectForegroundMatches"] = [no_masked_scores, same_frame]
        self.assertFalse(object_foreground_matches(record, 400))
        record["objectForegroundMatches"] = [valid] * 4
        self.assertEqual(len(object_foreground_matches(record, 400)), 3)
        scene = {"status": "ok", "frames": [{"timeMs": 100, "objects": [SOURCE], "text": []}]}
        for time in range(200, 1000, 100):
            scene["frames"].append({"timeMs": time, "text": [],
                                    "objects": [{**SOURCE, "objectForegroundMatches": [valid, valid]}]})
        profile, = object_layers(source_report(scene))["objectProfiles"]
        self.assertEqual([pair["toTimeMs"] for pair in profile["foregroundComparisons"]],
                         [200, 300, 400, 500, 600, 700])

    def test_ambiguous_exact_source_position_has_no_profile_reference(self):
        """Keep raw evidence when multiple source observations match its geometry."""
        scene = {"status": "ok", "frames": [
            {"timeMs": 100, "objects": [SOURCE, {**SOURCE, "score": .50}], "text": []},
            {"timeMs": 400, "objects": [{**SOURCE, "objectForegroundMatches": [comparison(SOURCE)]}], "text": []},
        ]}
        report = source_report(scene)
        profiles = object_layers(report)["objectProfiles"]
        self.assertTrue(profiles)
        self.assertFalse(any("foregroundComparisons" in profile for profile in profiles))
        graph = build_graph(report, kind="video", duration_ms=1200)
        observation, = [node for node in graph["nodes"]
                        if node["kind"] == "object_observation" and node["startMs"] == 400]
        self.assertEqual(observation["data"]["objectForegroundMatches"], [comparison(SOURCE)])
