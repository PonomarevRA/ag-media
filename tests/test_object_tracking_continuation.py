"""Exercise shot-local weak continuations without changing detector identity evidence."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf
from unittest.mock import patch

from voice_studio.graph.media_graph_ingest import apply_scene_evidence
from voice_studio.graph.media_graph_model import build_graph, object_layers
from voice_studio.graph.media_graph_store import MediaGraphStore, identify
from voice_studio.vision.object_tracking_contract import WEAK_CONTINUATION

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


def detection(box, score=.88):
    """Return one original measured detector result."""
    return {"label": "car", "box": list(box), "score": score}


def frame(stamp, items, shot=0):
    """Represent one independent detector sample on the original timeline."""
    return {"timeMs": stamp, "objects": list(items), "shot": shot, "text": []}


def weak(anchor, box, *, stamp=100):
    """Prepare one weak sample with explicit preceding detector provenance."""
    return {**detection(box, .28), "detailBasis": WEAK_CONTINUATION, "basis": "weak-detector-proposal",
            "trackingWitness": {"timeMs": stamp, "box": anchor["box"], "basis": "background-lk"}}


def source_report(frames, *, cuts=(0,), offset=0):
    """Assemble the same source-clock report used for graph and editor layers."""
    duration = offset + 1400
    coverage = {"scene": [], "shots": [{"startMs": offset, "endMs": duration, "status": "sampled"}]}
    report = {"language": "en", "durationMs": duration, "coverage": coverage,
              "videoEvidence": {"editorTimeline": {"shots": [{"startMs": offset + time} for time in cuts]}},
              "sceneEvidence": {"frames": []}}
    apply_scene_evidence(report, {"status": "ok", "frames": frames}, offset, duration)
    return report


class WeakTrackProjectionTests(TestCase):
    def test_existing_id_and_best_frame_survive_window_and_store(self):
        """Attach only to the earlier track and retain anchor times through SQLite."""
        base = detection((.1, .29, .11, .14), .54)
        strongest = detection((.1, .29, .11, .14), .95)
        other = detection((.6, .29, .11, .14))
        before = [frame(100, [base, other]), frame(200, []), frame(300, [strongest, other])]
        after = deepcopy(before)
        after[1]["objects"].append(weak(base, (.12, .29, .11, .14)))
        original = deepcopy(after)
        baseline = object_layers(source_report(before, offset=3200))
        report = source_report(after, offset=3200)
        result = object_layers(report)
        self.assertEqual(after, original)
        self.assertEqual(len(result["objectTimeline"]), 2)
        self.assertEqual({row["id"] for row in result["objectTimeline"]},
                         {row["id"] for row in baseline["objectTimeline"]})
        continued, = [row for row in result["objectTimeline"] if row.get("trackingContinuation")]
        self.assertEqual(continued["trackingContinuation"], [{"timeMs": 3400, "anchorMs": 3300, "basis": "background-lk"}])
        self.assertEqual(continued["representative"]["score"], .95)
        self.assertEqual(continued["representative"]["timeMs"], 3500)
        self.assertEqual(continued["sampleCount"], 3)
        graph = build_graph(report, kind="video", duration_ms=report["durationMs"])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source_path = root / "fixture.mp4"
            source_path.write_bytes(b"fixture")
            source = identify(source_path)
            store = MediaGraphStore(root / "graphs")
            snapshot = store.publish(source, {"kind": "video", "durationMs": report["durationMs"]}, graph, "en", {})
            restored, scene_coverage = store.scene_evidence(snapshot["snapshotId"], source["sha256"], "en")
        restored_report = {**report, "sceneEvidence": restored, "coverage": {**report["coverage"], "scene": scene_coverage}}
        self.assertEqual(object_layers(restored_report), result)

    def test_unanchored_and_cross_cut_weak_measurements_never_start_tracks(self):
        """Drop when anchor is absent, ambiguous, or on the other side of a cut."""
        anchor = detection((.1, .29, .11, .14))
        candidate = weak(anchor, (.12, .29, .11, .14))
        cases = ((source_report([frame(200, [candidate])]), 0),
                 (source_report([frame(100, [anchor]), frame(200, [candidate])], cuts=(0, 180)), 1),
                 (source_report([frame(100, [anchor, anchor]), frame(200, [candidate])]), 2),
                 (source_report([frame(100, [anchor]), frame(200, [candidate, candidate])]), 1))
        for report, count in cases:
            graph = object_layers(report)
            self.assertEqual(len(graph["objectTimeline"]), count)
            self.assertTrue(all(not row.get("trackingContinuation") for row in graph["objectTimeline"]))


@skipIf(cv2 is None, "OpenCV is required for the optional tracking worker")
class WeakTrackVisionTests(TestCase):
    def test_close_boxes_are_matched_after_confirmed_detections(self):
        """Unconfirmed candidates may continue a sole unmatched baseline track."""
        from voice_studio.vision.object_track_continuation import continue_object_tracks

        anchor = detection((.1, .29, .11, .14))
        good = detection((.11, .29, .11, .14), .29)
        other = detection((.7, .29, .11, .14), .29)
        frames = [frame(100, [anchor]), frame(200, []), frame(300, [], 1)]
        raw = {100: (None, [anchor]), 200: (None, [good, other]), 300: (None, [other])}
        original = deepcopy(raw)
        result = continue_object_tracks(frames, raw)
        self.assertEqual(raw, original)
        self.assertEqual(result["continued"], 1)
        self.assertEqual(frames[1]["objects"][0]["score"], .29)
        self.assertEqual(frames[1]["objects"][0]["trackingWitness"]["basis"], "box-iou")
        self.assertFalse(frames[2]["objects"])

    def test_nearby_distinct_tracks_can_veto_one_weak_proposal(self):
        """Mutual uniqueness preserves the two established same-class tracks."""
        from voice_studio.vision.object_track_continuation import continue_object_tracks

        first = detection((.1, .3, .14, .12))
        second = detection((.17, .3, .14, .12))
        ambiguous = detection((.135, .3, .14, .12), .29)
        frames = [frame(100, [first, second]), frame(200, [])]
        with patch("voice_studio.vision.object_track_continuation.camera_for_pair", return_value=None):
            stats = continue_object_tracks(frames, {100: (None, [first, second]), 200: (None, [ambiguous])})
        self.assertFalse(frames[1]["objects"])
        self.assertEqual(stats["continued"], 0)
        self.assertGreater(stats["ambiguousCandidates"], 0)

    def test_shifted_background_continues_a_disputed_candidate(self):
        """Fit a pan outside foreground boxes and rescue a real weak source rectangle."""
        from voice_studio.vision.object_track_continuation import continue_object_tracks
        from voice_studio.vision.object_tracking_camera import tracking_background

        rng = np.random.default_rng(153)
        gray = rng.integers(0, 255, (260, 380), dtype="uint8")
        old = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dx = 21
        moved = cv2.warpAffine(old, np.float32([[1, 0, dx], [0, 1, 0]]), (380, 260))
        first = detection((.43, .37, .10, .20))
        target = detection((.43 + dx / 380, .37, .10, .20), .28)
        background = tracking_background(old, [first["box"]])
        self.assertEqual(background[1][100, 175], 0)
        for image, item in ((old, first), (moved, target)):
            x, y, w, h = item["box"]
            image[int(260*y):int(260*(y+h)), int(380*x):int(380*(x+w))] = (200, 5, 70)
        frames = [frame(100, [first]), frame(250, [])]
        with TemporaryDirectory() as folder:
            a, b = (Path(folder) / name for name in ("a.png", "b.png"))
            self.assertTrue(cv2.imwrite(str(a), old))
            self.assertTrue(cv2.imwrite(str(b), moved))
            stats = continue_object_tracks(frames, {100: (a, [first]), 250: (b, [target])})
        self.assertEqual(stats["cameraContinued"], 1)
        self.assertEqual(stats["cameraPairs"], 1)
        self.assertEqual(frames[1]["objects"][0]["trackingWitness"]["basis"], "background-lk")
        self.assertEqual(stats["densePairs"], 0)

    def test_dense_motion_requires_texture_reciprocity_and_spread(self):
        """Reject unsupported flow and accept a spatially distributed translation."""
        from voice_studio.vision.object_tracking_camera import robust_translation

        height, width = 100, 180
        forward = np.full((height, width, 2), (4, 0), dtype=np.float32)
        backward = -forward
        texture = np.ones((height, width), dtype=np.float32)
        allowed = np.ones((height, width), dtype=np.uint8)
        good = robust_translation(forward, backward, texture, allowed, allowed, width, height)
        self.assertEqual(good["basis"], "background-dis")
        self.assertAlmostEqual(good["shiftX"], 4 / width, places=5)
        self.assertIsNone(robust_translation(forward, forward, texture, allowed, allowed, width, height))
        self.assertIsNone(robust_translation(forward, backward, texture * 0, allowed, allowed, width, height))
