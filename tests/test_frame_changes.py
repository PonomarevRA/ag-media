"""Verify native frame geometry, candidate budgets and graph exposure."""

from collections import Counter
from unittest import TestCase

from voice_studio.graph.media_graph_ingest import apply_scene_evidence
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.vision.frame_changes import augment_plan, bound_samples
from voice_studio.vision.frame_change_panels import assign_motion
from voice_studio.vision.scene_sampling import sample_plan


class RegionPlanTests(TestCase):
    """Use measured cases on variable source time and keep baseline semantics."""

    def test_routing_adds_short_between_anchor_witnesses_without_replacing_baseline(self):
        """A one-frame event missed by the ordinary schedule adds a crop request."""
        clock = [(n, n * 40) for n in range(100)]
        baseline = sample_plan(clock, {50}, "basic")
        original = {row["sourceFrame"]: dict(row) for row in baseline}
        target = next(n for n in range(10, 20) if n not in original)
        box = {"box": [.46, .25, .07, .06], "changedPixels": 200}
        rows, omitted = bound_samples([(target, [box]), (50, [box])], clock, {50}, for_objects=True)
        self.assertEqual(omitted, 0)
        self.assertEqual([row["sourceFrame"] for row in rows], [target])
        augment_plan(baseline, clock, {50}, {"mode": "objects", "status": "ok", "objectRows": rows})
        selected = {row["sourceFrame"]: row for row in baseline}
        for idx, row in original.items():
            self.assertEqual(selected[idx], row)
        self.assertEqual(selected[target]["changeRegions"], [box])
        self.assertTrue(selected[target]["changeOnly"])
        self.assertFalse(selected[target]["objects"])
        self.assertFalse(selected[target]["ocr"])

    def test_short_burst_keeps_first_changed_frame_as_an_object_candidate(self):
        """Keep appearance, movement and disappearance if all fit the per-second budget."""
        clock = [(n, n * 40) for n in range(25)]
        samples = [(n, [{"box": [.1, .1, .03, .03], "changedPixels": 30}]) for n in (17, 18, 19)]
        rows, _ = bound_samples(samples, clock, [], for_objects=True)
        self.assertEqual([r["startMs"] for r in rows], [680, 720, 760])

    def test_larger_graphic_keeps_recognition_even_without_small_object_request(self):
        """An animated card gets its own crop and exact graph reading."""
        from voice_studio.vision.frame_change_recognition import attach_change_recognition

        clock = [(index, index * 40) for index in range(30)]
        card = {"box": [.6, .15, .31, .27], "changedPixels": 3000}
        rows, _ = bound_samples([(7, [card])], clock, [])
        layer = {"mode": "assist", "status": "ok", "rows": rows}
        plan = sample_plan(clock, [], "basic")
        augment_plan(plan, clock, set(), layer)
        sample = next(row for row in plan if row["sourceFrame"] == 7)
        self.assertEqual(sample["changeRecognitionRegions"], [{"index": 0, "box": card["box"]}])
        self.assertNotIn("changeRegions", sample)
        line = {"text": "Map", "box": [.65, .2, .12, .04]}
        attach_change_recognition(layer, {"regions": [
            {"sourceFrame": 7, "index": 0, "status": "recognized", "text": [line], "objects": [],
             "sampleTimeMs": 440, "verifiedUntilMs": 680}]})
        report = {"language": "en", "sceneEvidence": {"frames": []}, "coverage": {"scene": []}}
        apply_scene_evidence(report, {"status": "ok", "frames": [], "frameChanges": layer}, 3000, 4200)
        graph = build_graph(report, kind="video", duration_ms=4200)
        change = next(node for node in graph["nodes"] if node["kind"] == "frame_change")
        self.assertEqual(change["data"]["recognition"]["text"], [line])
        self.assertEqual(change["data"]["recognition"]["verifiedUntilMs"], 3680)
        self.assertEqual((change["startMs"], change["endMs"]), (3280, 3320))
        self.assertIn((change["id"], "recognition_from_frame", "frame:3440"), graph["edges"])

    def test_dense_candidates_have_independent_measurement_and_inference_caps(self):
        """Use the dense clock before thinning and keep at most four extra frames per second."""
        clock = [(n, n * 10) for n in range(201)]
        samples = [(n, [{"box": [.1, .1, .03, .03], "changedPixels": 30}]) for n in range(1, 200)]
        measured, _ = bound_samples(samples, clock, [])
        planned, _ = bound_samples(samples, clock, [], for_objects=True)
        self.assertEqual(Counter(row["startMs"] // 1000 for row in measured), {0: 8, 1: 8})
        self.assertEqual(Counter(row["startMs"] // 1000 for row in planned), {0: 4, 1: 4})
        self.assertEqual([row["sourceFrame"] for row in planned[:2]], [1, 2])
        self.assertEqual([row["sourceFrame"] for row in planned[-2:]], [198, 199])

    def test_fallback_witnesses_use_only_free_measurement_and_recognition_slots(self):
        """A dense onset keeps all its original zones and its original OCR crop."""
        from voice_studio.vision.frame_changes import fill_supplemental_rows, recognition_witnesses

        def row(stamp, pixels):
            """Build one compact detection on a source-clock instant."""
            return {"startMs": stamp, "regions": [
                {"box": [.1, .1, .1, .1], "changedPixels": pixels},
            ]}

        baseline = [row(i * 40, 350) for i in range(8)] + [row(1040, 350)]
        extra = [row(360, 750), row(1200, 750), row(2200, 750)]
        rows, omitted = fill_supplemental_rows(baseline, extra)
        self.assertEqual(omitted, 1)
        self.assertEqual([item for item in rows if not item.get("supplemental")], baseline)
        self.assertEqual([item["startMs"] for item in rows if item.get("supplemental")], [1200, 2200])
        selected = [item["startMs"] for item, _ in recognition_witnesses(rows)]
        self.assertEqual(selected, [280, 1040, 2200])

    def test_assist_routes_motion_to_ocr_without_crossing_cuts(self):
        """Keep the side of the motion and discard local evidence from the previous shot."""
        left = {"box": [.1, .1, .05, .04], "changedPixels": 120}
        right = {"box": [.8, .1, .05, .04], "changedPixels": 200}
        plan = [{"timeMs": 100, "ocr": True}, {"timeMs": 360, "ocr": True}]
        rows = [{"startMs": 80, "regions": [left]}, {"startMs": 180, "regions": [left]},
                {"startMs": 300, "regions": [right]}]
        assign_motion(plan, rows, [0, 250])
        self.assertEqual((sum(plan[0]["changeBands"][:10]), sum(plan[0]["changeBands"][10:])), (120, 0))
        self.assertEqual((sum(plan[1]["changeBands"][:10]), sum(plan[1]["changeBands"][10:])), (0, 200))


class FrameChangeGraphTests(TestCase):
    """Verify graph geometry, exact frame edges and long-window offsets."""

    def test_partition_offsets_events_and_preserves_full_region_in_graph(self):
        """Query original time and geometry even though measurement ran on a partition."""
        scene = {"status": "ok", "frames": [], "frameChanges": {"status": "ok", "basis": "tile-difference",
                 "framesCompared": 30, "rows": [{"startMs": 120, "endMs": 160, "sourceFrame": 3,
                 "regions": [{"box": [.25, .1, .1, .2], "changedPixels": 155}]}]}}
        report = {"language": "en", "sceneEvidence": {"frames": []}, "coverage": {"scene": []}}
        apply_scene_evidence(report, scene, 12000, 13000)
        graph = build_graph(report, kind="video", duration_ms=13000)
        change = next(row for row in graph["nodes"] if row["kind"] == "frame_change")
        self.assertEqual((change["startMs"], change["endMs"]), (12120, 12160))
        self.assertEqual(change["data"], {"box": [.25, .1, .1, .2], "changedPixels": 155, "basis": "tile-difference"})
        self.assertTrue(any(source == change["id"] and relation == "at_frame"
                            for source, relation, _ in graph["edges"]))
        self.assertEqual(graph["coverage"]["scene"][0]["frameChanges"]["framesCompared"], 30)

    def test_verified_panel_reuses_an_exact_prior_reading_in_the_graph(self):
        """Store the OCR once and offset the reference along with partition timestamps."""
        panel = {"box": [.5, 0, .5, 1], "panelType": "ui", "basis": "frame-layout", "changeAssisted": True}
        line = {"role": "text_line", "basis": "ocr", "text": "Export", "box": [.55, .2, .15, .05]}
        frames = [{"timeMs": 100, "interfaces": [{**panel, "elements": [line]}]},
                  {"timeMs": 500, "interfaces": [{**panel, "readingTimeMs": 100}]}]
        report = {"language": "en", "sceneEvidence": {"frames": []}, "coverage": {"scene": []}}
        apply_scene_evidence(report, {"status": "ok", "frames": frames}, 1000, 2000)
        graph = build_graph(report, kind="video", duration_ms=2000)
        self.assertEqual(len([row for row in graph["nodes"] if row["kind"] == "ui_element"]), 1)
        self.assertIn(("screen_region:1500:0", "reading_from_frame", "frame:1100"), graph["edges"])


class ChangeTrackingTests(TestCase):
    """Only strong, unambiguous crop evidence may bridge a short fast displacement."""

    def test_fast_pair_joins_but_ambiguous_and_weak_pairs_do_not(self):
        """Exercise graph association for boxes with no overlap."""
        from voice_studio.graph.media_graph_objects import split_tracks

        previous_box = [.10, .20, .04, .05]
        next_box = [.18, .20, .04, .05]
        prior = {"startMs": 680, "label": "cup",
                 "data": {"box": previous_box, "detailBasis": "tile-difference"}}
        match = {"timeMs": 680, "label": "cup", "box": previous_box,
                 "objectSimilarity": .99, "edgeSimilarity": .96, "scaleRatio": 1.05}
        current = {"startMs": 720, "label": "cup",
                   "data": {"box": next_box, "detailBasis": "tile-difference", "objectMatches": [match]}}
        self.assertEqual([len(track) for track in split_tracks({680: [prior], 720: [current]})], [2])
        other = {**current, "data": {**current["data"], "box": [.23, .20, .04, .05]}}
        self.assertEqual([len(track) for track in split_tracks({680: [prior], 720: [current, other]})], [1, 1, 1])
        match["edgeSimilarity"] = .8
        self.assertEqual([len(track) for track in split_tracks({680: [prior], 720: [current]})], [1, 1])

    def test_displaced_change_sample_extends_a_unique_baseline_track(self):
        """Keep the primary ID and refuse the same pair if a second track claims it."""
        from voice_studio.graph.media_graph_objects import append_detail_tracks

        original_box = [.10, .20, .04, .05]
        base = {"id": "anchor", "startMs": 100, "label": "cup", "data": {"box": original_box}}
        match = {"timeMs": 100, "label": "cup", "box": original_box, "objectSimilarity": .99,
                 "edgeSimilarity": .95, "scaleRatio": 1.0, "comparisonBasis": "frame-change-anchor"}
        moved = {"startMs": 140, "label": "cup", "data": {"box": [.18, .20, .04, .05],
                 "detailBasis": "tile-difference", "objectMatches": [match]}}
        self.assertEqual(append_detail_tracks([[base]], [[moved]]), [(base, [base, moved])])
        competitor = {**base, "id": "other", "data": {"box": moved["data"]["box"]}}
        self.assertEqual(len(append_detail_tracks([[base], [competitor]], [[moved]])), 3)
