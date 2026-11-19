"""Check the NanoDet view cache in the existing OpenCV worker environment."""

import importlib.util
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

HAS_VISION = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None
if HAS_VISION:
    import cv2
    import numpy as np
    from voice_studio.vision.scene_objects import MEAN, PIXEL_LUT, STD, ObjectDetector, calibrate_tiles, refine_drinkware
    from voice_studio.vision.small_object_pass import SmallObjectPass


@skipUnless(HAS_VISION, "Requires the studio's OpenCV vision runtime")
class ObjectViewCacheTests(TestCase):
    """Cover exact input matching, isolation, and the bounded cache lifecycle."""

    @patch("voice_studio.vision.scene_objects.cv2.imread")
    def test_tiles_only_when_shot_anchor_recovers_a_small_confident_box(self, read):
        read.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        cup = {"label": "cup", "score": .71, "box": [.4, .4, .08, .12]}
        another_cup = {"label": "cup", "score": .62, "box": [.60, .4, .08, .12]}
        weak = {**another_cup, "score": .46}
        detector = Mock()
        detector.detect.side_effect = ([cup], [cup, weak], [cup], [cup, another_cup])
        plan = [{"shot": 0, "sectors": True}, {"shot": 1, "sectors": False}, {"shot": 2, "sectors": True}]
        self.assertEqual(calibrate_tiles(detector, plan, Path("/tmp")), {2})
        self.assertEqual(detector.detect.call_count, 4)

    def test_lut_matches_previous_float_normalization(self):
        pixels = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
        expected = np.asarray((pixels.astype(np.float32) - MEAN) / STD, dtype=np.float32)
        self.assertTrue(np.array_equal(cv2.LUT(pixels, PIXEL_LUT), expected))

    def test_cache_does_not_reuse_changed_pixels_or_mutated_boxes(self):
        with patch('voice_studio.vision.scene_objects.cv2.dnn.readNet'):
            detector = ObjectDetector("unused.onnx", reuse_views=True)
        image = np.zeros((416, 416, 3), dtype=np.uint8)
        changed = image.copy()
        changed[5, 6, 0] = 1
        box = [.3, .3, .2, .2]

        def detect(resized, people):
            return [{"label": "person" if people else "bus", "box": box.copy(), "score": .9}]

        with patch.object(detector, "detect_resized_view", side_effect=detect) as model:
            first = detector.detect_view(image)
            first[0]["box"][0] = .75
            self.assertEqual(detector.detect_view(image)[0]["box"], box)
            self.assertEqual(model.call_count, 1)
            detector.detect_view(changed)
            self.assertEqual(detector.detect_view(image, people=True)[0]["label"], "person")
            self.assertEqual(detector.detect_view(image)[0]["label"], "bus")
            self.assertEqual(model.call_count, 4)
            self.assertEqual(detector.cache_hits, 1)
            detector.clear_view_cache()
            detector.detect_view(image)
            self.assertEqual(model.call_count, 5)


    def test_weak_laptop_is_replaced_only_by_multiple_distinct_supported_drinkware_boxes(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        outer = {"label": "laptop", "score": .362, "box": [.46, .627, .11, .15]}
        proposals = [
            {"label": "cup", "score": .55, "box": [.48, .43, .075, .19]},
            {"label": "cup", "score": .52, "box": [.65, .45, .07, .18]},
        ]
        with patch.object(ObjectDetector, "detect_view", return_value=proposals):
            detector = ObjectDetector.__new__(ObjectDetector)
            items, review = refine_drinkware(detector, image, [outer])
            self.assertEqual([row["label"] for row in items], ["cup", "cup"])
            self.assertEqual(review["proposal"], outer)
            self.assertEqual(review["verifiedCups"], 2)
            proposals.append({"label": "laptop", "score": .66, "box": [.38, .15, .32, .44]})
            self.assertEqual(refine_drinkware(detector, image, [outer]), ([outer], None))


@skipUnless(HAS_VISION, "Requires the studio's OpenCV vision runtime")
class SmallObjectPassTests(TestCase):
    def test_only_an_unmatched_anchor_witness_opens_its_region(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        known = {"label": "cup", "box": [.1, .5, .04, .16], "score": .6}
        small = {"label": "cup", "box": [.71, .5, .04, .16], "score": .7}
        large = {"label": "book", "box": [.7, .2, .15, .25], "score": .8}
        detector = Mock()
        detector.detect_view.return_value = [known]
        detector.detect_region.side_effect = ([{**known, "score": .69}], [small, large])

        second_pass = SmallObjectPass()
        baseline = second_pass.observe_anchor(detector, image, shot=2)
        self.assertEqual([item["score"] for item in baseline], [.8, .7, .69])
        self.assertEqual(second_pass.targets, {2: {1: {"cup"}}})

    @patch("voice_studio.vision.small_object_pass.cv2.imread")
    def test_detail_appends_without_replacing_or_resorting_confirmed_objects(self, read):
        read.return_value = np.zeros((80, 160, 3), dtype=np.uint8)
        first = {"label": "cup", "box": [.41, .6, .05, .12], "score": .55}
        second = {"label": "cup", "box": [.70, .6, .05, .12], "score": .65}
        untouched = {"timeMs": 300, "shot": 1, "objects": []}
        measured = {"timeMs": 700, "shot": 2, "objects": [first.copy()]}
        detector = Mock()
        detector.model_runs = 0
        detector.detect_region.return_value = [{**first, "score": .9}, second,
                                              {**second, "label": "laptop", "score": .99}]

        pipeline = SmallObjectPass()
        pipeline.targets[2] = {1: {"cup"}}
        pipeline.add_sample(Path("untouched.jpg"), untouched)
        pipeline.add_sample(Path("measured.jpg"), measured)
        result = pipeline.run(detector)

        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(measured["objects"][0], first)
        self.assertEqual(measured["objects"][1]["box"], second["box"])
        self.assertEqual(measured["objects"][1]["detailBasis"], "small-object-region")
        self.assertEqual(untouched["objects"], [])
        self.assertEqual(detector.detect_region.call_count, 1)
        self.assertEqual(read.call_args.args, ("measured.jpg",))

    @patch("voice_studio.vision.small_object_pass.cv2.imread", return_value=None)
    def test_missing_crop_marks_pass_partial_and_keeps_existing_results(self, read):
        detector = Mock()
        detector.model_runs = 0
        pipeline = SmallObjectPass()
        baseline = [{"label": "cup", "box": [.1, .1, .05, .1], "score": .61}]
        row = {"timeMs": 0, "shot": 0, "objects": baseline.copy()}
        pipeline.targets[0] = {0: {"cup"}}
        pipeline.add_sample(Path("missing.jpg"), row)
        result = pipeline.run(detector)
        self.assertEqual((result["status"], result["frames"]), ("partial", 0))
        self.assertEqual(row["objects"], baseline)
        detector.detect_region.assert_not_called()
