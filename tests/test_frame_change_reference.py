"""Exercise strong, accumulated and bounded source-frame change selection."""

from copy import deepcopy
from unittest import TestCase, skipUnless

try:
    import numpy as np
    import cv2
except ImportError:
    cv2 = None

from voice_studio.vision.frame_change_events import REFERENCE_BASIS, group_events
from voice_studio.vision.frame_changes import MAX_FRAMES_PER_SECOND, augment_plan, bound_rows, reference_cut_guard


@skipUnless(cv2 is not None, "OpenCV vision runtime is required")
class ReferenceTests(TestCase):
    def test_jump_is_immediate_and_weak_change_needs_confirmation(self):
        """Require two weak changes relative to a stable patch before emitting pixels."""
        from voice_studio.vision.frame_change_reference import StableRegionChanges

        base = np.zeros((32, 32, 3), dtype=np.uint8)

        def changed(level):
            """Set one coherent, compact change within a tile."""
            frame = base.copy()
            frame[3:13, 3:13] = level
            return frame

        scan = StableRegionChanges(32, 16, 24)
        self.assertIsNotNone(scan.observe(base, changed(80), broad=False))
        self.assertEqual(scan.summary()["strongTiles"], 1)
        scan.reset(base)
        mild, next_mild = changed(15), changed(30)
        self.assertIsNone(scan.observe(base, mild, broad=False))
        output = scan.observe(mild, next_mild, broad=False)
        self.assertEqual(np.count_nonzero(output == 0), 100)
        self.assertEqual(scan.summary()["confirmedWeakTiles"], 1)
        self.assertIsNone(scan.observe(next_mild, changed(45), broad=True))
        self.assertIsNone(scan.observe(changed(45), changed(60), broad=False))

    def test_sharper_followup_can_win_before_thinning(self):
        """Choose a measured sharp frame on the same region and preserve the onset."""
        rows = []
        for frame, sharpness, pixels in ((1, 9.0, 100), (2, 30.0, 77)):
            stamp = frame * 33
            rows.append({"sourceFrame": frame, "startMs": stamp, "endMs": stamp + 33,
                         "regions": [{"box": [.2, .2, .1, .1], "changedPixels": pixels,
                                      "sharpness": sharpness, "detectionBasis": REFERENCE_BASIS}],
                         "_pixelSignatures": [None]})
        events, _ = group_events(rows, [])
        self.assertEqual((events[0]["onsetFrame"], events[0]["peakFrame"]), (1, 2))
        self.assertEqual(events[0]["eventBasis"], REFERENCE_BASIS)
        split_events, _ = group_events(deepcopy(rows), [66])
        self.assertEqual(len(split_events), 2)
        selected, _ = bound_rows(rows, event_sampling=True, stable_reference=True)
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(len(selected), MAX_FRAMES_PER_SECOND)

    def test_extra_reference_crop_keeps_ordinary_ocr_clock_unchanged(self):
        """An unscheduled reference crop stays out of ordinary persistence checks."""
        frames = [(0, 0), (1, 1080)]
        plan = [{"sourceFrame": 0, "timeMs": 0, "shot": 0}]
        reference_only = {"sourceFrame": 1, "startMs": 1080, "endMs": 1113,
                          "regions": [{"box": [.2, .2, .15, .15], "changedPixels": 700,
                                       "detectionBasis": REFERENCE_BASIS, "referenceOnly": True}]}
        report = {"status": "ok", "mode": "objects", "sampling": "stable-events", "rows": [reference_only]}
        augment_plan(plan, frames, set(), report)
        self.assertEqual(len(plan), 2)
        self.assertNotIn("supplementalOnly", plan[0])
        self.assertTrue(plan[1]["supplementalOnly"])
        self.assertTrue(plan[1]["changeRecognitionRegions"][0]["supplemental"])

    def test_guard_keeps_adjacent_evidence_and_drops_cross_cut_reference(self):
        """New reference-only data cannot enter the cut's warm-up frames."""
        normal = {"box": [.2, .2, .15, .15], "changedPixels": 700}
        reference = {**normal, "referenceOnly": True, "detectionBasis": REFERENCE_BASIS}
        frames = [(index + 80, index * 33) for index in range(7)]
        rows = [{"sourceFrame": 82, "startMs": 66, "endMs": 99, "regions": [reference, normal],
                 "eventWitnesses": [{"eventId": "lost", "index": 0}, {"eventId": "kept", "index": 1}]},
                {"sourceFrame": 86, "startMs": 198, "endMs": 231, "regions": [reference],
                 "eventWitnesses": [{"eventId": "lost", "index": 0}]}]
        guarded, invalid = reference_cut_guard(deepcopy(rows), frames, {1})
        self.assertEqual(invalid, {"lost"})
        self.assertEqual(guarded[0]["regions"], [normal])
        self.assertEqual(guarded[0]["eventWitnesses"], [{"eventId": "kept", "index": 0}])
        self.assertEqual(guarded[1]["regions"], [reference])
        self.assertFalse(guarded[1]["eventWitnesses"])
