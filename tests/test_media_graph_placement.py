"""Check replacement scale and prevent guesses for clipped geometry."""

from unittest import TestCase

from voice_studio.graph.media_graph_placement import placement_plan


class ObjectPlacementTests(TestCase):
    def test_fit_uses_visible_alpha_aspect_and_grounds_at_the_box_base(self):
        plan = placement_plan((1000, 1600), [.3, .5, .2, .2], (12, 20, 40, 80), height_ratio=.75)
        self.assertEqual(plan["anchorPx"], [400, 1120])
        self.assertEqual(plan["placedBoxPx"], [340, 880, 120, 240])
        self.assertEqual(plan["calibration"], "observed")

    def test_corner_clip_requests_both_calibrations(self):
        observed = [.85, .8, .15, .2]
        box = (0, 0, 50, 100)
        plan = placement_plan((1000, 1600), observed, box)
        self.assertEqual(plan["required"], ["height_px", "anchor_px"])
        plan = placement_plan((1000, 1600), observed, box, height_px=275, anchor_px=[963, 1580])
        self.assertEqual(plan["placedBoxPx"], [894, 1305, 138, 275])
        self.assertTrue(plan["placementClipped"])

    def test_side_clip_retains_measured_full_height(self):
        plan = placement_plan((1000, 1200), [.86, .4, .14, .2], (0, 0, 50, 100), anchor_px=[930, 720])
        self.assertEqual(plan["placedBoxPx"], [870, 480, 120, 240])
        self.assertEqual(plan["calibration"], "manual")


    def test_clipped_asset_can_have_a_base_below_the_frame(self):
        plan = placement_plan((1000, 1600), [.85, .8, .15, .2], (0, 0, 50, 100),
                              height_px=325, anchor_px=[975, 1650])
        self.assertEqual(plan["anchorPx"], [975, 1650])
        self.assertTrue(plan["placementClipped"])
