"""Check camera motion on deterministic pairs with known image transforms."""

from unittest import TestCase, skipIf

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@skipIf(cv2 is None, "OpenCV camera analysis runs in the visual worker environment")
class CameraMotionTests(TestCase):
    def setUp(self):
        """Make a non-repeating background with features spread to each edge."""
        from voice_studio.vision.camera_motion import CameraMotion

        self.motion = CameraMotion()
        cv2.setNumThreads(1)
        base = np.random.default_rng(42).integers(0, 255, (240, 360, 3), dtype=np.uint8)
        self.frame = cv2.GaussianBlur(base, (3, 3), 0)

    def warped(self, dx=0, dy=0, scale=1, roll=0):
        """Apply a known pixel-space transform about the center of the image."""
        affine = cv2.getRotationMatrix2D((180, 120), roll, scale)
        affine[:, 2] += (dx, dy)
        return cv2.warpAffine(self.frame, affine, (360, 240), borderMode=cv2.BORDER_REFLECT)

    def add(self, image, time_ms, shot=0, faces=()):
        """Send a scene sample with optional detected foreground faces."""
        self.motion.add(image, {"shot": shot, "timeMs": time_ms, "faces": faces})

    def test_camera_directions_invert_the_image_shift(self):
        self.add(self.frame, 120)
        self.add(self.warped(dx=-12, dy=8), 400)
        measured, = self.motion.intervals
        self.assertEqual((measured["startMs"], measured["endMs"]), (120, 400))
        self.assertEqual((measured["horizontal"], measured["vertical"], measured["zoom"]),
                         ("right", "up", "still"))
        self.assertAlmostEqual(measured["shiftX"], -12 / 360, delta=.003)
        self.assertAlmostEqual(measured["shiftY"], 8 / 240, delta=.003)
        self.assertGreater(measured["confidence"], .75)

    def test_center_zoom_has_no_directional_shift(self):
        self.add(self.frame, 0)
        self.add(self.warped(scale=1.065), 300)
        measured, = self.motion.intervals
        self.assertAlmostEqual(measured["scale"], 1.065, delta=.008)
        self.assertEqual((measured["horizontal"], measured["vertical"], measured["zoom"]),
                         ("still", "still", "in"))

    def test_unmeasured_conditions_do_not_bridge_boundaries(self):
        self.add(self.frame, 0)
        self.add(self.warped(roll=5), 300)
        self.add(self.frame, 800, shot=1)
        self.add(self.frame, 2200, shot=1)
        self.assertEqual(self.motion.intervals, [])
        self.add(self.frame, 2500, shot=1)
        self.assertEqual([(row["startMs"], row["endMs"]) for row in self.motion.intervals], [(2200, 2500)])

    def test_moving_foreground_is_not_reported_as_camera_motion(self):
        actor = self.warped(dx=14)[:140, 66:290]
        first, second = self.frame.copy(), self.frame.copy()
        first[30:170, 48:272] = actor
        second[35:175, 76:300] = actor
        self.add(first, 0, faces=((.4, .2, .28, .26),))
        self.add(second, 200, faces=((.47, .22, .28, .26),))
        measured, = self.motion.intervals
        self.assertEqual((measured["horizontal"], measured["vertical"], measured["zoom"]),
                         ("still", "still", "still"))


@skipIf(cv2 is None, "OpenCV frame changes run in the visual worker environment")
class FrameChangeMotionTests(TestCase):
    """Check residual changes, retained screen graphics and a hard cut."""

    def setUp(self):
        """Build a textured square corresponding to the shared analysis raster."""
        from voice_studio.vision.frame_changes import SIDE

        cv2.setNumThreads(1)
        noise = np.random.default_rng(12).integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8)
        self.before = cv2.GaussianBlur(noise, (3, 3), 0)

    def test_pan_is_removed_while_a_new_graphic_keeps_current_coordinates(self):
        """An uncovered border is excluded and the rectangle stays at its real location."""
        from voice_studio.vision.frame_change_motion import CameraCompensator
        from voice_studio.vision.frame_changes import SIDE, changed_regions

        affine = np.float32([[1, 0, 9], [0, 1, 4]])
        after = cv2.warpAffine(self.before, affine, (SIDE, SIDE), borderMode=cv2.BORDER_REFLECT)
        after[101:148, 178:229] = [242, 200, 30]
        camera = CameraCompensator()
        self.assertIsNone(changed_regions(self.before, after, 24))
        regions = changed_regions(self.before, after, 24, camera=camera)
        self.assertEqual(camera.compensated_frames, 1)
        self.assertEqual(len(regions), 1)
        for value, pixel in zip(regions[0]["box"], [178, 101, 51, 47]):
            self.assertAlmostEqual(value * SIDE, pixel, delta=1)

    def test_unrelated_cut_keeps_the_global_change(self):
        """Failed background matches must never turn a cut into a local witness."""
        from voice_studio.vision.frame_change_motion import CameraCompensator
        from voice_studio.vision.frame_changes import changed_regions

        camera = CameraCompensator()
        self.assertIsNone(changed_regions(self.before, 255 - self.before, 24, camera=camera))
        self.assertEqual(camera.compensated_frames, 0)

    def test_adaptive_threshold_keeps_a_faint_local_change_on_a_quiet_frame(self):
        """Detect a local edge just below the configured fixed threshold."""
        from voice_studio.vision.frame_changes import changed_regions

        before = np.full_like(self.before, 90)
        after = before.copy()
        after[12:44, 13:50, 1] += 21
        self.assertEqual(changed_regions(before, after, 24), [])
        self.assertEqual(len(changed_regions(before, after, 24, adaptive=True)), 1)

    def test_vertical_scroll_keeps_native_changes(self):
        """A scroll in an interface retains its fixed-threshold evidence."""
        from voice_studio.vision.frame_change_motion import CameraCompensator
        from voice_studio.vision.frame_changes import SIDE, changed_regions

        before = np.full_like(self.before, 255)
        for top in range(30, 300, 42):
            cv2.putText(before, "Map / Voice / Name", (25, top),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (16, 16, 16), 1)
        after = cv2.warpAffine(before, np.float32([[1, 0, 0], [0, 1, -8]]),
                               (SIDE, SIDE), borderMode=cv2.BORDER_REPLICATE)
        native = changed_regions(before, after, 24)
        self.assertTrue(native)
        camera = CameraCompensator()
        self.assertEqual(changed_regions(before, after, 24, adaptive=True, camera=camera), native)
        self.assertEqual(camera.compensated_frames, 0)

    def test_frame_wide_exposure_drift_keeps_the_small_change(self):
        """Keep a faint rectangular change during a small simultaneous exposure shift."""
        from voice_studio.vision.frame_changes import changed_regions

        before = np.full_like(self.before, 90)
        after = before + 12
        after[70:100, 112:139] += 23
        regions = changed_regions(before, after, 24)
        self.assertTrue(regions)
        self.assertEqual(changed_regions(before, after, 24, adaptive=True), regions)
        # A subtle local change on the darker side of a brightening frame.
        after[70:100, 112:139] = 79
        self.assertEqual(changed_regions(before, after, 24), [])
        regions = changed_regions(before, after, 24, adaptive=True)
        self.assertEqual(len(regions), 1)
