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
