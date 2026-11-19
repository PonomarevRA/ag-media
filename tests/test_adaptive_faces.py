"""Check detector sampling and conservative refinement on a fixed frame clock."""

from importlib.util import find_spec
from unittest import TestCase, skipUnless

HAS_VISION = find_spec("cv2") is not None and find_spec("numpy") is not None
if HAS_VISION:
    import numpy as np
    from voice_studio.vision.adaptive_faces import ScanFrame, detect_adaptively


@skipUnless(HAS_VISION, "Requires the studio's OpenCV vision runtime")
class AdaptiveFacesTests(TestCase):
    def scan(self, count, *, flashes=(), cuts=(), missing=()):
        """Prepare a small sequence and track the exact frames sent to the detector."""
        called = []

        def detect(frame):
            number = int(frame[0, 0, 0])
            called.append(number)
            if number in missing:
                return None
            return np.array([[20 + number / 4, 20, 30, 30]], dtype=np.float32)

        source = (
            ScanFrame(number, np.full((64, 64, 3), number, dtype=np.uint8), number in cuts, False,
                      np.full((64, 64), 80 if number in flashes else 0, dtype=np.uint8))
            for number in range(count)
        )
        results = list(detect_adaptively(source, detect))
        return called, results

    def test_stable_motion_reuses_anchors_without_changing_the_frame_clock(self):
        called, results = self.scan(13)
        self.assertEqual(called, [0, 6, 12])
        self.assertEqual([frame.number for frame, _, _ in results], list(range(13)))
        self.assertEqual(sum(measured for _, _, measured in results), 3)
        self.assertEqual(results[3][1][0, 0], 20.75)

    def test_brief_flash_refines_even_when_endpoint_images_look_identical(self):
        called, results = self.scan(7, flashes=(1,))
        self.assertIn(3, called)
        self.assertIn(1, called)
        self.assertTrue(results[1][2])
        self.assertNotIn(5, called)

    def test_cut_is_measured_and_missing_anchors_cannot_create_faces(self):
        called, results = self.scan(7, cuts=(2,), missing=(1, 2, 3))
        self.assertTrue({1, 2, 3}.issubset(called))
        self.assertEqual([(box, measured) for _, box, measured in results[1:4]], [(None, True)] * 3)
