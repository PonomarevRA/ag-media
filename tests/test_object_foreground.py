"""Check the opt-in foreground measurement and its conservative fallback."""

from unittest import TestCase, skipIf

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@skipIf(cv2 is None, "OpenCV object diagnostics run in the visual worker environment")
class ObjectForegroundTests(TestCase):
    box = (.18, .20, .64, .60)

    def pair(self):
        """Build an identical textured shape inside two contrasting backgrounds."""
        colors = ((33, 70, 92), (135, 130, 35))
        images = []
        for color in colors:
            frame = np.full((160, 220, 3), color, dtype=np.uint8)
            cv2.ellipse(frame, (110, 80), (58, 24), 20, 0, 360, (46, 38, 173), -1)
            cv2.circle(frame, (100, 74), 8, (26, 20, 220), -1)
            cv2.circle(frame, (125, 85), 5, (64, 49, 112), -1)
            images.append(frame)
        return images

    def test_stable_foreground_removes_context_from_similarity(self):
        """Provide interpretable baseline and masked scores from the same raster."""
        from voice_studio.vision.object_foreground import compare_foregrounds, describe_foreground

        first, second = (describe_foreground(image, self.box) for image in self.pair())
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        measured = compare_foregrounds(first, second)
        self.assertIsNotNone(measured)
        self.assertEqual(measured["basis"], "grabcut-foreground")
        self.assertGreater(measured["maskedObjectSimilarity"], .95)
        self.assertGreater(measured["maskedObjectSimilarity"], measured["baselineObjectSimilarity"])
        self.assertGreater(measured["maskOverlap"], .85)
        for key in ("sourceFraction", "targetFraction", "sourceStability", "targetStability"):
            self.assertGreater(measured[key], 0)
            self.assertLessEqual(measured[key], 1)

    def test_uniform_or_invalid_crop_produces_no_evidence(self):
        """An empty segmentation, missing input or too-small box cannot vote."""
        from voice_studio.vision.object_foreground import compare_foregrounds, describe_foreground

        frame = np.full((160, 220, 3), 75, dtype=np.uint8)
        self.assertIsNone(describe_foreground(frame, self.box))
        self.assertIsNone(describe_foreground(frame, (.5, .5, .02, .02)))
        self.assertIsNone(describe_foreground(frame, (-.1, .5, .4, .3)))
        self.assertIsNone(describe_foreground(frame, (0, 0, 1, 1)))
        self.assertIsNone(describe_foreground(None, self.box))
        self.assertIsNone(compare_foregrounds(None, None))

    def test_displaced_masks_do_not_produce_an_overlap_score(self):
        """Do not align, stretch or invent a score when foregrounds do not intersect."""
        from voice_studio.vision.object_foreground import ForegroundDescription, compare_foregrounds

        gray = np.arange(84 * 84, dtype=np.float32).reshape(84, 84)
        inside = np.ones(gray.shape, dtype=bool)
        left = np.zeros(gray.shape, dtype=bool)
        left[:, :40] = True
        right = np.zeros(gray.shape, dtype=bool)
        right[:, 44:] = True
        first = ForegroundDescription(gray, gray, inside, left, .47, .95)
        second = ForegroundDescription(gray, gray, inside, right, .47, .95)
        self.assertIsNone(compare_foregrounds(first, second))
