"""Check native candidate geometry, nested controls and conservative abstention."""

from unittest import TestCase, skipIf

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@skipIf(cv2 is None, "Native UI geometry runs in the visual worker environment")
class ScreenSmallControlTests(TestCase):
    width = 320
    height = 224

    def normalized(self, box):
        """Project fixture pixels to the same source coordinates as an interface layout."""
        return [value / scale for value, scale in zip(box, (self.width, self.height, self.width, self.height))]

    def image(self):
        """Build a flat panel with a background and no visible controls."""
        return np.full((self.height, self.width, 3), 236, dtype=np.uint8)

    def layout(self):
        """Provide a verified viewport offset from the whole-image origin."""
        return {"box": self.normalized((60, 25, 218, 163)), "panelType": "mobile-ui",
                "basis": "frame-layout", "elements": [], "connections": []}

    def test_nested_native_glyph_and_button_keep_original_indices_and_geometry(self):
        """A large measured outline can enclose smaller controls without consuming them."""
        from voice_studio.vision.screen_small_controls import enrich_small_controls

        image = self.image()
        layout = self.layout()
        existing = {"role": "outlined_region", "basis": "test", "box": self.normalized((78, 45, 180, 114))}
        layout["elements"].append(existing)
        # An unlabelled X, small icon-only button, and a readable line which already has OCR evidence.
        cv2.line(image, (99, 66), (111, 78), (35, 35, 35), 2)
        cv2.line(image, (111, 66), (99, 78), (35, 35, 35), 2)
        cv2.rectangle(image, (160, 96), (222, 125), (30, 30, 30), 2)
        cv2.line(image, (181, 110), (201, 110), (35, 35, 35), 2)
        cv2.putText(image, "OK", (92, 138), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        ocr = {"role": "text_line", "basis": "ocr", "text": "OK", "box": self.normalized((90, 124, 30, 18))}
        layout["elements"].append(ocr)

        self.assertEqual(enrich_small_controls(image, layout), 2)
        self.assertEqual(layout["elements"][:2], [existing, ocr])
        glyph, button = layout["elements"][2:]
        self.assertEqual((glyph["role"], button["role"]), ("icon_candidate", "button_candidate"))
        for candidate, center in ((glyph, (105, 72)), (button, (191, 110))):
            self.assertNotIn("text", candidate)
            x, y, w, h = candidate["pixelBox"]
            self.assertTrue(x <= center[0] < x + w and y <= center[1] < y + h)
            self.assertEqual(candidate["box"], [round(value, 6) for value in self.normalized((x, y, w, h))])
            self.assertGreater(candidate["confidence"], .5)
        self.assertEqual(enrich_small_controls(image, layout), 0)

    def test_empty_outline_and_unverified_or_outside_regions_do_not_create_controls(self):
        """Blank geometry and shapes outside the verified panel remain unclassified."""
        from voice_studio.vision.screen_small_controls import enrich_small_controls

        image = self.image()
        cv2.rectangle(image, (100, 60), (150, 81), (32, 32, 32), 1)
        cv2.line(image, (21, 50), (36, 68), (20, 20, 20), 3)
        cv2.line(image, (36, 50), (21, 68), (20, 20, 20), 3)
        layout = self.layout()
        self.assertEqual(enrich_small_controls(image, layout), 0)
        layout.pop("panelType")
        self.assertEqual(enrich_small_controls(image, layout), 0)
        self.assertEqual(layout["elements"], [])
