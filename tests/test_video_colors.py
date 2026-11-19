"""Check that caption guidance uses the actual background of the caption region."""

from unittest import TestCase

from voice_studio.vision.video_colors import PALETTE_HEIGHT, PALETTE_WIDTH, caption_colors


class CaptionColorTests(TestCase):
    def test_background_changes_suggest_an_opaque_backdrop(self):
        pixels = [((0, 0, 0) if x < PALETTE_WIDTH // 2 else (255, 255, 255))
                  for _ in range(PALETTE_HEIGHT) for x in range(PALETTE_WIDTH)]
        colors = caption_colors(pixels)
        self.assertEqual(colors["minSampledContrast"], 1)
        self.assertEqual((colors["foreground"], colors["backdrop"]), ("#FFFFFF", "#000000"))

    def test_bright_caption_area_uses_black_even_on_a_dark_frame(self):
        pixels = [((255, 255, 255) if y >= 21 else (0, 0, 0))
                  for y in range(PALETTE_HEIGHT) for _ in range(PALETTE_WIDTH)]
        colors = caption_colors(pixels)
        self.assertEqual(colors["minSampledContrast"], 21)
        self.assertEqual((colors["foreground"], colors["backdrop"]), ("#000000", None))
