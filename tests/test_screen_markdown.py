import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_studio.mcp.screen_mcp import screenshot_bytes
from voice_studio.vision.screen_crop import crop_screen
from voice_studio.vision.screen_markdown import screenshot_markdown, scroll_markdown


class ScreenMarkdownTests(unittest.TestCase):
    def test_visible_rows_and_exact_agent_coordinates(self):
        unsafe_text = "A | B <script>\n### More"
        result = {
            "imageSize": [1000, 500],
            "ocr": [{"text": unsafe_text, "box": [.05, .1, .21, .04]},
                    {"text": "Search", "box": [.75, .1, .14, .04]}],
            "controlGrid": {"grid": {"columns": 12, "rows": 6},
                            "controls": [{"role": "icon_candidate", "box": [.75, .4, .04, .08],
                                          "pixelBox": [750, 200, 40, 40], "column": 10, "row": 3}]},
        }
        report = screenshot_markdown(result, "en")
        visual = report.split("### Agent coordinates", 1)[0]
        self.assertIn("| A &#124; B &#60;script&#62; &#35;&#35;&#35; More | Search |", visual)
        self.assertNotIn("| Left | Center | Right |", visual)
        self.assertIn("#### Icon? · `icon_candidate`", report)
        self.assertIn("| — | C10 R3 | `750, 200, 40, 40` |", report)
        self.assertNotIn("<script>", report)

    def test_semantic_sections_allow_different_column_counts(self):
        result = {"imageSize": [400, 560],
                  "ocr": [{"text": "Town", "box": [.45, .12, .12, .05]},
                          {"text": "Cloudy", "box": [.42, .18, .18, .05]}],
                  "interface": {"elements": [
                      {"role": "icon_candidate", "text": "Browser", "box": [.08, .14, .20, .16]},
                      {"role": "icon_candidate", "text": "Search", "box": [.75, .14, .20, .16]},
                      {"role": "field_candidate", "box": [.07, .42, .85, .10]},
                      *({"role": "icon_candidate", "text": label, "box": [x, y, .12, .09]}
                        for x, y, label in ((.1, .64, "Camera"), (.3, .64, "Video"), (.5, .64, "Chat"),
                                            (.1, .82, "Store"), (.3, .82, "Gallery"), (.5, .82, "Music"),
                                            (.7, .82, "Maps"))),
                  ]}}
        report = screenshot_markdown(result, "ru")
        self.assertIn("| Browser | Town<br>Cloudy | Search |", report)
        self.assertIn("| Поле ввода? |", report)
        self.assertIn("| Camera | Video | Chat |  |", report)
        self.assertIn("| Store | Gallery | Music | Maps |", report)

    def test_scroll_distinguishes_mosaic_coordinates_from_source(self):
        report = scroll_markdown([{
            "startMs": 19000, "endMs": 24000, "width": 500, "height": 1200,
            "mosaicText": [{"text": "Create", "box": [.1, .5, .2, .03],
                            "sourceBox": [.72, .8, .1, .04], "timeMs": 22750}],
        }])
        self.assertIn("00:00:19.000–00:00:24.000", report)
        self.assertIn("| Create | `50, 600, 100, 36` | 00:00:22.750 | `0.7200, 0.8000, 0.1000, 0.0400` |", report)

    def test_crop_uses_pixel_and_normalized_coordinates(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:size=80x60",
                            "-frames:v", "1", "-threads", "1", str(source)], check=True)
            pixel, png = crop_screen(source, [20, 15, 40, 30], "pixel", None, 1024, screenshot_bytes)
            normalized, second = crop_screen(source, [.25, .25, .5, .5], "normalized", None, 1024, screenshot_bytes)
            self.assertEqual(pixel, normalized)
            self.assertEqual(pixel, {"pixelBox": (20, 15, 40, 30), "sourceSize": (80, 60),
                                     "imageSize": (40, 30), "timeMs": None})
            self.assertEqual(png, second)
            with self.assertRaisesRegex(ValueError, "границы"):
                crop_screen(source, [60, 15, 40, 30], "pixel", None, 1024, screenshot_bytes)
