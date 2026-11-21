"""Verify that stitched screen text remains anchored to an observed native frame."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from voice_studio.graph.media_graph_model import build_graph

HAS_VISION = find_spec("cv2") is not None and find_spec("numpy") is not None


def document_fixture(seed=29):
    """Draw unequal rows and text so a vertical displacement is identifiable."""
    import cv2
    import numpy as np

    image = np.full((880, 340, 3), 245, dtype=np.uint8)
    random = np.random.default_rng(seed)
    for index, top in enumerate(range(17, 850, 33)):
        x, width, shade = (int(v) for v in random.integers([15, 70, 15], [80, 230, 120]))
        cv2.rectangle(image, (x, top), (x + width, top + 6), (shade,) * 3, -1)
        cv2.putText(image, f"Content {index:02}", (x, top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (shade,) * 3, 1)
    return image


@skipUnless(HAS_VISION, "OpenCV is available in the media vision runtime")
class ScreenScrollTests(TestCase):
    def test_long_control_grid_keeps_icons_and_buttons_in_original_coordinates(self):
        """A compact glyph and a button far below it retain separate document cells."""
        import cv2
        import numpy as np
        from voice_studio.vision.screen_control_grid import control_grid

        image = np.full((1650, 300, 3), 244, dtype=np.uint8)
        cv2.circle(image, (45, 550), 14, (18, 18, 18), 3)
        cv2.line(image, (39, 550), (51, 550), (18, 18, 18), 3)
        cv2.rectangle(image, (90, 1270), (210, 1308), (18, 18, 18), 2)
        cv2.putText(image, "OK", (137, 1295), cv2.FONT_HERSHEY_SIMPLEX, .5, (18, 18, 18), 1)
        label = {"text": "OK", "box": [137 / 300, 1283 / 1650, 21 / 300, 15 / 1650]}
        result = control_grid(image, [label])
        icon = next(row for row in result["controls"] if row["role"] == "icon_candidate")
        button = next(row for row in result["controls"] if row["role"] == "button_candidate")
        self.assertEqual((icon["column"], button["column"]), (2, 7))
        self.assertLess(icon["row"], button["row"])
        self.assertEqual((result["grid"]["width"], result["grid"]["height"]), (300, 1650))

    def test_auxiliary_numeric_ocr_keeps_units_and_timestamps(self):
        from voice_studio.vision.scene_ocr import merge_lines

        readings = [{"text": value, "box": [.1, .1, .2, .2], "score": .90}
                    for value in ("30°", "97%", "1:23", "77")]
        self.assertEqual([row["text"] for row in merge_lines(readings, keep_numbers=True)],
                         ["30°", "97%", "1:23", "77"])

    def test_direction_and_unrelated_pages(self):
        from voice_studio.vision.screen_scroll import vertical_delta

        pixels = document_fixture()
        first, second = pixels[:480], pixels[123:603]
        self.assertAlmostEqual(vertical_delta(first, second), 123, delta=2)
        self.assertAlmostEqual(vertical_delta(second, first), -123, delta=2)
        self.assertEqual(vertical_delta(first, first.copy()), 0)
        self.assertIsNone(vertical_delta(first, document_fixture(890)[:480]))

    def test_overlapping_blocks_are_written_once_in_both_scroll_directions(self):
        """The join preserves the document raster across a seam inside the overlap."""
        import numpy as np
        from voice_studio.vision.screen_scroll import Page, View

        pixels = document_fixture()
        for offsets in ((0, 115, 220), (220, 115, 0)):
            views = [View(index * 200, (0, 0, 1, 1), pixels[top:top + 480])
                     for index, top in enumerate(offsets)]
            page = Page(views[0], views[0])
            for index, view in enumerate(views[1:], 1):
                page.add(view, offsets[index] - offsets[index - 1])
            np.testing.assert_array_equal(page.image(), pixels[:700])

    def test_revisiting_content_keeps_one_long_page_and_source_placements(self):
        """Scroll reversals preserve offsets without duplicating already seen pixels."""
        import numpy as np
        from voice_studio.vision.screen_scroll import Page, View

        document = document_fixture()
        offsets = [0, 145, 240, 180, 320, 400]
        views = [View(index * 200, (0, 0, 1, 1), document[top:top + 480])
                 for index, top in enumerate(offsets)]
        page = Page(views[0], views[0])
        for index, view in enumerate(views[1:], 1):
            page.add(view, offsets[index] - offsets[index - 1])
        self.assertEqual(page.extension, 400)
        self.assertEqual([offset for _, offset, _ in page.placements], offsets)
        np.testing.assert_array_equal(page.image(), document)

    def test_sticky_header_is_retained_once_for_scrolling_screenshots(self):
        """Register moving rows while keeping one fixed header and its anchor."""
        import cv2
        import numpy as np
        from voice_studio.vision.screen_scroll import collect_pages, project_reading

        document = document_fixture()
        header = document_fixture(58)[100:162]
        with TemporaryDirectory() as folder:
            candidates = []
            for index, y in enumerate((0, 95, 200, 155, 300)):
                path = Path(folder) / f"screen-{index}.png"
                cv2.imwrite(str(path), np.concatenate((header, document[y:y + 400])))
                region = {"box": [0, 0, 1, 1], "contentOnly": True}
                candidates.append((0, 200 * index, path, region))
            page, = collect_pages(candidates)
        self.assertAlmostEqual(page.header_height, 62, delta=4)
        self.assertAlmostEqual(len(page.image()), 762, delta=4)
        image = page.image()
        self.assertTrue(np.array_equal(image[6:40], header[6:40]))
        self.assertTrue(np.array_equal(image[-50:], document[650:700]))
        stamp, _, item = project_reading({"box": [.2, .02, .5, .01]}, page, len(image))
        self.assertEqual(stamp, 0)
        self.assertAlmostEqual(item["box"][1], .02 * len(image) / 462, places=3)

    def test_one_ocr_pass_projects_new_lines_onto_measured_frame_and_graph(self):
        import cv2
        from voice_studio.vision.screen_analysis import ScreenAnalysis

        document = document_fixture()
        panel = {"box": [0, 0, 1, 1], "basis": "frame-layout"}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dense = root / "screen-dense"
            dense.mkdir()
            candidates = []
            for index, top in enumerate((0, 115, 220)):
                path = (root if index else dense) / f"screen-{index}.png"
                cv2.imwrite(str(path), document[top:top + 480])
                candidates.append((2, 500 + 200 * index, path, panel))
            screens = ScreenAnalysis()
            screens.scroll_candidates = candidates[1:]
            layouts = {}
            with patch("voice_studio.vision.screen_scroll.recognize",
                       return_value=({0: [{"text": "New content", "box": [.25, .83, .20, .04]}]}, {})) as ocr:
                pages, _ = screens.scroll(layouts, candidates[:1], folder=root, include_mosaic_files=True)
            self.assertTrue((root / pages[0]["_mosaicFile"]).is_file())
        self.assertEqual(len(ocr.call_args.args[0]), 1)
        self.assertGreater(pages[0]["height"], 480)
        self.assertEqual(pages[0]["lines"], 1)
        text_stamps = [stamp for stamp, regions in layouts.items()
                       if any(row["role"] == "text_line" for region in regions for row in region["elements"])]
        self.assertEqual(text_stamps, [900])
        graph = build_graph({"sceneEvidence": {"frames": [{"timeMs": 900, "interfaces": layouts[900]}]}},
                            kind="video", duration_ms=1200)
        node, = [node for node in graph["nodes"] if node["kind"] == "ui_element" and node["data"]["role"] == "text_line"]
        self.assertEqual((node["startMs"], node["label"], node["data"]["basis"]), (900, "New content", "scroll-mosaic"))

    def test_cuts_and_gaps_break_the_scroll_route(self):
        from voice_studio.vision.screen_scroll import stable_runs

        panel = {"box": [.6, 0, .4, 1], "basis": "frame-layout"}
        samples = [(0, 0, None, panel), (1, 200, None, panel), (1, 2000, None, panel),
                   (1, 2300, None, panel)]
        runs = list(stable_runs(samples))
        self.assertEqual([(shot, [stamp for stamp, _, _ in rows]) for shot, rows in runs], [(1, [2000, 2300])])

    def test_successful_scroll_only_schedules_fixed_panel_margins(self):
        from voice_studio.vision.screen_analysis import ScreenAnalysis

        panel = (.60, .02, .36, .95)
        interior = (.62, .16, .32, .68)
        path = Path("image.jpg")
        screens = ScreenAnalysis()
        screens.candidates = [(100, path, {"box": panel}), (200, path, {"box": panel})]
        crops = []

        def collect_crop(image, target, bounds, stamp):
            crops.append((stamp, bounds))

        with (patch("voice_studio.vision.screen_analysis.cv2.imread", return_value=document_fixture()),
              patch("voice_studio.vision.screen_analysis.crop_job", side_effect=collect_crop),
              patch("voice_studio.vision.screen_analysis.outlined_text_regions", return_value=()),
              patch("voice_studio.vision.screen_analysis.recognize", return_value=({}, {})),
              patch("voice_studio.vision.screen_analysis.read_control_labels", return_value=({}, {}, 0))):
            screens.summarize({}, {}, [{"viewports": [{"timeMs": 100, "box": interior}]}])

        at_first = [box for stamp, box in crops if stamp == 100]
        self.assertEqual(len(at_first), 4)
        self.assertTrue(all(x + w <= interior[0] or x >= interior[0] + interior[2]
                            or y + h <= interior[1] or y >= interior[1] + interior[3]
                            for x, y, w, h in at_first))
        self.assertIn((200, panel), crops)
