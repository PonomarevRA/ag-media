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
    def test_direction_and_unrelated_pages(self):
        from voice_studio.vision.screen_scroll import vertical_delta

        pixels = document_fixture()
        first, second = pixels[:480], pixels[123:603]
        self.assertAlmostEqual(vertical_delta(first, second), 123, delta=2)
        self.assertAlmostEqual(vertical_delta(second, first), -123, delta=2)
        self.assertEqual(vertical_delta(first, first.copy()), 0)
        self.assertIsNone(vertical_delta(first, document_fixture(890)[:480]))

    def test_one_ocr_pass_projects_new_lines_onto_measured_frame_and_graph(self):
        import cv2
        from voice_studio.vision.screen_analysis import ScreenAnalysis

        document = document_fixture()
        panel = {"box": [0, 0, 1, 1], "basis": "frame-layout"}
        with TemporaryDirectory() as folder:
            candidates = []
            for index, top in enumerate((0, 115, 220)):
                path = Path(folder) / f"screen-{index}.png"
                cv2.imwrite(str(path), document[top:top + 480])
                candidates.append((2, 500 + 200 * index, path, panel))
            screens = ScreenAnalysis()
            screens.scroll_candidates = candidates
            layouts = {}
            with patch("voice_studio.vision.screen_scroll.recognize",
                       return_value=({0: [{"text": "New content", "box": [.25, .83, .20, .04]}]}, {})) as ocr:
                pages, _ = screens.scroll(layouts)
        self.assertEqual(len(ocr.call_args.args[0]), 1)
        self.assertGreater(pages[0]["height"], 480)
        self.assertEqual(pages[0]["lines"], 1)
        self.assertEqual(list(layouts), [900])
        graph = build_graph({"sceneEvidence": {"frames": [{"timeMs": 900, "interfaces": layouts[900]}]}},
                            kind="video", duration_ms=1200)
        node, = [node for node in graph["nodes"] if node["kind"] == "ui_element"]
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
