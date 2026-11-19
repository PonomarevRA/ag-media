"""Exercise screenshot structures using controlled screen geometry and OCR coordinates."""

from importlib.util import find_spec
from unittest import TestCase, skipUnless

HAS_VISION = find_spec("cv2") is not None and find_spec("numpy") is not None


def line(text, x, y, w, h, width=800, height=500):
    """Represent a measured OCR box relative to the source screenshot."""
    return {"text": text, "box": [x / width, y / height, w / width, h / height]}


@skipUnless(HAS_VISION, "OpenCV screenshot runtime required")
class ScreenshotStructureTests(TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        from voice_studio.vision.screenshot_structure import describe_structure

        self.cv2 = cv2
        self.image = np.full((500, 800, 3), 246, dtype=np.uint8)
        self.describe = describe_structure

    def test_mac_title_requires_pixel_controls(self):
        """A prominent heading alone is insufficient evidence of an OS window."""
        title = line("Recording Library", 103, 16, 153, 17)
        self.assertEqual(self.describe(self.image, [title])["windows"], [])
        for x, color in [(19, (75, 83, 255)), (46, (64, 188, 255)), (73, (75, 209, 30))]:
            self.cv2.circle(self.image, (x, 24), 7, color, -1)
        window, = self.describe(self.image, [title])["windows"]
        self.assertEqual(window["title"], "Recording Library")
        self.assertEqual(window["titleBox"], title["box"])

    def test_inactive_native_controls_need_a_continuous_chrome_divider(self):
        """Gray controls qualify with a real title-strip separator."""
        title = line("Recording Library", 103, 16, 153, 17)
        self.image[:] = 249
        for x in (19, 46, 73):
            self.cv2.circle(self.image, (x, 24), 8, (180, 180, 180), 1)
        self.assertEqual(self.describe(self.image, [title])["windows"], [])
        self.cv2.line(self.image, (0, 37), (799, 37), (211, 211, 211), 1)
        window, = self.describe(self.image, [title])["windows"]
        self.assertEqual(window["basis"], "ocr-with-inactive-window-controls")

    def test_aligned_rows_have_dividers_and_keep_heading_separate(self):
        """A saved-item list retains individual text boxes rather than absorbing headings."""
        ocr = [line("Saved recordings", 45, 72, 210, 28)]
        for y, text in [(151, "Interview sample"), (201, "Podcast outline"), (251, "Camera test")]:
            ocr.append(line(text, 59, y, 150, 17))
            self.cv2.line(self.image, (35, y + 32), (755, y + 32), (212, 212, 212), 1)
        group, = self.describe(self.image, ocr)["lists"]
        self.assertEqual(group["basis"], "aligned-ocr-with-dividers")
        self.assertEqual([row["text"] for row in group["rows"]], ["Interview sample", "Podcast outline", "Camera test"])

    def test_tightly_cropped_list_keeps_source_row_pitch(self):
        """A narrow scope preserves list evidence whose pitch is large relative to the crop."""
        rows = []
        for y, name in [(220, "First"), (263, "Second"), (306, "Third")]:
            rows.append(line(name, 59, y, 62, 15))
            self.cv2.line(self.image, (32, y + 24), (320, y + 24), (207, 207, 207), 1)
        group, = self.describe(self.image, rows, region=[.03, .42, .45, .26])["lists"]
        self.assertEqual([row["text"] for row in group["rows"]], ["First", "Second", "Third"])
        self.assertAlmostEqual(group["rows"][0]["box"][1], 220 / 500, places=4)

    def test_paragraph_is_not_a_list_by_alignment_alone(self):
        """Evenly aligned prose lacks recurring separate visual row evidence."""
        ocr = [line(f"Sentence {row}", 60, 100 + 24 * row, 250, 17) for row in range(4)]
        self.assertEqual(self.describe(self.image, ocr)["lists"], [])

    def test_fragmented_glyph_edges_do_not_mimic_list_dividers(self):
        """Text-break-like pixels with high total contrast still lack a continuous separator."""
        ocr = [line(f"Passage fragment {row}", 60, 100 + 34 * row, 250, 14) for row in range(4)]
        for y in (125, 159, 193):
            for x in range(60, 310, 7):
                self.cv2.line(self.image, (x, y), (x + 4, y), (40, 40, 40), 1)
        self.assertEqual(self.describe(self.image, ocr)["lists"], [])

    def test_timed_lanes_report_clip_boxes_without_media_timestamps(self):
        """Colored segments below a visible ruler remain bounded screenshot observations."""
        ocr = [line(text, x, 257, 35, 15) for text, x in [("0:00", 112), ("0:10", 343), ("0:20", 573)]]
        fixtures = [(100, 288, 210, "Opening"), (338, 288, 179, "Interview"),
                    (128, 333, 148, "Music"), (301, 333, 255, "Dialogue")]
        for index, (x, y, width, text) in enumerate(fixtures):
            fill = (150, 85, 31) if index < 2 else (45, 141, 176)
            self.cv2.rectangle(self.image, (x, y), (x + width, y + 25), fill, -1)
            ocr.append(line(text, x + 10, y + 7, 72, 12))
        clips = self.describe(self.image, ocr)["timelineClips"]
        self.assertEqual([clip["text"] for clip in clips], ["Opening", "Interview", "Music", "Dialogue"])
        self.assertEqual([clip["lane"] for clip in clips], [1, 1, 2, 2])
        self.assertTrue(all("startMs" not in clip for clip in clips))

    def test_single_lane_inside_scope_preserves_global_time_strip_boxes(self):
        """A cropped one-lane editor still reports original screenshot coordinates."""
        ruler = [line(text, x, 257, 35, 15) for text, x in [("0:00", 112), ("0:10", 343), ("0:20", 573)]]
        clips = [(100, 288, 210, "Opening"), (338, 288, 179, "Interview")]
        for x, y, width, text in clips:
            self.cv2.rectangle(self.image, (x, y), (x + width, y + 25), (150, 85, 31), -1)
            ruler.append(line(text, x + 10, y + 7, 72, 12))
        measured = self.describe(self.image, ruler, region=[.10, .50, .75, .20])["timelineClips"]
        self.assertEqual([clip["text"] for clip in measured], ["Opening", "Interview"])
        self.assertAlmostEqual(measured[0]["box"][0], .125, places=3)
        self.assertEqual([clip["lane"] for clip in measured], [1, 1])
        outside = self.describe(self.image, ruler, region=[0, 0, 1, .48])
        self.assertEqual(outside["timelineClips"], [])

    def test_repeated_colored_buttons_without_ruler_do_not_create_timeline(self):
        """A regular gallery of two button rows has no visible timeline evidence."""
        for y in (70, 113):
            for x in (35, 150, 265):
                self.cv2.rectangle(self.image, (x, y), (x + 94, y + 24), (141, 83, 34), -1)
        self.assertEqual(self.describe(self.image, [])["timelineClips"], [])
