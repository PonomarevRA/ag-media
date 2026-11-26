"""Verify source-clock UI field reads without conflating neighboring states."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np

from voice_studio.vision.screen_state_ocr import (
    PanelFrame, StateCrop, read_interface_state_ocr, reading_repeats, verified_plans,
)


PANEL = {"box": [.53, 0, .47, 1], "panelType": "mobile-ui"}
HEADER = [.57, .078, .29, .05]
FIELD = [.70, .50, .16, .08]


class InterfaceStateOCRTests(TestCase):
    def setUp(self):
        scratch = self.enterContext(TemporaryDirectory())
        self.folder = Path(scratch)

    def capture(self, stamp, value, *, shot=0, source=None):
        """Give fixed header glyphs and one field a distinct observed pixel signature."""
        image = np.full((480, 720, 3), 246, dtype=np.uint8)
        image[:, :382] = stamp % 220
        cv2.putText(image, "Account", (410, 61), cv2.FONT_HERSHEY_SIMPLEX, .85, (4, 4, 4), 2)
        cv2.rectangle(image, (505, 241), (617, 278), (value, value, value), -1)
        path = self.folder / f"frame-{shot}-{stamp}.png"
        self.assertTrue(cv2.imwrite(str(path), image))
        return shot, stamp, path, PANEL, stamp // 25 if source is None else source

    def read_fields(self, jobs, **_options):
        """Return only fields whose actual bounding boxes fit in the selected crop."""
        outputs = {}
        for key, path, bounds, _, _ in jobs:
            lines = []
            left, top, _width, height = bounds
            if top <= HEADER[1] and HEADER[1] + HEADER[3] <= top + height:
                lines.append({"text": "Account", "box": HEADER})
            if top <= FIELD[1] and FIELD[1] + FIELD[3] <= top + height:
                image = cv2.imread(str(path))
                x = round((.77 - left) / bounds[2] * image.shape[1])
                y = round((.539 - top) / height * image.shape[0])
                value = int(image[y, x, 0])
                lines.append({"text": "Price 1" if value < 100 else "Price 2", "box": FIELD})
            if lines:
                outputs[key] = lines
        return outputs, {"tesseract": "ok"}

    def test_stable_header_single_read_and_changed_field_separate(self):
        """A changed field gets its own exact witness and does not repeat the header."""
        frames = [self.capture(stamp, value) for stamp, value in ((0, 20), (100, 20), (200, 190), (300, 190))]
        observations, report = read_interface_state_ocr(
            self.folder, frames, {0: [PANEL]}, recognize=self.read_fields)
        lines = [line for crop in observations for line in crop["elements"]]
        header, = [line for line in lines if line["text"] == "Account"]
        first, second = sorted((line for line in lines if line["text"].startswith("Price")),
                               key=lambda row: row["verifiedFromMs"])
        self.assertEqual((header["verifiedFromMs"], header["verifiedUntilMs"]), (0, 301))
        self.assertEqual((first["text"], first["verifiedFromMs"], first["verifiedUntilMs"]), ("Price 1", 0, 101))
        self.assertEqual((second["text"], second["verifiedFromMs"], second["verifiedUntilMs"]), ("Price 2", 200, 301))
        for crop in observations:
            self.assertLessEqual(crop["startMs"], crop["timeMs"])
            self.assertLess(crop["timeMs"], crop["endMs"])
            self.assertEqual(crop["sourceFrame"], crop["timeMs"] // 25)
            self.assertTrue(all(row["recognizedAtMs"] == crop["timeMs"] for row in crop["elements"]))
        self.assertEqual(report["ocrStatuses"], {"tesseract": "ok"})

    def test_unseen_gap_and_cut_cannot_extend_time_support(self):
        """Split verified time ranges at a gap or a new shot and honor the crop limit."""
        frames = [self.capture(stamp, 20, shot=shot) for stamp, shot in (
            (0, 0), (100, 0), (500, 0), (600, 0), (700, 1), (800, 1))]
        observations, report = read_interface_state_ocr(
            self.folder, frames, {0: [PANEL], 700: [PANEL]}, recognize=self.read_fields)
        self.assertEqual([(crop["shot"], crop["startMs"], crop["endMs"]) for crop in observations],
                         [(0, 0, 101), (0, 500, 601), (1, 700, 801)])
        limited, status = read_interface_state_ocr(
            self.folder, frames, {0: [PANEL], 700: [PANEL]}, recognize=self.read_fields, max_crops=2)
        self.assertEqual((len(limited), status["ocrCrops"], report["ocrCrops"]), (2, 2, 3))

    def test_unverified_panel_skips_ocr(self):
        """Do not classify or read a moving image on unverified panel-like geometry."""
        frames = [self.capture(0, 20), self.capture(100, 190)]
        worker = Mock()
        observations, report = read_interface_state_ocr(self.folder, frames, {}, recognize=worker)
        self.assertEqual((observations, report["status"], report["ocrCrops"]), ([], "no-candidates", 0))
        worker.assert_not_called()

    def test_small_changed_digit_requires_independent_source_witness(self):
        """A one-digit edit splits a field while the rest of the panel stays stable."""
        frames = []
        panel = {"box": [0, 0, 1, 1], "panelType": "mobile-ui"}
        for stamp, digits in ((0, "12"), (100, "12"), (200, "13"), (300, "13")):
            image = np.full((720, 378, 3), 245, dtype=np.uint8)
            cv2.putText(image, digits, (220, 270), cv2.FONT_HERSHEY_SIMPLEX, .5, (12, 12, 12), 1, cv2.LINE_AA)
            path = self.folder / f"number-{stamp}.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            frames.append((0, stamp, path, panel, stamp // 25))
        boxes = [crop for crop in verified_plans(frames, {0: [panel]})
                 if crop.region[1] <= .367 < crop.region[1] + crop.region[3]]
        self.assertEqual([(crop.start, crop.last) for crop in boxes], [(0, 100), (200, 300)])
        initial, edited = boxes
        left, _, width, _ = edited.region
        self.assertEqual(initial.region[2], 1)
        self.assertLessEqual(width, .5)
        self.assertLess(left, 220 / 378)
        self.assertGreater(left + width, 242 / 378)
        self.assertIn(edited.representative.source_frame, (8, 12))

        transient = [crop for crop in verified_plans(frames[:3], {0: [panel]})
                     if crop.region[1] <= .367 < crop.region[1] + crop.region[3] and crop.kind == "changed"]
        self.assertEqual([crop.region[2] for crop in transient], [1])

    def test_registered_repeated_label_has_its_own_timestamp_and_box(self):
        """Only the spatially supported copy of repeated text may link to a previous reading."""
        rng = np.random.default_rng(9)
        pixels = rng.integers(0, 255, (192, 160, 3), dtype=np.uint8)
        first = PanelFrame(0, self.folder / "prior.png", 0, pixels)
        before = StateCrop(0, [0, 0, 1, 1], "mobile-ui", (0, 0, 1, 1), 0, first, 0, 0, 1, 1.0, "changed")
        second = PanelFrame(100, self.folder / "current.png", 3, np.roll(pixels, -19, axis=0))
        after = replace(before, representative=second, start=100, last=100)
        left = {"text": "Continue", "box": [.12, .6, .22, .045], "recognizedAtMs": 0, "sourceFrame": 0}
        linked = {"text": "Continue", "box": [.12, .5, .22, .045], "recognizedAtMs": 100, "sourceFrame": 3}
        separate = {"text": "Continue", "box": [.62, .5, .22, .045], "recognizedAtMs": 100, "sourceFrame": 3}
        stats = reading_repeats([(before, [left]), (after, [linked, separate])])
        self.assertEqual(linked["sameContentAs"], {"box": left["box"], "recognizedAtMs": 0, "sourceFrame": 0})
        self.assertEqual((linked["box"], linked["recognizedAtMs"]), ([.12, .5, .22, .045], 100))
        self.assertNotIn("sameContentAs", separate)
        self.assertEqual((stats["repeatedLabels"], stats["confirmedRepeats"], stats["unverifiedRepeats"]), (2, 1, 1))
