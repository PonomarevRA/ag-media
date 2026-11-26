"""Verify fixed frame-change OCR when optional witnesses add samples."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

from voice_studio.vision.frame_change_recognition import recognize_change_regions
from voice_studio.vision.frame_changes import augment_plan


@skipIf(cv2 is None, "OpenCV crop verification runs in the visual worker environment")
class FrameChangeRecognitionTests(TestCase):
    def test_supplemental_frame_does_not_change_fixed_crop_verification(self):
        """A supplemental frame before the fixed crop settles cannot invalidate it."""
        clock = list(enumerate((0, 500, 900, 1080, 1180, 1400, 1620, 1840)))
        box = [.1, .1, .1, .1]
        fixed = {"sourceFrame": 2, "startMs": 900, "regions": [{"box": box, "changedPixels": 400}]}
        supplemental = {
            "sourceFrame": 3, "startMs": 1080,
            "regions": [{"box": box, "changedPixels": 500}], "supplemental": True,
        }
        baseline = self.recognize(clock, [fixed])
        expanded = self.recognize(clock, [fixed, supplemental])

        fixed_result, = baseline
        self.assertEqual(fixed_result["sampleTimeMs"], 1180)
        self.assertEqual(fixed_result["verifiedUntilMs"], 1840)
        self.assertEqual(fixed_result["status"], "recognized")
        self.assertEqual([row for row in expanded if row["sourceFrame"] == 2], baseline)
        supplemental_result, = [row for row in expanded if row["sourceFrame"] == 3]
        self.assertEqual(supplemental_result["status"], "recognized")

    def recognize(self, clock, rows):
        """Run the production planner and persistence checks with deterministic OCR."""
        plan = [{"sourceFrame": ordinal, "timeMs": stamp, "shot": 0, "ocr": False}
                for ordinal, stamp in clock if ordinal != 3]
        augment_plan(plan, clock, set(), {"status": "ok", "mode": "objects", "rows": rows})

        def read_crops(jobs, *, line_limit):
            """Return a text line for each decoded crop without requiring an OCR runtime."""
            return {ordinal: [{"text": "Map"}] for ordinal, *_ in jobs}, {"status": "ok"}

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, sample in enumerate(plan):
                shade = 180 if sample["timeMs"] >= 1180 else 10
                image = np.full((192, 192, 3), shade, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / f"{index:04}.jpg"), image))
            return recognize_change_regions(None, plan, folder, recognize=read_crops)["regions"]
