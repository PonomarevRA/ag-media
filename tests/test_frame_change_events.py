"""Check native onset/peak selection and conservative direct-crop routing."""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase, skipIf

from voice_studio.vision.frame_change_events import constrain_events, group_events
from voice_studio.vision.frame_changes import (
    MAX_FRAMES_PER_SECOND, MAX_OBJECT_FRAMES_PER_SECOND, MAX_RECOGNITION_REGIONS, SIDE,
    augment_plan, bound_rows, clock_rows, crop_witnesses, scan_pixels,
)

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@skipIf(cv2 is None, "Pixel comparisons run in the visual worker environment")
class EventClockTests(TestCase):
    def test_keep_appearance_and_peak_before_thinning_on_native_clock(self):
        """Retain the real peak and never offer the return frame for direct OCR."""
        frames, samples = self.scan((0, 39, 81, 124, 166, 207, 246), (32, 32, (96, 2), 128, 128, 32, 32))
        dense = list(clock_rows(samples, frames, ()))
        events, omitted = group_events(dense, [])
        event, = events
        self.assertEqual((event["eventStartMs"], event["eventEndMs"]), (81, 207))
        self.assertEqual((event["onsetFrame"], event["peakFrame"]), (2, 3))
        self.assertEqual(event["eventBasis"], "roi-pixel-return")
        self.assertEqual(omitted, 0)
        rows, _ = bound_rows(dense, event_sampling=True)
        row, _, witness = crop_witnesses(rows, event_sampling=True)[0]
        self.assertEqual((row["sourceFrame"], witness["role"]), (3, "peak"))
        self.assertEqual(row["endMs"], 166)

    def test_cut_prevents_return_link_for_native_and_editor_boundaries(self):
        """A restored-looking image at a cut is not evidence of disappearance."""
        frames, samples = self.scan((0, 40, 80, 120), (32, 96, 32, 32))
        native = list(clock_rows(samples, frames, (2,)))
        native_events, _ = group_events(native, [80])
        self.assertFalse(native_events[0]["transient"])

        editor = list(clock_rows(samples, frames, ()))
        editor_events, _ = group_events(editor, [])
        self.assertTrue(editor_events[0]["transient"])
        report = {"events": editor_events, "rows": editor}
        constrain_events(report, [80])
        self.assertEqual(report["events"], [])
        self.assertFalse(editor[0]["eventWitnesses"])

    def test_similar_patch_return_after_half_second_is_not_direct_evidence(self):
        """Keep a long hold outside the short-event crop policy."""
        frames, samples = self.scan((0, 40, 80, 550, 590), (32, 96, 96, 32, 32))
        events, _ = group_events(list(clock_rows(samples, frames, ())), [])
        self.assertFalse(any(event["transient"] for event in events))

    def scan(self, times, shades):
        """Analyze real pixel differences without an external decoder or timestamps."""
        images = []
        for fill in shades:
            shade, step = fill if isinstance(fill, tuple) else (fill, 1)
            image = np.full((SIDE, SIDE, 3), 32, dtype=np.uint8)
            image[100:130, 140:180:step] = shade
            images.append(image)
        graphics = SimpleNamespace(observe=lambda image, index: None)
        stream = SimpleNamespace(stdout=BytesIO(b"".join(image.tobytes() for image in images)))
        samples, _, _, _, _ = scan_pixels(stream, 12, graphics, event_sampling=True)
        return list(enumerate(times)), samples


class EventBudgetTests(TestCase):
    def test_frame_caps_reserve_onset_and_peak_when_slots_fit(self):
        """Two useful witnesses survive dense traffic under the original frame limits."""
        rows = [measured_row(index, index * 20) for index in range(25)]
        for frame in (5, 6):
            rows[frame]["eventWitnesses"] = [witness(1, 100, 160, "onset" if frame == 5 else "peak")]
        for object_rows, limit in ((False, MAX_FRAMES_PER_SECOND), (True, MAX_OBJECT_FRAMES_PER_SECOND)):
            selected, omitted = bound_rows(rows, for_objects=object_rows, event_sampling=True)
            self.assertEqual(len(selected), limit)
            self.assertEqual(omitted, len(rows) - limit)
            self.assertTrue({5, 6} <= {row["sourceFrame"] for row in selected})
            stamps = [row["startMs"] for row in selected]
            self.assertEqual(stamps, sorted(stamps))

    def test_crop_budget_and_source_frame_match_in_both_event_modes(self):
        """Decode one crop per event, keep the cap, and carry exact native duration."""
        rows = []
        for offset in range(MAX_RECOGNITION_REGIONS + 5):
            event = offset * 1000
            onset, peak = (measured_row(len(rows) + step, event + stamp)
                           for step, stamp in enumerate((100, 140)))
            onset["eventWitnesses"] = [witness(offset, event + 100, event + 200, "onset")]
            peak["eventWitnesses"] = [witness(offset, event + 100, event + 200, "peak")]
            rows.extend((onset, peak))

        frames = [(row["sourceFrame"], row["startMs"]) for row in rows]
        plans = []
        for mode in ("event", "event-ocr-reuse"):
            plan = []
            augment_plan(plan, frames, set(), {"status": "ok", "mode": "objects",
                                              "sampling": mode, "rows": deepcopy(rows)})
            self.assertEqual(len(plan), MAX_RECOGNITION_REGIONS)
            for sample in plan:
                region, = sample["changeRecognitionRegions"]
                self.assertTrue(region["transient"])
                self.assertEqual(region["eventRole"], "peak")
                self.assertEqual(region["sampleEndMs"], sample["timeMs"] + 20)
                self.assertTrue(sample["supplementalOnly"])
            plans.append(plan)
        self.assertEqual(*plans)


def measured_row(frame, stamp):
    """Build one compact mock measurement in native clock coordinates."""
    return {"sourceFrame": frame, "startMs": stamp, "endMs": stamp + 20,
            "regions": [{"box": [.2, .2, .1, .1], "changedPixels": 600}]}


def witness(event, start, end, role):
    """Supply proven event provenance independently from the detection fixture."""
    return {"eventId": str(event), "eventStartMs": start, "eventEndMs": end, "eventSamples": 3,
            "eventBasis": "roi-pixel-return", "index": 0, "role": role, "transient": True}
