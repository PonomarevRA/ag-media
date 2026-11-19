"""Allocate native object and OCR samples on the source playback clock."""

from math import ceil
from typing import Literal

FRAME_MS = 40
SECOND = 1000
MIN_OBJECT_CHECKS = 10
OBJECT_FRAME_STEP = 3
DEFAULT_OBJECT_SAMPLING = "small-detail"
SMALL_OBJECT_DETAIL = "small-object-region"
ObjectSampling = Literal["small-detail", "basic", "every-third", "shot-tiles"]
OBJECT_SAMPLING = (DEFAULT_OBJECT_SAMPLING, "basic", "every-third", "shot-tiles")


def sample_plan(frames, cut_indices, object_sampling="basic"):
    """Select native frame ordinals for object checks and independent OCR anchors per shot."""
    if not frames:
        return []
    boundaries = sorted({0, len(frames), *(index for index in cut_indices if 0 < index < len(frames))})
    plan = []
    for shot, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        plan.extend(shot_samples(frames[start:end], shot, object_sampling))
    return plan


def shot_samples(frames, shot, object_sampling):
    """Keep the every-third cadence, adding distinct checks only if the shot needs its minimum."""
    length = len(frames)
    duration = frames[-1][1] - frames[0][1]
    count = min(length, max(MIN_OBJECT_CHECKS, ceil(duration / SECOND)))
    # Midpoints give the baseline and OCR reasonable coverage near both ends.
    baseline = [(2 * slot + 1) * length // (2 * count) for slot in range(count)]
    middle = baseline[count // 2]
    phases = {baseline[count // 5]: "early", baseline[count * 4 // 5]: "late", middle: "middle"}
    objects = set(baseline)
    if object_sampling == "every-third":
        objects = set(range(0, length, OBJECT_FRAME_STEP))
        missing = min(length, MIN_OBJECT_CHECKS) - len(objects)
        if missing > 0:
            unused = [index for index in baseline if index not in objects]
            objects.update(evenly_spaced(unused, missing))
    sector = min(objects, key=lambda index: (abs(index - middle), index))
    return [{"sourceFrame": frames[index][0], "timeMs": frames[index][1], "shot": shot,
             "objects": index in objects, "sectors": index == sector, "ocr": index in baseline,
             "anchor": index == middle, "phase": phases.get(index)}
            for index in sorted(objects.union(baseline))]


def label_visible_faces(plan, timeline):
    """Attach occupancy from the existing source-clock face track for task routing."""
    step = timeline.get("stepMs", FRAME_MS)
    frames = timeline.get("frames", [])
    for row in plan:
        index = row["timeMs"] // step
        row["faces"] = [] if not 0 <= index < len(frames) else [hit[1:5] for hit in frames[index]]


def evenly_spaced(items, limit):
    """Apply a fixed budget across the whole ordered interval, including both ends."""
    if limit < 2:
        return items[:max(0, limit)]
    if len(items) <= limit:
        return items
    return [items[slot * (len(items) - 1) // (limit - 1)] for slot in range(limit)]
