"""Read the contents of selected frame-change crops without inventing objects."""

from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from math import ceil, floor
import subprocess
from time import perf_counter


def recognition_bounds(box, width, height):
    """Add a small context margin without leaving the decoded source frame."""
    x, y, w, h = box
    return (max(0, floor((x - .12 * w) * width)), max(0, floor((y - .12 * h) * height)),
            min(width, ceil((x + 1.12 * w) * width)), min(height, ceil((y + 1.12 * h) * height)))


def persistent_crop(samples, stamp, box, rasters):
    """Find a settled crop that still matches a later check on the same shot."""
    import cv2
    import numpy as np

    slot = bisect_left(samples, (stamp + 130, -1))
    if slot >= len(samples) or samples[slot][0] > stamp + 445:
        return None
    settled, location = samples[slot]

    def patch(index):
        """Sample a fixed crop on a shared low-resolution coordinate grid."""
        raster = rasters(index)
        if raster is None:
            return None
        x, y, w, h = box
        return raster[round(y * 192):round((y + h) * 192), round(x * 192):round((x + w) * 192)]

    reference = patch(location)
    if reference is None or not reference.size:
        return None
    earlier_index = bisect_left(samples, (stamp - 220, -1)) - 1
    if earlier_index >= 0 and samples[earlier_index][0] >= stamp - 800:
        earlier = patch(samples[earlier_index][1])
        if earlier is not None and np.mean(cv2.absdiff(reference, earlier).max(axis=2) < 24) >= .72:
            return None
    checked = settled
    while checked < settled + 2100:
        next_slot = bisect_left(samples, (checked + 175, -1))
        if next_slot >= len(samples):
            break
        next_stamp, index = samples[next_slot]
        if next_stamp - checked > 620 or next_stamp > settled + 2400:
            break
        future = patch(index)
        if future is None or np.mean(cv2.absdiff(reference, future).max(axis=2) < 24) < .72:
            break
        checked = next_stamp
    return (settled, location, checked) if checked > settled else None


def recognize_change_regions(detector, samples, folder, *, recognize):
    """Run object detection and a single bounded OCR batch on selected source crops."""
    import cv2
    from voice_studio.vision.scene_objects import STRONG_SCORE, containing_fraction

    started = perf_counter()
    jobs, rows = [], []
    runs = detector.model_runs if detector else 0
    per_shot = defaultdict(list)
    for index, sample in enumerate(samples):
        per_shot[sample["shot"]].append((sample["timeMs"], index))

    @lru_cache(maxsize=24)
    def raster(index):
        """Decode small comparison images once for neighboring requests."""
        image = cv2.imread(str(folder / f"{index:04}.jpg"), cv2.IMREAD_REDUCED_COLOR_4)
        return cv2.resize(image, (192, 192)) if image is not None else None

    checked = 0
    for sample in samples:
        regions = sample.get("changeRecognitionRegions")
        if not regions:
            continue
        for region in regions:
            box = region["box"]
            checked += 1
            persisted = persistent_crop(per_shot[sample["shot"]], sample["timeMs"], box, raster)
            if not persisted:
                continue
            stamp, location, until = persisted
            image = cv2.imread(str(folder / f"{location:04}.jpg"))
            if image is None:
                continue
            height, width = image.shape[:2]
            bounds = recognition_bounds(box, width, height)
            x, y, right, bottom = bounds
            row = {"sourceFrame": sample["sourceFrame"], "index": region["index"],
                   "objects": [], "text": [], "status": "unclassified",
                   "sampleTimeMs": stamp, "verifiedUntilMs": until}
            if detector:
                try:
                    objects = [item for item in detector.detect_region(image, bounds)
                               if item["score"] >= STRONG_SCORE
                               and containing_fraction(box, item["box"]) >= .45]
                    row["objects"] = sorted(objects, key=lambda item: -item["score"])[:3]
                except (cv2.error, KeyError, ValueError):
                    row["detectorStatus"] = "failed"
            ordinal = len(rows)
            rows.append(row)
            file = folder / f"change-ocr-{ordinal:03}.jpg"
            crop = image[y:bottom, x:right]
            if cv2.imwrite(str(file), crop):
                normalized = (x / width, y / height, (right - x) / width, (bottom - y) / height)
                jobs.append((ordinal, file, normalized, right - x, bottom - y))
    try:
        readings, ocr = recognize(jobs, line_limit=6)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        readings, ocr = {}, {"status": "failed"}
    for index, row in enumerate(rows):
        row["text"] = readings.get(index, [])
        if row["text"] or row["objects"]:
            row["status"] = "recognized"
    return {"regions": rows, "ocr": ocr, "ocrCrops": len(jobs),
            "detectorRuns": detector.model_runs - runs if detector else 0, "candidatesChecked": checked,
            "elapsedMs": round((perf_counter() - started) * 1000)}


def attach_change_recognition(report, measured):
    """Attach only exact frame-and-region matches to the raw measurement layer."""
    lookup = {(row["sourceFrame"], row["index"]): row for row in measured.get("regions", ())}
    for row in report.get("rows", ()):
        for index, region in enumerate(row["regions"]):
            if facts := lookup.get((row["sourceFrame"], index)):
                keys = ("status", "text", "objects", "sampleTimeMs", "verifiedUntilMs")
                region["recognition"] = {key: facts[key] for key in keys if key in facts}
    report["recognition"] = {key: value for key, value in measured.items() if key != "regions"}
    report["recognition"]["reviewedRegions"] = len(lookup)
