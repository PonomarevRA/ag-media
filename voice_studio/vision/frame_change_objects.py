"""Use measured regions as additional, bounded detector requests."""

from math import ceil, floor
from bisect import bisect_left, bisect_right
from collections import defaultdict
from time import perf_counter

from voice_studio.vision.frame_changes import BASIS
from voice_studio.vision.object_appearance import ObjectAppearance, retain_confirmed_matches
from voice_studio.vision.scene_objects import STRONG_SCORE, containing_fraction, iou, unique_proposals
from voice_studio.vision.small_object_pass import already_detected


MAX_ADDITIONS = 4


def crop_bounds(box, width, height):
    """Give a small change enough original context without extending beyond the image."""
    x, y, w, h = box
    span_x = max(w * 2.2, .12)
    span_y = max(h * 2.2, .12)
    cx, cy = x + w / 2, y + h / 2
    return (max(0, floor((cx - span_x / 2) * width)),
            max(0, floor((cy - span_y / 2) * height)),
            min(width, ceil((cx + span_x / 2) * width)),
            min(height, ceil((cy + span_y / 2) * height)))


def region_objects(detector, image, regions, existing):
    """Accept confident boxes touching the changed pixels; keep original observations intact."""
    height, width = image.shape[:2]
    candidates = []
    for region in regions:
        box = region["box"]
        bounds = crop_bounds(box, width, height)
        for item in detector.detect_region(image, bounds):
            hit = item["box"]
            if (item["score"] >= STRONG_SCORE and hit[2] * hit[3] <= .12
                    and max(containing_fraction(box, hit), containing_fraction(hit, box)) >= .25):
                candidates.append(item)
    added = []
    for item in unique_proposals(candidates):
        if already_detected(item, existing) or already_detected(item, added):
            continue
        if any(iou(item["box"], prior["box"]) >= .72 for prior in existing):
            continue
        added.append({**item, "box": [round(value, 4) for value in item["box"]],
                      "basis": "score", "detailBasis": BASIS})
        if len(added) == MAX_ADDITIONS:
            break
    return added


def panel_checks(frames, interfaces):
    """Index measured panels, including explicit negatives at sampled times."""
    checks = defaultdict(list)
    for frame in frames:
        if frame.get("panelChecked"):
            stamp = frame["timeMs"]
            boxes = [item["box"] for item in interfaces.get(stamp, ()) if item.get("panelType")]
            checks[frame["shot"]].append((stamp, boxes))
    return checks


def physical_regions(regions, checks, stamp):
    """Skip changes enclosed by a nearby verified panel and prioritize compact crops."""
    at = bisect_left(checks, stamp, key=lambda row: row[0])
    nearest = min(checks[max(0, at - 1):at + 1], key=lambda row: abs(row[0] - stamp), default=None)
    panels = nearest[1] if nearest and abs(nearest[0] - stamp) <= 600 else ()
    eligible = [region for region in regions if region["box"][2] * region["box"][3] <= .06
                and not any(containing_fraction(panel, region["box"]) >= .90 for panel in panels)]
    return eligible[:1]


def attach_anchor_matches(frames, locations, changed, folder, cv2):
    """Measure appearance across baseline and supplemental frames without changing baseline matches."""
    stamps = sorted(frame["timeMs"] for frame in changed)
    appearance = ObjectAppearance(min_overlap=0, nearby_ms=160)
    for frame in frames:
        stamp = frame["timeMs"]
        if not frame["objects"] or bisect_left(stamps, stamp - 160) == bisect_right(stamps, stamp + 160):
            continue
        image = cv2.imread(str(folder / f"{locations[frame['sourceFrame']]:04}.jpg"))
        if image is None:
            continue
        originals = frame["objects"][:12]
        copies = [{key: value for key, value in item.items() if key != "objectMatches"} for item in originals]
        appearance.add(image, {**frame, "objects": copies}, mixed_passes=True)
        for item, measured in zip(originals, copies):
            if matches := measured.get("objectMatches"):
                matches = [{**match, "comparisonBasis": "frame-change-anchor"} for match in matches]
                prior = item.get("objectMatches", [])
                if item.get("detailBasis") == BASIS:
                    item["objectMatches"] = [*matches, *prior]
                else:
                    item["objectMatches"] = [*prior[:2], *matches, *prior[2:]]


def review_change_objects(detector, samples, frames, folder, *, interfaces=None):
    """Run extra crops after the baseline so they cannot change its confirmation decisions."""
    import cv2

    checks = [(index, sample) for index, sample in enumerate(samples) if sample.get("changeRegions")]
    if not checks:
        return None
    if detector is None:
        return {"status": "unavailable", "frames": 0, "added": 0, "modelRuns": 0}
    started = perf_counter()
    runs = detector.model_runs
    frames_by_number = {frame["sourceFrame"]: frame for frame in frames}
    appearance = ObjectAppearance(min_overlap=0, nearby_ms=160)
    assisted = interfaces is not None
    panels = panel_checks(frames, interfaces) if assisted else {}
    checked = added = skipped = 0
    changed = []
    status = "ok"
    for index, sample in checks:
        regions = sample["changeRegions"]
        if assisted:
            regions = physical_regions(regions, panels.get(sample["shot"], ()), sample["timeMs"])
        if not regions:
            skipped += 1
            continue
        frame = frames_by_number.get(sample["sourceFrame"])
        image = cv2.imread(str(folder / f"{index:04}.jpg")) if frame is not None else None
        if image is None:
            status = "partial"
            continue
        try:
            objects = region_objects(detector, image, regions, frame["objects"])
            if objects:
                appearance.add(image, {**frame, "objects": objects})
                frame["objects"].extend(objects)
                changed.append(frame)
            checked += 1
            added += len(objects)
        except (cv2.error, KeyError, ValueError):
            status = "partial"
    if assisted and changed:
        locations = {sample["sourceFrame"]: index for index, sample in enumerate(samples)}
        attach_anchor_matches(frames, locations, changed, folder, cv2)
    retain_confirmed_matches(frames)
    return {"status": status, "frames": checked, "added": added,
            "modelRuns": detector.model_runs - runs, "skippedFrames": skipped,
            "elapsedMs": round(1000 * (perf_counter() - started))}
