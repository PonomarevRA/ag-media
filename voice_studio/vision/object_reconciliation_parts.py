"""Bounded optional local-part measurements on original video object crops.

ORB describes observed 2D image parts. This does not reconstruct a 3D surface,
train a model, or change detector predictions. A failed read produces no match.
"""
from copy import deepcopy
from math import ceil
from pathlib import Path

import cv2
import numpy as np

from voice_studio.vision.vehicle_body_color import measure_vehicle_bodies
from voice_studio.vision.vehicle_evidence import VEHICLE_LABELS

MAX_FRAMES = 48
MAX_CROPS = 96
MAX_VEHICLE_FRAMES = 96
MAX_VEHICLE_CROPS = 144
MAX_GALLERY = 64
MAX_CANDIDATES = 16
MAX_FEATURES = 512
MAX_GAP_MS = 30_000


def spread(rows, maximum):
    """Keep deterministic time-spread input rows within a fixed measurement budget."""
    if maximum <= 1:
        return rows[-maximum:] if maximum else []
    if len(rows) <= maximum:
        return rows
    return [rows[round(index * (len(rows) - 1) / (maximum - 1))] for index in range(maximum)]


def crop_features(image, box, detector):
    """Measure ORB on a source crop resized to at most 384 pixels on either side."""
    if len(box) != 4 or not all(type(v) in (float, int) and np.isfinite(v) for v in box):
        return None
    height, width = image.shape[:2]
    x, y, w, h = box
    left, top = max(0, int(x * width)), max(0, int(y * height))
    right, bottom = min(width, ceil((x + w) * width)), min(height, ceil((y + h) * height))
    if right - left < 48 or bottom - top < 48:
        return None
    crop = image[top:bottom, left:right]
    scale = min(1, 384 / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # The central ellipse excludes box corners and margins; this is a crop mask, not segmentation.
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, (gray.shape[1] // 2, gray.shape[0] // 2),
                (int(gray.shape[1] * .45), int(gray.shape[0] * .45)), 0, 0, 360, 255, -1)
    points, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is None or len(points) < 8:
        return None
    coords = np.float32([point.pt for point in points])
    return coords, descriptors, (crop.shape[1], crop.shape[0])


def coverage(points, size):
    """Measure convex-hull extent and occupied 3x3 cells of supported crop parts."""
    width, height = size
    area = cv2.contourArea(cv2.convexHull(points)) / (width * height)
    normalized = points / np.float32([width, height])
    cells = {(min(2, int(x * 3)), min(2, int(y * 3))) for x, y in normalized}
    return round(area, 4), len(cells)


def compare_parts(first, second):
    """Verify reciprocal distinctive ORB matches with deterministic projective RANSAC.

    At least eight distributed inliers are required on both observed views.
    Central crop interiors are used; no surrounding context score is consulted.
    Bounding boxes are not segmentation: visible background inside a crop remains
    a limitation, so downstream temporal, color and ambiguity vetoes still apply.
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(first[1], second[1], k=2)
    backward = matcher.knnMatch(second[1], first[1], k=2)
    reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
               if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance}
    pairs = []
    for pair in forward:
        if len(pair) == 2:
            best, runner = pair
            if best.distance < .72 * runner.distance and reverse.get(best.trainIdx) == best.queryIdx:
                pairs.append(best)
    if len(pairs) < 8:
        return None
    a = np.float32([first[0][pair.queryIdx] for pair in pairs])
    b = np.float32([second[0][pair.trainIdx] for pair in pairs])
    # findHomography uses the calling thread's OpenCV RNG; a fixed seed keeps the pilot repeatable.
    cv2.setRNGSeed(0)
    matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, 3, maxIters=500, confidence=.995)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    chosen = mask.ravel().astype(bool)
    count, fraction = int(chosen.sum()), float(chosen.mean())
    if count < 8 or fraction < .55:
        return None
    source_area, source_cells = coverage(a[chosen], first[2])
    target_area, target_cells = coverage(b[chosen], second[2])
    if min(source_area, target_area) < .12 or min(source_cells, target_cells) < 3:
        return None
    return {'basis': 'orb-local-parts', 'inliers': count, 'inlierFraction': round(fraction, 4),
            'sourceCoverage': source_area, 'targetCoverage': target_area,
            'sourceCells': source_cells, 'targetCells': target_cells}


def candidate_views(previous, item, stamp):
    """Reserve part of the fixed search budget for visually plausible label changes."""
    named, alternatives = [], []
    width, height = item["box"][2:]
    for row in previous:
        previous_time, old, _ = row
        if not 0 < stamp - previous_time <= MAX_GAP_MS:
            continue
        if item["label"].casefold() == old["label"].casefold():
            named.append(row)
            continue
        if {item["label"], old["label"]} & {"mobile-ui", "ui", "person"}:
            continue
        other_width, other_height = old["box"][2:]
        if min(width, height, other_width, other_height) <= 0:
            continue
        aspect_ratio = (width / height) / (other_width / other_height)
        area_ratio = (width * height) / (other_width * other_height)
        if .45 <= aspect_ratio <= 2.2 and .35 <= area_ratio <= 2.85:
            alternatives.append(row)
    alternative_count = min(len(alternatives), MAX_CANDIDATES // 4)
    name_count = min(len(named), MAX_CANDIDATES - alternative_count)
    return spread(named, name_count) + spread(alternatives, min(len(alternatives), MAX_CANDIDATES - name_count))


def prepare_object_reconciliation(source: Path, report: dict) -> dict:
    """Return a copied report with bounded measured crop witnesses for optional reconciliation.

    Decode at most 48 existing source timestamps, describe at most 96 crops,
    retain 64 predecessor descriptors, compare at most 16 compatible non-simultaneous
    crops within 30 seconds, and retain at most three source witnesses per crop.
    The vehicle pass samples at most 96 timestamps and 144 vehicle crops and
    shares decoded frames with ORB when possible. Unreadable frames leave no evidence.
    Original scene predictions, boxes, labels and scores are preserved.
    """
    result = deepcopy(report)
    frames = sorted(result.get('sceneEvidence', {}).get('frames', []), key=lambda row: row['timeMs'])
    # Derived pilot pairs must be recomputed; stale copied witnesses cannot survive a failed read.
    for frame in frames:
        for item in frame.get('objects', []):
            item.pop('objectPartMatches', None)
            item.pop('vehicleBodyColor', None)
    selected = spread([row for row in frames if row.get('objects')], MAX_FRAMES)
    rows = [(frame, item) for frame in selected for item in frame.get('objects', [])[:12]]
    selected_rows = spread(rows, MAX_CROPS)
    by_stamp = {}
    for frame, item in selected_rows:
        by_stamp.setdefault(frame['timeMs'], []).append(item)
    vehicle_frames = [(frame, [item for item in frame.get('objects', ())[:12]
                              if item.get('label') in VEHICLE_LABELS]) for frame in frames]
    vehicle_rows = [(frame, item) for frame, items in spread(
        [(frame, items) for frame, items in vehicle_frames if items], MAX_VEHICLE_FRAMES) for item in items]
    vehicle_by_stamp = {}
    for frame, item in spread(vehicle_rows, MAX_VEHICLE_CROPS):
        _, items = vehicle_by_stamp.setdefault(frame["timeMs"], (frame["objects"][:12], []))
        items.append(item)
    stats = {'basis': 'bounded-source-crop-orb-pilot', 'decodedFrames': 0, 'describedCrops': 0,
             'comparisons': 0, 'measuredPairs': 0, 'maxFrames': MAX_FRAMES,
             'maxCrops': MAX_CROPS, 'maxGallery': MAX_GALLERY,
             'maxCandidates': MAX_CANDIDATES, 'status': 'no-evidence',
             'vehicleColorFrames': 0, 'vehicleColorCrops': 0}
    result['objectReconciliationEvidence'] = stats
    capture = cv2.VideoCapture(str(source))
    detector = cv2.ORB_create(nfeatures=MAX_FEATURES, edgeThreshold=12, fastThreshold=12)
    previous = []
    try:
        if not capture.isOpened():
            stats['status'] = 'source-unavailable'
            return result
        for stamp in sorted(by_stamp.keys() | vehicle_by_stamp.keys()):
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp)
            ok, image = capture.read()
            if not ok:
                continue
            if stamp in vehicle_by_stamp:
                siblings, vehicles = vehicle_by_stamp[stamp]
                stats['vehicleColorFrames'] += 1
                measure_vehicle_bodies(image, vehicles, siblings)
                stats['vehicleColorCrops'] += sum("vehicleBodyColor" in item for item in vehicles)
            if stamp not in by_stamp:
                continue
            stats['decodedFrames'] += 1
            current = []
            for item in by_stamp[stamp]:
                features = crop_features(image, item.get('box', []), detector)
                stats['describedCrops'] += 1
                if features is None:
                    continue
                matches = []
                for old_stamp, old, descriptor in candidate_views(previous, item, stamp):
                    stats['comparisons'] += 1
                    metrics = compare_parts(descriptor, features)
                    if metrics:
                        matches.append({'timeMs': old_stamp, 'label': old['label'],
                                        'box': list(old['box']), **metrics})
                matches.sort(key=lambda pair: (-pair['inliers'], -pair['inlierFraction'], pair['timeMs']))
                if matches:
                    item['objectPartMatches'] = matches[:3]
                    stats['measuredPairs'] += len(item['objectPartMatches'])
                current.append((stamp, item, features))
            previous = (previous + current)[-MAX_GALLERY:]
        if stats['measuredPairs'] or stats['vehicleColorCrops']:
            stats['status'] = 'measured'
    finally:
        capture.release()
    return result
