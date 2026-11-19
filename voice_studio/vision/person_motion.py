"""Estimate shot-local person motion and relative depth from existing geometry."""

from bisect import bisect_left, bisect_right
from collections import Counter
from math import isfinite
from statistics import median

WINDOW_MS = 400
MIN_WINDOW_MS = 160
MAX_CAMERA_WINDOW_MS = 1200
CAMERA_CONFIDENCE = .60


def finite_number(value):
    """Reject booleans and non-finite values from persisted evidence."""
    return type(value) in (int, float) and isfinite(value)


def valid_box(values):
    """Accept normalized, unclipped geometry from the oriented source frame."""
    if not isinstance(values, (list, tuple)) or len(values) != 4 or not all(map(finite_number, values)):
        return None
    x, y, width, height = values
    if x >= .008 and y >= .008 and width > 0 and height > 0 and x + width <= .992 and y + height <= .992:
        return values
    return None


def window_bounds(start, end):
    """Split areas without a camera measurement into short screen-space intervals."""
    while end - start >= MIN_WINDOW_MS:
        stop = min(start + WINDOW_MS, end)
        yield start, stop, None
        start = stop


def motion_windows(cuts, camera_samples):
    """Respect shot cuts and reserve reliable camera transforms for their exact intervals."""
    if not isinstance(camera_samples, (list, tuple)):
        camera_samples = ()
    camera = sorted((row["startMs"], row["endMs"], order, row)
                    for order, row in enumerate(camera_samples)
                    if isinstance(row, dict) and type(row.get("startMs")) is int and type(row.get("endMs")) is int
                    and MIN_WINDOW_MS <= row["endMs"] - row["startMs"] <= MAX_CAMERA_WINDOW_MS
                    and all(finite_number(row.get(key)) for key in ("confidence", "shiftX", "shiftY", "scale"))
                    and CAMERA_CONFIDENCE <= row["confidence"] <= 1 and .5 <= row["scale"] <= 2)
    stamps = [row[0] for row in camera]
    for begin, end in zip(cuts, cuts[1:]):
        cursor = begin
        for first, last, _, row in camera[bisect_left(stamps, begin):bisect_left(stamps, end)]:
            if first < cursor or last >= end:
                continue
            yield from window_bounds(cursor, first)
            yield first, last, row
            cursor = last
        yield from window_bounds(cursor, end)


def observed_tracks(hits):
    """Use the shot-local tracking ID and abstain when one identity appears twice."""
    valid = []
    for hit in hits:
        if not isinstance(hit, (tuple, list)) or len(hit) < 6:
            continue
        track = hit[0]
        face = hit[6] if len(hit) >= 7 else track
        if type(track) is not int or type(face) is not int or min(track, face) < 1:
            continue
        valid.append((track, face, valid_box(hit[1:5])))
    tracks = Counter(track for track, _, _ in valid)
    faces = Counter(face for _, face, _ in valid)
    return {track: (face, box) for track, face, box in valid if box and tracks[track] == 1 and faces[face] == 1}


def body_bottom(face_box, body_box):
    """Read the lower edge only when the corresponding body box is well inside the frame."""
    body = valid_box(body_box)
    if body is None:
        return None
    x, y, width, height = body
    face_x, face_y = box_center(face_box)
    if height >= 2.5 * face_box[3] and x <= face_x <= x + width and y <= face_y <= y + height * .45:
        return y + height
    return None


def box_center(box):
    """Return the center of a normalized rectangle."""
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def body_samples(rows, start, end, midpoint):
    """Choose a nearby person sample within the current shot-local window."""
    stamps, data = rows
    index = bisect_left(stamps, midpoint)
    candidates = [position for position in (index - 1, index)
                  if 0 <= position < len(stamps) and start <= stamps[position] < end]
    if candidates:
        nearest = min(candidates, key=lambda position: abs(stamps[position] - midpoint))
        if abs(stamps[nearest] - midpoint) <= 160:
            return {(row.get("track"), row.get("faceId")): row.get("box") for row in data[nearest]}
    return {}


def relative_order(one, other):
    """Compare concurrent face heights and use body position to check the ordering."""
    height, aspect, bottom = one
    peer_height, peer_aspect, peer_bottom = other
    if max(aspect, peer_aspect) / min(aspect, peer_aspect) > 1.35:
        return None
    ratio = max(height, peer_height) / min(height, peer_height)
    sign = 1 if height > peer_height else -1
    if ratio <= 1.15 and (bottom is None or peer_bottom is None or abs(bottom - peer_bottom) < .05):
        # Comparable projected sizes can share a layer when a third person supplies clear separation.
        return 0, "similar-projection", 0.0
    if bottom is not None and peer_bottom is not None:
        if ratio >= 1.38 and (bottom - peer_bottom) * sign >= .05:
            return sign, "relative-face-scale-body-position", .67
    elif ratio >= 1.80:
        return sign, "relative-face-scale", .42
    return None


def depth_hint(track, geometry):
    """Require a consistent comparison with every simultaneously visible supported peer."""
    unknown = {"depth": "unknown", "depthBasis": "unknown", "depthConfidence": 0.0}
    if track not in geometry:
        return unknown
    others = []
    for key, peer in geometry.items():
        if key == track:
            continue
        comparison = relative_order(geometry[track], peer)
        if comparison is None:
            return unknown
        if comparison[0]:
            others.append(comparison)
    if not others:
        return unknown
    signs = {value[0] for value in others}
    depth = "foreground" if signs == {1} else "background" if signs == {-1} else "midground"
    _, basis, confidence = min(others, key=lambda value: value[2])
    return {"depth": depth, "depthBasis": basis, "depthConfidence": confidence}


def velocity(first, last, camera, alignment):
    """Measure screen displacement after an optional matching global transform."""
    x, y = box_center(first)
    end_x, end_y = box_center(last)
    basis, confidence = "screen-space", .58
    tolerance_x, tolerance_y = max(.012, .07 * first[2]), max(.012, .07 * first[3])
    if camera is not None:
        scale = camera["scale"]
        x = .5 + (x - .5) * scale + camera["shiftX"]
        y = .5 + (y - .5) * scale + camera["shiftY"]
        tolerance_x += alignment * (abs(camera["shiftX"]) + abs(scale - 1) / 2)
        tolerance_y += alignment * (abs(camera["shiftY"]) + abs(scale - 1) / 2)
        basis, confidence = "camera-compensated", round(min(.82, camera["confidence"]), 2)
    dx, dy = end_x - x, end_y - y
    return {"horizontal": "left" if dx < -tolerance_x else "right" if dx > tolerance_x else "still",
            "vertical": "up" if dy < -tolerance_y else "down" if dy > tolerance_y else "still",
            "basis": basis, "confidence": confidence, "deltaX": round(dx, 4), "deltaY": round(dy, 4)}


def measure_window(start, end, camera, timeline, body, estimated, step):
    """Measure continuous IDs between observed endpoints without using estimated boxes as anchors."""
    first_frame = (start + step - 1) // step
    last_frame = min(len(timeline) - 1, end // step)
    if camera is None and last_frame * step == end:
        last_frame -= 1
    anchors = [index for index in range(first_frame, last_frame + 1) if index not in estimated]
    if len(anchors) < 3 or (anchors[-1] - anchors[0]) * step < MIN_WINDOW_MS:
        return []
    left, right = anchors[0], anchors[-1]
    anchor_offset = left * step - start + end - right * step
    if camera is not None and (left * step - start > step or end - right * step > step):
        camera = None
    samples = [observed_tracks(timeline[index]) for index in range(left, right + 1)]
    common = set(samples[0]).intersection(*samples[1:])
    middle = anchors[len(anchors) // 2] - left
    bodies = body_samples(body, start, end, (middle + left) * step)
    geometry, tracks = {}, {}
    for track in sorted(common):
        face = samples[0][track][0]
        if any(frame[track][0] != face for frame in samples):
            continue
        boxes = [samples[index - left][track][1] for index in anchors]
        median_height = median(box[3] for box in boxes)
        median_aspect = median(box[2] / box[3] for box in boxes)
        if median_height >= .018 and max(box[3] for box in boxes) <= 1.30 * min(box[3] for box in boxes):
            bottom = body_bottom(samples[middle][track][1], bodies.get((track, face)))
            geometry[track] = median_height, median_aspect, bottom
        tracks[track] = {"faceId": face, "track": track, "startMs": start, "endMs": end,
                         **velocity(boxes[0], boxes[-1], camera, anchor_offset / (end - start))}
    return [{**record, **depth_hint(track, geometry)} for track, record in tracks.items()]


def measure_people_motion(video_evidence, camera_samples, duration_ms, *, shot_starts=()):
    """Return source-clock motion intervals and ordinal depth for existing face tracks.

    Extra source-clock cuts come from scene analysis. Face-only depth is a weaker
    hint from simultaneous relative projection sizes. Deltas use frame fractions.
    """
    timeline = video_evidence.get("timeline") or {}
    frames, step = timeline.get("frames") or [], timeline.get("stepMs")
    if type(step) is not int or step <= 0 or not frames:
        return []
    end = min(duration_ms, len(frames) * step)
    if end < MIN_WINDOW_MS:
        return []
    editor = video_evidence.get("editorTimeline") or {}
    cuts = set(shot_starts) | {shot.get("startMs") for shot in editor.get("shots") or []}
    cuts.update((video_evidence.get("faceTracking") or {}).get("identityCutTimesMs") or [])
    bounds = [0, *sorted(stamp for stamp in cuts if type(stamp) is int and 0 < stamp < end), end]
    estimated = set(timeline.get("estimatedFrames") or [])
    body_rows = sorted((row["timeMs"], order, row.get("people") or [])
                       for order, row in enumerate((video_evidence.get("personLayer") or {}).get("frames") or [])
                       if type(row.get("timeMs")) is int)
    body = ([row[0] for row in body_rows], [row[2] for row in body_rows])
    result = []
    for start, stop, camera in motion_windows(bounds, camera_samples):
        result.extend(measure_window(start, stop, camera, frames, body, estimated, step))
    return result
