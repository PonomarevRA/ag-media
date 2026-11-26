"""Measure camera displacement outside object boxes for a disputed track continuation."""

import cv2
import numpy as np

from voice_studio.vision.camera_motion import MIN_MATCHES, estimate_transform, prepare_frame, spatial_support
from voice_studio.vision.vehicle_evidence import valid_box

MIN_BACKGROUND_SHARE = .30
MIN_SPARSE_CONFIDENCE = .70


def tracking_background(image, boxes):
    """Prepare a small grayscale frame and exclude known objects with a safety margin."""
    if image is None or (prepared := prepare_frame(image, ())) is None:
        return None
    gray, allowed = prepared
    height, width = gray.shape
    for box in boxes[:20]:
        if not valid_box(box):
            continue
        x, y, w, h = box
        left = max(0, int((x - .08 * w) * width))
        top = max(0, int((y - .08 * h) * height))
        right = min(width, int(np.ceil((x + 1.08 * w) * width)))
        bottom = min(height, int(np.ceil((y + 1.08 * h) * height)))
        allowed[top:bottom, left:right] = 0
    if cv2.countNonZero(allowed) / allowed.size < MIN_BACKGROUND_SHARE:
        return None
    return gray, allowed


def sparse_tracking_transform(first, second):
    """Use the existing reciprocal camera fitter when both backgrounds support it."""
    if first is None or second is None:
        return None
    try:
        motion = estimate_transform(first, second)
    except cv2.error:
        return None
    if motion and motion["confidence"] >= MIN_SPARSE_CONFIDENCE:
        return {**motion, "basis": "background-lk"}
    return None


def dense_tracking_transform(first, second):
    """Accept a translation only when reciprocal dense flow agrees across textured background."""
    if first is None or second is None or first[0].shape != second[0].shape:
        return None
    gray, allowed = first
    other, other_allowed = second
    height, width = gray.shape
    try:
        solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        forward = solver.calc(gray, other, None)
        backward = solver.calc(other, gray, None)
        texture = cv2.cornerMinEigenVal(gray, 5)
    except (AttributeError, cv2.error):
        return None
    return robust_translation(forward, backward, texture, allowed, other_allowed, width, height)


def robust_translation(forward, backward, texture, allowed, other_allowed, width, height):
    """Reject unsupported or spatially incoherent flow and remap reciprocal samples."""
    grid_y, grid_x = np.mgrid[5:height-5:9, 5:width-5:9]
    sample = (allowed[grid_y, grid_x] > 0) & (texture[grid_y, grid_x] >= .001)
    x, y = grid_x[sample], grid_y[sample]
    shifts = forward[y, x]
    finite = np.isfinite(shifts).all(axis=1)
    x, y, shifts = x[finite], y[finite], shifts[finite]
    destination = np.rint(np.column_stack((x, y)) + shifts).astype(np.int32)
    u, v = destination.T
    inside = (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
    x, y, shifts, u, v = x[inside], y[inside], shifts[inside], u[inside], v[inside]
    reciprocal = (other_allowed[v, u] > 0) & (np.linalg.norm(shifts + backward[v, u], axis=1) <= 1.6)
    origins, shifts = np.column_stack((x[reciprocal], y[reciprocal])), shifts[reciprocal]
    if len(shifts) < max(24, MIN_MATCHES):
        return None
    center = np.median(shifts, axis=0)
    inliers = np.linalg.norm(shifts - center, axis=1) <= 1.8
    ratio = np.count_nonzero(inliers) / len(shifts)
    support = spatial_support(origins[inliers], width, height) if ratio >= .78 else 0
    dx, dy = center / (width, height)
    if support < .55 or not .008 <= max(abs(dx), abs(dy)) <= .17:
        return None
    return {"shiftX": round(float(dx), 6), "shiftY": round(float(dy), 6), "scale": 1.0,
            "confidence": round(.7 * ratio + .3 * support, 3), "basis": "background-dis"}
