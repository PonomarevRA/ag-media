"""Estimate shot-local apparent camera motion from sparse background matches."""

from math import atan2, degrees, hypot

import cv2
import numpy as np

MAX_FRAME_SIDE = 384
MIN_STEP_MS = 160
MAX_STEP_MS = 1150
MIN_MATCHES = 12
GRID = 4


def prepare_frame(image, faces):
    """Reduce the frame and mask detected heads and their nearby lower silhouettes."""
    height, width = image.shape[:2]
    ratio = min(1, MAX_FRAME_SIDE / max(height, width))
    size = (round(width * ratio), round(height * ratio))
    if min(size) < 64:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if ratio < 1:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    width, height = size
    allowed = np.full(gray.shape, 255, dtype=np.uint8)
    for x, y, w, h in faces:
        left = max(0, int((x - 0.65 * w) * width))
        right = min(width, int((x + 1.65 * w) * width))
        top = max(0, int((y - 0.4 * h) * height))
        bottom = min(height, int((y + 4.0 * h) * height))
        if left < right and top < bottom:
            allowed[top:bottom, left:right] = 0
    return gray, allowed


def background_points(gray, allowed):
    """Cap corner votes per outer cell so a single textured area cannot dominate."""
    height, width = gray.shape
    found = []
    for row in range(GRID):
        for column in range(GRID):
            if 0 < row < GRID - 1 and 0 < column < GRID - 1:
                continue
            top, bottom = row * height // GRID, (row + 1) * height // GRID
            left, right = column * width // GRID, (column + 1) * width // GRID
            points = cv2.goodFeaturesToTrack(
                gray[top:bottom, left:right], maxCorners=14, qualityLevel=.015,
                minDistance=7, mask=allowed[top:bottom, left:right], blockSize=5,
            )
            if points is not None:
                found.append(points + np.array([left, top], dtype=np.float32))
    if found:
        return np.concatenate(found)
    return np.empty((0, 1, 2), dtype=np.float32)


def match_background(first, second):
    """Reject out-of-frame, masked and non-reciprocal corner tracks."""
    gray, allowed = first
    target, target_mask = second
    if gray.shape != target.shape:
        return None
    points = background_points(gray, allowed)
    if len(points) < MIN_MATCHES:
        return None
    options = {"winSize": (21, 21), "maxLevel": 3}
    predicted, status, error = cv2.calcOpticalFlowPyrLK(gray, target, points, None, **options)
    height, width = target.shape
    destination = predicted.reshape(-1, 2)
    x, y = destination.T
    valid = (status.ravel() != 0) & (error.ravel() <= 26)
    valid &= (x >= 1) & (x < width - 1) & (y >= 1) & (y < height - 1)
    origin, predicted = points[valid], predicted[valid]
    pixels = predicted.reshape(-1, 2).astype(np.int32)
    unmasked = target_mask[pixels[:, 1], pixels[:, 0]] != 0
    origin, predicted = origin[unmasked], predicted[unmasked]
    if len(origin) < MIN_MATCHES:
        return None
    backward, status, _ = cv2.calcOpticalFlowPyrLK(target, gray, predicted, None, **options)
    valid = (status.ravel() != 0) & (np.linalg.norm(backward[:, 0] - origin[:, 0], axis=1) <= 1.3)
    if np.count_nonzero(valid) < MIN_MATCHES:
        return None
    return origin[valid, 0], predicted[valid, 0], len(points)


def spatial_support(points, width, height):
    """Require matches spread across several outer regions and both image axes."""
    x, y = (points / (width, height)).T
    cells = set(zip((x * GRID).astype(int), (y * GRID).astype(int)))
    sides = sum((np.any(x < .25), np.any(x > .75), np.any(y < .25), np.any(y > .75)))
    if len(cells) < 5 or sides < 3 or np.ptp(x) < .50 or np.ptp(y) < .45:
        return 0.
    return min(1., len(cells) / 9.)


def estimate_transform(first, second):
    """Fit a translation/scale model only when a broad background supports it."""
    matches = match_background(first, second)
    if matches is None:
        return None
    origin, target, candidates = matches
    affine, inliers = cv2.estimateAffinePartial2D(
        origin, target, method=cv2.RANSAC, ransacReprojThreshold=1.8, maxIters=150,
    )
    if affine is None:
        return None
    accepted = inliers.ravel().astype(bool)
    count = np.count_nonzero(accepted)
    if count < MIN_MATCHES or count / len(origin) < .70 or count / candidates < .28:
        return None
    height, width = first[0].shape
    spread = spatial_support(origin[accepted], width, height)
    if not spread:
        return None
    scale = hypot(affine[0, 0], affine[1, 0])
    rotation = degrees(atan2(affine[1, 0], affine[0, 0]))
    shift = (affine[:, :2] @ [.5 * width, .5 * height] + affine[:, 2]) / [width, height] - .5
    dx, dy = shift
    # The published transform has no roll component. Abstain when roll would matter.
    if not .82 <= scale <= 1.18 or abs(rotation) > 1.25 or max(abs(dx), abs(dy)) > .20:
        return None
    confidence = min(1., (.65 * count / len(origin) + .35 * count / candidates) * (.6 + .4 * spread))
    return {
        "horizontal": "left" if dx > .005 else "right" if dx < -.005 else "still",
        "vertical": "up" if dy > .005 else "down" if dy < -.005 else "still",
        "zoom": "in" if scale > 1.010 else "out" if scale < .990 else "still",
        "shiftX": round(float(dx), 6), "shiftY": round(float(dy), 6),
        "scale": round(scale, 6), "confidence": round(float(confidence), 3),
    }


class CameraMotion:
    """Track at most one downsampled anchor and emit disjoint intervals within a shot."""

    def __init__(self):
        self.previous = None
        self.intervals = []

    def add(self, image, sample):
        """Measure between nearby anchors and omit cuts, gaps and unsupported pairs."""
        shot, stamp = sample["shot"], sample["timeMs"]
        previous = self.previous
        if previous and shot == previous[0] and stamp - previous[1] < MIN_STEP_MS:
            return
        current = prepare_frame(image, sample.get("faces", ()))
        self.previous = (shot, stamp, current)
        if not current or not previous or not previous[2] or shot != previous[0]:
            return
        start = previous[1]
        if stamp - start > MAX_STEP_MS:
            return
        try:
            motion = estimate_transform(previous[2], current)
        except cv2.error:
            return
        if motion:
            self.intervals.append({"startMs": start, "endMs": stamp, **motion})
