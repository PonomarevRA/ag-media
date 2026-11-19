"""Measure object appearance on the scene worker's decoded frames."""

from itertools import combinations, product

import cv2
import numpy as np

from voice_studio.vision.scene_objects import iou

OBJECT_SIZE = 64
GRID = np.arange(-41.5, 42, dtype=np.float32)
X, Y = np.meshgrid(GRID, GRID)
PAIR_GAP_MS = 1400
MAX_PAIRS_PER_SAMPLE = 3


def describe(image, box):
    """Resize the longest box side to 64px, preserving proportions and available context."""
    height, width = image.shape[:2]
    x, y, w, h = np.multiply(box, (width, height, width, height)).astype(np.float32)
    if min(w, h) < 12:
        return None
    step = max(w, h) / OBJECT_SIZE
    dx, dy = w / (2 * step), h / (2 * step)
    mx, my = X * step + x + w / 2, Y * step + y + h / 2
    valid = (mx >= 0) & (mx < width - 1) & (my >= 0) & (my < height - 1)
    patch = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    inside = valid & (abs(X) <= dx - 2) & (abs(Y) <= dy - 2)
    ring = valid & (abs(X) <= 1.30 * dx) & (abs(Y) <= 1.30 * dy)
    ring &= (abs(X) >= dx + 1) | (abs(Y) >= dy + 1)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    edge = cv2.magnitude(cv2.Sobel(gray, -1, 1, 0), cv2.Sobel(gray, -1, 0, 1))
    center = inside & ((X / dx) ** 2 + (Y / dy) ** 2 < .74)
    return gray, edge, inside, ring, max(w, h), crop_palette(patch[center])


def crop_palette(pixels):
    """Return three dominant approximate crop colors and their shares of valid center pixels."""
    if len(pixels) < 32:
        return []
    hsv = cv2.cvtColor(pixels[None], cv2.COLOR_BGR2HSV)[0].astype(np.int32)
    hue, saturation, value = hsv.T
    # Keep nearby shades together while separating bright colors from neutral shadows.
    chromatic = (saturation >= 65) & (value >= 48)
    hue_group = ((hue + 7) % 180) // 15
    groups = np.where(chromatic, hue_group + 6, value // 52)
    counts = np.bincount(groups)
    palette = []
    for group in sorted(np.flatnonzero(counts), key=lambda key: (-counts[key], key))[:3]:
        blue, green, red = np.median(pixels[groups == group], axis=0).astype(int)
        palette.append({"hex": f"#{red:02x}{green:02x}{blue:02x}", "share": round(int(counts[group]) / len(pixels), 3)})
    return palette


def correlation(left, right, mask):
    """Measure signed normalized correlation or return None for uniform or missing pixels."""
    if mask.sum() < 64:
        return None
    a, b = left[mask], right[mask]
    a, b = a - a.mean(), b - b.mean()
    energy = np.linalg.norm(a) * np.linalg.norm(b)
    if energy <= 1e-5:
        return None
    return round(float(np.clip(np.dot(a, b) / energy, -1, 1)), 3)


def compare(first, second):
    """Return foreground and context evidence without combining their scores."""
    inside = first[2] & second[2]
    return {"objectSimilarity": correlation(first[0], second[0], inside),
            "edgeSimilarity": correlation(first[1], second[1], inside),
            "backgroundSimilarity": correlation(first[0], second[0], first[3] & second[3]),
            "scaleRatio": round(float(second[4] / first[4]), 3)}


class ObjectAppearance:
    """Retain only descriptors from the preceding object sample in the same shot."""

    def __init__(self):
        self.previous = ()
        self.shot = None
        self.stamp = 0

    def add(self, image, frame):
        """Attach bounded evidence to frame items, indexed by the source box and time."""
        current = []
        for item in frame["objects"][:12]:
            if descriptor := describe(image, item["box"]):
                current.append((item, descriptor))
                if descriptor[5]:
                    item["objectPalette"] = descriptor[5]
        stamp, shot = frame["timeMs"], frame["shot"]
        comparisons = [(combinations(current, 2), stamp)]
        if shot == self.shot and 0 < stamp - self.stamp <= PAIR_GAP_MS:
            comparisons.append((product(self.previous, current), self.stamp))
        for pairs, previous_time in comparisons:
            for (left, first), (right, second) in pairs:
                if iou(left["box"], right["box"]) < .35:
                    continue
                metrics = compare(first, second)
                right.setdefault("objectMatches", []).append({
                    "timeMs": previous_time, "label": left["label"],
                    "box": [round(value, 4) for value in left["box"]], **metrics,
                })
        for item, _ in current:
            if matches := item.get("objectMatches"):
                matches.sort(key=lambda row: -(row["objectSimilarity"] if row["objectSimilarity"] is not None else -2))
                del matches[2 * MAX_PAIRS_PER_SAMPLE:]
        self.previous, self.shot, self.stamp = current, shot, stamp

def retain_confirmed_matches(frames):
    """Drop evidence to discarded detections and keep at most three pairs per observation."""
    known = {frame["timeMs"]: {(item["label"], tuple(item["box"])) for item in frame["objects"]}
             for frame in frames}
    for frame in frames:
        for item in frame["objects"]:
            raw = item.pop("objectMatches", ())
            matches = [pair for pair in raw if (pair["label"], tuple(pair["box"])) in known.get(pair["timeMs"], ())]
            if matches:
                item["objectMatches"] = matches[:MAX_PAIRS_PER_SAMPLE]
