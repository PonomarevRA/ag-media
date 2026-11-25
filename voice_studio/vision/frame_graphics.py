"""Track candidates for screen-fixed graphics from their appearance and persistent pixels."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

SAMPLE_STEP = 3
HISTORY_SIZE = 8
MAX_ACTIVE = 12
MAX_EVENTS = 4096
MAX_AREA_FRACTION = .24
MIN_PATCH_FRACTION = .004
CONNECT_PATCH = np.ones((9, 9), np.uint8)
SUPPORT_PATCH = np.ones((21, 21), np.uint8)
PANEL_GAP = np.ones((1, 29), np.uint8)


@dataclass
class Candidate:
    """Fixed raster coordinates and the pixels that distinguish a candidate from its background."""

    start: int
    last: int
    bounds: tuple
    outline: tuple
    pixels: np.ndarray
    checks: tuple


def candidate_patches(mask, seeds):
    """Keep separate compact patches, including neighboring artwork entering at the same time."""
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CONNECT_PATCH)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    area = mask.size
    image_height, image_width = mask.shape
    for index, (x, y, width, height, _) in enumerate(stats[1:], 1):
        if not (image_width * .045 <= width and image_height * .038 <= height
                and MIN_PATCH_FRACTION * area <= width * height <= MAX_AREA_FRACTION * area):
            continue
        section = np.s_[y:y + height, x:x + width]
        pixels = (labels[section] == index) & (mask[section] != 0)
        count = int(np.count_nonzero(pixels))
        foreground = seeds[section] & pixels
        seed_count = int(np.count_nonzero(foreground))
        if count >= .24 * width * height and seed_count >= max(.16 * count, MIN_PATCH_FRACTION * area):
            checks = ((pixels, count), (foreground, seed_count))
            yield (int(x), int(y), int(width), int(height)), section, checks


def same_area(left, right):
    """Reject duplicate tracks that still cover the same fixed patch."""
    x, y, w, h = left
    xx, yy, ww, hh = right
    overlap = max(0, min(x + w, xx + ww) - max(x, xx)) * max(0, min(y + h, yy + hh) - max(y, yy))
    return overlap >= .68 * min(w * h, ww * hh)


def graphic_patches(frame, history):
    """Find recent changes that are now fixed in place, excluding a broad scene transition."""
    novel = cv2.inRange(cv2.absdiff(history[0], frame), (0, 0, 0), (44, 44, 44)) == 0
    if np.count_nonzero(novel) > .47 * novel.size:
        return
    stable = np.ones(novel.shape, dtype=bool)
    for prior in (history[-1], history[-2]):
        quiet = cv2.inRange(cv2.absdiff(prior, frame), (0, 0, 0), (15, 15, 15))
        stable &= quiet != 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    vivid = cv2.inRange(hsv, (0, 126, 175), (179, 255, 255)) != 0
    light = cv2.inRange(frame, (200, 200, 200), (255, 255, 255)) != 0
    seeds = novel & stable & (vivid | light)
    if np.count_nonzero(seeds) < MIN_PATCH_FRACTION * seeds.size:
        return
    # Include dark strokes of an icon when held next to its bright foreground.
    surround = cv2.dilate(seeds.astype(np.uint8), SUPPORT_PATCH)
    mask = (novel & stable & (surround != 0)).astype(np.uint8)
    yield from candidate_patches(mask, seeds)


def graphic_outline(frame, bounds):
    """Include both light sides of compact card art when dark content divides its change mask."""
    x, y, width, height = bounds
    full_width = frame.shape[1]
    if height < frame.shape[0] * .08 or width >= full_width * .4:
        return bounds
    left_edge = max(0, x - round(full_width * .04))
    right_edge = min(full_width, x + round(full_width * .42))
    upper = frame[y:y + height // 2, left_edge:right_edge]
    white = np.count_nonzero(cv2.inRange(upper, (214, 214, 214), (255, 255, 255)), axis=0)
    present = (white >= max(3, len(upper) * .18)).astype(np.uint8)
    joined = cv2.morphologyEx(present.reshape(1, -1), cv2.MORPH_CLOSE, PANEL_GAP)[0]
    padded = np.pad(joined.astype(np.int8), 1)
    edges = np.diff(padded)
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        visible = np.flatnonzero(present[left:right])
        if not len(visible):
            continue
        right = left + int(visible[-1]) + 1 + left_edge
        left += int(visible[0]) + left_edge
        overlap = min(right, x + width) - max(left, x)
        if overlap >= .66 * width and width < right - left <= .42 * full_width:
            return int(min(x, left)), y, int(max(x + width, right) - min(x, left)), height
    return bounds


class GraphicTracker:
    """Bounded history in raster space; timings are attached from the native clock after decoding."""

    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)
        self.active = []
        self.finished = []

    def finish(self, candidate):
        """Keep a witness through its last matching sample."""
        if candidate.last - candidate.start >= 2 * SAMPLE_STEP and len(self.finished) < MAX_EVENTS:
            self.finished.append((candidate.start, candidate.last, candidate.outline, candidate.checks[0][1]))

    def match(self, frame, index):
        """Hold each graphic while most of its measured foreground stays at the same coordinates."""
        remaining = []
        for item in self.active:
            x, y, w, h = item.bounds
            patch = frame[y:y + h, x:x + w]
            equal = cv2.inRange(cv2.absdiff(patch, item.pixels), (0, 0, 0), (24, 24, 24))
            # Bright cores stay reliable when an animated object's darker outline drifts.
            if any(np.count_nonzero(equal[mask]) >= .70 * count for mask, count in item.checks):
                item.last = index
            if index - item.last <= SAMPLE_STEP:
                remaining.append(item)
            else:
                self.finish(item)
        self.active = remaining

    def observe(self, frame, index):
        """Admit a new graphic after confirming a compact, bright or vivid change across several samples."""
        if index % SAMPLE_STEP:
            return
        self.match(frame, index)
        if len(self.history) == HISTORY_SIZE and len(self.active) < MAX_ACTIVE:
            for box, section, checks in graphic_patches(frame, self.history):
                if len(self.active) >= MAX_ACTIVE:
                    break
                if any(same_area(box, item.bounds) for item in self.active):
                    continue
                outline = graphic_outline(frame, box)
                self.active.append(Candidate(index - 2 * SAMPLE_STEP, index, box, outline,
                                             frame[section].copy(), checks))
        self.history.append(frame)

    def report(self, frames, side, minimum_ms, basis):
        """Convert fixed pixels to time ranges on the native frame clock."""
        for item in self.active:
            self.finish(item)
        self.active.clear()
        rows = []
        for left, last_sample, bounds, count in self.finished:
            last = min(len(frames) - 1, last_sample + SAMPLE_STEP)
            start, end = frames[left][1], frames[last][1]
            if end - start < minimum_ms:
                continue
            rows.append({"startMs": start, "endMs": end, "regions": [{
                "box": [round(value / side, 5) for value in bounds],
                "changedPixels": count, "basis": basis,
            }]})
        return sorted(rows, key=lambda row: row["startMs"])
