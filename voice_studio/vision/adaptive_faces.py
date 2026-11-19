"""Reduce YuNet calls at 25 fps by refining unstable six-frame intervals."""

from itertools import islice
from typing import NamedTuple

import cv2
import numpy as np

ANCHOR_STEP = 6


class ScanFrame(NamedTuple):
    """One source-clock frame with cut evidence computed before any detection."""

    number: int
    color: np.ndarray
    cut: bool
    suspect_cut: bool
    gray: np.ndarray


def matched_boxes(left, right):
    """Match face boxes only when each has one geometrically plausible partner."""
    if left is None or right is None or not len(left) or len(left) != len(right):
        return None
    a, b = left[:, None, :4], right[None, :, :4]
    size = np.maximum(0, np.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:]) -
                      np.maximum(a[..., :2], b[..., :2]))
    common = size.prod(axis=2)
    iou = common / (a[..., 2:].prod(axis=2) + b[..., 2:].prod(axis=2) - common)
    eligible = iou >= .78
    if not (np.all(eligible.sum(axis=0) == 1) and np.all(eligible.sum(axis=1) == 1)):
        return None
    return left[:, :4], right[eligible.argmax(axis=1), :4]


def stable_images(images):
    """Check all intermediate frames, including motion that reverses between anchors."""
    for earlier, later in zip(images, images[1:]):
        delta = cv2.absdiff(earlier, later)
        if delta.mean() > 4 or np.count_nonzero(delta > 28) > delta.size * .025:
            return False
    return True


def refine(left, right, frames, images, hits, estimated, detect):
    """Check the two anchors, then split an unsafe interval until individual frames."""
    for index in (left, right):
        if index not in hits:
            hits[index] = detect(frames[index].color)
    span = right - left
    if span <= 1:
        return
    boxes = None
    if not any(frame.cut or frame.suspect_cut for frame in frames[left + 1:right + 1]):
        if stable_images(images[left:right + 1]):
            boxes = matched_boxes(hits[left], hits[right])
    if boxes is not None:
        first, last = boxes
        for offset in range(1, span):
            estimated[left + offset] = first + (last - first) * (offset / span)
        return
    midpoint = (left + right) // 2
    refine(left, midpoint, frames, images, hits, estimated, detect)
    refine(midpoint, right, frames, images, hits, estimated, detect)


def detect_adaptively(frames, detect):
    """Buffer at most seven color frames and reuse the shared measured anchor."""
    anchor = next(frames, None)
    if anchor is None:
        return
    last_hit = detect(anchor.color)
    yield anchor, last_hit, True
    while block := list(islice(frames, ANCHOR_STEP)):
        block.insert(0, anchor)
        images = [cv2.resize(frame.gray, (64, 64), interpolation=cv2.INTER_AREA) for frame in block]
        hits, estimated = {0: last_hit}, {}
        refine(0, len(block) - 1, block, images, hits, estimated, detect)
        for index, frame in enumerate(block[1:], 1):
            if index in hits:
                yield frame, hits[index], True
            else:
                yield frame, estimated[index], False
        anchor, last_hit = block[-1], hits[len(block) - 1]
