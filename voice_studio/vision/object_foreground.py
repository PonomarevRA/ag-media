"""Optional foreground-only diagnostics from the unmodified source crop."""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from voice_studio.vision.object_appearance import (
    OBJECT_SIZE, X, Y, correlation, describe,
)
from voice_studio.vision.vehicle_evidence import valid_box

BASIS = "grabcut-foreground"
MAX_SIDE = 160
MIN_SIDE = 18
MIN_MASK_PIXELS = 64
MIN_STABILITY = .66
EDGE_MARGIN = np.ones((7, 7), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class ForegroundDescription:
    """Native aspect-ratio appearance and a stable mask on the same normalized grid."""

    gray: np.ndarray
    edge: np.ndarray
    inside: np.ndarray
    mask: np.ndarray
    fraction: float
    stability: float


def trial_mask(image, box, inset):
    """Run one bounded GrabCut trial from a slightly varied detector rectangle."""
    left, top, right, bottom = box
    dx = round(inset * (right - left))
    dy = round(inset * (bottom - top))
    rectangle = (left + dx, top + dy, right - left - 2 * dx, bottom - top - 2 * dy)
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros_like(background)
    cv2.setRNGSeed(0)
    cv2.grabCut(image, labels, rectangle, background, foreground, 2, cv2.GC_INIT_WITH_RECT)
    return (labels & 1) != 0


def stable_mask(image, box):
    """Keep only foreground shared by both seeds and abstain on implausible masks."""
    left, top, right, bottom = box
    area = (right - left) * (bottom - top)
    if min(right - left, bottom - top) < MIN_SIDE or area == image.shape[0] * image.shape[1]:
        return None
    try:
        outer = trial_mask(image, box, 0.0)
        inner = trial_mask(image, box, .04)
    except cv2.error:
        return None
    common = outer & inner
    shared = np.count_nonzero(common)
    fraction = shared / area
    outer_fraction = np.count_nonzero(outer) / area
    stability = shared / max(1, np.count_nonzero(outer | inner))
    if not .16 <= fraction <= outer_fraction <= .83 or stability < MIN_STABILITY:
        return None
    return common.astype(np.uint8), float(fraction), float(stability)


def describe_foreground(image, box):
    """Estimate a stable foreground mask from a detector box, including nearby background.

    Input is a BGR uint8 image and a normalized box. A small crop, weak support,
    or an OpenCV failure leaves no measurement. Mask stability measures agreement
    between seeds and is not confidence that the pixels belong to the object.
    """
    if not valid_box(box) or image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    height, width = image.shape[:2]
    x, y, w, h = np.multiply(box, (width, height, width, height)).astype(np.float32)
    left, top, right, bottom = floor(x), floor(y), min(width, ceil(x + w)), min(height, ceil(y + h))
    if min(right - left, bottom - top) < MIN_SIDE:
        return None
    margin_x = max(4, (right - left) // 8)
    margin_y = max(4, (bottom - top) // 8)
    start_x, start_y = max(0, left - margin_x), max(0, top - margin_y)
    end_x, end_y = min(width, right + margin_x), min(height, bottom + margin_y)
    crop = image[start_y:end_y, start_x:end_x]
    scale = min(1.0, MAX_SIDE / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    limits = (left - start_x, top - start_y, right - start_x, bottom - start_y)
    support = stable_mask(crop, tuple(round(value * scale) for value in limits))
    if support is None:
        return None
    patch, fraction, stability = support
    appearance = describe(image, box)
    if appearance is None:
        return None
    gray, edge, inside = appearance[:3]
    # Use the appearance raster's source-pixel centers to avoid stretching to a square.
    step = max(w, h) / OBJECT_SIZE
    mx = (X * step + x + w / 2 - start_x + .5) * scale - .5
    my = (Y * step + y + h / 2 - start_y + .5) * scale - .5
    sampled = cv2.remap(patch, mx, my, cv2.INTER_NEAREST)
    # Exclude mask edges where the shared Gaussian/Sobel raster mixes background pixels.
    mask = cv2.erode(sampled, EDGE_MARGIN).astype(bool) & inside
    if np.count_nonzero(mask) < MIN_MASK_PIXELS:
        return None
    return ForegroundDescription(gray, edge, inside, mask, fraction, stability)


def compare_foregrounds(first, second):
    """Compare directly overlapping normalized masks without searching or inferring identity."""
    if first is None or second is None:
        return None
    shared = first.mask & second.mask
    pixels = np.count_nonzero(shared)
    if pixels < MIN_MASK_PIXELS:
        return None
    masked_gray = correlation(first.gray, second.gray, shared)
    masked_edge = correlation(first.edge, second.edge, shared)
    if masked_gray is None and masked_edge is None:
        return None
    baseline = first.inside & second.inside
    overlap = pixels / np.count_nonzero(first.mask | second.mask)
    return {"basis": BASIS,
            "baselineObjectSimilarity": correlation(first.gray, second.gray, baseline),
            "baselineEdgeSimilarity": correlation(first.edge, second.edge, baseline),
            "maskedObjectSimilarity": masked_gray, "maskedEdgeSimilarity": masked_edge,
            "maskOverlap": round(float(overlap), 3),
            "sourceFraction": round(first.fraction, 3), "targetFraction": round(second.fraction, 3),
            "sourceStability": round(first.stability, 3), "targetStability": round(second.stability, 3)}
