"""Bounded native-pixel detection of small controls in a verified UI region."""

from dataclasses import dataclass

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import pixel_view
from voice_studio.vision.scene_objects import iou
from voice_studio.vision.screen_layout import _box, _intersection

MAX_PANEL_PIXELS = 1_200_000
MAX_ADDITIONS = 24
MAX_GEOMETRY_CANDIDATES = 192
PANEL_TYPES = frozenset(("ui", "mobile-ui"))
GLYPH_CLOSE = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class Control:
    """A visible local contour, with a heuristic strength and no inferred action."""

    box: tuple
    role: str
    basis: str
    confidence: float


def local_support(gray, edges, box):
    """Find interior strokes separated from neighboring texture and text."""
    x, y, w, h = box
    margin = max(3, min(7, min(w, h) // 3))
    if min(x, y, gray.shape[1] - x - w, gray.shape[0] - y - h) < margin:
        return None
    local = gray[y - margin:y + h + margin, x - margin:x + w + margin]
    surrounding = edges[y - margin:y + h + margin, x - margin:x + w + margin]
    border = np.ones(local.shape, dtype=bool)
    border[margin:margin + h, margin:margin + w] = False
    ring = local[border]
    dark, bright = np.percentile(ring, (10, 90))
    clutter = np.count_nonzero(surrounding[border]) / ring.size
    if bright - dark > 30 or clutter > .16:
        return None
    ink = gray[y:y + h, x:x + w]
    contrast = float(max(np.percentile(ink, 90) - dark, bright - np.percentile(ink, 10)))
    if contrast < 24:
        return None
    pad = max(1, round(min(w, h) * .27))
    interior = edges[y + pad:y + h - pad, x + pad:x + w - pad]
    if np.count_nonzero(interior) < 3:
        return None
    return contrast, clutter


def small_buttons(gray, edges, maximum):
    """Recognize closed compact outlines that contain visible content."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        _, _, w, h = box
        if not 13 <= h <= maximum or not 16 <= w <= 140 or not 1.22 <= w / h <= 5:
            continue
        closure = cv2.contourArea(contour) / (w * h)
        if closure >= .63:
            candidates.append((closure, box))
    candidates.sort(reverse=True)
    for closure, box in candidates[:MAX_GEOMETRY_CANDIDATES]:
        if support := local_support(gray, edges, box):
            contrast, clutter = support
            confidence = min(.96, .52 + .20 * closure + min(.16, contrast / 420) - .25 * clutter)
            yield Control(box, "button_candidate", "native-small-control-outline", confidence)


def small_glyphs(gray, edges, maximum):
    """Recover isolated compact strokes, including glyphs below the overview threshold."""
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, GLYPH_CLOSE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    candidates = stats[1:]
    _, _, widths, heights, areas = candidates.T
    area = widths * heights
    compatible = (
        (widths >= 6) & (heights >= 6) & (widths <= maximum) & (heights <= maximum)
        & (widths >= .57 * heights) & (heights >= .57 * widths)
        & (areas >= .08 * area) & (areas <= .78 * area)
    )
    candidates = sorted(candidates[compatible].tolist(), key=lambda row: -row[4])[:MAX_GEOMETRY_CANDIDATES]
    for x, y, w, h, pixels in candidates:
        box = (x, y, w, h)
        if support := local_support(gray, edges, box):
            contrast, clutter = support
            density = pixels / (w * h)
            confidence = min(.92, .53 + min(.22, contrast / 360) + .12 * density - .30 * clutter)
            yield Control(box, "icon_candidate", "native-small-control-glyph", confidence)


def existing_control(box, role, other, other_role):
    """Drop duplicate glyphs and text ink while preserving genuinely nested controls."""
    if iou(box, other) >= .57:
        return True
    _, _, w, h = box
    _, _, p, q = other
    shared = _intersection(box, other)
    other_area = p * q
    if role == "icon_candidate" and other_role == "text_line" and shared >= .42 * w * h:
        return True
    smaller, larger = sorted((w * h, other_area))
    # A selected icon may have a background tile. It is still one visual control.
    same_icon = role == other_role == "icon_candidate"
    return (same_icon or smaller >= .48 * larger) and shared >= .81 * smaller


def enrich_small_controls(image, layout):
    """Append unnamed native-resolution candidates to one already verified UI layout.

    The input image is the oriented BGR source frame and layout coordinates refer
    to that frame. pixelBox uses source pixels. confidence measures the strength
    of supporting geometry only; text and interaction behavior are unverified.
    """
    if not isinstance(layout, dict) or layout.get("panelType") not in PANEL_TYPES or not isinstance(layout.get("elements"), list):
        return 0
    region = _box(layout.get("box"))
    if region is None or image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return 0
    height, width = image.shape[:2]
    crop, bounds = pixel_view(image, region)
    pixels = crop.shape[0] * crop.shape[1]
    if min(crop.shape[:2]) < 32 or pixels > MAX_PANEL_PIXELS:
        return 0
    left, top = round(bounds[0] * width), round(bounds[1] * height)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 34, 95, L2gradient=True)
    maximum = min(56, max(18, crop.shape[1] // 6))
    candidates = [*small_buttons(gray, edges, maximum), *small_glyphs(gray, edges, maximum)]
    candidates.sort(key=lambda item: (item.role != "button_candidate", -item.confidence, item.box[1], item.box[0]))

    elements = layout["elements"]
    known = []
    for entry in elements:
        if isinstance(entry, dict) and (box := _box(entry.get("box"))) is not None:
            known.append((box, entry.get("role")))
    added = []
    for candidate in candidates:
        x, y, w, h = candidate.box
        pixel_box = [left + x, top + y, w, h]
        box = [round(value / dimension, 6) for value, dimension in zip(pixel_box, (width, height, width, height))]
        if any(existing_control(box, candidate.role, prior, role) for prior, role in known):
            continue
        known.append((box, candidate.role))
        added.append({"role": candidate.role, "basis": candidate.basis, "confidence": round(candidate.confidence, 3),
                      "box": box, "pixelBox": pixel_box})
        if len(added) >= MAX_ADDITIONS:
            break
    elements.extend(sorted(added, key=lambda item: (item["pixelBox"][1], item["pixelBox"][0])))
    return len(added)
