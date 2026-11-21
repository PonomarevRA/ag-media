"""Locate compact UI controls in overlapping bands without shrinking a tall image."""

from math import ceil

import cv2
import numpy as np

from voice_studio.vision.scene_objects import iou
from voice_studio.vision.screen_ui import analyze_interface, measured_controls
from voice_studio.vision.scene_sampling import evenly_spaced

GRID_COLUMNS = 12
MAX_BANDS = 16
MAX_CONTROLS = 96
CONTROL_ROLES = {"button_candidate", "field_candidate", "icon_candidate"}


def compact_icons(image, region, lines, controls):
    """Find isolated compact glyphs in a UI band; roles remain visual candidates."""
    height, width = image.shape[:2]
    x, y, w, h = region
    left, top, right, bottom = round(x * width), round(y * height), round((x + w) * width), round((y + h) * height)
    pixels = image[top:bottom, left:right]
    if min(pixels.shape[:2]) < 32:
        return []
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 52, 128)
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    max_side = min(72, max(20, (right - left) // 5))
    found = []
    for a, b, p, q, area in stats[1:]:
        if not 10 <= p <= max_side or not 10 <= q <= max_side or not .65 <= p / q <= 1.54:
            continue
        if not .085 <= area / (p * q) <= .64 or min(a, b, right - left - a - p, bottom - top - b - q) < 3:
            continue
        # Adjacent ink on the same baseline usually indicates an OCR-missed word.
        margin = min(8, p // 2)
        side_ink = [edges[b:b + q, max(0, a - margin):a],
                    edges[b:b + q, a + p:a + p + margin]]
        if any(side.size and np.count_nonzero(side) / side.size > .08 for side in side_ink):
            continue
        box = [(left + a) / width, (top + b) / height, p / width, q / height]
        if any(covers(box, row["box"]) for row in (*lines, *controls, *found)):
            continue
        found.append({"role": "icon_candidate", "basis": "compact-shaped-glyph", "box": box})
    return sorted(found, key=lambda item: (item["box"][1], item["box"][0]))[:16]


def covers(box, other):
    """Suppress glyphs contained by a measured text line or a complete control."""
    x, y, w, h = box
    a, b, p, q = other
    width = max(0, min(x + w, a + p) - max(x, a))
    height = max(0, min(y + h, b + q) - max(y, b))
    return width * height >= .64 * w * h


def control_grid(image, lines, region=(0, 0, 1, 1)):
    """Return bounded controls in whole-image coordinates and one-based grid cells."""
    height, width = image.shape[:2]
    left, top, span, extent = region
    first, end = round(top * height), round((top + extent) * height)
    step = max(ceil((end - first) / MAX_BANDS), min(600, max(160, round(span * width))))
    starts = list(range(first, max(first + 1, end - step), step))
    starts.append(max(first, end - step))
    controls = measured_controls(image, region, lines)
    for start in evenly_spaced(sorted(set(starts)), MAX_BANDS):
        band = [left, start / height, span, (min(end, start + step + 110) - start) / height]
        layout = analyze_interface(image, {"box": band, "basis": "screenshot"}, lines, detailed=False) or {}
        candidates = [row for row in layout.get("elements", ()) if row["role"] in CONTROL_ROLES]
        candidates.extend(compact_icons(image, band, lines, candidates))
        for item in candidates:
            if all(iou(item["box"], prior["box"]) < .58 for prior in controls):
                controls.append(item)
    grid = {"columns": GRID_COLUMNS, "rows": max(1, ceil(height * GRID_COLUMNS / width)),
            "width": width, "height": height, "origin": "top-left", "coordinateSpace": "normalized-image"}
    items = []
    for row in evenly_spaced(sorted(controls, key=lambda item: (item["box"][1], item["box"][0])), MAX_CONTROLS):
        x, y, w, h = row["box"]
        left, top = round(x * width), round(y * height)
        items.append({**row, "box": [round(value, 5) for value in (x, y, w, h)],
                      "pixelBox": [left, top, min(width, round((x + w) * width)) - left,
                                   min(height, round((y + h) * height)) - top],
                      "column": min(grid["columns"], int((x + w / 2) * grid["columns"]) + 1),
                      "row": min(grid["rows"], int((y + h / 2) * grid["rows"]) + 1)})
    return {"grid": grid, "controls": items}
