"""Shared visual UI interpretation for screenshots, video frames and mosaics."""

import cv2

from voice_studio.vision.scene_ocr import pixel_view
from voice_studio.vision.screen_layout import _project_box, analyze_layout
from voice_studio.vision.scene_objects import iou


def shape_candidates(crop):
    """
    Separate large rounded controls and square icon shapes.

    crop is an oriented BGR viewport. Return bounding boxes in crop pixels;
    content and control behavior are not inferred from the contour.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 90)
    _, light = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY)
    fields, icons = [], []
    for mask in (edges, light):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = box = cv2.boundingRect(contour)
            if w < 23 or h < 18 or x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                continue
            fill = cv2.contourArea(contour) / (w * h)
            if .52 * width <= w <= .94 * width and 4.5 <= w / h <= 17 and fill >= .82:
                center = gray[y + h // 4:y + 3 * h // 4, x + w // 5:x + 4 * w // 5]
                if float(center.mean()) < 150 or float(center.std()) > 34:
                    continue
                if all(iou(box, item) < .75 for item in fields):
                    fields.append(box)
            if .055 * width <= w <= .22 * width and .72 <= w / h <= 1.4 and fill >= .57:
                if all(iou(box, item) < .70 for item in icons):
                    icons.append(box)
    return fields[:8], icons[:80]


def measured_controls(image, scope, lines):
    """
    Connect OCR captions to nearby icon shapes and describe evidence-only fields.

    image and scope refer to the oriented original and its normalized viewport.
    Return candidate controls in whole-image coordinates. The same caption
    stays in OCR evidence so agents can distinguish label from icon geometry.
    """
    crop, bounds = pixel_view(image, scope)
    height, width = crop.shape[:2]
    if min(height, width) < 96:
        return []
    fields, icons = shape_candidates(crop)
    left, top, span, extent = bounds
    controls = [{"role": "field_candidate", "basis": "wide-neutral-control",
                 "box": _project_box(box, bounds, width, height)} for box in fields]
    taken = set()
    for line in lines:
        text = line.get("text", "")
        if len(text) > 28:
            continue
        x, y, w, h = line["box"]
        baseline = (y - top) / extent * height
        middle = (x + w / 2 - left) / span * width
        choices = [(abs(middle - a - p / 2) + gap, box) for box in icons if box not in taken
                   for a, b, p, q in (box,) if -2 <= (gap := baseline - b - q) <= max(8, .36 * q)
                   and abs(middle - a - p / 2) <= .33 * p]
        if not choices:
            continue
        _, icon = min(choices)
        taken.add(icon)
        a, b, p, q = _project_box(icon, bounds, width, height)
        controls.append({"role": "icon_candidate", "basis": "icon-with-visible-caption", "text": text,
                         "box": [a, b, p, min(1 - b, max(q, y + h - b))]})
    return controls


def analyze_interface(image, region, lines, *, detailed=True):
    """
    Interpret the same screen geometry on each oriented capture surface.

    image and region provide pixels and normalized source bounds. lines
    carries OCR observations in that source. detailed enables standalone
    shape evidence; video geometry requires meaningful supporting evidence
    before declaring an otherwise unknown region to be an interface.
    """
    layout = analyze_layout(image, region, lines)
    if not detailed:
        return layout
    observed = measured_controls(image, region["box"], lines)
    if not observed:
        return layout
    if layout is None:
        trusted = region.get("knownPanel") or region.get("basis") == "screenshot"
        text_evidence = sum(bool(item.get("text")) for item in observed)
        if not trusted and text_evidence < 2:
            return None
        layout = {**region, "elements": [], "connections": []}
    elements = layout["elements"]
    for item in observed:
        if all(iou(item["box"], prior["box"]) < .60 for prior in elements):
            elements.append(item)
    return layout
