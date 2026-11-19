"""Recover visible layout evidence from an already decoded frame and its OCR boxes."""

from math import hypot, isfinite

import cv2
import numpy as np

from voice_studio.vision.scene_objects import iou

DEVICE_LABELS = {"cell phone", "laptop", "tv"}
MAX_DEVICE_REGIONS = 2
MAX_OUTLINES = 20
MAX_TEXT_LINES = 36
WORKING_SIDE = 960


def _box(value):
    """Validate a nonempty rectangle and clamp small rounding overshoot to the frame."""
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    if any(type(coordinate) not in (int, float) or not isfinite(coordinate) for coordinate in value):
        return None
    x, y, w, h = value
    if min(x, y) < 0 or min(w, h) <= 0 or max(x + w, y + h) > 1.002 or max(x, y) >= 1:
        return None
    return [x, y, min(w, 1 - x), min(h, 1 - y)]


def _intersection(first, second):
    """Return overlap area in the common coordinate space."""
    x, y, w, h = first
    a, b, p, q = second
    return max(0, min(x + w, a + p) - max(x, a)) * max(0, min(y + h, b + q) - max(y, b))


def device_regions(objects):
    """Return up to two nonduplicate device proposals large enough to inspect."""
    candidates = []
    for item in objects:
        label, score = item.get("label"), item.get("score")
        if label not in DEVICE_LABELS or type(score) not in (int, float) or not .38 <= score <= 1:
            continue
        box = _box(item.get("box"))
        if box and box[2] * box[3] >= .022 and min(box[2:]) >= .08:
            candidates.append((score, box, label))
    regions = []
    for _, box, label in sorted(candidates, key=lambda row: -row[0]):
        if any(iou(box, region["box"]) >= .55 for region in regions):
            continue
        regions.append({"box": box, "basis": "device-crop", "deviceLabel": label})
        if len(regions) == MAX_DEVICE_REGIONS:
            break
    return regions


def _crop(image, region):
    """Crop the normalized box and retain the exact rounded-pixel transform."""
    height, width = image.shape[:2]
    x, y, w, h = region
    left, top = round(x * width), round(y * height)
    right, bottom = round((x + w) * width), round((y + h) * height)
    crop = image[top:bottom, left:right]
    if min(crop.shape[:2]) < 48:
        return None, None
    scale = min(1, WORKING_SIDE / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return crop, (left / width, top / height, (right - left) / width, (bottom - top) / height)


def _outlines(edges):
    """Find closed outline candidates while excluding glyph-sized contours and the crop border."""
    height, width = edges.shape
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(24, .055 * width) or h < max(13, .018 * height):
            continue
        if w >= .95 * width or h >= .56 * height or w * h >= .36 * width * height:
            continue
        if x < 2 or y < 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        area = cv2.contourArea(contour)
        if area < .66 * w * h:
            _, (rotated_width, rotated_height), angle = cv2.minAreaRect(contour)
            if min(angle % 90, 90 - angle % 90) > 18 or area < .78 * rotated_width * rotated_height:
                continue
        polygon = cv2.approxPolyDP(contour, .024 * cv2.arcLength(contour, True), True)
        if not 4 <= len(polygon) <= 8 or not cv2.isContourConvex(polygon):
            continue
        box = (x, y, w, h)
        if all(iou(box, other) < .72 for other in boxes):
            boxes.append(box)
    return sorted(boxes, key=lambda box: (-box[2] * box[3], box[1], box[0]))[:MAX_OUTLINES]


def _filled_controls(crop):
    """Find saturated, filled pills carrying compact bright glyphs, without guessing the label."""
    if crop.ndim != 3:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 105, 65), (179, 255, 255))
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(24, .055 * width) or h < max(12, .016 * height) or h > .17 * height:
            continue
        if not 1.3 <= w / h <= 7.5 or w > .85 * width or cv2.contourArea(contour) < .72 * w * h:
            continue
        inner = hsv[y + h // 5:y + h - h // 5, x + w // 7:x + w - w // 7]
        glyphs = cv2.inRange(inner, (0, 0, 150), (179, 110, 255))
        _, _, components, _ = cv2.connectedComponentsWithStats(glyphs, connectivity=8, ltype=cv2.CV_32S)
        aligned = [row for row in components[1:] if .13 * h <= row[3] <= .59 * h
                   and row[4] >= 3 and row[2] <= .22 * w]
        if 2 <= len(aligned) <= 16 and np.ptp([row[1] + row[3] / 2 for row in aligned]) <= .22 * h:
            candidates.append((x, y, w, h))
    return candidates[:MAX_OUTLINES]


def _encloses_separate_outlines(box, outlines):
    """Exclude composite contours traced around multiple distinct controls or diagram blocks."""
    area = box[2] * box[3]
    children = []
    for other in outlines:
        child_area = other[2] * other[3]
        if .16 * area <= child_area <= .68 * area and _intersection(box, other) >= .96 * child_area:
            if any(iou(other, child) < .1 for child in children):
                return True
            children.append(other)
    return False


def _geometry(image, box):
    """Measure edges and complementary filled controls in a bounded frame crop."""
    crop, source_box = _crop(image, box)
    if crop is None:
        return None
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110)
    filled = _filled_controls(crop)
    outlines = _outlines(edges)
    boxes = filled + [outline for outline in outlines
                      if all(iou(outline, item) < .66 for item in filled)
                      and not _encloses_separate_outlines(outline, outlines)]
    return edges, source_box, boxes[:MAX_OUTLINES], len(filled)


def outlined_text_regions(image, region):
    """Return at most ten tight source-frame crops for independent outlined-control OCR."""
    box = _box(region.get("box"))
    geometry = _geometry(image, box) if box else None
    if geometry is None:
        return []
    edges, source_box, boxes, filled_count = geometry
    height, width = edges.shape
    plain_boxes = [outline for outline in boxes[filled_count:]
                   if outline[2] >= 1.2 * outline[3] and outline[2] <= .85 * width
                   and outline[3] <= .27 * height]
    selected = boxes[:filled_count] + sorted(plain_boxes, key=lambda item: item[3])
    output = []
    for x, y, w, h in selected[:10]:
        inset = min(5, max(2, h // 8))
        output.append(_project_box((x + inset, y + inset, w - 2 * inset, h - 2 * inset),
                                   source_box, width, height))
    return output


def _local_lines(lines, region, width, height):
    """Map OCR lines into the working image only where the crop contains the text."""
    x, y, rw, rh = region
    relevant = []
    for item in lines:
        text = str(item.get("text") or "").strip()[:160]
        box = _box(item.get("box"))
        if not text or not box or _intersection(region, box) < .93 * box[2] * box[3]:
            continue
        left, top, w, h = box
        local = ((left - x) * width / rw, (top - y) * height / rh, w * width / rw, h * height / rh)
        if not any(text.casefold() == value.casefold() and iou(local, known) >= .45
                   for known, value in relevant):
            relevant.append((local, text))
    return sorted(relevant, key=lambda item: (item[0][1], item[0][0]))[:MAX_TEXT_LINES]


def _assign_lines(boxes, lines):
    """Assign each line to the smallest plausible containing outline once."""
    grouped = [[] for _ in boxes]
    free = []
    for line, text in lines:
        area = line[2] * line[3]
        owners = [index for index, box in enumerate(boxes) if _intersection(box, line) >= .9 * area
                  and box[2] * box[3] >= 1.2 * area]
        if owners:
            index = min(owners, key=lambda i: boxes[i][2] * boxes[i][3])
            grouped[index].append((line, text))
        else:
            free.append((line, text))
    return grouped, free


def _edge_distance(point, box):
    """Measure distance to the outline, even when the endpoint lies slightly inside."""
    px, py = point
    x, y, w, h = box
    dx, dy = max(x - px, 0, px - x - w), max(y - py, 0, py - y - h)
    if dx or dy:
        return hypot(dx, dy)
    return min(px - x, x + w - px, py - y, y + h - py)


def _nearest_outline(point, boxes, tolerance):
    """Find an unambiguous nearest outline for a connector endpoint."""
    candidates = sorted((_edge_distance(point, box), index) for index, box in enumerate(boxes))
    if not candidates or candidates[0][0] > tolerance:
        return None
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 2:
        return None
    return candidates[0][1]


def _exits_outline(point, other, box):
    """Require the stroke to leave a side rather than follow the box border."""
    px, py = point
    dx, dy = other[0] - px, other[1] - py
    x, y, w, h = box
    options = []
    if y + .09 * h <= py <= y + .91 * h:
        options.extend(((abs(px - x), -dx), (abs(px - x - w), dx)))
    if x + .09 * w <= px <= x + .91 * w:
        options.extend(((abs(py - y), -dy), (abs(py - y - h), dy)))
    return bool(options) and min(options, key=lambda row: row[0])[1] >= .4 * hypot(dx, dy)


def _connectors(edges, boxes):
    """Return visible straight strokes between distinct outlines, with no inferred direction."""
    if len(boxes) < 2:
        return []
    height, width = edges.shape
    scale = min(width, height)
    minimum = max(17, round(scale * .045))
    tolerance = max(5, scale * .015)
    lines = cv2.HoughLinesP(edges, 1, .01745, threshold=max(14, minimum // 2),
                            minLineLength=minimum, maxLineGap=max(4, round(scale * .018)))
    if lines is None:
        return []
    hits = {}
    for ax, ay, bx, by in lines.reshape(-1, 4):
        left, right = (int(ax), int(ay)), (int(bx), int(by))
        first = _nearest_outline(left, boxes, tolerance)
        second = _nearest_outline(right, boxes, tolerance)
        if first is None or second is None or first == second:
            continue
        # Reject a rectangle's own edge and lines that merely graze the sides of both boxes.
        a, b = boxes[first], boxes[second]
        if _intersection(a, b) or not _exits_outline(left, right, a) or not _exits_outline(right, left, b):
            continue
        mx, my = (ax + bx) / 2, (ay + by) / 2
        if any(x - 2 <= mx <= x + w + 2 and y - 2 <= my <= y + h + 2 for x, y, w, h in (a, b)):
            continue
        key = tuple(sorted((first, second)))
        gap = _edge_distance(left, a) + _edge_distance(right, b)
        if key not in hits or gap < hits[key][0]:
            hits[key] = (gap, left, right, first, second)
    return list(hits.values())[:12]


def _project_box(box, region, width, height):
    """Project a working-image rectangle to normalized original frame coordinates."""
    x, y, w, h = box
    a, b, rw, rh = region
    left, top = max(a, a + rw * x / width), max(b, b + rh * y / height)
    right, bottom = min(a + rw, a + rw * (x + w) / width), min(b + rh, b + rh * (y + h) / height)
    return [round(part, 4) for part in (left, top, right - left, bottom - top)]


def _outline_element(box, lines, connected):
    """Name visual candidates without inferring semantics absent from the image."""
    if connected:
        return "diagram_block", "outline-and-connector"
    if len(lines) != 1:
        return "outlined_region", "outline-and-ocr"
    x, _, w, h = box
    (left, _, text_width, _), _ = lines[0]
    if abs(left + text_width / 2 - (x + w / 2)) <= w * .15 and 1.2 <= w / h <= 8:
        return "button_candidate", "centered-text-in-outline"
    if w / h >= 3.5 and (left - x) <= w * .24:
        return "field_candidate", "left-aligned-text-in-outline"
    return "outlined_region", "outline-and-ocr"


def analyze_layout(image, region, lines):
    """Measure visible text, outlined controls and straight links in one screen proposal."""
    box = _box(region.get("box"))
    if not box:
        return None
    geometry = _geometry(image, box)
    if geometry is None:
        return None
    edges, source_box, boxes, filled_count = geometry
    height, width = edges.shape
    text = _local_lines(lines, source_box, width, height)
    assigned, free = _assign_lines(boxes, text)
    links = _connectors(edges, boxes)
    explicit = region.get("basis") == "screenshot"
    if not explicit:
        links = [link for link in links if assigned[link[3]] and assigned[link[4]]]
    connected = {index for _, _, _, first, second in links for index in (first, second)}
    outlined_text = sum(bool(group) for group in assigned)
    if region.get("basis") == "frame-layout" and not links:
        panel = source_box[2] <= .5 and source_box[3] >= .9
        if panel:
            if outlined_text < 1 or len(text) < 3:
                return None
        elif outlined_text < 2:
            return None
    if not text and not links and not (explicit and filled_count):
        return None

    elements, ids = [], {}
    for index, (outline, labels) in enumerate(zip(boxes, assigned)):
        filled = index < filled_count
        if not labels and index not in connected and not filled:
            continue
        if filled and index not in connected:
            role, basis = "button_candidate", "filled-shape-and-centered-glyphs"
        else:
            role, basis = _outline_element(outline, labels, index in connected)
        element = {"role": role, "box": _project_box(outline, source_box, width, height), "basis": basis}
        if labels:
            element["text"] = " ".join(value for _, value in labels)[:160]
        ids[index] = len(elements)
        elements.append(element)
    for text_box, label in free:
        elements.append({"role": "text_line", "text": label, "basis": "ocr",
                         "box": _project_box(text_box, source_box, width, height)})

    x, y, rw, rh = source_box
    connections = []
    for _, left, right, first, second in links:
        segment = [[round(x + rw * px / width, 4), round(y + rh * py / height, 4)]
                   for px, py in (left, right)]
        connections.append({"from": ids[first], "to": ids[second], "segment": segment})
    return {**region, "box": list(source_box), "elements": elements, "connections": connections}
