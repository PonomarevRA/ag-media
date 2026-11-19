"""Turn OCR lines into conservative list and title-bar evidence for a still image."""

from collections import defaultdict
from math import isfinite
import re
from statistics import median
from itertools import combinations

import cv2
import numpy as np

MAX_LINES = 120
MAX_LISTS = 8
MAX_ROWS = 30
MAX_CLIPS = 36
TIMECODE = re.compile(r"(?:\d{1,3}:){1,2}\d{2}(?:[.,:]\d{1,3})?")


def _ocr_lines(lines):
    """Keep usable normalized OCR boxes and stable text order."""
    output = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            continue
        text = line["text"].strip()
        box = line.get("box")
        if not text or not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        if any(type(v) not in (int, float) or not isfinite(v) for v in box):
            continue
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or max(x + w, y + h) > 1.002:
            continue
        output.append({"text": text[:160], "box": [x, y, min(w, 1 - x), min(h, 1 - y)]})
        if len(output) == MAX_LINES:
            break
    return sorted(output, key=lambda row: (row["box"][1], row["box"][0]))


def _control_circles(mask, limit):
    """Return compact circular contours, removing duplicate edges from rings."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not 4 <= min(w, h) <= max(w, h) <= limit or not .76 <= w / h <= 1.32:
            continue
        area = cv2.contourArea(contour)
        if not .50 * w * h <= area <= .88 * w * h:
            continue
        item = (x + w / 2, y + h / 2, (w + h) / 2)
        if not any(abs(cx - item[0]) < 3 and abs(cy - item[1]) < 3 for cx, cy, _ in centers):
            centers.append(item)
    return sorted(centers)[:14]


def _is_control_cluster(centers):
    """Check equal circular size, horizontal alignment and macOS spacing."""
    size = median(item[2] for item in centers)
    gaps = [right[0] - left[0] for left, right in zip(centers, centers[1:])]
    return (max(item[2] for item in centers) - min(item[2] for item in centers) <= .25 * size
            and all(1.05 * size <= gap <= 3.3 * size for gap in gaps)
            and .68 <= gaps[0] / gaps[1] <= 1.47
            and max(item[1] for item in centers) - min(item[1] for item in centers) <= .45 * size)


def _window_controls(image):
    """Find three recognizable left-side macOS window controls, including inactive chrome."""
    if image.ndim != 3 or image.shape[2] != 3:
        return ()
    height, width = image.shape[:2]
    crop = image[:min(height, max(16, round(height * .1))), :max(16, round(width * .16))]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 115, 120), (13, 255, 255)) | cv2.inRange(hsv, (166, 115, 120), (179, 255, 255))
    yellow = cv2.inRange(hsv, (14, 115, 135), (38, 255, 255))
    green = cv2.inRange(hsv, (39, 105, 95), (92, 255, 255))
    limit = .055 * min(width, height) + 3
    colored = [_control_circles(mask, limit) for mask in (red, yellow, green)]
    if all(colored):
        controls = [candidates[0] for candidates in colored]
        if _is_control_cluster(controls):
            return controls, "ocr-with-window-controls"

    # Inactive native windows render the same controls as gray rings. A window
    # separator is required here, because the rings alone can also be form icons.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    candidates = _control_circles(cv2.Canny(gray, 10, 28), limit)
    for controls in combinations(candidates, 3):
        if not _is_control_cluster(controls):
            continue
        cy = median(item[1] for item in controls)
        size = median(item[2] for item in controls)
        lower, upper = round(cy + .7 * size), round(cy + 1.9 * size)
        strip = image[lower:upper, width // 5:width * 4 // 5, 1].astype(np.int16)
        if len(strip) > 2 and _divider_strength(strip, 6).max() >= .72:
            return controls, "ocr-with-inactive-window-controls"
    return ()

def _window_title(control_evidence, lines, image):
    """Choose a line within the control band and report its exact OCR box."""
    if not control_evidence:
        return []
    controls, basis = control_evidence
    height, width = image.shape[:2]
    center = median(item[1] for item in controls) / height
    size = median(item[2] for item in controls) / height
    cutoff = (controls[-1][0] + 1.25 * controls[-1][2]) / width
    candidates = []
    for line in lines:
        x, y, w, h = line["box"]
        if x < cutoff or abs(y + h / 2 - center) > max(.8 * size, .009):
            continue
        if h > 2.6 * size or sum(ch.isalnum() for ch in line["text"]) < 2:
            continue
        # Prefer conventional centered chrome to nearby toolbar labels.
        distance = abs(x + w / 2 - .5)
        candidates.append((distance, line))
    if not candidates:
        return []
    score, title = min(candidates, key=lambda row: row[0])
    if score > .14 and len(candidates) != 1:
        return []
    box = title["box"]
    return [{"title": title["text"], "titleBox": box, "box": box, "basis": basis}]


def _midline(box):
    """Measure the vertical center of a normalized line box."""
    return box[1] + box[3] / 2


def _column_lines(rows):
    """Prefer a complete OCR line over overlapping fragments at the same anchor."""
    output = []
    for row in sorted(rows, key=lambda item: _midline(item["box"])):
        box = row["box"]
        if output and abs(_midline(box) - _midline(output[-1]["box"])) < .5 * box[3]:
            if box[2] > output[-1]["box"][2]:
                output[-1] = row
        else:
            output.append(row)
    return output


def _matching_steps(group, vertical_span):
    """Follow a recurring vertical step, allowing secondary text within a row."""
    for index, first in enumerate(group[:-2]):
        height = first["box"][3]
        for follow in range(index + 1, min(index + 6, len(group) - 1)):
            second = group[follow]
            gap = _midline(second["box"]) - _midline(first["box"])
            if not 1.10 * height <= gap <= min(.15 / vertical_span, 7.2 * height) or not .66 <= second["box"][3] / height <= 1.50:
                continue
            run = [first, second]
            for row in group[follow + 1:]:
                expected = _midline(run[-1]["box"]) + gap
                delta = _midline(row["box"]) - expected
                if delta > .20 * gap:
                    break
                if abs(delta) <= .20 * gap and .66 <= row["box"][3] / height <= 1.50:
                    run.append(row)
            if len(run) >= 3:
                yield run


def _aligned_groups(lines, tolerance, vertical_span):
    """Collect row candidates even when labels and dates share their left edge."""
    anchors = []
    groups = defaultdict(list)
    for row in sorted(lines, key=lambda item: item["box"][0]):
        x = row["box"][0]
        if not anchors or x - anchors[-1] > tolerance:
            anchors.append(x)
        groups[len(anchors) - 1].append(row)
    for group in groups.values():
        candidates = list(_matching_steps(_column_lines(group), vertical_span))
        candidates.sort(key=lambda run: (-len(run), -median(row["box"][3] for row in run)))
        yield from candidates


def _gray_frame(image):
    """Bound geometry work while retaining horizontal separators and icon edges."""
    height, width = image.shape[:2]
    scale = min(1, 1024 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.int16)


def _divider_strength(gray, threshold=7):
    """Measure the fraction of a horizontal stripe changing across each y pixel."""
    return (np.abs(np.diff(gray, axis=0)) >= threshold).mean(axis=1)


def _separator_phases(region):
    """Find continuous strokes, excluding the individual glyph edges of wrapped prose."""
    contrast = np.abs(np.diff(region, axis=0)) >= 7
    minimum = max(7, round(.55 * region.shape[1]))
    phases = []
    for index in np.flatnonzero(contrast.mean(axis=1) >= .67):
        stripe = np.pad(contrast[index].astype(np.int8), 1)
        transitions = np.flatnonzero(np.diff(stripe))
        if (transitions[1::2] - transitions[::2]).max(initial=0) >= minimum:
            phases.append(index / len(region))
    return phases


def _visible_list_evidence(gray, run):
    """Require recurring dividers or compact markers to distinguish a list from prose."""
    height, width = gray.shape
    boxes = [row["box"] for row in run]
    left = min(box[0] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    if right - left < .025:
        return None
    x1, x2 = round(left * width), round(right * width)
    if x2 - x1 < 10:
        return None
    phases = []
    for previous, current in zip(boxes, boxes[1:]):
        lo = max(0, round((previous[1] + previous[3]) * height))
        hi = min(height, round(current[1] * height))
        if hi - lo <= 2:
            continue
        region = gray[lo:hi, x1:x2]
        phases.append(_separator_phases(region))
    required = max(2, len(boxes) - 2)
    for candidate in (phase for gap in phases for phase in gap):
        if sum(any(abs(phase - candidate) <= .13 for phase in gap) for gap in phases) >= required:
            return "aligned-ocr-with-dividers"

    markers = 0
    for x, y, _, h in boxes:
        margin = min(.04, h * height / width * 2)
        x1, x2 = max(0, round((x - margin) * width)), round((x - .004) * width)
        y1, y2 = max(0, round(y * height)), min(height, round((y + h) * height))
        area = gray[y1:y2, x1:x2]
        if area.shape[0] < 3 or area.shape[1] < 4:
            continue
        ground = median((int(area[0, 0]), int(area[0, -1]), int(area[-1, 0]), int(area[-1, -1])))
        contrast = np.abs(area - ground) > 28
        if .04 <= contrast.mean() <= .6 and contrast.any(axis=0).sum() >= 2:
            markers += 1
    if markers >= max(3, len(boxes) - 1):
        return "aligned-ocr-with-row-markers"
    return None


def _union(rows):
    """Bound recognized row text in normalized source-image coordinates."""
    boxes = [row["box"] for row in rows]
    x, y = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[0] + box[2] for box in boxes), max(box[1] + box[3] for box in boxes)
    return [round(value, 4) for value in (x, y, right - x, bottom - y)]


def _trim_heading(run):
    """Drop a larger introductory heading before a recurring measured row pattern."""
    first, *rows = run
    width = median(row["box"][2] for row in rows)
    height = median(row["box"][3] for row in rows)
    if first["box"][2] >= 1.55 * width and first["box"][3] >= 1.17 * height:
        return rows
    return run


def _shares_column(group, box, tolerance):
    """Suppress overlapping alternate parses of one visible list column."""
    left, top, _, height = group["box"]
    if abs(left - box[0]) > tolerance:
        return False
    overlap = min(top + height, box[1] + box[3]) - max(top, box[1])
    return overlap >= .66 * min(height, box[3])


def _merge_adjacent_columns(lists):
    """Pair visually aligned cells without repeating separate lists for each column."""
    result = []
    for group in sorted(lists, key=lambda item: item["box"][0]):
        rows = group["rows"]
        merged = False
        for previous in result:
            leaders = previous["rows"]
            if (len(leaders) != len(rows) or previous["basis"] != "aligned-ocr-with-dividers"
                    or group["basis"] != previous["basis"]):
                continue
            pitch = (_midline(leaders[-1]["box"]) - _midline(leaders[0]["box"])) / (len(leaders) - 1)
            if any(abs(_midline(a["box"]) - _midline(b["box"])) > .28 * pitch
                   or a["box"][0] + a["box"][2] >= b["box"][0] for a, b in zip(leaders, rows)):
                continue
            for primary, cell in zip(leaders, rows):
                cells = primary.setdefault("cells", [{"text": primary["text"], "box": primary["box"]}])
                cells.append(cell)
                primary["box"] = _union(cells)
            previous["box"] = _union(leaders)
            merged = True
            break
        if not merged:
            result.append(group)
    return result


def _filled_clip_shapes(image, vertical_span):
    """Find strongly colored, long, contiguous shapes at screen rather than pixel scale."""
    if image.ndim != 3 or image.shape[2] != 3:
        return []
    height, width = image.shape[:2]
    scale = min(1, 960 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 66) & (hsv[:, :, 2] >= 38)
    if saturated.mean() < .012:
        return []
    bands = hsv[:, :, 0] // 15
    max_height = min(.145 / vertical_span, .55) * height
    shapes = []
    hue_counts = np.bincount(bands[saturated], minlength=12)
    for hue in np.flatnonzero(hue_counts >= 35):
        mask = np.where(saturated & (bands == hue), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not max(8, .014 * height) <= h <= max_height or not .028 * width <= w <= .77 * width:
                continue
            if w < 1.50 * h or cv2.contourArea(contour) < .69 * w * h:
                continue
            shapes.append((x / width, y / height, w / width, h / height))
    shapes = sorted(shapes, key=lambda box: (-box[2] * box[3], box[1], box[0]))[:160]
    return sorted(shapes, key=lambda box: (box[1] + box[3] / 2, box[0]))


def _clip_lanes(shapes):
    """Group clips with the same visual lane and reject solitary decorative blocks."""
    lanes = []
    for shape in shapes:
        _, y, _, h = shape
        center = y + h / 2
        if not lanes or abs(center - (lanes[-1][0][1] + lanes[-1][0][3] / 2)) > .35 * h:
            lanes.append([])
        lanes[-1].append(shape)
    return [sorted(lane) for lane in lanes if len(lane) >= 2]


def _ruler_bounds(lines):
    """Locate a time ruler by aligned ticks or by a tick beside an explicit axis label."""
    ticks = []
    for row in lines:
        x, y, w, h = row["box"]
        text = "".join(row["text"].split())
        if TIMECODE.fullmatch(text):
            ticks.append((x, y + h / 2, y + h, w, text))
    groups = []
    for _, y, _, _, _ in ticks:
        aligned = sorted(tick for tick in ticks if abs(tick[1] - y) < .016)
        if len({tick[4] for tick in aligned}) >= 3 and aligned[-1][0] - aligned[0][0] >= .16:
            groups.append(aligned)
    if not groups:
        # An explicit axis label can corroborate a single readable tick in an overview.
        axis = [row["box"] for row in lines if row["text"].strip().casefold().rstrip(":") in {"time", "время"}]
        for tick in sorted(ticks):
            if any(box[0] + box[2] + .04 < tick[0] and abs(box[1] + box[3] / 2 - tick[1]) < .025
                   for box in axis):
                return max(0, tick[0] - .035), tick[2] + .002
        return None
    widest = max(groups, key=lambda group: group[-1][0] - group[0][0])
    return max(0, widest[0][0] - .05), max(tick[2] for tick in widest) + .002


def _clip_text(box, lines):
    """Collect OCR carried by one visual clip, excluding text outside its colored body."""
    x, y, w, h = box
    labels = []
    for row in lines:
        left, top, width, height = row["box"]
        center_x, center_y = left + width / 2, top + height / 2
        if x <= center_x <= x + w and y <= center_y <= y + h and width <= 1.15 * w:
            labels.append(row["text"])
    full = []
    for label in sorted(labels, key=len, reverse=True):
        if not any(label.casefold() in previous.casefold() for previous in full):
            full.append(label)
    return " ".join(full)[:160]


def _timeline_clips(image, lines, vertical_span):
    """Use an explicit ruler or irregular multi-lane shape evidence for clip candidates."""
    shapes = _filled_clip_shapes(image, vertical_span)
    ruler = _ruler_bounds(lines)
    if ruler:
        first_x, first_y = ruler
        shapes = [box for box in shapes if box[0] >= first_x and box[1] >= first_y]
    lanes = _clip_lanes(shapes)
    if not lanes:
        return []
    first = lanes[0]
    staggered = any(all(abs(box[0] - reference[0]) > .018 for reference in first)
                    for lane in lanes[1:] for box in lane)
    if not ruler and (len(lanes) < 3 or not staggered or not any(len(lane) >= 3 for lane in lanes)):
        return []
    basis = "filled-clips-below-time-ruler" if ruler else "staggered-filled-clips-in-lanes"
    result = []
    for index, lane in enumerate(lanes, 1):
        for box in lane:
            element = {"box": [round(part, 4) for part in box], "lane": index, "basis": basis}
            if text := _clip_text(box, lines):
                element["text"] = text
            result.append(element)
            if len(result) == MAX_CLIPS:
                return result
    return result


def has_distinct_rows(rows):
    """Drop repeated action labels while retaining rows with differing secondary cells."""
    signatures = {tuple(cell["text"].casefold() for cell in row.get("cells", [row])) for row in rows}
    return len(signatures) >= 2


def _describe_pixels(image, lines, vertical_span=1):
    """Return titles and lists supported by screenshot pixels and source-space OCR lines.

    The window result reports the visible title text box. A list rectangle encloses
    the measured row fields rather than extending to an inferred container border.
    """
    if image is None or min(image.shape[:2]) < 16:
        return {"windows": [], "lists": [], "timelineClips": []}
    windows = _window_title(_window_controls(image), lines, image)
    clips = _timeline_clips(image, lines, vertical_span)
    if len(lines) < 3:
        return {"windows": windows, "lists": [], "timelineClips": clips}
    gray = _gray_frame(image)
    lists = []
    tolerance = max(.006, 8 / image.shape[1])
    for candidate in _aligned_groups(lines, tolerance, vertical_span):
        run = _trim_heading(candidate)
        if len(run) < 3:
            continue
        basis = _visible_list_evidence(gray, run)
        if not basis:
            continue
        rows = [dict(row) for row in run[:MAX_ROWS]]
        box = _union(rows)
        if any(_shares_column(item, box, tolerance) for item in lists):
            continue
        lists.append({"box": box, "basis": basis, "rows": rows})
        if len(lists) == MAX_LISTS:
            break
    lists = _merge_adjacent_columns(lists)
    lists = [group for group in lists if has_distinct_rows(group["rows"])]
    return {"windows": windows, "lists": lists, "timelineClips": clips}


def _parse_region(region, image):
    """Map a finite normalized crop to exact source pixel boundaries."""
    height, width = image.shape[:2]
    if region is None:
        return 0, 0, width, height
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    if any(type(value) not in (int, float) or not isfinite(value) for value in region):
        return None
    x, y, w, h = region
    if min(x, y) < 0 or min(w, h) <= 0 or max(x + w, y + h) > 1.002:
        return None
    return round(x * width), round(y * height), min(width, round((x + w) * width)), min(height, round((y + h) * height))


def _project(box, region):
    """Transform normalized crop evidence back to full-image coordinates."""
    x, y, w, h = box
    left, top, rw, rh = region
    return [round(left + x * rw, 4), round(top + y * rh, 4), round(w * rw, 4), round(h * rh, 4)]


def _project_structure(result, region):
    """Remap every list, cell, clip and title rectangle from the same cropped screenshot."""
    for window in result["windows"]:
        window["box"] = _project(window["box"], region)
        window["titleBox"] = _project(window["titleBox"], region)
    for group in result["lists"]:
        group["box"] = _project(group["box"], region)
        for row in group["rows"]:
            row["box"] = _project(row["box"], region)
            for cell in row.get("cells", []):
                cell["box"] = _project(cell["box"], region)
    for clip in result["timelineClips"]:
        clip["box"] = _project(clip["box"], region)
    return result


def describe_structure(image, lines, *, region=None):
    """Measure visible screenshot evidence within an optional normalized source crop."""
    empty = {"windows": [], "lists": [], "timelineClips": []}
    if image is None or min(image.shape[:2]) < 16:
        return empty
    bounds = _parse_region(region, image)
    if not bounds:
        return empty
    left, top, right, bottom = bounds
    if min(right - left, bottom - top) < 16:
        return empty
    lines = _ocr_lines(lines)
    height, width = image.shape[:2]
    if bounds == (0, 0, width, height):
        return _describe_pixels(image, lines)
    actual = [left / width, top / height, (right - left) / width, (bottom - top) / height]
    rx, ry, rw, rh = actual
    inside = []
    for line in lines:
        x, y, w, h = line["box"]
        tolerance = .00011
        if (x >= rx - tolerance and y >= ry - tolerance
                and x + w <= rx + rw + tolerance and y + h <= ry + rh + tolerance):
            x1, y1 = max(x, rx), max(y, ry)
            x2, y2 = min(x + w, rx + rw), min(y + h, ry + rh)
            if x2 > x1 and y2 > y1:
                inside.append({"text": line["text"], "box": [
                    (x1 - rx) / rw, (y1 - ry) / rh, (x2 - x1) / rw, (y2 - y1) / rh]})
    result = _describe_pixels(image[top:bottom, left:right], inside, vertical_span=rh)
    return _project_structure(result, actual)
