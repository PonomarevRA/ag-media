"""Register vertical screen motion and keep each OCR reading on the source clock."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import FULL, pixel_view, recognize
from voice_studio.vision.scene_objects import iou

MAX_PAGES = 16
MAX_PAGE_SCREENS = 4
MAX_VISUAL_GAP_MS = 1600
MIN_CORRELATION = .73
MIN_OVERLAP = .42


@dataclass
class View:
    """Pixels and source-frame location for a measured browser viewport."""

    stamp: int
    box: tuple
    pixels: np.ndarray


@dataclass
class Page:
    """One uninterrupted vertical scroll, with its observed viewport placements."""

    first: View
    last: View
    step: int = 0
    placements: list = field(default_factory=list)
    strips: list = field(default_factory=list)
    extension: int = 0

    def __post_init__(self):
        """Record the origin viewport so lines already visible retain their source frame."""
        self.placements.append((self.first.stamp, 0, self.first.box))

    def add(self, view, delta):
        """Add only new rows from a verified motion in the same direction."""
        if delta:
            self.step = delta
            self.extension += abs(delta)
            fresh = view.pixels[-delta:] if delta > 0 else view.pixels[:-delta]
            self.strips.append(fresh.copy())
        offset = self.extension if self.step >= 0 else -self.extension
        self.placements.append((view.stamp, offset, view.box))
        self.last = view

    def image(self):
        """Join new strips in document order, with a single copy of overlapping rows."""
        if self.step > 0:
            strips = [self.first.pixels, *self.strips]
        else:
            strips = [*reversed(self.strips), self.first.pixels]
        return np.concatenate(strips)


def registration_pixels(pixels):
    """Downsample to a fixed working height and omit side chrome from motion matching."""
    width = pixels.shape[1]
    inset = max(2, round(.12 * width))
    center = pixels[:, inset:width - inset]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (min(208, center.shape[1]), 320), interpolation=cv2.INTER_AREA)


def vertical_delta(previous, current):
    """Return content offset in viewport pixels only when separate bands agree."""
    if previous.shape != current.shape:
        return None
    a, b = registration_pixels(previous), registration_pixels(current)
    height = len(a)
    if np.mean(cv2.absdiff(a, b)) < 1.2:
        return 0
    band_height = 44
    matches = []
    for top in (32, 84, 136, 188, 240):
        template = a[top:top + band_height]
        if template.std() < 14:
            continue
        scores = cv2.matchTemplate(b, template, cv2.TM_CCOEFF_NORMED).ravel()
        best = int(scores.argmax())
        score = float(scores[best])
        if score < MIN_CORRELATION or abs(top - best) > height * (1 - MIN_OVERLAP):
            continue
        # Repeated list rows may give several equally plausible displacements.
        other = np.concatenate((scores[:max(0, best - 4)], scores[best + 5:]))
        if len(other) and score - float(other.max()) >= .045:
            matches.append((top - best, score))
    if len(matches) < 2:
        return None
    ranked = sorted(
        (([row for row in matches if abs(row[0] - delta) <= 2], delta) for delta, _ in matches),
        key=lambda item: (len(item[0]), sum(row[1] for row in item[0])),
    )
    support, delta = ranked[-1]
    if len(support) < 2 or len(support) * 2 < len(matches):
        return None
    delta = round(float(np.median([row[0] for row in support])))
    if abs(delta) < 3:
        return 0
    if any(abs(other - delta) > 2 and len(rows) == len(support) for rows, other in ranked[:-1]):
        return None
    if delta > 0:
        overlap = cv2.absdiff(a[delta:], b[:-delta])
    else:
        overlap = cv2.absdiff(a[:delta], b[-delta:])
    if float(np.median(np.mean(overlap, axis=1))) > 19:
        return None
    return round(delta * len(previous) / height)


def view_for(path, bounds, stamp, *, content_only=False):
    """Load the stable panel interior and retain the exact rounded source transform."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    x, y, w, h = bounds
    # Leave space for fixed browser chrome and floating buttons at the bottom.
    # Whole-frame OCR still reads those margins on its regular schedule.
    interior = bounds if content_only else (x + .055 * w, y + .13 * h, w * .92, h * .72)
    pixels, box = pixel_view(image, interior)
    if min(pixels.shape[:2]) < 50:
        return None
    height, width = pixels.shape[:2]
    scale = min(1.0, 920 / height, 640 / width)
    if scale < 1:
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return View(stamp, box, pixels)


def stable_runs(candidates):
    """Separate panels at cuts, losses, and geometric jumps before motion analysis."""
    grouped = defaultdict(list)
    for shot, stamp, path, region in candidates:
        x, _, width, _ = region["box"]
        side = "left" if x + width / 2 < .5 else "right"
        grouped[shot, side].append((stamp, path, region))
    for (shot, _), rows in grouped.items():
        current = []
        for sample in rows:
            stamp, _, region = sample
            if current:
                previous_stamp, _, previous_region = current[-1]
                if stamp - previous_stamp > MAX_VISUAL_GAP_MS or iou(region["box"], previous_region["box"]) < .68:
                    if len(current) >= 2:
                        yield shot, current
                    current = []
            current.append(sample)
        if len(current) >= 2:
            yield shot, current


def scrolling_intervals(candidates, *, padding_ms):
    """Request dense samples only near visible changes to a stable screen panel."""
    intervals = defaultdict(list)
    for shot, group in stable_runs(candidates):
        bounds = np.median([region["box"] for _, _, region in group], axis=0)
        previous = None
        for stamp, path, _ in group:
            view = view_for(path, bounds, stamp)
            if view is None:
                continue
            signature = registration_pixels(view.pixels)
            if previous is not None and np.mean(cv2.absdiff(previous[1], signature)) >= 4:
                start, end = previous[0] - padding_ms, stamp + padding_ms
                ranges = intervals[shot]
                if ranges and ranges[-1][1] >= start:
                    ranges[-1][1] = max(ranges[-1][1], end)
                else:
                    ranges.append([start, end])
            previous = stamp, signature
    return intervals


def collect_pages(candidates):
    """Keep bounded mosaics and restart on cuts, reversal, or weak registration."""
    pages = []
    for _, group in stable_runs(candidates):
        dimensions = np.median([region["box"] for _, _, region in group], axis=0)
        page = None
        for stamp, path, region in group:
            view = view_for(path, dimensions, stamp, content_only=region.get("contentOnly", False))
            if view is None:
                continue
            if page is None:
                page = Page(view, view)
                continue
            delta = vertical_delta(page.last.pixels, view.pixels)
            reversal = delta and page.step * delta < 0
            too_tall = delta and page.extension + abs(delta) > (MAX_PAGE_SCREENS - 1) * len(view.pixels)
            if delta is None or reversal or too_tall:
                if page.extension >= .10 * len(page.first.pixels):
                    pages.append(page)
                if delta is None:
                    page = Page(view, view)
                else:
                    page = Page(page.last, page.last)
                    page.add(view, delta)
                if len(pages) == MAX_PAGES:
                    return pages
                continue
            page.add(view, delta)
        if page is not None and page.extension >= .10 * len(page.first.pixels):
            pages.append(page)
        if len(pages) == MAX_PAGES:
            return pages
    return pages


def project_reading(line, page, page_height):
    """Place a full OCR line on the first measured frame where it was visible."""
    x, y, w, h = line["box"]
    viewport_height = len(page.first.pixels)
    origin = 0 if page.step > 0 else -page.extension
    top = y * page_height + origin
    bottom = top + h * page_height
    for stamp, offset, (left, upper, width, height) in page.placements:
        if offset <= top and bottom <= offset + viewport_height:
            box = [left + x * width, upper + (top - offset) / viewport_height * height,
                   w * width, h * page_height / viewport_height * height]
            return stamp, [left, upper, width, height], {
                "role": "text_line", "basis": "scroll-mosaic", "text": line["text"],
                "box": [round(value, 4) for value in box],
            }
    return None


def read_scroll_pages(candidates, folder=None, *, include_files=False):
    """Use at most one OCR crop per confirmed trajectory and restore source positions."""
    pages = collect_pages(candidates)
    if not pages:
        return {}, [], {}
    if folder is None:
        folder = candidates[0][2].parent
    jobs, valid = [], {}
    for key, page in enumerate(pages):
        path = Path(folder) / f"screen-scroll-{key:02}.png"
        pixels = page.image()
        if cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            height, width = pixels.shape[:2]
            jobs.append((key, path, FULL, width, height))
            valid[key] = page, height, width, path.name
    readings, statuses = recognize(jobs, line_limit=100, keep_numbers=True)
    by_stamp, summary = defaultdict(list), []
    for key, (page, height, width, filename) in valid.items():
        count = 0
        for line in readings.get(key, ()):
            if match := project_reading(line, page, height):
                stamp, region, row = match
                by_stamp[stamp].append((region, row))
                count += 1
        summary.append({"startMs": page.first.stamp, "endMs": page.last.stamp,
                        "samples": len(page.placements), "width": width, "height": height, "lines": count,
                        **({"_mosaicFile": filename} if include_files and count else {}),
                        "viewports": [{"timeMs": stamp, "box": bounds} for stamp, _, bounds in page.placements]
                                     if count else []})
    return by_stamp, summary, statuses


def screenshot_scroll_layout(request, folder):
    """Reuse scroll registration on a sequence of local captures in their supplied order."""
    frame_ids = {}
    candidates = []
    for index, sample in enumerate(request["captures"]):
        stamp = index * 250
        frame_ids[stamp] = index, sample["timeMs"]
        region = {"box": request["region"], "contentOnly": request["contentOnly"]}
        candidates.append((0, stamp, Path(sample["path"]), region))
    readings, pages, statuses = read_scroll_pages(candidates, folder)
    lines = []
    for stamp, projected in sorted(readings.items()):
        index, actual = frame_ids[stamp]
        for _, line in projected:
            lines.append({**line, "captureIndex": index, "timeMs": actual})
    for page in pages:
        page["startMs"] = frame_ids[page["startMs"]][1]
        page["endMs"] = frame_ids[page["endMs"]][1]
        for view in page["viewports"]:
            view["captureIndex"], view["timeMs"] = frame_ids[view["timeMs"]]
    return {"status": "draft" if lines else "no-registered-scroll", "ocrJobs": len(pages),
            "ocrStatuses": statuses, "coordinateSpace": "normalized-oriented-image",
            "stretches": pages, "lines": lines}
