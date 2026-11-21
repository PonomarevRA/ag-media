"""Register vertical screen motion and keep each OCR reading on the source clock."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import FULL, pixel_view, recognize
from voice_studio.vision.scene_objects import iou

MAX_PAGES = 16
MAX_PAGE_SCREENS = 5
MAX_REGISTRATION_GAP_MS = 500
MIN_PAGE_GROWTH = .18
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
    placements: list = field(default_factory=list)
    strips: deque = field(default_factory=deque)
    extension: int = 0
    offset: int = 0
    origin: int = 0
    registration: View = field(init=False)
    fixed_top: View | None = None

    def __post_init__(self):
        """Record the first viewport and its exact source frame."""
        self.placements.append((self.first.stamp, 0, self.first.box))
        self.strips.append(self.first.pixels)
        self.registration = self.first

    def detach_header(self, rows):
        """
        Keep stationary top chrome once and turn existing placements into body coordinates.

        rows is the verified sticky prefix height in viewport pixels. Only
        call before the first nonzero scroll displacement is accumulated.
        """
        self.fixed_top = View(self.first.stamp, view_slice_box(self.first.box, len(self.first.pixels), 0, rows),
                              self.first.pixels[:rows].copy())
        original_height = len(self.first.pixels)
        self.first = body_view(self.first, rows)
        self.registration = self.first
        self.last = body_view(self.last, rows)
        self.strips = deque([self.first.pixels])
        self.placements = [(stamp, offset, view_slice_box(box, original_height, rows, original_height))
                           for stamp, offset, box in self.placements]

    def growth(self, view, delta):
        """Calculate new document rows, including reversals into already seen content."""
        at = self.offset + delta
        right = self.origin + len(self.first.pixels) + self.extension
        return max(0, self.origin - at), max(0, at + len(view.pixels) - right)

    def add(self, view, delta):
        """Register one observed viewport and materialize only novel document rows."""
        above, below = self.growth(view, delta)
        at = self.offset + delta
        height = len(view.pixels)
        if below:
            overlap = height - below
            # If the last view touches this edge, cut inside a matching block
            # instead of at a moving browser footer.
            seam = overlap_seam(self.registration.pixels[delta:], view.pixels[:overlap]) if below == delta else overlap
            trim_strips(self.strips, overlap - seam, from_bottom=True)
            self.strips.append(view.pixels[seam:].copy())
        elif above:
            overlap = height - above
            seam = overlap_seam(view.pixels[above:], self.registration.pixels[:overlap], fallback=0) if above == -delta else 0
            trim_strips(self.strips, seam, from_bottom=False)
            self.strips.appendleft(view.pixels[:above + seam].copy())
            self.origin = at
        self.extension += above + below
        self.offset = at
        self.placements.append((view.stamp, at, view.box))
        self.last = view
        if delta:
            self.registration = view

    def image(self):
        """Join new strips in document order, with a single copy of overlapping rows."""
        header = (self.fixed_top.pixels,) if self.fixed_top is not None else ()
        return np.concatenate((*header, *self.strips))

    @property
    def header_height(self):
        """Return zero when the render has no detached fixed header."""
        return len(self.fixed_top.pixels) if self.fixed_top is not None else 0


def view_slice_box(box, height, start, end):
    """Map a vertical slice of a viewport raster to its normalized source box."""
    x, y, w, h = box
    return x, y + h * start / height, w, h * (end - start) / height


def body_view(view, removed):
    """Keep the moving rows from a measured view and its exact source geometry."""
    height = len(view.pixels)
    return View(view.stamp, view_slice_box(view.box, height, removed, height), view.pixels[removed:])


def sticky_prefix(previous, current, delta):
    """
    Find a textured prefix that matches in place while the body scrolls.

    previous and current are equal-size viewport rasters. delta is an
    independently registered nonzero displacement, in pixels. Return zero
    unless a stationary band has a well-supported boundary above moving rows.
    """
    height, width = previous.shape[:2]
    if abs(delta) < 8 or height < 100:
        return 0
    inset = width // 9
    a, b = [frame[:, inset:width - inset:3] for frame in (previous, current)]
    direct = cv2.absdiff(a, b).mean(axis=(1, 2))
    least, most = max(18, round(.045 * height)), round(.29 * height)
    prefix = 0
    for row in range(least, most):
        if np.count_nonzero(direct[row:row + 12] > 3.0) >= 7:
            prefix = row + int(np.argmax(direct[row:row + 12] > 3.0))
            break
    if not prefix or np.quantile(direct[:prefix], .80) > 2.6:
        return 0
    # Background flat color by itself is insufficient. A persistent menu or
    # header should contain aligned edges that would change under translation.
    shifted, stationary = (a, b) if delta > 0 else (b, a)
    distance = abs(delta)
    overlap = min(prefix, height - distance)
    if overlap < 12:
        return 0
    moving_alignment = cv2.absdiff(shifted[distance:distance + overlap], stationary[:overlap])
    texture = a[:prefix].std(axis=1).mean()
    return prefix if texture >= 9 and float(moving_alignment.mean()) >= 5 else 0



def trim_strips(strips, rows, *, from_bottom):
    """Replace a seam's overlapping rows without rebuilding the full page."""
    while rows:
        edge = strips.pop() if from_bottom else strips.popleft()
        if rows < len(edge):
            if from_bottom:
                strips.append(edge[:-rows])
            else:
                strips.appendleft(edge[rows:])
            break
        rows -= len(edge)


def overlap_seam(earlier, later, *, fallback=None):
    """Choose a matched block near the center to avoid cutting at viewport chrome."""
    height, width = earlier.shape[:2]
    if fallback is None:
        fallback = height
    if height < 36:
        return fallback
    inset = width // 8
    a = earlier[:, inset:width - inset:3]
    b = later[:, inset:width - inset:3]
    per_row = cv2.absdiff(a, b).mean(axis=(1, 2))
    middle = height // 2
    choices = range(max(8, height // 4), min(height - 8, 3 * height // 4), 4)
    costs = [(float(per_row[at - 7:at + 7].mean()) + .02 * abs(at - middle), at) for at in choices]
    cost, seam = min(costs, default=(float("inf"), fallback))
    return seam if cost <= 9 else fallback


def registration_pixels(pixels):
    """Downsample to a fixed working height and omit side chrome from motion matching."""
    width = pixels.shape[1]
    inset = max(2, round(.12 * width))
    center = pixels[:, inset:width - inset]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (min(208, center.shape[1]), 320), interpolation=cv2.INTER_AREA)


def refine_overlap(previous, current, coarse):
    """Align several overlapping blocks on the native vertical raster before adding a strip."""
    height, width = previous.shape[:2]
    half = min(64, width // 3)
    inset = (width - 2 * half) // 2
    a, b = [cv2.cvtColor(frame[:, inset:inset + 2 * half], cv2.COLOR_BGR2GRAY)
            for frame in (previous, current)]
    radius = max(2, round(height / 160))
    first = max(0, coarse) + radius
    stop = min(height, height + coarse) - radius
    blocks = []
    for top in range(first, stop - 20, max(25, (stop - first) // 4)):
        for x in (0, half):
            template = a[top:top + 20, x:x + half]
            if template.std() < 10:
                continue
            center = top - coarse
            search = b[center - radius:center + 20 + radius, x:x + half]
            scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).ravel()
            best = int(scores.argmax())
            if scores[best] >= .82:
                blocks.append(coarse + radius - best)
    if len(blocks) < 2:
        return coarse
    refined = round(float(np.median(blocks)))
    if sum(abs(delta - refined) <= 1 for delta in blocks) * 3 >= len(blocks) * 2:
        return refined
    return coarse


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
    if abs(delta) < 2:
        return 0
    if any(abs(other - delta) > 2 and len(rows) == len(support) for rows, other in ranked[:-1]):
        return None
    if delta > 0:
        overlap = cv2.absdiff(a[delta:], b[:-delta])
    else:
        overlap = cv2.absdiff(a[:delta], b[-delta:])
    if float(np.median(np.mean(overlap, axis=1))) > 19:
        return None
    return refine_overlap(previous, current, round(delta * len(previous) / height))


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
    """
    Accumulate a bounded document per stable panel, skipping one noisy moment.

    candidates are source frames with measured panel rectangles and shot IDs.
    Continue through direction changes only when pairwise pixel registration
    succeeds. Never paste a nonoverlapping or unrelated screen into the mosaic.
    """
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
            gap = stamp - page.last.stamp
            raw_view = view
            header_changed = False
            if page.fixed_top is not None:
                fixed = page.fixed_top.pixels
                observed = raw_view.pixels[:page.header_height]
                header_changed = fixed.shape != observed.shape or float(cv2.absdiff(fixed, observed).mean()) > 14
                if not header_changed:
                    view = body_view(raw_view, page.header_height)
            delta = (vertical_delta(page.registration.pixels, view.pixels)
                     if not header_changed and gap <= MAX_VISUAL_GAP_MS else None)
            if delta and not page.extension and not page.header_height:
                height = sticky_prefix(page.registration.pixels, view.pixels, delta)
                if height:
                    moving = body_view(raw_view, height)
                    estimate = vertical_delta(page.registration.pixels[height:], moving.pixels)
                    if estimate is not None and abs(estimate - delta) <= max(3, abs(delta) * .06):
                        page.detach_header(height)
                        view, delta = moving, estimate
            if delta is None and not header_changed and gap <= MAX_REGISTRATION_GAP_MS:
                continue
            too_tall = (delta is not None and page.extension + sum(page.growth(view, delta))
                        > (MAX_PAGE_SCREENS - 1) * len(view.pixels))
            if delta is None or too_tall:
                if page.extension >= MIN_PAGE_GROWTH * len(page.first.pixels):
                    pages.append(page)
                if len(pages) == MAX_PAGES:
                    return pages
                if delta is None:
                    page = Page(raw_view, raw_view)
                    continue
                # Start a new raster at the current full viewport after reaching
                # the page size limit. Its fixed header is measured anew.
                page = Page(raw_view, raw_view)
                continue
            page.add(view, delta)
        if page is not None and page.extension >= MIN_PAGE_GROWTH * len(page.first.pixels):
            pages.append(page)
        if len(pages) == MAX_PAGES:
            return pages
    return pages


def project_reading(element, page, page_height):
    """Place a full element on the first measured frame where it was visible."""
    x, y, w, h = element["box"]
    viewport_height = len(page.first.pixels)
    origin = page.origin
    top_on_page = y * page_height
    bottom_on_page = (y + h) * page_height
    header_height = page.header_height
    if page.fixed_top is not None and bottom_on_page <= header_height:
        left, upper, width, height = page.fixed_top.box
        box = [left + x * width, upper + top_on_page / header_height * height,
               w * width, (bottom_on_page - top_on_page) / header_height * height]
        return page.fixed_top.stamp, [left, upper, width, height], {**element, "box": [round(v, 4) for v in box]}
    if top_on_page < header_height:
        return None
    top = top_on_page - header_height + origin
    bottom = bottom_on_page - header_height + origin
    for stamp, offset, (left, upper, width, height) in page.placements:
        if offset <= top and bottom <= offset + viewport_height:
            box = [left + x * width, upper + (top - offset) / viewport_height * height,
                   w * width, h * page_height / viewport_height * height]
            return stamp, [left, upper, width, height], {**element, "box": [round(value, 4) for value in box]}
    return None


def resolve_repeated_mosaic_lines(lines):
    """
    Prefer the fuller reading where overlapping OCR bands read the same line.

    lines have normalized mosaic positions. Keep identical labels in different
    rows and avoid treating divergent readings in one position as new blocks.
    """
    selected = []
    for row in sorted(lines, key=lambda item: -len(item["text"])):
        label = "".join(char for char in row["text"].casefold() if char.isalnum())
        duplicate = False
        for prior, other in selected:
            if iou(row["box"], prior["box"]) >= .49 and (
                    label == other or (min(len(label), len(other)) >= 5
                                       and SequenceMatcher(None, label, other).ratio() >= .80)):
                duplicate = True
                break
        if not duplicate:
            selected.append((row, label))
    return sorted((row for row, _ in selected), key=lambda item: (item["box"][1], item["box"][0]))


def mosaic_ocr_jobs(key, path, pixels, viewport_height, folder, band_budget):
    """
    Prepare one full-page OCR read and optional overlapping bands for small text.

    key identifies the mosaic; path is the saved PNG. pixels holds its BGR
    raster. viewport_height is the original sampled window. folder receives
    private temporary crops and band_budget caps the additional jobs globally.
    All band readings project into the same mosaic coordinate space.
    """
    height, width = pixels.shape[:2]
    jobs = [(key, path, FULL, width, height)]
    if band_budget < 2 or height < 900 or height < 1.6 * viewport_height:
        return jobs
    rows = round(.62 * height)
    for index, top in enumerate((0, height - rows)):
        band = pixels[top:top + rows]
        scale = min(2, 1360 / max(band.shape[:2]))
        enlarged = cv2.resize(band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        target = Path(folder) / f"scroll-ocr-{key}-{index}.png"
        if cv2.imwrite(str(target), enlarged, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            jobs.append((key, target, (0, top / height, 1, rows / height),
                         enlarged.shape[1], enlarged.shape[0]))
    return jobs


def read_scroll_pages(candidates, folder=None, *, include_files=False):
    """Use at most one OCR crop per confirmed trajectory and restore source positions."""
    pages = collect_pages(candidates)
    if not pages:
        return {}, [], {}
    if folder is None:
        folder = candidates[0][2].parent
    jobs, valid, band_budget = [], {}, 8
    for key, page in enumerate(pages):
        path = Path(folder) / f"screen-scroll-{key:02}.png"
        pixels = page.image()
        if cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            height, width = pixels.shape[:2]
            prepared = mosaic_ocr_jobs(key, path, pixels, len(page.first.pixels) + page.header_height, folder, band_budget)
            jobs.extend(prepared)
            band_budget -= len(prepared) - 1
            valid[key] = page, height, width, path.name
    readings, statuses = recognize(jobs, line_limit=100, keep_numbers=True)
    from voice_studio.vision.screen_control_grid import control_grid

    by_stamp, summary = defaultdict(list), []
    for key, (page, height, width, filename) in valid.items():
        count = 0
        lines = resolve_repeated_mosaic_lines(readings.get(key, ()))
        mosaic_text = []
        for line in lines:
            if match := project_reading({**line, "role": "text_line", "basis": "scroll-mosaic"}, page, height):
                stamp, region, row = match
                by_stamp[stamp].append((region, row))
                count += 1
            mosaic_text.append({**line, **({"timeMs": stamp, "sourceBox": row["box"]} if match else {})})
        image = cv2.imread(str(Path(folder) / filename))
        layout = control_grid(image, lines) if image is not None else {"controls": []}
        controls = []
        for control in layout["controls"]:
            projected = project_reading(control, page, height)
            if projected:
                stamp, region, row = projected
                source_row = {field: row[field] for field in ("role", "box", "basis", "text") if field in row}
                source_row["mosaic"] = {"pageStartMs": page.first.stamp, "pageIndex": key,
                                        **{field: control[field] for field in ("box", "column", "row", "pixelBox")}}
                by_stamp[stamp].append((region, source_row))
                control = {**control, "timeMs": stamp, "sourceBox": row["box"]}
            controls.append(control)
        has_content = bool(count or controls)
        summary.append({"startMs": page.first.stamp, "endMs": page.last.stamp,
                        "samples": len(page.placements), "width": width, "height": height, "lines": count,
                        "viewportHeight": len(page.first.pixels) + page.header_height, "addedPixels": page.extension,
                        "fixedHeaderPixels": page.header_height,
                        "mosaicText": mosaic_text,
                        **({"grid": layout["grid"], "controls": controls} if image is not None else {}),
                        **({"_mosaicFile": filename} if include_files else {}),
                        "viewports": [{"timeMs": stamp, "box": bounds} for stamp, _, bounds in page.placements]
                                     if has_content or include_files else []})
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
            if line["role"] == "text_line":
                lines.append({**line, "captureIndex": index, "timeMs": actual})
    for page in pages:
        page["startMs"] = frame_ids[page["startMs"]][1]
        page["endMs"] = frame_ids[page["endMs"]][1]
        for view in page["viewports"]:
            view["captureIndex"], view["timeMs"] = frame_ids[view["timeMs"]]
        for item in (*page.get("controls", ()), *page.get("mosaicText", ())):
            if "timeMs" in item:
                item["captureIndex"], item["timeMs"] = frame_ids[item["timeMs"]]
    return {"status": "draft" if pages else "no-registered-scroll", "ocrJobs": len(pages),
            "ocrStatuses": statuses, "coordinateSpace": "normalized-oriented-image",
            "stretches": pages, "lines": lines}
