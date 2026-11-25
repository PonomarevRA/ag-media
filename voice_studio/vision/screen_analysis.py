"""Select screen crops and run local OCR without increasing ordinary scene sampling."""

from collections import defaultdict
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from shutil import which
from time import perf_counter

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import FULL, TESSDATA, pixel_view, recognize
from voice_studio.vision.frame_change_panels import SplitPanelProbe
from voice_studio.vision.scene_objects import iou
from voice_studio.vision.scene_sampling import evenly_spaced
from voice_studio.vision.screen_layout import device_regions, outlined_text_regions
from voice_studio.vision.screen_ui import analyze_interface
from voice_studio.vision.screen_scroll import read_scroll_pages
from voice_studio.vision.screen_regions import ScreenRegionStates
from voice_paths import temporary_directory

MAX_SCREEN_CROPS = 24
MAX_CONTROL_CROPS = 12
MAX_SUPPLEMENTAL_UI_CROPS = 16
FRAME_LAYOUT = {"box": list(FULL), "basis": "frame-layout"}


def side_panel(image):
    """Find a strong vertical seam compatible with a mobile recording beside a presenter."""
    height, width = image.shape[:2]
    if width < height * 1.15:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (192, 96), interpolation=cv2.INTER_AREA).astype(np.int16)
    differences = np.abs(np.diff(gray[8:-8], axis=1))
    seams = (differences > 24).mean(axis=0)
    best = None
    for column in np.argsort(-seams):
        if seams[column] < .58:
            break
        split = (int(column) + 1) / 192
        fraction = min(split, 1 - split)
        if not .18 <= fraction <= .46 or not .30 <= fraction * width / height <= .95:
            continue
        start = 0 if split <= .5 else split
        box = [round(start, 4), 0, round(fraction, 4), 1]
        best = {"box": box, "basis": "frame-layout"}
        break
    return best


def crop_job(image, path, region, stamp):
    """Enlarge small interface text and keep its original coordinate transform."""
    pixels, bounds = pixel_view(image, region)
    height, width = pixels.shape[:2]
    if min(width, height) < 24:
        return None
    scale = min(2.5, 640 / min(width, height), 1800 / max(width, height))
    if scale > 1.15:
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        return None
    height, width = pixels.shape[:2]
    return stamp, path, bounds, width, height


def control_job(image, path, region, stamp):
    """Prepare a single UI label and account for padding in the OCR position transform."""
    pixels, bounds = pixel_view(image, region)
    if min(pixels.shape[:2]) < 6:
        return None
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    if np.median(gray) < 125:
        gray = 255 - gray
    height, width = gray.shape
    scale = min(5, 82 / height, 1450 / width)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height, width = gray.shape
    pad = 12
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    if not cv2.imwrite(str(path), gray):
        return None
    x, y, w, h = bounds
    ox, oy = w * pad / width, h * pad / height
    return stamp, path, (x - ox, y - oy, w + 2 * ox, h + 2 * oy), width + 2 * pad, height + 2 * pad


def read_control_labels(candidates, limit=MAX_CONTROL_CROPS):
    """Read a bounded set of outline interiors using the one-line OCR mode."""
    jobs = []
    previous_path, image = None, None
    for index, (stamp, path, bounds) in enumerate(evenly_spaced(candidates, limit)):
        if path != previous_path:
            previous_path, image = path, cv2.imread(str(path))
        if image is not None and (job := control_job(image, path.with_name(f"control-{index}.png"), bounds, stamp)):
            jobs.append(job)
    readings, statuses = recognize(jobs, line_limit=48, keep_numbers=True, page_mode=7)
    return readings, statuses, len(jobs)


def merge_readings(*groups):
    """Keep one reading per location, preferring OCR from the tightest crop."""
    kept = []
    for rows in groups:
        for row in rows:
            normalized = " ".join(row["text"].casefold().split())
            duplicate = False
            for existing in kept:
                overlap = iou(row["box"], existing["box"])
                prior = " ".join(existing["text"].casefold().split())
                if overlap >= .72 or (overlap >= .35 and normalized == prior):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
    return sorted(kept, key=lambda row: (row["box"][1], row["box"][0]))


def panel_ocr_regions(panel, stitched):
    """Preserve fixed strips around a successfully recognized scrolling interior."""
    if stitched is None:
        return [panel]
    x, y, w, h = panel
    a, b, u, v = stitched
    left, top = max(x, a), max(y, b)
    right, bottom = min(x + w, a + u), min(y + h, b + v)
    if left >= right or top >= bottom:
        return [panel]
    return [(x, y, w, top - y), (x, bottom, w, y + h - bottom),
            (x, top, left - x, bottom - top), (right, top, x + w - right, bottom - top)]


def control_outside_content(box, stitched):
    """Keep control OCR outside the already-read scroll interior."""
    if stitched is None:
        return True
    x, y, w, h = box
    left, top, width, height = stitched
    return not (left <= x + w / 2 <= left + width and top <= y + h / 2 <= top + height)


class ScreenAnalysis:
    """Keep temporally distributed, deduplicated screen observations per scene pass."""

    def __init__(self):
        self.candidates = []
        self.region_frames = set()
        self.scroll_candidates = []
        self.panel_samples = defaultdict(list)
        self.history = {}
        self.split_probe = SplitPanelProbe()
        self.detail_states = ScreenRegionStates()

    def sample(self, image, path, sample, objects):
        """Look for device crops and split-screen panels on already sampled frames."""
        regions = device_regions(objects)
        panel = self.split_probe.sample(image, sample) if "changeBands" in sample else None
        panel = panel or side_panel(image)
        self.detail_states.sample(image, sample, panel)
        if panel:
            # A detected handset can appear beside a separate screen recording.
            regions = [panel, *[region for region in regions if iou(panel["box"], region["box"]) < .65]]
        if not regions and sample["anchor"] and len(outlined_text_regions(image, FRAME_LAYOUT)) >= 2:
            regions = [FRAME_LAYOUT]
        stamp = sample["timeMs"]
        if panel:
            self.scroll_candidates.append((sample["shot"], stamp, path, panel))
        self.panel_samples[sample["shot"]].append((stamp, panel))
        if regions:
            self.region_frames.add(stamp)
        for index, region in enumerate(regions):
            pixels, _ = pixel_view(image, region["box"])
            if min(pixels.shape[:2]) < 24:
                continue
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            assisted = region.get("changeAssisted", False)
            size = (128, 96) if assisted else (32, 32)
            signature = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            key = sample["shot"], index, region.get("deviceLabel", region["basis"])
            if previous := self.history.get(key):
                prior_stamp, prior_box, prior = previous
                if prior.shape == signature.shape:
                    if stamp - prior_stamp < 350:
                        continue
                    if iou(region["box"], prior_box) >= (.98 if assisted else .7):
                        delta = cv2.absdiff(prior, signature)
                        if assisted and cv2.mean(delta)[0] < .45 and (delta >= 12).mean() < .004:
                            region["readingTimeMs"] = prior_stamp
                            continue
                        if not assisted and cv2.mean(delta)[0] < 4:
                            continue
            self.history[key] = stamp, region["box"], signature
            self.candidates.append((stamp, path, region))

    def retain_recognized_screens(self, readings):
        """Add text-supported UI regions to the same dense sampling plan as split panels."""
        existing = {at for _, at, _, _ in self.scroll_candidates}
        candidates = {}
        for stamp, path, region in self.candidates:
            if stamp in existing or stamp in candidates:
                continue
            x, y, w, h = region["box"]
            inside = 0
            for line in readings.get(stamp, ()):
                a, b, p, q = line["box"]
                inside += x <= a + p / 2 <= x + w and y <= b + q / 2 <= y + h
            if inside < 3:
                continue
            kind = "mobile-ui" if region.get("deviceLabel") == "cell phone" else "ui"
            candidates[stamp] = (path, {**region, "knownPanel": True, "candidateType": kind})
        for shot, observations in self.panel_samples.items():
            for index, (stamp, panel) in enumerate(observations):
                if panel is None and stamp in candidates:
                    path, region = candidates[stamp]
                    self.scroll_candidates.append((shot, stamp, path, region))
                    observations[index] = stamp, region
        self.scroll_candidates.sort(key=lambda row: row[1])

    def summarize(self, frame_paths, readings, scrolls=(), *, supplemental=()):
        """Add optional interface details with a bounded, separate OCR budget."""
        jobs, targets, controls = [], defaultdict(list), []
        covered = defaultdict(list)
        for page in scrolls:
            for viewport in page.get("viewports", ()):
                covered[viewport["timeMs"]].append(viewport["box"])
        selected = evenly_spaced(self.candidates, MAX_SCREEN_CROPS)
        # Decode often in a stable viewport, then OCR only a bounded subset outside
        # the mosaic interior, which was already read in its document coordinates.
        supplemental = evenly_spaced(list(supplemental), MAX_SUPPLEMENTAL_UI_CROPS)
        paths = {**frame_paths, **{stamp: path for _, stamp, path, _, _ in supplemental}}
        dense_stamps = {stamp for _, stamp, _, _, _ in supplemental}
        selected.extend((stamp, path, region) for _, stamp, path, region, _ in supplemental)
        for index, (stamp, path, region) in enumerate(selected):
            image = cv2.imread(str(path))
            if image is None:
                continue
            stitched = next((box for box in covered[stamp] if iou(box, region["box"]) >= .4), None)
            if stamp in dense_stamps and stitched is not None:
                continue
            if region is not FRAME_LAYOUT:
                for piece, bounds in enumerate(panel_ocr_regions(region["box"], stitched)):
                    target = path.with_name(f"screen-{index}-{piece}.jpg")
                    if job := crop_job(image, target, bounds, stamp):
                        jobs.append(job)
            targets[stamp].append(region)
            controls.extend((stamp, path, box) for box in outlined_text_regions(image, region)
                            if control_outside_content(box, stitched))
        screen_readings, statuses = recognize(jobs, line_limit=48, keep_numbers=True)
        label_readings, label_statuses, control_count = read_control_labels(controls)
        statuses.update({f"controls-{key}": value for key, value in label_statuses.items()})
        results = {}
        for stamp, path in paths.items():
            base = readings.get(stamp, ())
            if stamp in targets:
                regions = targets[stamp]
            elif stamp not in self.region_frames and len(base) >= 2:
                regions = (FRAME_LAYOUT,)
            else:
                continue
            image = cv2.imread(str(path))
            if image is not None:
                lines = merge_readings(label_readings.get(stamp, ()), screen_readings.get(stamp, ()), base)
                layouts = [layout for region in regions if (layout := analyze_interface(image, region, lines))]
                if layouts:
                    results[stamp] = layouts
        return results, len(jobs) + control_count, statuses

    def retain_panels(self, layouts, pages, extra_samples):
        """Keep consecutive split panels supported by local OCR or a measured scrolling viewport."""
        shots = {stamp: shot for shot, stamp, _, _ in self.scroll_candidates}
        shots.update((stamp, shot) for shot, stamp, *_ in extra_samples)
        viewports = defaultdict(list)
        for page in pages:
            if page["lines"] >= 3:
                for view in page["viewports"]:
                    at = view["timeMs"]
                    if at in shots:
                        viewports[shots[at]].append((at, view["box"]))
        for shot, samples in self.panel_samples.items():
            run, confirmed = [], False
            for stamp, panel in [*samples, (0, None)]:
                if (not panel or (run and (stamp - run[-1][0] > 1800
                                            or iou(panel["box"], run[-1][1]["box"]) < .8))):
                    if confirmed and (len(run) >= 2 or not run[0][1].get("changeAssisted")):
                        for at, measured in run:
                            regions = layouts.setdefault(at, [])
                            exact = next((row for row in regions if iou(row["box"], measured["box"]) >= .95), None)
                            if exact is None:
                                exact = {**measured, "elements": [], "connections": []}
                                regions.insert(0, exact)
                            exact["panelType"] = measured.get("candidateType", "mobile-ui")
                            reading = measured.get("readingTimeMs")
                            if (not exact["elements"] and reading is not None
                                    and any(prior["elements"] and iou(prior["box"], exact["box"]) >= .98
                                            for prior in layouts.get(reading, ()))):
                                exact["readingTimeMs"] = reading
                            else:
                                exact.pop("readingTimeMs", None)
                    run, confirmed = [], False
                if not panel:
                    continue
                run.append((stamp, panel))
                confirmed |= any(abs(at - stamp) <= 600 and iou(bounds, panel["box"]) >= .5
                                 for at, bounds in viewports.get(shot, ()))
                confirmed |= any(
                    (row["basis"] == "frame-layout" or panel.get("knownPanel")) and iou(row["box"], panel["box"]) >= .5
                    and sum(bool(element.get("text")) for element in row["elements"]) >= 3
                    for row in layouts.get(stamp, ())
                )

    def scroll(self, layouts, extra_samples=(), *, folder, include_mosaic_files=False):
        """Write mosaics to the worker's output directory and project lines onto sampled frames."""
        samples = {(shot, stamp): (shot, stamp, path, region)
                   for shot, stamp, path, region in self.scroll_candidates}
        for shot, stamp, path, region in extra_samples:
            samples.setdefault((shot, stamp), (shot, stamp, path, region))
        readings, pages, statuses = read_scroll_pages(sorted(samples.values(), key=lambda row: row[1]),
                                                       folder=folder,
                                                       include_files=include_mosaic_files)
        for stamp, rows in readings.items():
            regions = layouts.setdefault(stamp, [])
            for bounds, row in rows:
                add_scroll_line(regions, bounds, row)
        return pages, statuses


def merge_scroll_layouts(layouts, mosaics):
    """Prefer recognized mosaic lines at measured frames while retaining static layout traces."""
    for stamp, regions in mosaics.items():
        for region in regions:
            for line in region["elements"]:
                add_scroll_line(layouts.setdefault(stamp, []), region["box"], line)


def add_scroll_line(regions, bounds, line):
    """Attach the source-frame box to an enclosing region compatible with the video graph."""
    for region in regions:
        x, y, w, h = line["box"]
        left, upper, width, height = region["box"]
        if x < left or y < upper or x + w > left + width + .001 or y + h > upper + height + .001:
            continue
        elements = region["elements"]
        for element in elements:
            same_text = element.get("text", "").casefold() == line.get("text", "").casefold()
            same_role = element["role"] == line["role"]
            if same_text and (same_role or bool(line.get("text"))) and iou(element["box"], line["box"]) > .3:
                if element["role"] == "text_line":
                    element.update(line)
                elif "mosaic" in line:
                    element["mosaic"] = line["mosaic"]
                return
        if len(elements) < 48:
            elements.append(line)
        return
    if len(regions) < 2:
        regions.append({"box": bounds, "basis": "frame-layout", "elements": [line], "connections": []})




def screenshot_crop(image, path, bounds, *, max_side=1280, enlarge=False):
    """Write one oriented OCR input with geometry still relative to the whole screenshot."""
    pixels, box = pixel_view(image, bounds)
    height, width = pixels.shape[:2]
    if min(height, width) < 12:
        raise ValueError("Область скриншота слишком мала.")
    scale = min(max_side / max(height, width), 2.5 if enlarge else 1)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=interpolation)
    if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise ValueError("Не удалось подготовить скриншот для OCR.")
    height, width = pixels.shape[:2]
    return 0, path, box, width, height


def screenshot_tiles(box):
    """Produce four overlapping tiles inside the requested normalized rectangle."""
    left, top, width, height = box
    return [(left + x * width, top + y * height, .56 * width, .56 * height)
            for y in (0, .44) for x in (0, .44)]


def leading_caption_band(lines):
    """
    Suggest one detailed OCR crop above a vertically aligned text group.

    lines contains recognized boxes in whole-image coordinates. Only a
    sizable blank band immediately above a corroborated group is considered.
    This can recover isolated numerals or headings missed in a global pass.
    """
    for line in lines:
        x, y, w, h = line["box"]
        if not .10 <= y <= .30 or w > .42:
            continue
        center = x + w / 2
        stacked = sum(abs(other["box"][0] + other["box"][2] / 2 - center) <= .095
                      and y + h < other["box"][1] < y + .13 for other in lines)
        if stacked < 2:
            continue
        top = max(.012, y - .145)
        left, right = max(0, center - .16), min(1, center + .16)
        if any(other["box"][1] < y - .006 and other["box"][1] + other["box"][3] > top
               and other["box"][0] < right and other["box"][0] + other["box"][2] > left
               for other in lines):
            continue
        return left, top, right - left, y - .006 - top
    return None


def sparse_caption_jobs(image, folder, band):
    """
    Split separated high-contrast groups in a shallow caption band for local OCR.

    image is the original oriented raster. folder receives at most two
    monochrome crops. band is normalized; every job retains source geometry.
    """
    pixels, bounds = pixel_view(image, band)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    reverse = cv2.THRESH_BINARY_INV if float(np.median(gray)) >= 128 else cv2.THRESH_BINARY
    _, foreground = cv2.threshold(gray, 0, 255, reverse | cv2.THRESH_OTSU)
    columns = np.flatnonzero(np.count_nonzero(foreground, axis=0) >= .05 * height)
    groups = np.split(columns, np.flatnonzero(np.diff(columns) >= max(5, .05 * width)) + 1)
    jobs = []
    a, b, dx, dy = bounds
    for group in groups:
        if not len(group):
            continue
        left, right = max(0, int(group[0]) - 3), min(width, int(group[-1]) + 4)
        if right - left < 13:
            continue
        bitmap = cv2.resize(cv2.bitwise_not(foreground[:, left:right]), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        target = folder / f"heading-{len(jobs)}.png"
        if cv2.imwrite(str(target), bitmap):
            jobs.append((0, target, (a + left / width * dx, b, (right - left) / width * dx, dy),
                         bitmap.shape[1], bitmap.shape[0]))
        if len(jobs) == 2:
            break
    return jobs


def read_screenshot(image, region, mode, work_dir=None):
    """Return OCR and a job count for either a one-pass overview or a bounded detailed read."""
    scope = tuple(region["box"])
    workspace = nullcontext(work_dir) if work_dir is not None else temporary_directory()
    with workspace as folder:
        root = Path(folder)
        primary = screenshot_crop(image, root / "oriented.png", scope,
                                  max_side=1280 if mode == "overview" else 8192,
                                  enlarge=scope != FULL and mode == "full")
        jobs = [primary]
        if mode == "full" and max(primary[3:]) >= 900:
            for index, bounds in enumerate(screenshot_tiles(scope)):
                if job := crop_job(image, root / f"screen-{index}.jpg", bounds, 0):
                    jobs.append(job)
        readings, statuses = recognize(jobs, line_limit=80, keep_numbers=True, compile_vision=mode == "full")
        compact_readings = readings.get(0, ())
        heading_jobs = []
        if mode == "full" and scope == FULL and max(image.shape[:2]) < 850:
            if band := leading_caption_band(compact_readings):
                heading_jobs = sparse_caption_jobs(image, root, band)
        # Tesseract's single-line mode is useful on isolated, thresholded labels.
        tess_ready = heading_jobs and which("tesseract") and all(
            (TESSDATA / f"{name}.traineddata").is_file() for name in ("rus", "eng"))
        headings, heading_statuses = recognize(heading_jobs, line_limit=8, keep_numbers=True, page_mode=7,
                                               compile_vision=False, backend="tesseract" if tess_ready else None)
        label_jobs = []
        if mode == "full":
            for index, box in enumerate(outlined_text_regions(image, region)[:MAX_CONTROL_CROPS]):
                if job := control_job(image, root / f"label-{index}.png", box, 0):
                    label_jobs.append(job)
        labels, label_statuses = recognize(label_jobs, line_limit=48, keep_numbers=True, page_mode=7)
    statuses.update({f"controls-{key}": value for key, value in label_statuses.items()})
    statuses.update({f"heading-{key}": value for key, value in heading_statuses.items()})
    return (merge_readings(labels.get(0, ()), compact_readings, headings.get(0, ())),
            statuses, len(jobs) + len(label_jobs) + len(heading_jobs))


def screenshot_layout(source, mode="full", region=None, session_dir=None, work_dir=None,
                      include_ui=True, include_objects=False):
    """Analyze one oriented image with the shared OCR/layout path and optional existing object detector."""
    from voice_studio.vision.screenshot_structure import describe_structure
    from voice_studio.vision.screen_session import cached_layout, save_layout

    started = perf_counter()
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("Не удалось декодировать скриншот.")
    height, width = image.shape[:2]
    scope = {"box": list(FULL if region is None else region), "basis": "screenshot"}
    pixels, _ = pixel_view(image, scope["box"])
    cache_path = None
    if session_dir:
        fingerprint = sha256(pixels.tobytes())
        fingerprint.update(json.dumps([mode, scope["box"], width, height, pixels.shape, include_ui, include_objects, 8]).encode())
        cache_path = Path(session_dir) / (fingerprint.hexdigest() + ".json")
        if result := cached_layout(cache_path):
            return {**result, "reuse": "unchanged-pixels", "ocrJobs": 0,
                    "elapsedMs": round(1000 * (perf_counter() - started), 1)}
    lines, statuses, count = [], {}, 0
    layout, structure = None, {"windows": [], "lists": [], "timelineClips": []}
    if include_ui:
        lines, statuses, count = read_screenshot(image, scope, mode, work_dir)
        layout = analyze_interface(image, scope, lines, detailed=mode == "full")
        structure = describe_structure(image, lines, region=scope["box"])
    result = {"imageSize": [width, height], "ocr": lines, "interface": layout, "ocrStatuses": statuses,
              "mode": mode, "scopeBox": scope["box"], "structure": structure, "ocrJobs": count}
    if include_ui and mode == "full" and pixels.shape[0] > 1.5 * pixels.shape[1]:
        from voice_studio.vision.screen_control_grid import control_grid
        result["controlGrid"] = control_grid(image, lines, scope["box"])
    if include_objects:
        from voice_studio.vision.image_objects import detect_image_objects
        result.update(detect_image_objects(image, mode))
    result["uiStatus"] = "skipped"
    if include_ui:
        from voice_studio.vision.image_status import ocr_coverage
        result["uiStatus"] = ocr_coverage(statuses, has_lines=bool(lines))
    if cache_path and result.get("objectStatus", "ok") == "ok" and all(
        statuses.get(f"{prefix}tesseract", statuses.get(f"{prefix}apple-vision", "ok")) == "ok"
        for prefix in ("", "controls-")
    ):
        save_layout(cache_path, result)
    return {**result, "reuse": None, "elapsedMs": round(1000 * (perf_counter() - started), 1)}
