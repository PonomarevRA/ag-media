"""Read independently changing areas of verified panels on their source clock."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import recognize
from voice_studio.vision.scene_ocr_reuse import ACTIVE_SCENE_OCR
from voice_studio.vision.scene_objects import iou
from voice_studio.vision.scene_sampling import evenly_spaced
from voice_studio.vision.screen_analysis import crop_job
from voice_studio.vision.screen_scroll import stable_runs, vertical_delta, view_for

SIGNATURE_WIDTH = 160
SIGNATURE_HEIGHT = 192
BANDS = 16
COLUMNS = 4
MAX_SAMPLE_GAP_MS = 180
MOTION_BUCKET_MS = 180
NEAR_READING_MS = 240
MAX_EPOCH_MS = 12_000
PIXEL_THRESHOLD = 20
CHANGED_TILE_FRACTION = .022
STRONG_PIXEL_THRESHOLD = 44
STRONG_TILE_FRACTION = .010
MIN_ZONE_BANDS = 2
MAX_CROPS = 48


@dataclass(slots=True)
class PanelFrame:
    """One measured viewport and a small raster used only for choosing OCR crops."""

    stamp: int
    path: Path
    source_frame: int
    signature: np.ndarray


@dataclass(slots=True)
class StateCrop:
    """A source-frame crop and the observed time support for its visible contents."""

    shot: int
    panel_box: list
    panel_type: str
    region: tuple
    epoch_start: int
    representative: PanelFrame
    start: int
    last: int
    count: int
    sharpness: float
    kind: str


def tile_fractions(delta, threshold):
    """Measure changed pixel shares in individual parts of each band."""
    changed = delta >= threshold
    return changed.reshape(BANDS, SIGNATURE_HEIGHT // BANDS, COLUMNS, SIGNATURE_WIDTH // COLUMNS).mean(axis=(1, 3))


def changed_tiles(first, second):
    """Detect compact strong changes such as one edited digit."""
    delta = cv2.absdiff(first, second).max(axis=2)
    noisy_change = tile_fractions(delta, PIXEL_THRESHOLD) >= CHANGED_TILE_FRACTION
    glyph_change = tile_fractions(delta, STRONG_PIXEL_THRESHOLD) >= STRONG_TILE_FRACTION
    return noisy_change | glyph_change


def panel_epochs(frames):
    """Separate large page changes and stop time support at an unobserved gap."""
    start = 0
    for index, current in enumerate(frames[1:], start=1):
        previous = frames[index - 1]
        delta = cv2.absdiff(previous.signature, current.signature).max(axis=2)
        changes = tile_fractions(delta, PIXEL_THRESHOLD)
        page_jump = (changes.max(axis=1) > .10).mean() >= .88 and changes.mean() >= .15
        if (current.stamp - previous.stamp > MAX_SAMPLE_GAP_MS
                or current.stamp - frames[start].stamp > MAX_EPOCH_MS or page_jump):
            yield frames[start:index]
            start = index
    if frames:
        yield frames[start:]


def panel_zones(frames):
    """Split the fixed crop into coherent horizontal bands for independent reads."""
    varying = np.zeros(BANDS, dtype=bool)
    for first, second in zip(frames, frames[1:]):
        varying |= changed_tiles(first.signature, second.signature).any(axis=1)
    zones = []
    start = 0
    for band in range(1, BANDS):
        if varying[band] != varying[band - 1]:
            zones.append([start, band, bool(varying[band - 1])])
            start = band
    zones.append([start, BANDS, bool(varying[-1])])
    # A narrow leftover band may contain the top or bottom pixels of a glyph.
    while len(zones) > 1:
        short = next((index for index, (top, end, _) in enumerate(zones) if end - top < MIN_ZONE_BANDS), None)
        if short is None:
            break
        neighbor = short - 1 if short == len(zones) - 1 else short + 1
        left, right = sorted((short, neighbor))
        zones[left:right + 1] = [[zones[left][0], zones[right][1], zones[left][2] or zones[right][2]]]
    return zones


def representative_frame(frames, top, bottom, left, right):
    """Choose the sharpest actually observed raster inside the selected crop."""
    chosen, quality = frames[0], -1.0
    for frame in frames:
        pixels = frame.signature[top:bottom, left:right]
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if sharpness > quality:
            chosen, quality = frame, sharpness
    return chosen, quality


def zone_states(frames, top, bottom):
    """Keep time support only while each measured band agrees with its anchor."""
    anchor = 0
    for index, current in enumerate(frames[1:], start=1):
        changed = changed_tiles(frames[anchor].signature, current.signature)
        if changed[top:bottom].any():
            yield frames[anchor:index]
            anchor = index
    yield frames[anchor:]


def field_columns(before, after, top, bottom):
    """Narrow a corroborated local edit and leave room for the surrounding glyphs."""
    if len(before) < 2 or len(after) < 2:
        return 0.0, 1.0
    changed = changed_tiles(before[0].signature, after[0].signature)
    columns = np.flatnonzero(changed.any(axis=0))
    if len(columns) != 1 or changed[:top].any() or changed[bottom:].any():
        return 0.0, 1.0
    column = int(columns[0])
    margin = .5  # Half a column on either side to retain nearby text.
    return max(0.0, (column - margin) / COLUMNS), min(1.0, (column + 1 + margin) / COLUMNS)


def plan_crops(shot, frames, panel_box, panel_type):
    """Find stable observed spans and pick a bounded-cadence view of moving content."""
    x, y, width, height = panel_box
    for epoch in panel_epochs(frames):
        for top, bottom, moving in panel_zones(epoch):
            fixed, changing = [], {}
            lo, hi = top * SIGNATURE_HEIGHT // BANDS, bottom * SIGNATURE_HEIGHT // BANDS
            previous = ()
            for state in zone_states(epoch, top, bottom):
                left, right = field_columns(previous, state, top, bottom)
                region = (x + width * left, y + height * top / BANDS, width * (right - left),
                          height * (bottom - top) / BANDS)
                representative, sharpness = representative_frame(
                    state, lo, hi, round(left * SIGNATURE_WIDTH), round(right * SIGNATURE_WIDTH))
                previous = state
                stable = len(state) >= 2
                candidate = StateCrop(shot, panel_box, panel_type, region, epoch[0].stamp, representative, state[0].stamp,
                                      state[-1].stamp, len(state), sharpness, "stable" if stable else "changed")
                if stable or not moving:
                    fixed.append(candidate)
                else:
                    bucket = representative.stamp // MOTION_BUCKET_MS
                    prior = changing.get(bucket)
                    if prior is None or prior.sharpness < sharpness:
                        changing[bucket] = candidate
            yield from fixed
            yield from changing.values()


def verified_plans(candidates, layouts):
    """Keep source-frame witnesses inside a UI run supported by existing layout analysis."""
    ordered = sorted(candidates, key=lambda row: row[1])
    ordinals = {(shot, stamp, path): source for shot, stamp, path, _, source in ordered}
    inputs = [(shot, stamp, path, region) for shot, stamp, path, region, _ in ordered]
    for shot, rows in stable_runs(inputs):
        panel = next((panel for stamp, _, region in rows for panel in layouts.get(stamp, ())
                      if panel.get("panelType") in ("mobile-ui", "ui") and iou(panel["box"], region["box"]) >= .7), None)
        if panel is None:
            continue
        box = np.median([region["box"] for _, _, region in rows], axis=0).tolist()
        frames = []
        for stamp, path, region in rows:
            if iou(region["box"], box) < .90:
                continue
            view = view_for(path, box, stamp, content_only=True)
            if view is not None:
                signature = cv2.resize(view.pixels, (SIGNATURE_WIDTH, SIGNATURE_HEIGHT), interpolation=cv2.INTER_AREA)
                frames.append(PanelFrame(stamp, path, ordinals[shot, stamp, path], signature))
        yield from plan_crops(shot, frames, box, panel["panelType"])


def choose_crops(planned, limit):
    """Reserve most reads for changes, using remaining slots for stable comparisons."""
    stable = sorted((crop for crop in planned if crop.kind == "stable"), key=lambda crop: crop.start)
    changed = sorted((crop for crop in planned if crop.kind == "changed"), key=lambda crop: crop.start)
    reserve = min(len(stable), max(1, limit // 4)) if changed and limit else 0
    picked_changed = evenly_spaced(changed, limit - reserve)
    picked_stable = evenly_spaced(stable, limit - len(picked_changed))
    return sorted((*picked_stable, *picked_changed), key=lambda crop: crop.representative.stamp)


def visible_label(line):
    """Collapse whitespace and case without correcting or equating OCR glyphs."""
    return " ".join(line["text"].casefold().split())


def row_matches(previous, present, offset):
    """Pair unique equal lines only at the registered position on both crops."""
    labeled = {}
    for at, line in enumerate(previous):
        x, y, width, height = line["box"]
        label = visible_label(line)
        box = [x, y - offset, width, height]
        labeled.setdefault(label, []).append((at, box))
    candidates = []
    for current in present:
        candidates.append([at for at, box in labeled.get(visible_label(current), ())
                           if iou(box, current["box"]) >= .58])
    demand = {}
    for matches in candidates:
        for at in matches:
            demand[at] = demand.get(at, 0) + 1
    for index, matches in enumerate(candidates):
        if len(matches) == 1 and demand[matches[0]] == 1:
            yield previous[matches[0]], present[index]


def reading_repeats(readings):
    """Record verified repeats without moving the time or geometry of either reading."""
    seen, recent = set(), {}
    count = repeats = confirmed = 0
    for crop, lines in readings:
        count += len(lines)
        labels = set()
        for line in lines:
            label = visible_label(line)
            repeats += label in seen
            seen.add(label)
            labels.add(label)
        key = crop.shot, tuple(crop.panel_box), crop.region, crop.epoch_start
        previous = recent.get(key)
        recent[key] = (crop, lines, labels)
        if previous is None:
            continue
        prior, before, prior_labels = previous
        gap = crop.representative.stamp - prior.representative.stamp
        if not labels & prior_labels or not 0 < gap <= NEAR_READING_MS:
            continue
        _, y, _, height = crop.region
        top = round((y - crop.panel_box[1]) / crop.panel_box[3] * SIGNATURE_HEIGHT)
        bottom = top + round(height / crop.panel_box[3] * SIGNATURE_HEIGHT)
        movement = vertical_delta(prior.representative.signature[top:bottom],
                                  crop.representative.signature[top:bottom])
        if movement is None:
            continue
        shift = movement / SIGNATURE_HEIGHT * crop.panel_box[3]
        for source, target in row_matches(before, lines, shift):
            target["sameContentAs"] = {"recognizedAtMs": source["recognizedAtMs"], "sourceFrame": source["sourceFrame"],
                                       "box": source["box"]}
            confirmed += 1
    return {"ocrLines": count, "distinctLabels": len(seen), "repeatedLabels": repeats,
            "confirmedRepeats": confirmed, "unverifiedRepeats": repeats - confirmed}


def read_interface_state_ocr(folder, candidates, layouts, *, recognize=recognize, max_crops=MAX_CROPS):
    """Read source-frame regions once per measured state and attach exact reading provenance."""
    started = perf_counter()
    planned = list(verified_plans(candidates, layouts))
    chosen = choose_crops(planned, min(MAX_CROPS, max(0, max_crops)))
    jobs, witnesses = [], []
    path, image = None, None
    for crop in chosen:
        frame = crop.representative
        if frame.path != path:
            path = frame.path
            image = cv2.imread(str(path))
        target = folder / f"state-ocr-{len(jobs):03}.jpg"
        if image is not None and (job := crop_job(image, target, crop.region, len(jobs))):
            witnesses.append((crop, job[2]))
            jobs.append(job)
    scope = ACTIVE_SCENE_OCR.get()
    before = scope.summary() if scope is not None else {}
    readings, statuses = recognize(jobs, line_limit=48, keep_numbers=True) if jobs else ({}, {})
    observations, recognized = [], []
    for key, (crop, bounds) in enumerate(witnesses):
        frame = crop.representative
        elements = [{**row, "role": "text_line", "basis": "stateful-crop", "sourceFrame": frame.source_frame,
                     "recognizedAtMs": frame.stamp, "verifiedFromMs": crop.start, "verifiedUntilMs": crop.last + 1}
                    for row in readings.get(key, ())]
        if elements:
            recognized.append((crop, elements))
            observations.append({"shot": crop.shot, "timeMs": frame.stamp, "sourceFrame": frame.source_frame,
                                 "panelBox": crop.panel_box, "panelType": crop.panel_type, "box": list(bounds),
                                 "kind": crop.kind, "startMs": crop.start, "endMs": crop.last + 1,
                                 "lastObservedMs": crop.last, "supportCount": crop.count, "elements": elements})
    report = {"status": "ok" if jobs else "no-candidates", "planned": len(planned), "ocrCrops": len(jobs),
              "stableCrops": sum(crop.kind == "stable" for crop, _ in witnesses),
              "changedCrops": sum(crop.kind == "changed" for crop, _ in witnesses), "recognized": len(observations),
              "ocrInputPixels": sum(job[3] * job[4] for job in jobs), **reading_repeats(recognized),
              "ocrStatuses": statuses, "elapsedMs": round((perf_counter() - started) * 1000)}
    if scope is not None:
        after = scope.summary()
        report.update({key: after[key] - before[key] for key in ("dispatched", "reused", "dropped", "engineCalls")})
    return observations, report
