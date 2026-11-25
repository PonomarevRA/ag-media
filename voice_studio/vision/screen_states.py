"""Find stable screen changes inside a measured viewport and read each representative once."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from voice_studio.vision.scene_ocr import recognize
from voice_studio.vision.scene_objects import iou
from voice_studio.vision.screen_scroll import stable_runs, vertical_delta, view_for

OCR_BATCH_SIZE = 24
MIN_STABLE_MS = 120


@dataclass
class ScreenState:
    """A representative capture and the last bounded time of the observed state."""

    start: int
    end: int
    path: Path


def changed_screen(first, second):
    """Require distributed pixel changes so a cursor or a single animated control cannot split a screen."""
    delta = cv2.absdiff(first, second).astype(np.float32).mean(axis=2)
    blocks = delta.reshape(4, 24, 4, 16).mean(axis=(1, 3))
    return float(delta.mean()) >= 4.0 and int(np.count_nonzero(blocks >= 5.5)) >= 4


def screen_runs(rows, bounds):
    """Confirm a changed page with a second stable capture and reject registration-supported scrolling."""
    states = []
    previous = pending = active = active_signature = None
    for stamp, path, _ in rows:
        view = view_for(path, bounds, stamp, content_only=True)
        if view is None:
            continue
        # The center limits edge jitter and chroma differences catch dark/light switches.
        pixels = view.pixels
        h, w = pixels.shape[:2]
        interior = pixels[round(.06 * h):round(.94 * h), round(.08 * w):round(.92 * w)]
        signature = cv2.resize(interior, (64, 96), interpolation=cv2.INTER_AREA)
        if active is None:
            active = ScreenState(stamp, stamp + 1, path)
            states.append(active)
            active_signature = signature
        elif changed_screen(previous[1], signature):
            delta = vertical_delta(previous[2], interior)
            if delta:
                active_signature = signature
            pending = (stamp, path, signature) if not delta and changed_screen(active_signature, signature) else None
        elif pending and stamp - pending[0] >= MIN_STABLE_MS and not changed_screen(pending[2], signature):
            start, representative, _ = pending
            active.end = start
            active = ScreenState(start, stamp + 1, representative)
            states.append(active)
            active_signature = signature
            pending = None
        active.end = max(active.end, stamp + 1)
        previous = (stamp, signature, interior)
    return states


def read_screen_states(folder, candidates, layouts):
    """Run isolated OCR on observed states of verified UI panels and attach it on the source clock."""
    from voice_studio.vision.screen_analysis import crop_job
    from voice_studio.vision.screen_ui import analyze_interface

    confirmed = defaultdict(list)
    for stamp, regions in layouts.items():
        for panel in regions:
            if panel.get("panelType") in ("mobile-ui", "ui"):
                confirmed[stamp].append(panel)

    states = []
    for _, rows in stable_runs(sorted(candidates, key=lambda row: row[1])):
        supporting = [panel for stamp, _, region in rows for panel in confirmed.get(stamp, ())
                      if iou(panel["box"], region["box"]) >= .7]
        if not supporting:
            continue
        panel_type = supporting[0]["panelType"]
        box = np.median([region["box"] for _, _, region in rows], axis=0).tolist()
        for state in screen_runs(rows, box):
            states.append((state, box, panel_type))
    states.sort(key=lambda row: row[0].start)
    jobs, selected = [], []
    for key, (state, box, panel_type) in enumerate(states):
        image = cv2.imread(str(state.path))
        if image is None:
            continue
        job = crop_job(image, folder / f"screen-state-{key}.jpg", box, key)
        if job is not None:
            jobs.append(job)
            selected.append((key, state, box, panel_type))
    readings, statuses = {}, {}
    for index in range(0, len(jobs), OCR_BATCH_SIZE):
        batch, batch_statuses = recognize(jobs[index:index + OCR_BATCH_SIZE], line_limit=48, keep_numbers=True)
        readings.update(batch)
        for engine, status in batch_statuses.items():
            if engine not in statuses or status != "ok":
                statuses[engine] = status
    for key, state, box, panel_type in selected:
        image = cv2.imread(str(state.path))
        if image is None:
            continue
        region = {"box": box, "basis": "frame-layout", "panelType": panel_type, "knownPanel": True}
        lines = readings.get(key, ())
        layout = analyze_interface(image, region, lines)
        if layout is None:
            layout = {**region, "elements": [{**line, "role": "text_line", "basis": "ocr"} for line in lines],
                      "connections": []}
        else:
            elements = layout["elements"]
            for line in lines:
                if not any(element.get("text") == line["text"] and iou(element["box"], line["box"]) > .4
                           for element in elements):
                    elements.append({**line, "role": "text_line", "basis": "ocr"})
        layout["screenStateEndMs"] = state.end
        layout["screenStateElements"] = layout["elements"][:48]
        rows = layouts.setdefault(state.start, [])
        match = next((index for index, prior in enumerate(rows) if iou(prior["box"], box) >= .7), None)
        if match is not None:
            rows[match] = {**rows[match], "screenStateEndMs": state.end,
                           "screenStateElements": layout["screenStateElements"]}
        else:
            rows.insert(0, layout)
    return len(jobs), statuses
