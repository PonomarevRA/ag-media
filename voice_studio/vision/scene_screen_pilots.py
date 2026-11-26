"""Attach optional video UI evidence to exact source-frame observations."""

from bisect import bisect_left
from contextlib import nullcontext
import logging
import subprocess
from time import perf_counter

import cv2

from voice_studio.vision.scene_objects import iou
from voice_studio.vision.scene_sampling import evenly_spaced
from voice_studio.mods.errors import ModError

MAX_CONTROL_FRAMES = 16
MAX_STATE_GAP_FRAMES = 96


def attach_state_reads(interfaces, observations):
    """Keep provenance on the measured frame, including coincident older reads."""
    for row in observations:
        if not row["elements"]:
            continue
        layouts = interfaces.setdefault(row["timeMs"], [])
        box = row["panelBox"]
        panel = next((item for item in layouts if item.get("panelType") in ("ui", "mobile-ui")
                      and item.get("basis") == "frame-layout" and iou(item["box"], box) >= .97), None)
        if panel is None:
            panel = {"box": box, "basis": "frame-layout", "panelType": row["panelType"],
                     "connections": [], "elements": []}
            layouts.insert(0, panel)
        for element in row["elements"]:
            label = element["text"].casefold().strip()
            evidence = {key: element[key] for key in
                        ("recognizedAtMs", "verifiedFromMs", "verifiedUntilMs", "sourceFrame")}
            evidence.update({"cropBox": row["box"], "kind": row["kind"], "supportCount": row["supportCount"]})
            if "sameContentAs" in element:
                evidence["sameContentAs"] = element["sameContentAs"]
            if previous := next((old for old in panel["elements"]
                                 if old.get("text", "").casefold().strip() == label
                                 and iou(old["box"], element["box"]) >= .55), None):
                # Keep the old position for connectors and retain the exact confirming crop.
                previous["stateEvidence"] = evidence
            else:
                panel["elements"].append({**element, "stateEvidence": evidence})


def state_ocr_candidates(samples, supplemental, candidates):
    """Use one measured path for each proxy frame, retaining native source ordinals."""
    source_frames = {sample["timeMs"]: sample["sourceFrame"] for sample in samples}
    rows = {(row[0], row[1]): row for row in supplemental}
    for shot, stamp, path, panel in candidates:
        if stamp in source_frames:
            rows[shot, stamp] = (shot, stamp, path, panel, source_frames[stamp])
    return [row for _, row in sorted(rows.items())]


def changing_panel_gaps(candidates, interfaces):
    """Find missed samples between two measured, visibly changing panel views."""
    from voice_studio.vision.screen_scroll import view_for

    gaps = []
    for first, second in zip(candidates, candidates[1:]):
        shot, start, before, left, _ = first
        next_shot, end, after, right, _ = second
        box = right["box"]
        if shot != next_shot or not 180 < end - start <= 1400 or iou(left["box"], box) < .92:
            continue
        verified = any(layout.get("panelType") in ("mobile-ui", "ui") and iou(layout["box"], box) >= .9
                       for stamp in (start, end) for layout in interfaces.get(stamp, ()))
        if not verified:
            continue
        a, b = view_for(before, box, start, content_only=True), view_for(after, box, end, content_only=True)
        if a is None or b is None:
            continue
        first_pixels, second_pixels = (cv2.resize(view.pixels, (64, 96), interpolation=cv2.INTER_AREA)
                                       for view in (a, b))
        delta = cv2.absdiff(first_pixels, second_pixels)
        changed = 1 - cv2.mean(cv2.inRange(delta, (0, 0, 0), (21, 21, 21)))[0] / 255
        if .03 <= changed <= .75:
            gaps.append((shot, start, end, right))
    return gaps


def sample_panel_gaps(folder, request, candidates, interfaces, decode_samples):
    """Decode a bounded set of same-shot holes in existing panel measurements."""
    from voice_studio.vision.frame_change_panels import matching_panel
    from voice_studio.vision.screen_analysis import side_panel

    clock = request.get("scrollClock")
    if not clock or not request.get("scrollInputs"):
        return []
    stamps = [stamp for _, stamp, _ in clock]
    selected = {}
    for shot, start, end, hint in changing_panel_gaps(candidates, interfaces):
        left, right = bisect_left(stamps, start + 1), bisect_left(stamps, end)
        for current_shot, stamp, ordinal in clock[left:right]:
            if shot == current_shot:
                selected[ordinal] = (shot, stamp, ordinal, hint)
    frames = evenly_spaced([selected[ordinal] for ordinal in sorted(selected)], MAX_STATE_GAP_FRAMES)
    if not frames:
        return []
    target = folder / "screen-state-gap"
    target.mkdir()
    plan = [{"sourceFrame": ordinal} for _, _, ordinal, _ in frames]
    try:
        decode_samples(request["scrollInputs"], target, plan, timeout=30)
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    additions = []
    for index, (shot, stamp, ordinal, hint) in enumerate(frames):
        path = target / f"{index:04}.jpg"
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None:
            continue
        panel = matching_panel(image, hint) if hint.get("changeAssisted") else None
        panel = panel or side_panel(image)
        if panel and iou(panel["box"], hint["box"]) >= .9:
            additions.append((shot, stamp, path, panel, ordinal))
    return additions


def analyze_screen_pilots(folder, request, interfaces, supplemental, panels, decode_samples, *, ocr=None):
    """Finish independent, separately selectable screen passes without affecting baseline OCR."""
    features = request["pilotFeatures"]
    candidates = state_ocr_candidates(request["samples"], supplemental, panels)
    reports = {}
    extras = []
    if features.get("interfaceStateOcr"):
        from voice_studio.vision.screen_state_ocr import MAX_CROPS, read_interface_state_ocr

        try:
            with ocr.targeted() if ocr is not None else nullcontext():
                limit = ocr.available(MAX_CROPS) if ocr is not None else MAX_CROPS
                if limit:
                    extras = sample_panel_gaps(folder, request, candidates, interfaces, decode_samples)
                    rows, stats = read_interface_state_ocr(folder, [*candidates, *extras], interfaces,
                                                           max_crops=limit)
                    attach_state_reads(interfaces, rows)
                    stats["gapFrames"] = len(extras)
                else:
                    stats = {"status": "crop-budget", "ocrCrops": 0}
        except (OSError, ValueError, RuntimeError, cv2.error, ModError, subprocess.SubprocessError):
            logging.getLogger(__name__).exception("Interface state OCR failed")
            stats = {"status": "failed", "ocrCrops": 0}
        reports["interfaceStateOcr"] = stats
    if features.get("smallUiControls"):
        source_frames = {sample["timeMs"]: sample["sourceFrame"] for sample in request["samples"]}
        source_frames.update({stamp: ordinal for _, stamp, _, _, ordinal in (*supplemental, *extras)})
        reports["smallUiControls"] = measure_small_controls(
            folder, request, interfaces, source_frames, decode_samples,
        )
    return reports, extras


def measure_small_controls(folder, request, interfaces, source_frames, decode_samples):
    """Read a time-distributed set of verified screen states at original resolution."""
    from voice_studio.vision.screen_small_controls import enrich_small_controls

    started = perf_counter()
    candidates = []
    for stamp, layouts in sorted(interfaces.items()):
        if stamp not in source_frames:
            continue
        panels = [layout for layout in layouts if layout.get("panelType") in ("ui", "mobile-ui")]
        if panels:
            candidates.append((stamp, panels))
    native = folder / "screen-small-controls"
    if not candidates or not request.get("scrollInputs"):
        return {"status": "no-candidates", "frames": 0, "added": 0}
    native.mkdir()
    states = [row for row in candidates if any("previousStateMs" in panel for panel in row[1])]
    selected = evenly_spaced(states, MAX_CONTROL_FRAMES // 2)
    used = {stamp for stamp, _ in selected}
    other = [row for row in candidates if row[0] not in used]
    selected.extend(evenly_spaced(other, MAX_CONTROL_FRAMES - len(selected)))
    selected.sort(key=lambda row: source_frames[row[0]])
    plan = [{"sourceFrame": source_frames[stamp]} for stamp, _ in selected]
    try:
        decode_samples(request["scrollInputs"], native, plan, native=True, timeout=30)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "failed", "frames": 0, "added": 0}
    added = checked = failed = 0
    for index, (_, panels) in enumerate(selected):
        path = native / f"{index:04}.png"
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None:
            failed += 1
            continue
        checked += 1
        for panel in panels[:2]:
            try:
                added += enrich_small_controls(image, panel)
            except cv2.error:
                failed += 1
    return {"status": "partial" if failed else "ok", "frames": checked, "added": added,
            "elapsedMs": round(1000 * (perf_counter() - started))}
