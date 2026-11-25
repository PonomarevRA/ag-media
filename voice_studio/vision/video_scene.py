"""Plan local scene OCR and object recognition without re-running face detection."""

from voice_studio.core.analysis_stages import ignore_progress, queued_stage_slot
from voice_studio.core.process_usage import process_usage
from voice_studio.mod_runtime import current_snapshot, model_provider, is_replacement
from voice_studio.mods.errors import ModError

from bisect import bisect_left
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

from voice_studio.synthesis.engines import ROOT, ffmpeg
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING, evenly_spaced, label_visible_faces, sample_plan
from voice_studio.vision.video_colors import hex_rgb, mean_rgb
from voice_studio.vision.video_revolver import VISION, VISUAL_SLOT, probe_video, run_worker

from voice_paths import model_path, temporary_directory
MAX_OCR_REGIONS = 128
MAX_SCROLL_FRAMES = 320
SCROLL_SAMPLE_MS = 65
PANEL_SAMPLE_MS = 400
SCENE_CUT_THRESHOLD = 0.30
LOW_CONTRAST_CUT = 0.16
CLOCK_FILTER = ("scale=112:112:force_original_aspect_ratio=decrease,"
                "select='gte(scene,0)',metadata=mode=print:key=lavfi.scene_score:file=frames.log")
SCENE_FRAME = re.compile(r"^frame:(\d+)\s+pts:\S+\s+pts_time:(\S+)")


@model_provider('vision.scene-batch', 'builtin.scene-batch')
def review_scene(source, work, formats, evidence=None, *, object_sampling=DEFAULT_OBJECT_SAMPLING, window=None, colors_only=False,
                 publish_mosaic=None, emit=ignore_progress):
    """Analyze the source or a graph partition, with a thumbnail-only path for shot colors."""
    source = Path(source).resolve()
    if probe_video(source, formats) is False:
        return {"status": "no-video"}
    try:
        with queued_stage_slot(VISUAL_SLOT, emit, "scene", already_started=True), temporary_directory() as folder:
            started = perf_counter()
            root = Path(folder)
            evidence = evidence or {}
            inputs = ["-protocol_whitelist", "file,pipe", "-format_whitelist", formats]
            if window is not None:
                start, end = window
                inputs.extend(["-ss", f"{start / 1000:.3f}", "-t", f"{(end - start) / 1000:.3f}"])
            inputs.extend(["-i", str(source)])
            frames, native_cuts, changes = scan_scene(inputs, root, colors_only=colors_only)
            supplied_cuts = evidence.get("editorTimeline", {}).get("shots", [])
            cuts = merge_shot_cuts(frames, native_cuts, supplied_cuts)
            plan = sample_plan(frames, cuts, object_sampling)
            if changes and changes["status"] == "ok":
                from voice_studio.vision.frame_changes import augment_plan, constrain_graphics
                cuts_by_ordinal = {frames[index][0] for index in cuts if 0 <= index < len(frames)}
                kept = [row for row in changes["rows"] if row["sourceFrame"] not in cuts_by_ordinal]
                changes["cutCandidatesRemoved"] = len(changes["rows"]) - len(kept)
                changes["rows"] = kept
                boundaries_ms = sorted(frames[index][1] for index in cuts if 0 < index < len(frames))
                changes["graphicRows"] = list(constrain_graphics(changes.get("graphicRows", ()), boundaries_ms))
                augment_plan(plan, frames, cuts, changes)
                changes.pop("objectRows", None)
            if not plan:
                return {"status": "no-video"}
            label_visible_faces(plan, evidence.get("timeline") or {})
            decode_samples(inputs, root, plan, thumbnails=colors_only)
            decoded = perf_counter()
            request = {"sampling": "shots", "objectSampling": object_sampling, "samples": plan,
                       "colorsOnly": colors_only}
            if not colors_only:
                request["includeMosaicFiles"] = publish_mosaic is not None
                request["scrollInputs"] = inputs
                request["scrollClock"] = scroll_clock(frames, cuts)
            (root / "samples.json").write_text(json.dumps(request), encoding="utf-8")
            result = run_worker("voice_studio.vision.video_scene", [root], [VISION], timeout=max(420, len(plan)))
            for index, page in enumerate(result.get("interfaceScrolls", [])):
                filename = page.pop("_mosaicFile", None)
                if publish_mosaic and isinstance(filename, str) and re.fullmatch(r"screen-scroll-[0-9]+\.png", filename):
                    url = publish_mosaic(index, root / filename, mosaic_markdown(page))
                    if url:
                        page["previewUrl"] = url
            if changes:
                if measured := result.pop("frameChangeRecognition", None):
                    from voice_studio.vision.frame_change_recognition import attach_change_recognition
                    attach_change_recognition(changes, measured)
                result["frameChanges"] = changes
                if changes["status"] != "ok":
                    result.setdefault("stages", {})["frame-changes"] = changes["status"]
                objects_status = (result.get("frameChangeObjects") or {}).get("status", "ok")
                if objects_status != "ok":
                    result.setdefault("stages", {})["frame-change-objects"] = objects_status
            result["shotStartsMs"] = [frames[index][1] for index in sorted(cuts) if 0 < index < len(frames)]
            starts = [0, *result["shotStartsMs"]]
            for shot in result.get("shotColors", []):
                shot["startMs"] = starts[shot.pop("shot")]
            result["decodeMs"] = round((decoded - started) * 1000)
            result["elapsedMs"] = round((perf_counter() - started) * 1000)
            return result
    except (BrokenPipeError, ConnectionResetError):
        raise
    except (OSError, ValueError, subprocess.SubprocessError):
        logging.getLogger(__name__).exception("Scene analysis failed")
        return {"status": "failed", "reason": "Не удалось обработать сцену. Пересчитайте «Надписи, предметы и камера»."}


def mosaic_markdown(page):
    """Format a screenshot page from saved OCR without running its vision models again."""
    from voice_studio.vision.screen_markdown import scroll_markdown

    return scroll_markdown([page]) if "mosaicText" in page else None


def scan_scene(inputs, folder, *, colors_only):
    """Use the optional algorithm in the early pass and keep baseline usable on failure."""
    from voice_studio.mods import ModError
    from voice_studio.vision.frame_changes import BASIS, CAPABILITY, MOD_ID

    changes = None
    if not colors_only:
        snapshot = current_snapshot()
        record = snapshot.to_wire()["mods"].get(MOD_ID, {})
        if record.get("enabled"):
            try:
                changes = snapshot.invoke_mod(MOD_ID, CAPABILITY, {
                    "inputs": inputs, "folder": folder, "clockFilter": CLOCK_FILTER,
                })
                frames, cuts = read_frame_clock(folder)
                return frames, cuts, changes
            except (ModError, OSError, ValueError, subprocess.SubprocessError):
                changes = {"status": "failed", "basis": BASIS}
    frames, cuts = scan_frames(inputs, folder)
    return frames, cuts, changes


def scan_frames(inputs, folder):
    """Read native timestamps and hard cuts from a low-resolution decode of the selected input."""
    ffmpeg([*inputs, "-map", "0:V:0", "-an", "-sn", "-filter_threads", "2", "-vf", CLOCK_FILTER,
            "-f", "null", "-"], cwd=folder, timeout=900)
    return read_frame_clock(folder)


def read_frame_clock(folder):
    """Read the same low-resolution cut scores for either scan path."""
    frames, scores = [], []
    with (folder / "frames.log").open() as lines:
        for line in lines:
            match = SCENE_FRAME.match(line)
            if match:
                frame, pts = match.groups()
                frames.append((int(frame), round(float(pts) * 1000)))
                scores.append(0)
            elif frames and line.startswith("lavfi.scene_score="):
                scores[-1] = float(line.partition("=")[2])
    return frames, scene_cuts(scores)


def scene_cuts(scores):
    """Recover isolated hard transitions whose low contrast hides them from the main threshold."""
    cuts = []
    for index, score in enumerate(scores):
        if score > SCENE_CUT_THRESHOLD:
            cuts.append(index)
        elif index and score >= LOW_CONTRAST_CUT:
            surrounding = scores[max(0, index - 3):index] + scores[index + 1:index + 4]
            if surrounding and score > 3 * max(surrounding):
                cuts.append(index)
    return cuts


def merge_shot_cuts(frames, native_cuts, supplied_cuts):
    """Keep native cuts precise and use the shared visual timeline to recover additional cuts."""
    stamps = [stamp for _, stamp in frames]
    native_stamps = sorted(stamps[index] for index in native_cuts)
    cuts = set(native_cuts)
    for row in supplied_cuts:
        stamp = row["startMs"]
        closest = bisect_left(native_stamps, stamp - 80)
        if closest < len(native_stamps) and native_stamps[closest] <= stamp + 80:
            continue
        cuts.add(bisect_left(stamps, stamp))
    return cuts


def frame_selection(plan):
    """Encode adjacent arithmetic runs compactly for FFmpeg's source-frame selector."""
    indices = [row["sourceFrame"] for row in plan]
    terms = []
    start = 0
    while start < len(indices):
        first = indices[start]
        if start + 1 == len(indices):
            terms.append(f"eq(n\\,{first})")
            break
        stride = indices[start + 1] - first
        end = start + 1
        while end + 1 < len(indices) and indices[end + 1] - indices[end] == stride:
            end += 1
        band = f"between(n\\,{first}\\,{indices[end]})"
        if stride > 1:
            band += f"*not(mod(n-{first}\\,{stride}))"
        terms.append(band)
        start = end + 1
    return "+".join(terms)


def decode_samples(inputs, folder, plan, *, thumbnails=False, native=False, timeout=900):
    """Extract only the planned frames, retaining detail for small background text."""
    expression = frame_selection(plan)
    limit = 48 if thumbnails else 1920
    filters = f"select='{expression}'"
    if not native:
        filters += f",scale='min({limit},iw)':'min({limit},ih)':force_original_aspect_ratio=decrease"
    filter_file = folder / "frames.filter"
    filter_file.write_text(filters, encoding="utf-8")
    encoding = ["-pix_fmt", "rgb24"] if native else ["-q:v", "3"]
    output = "%04d.png" if native else "%04d.jpg"
    ffmpeg([*inputs, "-map", "0:V:0", "-an", "-sn", "-filter_threads", "2", "-filter_script:v", str(filter_file),
            "-fps_mode", "passthrough", "-start_number", "0", *encoding, "-threads:v", "2",
            "-frames:v", str(len(plan)), str(folder / output)], timeout=timeout)


def scroll_clock(frames, cuts):
    """Keep native frame ordinals for a potential high-rate UI pass within each shot."""
    clock = []
    shot, previous = -1, float("-inf")
    for index, (source_frame, stamp) in enumerate(frames):
        if index == 0 or index in cuts:
            shot += 1
            previous = float("-inf")
        if stamp - previous >= SCROLL_SAMPLE_MS:
            clock.append((shot, stamp, source_frame))
            previous = stamp
    return clock


def dense_scroll_samples(folder, request, candidates):
    """Decode only a capped subset of panel-motion intervals; OCR is deferred to the mosaic."""
    import cv2
    from voice_studio.vision.frame_change_panels import matching_panel
    from voice_studio.vision.scene_objects import iou
    from voice_studio.vision.screen_analysis import side_panel
    from voice_studio.vision.screen_scroll import scrolling_intervals, stable_runs

    if not candidates or not request.get("scrollClock"):
        return []
    ranges = scrolling_intervals(candidates, padding_ms=SCROLL_SAMPLE_MS)
    stable = {}
    for shot, rows in stable_runs(candidates):
        stable.setdefault(shot, []).append((rows[0][0], rows[-1][0]))
    current = {candidate[1] for candidate in candidates}
    hints = {}
    for shot, stamp, _, region in candidates:
        if region.get("changeAssisted") or region.get("knownPanel"):
            hints.setdefault(shot, []).append((stamp, region))
    moving, still, last_still = [], [], {}
    for shot, stamp, source_frame in request["scrollClock"]:
        if stamp in current:
            continue
        entry = {"sourceFrame": source_frame, "timeMs": stamp, "shot": shot}
        if any(start <= stamp <= end for start, end in ranges.get(shot, ())):
            moving.append(entry)
        elif (stamp - last_still.get(shot, -PANEL_SAMPLE_MS) >= PANEL_SAMPLE_MS
              and any(start <= stamp <= end for start, end in stable.get(shot, ()))):
            still.append(entry)
            last_still[shot] = stamp
    # Preserve temporal coverage on stable panels while assigning most of the budget to scroll.
    motion_limit = MAX_SCROLL_FRAMES - min(len(still), MAX_SCROLL_FRAMES // 4)
    selected = evenly_spaced(moving, motion_limit)
    selected.extend(evenly_spaced(still, MAX_SCROLL_FRAMES - len(selected)))
    if not selected:
        return []
    selected.sort(key=lambda row: row["sourceFrame"])
    target = folder / "screen-dense"
    target.mkdir()
    try:
        decode_samples(request["scrollInputs"], target, selected, timeout=35)
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    samples = []
    for index, sample in enumerate(selected):
        path = target / f"{index:04}.jpg"
        if not path.is_file():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        nearby = hints.get(sample["shot"], ())
        stamp = sample["timeMs"]
        index = bisect_left(nearby, stamp, key=lambda item: item[0])
        neighbors = nearby[max(0, index - 1):index + 1]
        hint = min(neighbors, key=lambda item: abs(item[0] - stamp), default=None)
        region = None
        if hint and abs(hint[0] - stamp) <= 1200:
            if hint[1].get("changeAssisted"):
                region = matching_panel(image, hint[1])
            elif (len(neighbors) == 2 and neighbors[0][0] <= stamp <= neighbors[1][0]
                  and all(row.get("knownPanel") and iou(row["box"], hint[1]["box"]) >= .9 for _, row in neighbors)):
                region = hint[1]
        if region := region or side_panel(image):
            samples.append((sample["shot"], sample["timeMs"], path, region, sample["sourceFrame"]))
    return samples


def detail_screen_states(folder, request, screens, interfaces):
    """Read changed large regions from native frames using the screenshot analyzer."""
    import cv2
    from voice_studio.vision.scene_objects import iou
    from voice_studio.vision.screen_analysis import screenshot_layout

    started = perf_counter()
    plan = screens.detail_states.plan()
    if not plan or not request.get("scrollInputs"):
        return {"status": "no-candidates", "planned": 0, "analyzed": 0}
    native = folder / "screen-native"
    native.mkdir()
    # A single decode selects source ordinals; avoid one slow seek per region.
    frames = list({row["sourceFrame"]: row for row in plan}.values())
    try:
        decode_samples(request["scrollInputs"], native, frames, native=True, timeout=35)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "partial", "planned": len(plan), "analyzed": 0}
    paths = {row["sourceFrame"]: native / f"{index:04}.png" for index, row in enumerate(frames)}
    analyzed = 0
    failures = 0
    ocr_jobs = 0
    for index, state in enumerate(plan):
        target = native / f"ocr-{index}"
        target.mkdir()
        path = paths[state["sourceFrame"]]
        try:
            reading = screenshot_layout(path, region=state["box"], work_dir=target)
        except (cv2.error, OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            failures += 1
            continue
        ocr_jobs += reading["ocrJobs"]
        layout = reading.get("interface") or {"box": state["box"], "elements": [], "connections": []}
        if not reading["ocr"] and not layout["elements"]:
            continue
        elements = list(layout["elements"])
        for item in reading.get("controlGrid", {}).get("controls", ()):
            if all(iou(item["box"], present["box"]) < .5 for present in elements):
                elements.append(item)
        detailed = {**layout, "elements": elements, "basis": "frame-layout",
                    "regionKind": state["regionKind"], "lastObservedMs": state["lastObservedMs"],
                    "cropAvailable": True, "detailBasis": "original-resolution"}
        if state["regionKind"] == "panel":
            detailed["panelType"] = state.get("panelType", "mobile-ui")
        if "previousStateMs" in state:
            detailed["previousStateMs"] = state["previousStateMs"]
        for line in reading["ocr"]:
            if all(iou(line["box"], item["box"]) < .55 for item in elements):
                elements.append({**line, "role": "screen_text", "basis": "ocr"})
        rows = interfaces.setdefault(state["timeMs"], [])
        remaining = []
        for item in rows:
            if iou(item["box"], state["box"]) < .65:
                remaining.append(item)
                continue
            for row in item["elements"]:
                if row.get("basis") == "scroll-mosaic" and all(iou(row["box"], part["box"]) < .55 for part in elements):
                    elements.append(row)
        rows[:] = [detailed, *remaining]
        analyzed += 1
    return {"status": "partial" if failures else "ok", "planned": len(plan),
            "analyzed": analyzed, "ocrJobs": ocr_jobs, "elapsedMs": round(1000 * (perf_counter() - started))}


def ocr_jobs(image, path, sample, history):
    """Schedule OCR on changed sectors and reserve full-frame reads for shot anchors."""
    import cv2
    from voice_studio.vision.scene_ocr import BOTTOM, CENTER, FULL, pixel_view

    stamp = sample["timeMs"]
    faces = sample["faces"]
    close_face = any(w * h >= .035 for _, _, w, h in faces)
    phase = sample["phase"]
    if close_face:
        regions = (FULL, BOTTOM) if sample["anchor"] else (BOTTOM,)
    elif phase == "middle":
        regions = (FULL, CENTER)
    elif phase:
        regions = (FULL,)
    else:
        return []
    jobs = []
    for number, region in enumerate(regions):
        crop, bounds = pixel_view(image, region)
        signature = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (24, 24), interpolation=cv2.INTER_AREA)
        key = sample["shot"], region
        previous = history.get(key)
        if previous is not None:
            prior_stamp, prior = previous
            # Throttle moving scenes and skip static sectors for the rest of the shot.
            cooldown = 1700 if close_face and region == BOTTOM else 350
            if stamp - prior_stamp < cooldown or cv2.mean(cv2.absdiff(signature, prior))[0] < 5:
                continue
        if region == FULL:
            target = path
        else:
            target = path.with_name(f"ocr-{path.stem}-{number}.jpg")
            if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                continue
        height, width = crop.shape[:2]
        history[key] = stamp, signature
        jobs.append((stamp, target, bounds, width, height))
    return jobs


def add_shot_color(colors, sample, image, cv2):
    """Collect equally spaced frame means without letting dense object mode change the weighting."""
    if sample["ocr"]:
        blue, green, red, _ = cv2.mean(image)
        colors.setdefault(sample["shot"], []).append((red, green, blue))


def summarize_shot_colors(colors):
    """Keep the mean and the size of the sample behind it for each measured shot."""
    return [{"shot": shot, "average": hex_rgb(mean_rgb(samples)), "sampleCount": len(samples)}
            for shot, samples in colors.items()]


def read_shot_colors(folder, samples, cv2):
    """Average tiny decoded thumbnails without loading object or text recognition."""
    colors = {}
    for index, sample in enumerate(samples):
        image = cv2.imread(str(folder / f"{index:04}.jpg"))
        if image is not None:
            add_shot_color(colors, sample, image, cv2)
    return {"status": "ok", "shotColors": summarize_shot_colors(colors)}


def analyze_frames(folder):
    """Execute bounded tasks and keep all results tied to their sampled source time."""
    import cv2
    cv2.setNumThreads(1)
    request = json.loads((folder / "samples.json").read_bytes())
    if request.get("colorsOnly"):
        return read_shot_colors(folder, request["samples"], cv2)
    from voice_studio.vision.camera_motion import CameraMotion
    from voice_studio.vision.frame_change_objects import review_change_objects
    from voice_studio.vision.frame_change_recognition import recognize_change_regions
    from voice_studio.vision.object_appearance import ObjectAppearance, retain_confirmed_matches
    from voice_studio.vision.scene_objects import ObjectDetector, calibrate_tiles, confirm_objects, refine_drinkware
    from voice_studio.vision.scene_ocr import recognize
    from voice_studio.vision.screen_analysis import ScreenAnalysis, merge_scroll_layouts, merge_readings
    from voice_studio.vision.screen_states import read_screen_states
    from voice_studio.vision.small_object_pass import SmallObjectPass
    from voice_studio.vision.ui_activity import UIActivity

    weights = model_path("vision", "nanodet.onnx")
    try:
        detector = ObjectDetector(weights, reuse_views=True) if weights.is_file() or is_replacement('vision.objects', 'builtin.nanodet') else None
    except cv2.error:
        detector = None
    stages = {"nanodet": "ok" if detector else "unavailable"}
    frames, jobs, history = [], [], {}
    checks_by_shot = {}
    colors = {}
    active_shot = None
    object_seconds = appearance_seconds = 0
    tiled_shots = set()
    if detector is not None and request["objectSampling"] == "shot-tiles":
        calibration_started = perf_counter()
        try:
            tiled_shots = calibrate_tiles(detector, request["samples"], folder)
        except (cv2.error, KeyError, ValueError):
            stages["nanodet"] = "partial"
        object_seconds += perf_counter() - calibration_started
    small_objects = SmallObjectPass() if detector and request["objectSampling"] == "small-detail" else None
    appearance = ObjectAppearance()
    camera = CameraMotion()
    screens = ScreenAnalysis()
    ui_activity = UIActivity()
    frame_paths = {}
    motion_seconds = 0
    for index, sample in enumerate(request["samples"]):
        shot = sample["shot"]
        checks = checks_by_shot.setdefault(shot, {"shot": shot, "planned": 0, "checked": 0})
        if sample["objects"]:
            checks["planned"] += 1
        path = folder / f"{index:04}.jpg"
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None:
            if sample["objects"] and detector:
                stages["nanodet"] = "partial"
            continue
        if not sample.get("changeOnly"):
            add_shot_color(colors, sample, image, cv2)
            ui_activity.add(image, sample)
            motion_started = perf_counter()
            camera.add(image, sample)
            motion_seconds += perf_counter() - motion_started
        frame = {"timeMs": sample["timeMs"], "sourceFrame": sample["sourceFrame"],
                 "shot": shot, "text": [], "objects": [], "objectChecked": False}
        if detector and sample["objects"]:
            if shot != active_shot:
                detector.clear_view_cache()
                active_shot = shot
            object_started = perf_counter()
            try:
                if small_objects is not None and sample["sectors"]:
                    frame["objects"] = small_objects.observe_anchor(detector, image, shot)
                else:
                    frame["objects"] = detector.detect(image, sectors=sample["sectors"] or shot in tiled_shots)
                refined, correction = refine_drinkware(detector, image, frame["objects"])
                frame["objects"] = refined
                if correction:
                    frame["objectReview"] = correction
                frame["objectChecked"] = True
                checks["checked"] += 1
                if small_objects is not None:
                    small_objects.add_sample(path, frame)
            except (cv2.error, KeyError, ValueError):
                stages["nanodet"] = "partial"
            finally:
                object_seconds += perf_counter() - object_started
            appearance_started = perf_counter()
            appearance.add(image, frame)
            appearance_seconds += perf_counter() - appearance_started
        if sample["ocr"]:
            frame["panelChecked"] = True
            jobs.extend(ocr_jobs(image, path, sample, history))
            frame_paths[frame["timeMs"]] = path
            screens.sample(image, path, sample, frame["objects"])
        frames.append(frame)
    confirm_objects(frames)
    retain_confirmed_matches(frames)
    small_report = None
    if small_objects is not None:
        small_report = small_objects.run(detector)
        retain_confirmed_matches(frames)
        object_seconds += small_report["elapsedMs"] / 1000
        if small_report["status"] == "partial":
            stages["nanodet"] = "partial"
    jobs = evenly_spaced(jobs, MAX_OCR_REGIONS)
    ocr_started = perf_counter()
    readings, ocr_stages = recognize(jobs)
    ocr_ms = round((perf_counter() - ocr_started) * 1000)
    started_interface = perf_counter()
    scroll_started = perf_counter()
    screens.retain_recognized_screens(readings)
    supplemental = dense_scroll_samples(folder, request, screens.scroll_candidates)
    mosaics = {}
    scrolls, scroll_statuses = screens.scroll(mosaics, (row[:4] for row in supplemental),
                                               folder=folder,
                                               include_mosaic_files=request.get("includeMosaicFiles", False))
    scroll_ms = round(1000 * (perf_counter() - scroll_started))
    interfaces, screen_crops, screen_statuses = screens.summarize(frame_paths, readings, scrolls, supplemental=supplemental)
    merge_scroll_layouts(interfaces, mosaics)
    screens.retain_panels(interfaces, scrolls, supplemental)
    screen_details = detail_screen_states(folder, request, screens, interfaces)
    state_samples = {(shot, stamp): (shot, stamp, path, region)
                     for shot, stamp, path, region, *_ in [*supplemental, *screens.scroll_candidates]}
    try:
        screen_state_count, state_statuses = read_screen_states(folder, state_samples.values(), interfaces)
    except (OSError, ValueError, cv2.error, ModError, subprocess.SubprocessError):
        logging.getLogger(__name__).exception("Screen state analysis failed; preserving scene evidence")
        screen_state_count, state_statuses = 0, {"screen-state": "failed"}
    screen_statuses.update({f"states-{key}": value for key, value in state_statuses.items()})
    assisted = any("changeBands" in sample for sample in request["samples"])
    change_objects = review_change_objects(detector, request["samples"], frames, folder,
                                           interfaces=interfaces if assisted else None)
    change_recognition = None
    if any(sample.get("changeRecognitionRegions") for sample in request["samples"]):
        change_recognition = recognize_change_regions(detector, request["samples"], folder, recognize=recognize)
    present = {frame["timeMs"] for frame in frames}
    for shot, stamp, _, _, source_frame in supplemental:
        if stamp not in present and stamp in interfaces:
            frames.append({"timeMs": stamp, "sourceFrame": source_frame, "shot": shot,
                           "text": [], "objects": [], "objectChecked": False})
    frames.sort(key=lambda row: row["timeMs"])
    screen_statuses.update({f"scroll-{key}": value for key, value in scroll_statuses.items()})
    for frame in frames:
        stamp = frame["timeMs"]
        frame["text"] = readings.get(stamp, [])
        if stamp in interfaces:
            frame["interfaces"] = interfaces[stamp]
            scroll_lines = [{"text": row["text"], "box": row["box"]}
                            for region in interfaces[stamp] for row in region["elements"]
                            if row["basis"] == "scroll-mosaic" and row.get("text")]
            if scroll_lines:
                frame["text"] = merge_readings(scroll_lines, frame["text"])
    screen_complete = state_statuses.get("screen-state") != "failed" and all(
        screen_statuses.get(f"{prefix}tesseract", screen_statuses.get(f"{prefix}apple-vision", "ok")) == "ok"
        for prefix in ("", "controls-", "scroll-", "states-")
    )
    cpu_ms, peak_mib = process_usage()
    return {"status": "ok", "stages": {**stages, **ocr_stages}, "sampling": request["sampling"],
            **({"objectTiledShots": sorted(tiled_shots)} if request["objectSampling"] == "shot-tiles" else {}),
            **({"smallObjectPass": small_report} if small_report is not None else {}),
            "objectSampling": request["objectSampling"], "objectChecksByShot": list(checks_by_shot.values()),
            "interfaceFrames": len(interfaces), "interfaceOcrCrops": screen_crops,
            "interfaceDetail": screen_details,
            "interfaceStateOcrCrops": screen_state_count,
            "interfaceScrolls": scrolls, "interfaceScrollMs": scroll_ms, "interfaceScrollFrames": len(supplemental),
            "interfaceStatus": "ok" if screen_complete else "partial", "interfaceOcrStatuses": screen_statuses, "interfaceMs": round(1000 * (perf_counter() - started_interface)),
            "ocrPasses": len(jobs), "ocrMs": ocr_ms, "objectMs": round(object_seconds * 1000),
            "objectAppearanceMs": round(appearance_seconds * 1000),
            "objectFrames": sum(row["checked"] for row in checks_by_shot.values()),
            **({"frameChangeObjects": change_objects} if change_objects is not None else {}),
            **({"frameChangeRecognition": change_recognition} if change_recognition is not None else {}),
            "objectModelRuns": detector.model_runs if detector else 0,
            "objectViewCacheHits": detector.cache_hits if detector else 0,
            "cameraMotion": camera.intervals, "cameraMotionMs": round(motion_seconds * 1000),
            "cameraMotionStatus": "ok" if camera.intervals else "unmeasured",
            "pythonWorkerCpuMs": cpu_ms,
            "pythonWorkerPeakMiB": peak_mib,
            "frames": frames, "shotColors": summarize_shot_colors(colors), "uiActivity": ui_activity.summary()}


if __name__ == "__main__":
    if sys.argv[1] == "--screenshot":
        from voice_studio.vision.screen_analysis import screenshot_layout
        options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        report = screenshot_layout(Path(sys.argv[2]), **options)
    elif sys.argv[1] == "--scroll-screenshots":
        from voice_studio.vision.screen_scroll import screenshot_scroll_layout
        request = Path(sys.argv[2])
        report = screenshot_scroll_layout(json.loads(request.read_text(encoding="utf-8")), request.parent)
    else:
        report = analyze_frames(Path(sys.argv[1]))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
