"""Measure bounded pixel-change candidates before selecting OCR and object frames."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from threading import Timer
from time import perf_counter


MOD_ID = "builtin.frame-changes"
CAPABILITY = "vision.frame-changes"
BASIS = "tile-difference"
GRAPHIC_BASIS = "overlay-persistence"
MIN_GRAPHIC_MS = 420
SIDE = 320
TILE = 16
MAX_REGIONS = 4
MAX_FRAMES_PER_SECOND = 8
MAX_PENDING_FRAMES = 16384
MAX_SUPPLEMENTAL_FRAMES = 4096
MAX_OBJECT_FRAMES_PER_SECOND = 4
MAX_OBJECT_REGIONS = 2
MAX_RECOGNITION_REGIONS = 64
GLOBAL_FRACTION = .32
TIMEOUT = 480


def assist_settings():
    """Read the captured frame-change mod settings only for its assist mode."""
    from voice_studio.mod_runtime import current_snapshot

    record = current_snapshot().to_wire()["mods"].get(MOD_ID, {})
    settings = record.get("settings", {})
    return settings if record.get("enabled") and settings.get("mode") == "assist" else {}


def run_scan(inputs, folder, clock_filter, settings):
    """Reuse the vision interpreter for one scan; return only bounded measurements."""
    from voice_studio.vision.video_revolver import VISION, run_worker

    request = Path(folder) / "change-scan.json"
    options = {"inputs": inputs, "clockFilter": clock_filter,
               "threshold": settings["pixelThreshold"], "mode": settings["mode"],
               "adaptiveThreshold": settings.get("adaptiveThreshold", False),
               "compensateCamera": settings.get("compensateCamera", False)}
    request.write_text(json.dumps(options), encoding="utf-8")
    result = run_worker("voice_studio.vision.frame_changes", [request], [VISION], timeout=TIMEOUT + 15)
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError("Frame change worker returned no completed scan")
    return {**result, "mode": settings["mode"]}


def changed_regions(previous, current, threshold, *, adaptive=False, camera=None):
    """Return local boxes for a pair or None for a broad change."""
    regions, _ = compare_frames(previous, current, threshold, adaptive=adaptive, camera=camera)
    return regions


def compare_frames(previous, current, threshold, *, adaptive=False, camera=None):
    """Measure a frame pair and indicate whether the result came from a fallback."""
    from voice_studio.vision.frame_change_motion import adaptive_quiet_pixels, quiet_pixels

    quiet = quiet_pixels(previous, current, threshold)
    regions = regions_from_mask(quiet)
    if regions:
        return regions, False
    if camera is not None:
        aligned = camera.quiet_mask(previous, current, threshold, quiet)
        if aligned is not quiet:
            return regions_from_mask(aligned), True
    if adaptive:
        expanded = regions_from_mask(adaptive_quiet_pixels(previous, current, threshold))
        if expanded:
            return expanded, True
    return regions, False


def regions_from_mask(quiet):
    """Merge neighboring changed cells and return tight normalized pixel bounds."""
    import cv2
    import numpy as np

    changed = SIDE * SIDE - cv2.countNonZero(quiet)
    if changed >= SIDE * SIDE * GLOBAL_FRACTION:
        return None
    if changed < 5:
        return []
    mask = quiet == 0
    tiles = mask.reshape(SIDE // TILE, TILE, SIDE // TILE, TILE).sum(axis=(1, 3))
    active = (tiles >= 5).astype("uint8")
    if not active.any():
        return []
    _, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    regions = []
    for label, (left, top, width, height, _) in enumerate(stats[1:], start=1):
        x, y = left * TILE, top * TILE
        pixels = mask[y:y + height * TILE, x:x + width * TILE]
        component = labels[top:top + height, left:left + width] == label
        area = pixels & component.repeat(TILE, axis=0).repeat(TILE, axis=1)
        ys, xs = np.nonzero(area)
        if not len(xs):
            continue
        x, y = int(x + xs.min()), int(y + ys.min())
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if w * h > SIDE * SIDE * GLOBAL_FRACTION:
            continue
        regions.append({"box": [round(value / SIDE, 5) for value in (x, y, w, h)],
                        "changedPixels": len(xs)})
    regions.sort(key=lambda item: -item["changedPixels"])
    return regions[:MAX_REGIONS]


def scan_pixels(process, threshold, graphics, *, adaptive=False, camera=None):
    """Compare consecutive frames and retain a small, time-distributed set per source second.

    Read FFmpeg's clock in a separate pass after decoding; the worker retains no
    video frames beyond the adjacent pair. Compact accumulated candidates if
    a long or high-rate input reaches the fixed pending metadata budget.
    """
    import numpy as np

    samples = []
    supplemental = []
    previous = None
    wide = 0
    frame_count = dropped = 0
    stride = SIDE * SIDE * 3
    for raw in iter(partial(process.stdout.read, stride), b""):
        if len(raw) != stride:
            raise ValueError("Incomplete video pixel frame")
        current = np.frombuffer(raw, dtype="uint8").reshape(SIDE, SIDE, 3)
        graphics.observe(current, frame_count)
        if previous is not None:
            regions, is_supplemental = compare_frames(
                previous, current, threshold, adaptive=adaptive, camera=camera,
            )
            if regions is None:
                wide += 1
            elif regions:
                target = supplemental if is_supplemental else samples
                limit = MAX_SUPPLEMENTAL_FRAMES if is_supplemental else MAX_PENDING_FRAMES
                target.append((frame_count, regions))
                if len(target) >= limit:
                    dropped += len(target) // 2
                    del target[1::2]
        previous = current
        frame_count += 1
    return samples, supplemental, wide, frame_count, dropped


def bound_samples(samples, frames, cuts, *, for_objects=False):
    """Keep first, last and evenly spaced witnesses in each second without crossing a cut."""
    from voice_studio.vision.scene_sampling import evenly_spaced

    cut_frames = set(cuts)
    buckets = defaultdict(list)
    last_stamp = None
    for index, regions in samples:
        if index in cut_frames or not 0 < index < len(frames):
            continue
        source_frame, stamp = frames[index]
        if index + 1 < len(frames):
            end = frames[index + 1][1]
        else:
            end = stamp + max(1, stamp - frames[index - 1][1])
        if stamp < 0 or end <= stamp or stamp == last_stamp:
            continue
        last_stamp = stamp
        buckets[stamp // 1000].append({"sourceFrame": source_frame, "startMs": stamp,
                                       "endMs": end, "regions": regions})
    rows = []
    omitted = 0
    for bucket in buckets.values():
        if for_objects:
            # Select adjacent candidate pairs from the original clock, before thinning.
            starts = range(0, max(1, len(bucket) - 1), 2)
            pairs = [bucket[index:index + 2] for index in starts]
            if len(bucket) > 2 and len(bucket) % 2:
                pairs.append(bucket[-2:])
            selected = evenly_spaced(pairs, MAX_OBJECT_FRAMES_PER_SECOND // 2)
            kept = list({row["sourceFrame"]: row for pair in selected for row in pair}.values())
        else:
            kept = evenly_spaced(bucket, MAX_FRAMES_PER_SECOND)
        rows.extend(kept)
        omitted += len(bucket) - len(kept)
    return rows, omitted


def fill_supplemental_rows(rows, supplemental):
    """Use free per-second slots for fallback results while retaining the fixed sample set."""
    from voice_studio.vision.scene_sampling import evenly_spaced

    counts = defaultdict(int)
    baseline_stamps = set()
    for row in rows:
        stamp = row["startMs"]
        baseline_stamps.add(stamp)
        counts[stamp // 1000] += 1
    buckets = defaultdict(list)
    for row in supplemental:
        stamp = row["startMs"]
        if stamp not in baseline_stamps:
            buckets[stamp // 1000].append(row)
    extra = []
    for second, bucket in buckets.items():
        available = MAX_FRAMES_PER_SECOND - counts[second]
        if available > 0:
            extra.extend({**row, "supplemental": True} for row in evenly_spaced(bucket, available))
    return sorted(rows + extra, key=lambda row: row["startMs"]), len(supplemental) - len(extra)


def measure(request):
    """Decode one shared clock and RGB stream, timing out the owned FFmpeg subprocess."""
    import cv2
    from voice_studio.vision.frame_change_motion import CameraCompensator
    from voice_studio.vision.frame_graphics import GraphicTracker
    from voice_studio.vision.video_scene import read_frame_clock

    cv2.setNumThreads(1)
    folder = request.parent
    options = json.loads(request.read_text(encoding="utf-8"))
    started = perf_counter()
    filters = (f"[0:V:0]split=2[time][local];[time]{options['clockFilter']}[clock];"
               f"[local]scale={SIDE}:{SIDE},format=rgb24[pixels]")
    command = ["ffmpeg", "-nostdin", "-v", "error", *options["inputs"],
               "-filter_complex_threads", "2", "-filter_complex", filters,
               "-map", "[clock]", "-an", "-sn", "-f", "null", "-",
               "-map", "[pixels]", "-an", "-sn", "-pix_fmt", "rgb24", "-threads:v", "1",
               "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    graphics = GraphicTracker()
    adaptive = options.get("adaptiveThreshold", False)
    camera = CameraCompensator() if options.get("compensateCamera") else None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=folder) as process:
        timer = Timer(TIMEOUT, process.kill)
        timer.daemon = True
        timer.start()
        try:
            samples, supplemental, wide, count, dropped = scan_pixels(
                process, options["threshold"], graphics, adaptive=adaptive, camera=camera,
            )
            if process.wait() != 0:
                raise ValueError("FFmpeg frame change decode failed")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
    frames, cuts = read_frame_clock(folder)
    if len(frames) != count:
        raise ValueError("Frame change pixels and timestamps do not match")
    rows, omitted = bound_samples(samples, frames, cuts)
    if supplemental:
        extra, extra_omitted = bound_samples(supplemental, frames, cuts)
        rows, full_slots = fill_supplemental_rows(rows, extra)
        omitted += extra_omitted + full_slots
    object_rows, _ = bound_samples(samples, frames, cuts, for_objects=True) if options["mode"] in ("objects", "assist") else ([], 0)
    return {"status": "ok", "basis": BASIS, "objectRows": object_rows, "framesCompared": max(0, count - 1),
            "globalFrames": wide, "omittedFrames": omitted + dropped, "rows": rows,
            "adaptiveThreshold": adaptive, "compensateCamera": camera is not None,
            "compensatedFrames": camera.compensated_frames if camera is not None else 0,
            "graphicRows": graphics.report(frames, SIDE, MIN_GRAPHIC_MS, GRAPHIC_BASIS),
            "elapsedMs": round((perf_counter() - started) * 1000)}


def constrain_graphics(rows, boundaries):
    """Keep persistent graphics on both sides of a cut as separate graph witnesses."""
    for row in rows:
        start, end = row["startMs"], row["endMs"]
        cuts = boundaries[bisect_right(boundaries, start):bisect_left(boundaries, end)]
        for stop in (*cuts, end):
            if stop - start >= MIN_GRAPHIC_MS:
                yield {**row, "startMs": start, "endMs": stop}
            start = stop


def augment_plan(plan, frames, cuts, report):
    """Route capped extra source frames/regions without altering baseline flags."""
    mode = report.get("mode")
    if mode not in ("objects", "assist") or report.get("status") != "ok":
        return
    by_frame = {sample["sourceFrame"]: sample for sample in plan}
    boundaries = sorted({index for index in cuts if 0 < index < len(frames)})
    frame_indices = {ordinal: index for index, (ordinal, _) in enumerate(frames)}
    rows = report.pop("objectRows", [])
    if mode == "assist":
        from voice_studio.vision.frame_change_panels import assign_motion

        buckets = defaultdict(list)
        for row in rows:
            buckets[row["startMs"] // 1000].append(row)
        rows = [row for bucket in buckets.values() for row in local_pair(bucket)]
        baseline = [row for row in report.get("rows", ()) if not row.get("supplemental")]
        assign_motion(plan, baseline, [0, *(frames[index][1] for index in boundaries)])
    def sample_for(row):
        """Reuse a planned frame or schedule this exact source frame once."""
        ordinal = row["sourceFrame"]
        index = frame_indices[ordinal]
        if index in cuts:
            return None
        sample = by_frame.get(ordinal)
        if sample is None:
            sample = {"sourceFrame": ordinal, "timeMs": row["startMs"], "shot": bisect_right(boundaries, index),
                      "objects": False, "sectors": False, "ocr": False, "anchor": False, "phase": None,
                      "changeOnly": True}
            plan.append(sample)
            by_frame[ordinal] = sample
        return sample
    for row in rows:
        if sample := sample_for(row):
            sample["changeRegions"] = row["regions"][:MAX_OBJECT_REGIONS]
    for row, index in recognition_witnesses(report.get("rows", ())):
        existing = row["sourceFrame"] in by_frame
        if sample := sample_for(row):
            region = {"index": index, "box": row["regions"][index]["box"]}
            if row.get("supplemental"):
                region["supplemental"] = True
                if not existing:
                    sample["supplementalOnly"] = True
            sample.setdefault("changeRecognitionRegions", []).append(region)
    plan.sort(key=lambda row: row["sourceFrame"])


def local_pair(rows):
    """Keep at most the first compact change pair per second in balanced assist mode."""
    local = [row for row in rows if any(region["box"][2] * region["box"][3] <= .06 for region in row["regions"])]
    if len(local) >= 2 and local[1]["startMs"] - local[0]["startMs"] <= 160:
        return local[:2]
    return local[:1]


def recognition_witnesses(rows):
    """Give fixed regions first refusal on crop slots and use fallback regions in the gaps."""
    from voice_studio.vision.scene_sampling import evenly_spaced

    baseline, supplemental = [], []
    for row in rows:
        target = supplemental if row.get("supplemental") else baseline
        target.append(row)
    known = evenly_spaced(best_crop_regions(baseline), MAX_RECOGNITION_REGIONS)
    available = MAX_RECOGNITION_REGIONS - len(known)
    if not supplemental or not available:
        return known
    used_seconds = {row["startMs"] // 1000 for row, _ in known}
    candidates = [row for row in supplemental if row["startMs"] // 1000 not in used_seconds]
    return known + evenly_spaced(best_crop_regions(candidates), available)


def best_crop_regions(rows):
    """Pick the strongest compact region per second from one source of measurements."""
    candidates = defaultdict(list)
    for row in rows:
        second = row["startMs"] // 1000
        for index, region in enumerate(row["regions"]):
            _, _, width, height = region["box"]
            area = width * height
            pixels = region["changedPixels"]
            if not .008 <= area <= .16 or min(width, height) < .055 or pixels < 240:
                continue
            density = pixels / (area * SIDE * SIDE)
            if density < .12:
                continue
            score = density * pixels
            candidates[second].append((score, row, index))
    selected = []
    for bucket in candidates.values():
        score, row, index = max(bucket, key=lambda entry: entry[0])
        box = row["regions"][index]["box"]
        # During an entrance, the later comparable box more often encloses the settled graphic.
        later = [entry for entry in bucket if entry[0] >= .25 * score
                 and 0 < entry[1]["startMs"] - row["startMs"] <= 350
                 and comparable_region(box, entry[1]["regions"][entry[2]]["box"])]
        if later:
            _, row, index = max(later, key=lambda entry: (entry[1]["startMs"], entry[0]))
        selected.append((row, index))
    return selected


def comparable_region(first, second):
    """Require material overlap and similar extent before preferring a later witness."""
    x, y, width, height = first
    a, b, w, h = second
    common = max(0, min(x + width, a + w) - max(x, a)) * max(0, min(y + height, b + h) - max(y, b))
    smaller, larger = sorted((width * height, w * h))
    return smaller >= .60 * larger and common >= .48 * smaller


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1])), allow_nan=False))
