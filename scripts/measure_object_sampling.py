"""Compare object sampling strategies on local files with fixed source-time annotations."""

import argparse
from copy import deepcopy
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from voice_paths import model_path
from voice_studio.graph.media_graph_model import object_layers
from voice_studio.vision.object_appearance import ObjectAppearance, retain_confirmed_matches
from voice_studio.vision.scene_objects import ObjectDetector, calibrate_tiles, confirm_objects, iou, refine_drinkware
from voice_studio.vision.scene_sampling import sample_plan
from voice_studio.vision.small_object_pass import SmallObjectPass
from voice_studio.vision.video_scene import decode_samples, scan_frames

FORMATS = "mov,mp4,m4a,3gp,3g2,mj2"
WIDTH, HEIGHT = 96, 54
PROFILES = ("basic", "every-third", "adaptive", "adaptive-tiles", "basic-tiles", "adaptive-safe",
            "basic-roi", "shot-tiles", "small-detail")


def source_digest(path):
    """Identify the exact source or weight file in the result."""
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def inputs(source):
    """Limit FFmpeg to local media formats used by the studio."""
    return ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source)]


def thumbnails(source, plan, work):
    """Read low-cost candidates at the same native ordinals as detailed analysis."""
    from voice_studio.vision.video_scene import frame_selection

    selector = work / "thumbnail.filter"
    selector.write_text(f"select='{frame_selection(plan)}',scale={WIDTH}:{HEIGHT},format=gray")
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-nostdin", *inputs(source), "-map", "0:V:0", "-an",
         "-filter_threads", "1", "-filter_script:v", str(selector), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-threads:v", "1", "pipe:1"],
        timeout=180,
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)
    if len(frames) != len(plan):
        raise ValueError("Thumbnail decode and candidate clock differ")
    return frames


def adaptation(base, candidates, pixels, *, threshold, gap_ms):
    """Keep every baseline check and add frames with strong local change."""
    baseline = {row["sourceFrame"]: row for row in base}
    selected = []
    previous = None
    for row, image in zip(candidates, pixels, strict=True):
        anchor = baseline.get(row["sourceFrame"])
        if anchor is not None:
            selected.append(anchor)
            previous = row, image
            continue
        if previous is None or row["shot"] != previous[0]["shot"]:
            continue
        age = row["timeMs"] - previous[0]["timeMs"]
        if age < gap_ms:
            continue
        difference = cv2.absdiff(previous[1], image)
        # A moving foreground object may occupy only one block of the whole picture.
        blocks = (difference > 18).reshape(6, 9, 8, 12).mean(axis=(1, 3))
        if blocks.max() < threshold:
            continue
        top, left = np.unravel_index(blocks.argmax(), blocks.shape)
        selected.append({**row, "motionTile": (int(left), int(top)), "sectors": False})
        previous = row, image
    return selected


def targeted_tile(detector, image, objects, center):
    """Retry one zoomed crop around a normalized point and project measured boxes."""
    height, width = image.shape[:2]
    left, top = center
    x = min(.50, max(0, left - .25))
    y = min(.50, max(0, top - .25))
    x1, x2 = round(x * width), round((x + .50) * width)
    y1, y2 = round(y * height), round((y + .50) * height)
    for hit in detector.detect_view(image[y1:y2, x1:x2]):
        u, v, w, h = hit["box"]
        if min(u, v, 1 - u - w, 1 - v - h) < .015:
            continue
        box = [(x1 + u * (x2 - x1)) / width, (y1 + v * (y2 - y1)) / height,
               w * (x2 - x1) / width, h * (y2 - y1) / height]
        if hit["score"] >= .40 and not any(hit["label"] == other["label"] and iou(box, other["box"]) >= .32
                                          for other in objects):
            objects.append({**hit, "box": box, "detailBasis": "motion-tile"})
    return sorted(objects, key=lambda row: -row["score"])[:20]


def small_object_center(objects):
    """Use a compact proposal as a spatial hint for a single additional native crop."""
    candidates = []
    for item in objects:
        x, y, w, h = item["box"]
        if .035 <= w and .04 <= h and w * h <= .014 and item["score"] >= .36:
            candidates.append((w * h, -item["score"], x + w / 2, y + h / 2))
    return min(candidates)[2:] if candidates else None


def observe(plan, work, weights, profile):
    """Run the production NanoDet and evidence filters over a fixed plan."""
    detector = ObjectDetector(weights, reuse_views=True)
    tiled_shots = calibrate_tiles(detector, plan, work) if profile == "shot-tiles" else set()
    small_objects = SmallObjectPass() if profile == "small-detail" else None
    appearances = ObjectAppearance()
    frames = []
    anchors = []
    anchor_appearance = ObjectAppearance() if profile == "adaptive-safe" else None
    last_shot = None
    for index, sample in enumerate(plan):
        shot = sample["shot"]
        if shot != last_shot:
            detector.clear_view_cache()
            last_shot = shot
        path = work / f"{index:04}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing decoded frame {sample['sourceFrame']}")
        sectors = profile == "basic-tiles" or sample["sectors"] or shot in tiled_shots
        if small_objects is not None and sample["sectors"]:
            objects = small_objects.observe_anchor(detector, image, shot)
        else:
            objects = detector.detect(image, sectors=sectors)
        if profile == "adaptive-tiles" and "motionTile" in sample:
            left, top = sample["motionTile"]
            objects = targeted_tile(detector, image, objects, ((left + .5) / 8, (top + .5) / 6))
        if profile == "basic-roi" and (center := small_object_center(objects)):
            objects = targeted_tile(detector, image, objects, center)
        objects, _ = refine_drinkware(detector, image, objects)
        frame = {"timeMs": sample["timeMs"], "shot": shot, "objects": objects}
        if anchor_appearance is not None and "motionTile" not in sample:
            baseline_frame = deepcopy(frame)
            anchor_appearance.add(image, baseline_frame)
            anchors.append(baseline_frame)
        appearances.add(image, frame)
        frames.append(frame)
        if small_objects is not None:
            small_objects.add_sample(path, frame)
    confirm_objects(frames)
    if profile == "adaptive-safe":
        confirm_objects(anchors)
        measured = {row["timeMs"]: row["objects"] for row in anchors}
        for frame in frames:
            if frame["timeMs"] in measured:
                frame["objects"] = measured[frame["timeMs"]]
            else:
                frame["objects"] = [item for item in frame["objects"] if item["score"] >= .65]
    retain_confirmed_matches(frames)
    detail = small_objects.run(detector) if small_objects is not None else None
    if detail is not None:
        retain_confirmed_matches(frames)
    return frames, detector.model_runs, detector.cache_hits, sorted(tiled_shots), detail


def matches(annotation, frame):
    """Check label, time tolerance and spatial overlap on a measured frame."""
    if abs(frame["timeMs"] - annotation["timeMs"]) > annotation.get("toleranceMs", 450):
        return []
    return [item for item in frame["objects"] if item["label"] in annotation["labels"]
            and iou(item["box"], annotation["box"]) >= annotation.get("minIou", .25)]


def evaluate(annotations, frames, profiles, graph):
    """Report measured target recall, coverage by one profile, and negative control hits."""
    groups = {}
    for row in annotations:
        group = groups.setdefault(row["id"], {"checked": 0, "hit": 0, "negative": bool(row.get("negative")),
                                               "profileCoverage": {}, "matchedTimesMs": []})
        group["checked"] += 1
        observed = [frame for frame in frames if matches(row, frame)]
        if not observed:
            continue
        group["hit"] += 1
        group["matchedTimesMs"].append(row["timeMs"])
        for profile, related in profiles.items():
            if any(matches(row, frame) for frame in related):
                counts = group["profileCoverage"]
                counts[profile] = 1 + counts.get(profile, 0)
    for group in groups.values():
        coverage = group.pop("profileCoverage")
        group["profiles"] = len(coverage)
        group["bestProfile"] = max(coverage.values(), default=0)
    return {"groups": groups, "confirmedDetections": sum(len(frame["objects"]) for frame in frames),
            "tracks": len(graph["objectTimeline"]), "profiles": len(graph["objectProfiles"])}


def profile_evidence(report):
    """Assign measured boxes to their canonical graph profiles for continuity scoring."""
    from voice_studio.graph.media_graph_model import build_graph

    graph = build_graph(report, kind="video", duration_ms=report["durationMs"])
    track_by_sample = {left: right for left, relation, right in graph["edges"] if relation == "sample_of"}
    profile_by_track = {left: right for left, relation, right in graph["edges"] if relation == "part_of_profile"}
    result = {}
    for node in graph["nodes"]:
        profile = profile_by_track.get(track_by_sample.get(node["id"]))
        if profile is not None:
            result.setdefault(profile, []).append({"timeMs": node["startMs"],
                "objects": [{"label": node["label"], "box": node["data"]["box"]}]})
    return result


def measure(source, annotations, modes, output, threshold, gap_ms):
    """Execute each full object-only variant and preserve raw metrics for comparison."""
    weights = model_path("vision", "nanodet.onnx")
    with TemporaryDirectory(prefix="objects-bench-") as folder:
        work = Path(folder)
        started = perf_counter()
        clock, cuts = scan_frames(inputs(source), work)
        scan_ms = round((perf_counter() - started) * 1000)
        base = sample_plan(clock, cuts, "basic")
        detailed = sample_plan(clock, cuts, "every-third")
        duration = clock[-1][1] + 1
        editor = {"shots": [{"startMs": clock[index][1]} for index in sorted({0, *cuts}) if index < len(clock)]}
        metadata = {"sourceSha256": source_digest(source), "weightsSha256": source_digest(weights),
                    "frames": len(clock), "shots": len(editor["shots"]), "scanMs": scan_ms}
        result = {**metadata, "threshold": threshold, "minimumMotionGapMs": gap_ms, "variants": {}}
        for mode in modes:
            started = perf_counter()
            if mode.startswith("adaptive"):
                pixels = thumbnails(source, detailed, work)
                plan = adaptation(base, detailed, pixels, threshold=threshold, gap_ms=gap_ms)
            else:
                plan = [row for row in detailed if row["objects"]] if mode == "every-third" else base
            prepared = perf_counter()
            with TemporaryDirectory(dir=work) as decode:
                decode = Path(decode)
                decode_samples(inputs(source), decode, plan)
                decoded = perf_counter()
                frames, passes, hits, tiled_shots, detail = observe(plan, decode, weights, mode)
            observed = perf_counter()
            report = {"durationMs": duration, "videoEvidence": {"editorTimeline": editor},
                      "sceneEvidence": {"frames": frames}}
            layers = object_layers(report)
            facts = evaluate(annotations, frames, profile_evidence(report), layers)
            result["variants"][mode] = {**facts, "requestedFrames": len(plan), "modelRuns": passes, "cacheHits": hits,
                "tiledShots": tiled_shots,
                **({"smallObjectPass": detail} if detail is not None else {}),
                "scoreMs": round((observed - decoded) * 1000), "decodeMs": round((decoded - prepared) * 1000),
                "planningMs": round((prepared - started) * 1000),
                "wallMs": scan_ms + round((perf_counter() - started) * 1000)}
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(source.name, mode, result["variants"][mode], flush=True)


def main():
    """Read fixed annotations and run selected variants in a one-thread OpenCV process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=PROFILES, default=PROFILES)
    parser.add_argument("--motion-threshold", type=float, default=.32)
    parser.add_argument("--motion-gap-ms", type=int, default=330)
    args = parser.parse_args()
    if not 0 < args.motion_threshold <= 1 or args.motion_gap_ms < 0:
        parser.error("Motion threshold must be in (0, 1] with a nonnegative minimum gap.")
    cv2.setNumThreads(1)
    annotations = json.loads(args.annotations.read_text())[args.source.name]
    measure(args.source, annotations, args.modes, args.output, args.motion_threshold, args.motion_gap_ms)


if __name__ == "__main__":
    main()
