"""Extract a compact shot palette and visible-person statistics from sampled video pixels."""

from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from voice_paths import temporary_directory
from voice_studio.synthesis.engines import ffmpeg
from voice_studio.vision.scene_sampling import evenly_spaced
from voice_studio.vision.video_colors import hex_rgb

WIDTH, HEIGHT = 96, 54
SAMPLES_PER_SHOT = 4


def palette(histogram, limit=4):
    """Report dominant color clusters and their share of the measured region."""
    clusters = defaultdict(lambda: [0, 0, 0, 0])
    for color, count in histogram.items():
        cluster = clusters[tuple((value + 16) // 32 for value in color)]
        cluster[0] += count
        for index, value in enumerate(color, 1):
            cluster[index] += value * count
    total = sum(histogram.values())
    return [{"hex": hex_rgb(tuple(value / count for value in channels)),
             "share": round(count / total, 3)}
            for count, *channels in sorted(clusters.values(), key=lambda row: -row[0])[:limit]]


def pixel_rows(box):
    """Convert a normalized region to clipped thumbnail coordinates."""
    x, y, width, height = box
    return (max(0, min(WIDTH, round(x * WIDTH))), max(0, min(HEIGHT, round(y * HEIGHT))),
            max(0, min(WIDTH, round((x + width) * WIDTH))),
            max(0, min(HEIGHT, round((y + height) * HEIGHT))))


def estimated_figure(face):
    """Reserve space around a face when full figure geometry is absent."""
    x, y, w, h = face
    return (x - w, y - .2 * h, 3 * w, 5.2 * h)


def clothing_region(face, body):
    """Measure a central area below the face and within the visible figure."""
    x, y, w, h = face
    left, _, width, _ = body
    top = y + h * 1.25
    bottom = min(y + h * 2.9, body[1] + body[3])
    return (max(left + width * .18, x - w * .42), top,
            min(left + width * .82, x + w * 1.42) - max(left + width * .18, x - w * .42),
            bottom - top)


def nearby_people(evidence, stamp, sample_index, frames):
    """Match body geometry by exact track and identity within 150 ms of the thumbnail."""
    hits = frames[sample_index]
    layer = evidence.get("personLayer") or {}
    body_frames = layer.get("frames") or []
    stamps = layer.get("_sample_times", ())
    i = bisect_right(stamps, stamp)
    closest = min(body_frames[max(i - 1, 0):i + 1], key=lambda row: abs(row["timeMs"] - stamp), default=None)
    bodies = {(row["track"], row["faceId"]): row.get("box") for row in closest["people"]} if (
        closest is not None and abs(closest["timeMs"] - stamp) <= 150) else {}
    people = {}
    for hit in hits:
        track, face = hit[0], hit[6] if len(hit) > 6 else hit[0]
        if face in people:
            continue
        face_box = hit[1:5]
        body = bodies.get((track, face))
        region = body or estimated_figure(face_box)
        people[face] = (pixel_rows(region), pixel_rows(clothing_region(face_box, region)), bool(body))
    return people


def collect_colors(raw, people):
    """Count whole-frame pixels and background outside approximate figure regions."""
    whole, background = Counter(), Counter()
    outfits = {face: Counter() for face in people}
    for i in range(WIDTH * HEIGHT):
        x, y = i % WIDTH, i // WIDTH
        pixel = tuple(raw[i * 3:i * 3 + 3])
        whole[pixel] += 1
        occupants = [face for face, ((x1, y1, x2, y2), _, _) in people.items()
                     if x1 <= x < x2 and y1 <= y < y2]
        if not occupants:
            background[pixel] += 1
        if len(occupants) == 1:
            face = occupants[0]
            _, (x1, y1, x2, y2), _ = people[face]
            if x1 <= x < x2 and y1 <= y < y2:
                outfits[face][pixel] += 1
    return whole, background, outfits


def summarize_shot(start, end, samples, evidence, stream, time_offset):
    """Combine measured frames while separating visibility and pixel color denominators."""
    timeline = evidence.get("timeline") or {}
    frames, step = timeline.get("frames") or [], timeline.get("stepMs") or 40
    whole, background = Counter(), Counter()
    colors, areas, body_counts, color_counts = defaultdict(Counter), Counter(), Counter(), Counter()
    measured = frames[(start + step - 1) // step:(end + step - 1) // step]
    counts = Counter(face for frame in measured for face in {hit[6] if len(hit) > 6 else hit[0] for hit in frame})
    background_frames = 0
    for row in samples:
        raw = stream.read(WIDTH * HEIGHT * 3)
        if len(raw) != WIDTH * HEIGHT * 3:
            raise ValueError("Moodbook thumbnail missing")
        t = row["timeMs"] + time_offset
        index = t // step
        covered = 0 <= index < len(frames)
        people = nearby_people(evidence, t, index, frames) if covered else {}
        scene_colors, backdrop, outfits = collect_colors(raw, people)
        whole.update(scene_colors)
        if not covered:
            continue
        background_frames += 1
        background.update(backdrop)
        for face, ((x1, y1, x2, y2), _, measured_body) in people.items():
            if measured_body:
                areas[face] += (x2 - x1) * (y2 - y1) / (WIDTH * HEIGHT)
                body_counts[face] += 1
            if outfits[face]:
                colors[face].update(outfits[face])
                color_counts[face] += 1
    backdrop_palette = palette(background)
    return {"startMs": start, "endMs": end, "sampleCount": len(samples), "personSampleCount": len(measured),
            "representativeMs": samples[len(samples) // 2]["timeMs"],
            "palette": palette(whole), "backgroundPalette": backdrop_palette,
            "backgroundKeyColor": backdrop_palette[0]["hex"] if backdrop_palette else None,
            "backgroundBasis": "outside-tracked-figure-regions" if background_frames else "unmeasured",
            "characters": [
                {"faceId": face, "visibleFrames": count, "visibilityShare": round(count / len(measured), 3),
                 "frameAreaShare": round(areas[face] / body_counts[face], 3) if body_counts[face] else None,
                 "areaSamples": body_counts[face], "colorSampleCount": color_counts[face],
                 "clothingPalette": palette(colors[face]),
                 "clothingBasis": "upper-figure-region"}
                for face, count in counts.most_common()
            ]}


def build_moodbook(source, formats, scene, evidence, duration, *, window=None):
    """Sample a small fixed number of native frames per shot in one bounded thumbnail pass."""
    from voice_studio.vision.video_scene import frame_selection

    rows = defaultdict(list)
    for frame in scene.get("frames", []):
        if 0 <= frame["timeMs"] < duration and type(frame.get("sourceFrame")) is int:
            rows[frame["shot"]].append(frame)
    if not rows:
        return []
    starts = [0, *scene.get("shotStartsMs", ()), duration]
    plan = [(starts[shot], starts[shot + 1], evenly_spaced(samples, SAMPLES_PER_SHOT))
            for shot, samples in sorted(rows.items()) if 0 <= shot < len(starts) - 1]
    if not plan:
        return []
    evidence = evidence or {}
    if layer := evidence.get("personLayer"):
        evidence = {**evidence, "personLayer": {**layer,
                    "_sample_times": [r["timeMs"] for r in layer.get("frames") or []]}}
    with temporary_directory(prefix="moodbook-") as folder:
        root = Path(folder)
        inputs = ["-protocol_whitelist", "file,pipe", "-format_whitelist", formats]
        if window:
            begin, end = window
            inputs += ["-ss", f"{begin / 1000:.3f}", "-t", f"{(end - begin) / 1000:.3f}"]
        selection = frame_selection([row for _, _, rows in plan for row in rows])
        filters = root / "frames.filter"
        filters.write_text(f"select='{selection}',scale={WIDTH}:{HEIGHT}:flags=area", encoding="utf-8")
        output = root / "frames.rgb"
        ffmpeg([*inputs, "-i", str(source), "-map", "0:V:0", "-an", "-sn", "-filter_threads", "1",
                "-filter_script:v", str(filters), "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
                "-threads:v", "1", "-f", "rawvideo", str(output)], timeout=180)
        with output.open("rb") as stream:
            return [summarize_shot(begin, end, samples, evidence, stream, 0)
                    for begin, end, samples in plan]
