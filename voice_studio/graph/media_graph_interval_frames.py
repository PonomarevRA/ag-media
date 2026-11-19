"""
Build a bounded source-frame grid with measured timestamps and separate captions.
"""

from fractions import Fraction
from pathlib import Path
import re
import subprocess
import textwrap
from time import monotonic

from voice_paths import temporary_directory
from voice_studio.graph.media_graph_frame import FRAME_SLOT, image_dimensions, time_argument
from voice_studio.speech.media_transcript import FORMATS
from voice_studio.vision.video_export import ass_header, event, safe_text

MAX_SAMPLES = 8
MAX_INTERVAL_MS = 60_000
DENSE_SPAN_MS = 2000
CAPTION_HEIGHT = 96
TIME_BASE = re.compile(r"config in time_base: (\d+)/(\d+)")
FRAME_PTS = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:")


def validate_samples(samples, end_ms, tile_width):
    """
    Reject unbounded input before accessing media; retain input order and duplicates.

    samples contains 1..8 dictionaries with integer atMs and optional <=240-character
    caption. end_ms is the exclusive source-millisecond limit, at most sixty seconds
    after the earliest request. tile_width is an even 320..640-pixel picture width.
    Return new metadata dictionaries; raise ValueError for an invalid argument.
    """
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("The grid requires between one and eight samples.")
    if type(end_ms) is not int or end_ms <= 0:
        raise ValueError("The interval end must be a positive integer in milliseconds.")
    if type(tile_width) is not int or not 320 <= tile_width <= 640 or tile_width % 2:
        raise ValueError("Grid tile width must be an even integer between 320 and 640.")
    result = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each grid sample must contain an integer atMs.")
        at_ms = sample.get("atMs")
        caption = sample.get("caption", "")
        if type(at_ms) is not int or not 0 <= at_ms < end_ms:
            raise ValueError("Grid sample times must be nonnegative and before the interval end.")
        if not isinstance(caption, str) or len(caption) > 240:
            raise ValueError("A grid caption must contain at most 240 characters.")
        result.append({"requestedMs": at_ms, "caption": caption})
    if end_ms - min(item["requestedMs"] for item in result) > MAX_INTERVAL_MS:
        raise ValueError("A grid interval must not exceed sixty seconds.")
    return result


def decode_frames(source, folder, times, end_ms, tile_width, timeout):
    """
    Decode one bounded seek window, retaining integer native PTS for each output PNG.

    source is a checked local Path; folder owns temporary PNGs. times is a sorted
    nonempty list of unique source-millisecond requests before exclusive end_ms.
    tile_width bounds picture pixels; timeout limits this subprocess in seconds.
    Return frame paths and exact rational clocks, or an empty list at EOF.
    Raise ValueError for an incomplete clock/image pairing; subprocess failures and
    OSError propagate to the public renderer for conversion to ValueError.
    """
    crossings = []
    for at_ms in times:
        at = time_argument(at_ms)
        crossings.append(f"gte(t,{at})*(isnan(prev_t)+lt(prev_t,{at}))")
    # Several requests may cross the same native frame. Emit a bounded tail after
    # the last request so -frames:v can stop without decoding the rest of the interval.
    crossings.append(f"gte(t,{time_argument(times[-1])})")
    select = f"select='lt(t,{time_argument(end_ms)})*({'+'.join(crossings)})'"
    picture_height = tile_width * 9 // 16
    scale = (f"scale={tile_width}:{picture_height}:force_original_aspect_ratio=decrease,"
             "setsar=1")
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "info", "-nostdin", "-y", "-copyts", "-start_at_zero",
         "-ss", time_argument(times[0]), "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
         "-max_pixels", "67108864", "-threads", "2", "-i", str(source),
         "-to", time_argument(end_ms), "-map", "0:v:0", "-an", "-sn", "-filter_threads", "1",
         "-vf", f"{select},showinfo,{scale}", "-frames:v", str(len(times)),
         "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads:v", "1", "sample-%02d.png"],
        cwd=folder, capture_output=True, timeout=timeout, check=True,
    )
    # Integer PTS and the filter's time base preserve precision for VFR sources.
    # start_at_zero follows the same source clock as the existing single-frame tool.
    log = process.stderr.decode("utf-8", errors="replace")
    base_match = TIME_BASE.search(log)
    matches = FRAME_PTS.findall(log)
    if not matches:
        return []
    if base_match is None or int(base_match[2]) == 0:
        raise ValueError("The decoder did not report the source frame clock.")
    numerator, denominator = map(int, base_match.groups())
    result = []
    for index, pts_text in matches:
        frame = folder / f"sample-{int(index) + 1:02d}.png"
        if not frame.is_file():
            raise ValueError("The decoder frame and timestamp counts disagree.")
        pts = int(pts_text)
        actual = Fraction(pts * numerator * 1000, denominator)
        if actual < end_ms:
            result.append({"path": frame, "time": actual, "sourcePts": pts,
                           "sourceTimeBase": f"{numerator}/{denominator}"})
    return result


def decode_strategy(samples):
    """
    Choose one shared decode for nearby requests, or bounded seeks for sparse requests.

    samples contains validated requestedMs metadata. Return a strategy name and
    sorted unique time batches; requests spanning <=2000 ms share a decode.
    """
    times = sorted({item["requestedMs"] for item in samples})
    if times[-1] - times[0] <= DENSE_SPAN_MS:
        return "dense-single-pass", [times]
    return "sparse-accurate-seeks", [[at_ms] for at_ms in times]


def selected_frames(source, folder, samples, end_ms, tile_width):
    """
    Select native frames under a shared twenty-second budget, preserving source-clock PTS.

    source is the checked local file; folder owns all per-seek scratch directories.
    samples is validated request metadata; end_ms is exclusive source milliseconds.
    tile_width bounds decoded picture pixels. Return source-ordered frame records.
    Raise TimeoutExpired when the shared budget expires; decode_frames errors propagate.
    """
    _, batches = decode_strategy(samples)
    deadline = monotonic() + 20
    result = []
    for index, times in enumerate(batches):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("ffmpeg-grid-decode", 20)
        target = folder / f"decode-{index}"
        target.mkdir()
        result.extend(decode_frames(source, target, times, end_ms, tile_width, remaining))
    return result


def assign_frames(samples, decoded, end_ms):
    """
    Map selected frames to each original request without hiding unavailable images.

    samples preserves validated request order and duplicates; decoded contains
    source-ordered frame records with exact rational time. end_ms is exclusive
    source milliseconds. Return (metadata, temporary Path or None) pairs; no match
    yields unavailable status and null actualMs/deltaMs/PTS fields.
    """
    result = []
    for sample in samples:
        selected = next((frame for frame in decoded
                         if sample["requestedMs"] <= frame["time"] < end_ms), None)
        item = {**sample, "status": "unavailable", "actualMs": None, "deltaMs": None,
                "sourcePts": None, "sourceTimeBase": None}
        if selected is not None:
            item.update(status="available", actualMs=float(selected["time"]),
                        deltaMs=float(selected["time"] - sample["requestedMs"]),
                        sourcePts=selected["sourcePts"], sourceTimeBase=selected["sourceTimeBase"])
        result.append((item, selected["path"] if selected is not None else None))
    return result


def clock_label(milliseconds):
    """
    Format source-clock milliseconds for a human-readable grid label.

    milliseconds is finite source time; return HH:MM:SS.mmm rounded for display.
    Exact native PTS remains separately available in sample metadata.
    """
    seconds, ms = divmod(round(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def grid_captions(assignments, tile_width, tile_height, columns, width, height):
    """
    Place all text in caption bands outside unannotated source-image rectangles.

    assignments contains (sample metadata, temporary Path or None) pairs.
    tile_width/tile_height define each cell in pixels; columns defines row order.
    width/height define the full ASS coordinate space in pixels. Yield an ASS
    header and safe bounded text events; caption display may be truncated to two
    lines while full bounded caption text remains in returned metadata.
    """
    yield ass_header(width, height)
    picture_height = tile_height - CAPTION_HEIGHT
    for index, (sample, _) in enumerate(assignments):
        x = (index % columns) * tile_width + 10
        y = (index // columns) * tile_height + picture_height + 7
        lines = [f"#{index + 1} requested {clock_label(sample['requestedMs'])}"]
        if sample["status"] == "available":
            lines.append(f"source {clock_label(sample['actualMs'])}  +{sample['deltaMs']:.3f} ms")
        else:
            lines.append("NO NATIVE FRAME AT/AFTER REQUEST")
        caption = safe_text(sample["caption"])
        # Limit by both characters and lines; arbitrary unbroken words cannot spill into another tile.
        wrapped = textwrap.wrap(caption, width=(tile_width - 24) // 18)
        if len(wrapped) > 2:
            wrapped = [wrapped[0], wrapped[1][:-1] + "…"]
        lines.extend(wrapped)
        for line_index, line in enumerate(lines):
            top = y + line_index * 20
            right = (index % columns + 1) * tile_width - 8
            band_top = (index // columns) * tile_height + picture_height + 2
            band_bottom = (index // columns + 1) * tile_height - 2
            # Remove opaque boxes that can erase nearby labels after font fallback.
            # Clip the whole caption band, retaining complex-script ascenders/descenders.
            position = (rf"{{\an7\pos({x},{top})\fs16\q2\bord0\shad0"
                        rf"\clip({x - 2},{band_top},{right},{band_bottom})}}")
            yield event(1, 0, 1000, "Label", position + safe_text(line))


def compose_grid(folder, assignments, tile_width):
    """
    Compose at most eight letterboxed pictures and safe libass caption bands into PNG.

    folder owns scratch files; assignments contains validated sample metadata and
    paths under that folder, or None for visible gaps. tile_width is validated
    pixel width. Return (PNG bytes, full width, full height), bounded at 1920x1368.
    Raise ValueError for a dimension mismatch; FFmpeg errors and OSError propagate
    to the public renderer. RGBA packing prevents packed-RGB24 libass corruption
    after xstack while preserving source RGB pixels exactly.
    """
    columns = min(2, len(assignments)) if len(assignments) <= 4 else 3
    rows = (len(assignments) + columns - 1) // columns
    picture_height = tile_width * 9 // 16
    tile_height = picture_height + CAPTION_HEIGHT
    width, height = columns * tile_width, rows * tile_height
    (folder / "captions.ass").write_text("".join(grid_captions(
        assignments, tile_width, tile_height, columns, width, height)), encoding="utf-8")
    inputs, filters, labels, positions = [], [], [], []
    for index, (_, frame) in enumerate(assignments):
        if frame is None:
            inputs.extend(["-f", "lavfi", "-i", f"color=c=0x242831:s={tile_width}x{picture_height}:d=0.04"])
        else:
            inputs.extend(["-threads", "1", "-i", str(frame.relative_to(folder))])
        filters.append(f"[{index}:v]pad={tile_width}:{picture_height}:(ow-iw)/2:(oh-ih)/2:color=0x10141c,"
                       f"pad={tile_width}:{tile_height}:0:0:color=0x10141c,setsar=1[v{index}]")
        labels.append(f"[v{index}]")
        positions.append(f"{index % columns * tile_width}_{index // columns * tile_height}")
    if len(assignments) == 1:
        filters.append("[v0]format=rgba,ass=captions.ass[out]")
    else:
        filters.append(f"{''.join(labels)}xstack=inputs={len(assignments)}:layout={'|'.join(positions)}:"
                       "fill=0x10141c,format=rgba,ass=captions.ass[out]")
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe,lavfi", *inputs,
         "-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[out]",
         "-frames:v", "1", "-pix_fmt", "rgb24", "-threads:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"],
        cwd=folder, capture_output=True, timeout=8, check=True,
    ).stdout
    if image_dimensions(png) != (width, height):
        raise ValueError("The grid dimensions do not match the bounded layout.")
    return png, width, height


def render_interval_grid(source: Path, samples: list[dict], *, end_ms: int,
                         tile_width: int = 480) -> tuple[bytes, dict]:
    """
    Return PNG and actual frame metadata for <=8 requests spanning <=60 seconds.

    source is a local media Path. samples contains atMs integer source milliseconds
    and optional caption text up to 240 characters. end_ms is exclusive source
    milliseconds, at most sixty seconds after the earliest request. tile_width
    defaults to 480 pixels and must be an even integer from 320 through 640.
    Times use normalized source milliseconds. The first
    native frame at or after each request is chosen, without interpolation or
    boxes; a frame outside the interval is never substituted. Input order and
    duplicates are retained. Missing images get visible gap tiles. Unavailable
    files, a busy renderer, invalid input and FFmpeg failures raise ValueError.
    """
    validated = validate_samples(samples, end_ms, tile_width)
    try:
        source = Path(source).resolve(strict=True)
    except (TypeError, OSError, RuntimeError) as exc:
        raise ValueError("The original media file is unavailable.") from exc
    if not source.is_file():
        raise ValueError("The original media file is unavailable.")
    if not FRAME_SLOT.acquire(blocking=False):
        raise ValueError("Screenshot renderer is busy. Try again.")
    try:
        with temporary_directory(prefix="voice-graph-grid-") as raw:
            folder = Path(raw)
            decoded = selected_frames(source, folder, validated, end_ms, tile_width)
            assignments = assign_frames(validated, decoded, end_ms)
            png, width, height = compose_grid(folder, assignments, tile_width)
            strategy, batches = decode_strategy(validated)
            available = sum(item["status"] == "available" for item, _ in assignments)
            status = "unavailable"
            if available == len(samples):
                status = "available"
            elif available:
                status = "partial"
            return png, {
                "status": status,
                "requested": len(samples), "returned": available, "samples": [item for item, _ in assignments],
                "selection": "first-native-frame-at-or-after-request-before-end",
                "clock": "source-normalized-start-zero", "endMs": end_ms,
                "width": width, "height": height, "decodePasses": len(batches), "strategy": strategy,
                "limits": {"maxSamples": MAX_SAMPLES, "maxIntervalMs": MAX_INTERVAL_MS,
                           "decodeTimeoutSeconds": 20, "composeTimeoutSeconds": 8,
                           "maxWidth": 1920, "maxHeight": 1368},
            }
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not render a frame grid from the original file.") from exc
    finally:
        FRAME_SLOT.release()
