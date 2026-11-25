"""Render a source frame at a media-graph time without storing screenshot blobs."""

from math import isfinite
from pathlib import Path
import struct
import subprocess
from voice_paths import temporary_directory
from threading import BoundedSemaphore

from voice_studio.graph.media_graph_frame_cache import FrameCache
from voice_studio.speech.media_transcript import FORMATS
from voice_studio.vision.video_colors import frame_colors
from voice_studio.vision.video_export import ass_header, event, safe_text

FRAME_SLOT = BoundedSemaphore(2)
RAW_FRAMES = FrameCache()
MAX_TIME_MS = 2 * 60 * 60 * 1000
PNG_HEADER = b"\x89PNG\r\n\x1a\n"
BOX_COLORS = {
    "face": "E8F1F5",
    "speaker": "4EB3FF",
    "body": "DBDD70",
    "object": "78DC89",
    "text": "E8E671",
    "change": "66DFFF",
    "target": "E5A1FF",
}


def time_argument(milliseconds, nudge_us=0):
    """Format an exact FFmpeg time without rounding through a float."""
    seconds, microseconds = divmod(milliseconds * 1000 + nudge_us, 1_000_000)
    return f"{seconds}.{microseconds:06d}"


def extract_frame(source, folder, at_ms, max_side, *, region=None):
    """Keep the last native frame at or before the requested time on the source clock."""
    output = folder / "frame.png"
    scale = (f"scale='min(iw,{max_side})':'min(ih,{max_side})':"
             "force_original_aspect_ratio=decrease,setsar=1")
    if region is not None:
        x, y, w, h = region
        # Crop precedes scale, preserving small details from the oriented source.
        scale = (f"crop=w=max(1\\,min(iw-floor(iw*{x})\\,round(iw*{w})))"
                 f":h=max(1\\,min(ih-floor(ih*{y})\\,round(ih*{h})))"
                 f":x=floor(iw*{x}):y=floor(ih*{y}):exact=1," + scale)
    # Accurate input seek skips old GOPs. image2 replaces the same file for each frame.
    # The microsecond on the exclusive end includes a frame exactly at the requested time.
    for lookback_ms in (200, 2000, None):
        if lookback_ms is None:
            # Sparse video might hold one presentation frame for several seconds.
            seek = ["-ss", time_argument(at_ms), "-noaccurate_seek"]
        else:
            seek = ["-ss", time_argument(max(0, at_ms - lookback_ms))]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-start_at_zero", *seek,
             "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
             "-max_pixels", "67108864", "-threads", "2", "-i", str(source),
             "-to", time_argument(at_ms, 1), "-map", "0:v:0", "-an", "-sn",
             "-filter_threads", "1", "-vf", scale, "-pix_fmt", "rgb24", "-threads:v", "1",
             "-fps_mode", "passthrough", "-update", "1", str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=True,
        )
        if output.is_file() and output.stat().st_size:
            return output.read_bytes()
        if lookback_ms is not None and at_ms <= lookback_ms:
            break
    raise ValueError("No video frame is available at the requested time.")


def image_dimensions(png):
    """Read the PNG frame size from the FFmpeg-produced image header."""
    if len(png) < 24 or png[:8] != PNG_HEADER or png[12:16] != b"IHDR":
        raise ValueError("Could not decode a PNG frame.")
    return struct.unpack(">II", png[16:24])


def pixel_box(box, width, height):
    """Convert finite normalized geometry to a clipped source-image rectangle."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(type(value) in (int, float) and isfinite(value) and 0 <= value <= 1 for value in box):
        return None
    x, y, w, h = box
    left, top = round(x * width), round(y * height)
    right, bottom = min(width, round((x + w) * width)), min(height, round((y + h) * height))
    if min(right - left, bottom - top) < 8:
        return None
    return left, top, right - left, bottom - top


def box_style(box, label, fill, width, height):
    """Draw a hollow boundary and optional annotation for one recorded observation."""
    x, y, w, h = box
    line = max(2, min(width, height) // 300)
    if min(w, h) <= 2 * line:
        return
    outer = f"m 0 0 l {w} 0 {w} {h} 0 {h} 0 0"
    inner = f"m {line} {line} l {line} {h-line} {w-line} {h-line} {w-line} {line} {line} {line}"
    style = rf"{{\an7\pos({x},{y})\1c&H{fill}&\p1}}"
    yield event(0, 0, 1000, "Box", f"{style}{outer} {inner}")
    if isinstance(label, str) and label.strip():
        top = max(0, y - max(16, min(width, height) // 31))
        left = min(x, max(0, width - 96))
        prefix = rf"{{\an7\pos({left},{top})\1c&H{fill}&}}"
        yield event(1, 0, 1000, "Label", prefix + safe_text(label[:160]))


def overlay_script(context, at_ms, width, height):
    """Assemble bounded libass annotations from graph data, with no executable text."""
    yield ass_header(width, height)
    boxes = context.get("boxes")
    for item in boxes[:24] if isinstance(boxes, (list, tuple)) else ():
        if not isinstance(item, dict):
            continue
        box = pixel_box(item.get("box"), width, height)
        kind = item.get("kind")
        fill = BOX_COLORS.get(kind) if isinstance(kind, str) else None
        if box and fill:
            yield from box_style(box, item.get("label"), fill, width, height)
    seconds, ms = divmod(at_ms, 1000)
    minutes, ss = divmod(seconds, 60)
    hh, mm = divmod(minutes, 60)
    header = f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
    shot = context.get("shot")
    if isinstance(shot, str) and shot.strip():
        header += "  ·  " + shot[:100]
    yield event(2, 0, 1000, "Label", r"{\an7\pos(12,12)}" + safe_text(header))
    caption = context.get("caption")
    if isinstance(caption, str) and caption.strip():
        yield event(2, 0, 1000, "Caption", safe_text(caption[:300]))


def annotate_frame(folder, context, at_ms, png):
    """Paint the technical layer over the already selected source pixels."""
    width, height = image_dimensions(png)
    (folder / "technical.ass").write_text("".join(overlay_script(context, at_ms, width, height)), encoding="utf-8")
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
         "-i", "frame.png", "-vf", "ass=technical.ass", "-frames:v", "1",
         "-pix_fmt", "rgb24", "-threads:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"],
        cwd=folder, capture_output=True, timeout=8, check=True,
    ).stdout


def raw_frame(source, folder, at_ms, max_side, include_palette, region=None):
    """
    Reuse exact source pixels, leaving graph-specific decoration to the caller.

    Inputs source/folder name a resolved original and private request directory.
    at_ms/max_side select source time/size; include_palette requests color measurements.
    Result is (PNG bytes, palette or None). File/decode errors propagate to render_frame.
    A changing source is never inserted; source verification remains the MCP caller's duty.
    """
    before = source.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    key = (str(source), identity, at_ms, max_side)
    if region is not None:
        key += (tuple(region),)
    cached = RAW_FRAMES.get(key)
    if cached is None:
        png = (extract_frame(source, folder, at_ms, max_side) if region is None
               else extract_frame(source, folder, at_ms, max_side, region=region))
        image_dimensions(png)
        palette = None
    else:
        png, palette = cached
        # libass reads the selected PNG from this request's private folder.
        (folder / "frame.png").write_bytes(png)
    if include_palette and palette is None:
        palette = frame_colors(png)
    after = source.stat()
    current = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if current == identity:
        RAW_FRAMES.put(key, png, palette)
    return png, palette if include_palette else None


def render_frame(source: Path, at_ms: int, *, max_side: int = 1280, context: dict | None = None,
                 include_palette: bool = False, region=None) -> bytes | tuple[bytes, dict]:
    """
    Return source PNG pixels and optional color measurements before any overlay.

    Inputs source/at_ms select a local frame within two hours; max_side is 160–1920.
    include_palette enables color measurements; context supplies technical decoration.
    Context is reapplied on every request. Result is bytes or (bytes, palette).
    Invalid inputs, unavailable originals, busy slots and decode failures raise ValueError.
    """
    if region is not None and (context is not None or not pixel_box(region, 10000, 10000)):
        raise ValueError("A crop requires normalized geometry and no full-frame overlay.")
    if type(at_ms) is not int or not 0 <= at_ms < MAX_TIME_MS:
        raise ValueError("The screenshot time must be within the first two hours.")
    if type(max_side) is not int or not 160 <= max_side <= 1920:
        raise ValueError("Screenshot size must be between 160 and 1920 pixels.")
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("The original media file is unavailable.") from exc
    if not source.is_file():
        raise ValueError("The original media file is unavailable.")
    if not FRAME_SLOT.acquire(blocking=False):
        raise ValueError("Screenshot renderer is busy. Try again.")
    try:
        with temporary_directory(prefix="voice-graph-frame-") as raw:
            folder = Path(raw)
            png, palette = (raw_frame(source, folder, at_ms, max_side, include_palette) if region is None
                            else raw_frame(source, folder, at_ms, max_side, include_palette, region))
            if context is not None:
                png = annotate_frame(folder, context, at_ms, png)
            image_dimensions(png)
            return (png, palette) if include_palette else png
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not render a screenshot from the original file.") from exc
    finally:
        FRAME_SLOT.release()


def render_audio(source: Path, start_ms: int, duration_ms: int, stream_index: int | None = None) -> bytes:
    """Decode at most ten seconds from an indexed source stream to a playable PCM WAV."""
    if not 1 <= duration_ms <= 10000 or not 0 <= start_ms < MAX_TIME_MS:
        raise ValueError("Вырезка звука должна укладываться в 10 секунд.")
    mapping = "0:a:0" if stream_index is None else f"0:{stream_index}"
    if not FRAME_SLOT.acquire(blocking=False):
        raise ValueError("Извлечение из исходника занято другим запросом.")
    try:
        with temporary_directory(prefix="voice-graph-audio-") as raw:
            target = Path(raw) / "clip.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", time_argument(start_ms),
                 "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                 "-i", str(source), "-t", time_argument(duration_ms), "-map", mapping,
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=True,
            )
            payload = target.read_bytes()
            if len(payload) <= 80:
                raise ValueError("На выбранном интервале нет звука.")
            return payload
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось извлечь звук из исходника.") from exc
    finally:
        FRAME_SLOT.release()
