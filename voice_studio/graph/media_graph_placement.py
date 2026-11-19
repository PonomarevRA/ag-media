"""Fit a transparent replacement asset to one measured object on the source clock."""

from hashlib import sha256
from math import isfinite
from pathlib import Path
import re
from voice_paths import temporary_directory
import subprocess

from voice_studio.graph.media_graph_frame import FRAME_SLOT, image_dimensions, render_frame

MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_ASSET_SIDE = 4096
PNG_INPUT = ("-protocol_whitelist", "file,pipe", "-f", "png_pipe", "-threads", "1")
ALPHA_BOUNDS = re.compile(r"x1:(\d+) x2:\d+ y1:(\d+) y2:\d+ w:(\d+) h:(\d+)")


def png_asset(path):
    """Read an explicitly selected local PNG once and stage those exact bytes for FFmpeg."""
    source = Path(path).expanduser()
    if not source.is_absolute() or not source.is_file():
        raise ValueError("Нужен абсолютный путь к локальному PNG.")
    try:
        with source.open("rb") as stream:
            content = stream.read(MAX_ASSET_BYTES + 1)
    except OSError as exc:
        raise ValueError("PNG объекта недоступен.") from exc
    if len(content) > MAX_ASSET_BYTES or max(image_dimensions(content)) > MAX_ASSET_SIDE:
        raise ValueError("PNG должен быть не больше 8 МиБ и 4096 пикселей по стороне.")
    return content


def alpha_bounds(asset):
    """Find the visible part of the asset, excluding fully transparent padding."""
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-nostdin", *PNG_INPUT, "-i", str(asset), "-map", "0:v:0",
         "-vf", "format=rgba,alphaextract,bbox=min_val=8", "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    bounds = ALPHA_BOUNDS.search(result.stderr)
    if bounds is None:
        raise ValueError("В PNG не обнаружена видимая область.")
    return tuple(map(int, bounds.groups()))


def placement_plan(frame_size, observed, crop, *, height_ratio=1.0, height_px=None, anchor_px=None):
    """Fit the visible asset by height and ground it at the observed bottom-center."""
    frame_w, frame_h = frame_size
    x, y, w, h = observed
    if type(height_ratio) not in (int, float) or not isfinite(height_ratio) or not .15 <= height_ratio <= 4:
        raise ValueError("Коэффициент высоты должен быть от 0,15 до 4.")
    if height_px is not None and (type(height_px) is not int or not 1 <= height_px <= frame_h * 2):
        raise ValueError("Укажи высоту от одного пикселя до двух высот кадра.")
    if anchor_px is not None and (len(anchor_px) != 2 or any(type(n) is not int for n in anchor_px)
                                  or not -frame_w <= anchor_px[0] <= 2 * frame_w
                                  or not -frame_h <= anchor_px[1] <= 2 * frame_h):
        raise ValueError("Точка опоры слишком далеко от исходного кадра.")
    if height_px is not None and height_ratio != 1:
        raise ValueError("Укажи высоту в пикселях или коэффициент высоты.")
    side_edge = x <= .01 or x + w >= .99
    top_edge, bottom_edge = y <= .01, y + h >= .99
    missing = []
    if height_px is None and (top_edge or bottom_edge):
        missing.append("height_px")
    if anchor_px is None and (side_edge or bottom_edge):
        missing.append("anchor_px")
    details = {"frameSize": list(frame_size), "observedBox": observed, "assetCropPx": list(crop),
               "nearFrameEdge": side_edge or top_edge or bottom_edge}
    if missing:
        return {**details, "status": "needs_calibration", "required": missing}
    height = height_px if height_px is not None else max(1, round(h * frame_h * height_ratio))
    width = max(1, round(height * crop[2] / crop[3]))
    if width > frame_w * 3:
        raise ValueError("Ширина объекта после масштабирования превышает три ширины кадра.")
    anchor = anchor_px if anchor_px is not None else [round((x + w / 2) * frame_w), round((y + h) * frame_h)]
    left, top = anchor[0] - width // 2, anchor[1] - height
    if left >= frame_w or top >= frame_h or left + width <= 0 or top + height <= 0:
        raise ValueError("Объект целиком выходит за границы кадра.")
    return {**details, "status": "placed", "anchorPx": anchor,
            "placedBoxPx": [left, top, width, height], "heightRatio": height_ratio,
            "calibration": "manual" if height_px is not None or anchor_px is not None else "observed",
            "placementClipped": left < 0 or top < 0 or left + width > frame_w or top + height > frame_h}


def composite(frame, asset, output, plan):
    """Composite the scaled alpha asset over the source-frame pixels."""
    frame_size = image_dimensions(frame.read_bytes())
    sx, sy = (view / original for view, original in zip(frame_size, plan["frameSize"]))
    left, top, w, h = plan["placedBoxPx"]
    x, y, width, height = round(left * sx), round(top * sy), max(1, round(w * sx)), max(1, round(h * sy))
    cx, cy, cw, ch = plan["assetCropPx"]
    filters = (f"[1:v]format=rgba,crop={cw}:{ch}:{cx}:{cy}:exact=1,scale={width}:{height}:flags=lanczos[asset];"
               f"[0:v]format=rgba[base];[base][asset]overlay=x={x}:y={y}:format=auto,format=rgb24")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *PNG_INPUT, "-i", str(frame),
         *PNG_INPUT, "-i", str(asset), "-filter_complex_threads", "1", "-filter_complex", filters,
         "-frames:v", "1", "-threads:v", "1", str(output)],
        capture_output=True, check=True, timeout=15,
    )
    return output.read_bytes()


def place_object(source, stamp, observed, frame_size, asset_path, *, height_ratio=1.0,
                 height_px=None, anchor_px=None, preview=True, max_side=1024):
    """Return plan metadata and optionally a PNG proof of the computed placement."""
    asset_bytes = png_asset(asset_path)
    try:
        with temporary_directory(prefix="voice-graph-placement-") as folder:
            work = Path(folder)
            asset = work / "asset.png"
            asset.write_bytes(asset_bytes)
            if not FRAME_SLOT.acquire(blocking=False):
                raise ValueError("Обработка кадров занята другим запросом.")
            try:
                crop = alpha_bounds(asset)
            finally:
                FRAME_SLOT.release()
            plan = placement_plan(frame_size, observed, crop, height_ratio=height_ratio, height_px=height_px,
                                  anchor_px=anchor_px)
            plan["assetSha256"] = sha256(asset_bytes).hexdigest()
            if not preview or plan["status"] != "placed":
                return plan, None
            frame = work / "frame.png"
            frame.write_bytes(render_frame(source, stamp, max_side=max_side))
            if not FRAME_SLOT.acquire(blocking=False):
                raise ValueError("Обработка кадров занята другим запросом.")
            try:
                image = composite(frame, asset, work / "placement.png", plan)
            finally:
                FRAME_SLOT.release()
            return plan, image
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось подготовить PNG объекта или кадр примерки.") from exc
