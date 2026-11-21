"""Extract a local screenshot region or timed video region for use as visual evidence."""

from math import isfinite
from pathlib import Path
import subprocess

from voice_paths import temporary_directory
from voice_studio.graph.media_graph_frame import MAX_TIME_MS, extract_frame, image_dimensions
from voice_studio.vision.image_input import verify_image


def crop_parameters(region, coordinate_space, at_ms, max_side):
    """Reject invalid shapes and timestamps before touching the source file."""
    if len(region) != 4 or any(type(value) not in (int, float) or not isfinite(value) for value in region):
        raise ValueError("Передай рамку [x, y, width, height] из четырёх чисел.")
    x, y, width, height = region
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Начало рамки должно быть неотрицательным, размеры — положительными.")
    if coordinate_space == "normalized" and max(x + width, y + height) > 1.00001:
        raise ValueError("Нормализованная рамка должна находиться внутри [0, 1].")
    if coordinate_space == "pixel" and (
            max(region) > 8192 or any(int(value) != value for value in region)):
        raise ValueError("Пиксельная рамка должна содержать целые координаты до 8192.")
    if at_ms is not None and (type(at_ms) is not int or not 0 <= at_ms < MAX_TIME_MS):
        raise ValueError("Таймкод должен приходиться на первые два часа видео.")
    if type(max_side) is not int or not 64 <= max_side <= 1920:
        raise ValueError("Максимальная сторона PNG должна быть от 64 до 1920 пикселей.")


def pixel_region(region, coordinate_space, image_size):
    """Resolve a region in the same oriented coordinates as screenshot and graph recognition."""
    x, y, width, height = region
    frame_width, frame_height = image_size
    if coordinate_space == "normalized":
        left, top = round(x * frame_width), round(y * frame_height)
        right, bottom = round((x + width) * frame_width), round((y + height) * frame_height)
    else:
        left, top, right, bottom = map(int, (x, y, x + width, y + height))
    if not 0 <= left < right <= frame_width or not 0 <= top < bottom <= frame_height:
        raise ValueError("Рамка выходит за границы кадра или меньше одного пикселя.")
    return left, top, right - left, bottom - top


def encode_png(source, filters=None):
    """Decode a single staged frame with the same local-only FFmpeg policy as other media tools."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
               "-format_whitelist", "png_pipe,jpeg_pipe", "-max_pixels", "67108864",
               "-threads", "2", "-i", str(source), "-frames:v", "1", "-an", "-sn"]
    if filters:
        command.extend(["-filter_threads", "1", "-vf", filters])
    command.extend(["-pix_fmt", "rgba", "-c:v", "png", "-threads:v", "1", "-f", "image2pipe", "pipe:1"])
    return subprocess.run(command, capture_output=True, check=True, timeout=15).stdout


def crop_screen(source, region, coordinate_space, at_ms, max_side, screenshot_bytes):
    """Return a crop, its actual source pixel box, and row-independent output dimensions."""
    crop_parameters(region, coordinate_space, at_ms, max_side)
    source = Path(source).expanduser()
    if not source.is_absolute():
        raise ValueError("Укажи абсолютный путь к локальному исходнику.")
    try:
        with temporary_directory(prefix="voice-screen-crop-") as directory:
            root = Path(directory)
            original = root / "oriented.png"
            if at_ms is None:
                image = root / "screenshot"
                image.write_bytes(screenshot_bytes(source))
                verify_image(image)
                original.write_bytes(encode_png(image))
                with original.open("rb") as stream:
                    header = stream.read(24)
            else:
                if not source.is_file():
                    raise ValueError("Исходное видео недоступно.")
                header = extract_frame(source.resolve(), root, at_ms, 8192)[:24]
                original = root / "frame.png"
            size = image_dimensions(header)
            x, y, width, height = box = pixel_region(region, coordinate_space, size)
            filters = f"format=rgba,crop={width}:{height}:{x}:{y}:exact=1"
            if max(width, height) > max_side:
                output_width = max(1, round(width * max_side / max(width, height)))
                output_height = max(1, round(height * max_side / max(width, height)))
                filters += f",scale={output_width}:{output_height}"
            png = encode_png(original, filters)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось вырезать область из локального исходника.") from exc
    return {"pixelBox": box, "sourceSize": size, "imageSize": image_dimensions(png), "timeMs": at_ms}, png
