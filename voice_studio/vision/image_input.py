"""Shared bounds and decoder validation for browser image uploads and MCP screenshots."""

import json
import subprocess
from threading import BoundedSemaphore

MAX_SCREENSHOT_BYTES = 24 * 1024 * 1024
MAX_SCREENSHOT_PIXELS = 24_000_000
IMAGE_SLOT = BoundedSemaphore(1)


def verify_image(path):
    """Reject oversized, streaming, or non-image input before decoding pixels."""
    try:
        with path.open("rb") as source:
            signature = source.read(8)
        if signature == b"\x89PNG\r\n\x1a\n":
            image_format = "png_pipe"
        elif signature.startswith(b"\xff\xd8\xff"):
            image_format = "jpeg_pipe"
        else:
            raise ValueError("Unsupported image signature")
        # A .jpg extension otherwise selects image2, which can open numbered file sequences.
        # Force a single-image demuxer selected from actual bytes, never the browser MIME or filename.
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file",
             "-f", image_format, "-format_whitelist", "png_pipe,jpeg_pipe", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path)],
            stderr=subprocess.DEVNULL, timeout=8,
        )
        stream = json.loads(raw)["streams"][0]
        width, height = stream["width"], stream["height"]
    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as exc:
        raise ValueError("Нужен локальный скриншот PNG или JPEG.") from exc
    if (stream.get("codec_name") not in {"png", "mjpeg"}
            or min(width, height) < 16 or max(width, height) > 8192 or width * height > MAX_SCREENSHOT_PIXELS):
        raise ValueError("Размеры скриншота выходят за допустимые пределы.")

