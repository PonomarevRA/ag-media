"""Measure representative colors and caption contrast from source video pixels."""

from collections import defaultdict
import subprocess

PALETTE_WIDTH = 40
PALETTE_HEIGHT = 30
CAPTION_COLUMNS = range(6, 34)
CAPTION_ROWS = range(21, 27)
TEXT_CONTRAST = 4.5


def mean_rgb(samples):
    """Average equally weighted RGB samples, keeping fractional channels until rendering."""
    return tuple(sum(channel) / len(samples) for channel in zip(*samples))


def hex_rgb(rgb):
    """Serialize RGB channels as a CSS-safe six-digit sRGB color."""
    return "#" + "".join(f"{round(channel):02X}" for channel in rgb)


def linear(channel):
    """Convert an eight-bit sRGB channel to linear light for relative luminance."""
    value = channel / 255
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4


LINEAR = tuple(linear(channel) for channel in range(256))


def caption_colors(pixels):
    """Choose black or white based on the least favorable sampled tile in the caption area."""
    luminances = []
    for y in CAPTION_ROWS:
        for x in CAPTION_COLUMNS:
            red, green, blue = pixels[y * PALETTE_WIDTH + x]
            luminances.append(.2126 * LINEAR[red] + .7152 * LINEAR[green] + .0722 * LINEAR[blue])
    white = 1.05 / (max(luminances) + .05)
    black = (min(luminances) + .05) / .05
    foreground, opposite = ("#FFFFFF", "#000000") if white >= black else ("#000000", "#FFFFFF")
    ratio = max(white, black)
    region = [CAPTION_COLUMNS.start / PALETTE_WIDTH, CAPTION_ROWS.start / PALETTE_HEIGHT,
              len(CAPTION_COLUMNS) / PALETTE_WIDTH, len(CAPTION_ROWS) / PALETTE_HEIGHT]
    return {"region": region, "foreground": foreground, "minSampledContrast": round(ratio, 3),
            "backdrop": opposite if ratio < TEXT_CONTRAST else None}


def frame_colors(png):
    """Return a compact palette and lower-third guidance from the exact undecorated PNG."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "pipe", "-f", "image2pipe",
         "-threads", "1", "-i", "pipe:0", "-vf", f"scale={PALETTE_WIDTH}:{PALETTE_HEIGHT}:flags=area",
         "-frames:v", "1", "-pix_fmt", "rgb24", "-threads:v", "1", "-f", "rawvideo", "pipe:1"],
        input=png, capture_output=True, check=True, timeout=5,
    ).stdout
    if len(raw) != PALETTE_WIDTH * PALETTE_HEIGHT * 3:
        raise ValueError("Could not measure frame colors.")
    pixels = list(zip(raw[::3], raw[1::3], raw[2::3]))
    buckets = defaultdict(list)
    for pixel in pixels:
        buckets[tuple((channel + 16) // 32 for channel in pixel)].append(pixel)
    dominant = sorted(buckets.values(), key=len, reverse=True)[:5]
    return {"average": hex_rgb(mean_rgb(pixels)),
            "colors": [{"hex": hex_rgb(mean_rgb(bucket)), "share": round(len(bucket) / len(pixels), 3)}
                       for bucket in dominant],
            "caption": caption_colors(pixels)}
