"""Bake timed face boxes, shot numbers and captions into a local MP4."""

import json
from math import isfinite
import re
import subprocess
from threading import BoundedSemaphore

from voice_studio.synthesis.engines import ffmpeg
from voice_studio.speech.media_transcript import FORMATS, MAX_MEDIA_SECONDS
from voice_studio.speech.speech_overlap import mixed_label

MAX_METADATA = 12 * 1024 * 1024
EXPORT_SLOT = BoundedSemaphore(1)
FONT = "DejaVu Sans"


def timestamp(ms):
    """Return an ASS timestamp with centisecond precision."""
    seconds, hundredths = divmod(ms // 10, 100)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes // 60}:{minutes % 60:02}:{seconds:02}.{hundredths:02}"


def safe_text(value):
    """Protect the subtitle renderer from markup embedded in a transcript."""
    return " ".join(value.split()).replace("\\", "＼").replace("{", "｛").replace("}", "｝")[:650]


def dimensions(source):
    """Read the geometry after FFmpeg's automatic display rotation."""
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
            "-select_streams", "v:0", "-show_entries",
            "stream=width,height:stream_side_data=rotation:format=duration", "-of", "json", str(source),
        ], timeout=8)
        info = json.loads(raw)
        stream = info["streams"][0]
        width, height = stream["width"], stream["height"]
        rotation = next((entry["rotation"] for entry in stream.get("side_data_list", []) if "rotation" in entry), 0)
        if rotation % 180:
            width, height = height, width
        seconds = float(info["format"]["duration"])
        if not 1 <= width <= 8192 or not 1 <= height <= 8192 or not 0 < seconds <= MAX_MEDIA_SECONDS:
            raise ValueError
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        raise ValueError("Для экспорта нужно видео длительностью до 15 минут.") from exc
    scale = min(1, 1920 / width, 1920 / height)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def ass_header(width, height):
    """Set subtitle sizing and coordinate space for portrait and landscape."""
    size = max(16, round(min(width, height) * .043))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT},{size},&H00E8F1F5,&H00FFFFFF,&H20100D0A,&H20100D0A,-1,0,0,0,100,100,0,0,3,3,0,2,18,18,{size},1
Style: Label,{FONT},{max(12, size // 2)},&H00E8F1F5,&H00FFFFFF,&H00100D0A,&H00100D0A,-1,0,0,0,100,100,0,0,3,2,0,7,0,0,0,1
Style: Box,{FONT},12,&H00E8F1F5,&H00FFFFFF,&H00E8F1F5,&H00100D0A,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def event(layer, start, end, style, text):
    """Format one validated ASS event."""
    return f"Dialogue: {layer},{timestamp(start)},{timestamp(end)},{style},,0,0,0,,{text}\n"


def intervals(rows):
    """Yield bounded, ordered millisecond ranges from analyser metadata."""
    last = -1
    for row in rows:
        start, end = row["startMs"], row["endMs"]
        if (type(start) is not int or type(end) is not int
                or not max(0, last) <= start < end <= MAX_MEDIA_SECONDS * 1000 + 1000):
            raise ValueError("Некорректные отметки времени технического слоя.")
        last = start
        yield start, end, row


def box_events(row, width, height, step, start, active, mapping, *, body=False, language="ru"):
    """Emit a hollow rectangle and a label for a single normalized face hit."""
    track, x, y, w, h, _, *identity = row
    if not all(type(v) in (int, float) and isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
        raise ValueError("Некорректные координаты лица.")
    face = identity[0] if identity else track
    bound = mapping.get(str(face), {})
    speaker = bound.get("speaker")
    speaker_name, face_name = ("Speaker", "Face") if language == "en" else ("Спикер", "Лицо")
    label = f"{speaker_name} {speaker}" if type(speaker) is int and bound.get("basis") != "visible-only" else f"{face_name} {face}"
    x, y, w, h = round(x * width), round(y * height), round(w * width), round(h * height)
    w, h = min(w, width - x), min(h, height - y)
    border = max(2, min(width, height) // 250)
    if min(w, h) < border * 3:
        return
    fill = "DBDD70" if body else "4EB3FF" if active else "E8F1F5"
    outer = f"m 0 0 l {w} 0 {w} {h} 0 {h} 0 0"
    inner = f"m {border} {border} l {border} {h-border} {w-border} {h-border} {w-border} {border} {border} {border}"
    drawing = rf"{{\an7\pos({x},{y})\1c&H{fill}&\p1}}"
    yield event(0, start, start + step, "Box", f"{drawing}{outer} {inner}")
    if body:
        return
    label_position = rf"{{\an7\pos({min(x, width - 96)},{max(0, y - max(16, min(width, height) // 40))})\1c&H{fill}&}}"
    yield event(1, start, start + step, "Label", label_position + safe_text(label))


def caption_windows(start, end, cue):
    """Use the mixed caption only during the actual overlapping speech intervals."""
    if not mixed_label(cue) or not cue.get("overlaps"):
        yield start, end, cue
        return
    overlaps = []
    for span in cue["overlaps"]:
        left, right = max(start, span["startMs"]), min(end, span["endMs"])
        if left < right:
            overlaps.append({**span, "startMs": left, "endMs": right})
    bounds = sorted({start, end, *(value for span in overlaps for value in (span["startMs"], span["endMs"]))})
    for left, right in zip(bounds, bounds[1:]):
        active = [span for span in overlaps if span["startMs"] <= left < span["endMs"]]
        yield left, right, {**cue, "overlap": bool(active), "overlaps": active}


def write_overlay(layer, destination, width, height, language="ru"):
    """Write bounded editor markers, using the same active-track decisions as the preview."""
    cues = list(intervals(layer.get("segments", [])))
    frames = layer.get("frames", [])
    step = layer.get("stepMs", 40)
    if type(step) is not int or not 20 <= step <= 1000 or len(frames) > 3000 or len(cues) > 6000:
        raise ValueError("Технический слой слишком большой.")
    mapping = layer.get("faceSpeakers", {})
    active_tracks = layer.get("activeTracks", [])
    bodies = layer.get("bodies", [])
    if len(active_tracks) != len(frames) or (bodies and len(bodies) != len(frames)):
        raise ValueError("Не совпадают кадры и активность лиц.")
    with destination.open("w", encoding="utf-8") as output:
        output.write(ass_header(width, height))
        for index, (hits, active) in enumerate(zip(frames, active_tracks)):
            if len(hits) > 24 or not isinstance(active, list):
                raise ValueError("Слишком много лиц в кадре.")
            boxes = bodies[index] if bodies else []
            if not isinstance(boxes, list) or len(boxes) > 4:
                raise ValueError("Некорректные рамки фигур.")
            for box in boxes:
                output.writelines(box_events([0, *box, 0], width, height, step, index * step, False, {}, body=True))
            for hit in hits:
                output.writelines(box_events(hit, width, height, step, index * step, hit[0] in active, mapping,
                                              language=language))
        speaker_name = "Speaker" if language == "en" else "Спикер"
        for start, end, row in cues:
            for left, right, cue in caption_windows(start, end, row):
                number = cue.get("number")
                prefix = mixed_label(cue, language) or cue.get("speakerName")
                if not prefix and type(number) is int:
                    prefix = f"{speaker_name} {number}"
                text = f"{prefix} · {cue['text']}" if prefix else cue["text"]
                if timestamp(left) != timestamp(right):
                    output.write(event(2, left, right, "Caption", safe_text(text)))
        for index, shot in enumerate(layer.get("shots", [])[:3000], 1):
            if type(shot) is int and 0 <= shot < MAX_MEDIA_SECONDS * 1000:
                shot_name = "Shot" if language == "en" else "План"
                marker = rf"{{\an7\pos(12,12)}}{shot_name} {index}"
                output.write(event(1, shot, shot + 600, "Label", marker))


def preview_color(value):
    """Translate the preview's fixed CSS colors into ASS color and alpha tags."""
    if not isinstance(value, str) or len(value) > 48:
        raise ValueError("Некорректный цвет технического слоя.")
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        red, green, blue = [int(value[index:index + 2], 16) for index in (1, 3, 5)]
        alpha = 255
    else:
        match = re.fullmatch(r"rgba\((\d{1,3}), *(\d{1,3}), *(\d{1,3}), *([01]?(?:\.\d+)?)\)", value)
        if not match:
            raise ValueError("Некорректный цвет технического слоя.")
        red, green, blue = map(int, match.groups()[:3])
        alpha = round(float(match[4]) * 255)
        if max(red, green, blue, alpha) > 255:
            raise ValueError("Некорректный цвет технического слоя.")
    return rf"\1c&H{blue:02X}{green:02X}{red:02X}&\1a&H{255 - alpha:02X}&"


def rectangle_vector(x, y, w, h, *, reverse=False):
    """Emit an ASS polygon on the preview's coordinate scale."""
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if reverse:
        points.reverse()
    origin, *remaining = points
    return f"m {round(origin[0])} {round(origin[1])} l " + " ".join(
        f"{round(px)} {round(py)}" for px, py in [*remaining, origin])


def dashed_edge(x, y, length, thickness, dash, *, vertical=False):
    """Render a dashed side in the same reference pixels as the canvas."""
    at = 0
    while at < length:
        part = min(dash[0], length - at)
        if vertical:
            yield rectangle_vector(x - thickness / 2, y + at, thickness, part)
        else:
            yield rectangle_vector(x + at, y - thickness / 2, part, thickness)
        at += sum(dash)


def preview_stroke(x, y, w, h, line_width, dash):
    """Convert a centered canvas stroke to a filled vector, including dashed bodies."""
    if dash:
        return " ".join([
            *dashed_edge(x, y, w, line_width, dash),
            *dashed_edge(x, y + h, w, line_width, dash),
            *dashed_edge(x, y, h, line_width, dash, vertical=True),
            *dashed_edge(x + w, y, h, line_width, dash, vertical=True),
        ])
    half = line_width / 2
    outer = rectangle_vector(x - half, y - half, w + line_width, h + line_width)
    if min(w, h) > line_width:
        inner = rectangle_vector(x + half, y + half, w - line_width, h - line_width, reverse=True)
        return f"{outer} {inner}"
    return outer


def bounded_coordinates(values, limit):
    """Reject arbitrary commands, NaN and coordinates far outside the reference frame."""
    if not all(type(value) in (int, float) and isfinite(value) and abs(value) <= limit for value in values):
        raise ValueError("Некорректные координаты предпросмотра.")


def preview_event(command, start, end, limit):
    """Compile one validated paint command. Never accept raw client ASS."""
    if not isinstance(command, list) or len(command) not in (6, 8):
        raise ValueError("Некорректная команда предпросмотра.")
    kind, color, x, y, *arguments = command
    tint = preview_color(color)
    bounded_coordinates((x, y), limit)
    if kind == "text" and len(command) == 6:
        size, text = arguments
        if type(size) is not int or not 8 <= size <= 45 or not isinstance(text, str):
            raise ValueError("Некорректная подпись предпросмотра.")
        text_style = rf"{{\an7\pos({round(x)},{round(y - size)})\fs{size}\b1\q2{tint}}}"
        return event(0, start, end, "Box", text_style + safe_text(text))
    if kind not in ("fill", "stroke") or (kind == "fill") != (len(command) == 6):
        raise ValueError("Неизвестная команда предпросмотра.")
    w, h, *stroke = arguments
    bounded_coordinates((w, h), limit)
    if min(w, h) <= 0:
        return ""
    if stroke:
        line_width, dash = stroke
        if type(line_width) not in (int, float) or not 0 < line_width <= 8 or dash not in ([], [7, 4], [5, 3]):
            raise ValueError("Некорректная рамка предпросмотра.")
        drawing = preview_stroke(x, y, w, h, line_width, dash)
    else:
        drawing = rectangle_vector(x, y, w, h)
    return event(0, start, end, "Box", rf"{{\an7\pos(0,0){tint}\p1}}" + drawing)


def write_preview_overlay(plan, destination):
    """Compile the browser's shared paint plan at its original preview dimensions."""
    width, height = plan.get("width"), plan.get("height")
    samples = plan.get("samples")
    if (type(width) is not int or type(height) is not int or not 1 <= min(width, height)
            or max(width, height) > 8192 or not isinstance(samples, list) or len(samples) > 20_000):
        raise ValueError("Некорректный план технического слоя.")
    last_end = 0
    commands_total = 0
    written = 0
    with destination.open("w", encoding="utf-8") as output:
        output.write(ass_header(width, height))
        for start, end, commands in samples:
            if (type(start) is not int or type(end) is not int
                    or not last_end <= start < end <= MAX_MEDIA_SECONDS * 1000 + 1000
                    or not isinstance(commands, list) or len(commands) > 256):
                raise ValueError("Некорректные интервалы предпросмотра.")
            last_end = end
            commands_total += len(commands)
            if commands_total > 220_000:
                raise ValueError("Технический слой слишком большой.")
            if timestamp(start) != timestamp(end):
                for command in commands:
                    row = preview_event(command, start, end, max(width, height) * 2)
                    written += len(row)
                    if written > 48 * 1024 * 1024:
                        raise ValueError("Технический слой слишком большой.")
                    output.write(row)


def export_video(source, layer, destination, language="ru"):
    """Render original video and optional audio once, limiting concurrent exports."""
    if not EXPORT_SLOT.acquire(blocking=False):
        raise ValueError("Уже выгружается другое видео. Дождись завершения.")
    try:
        width, height = dimensions(source)
        try:
            overlay = source.parent / "technical.ass"
            if layer.get("format") == "preview-v1":
                write_preview_overlay(layer, overlay)
            else:
                write_overlay(layer, overlay, width, height, language)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError("Некорректные данные технического слоя.") from exc
        try:
            ffmpeg([
                "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "-1", "-map_chapters", "-1",
                "-vf", f"scale={width}:{height},setsar=1,ass=technical.ass", "-filter_threads", "2",
                "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "21",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                str(destination),
            ], timeout=360, cwd=source.parent)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            raise ValueError("FFmpeg не смог записать MP4 с техническим слоем.") from exc
    finally:
        EXPORT_SLOT.release()
