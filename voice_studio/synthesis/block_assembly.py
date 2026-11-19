"""Validate time ranges from approved takes and join their normalized PCM."""

from array import array
from contextlib import ExitStack
import sys
import wave

RATE = 24000
MIN_MS = 80
MAX_MS = 90000
FADE_FRAMES = RATE * 6 // 1000


def checked_blocks(blocks, sources):
    """Resolve and validate an ordered set of bounded, manually approved ranges."""
    if not isinstance(blocks, list) or not 2 <= len(blocks) <= 12:
        raise ValueError("Добавь от 2 до 12 фрагментов в нужном порядке.")
    result = []
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("id"), str):
            raise ValueError("Фрагмент недоступен.")
        key = block["id"]
        source = sources.get(key)
        start, end = block.get("startMs"), block.get("endMs")
        if not source:
            raise ValueError("Дубль для фрагмента недоступен.")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end - MIN_MS or end > round(source["seconds"] * 1000)):
            raise ValueError("Проверь границы фрагмента. Минимум — 80 мс.")
        result.append({"id": key, "startMs": start, "endMs": end})
    if sum(part["endMs"] - part["startMs"] for part in result) > MAX_MS:
        raise ValueError("Сборка должна быть не длиннее 90 секунд.")
    return result


def read_block(audio, part):
    """Read the selected frames and taper six milliseconds at both cut points."""
    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, RATE, "NONE"):
        raise ValueError("Формат дубля не подходит для нарезки.")
    limit = audio.getnframes()
    start = min(limit, part["startMs"] * RATE // 1000)
    end = min(limit, part["endMs"] * RATE // 1000)
    audio.setpos(start)
    pcm = array("h", audio.readframes(end - start))
    if sys.byteorder != "little":
        pcm.byteswap()
    frames = min(FADE_FRAMES, len(pcm) // 2)
    for index in range(frames):
        gain = index / frames
        pcm[index] = round(pcm[index] * gain)
        pcm[-1-index] = round(pcm[-1-index] * gain)
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm


def render_blocks(parts, media, output):
    """Render a standalone WAV and return its measured duration."""
    with ExitStack() as stack:
        tracks = {}
        for key in dict.fromkeys(part["id"] for part in parts):
            try:
                tracks[key] = stack.enter_context(wave.open(str(media / f"{key}.wav"), "rb"))
            except (FileNotFoundError, wave.Error, EOFError) as exc:
                raise ValueError("Исходный WAV недоступен.") from exc
        writer = stack.enter_context(wave.open(str(output), "wb"))
        writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for part in parts:
            writer.writeframesraw(read_block(tracks[part["id"]], part))
        return writer.getnframes() / RATE
