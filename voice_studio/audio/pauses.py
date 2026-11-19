"""Measure quiet spans and suggest punctuation at recognized boundaries."""

from array import array
from math import log10
import sys
import wave

MIN_PAUSE_MS = 350
LONG_PAUSE_MS = 800
WINDOW = 0.02


def quiet_spans(path, min_duration_ms=MIN_PAUSE_MS):
    """Detect internal quiet intervals at or above a minimum duration in milliseconds."""
    with wave.open(str(path)) as wav:
        rate = wav.getframerate()
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    width = max(1, round(rate * WINDOW))
    levels = []
    for start in range(0, len(samples), width):
        window = samples[start:start + width]
        mean_power = sum(value * value for value in window) / len(window)
        levels.append(10 * log10(max(1, mean_power) / 32768**2))
    if not levels or max(levels) < -50:
        return []
    sorted_levels = sorted(levels)
    threshold = max(-55, min(-32, sorted_levels[int(0.9 * (len(levels) - 1))] - 20))
    pauses = []
    silence_start = None
    # Ignore leading quiet and quiet that never returns to speech.
    heard_signal = False
    for index, level in enumerate(levels):
        if level <= threshold:
            if heard_signal and silence_start is None:
                silence_start = index
            continue
        if silence_start is not None:
            start = silence_start * width / rate
            end = index * width / rate
            if round((end - start) * 1000) >= min_duration_ms:
                pauses.append({"start": round(start, 3), "end": round(end, 3)})
            silence_start = None
        heard_signal = True
    return pauses


def boundary_pause(left, right, pauses):
    """Find measured silence close to the neighboring Whisper segment edges."""
    offsets = (left.get("offsets") or {}, right.get("offsets") or {})
    stop, start = offsets[0].get("to"), offsets[1].get("from")
    if not isinstance(stop, (int, float)) or not isinstance(start, (int, float)):
        return None
    stop, start = stop / 1000, start / 1000
    for pause in pauses:
        if abs(pause["start"] - stop) <= 0.3 and abs(pause["end"] - start) <= 0.3:
            return pause
    return None


def annotate(segments, pauses):
    """Keep ASR text intact and return a separate optional punctuation draft."""
    segments = [segment for segment in segments if segment["text"].strip()]
    phrases = [segment["text"].strip() for segment in segments]
    draft = phrases.copy()
    for index in range(len(draft) - 1):
        left, right = phrases[index:index + 2]
        if not left[-1].isalpha() or not right[0].isalpha():
            continue
        pause = boundary_pause(segments[index], segments[index + 1], pauses)
        if not pause:
            continue
        if round((pause["end"] - pause["start"]) * 1000) >= LONG_PAUSE_MS:
            draft[index] += "."
            draft[index + 1] = right[0].upper() + right[1:]
        elif right[0].islower():
            draft[index] += ","
    return {"text": " ".join(phrases), "suggestion": " ".join(draft), "pauses": pauses}
