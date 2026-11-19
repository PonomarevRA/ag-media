"""Estimate phrase-level pitch contours from the original speech PCM."""

from array import array
from math import log2, sqrt
import os
from statistics import median
import sys
import wave

from voice_studio.audio import pitch_vdsp

FRAME_MS = 45
ANALYSIS_RATE = 4000
MIN_VOICED = 3


def python_scores(frame, first_lag, last_lag):
    """Return normalized autocorrelation using the portable reference implementation."""
    scores = []
    for lag in range(first_lag, last_lag + 1):
        left, right = frame[:-lag], frame[lag:]
        denominator = sqrt(sum(x * x for x in left) * sum(y * y for y in right))
        scores.append(sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0)
    return scores


def estimate_pitch(samples, rate, score=python_scores):
    """Find the first strong normalized autocorrelation peak in the speech range."""
    mean = sum(samples) / len(samples)
    frame = [sample - mean for sample in samples]
    energy = sum(sample * sample for sample in frame)
    if energy < len(frame) * 90 ** 2:
        return None
    first = max(2, rate // 440)
    last = min(rate // 70, len(frame) // 2)
    scores = score(frame, first - 1, last + 1)
    threshold = max(.68, max(scores) * .91)
    for index in range(1, len(scores) - 1):
        if scores[index] >= threshold and scores[index - 1] < scores[index] >= scores[index + 1]:
            peak = (scores[index - 1] - scores[index + 1]) / (2 * (scores[index - 1] - 2 * scores[index] + scores[index + 1]))
            return rate / (first - 1 + index + peak)
    return None


def sample_zone(audio, center_ms, radius_ms, score=python_scores):
    """Read only a bounded zone and reduce its sample rate by local averaging."""
    rate = audio.getframerate()
    stride = max(1, rate // ANALYSIS_RATE)
    position = max(0, round((center_ms - radius_ms) * rate / 1000))
    if position >= audio.getnframes():
        return []
    audio.setpos(position)
    raw = array("h", audio.readframes(round(2 * radius_ms * rate / 1000)))
    if sys.byteorder != "little":
        raw.byteswap()
    reduced = [sum(raw[index:index + stride]) / stride for index in range(0, len(raw) - stride + 1, stride)]
    effective = rate / stride
    size = round(effective * FRAME_MS / 1000)
    step = round(effective * .025)
    pitches = []
    for position in range(0, len(reduced) - size + 1, step):
        pitch = estimate_pitch(reduced[position:position + size], int(effective), score)
        if pitch:
            pitches.append(pitch)
    if len(pitches) < MIN_VOICED:
        return []
    reference = median(pitches)
    return [pitch for pitch in pitches if abs(log2(pitch / reference)) <= .32]


def measure_intonation(wav, cues, offset_ms, overlaps=()):
    """Compare the early, middle and late voiced zones of each sufficiently clear phrase."""
    result = []
    score = pitch_vdsp.scores if os.getenv("VOICE_PITCH_BACKEND") == "vdsp" and pitch_vdsp.correlate() else python_scores
    with wave.open(str(wav)) as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1 or audio.getframerate() < ANALYSIS_RATE:
            return result
        last_ms = audio.getnframes() * 1000 // audio.getframerate()
        for cue in cues:
            begin, stop = cue["startMs"], cue["endMs"]
            start = max(0, begin - offset_ms)
            end = min(last_ms, stop - offset_ms)
            length = end - start
            if length < 430 or cue.get("overlap") or cue.get("overlaps"):
                continue
            if any(row["startMs"] < stop and row["endMs"] > begin for row in overlaps):
                continue
            radius = min(length * .13, 100)
            zones = [sample_zone(audio, start + length * fraction, radius, score) for fraction in (.2, .5, .8)]
            if min(len(zone) for zone in zones) < MIN_VOICED:
                continue
            if any(max(zone) / min(zone) > 1.42 for zone in zones):
                continue
            first, middle, last = (round(median(zone), 1) for zone in zones)
            rises = 12 * log2(last / first)
            half = 12 * log2(middle / first)
            if abs(rises) > 9 or abs(half) > 9:
                continue
            if half > max(0, rises) + 1.7:
                label = "rise-fall"
            elif half < min(0, rises) - 1.7:
                label = "fall-rise"
            elif rises >= 1.5:
                label = "rising"
            elif rises <= -1.5:
                label = "falling"
            else:
                label = "steady"
            result.append({"kind": "intonation", "label": label, "startMs": begin, "endMs": stop,
                           "basis": "speech-f0", "pitchHz": [first, middle, last],
                           "semitones": round(rises, 2)})
    return result
