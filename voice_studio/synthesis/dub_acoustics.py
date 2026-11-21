"""Conservative post-fit timing helpers for dubbing with a fixed source clock."""

from array import array
from math import cos, isfinite, log10, log1p, pi, sqrt
from pathlib import Path
from statistics import mean, median
import sys

from voice_paths import temporary_directory
from voice_studio.audio.pauses import quiet_spans
from voice_studio.audio.speech_intonation import python_scores
from voice_studio.synthesis.dub_timing import speech_edges
from voice_studio.synthesis.prosody_alignment import RATE, fit_phrase, read_pcm, write_pcm

STEP_MS = 10
SAMPLES_PER_MS = RATE // 1000


def envelope(pcm):
    """
    Measure centered 20 ms energy windows every 10 ms.

    pcm is little-endian mono 24 kHz PCM16. The return values are log RMS
    samples indexed in STEP_MS blocks, with partial windows at clip edges.
    """
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    stride = STEP_MS * SAMPLES_PER_MS
    result = []
    for point in range(0, len(samples), stride):
        window = samples[max(0, point - stride):point + stride]
        result.append(log1p(sqrt(mean(sample * sample for sample in window))))
    return result


def correlate_aperture(points, levels, delay_ms):
    """
    Return a correlation proxy between measured aperture and shifted audio energy.

    points contains (cue-relative time in ms, aperture) pairs from a visible
    mouth track. levels is an envelope indexed in STEP_MS blocks. Positive
    delay_ms moves the audio signal later. Missing or flat evidence returns None.
    """
    audio, mouth = [], []
    for at_ms, aperture in points:
        index = round((at_ms - delay_ms) / STEP_MS)
        if 0 <= index < len(levels):
            audio.append(levels[index])
            mouth.append(aperture)
    if len(audio) < 8:
        return None
    ac, mc = mean(audio), mean(mouth)
    xa = [value - ac for value in audio]
    xm = [value - mc for value in mouth]
    norm = sqrt(sum(value * value for value in xa) * sum(value * value for value in xm))
    return sum(a * b for a, b in zip(xa, xm, strict=True)) / norm if norm > 0 else None


def shift_in_slot(pcm, delay_ms):
    """
    Shift a PCM phrase without changing length or discarding a nonzero sample.

    A positive delay_ms pads the start. A negative value removes leading
    zeros. Reject insufficient zero padding by returning None.
    """
    shift = abs(delay_ms) * SAMPLES_PER_MS * 2
    if not shift:
        return pcm
    if shift >= len(pcm):
        return None
    edge = pcm[-shift:] if delay_ms > 0 else pcm[:shift]
    if any(edge):
        return None
    zeros = bytes(shift)
    return zeros + pcm[:-shift] if delay_ms > 0 else pcm[shift:] + zeros


def mouth_shift(source, candidate, points, maximum_ms=100):
    """
    Propose a limited timing shift where the original has measurable correspondence.

    source and candidate are equal-length mono PCM clips for one cue. points
    contains a time-ordered (cue-relative ms, aperture) track from a verified
    visible speaker. maximum_ms caps the search at 100 ms. Return unchanged
    candidate and None on insufficient evidence, otherwise shifted PCM and
    scores on alternating held-out frames. This measures an acoustic proxy.
    """
    if len(source) != len(candidate) or len(points) < 18 or not 10 <= maximum_ms <= 100:
        return candidate, None
    original, current = envelope(source), envelope(candidate)
    # Keep the same frames in every comparison. Boundary frames would otherwise
    # disappear from the correlation at one of the extreme offsets.
    last_ms = (len(original) - 1) * STEP_MS
    middle = [(at, aperture) for at, aperture in points if maximum_ms <= at <= last_ms - maximum_ms]
    train, holdout = middle[::2], middle[1::2]
    if len(holdout) < 8:
        return candidate, None
    source_supported = correlate_aperture(holdout, original, 0)
    if source_supported is None or source_supported < .30:
        return candidate, None
    best, baseline = 0, correlate_aperture(train, current, 0)
    if baseline is None:
        return candidate, None
    best_score = baseline
    bound = maximum_ms // STEP_MS * STEP_MS
    for delay in range(-bound, bound + 1, STEP_MS):
        if not delay or shift_in_slot(candidate, delay) is None:
            continue
        score = correlate_aperture(train, current, delay)
        if score is not None and score - abs(delay) * .0005 > best_score:
            best, best_score = delay, score - abs(delay) * .0005
    before = correlate_aperture(holdout, current, 0)
    after = correlate_aperture(holdout, current, best)
    if not best or before is None or after is None or after < .28 or after - before < .07:
        return candidate, None
    return shift_in_slot(candidate, best), {
        "delayMs": best, "reference": round(source_supported, 4),
        "before": round(before, 4), "after": round(after, 4), "heldOutFrames": len(holdout),
    }


def pause_bounds(path, minimum_ms, minimum_side_ms=220):
    """
    Return internal quiet intervals in samples away from both audible speech edges.

    path is mono PCM16 at 24 kHz. minimum_ms controls the required quiet
    duration, and minimum_side_ms retains room for speech on both sides.
    """
    pcm = read_pcm(path)
    first, last = speech_edges(pcm, RATE)
    margin = minimum_side_ms * SAMPLES_PER_MS
    spans = []
    for gap in quiet_spans(path, minimum_ms):
        left, right = (round(gap[key] * RATE) for key in ("start", "end"))
        if first + margin <= left < right <= last - margin:
            spans.append((left, right))
    return pcm, first, last, spans


def transfer_pause(source_path, candidate_path):
    """
    Transfer one shared interior pause inside an equal-length dub slot.

    source_path and candidate_path contain original separated speech and
    already fitted generated speech on the same 24 kHz clock. Both must have
    a plausible matching quiet interval. Place the original silence exactly
    and fit adjacent speech within 0.84–1.20x. Return candidate and None if
    the match is ambiguous, natural bounds fail, or the change is under 40 ms.
    """
    source, source_first, source_last, original = pause_bounds(source_path, 120)
    candidate, _, _, generated = pause_bounds(candidate_path, 80)
    samples = array("h", candidate)
    first = next((index for index, value in enumerate(samples) if value), 0)
    last = len(samples) - next((index for index, value in enumerate(reversed(samples)) if value), 0)
    if len(source) != len(candidate) or len(original) != 1 or not generated:
        return candidate, None
    start, stop = original[0]
    offset = (start + stop) / len(source)
    ranked = sorted((abs((a + b) / len(candidate) - offset), a, b) for a, b in generated)
    distance, left, right = ranked[0]
    if distance > .18 or (len(ranked) > 1 and ranked[1][0] - distance < .08):
        return candidate, None
    if max(abs(start - left), abs(stop - right)) < 40 * SAMPLES_PER_MS:
        return candidate, None
    fitted = []
    with temporary_directory(prefix="dub-acoustics-") as temporary:
        parts = ((candidate[first * 2:left * 2], start - source_first, 30),
                 (candidate[right * 2:last * 2], source_last - stop, 170))
        for part, target, maximum_padding in parts:
            # A final clause may finish slightly early if stretching it would sound forced.
            implied = len(part) / (2 * target)
            if not .70 <= implied <= 1.2:
                return candidate, None
            try:
                audio, speed, padding = fit_phrase(part, target, Path(temporary), .84, 1.2)
            except ValueError:
                return candidate, None
            if padding > maximum_padding * SAMPLES_PER_MS:
                return candidate, None
            fitted.append((audio, speed))
    spaced = bytearray(len(candidate))
    spaced[source_first * 2:start * 2] = fitted[0][0]
    spaced[stop * 2:source_last * 2] = fitted[1][0]
    return bytes(spaced), {
        "sourcePauseMs": [round(start / SAMPLES_PER_MS), round(stop / SAMPLES_PER_MS)],
        "priorPauseMs": [round(left / SAMPLES_PER_MS), round(right / SAMPLES_PER_MS)],
        "tempo": [round(result[1], 4) for result in fitted],
    }


def phonation_profile(pcm):
    """
    Measure periodicity and low-register evidence on substantially voiced frames.

    pcm is one mono 24 kHz PCM16 phrase without a music bed. Return robust
    statistics of normalized autocorrelation peaks (a relative HNR proxy),
    or None when too few frames are usable. Compare the same speaker across
    recordings. This descriptor can also vary with noise and microphone setup.
    """
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    down = [mean(samples[index:index + 6]) for index in range(0, len(samples), 6)]
    windows = [down[index:index + 240] for index in range(0, len(down) - 239, 100)]
    if not windows:
        return None
    energy = [mean(value * value for value in window) for window in windows]
    active = max(90 ** 2, max(energy) * .018)
    harmonic, frequency = [], []
    for power, window in zip(energy, windows, strict=True):
        if power < active:
            continue
        average = mean(window)
        scores = python_scores([value - average for value in window], 9, 100)
        peaks = [(score, offset + 9) for offset, score in enumerate(scores[1:-1], 1)
                 if scores[offset - 1] < score >= scores[offset + 1]]
        if not peaks or max(peaks)[0] < .52:
            continue
        threshold = max(.52, max(peaks)[0] * .88)
        peak, lag = next((value, index) for value, index in peaks if value >= threshold)
        peak = min(.999, peak)
        harmonic.append(10 * log10(peak / (1 - peak)))
        frequency.append(4000 / lag)
    if len(harmonic) < 9:
        return None
    return {"hnrProxyDb": round(median(harmonic), 2),
            "lowerHnrProxyDb": round(sorted(harmonic)[len(harmonic) // 4], 2),
            "medianF0Hz": round(median(frequency), 1),
            "frames": len(harmonic)}


def periodic_grain(pcm, reference, depth):
    """
    Add bounded alternate-cycle amplitude variation to an overly smooth take.

    pcm and reference are separate mono 24 kHz PCM16 cue clips. depth is the
    maximum gain variation, between 0 and .25. Require measured voiced
    evidence with more aperiodicity in the reference. The input wording,
    duration and precise zero samples stay fixed. This approximates one
    aspect of rough phonation; acceptance belongs to the caller's A/B gate.
    """
    if not isfinite(depth) or not 0 < depth <= .25:
        raise ValueError("Grain depth must be above0 and at most .25.")
    target, baseline = phonation_profile(reference), phonation_profile(pcm)
    if not target or not baseline or target["lowerHnrProxyDb"] >= baseline["lowerHnrProxyDb"] - .35:
        return pcm
    signals = array("h", pcm)
    if sys.byteorder != "little":
        signals.byteswap()
    stride = 10 * SAMPLES_PER_MS
    levels = envelope(pcm)
    maximum = max(levels)
    peak = max(abs(value) for value in signals)
    depth = max(0, min(depth, 32767 / peak - 1))
    if not depth:
        return pcm
    # Use the subharmonic of the generated register. A quiet consonant, pause
    # or edge should keep its own texture and settle to unity gain.
    phase_step = pi * baseline["medianF0Hz"] / RATE
    for index, signal in enumerate(signals):
        cell, remainder = divmod(index, stride)
        position = remainder / stride
        level = (1 - position) * levels[cell] + position * levels[min(cell + 1, len(levels) - 1)]
        confidence = max(0, min(1, (level - maximum + 2.5) / 2))
        edge = min(1, index / 240, (len(signals) - 1 - index) / 240)
        gain = 1 + depth * confidence * edge * cos(index * phase_step)
        signals[index] = max(-32768, min(32767, round(signal * gain)))
    if sys.byteorder != "little":
        signals.byteswap()
    return signals.tobytes()
