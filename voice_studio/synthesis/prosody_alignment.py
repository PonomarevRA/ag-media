"""Acoustic phrase placement and conservative F0 transfer, independent of any TTS model."""

from math import isfinite, log2
from array import array
from pathlib import Path
from statistics import median
import wave

from voice_paths import temporary_directory
from voice_studio.audio.pauses import quiet_spans
from voice_studio.audio.speech_intonation import measure_intonation, sample_zone
from voice_studio.synthesis.engines import ffmpeg

RATE = 24_000


def read_pcm(path):
    """Read bounded mono24kHz/16-bit speech; reject empty, non-PCM and oversized inputs."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("Alignment requires mono24kHz16-bit PCM.")
        if not 0 < audio.getnframes() <= RATE * 90:
            raise ValueError("Alignment accepts up to90 seconds of nonempty PCM.")
        return audio.readframes(audio.getnframes())


def write_pcm(path, pcm):
    """Write one mono24kHz/16-bit result without changing the source samples."""
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)


def reference_rhythm(path, minimum_pause_ms=120):
    """Measure quiet spans and speech islands on the local clip clock; do not invent missing pauses."""
    if type(minimum_pause_ms) is not int or not 80 <= minimum_pause_ms <= 2000:
        raise ValueError("Minimum pause must be80–2000 milliseconds.")
    pcm = read_pcm(path)
    if not any(pcm):
        raise ValueError("A silent reference cannot define speech rhythm.")
    # Only exact zero at source edges is safe to discard; quiet consonants remain in the slot.
    samples = array('h', pcm)
    first = next(index for index, value in enumerate(samples) if value)
    last = len(samples) - next(index for index, value in enumerate(reversed(samples)) if value)
    window = RATE // 50
    first = first // window * window
    last = min(len(samples), (last + window - 1) // window * window)
    pauses, phrases = [], []
    position = first
    for gap in quiet_spans(path, minimum_pause_ms):
        start, end = (round(gap[key] * RATE) for key in ("start", "end"))
        # Short fragments near an edge are not safe standalone synthesis phrases.
        if start - position < RATE * .16 or last - end < RATE * .16:
            continue
        phrases.append({"startSample": position, "endSample": start})
        pauses.append({"startSample": start, "endSample": end, "basis": "pcm-quiet"})
        position = end
    phrases.append({"startSample": position, "endSample": last})
    return {"sampleRate": RATE, "frames": len(pcm) // 2, "phrases": phrases, "pauses": pauses,
            "basis": "acoustic-envelope; not phoneme or visual lip alignment"}


def validate_rhythm(plan):
    """Validate exact integer source-sample slots before allocating or rendering audio."""
    if not isinstance(plan, dict) or plan.get("sampleRate") != RATE:
        raise ValueError("Unsupported rhythm sample rate.")
    frames, phrases = plan.get("frames"), plan.get("phrases")
    if type(frames) is not int or not 0 < frames <= RATE * 90:
        raise ValueError("Invalid reference duration.")
    if not isinstance(phrases, list) or not 1 <= len(phrases) <= 100:
        raise ValueError("Invalid phrase slots.")
    position = 0
    for row in phrases:
        if not isinstance(row, dict):
            raise ValueError("Invalid phrase slot.")
        start, end = row.get("startSample"), row.get("endSample")
        if type(start) is not int or type(end) is not int or not position <= start < end <= frames:
            raise ValueError("Phrase slots overlap or exceed the reference.")
        position = end
    return phrases


def fit_phrase(pcm, target_frames, folder, min_speed, max_speed):
    """Fit speech without cutting a spoken tail; reject slots requiring excessive rate change."""
    if not any(pcm):
        raise ValueError("The generated phrase contains no signal.")
    speech = pcm
    speed = max(min_speed, len(speech) / 2 / target_frames)
    if speed > max_speed:
        raise ValueError(f"Phrase needs tempo{speed:.3f}; shorten/rewrite or regenerate it.")
    source, fitted = folder / "phrase.wav", folder / "fitted.wav"
    write_pcm(source, speech)
    for _ in range(3):
        ffmpeg(["-i", str(source), "-af", f"atempo={speed:.8f}", "-c:a", "pcm_s16le", str(fitted)])
        result = read_pcm(fitted)
        length = len(result) // 2
        if length <= target_frames:
            return result + bytes((target_frames - length) * 2), speed, target_frames - length
        speed *= length / target_frames * 1.002
        if speed > max_speed:
            break
    raise ValueError("Tempo fitting cannot preserve the spoken tail inside this slot.")


def align_phrases(paths, plan, output, *, min_speed=.8, max_speed=1.8):
    """Place one independently synthesized phrase per source island, preserving all measured gaps."""
    phrases = validate_rhythm(plan)
    if not (.5 <= min_speed <= 1 <= max_speed <= 2):
        raise ValueError("Invalid tempo bounds.")
    if len(paths) != len(phrases):
        raise ValueError("Provide one translated phrase for every source speech island.")
    result = bytearray(plan["frames"] * 2)
    audit = []
    # Publish only after every phrase fits; a failed attempt leaves a previous WAV untouched.
    with temporary_directory(prefix="prosody-fit-") as temp:
        for path, slot in zip(paths, phrases, strict=True):
            start, end = slot["startSample"], slot["endSample"]
            fitted, speed, padding = fit_phrase(read_pcm(path), end - start, Path(temp), min_speed, max_speed)
            result[start * 2:end * 2] = fitted
            audit.append({**slot, "tempo": round(speed, 6), "paddingFrames": padding})
    write_pcm(output, result)
    return {"frames": plan["frames"], "sampleRate": RATE, "phrases": audit,
            "insertedQuietFrames": plan["frames"] - sum(x["endSample"] - x["startSample"] for x in phrases)}


def measured_contour(path):
    """Retry sparse three-zone evidence with wider voiced windows, retaining missing-data abstention."""
    raw = read_pcm(path)
    duration = len(raw) * 500 / RATE
    rows = measure_intonation(path, [{"startMs": 0, "endMs": round(duration)}], 0)
    if rows:
        return rows[0]
    if duration < 800:
        return None
    with wave.open(str(path)) as audio:
        zones = [sample_zone(audio, duration * part, min(250, duration * .18)) for part in (.2, .5, .8)]
    if any(len(zone) < 3 or max(zone) / min(zone) > 1.6 for zone in zones):
        return None
    return {"pitchHz": [round(median(zone), 1) for zone in zones], "basis": "speech-f0-wide-zone"}


def centred_corrections(values, bound):
    """Project shifts onto bounded zero-mean corrections so clamping does not shift the register."""
    lower, upper = min(values) - bound, max(values) + bound
    for _ in range(60):
        centre = (lower + upper) / 2
        total = sum(max(-bound, min(bound, value - centre)) for value in values)
        if total > 0:
            lower = centre
        else:
            upper = centre
    centre = (lower + upper) / 2
    return [max(-bound, min(bound, value - centre)) for value in values]


def contour_error(source, candidate):
    """Measure relative three-zone F0 shape error in semitones; missing evidence stays unknown."""
    if not source or not candidate:
        return None
    zones = [[12 * log2(hz) for hz in row['pitchHz']] for row in (source, candidate)]
    centred = [[value - sum(row)/len(row) for value in row] for row in zones]
    return (sum((a-b)**2 for a,b in zip(*centred))/3)**.5


def transfer_intonation(source, candidate, output, *, maximum_semitones=2, rhythm=None):
    """Transfer measured relative F0; preserve explicit rhythm gaps and abstain on filter clock drift."""
    if not isfinite(maximum_semitones) or not 0 < maximum_semitones <= 3:
        raise ValueError("Pitch correction must be above0 and at most3 semitones.")
    pcm = read_pcm(candidate)
    slots = validate_rhythm(rhythm) if rhythm is not None else None
    if rhythm is not None and rhythm['frames'] != len(pcm) // 2:
        raise ValueError("The rhythm and candidate must use the same sample clock.")
    rows = []
    for path in (source, candidate):
        measured = measured_contour(path)
        if not measured:
            return {"status": "skipped", "reason": "insufficient reliable F0 evidence"}
        rows.append(measured)
    target, current = ([12 * log2(hz) for hz in row["pitchHz"]] for row in rows)
    # Relative shape preserves the candidate's pitch centre rather than forcing another register.
    shifts = [(a - sum(target) / 3) - (b - sum(current) / 3) for a, b in zip(target, current)]
    limited = centred_corrections(shifts, maximum_semitones)
    duration = len(pcm) / (2 * RATE)
    commands = []
    for index in range(max(2, round(duration / .05))):
        fraction = index * .05 / duration
        coordinate = max(0, min(2, (fraction - .2) / .3))
        left = min(1, int(coordinate))
        delta = limited[left] + (limited[left + 1] - limited[left]) * (coordinate - left)
        commands.append(f"{index * .05:.3f} rubberband@prosody pitch {2 ** (delta / 12):.8f};")
    with temporary_directory(prefix="prosody-pitch-") as temp:
        folder = Path(temp)
        (folder / "pitch.commands").write_text("\n".join(commands))
        rendered = folder / "pitch.wav"
        ffmpeg(["-i", str(Path(candidate).resolve()), "-af",
                "asendcmd=f=pitch.commands,rubberband@prosody=formant=preserved",
                "-c:a", "pcm_s16le", str(rendered)], cwd=folder)
        result = read_pcm(rendered)
    if len(result) != len(pcm):
        return {"status": "skipped", "reason": "pitch processor changed the sample clock"}
    if slots is not None:
        # A time-frequency filter spreads signal into gaps. Restore only declared source gaps.
        result = bytearray(result)
        position = 0
        for slot in slots:
            start = slot['startSample']
            result[position * 2:start * 2] = bytes((start - position) * 2)
            position = slot['endSample']
        result[position * 2:] = bytes(len(result) - position * 2)
    write_pcm(output, result)
    return {"status": "applied", "source": rows[0], "before": rows[1], "semitoneCorrections": limited,
            "after": measured_contour(output), "clamped": any(abs(a-b) > 1e-6 for a,b in zip(limited,shifts)),
            "frames": len(pcm) // 2,
            "basis": "measured F0 shape; perceptual naturalness requires audition"}
