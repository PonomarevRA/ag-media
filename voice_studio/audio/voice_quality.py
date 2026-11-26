"""Measure dialogue and render an optional restored copy on its original clock."""

from math import isfinite
import subprocess
from typing import Literal
import wave

from voice_studio.audio.audio_processing import band_levels, rms
from voice_studio.synthesis.engines import ffmpeg

WINDOW_MS = 100
MIN_SPEECH_WINDOWS = 12
LEVEL_KEY = "lavfi.astats.Overall.RMS_level"
RestorationMode = Literal["clarity", "quiet-speech", "denoise"]


def window_levels(path):
    """Read RMS in consecutive 100 ms windows without adding synthetic silence."""
    chain = (
        "aformat=sample_rates=24000:channel_layouts=mono,asetnsamples=n=2400:p=0,"
        "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
        f"ametadata=mode=print:key={LEVEL_KEY}:file=-"
    )
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-af", chain, "-f", "null", "-"],
        capture_output=True, text=True, check=True, timeout=90,
    )
    return [float(line.split("=", 1)[1]) for line in result.stdout.splitlines()
            if line.startswith(f"{LEVEL_KEY}=")]


def percentile(ordered, fraction):
    """Return an observed sample quantile without extrapolating into pauses."""
    return ordered[round((len(ordered) - 1) * fraction)]


def measure_dialogue(path, spans, *, offset_ms=0):
    """Measure speech dynamics and stable noise outside known speech windows.

    Spans are in source-clock milliseconds. A decoded original may start after
    time zero. Only complete windows with a margin from speech estimate noise.
    """
    levels = window_levels(path)
    ordered_spans = sorted((start, end) for start, end in spans if start < end)
    if not ordered_spans:
        return {"status": "insufficient-speech"}
    speech, gaps = [], []
    cue = 0
    for index, level in enumerate(levels):
        middle = offset_ms + index * WINDOW_MS + WINDOW_MS // 2
        while cue + 1 < len(ordered_spans) and ordered_spans[cue + 1][0] <= middle:
            cue += 1
        start, end = ordered_spans[cue]
        if start <= middle < end:
            if level >= -50:
                speech.append(level)
        elif cue + 1 < len(ordered_spans) and end + 150 <= middle <= ordered_spans[cue + 1][0] - 150:
            gaps.append(level)
    if len(speech) < MIN_SPEECH_WINDOWS:
        return {"status": "insufficient-speech"}

    speech.sort()
    typical = percentile(speech, .5)
    quiet = percentile(speech, .2)
    gaps.sort()
    noise = None
    if len(gaps) >= 5:
        lower, median, upper = (percentile(gaps, fraction) for fraction in (.25, .5, .75))
        if isfinite(lower) and -55 <= median <= -30 and upper - lower <= 6 and 8 <= typical - median <= 25:
            noise = round(median, 1)
    total = rms(path)
    low, middle, presence = (level - total for level in band_levels(path))
    return {"status": "measured", "speechWindows": len(speech), "noiseWindows": len(gaps),
            "speechDbfs": round(typical, 1), "quietSpeechDbfs": round(quiet, 1), "steadyNoiseDbfs": noise,
            "lowBandDb": round(low, 1), "midBandDb": round(middle, 1), "presenceBandDb": round(presence, 1)}


def recommended_modes(quality):
    """Return the presets for which the measurements provide evidence."""
    modes = []
    if quality["status"] == "measured":
        if quality["steadyNoiseDbfs"] is not None:
            modes.append("denoise")
        if quality["presenceBandDb"] <= -21 and quality["lowBandDb"] - quality["midBandDb"] >= 4:
            modes.append("clarity")
        if quality["quietSpeechDbfs"] <= -33 and quality["speechDbfs"] <= -25:
            modes.append("quiet-speech")
    return modes


def restoration_filters(quality, rate, modes=None):
    """Translate automatic evidence or explicit presets into bounded corrections."""
    selected = recommended_modes(quality) if modes is None else modes
    actions, filters = {}, []
    if "denoise" in selected:
        # FFT overlap adds two 12.5 ms hops. Pad first to preserve the last speech samples.
        delay = 2 * (rate // 80)
        measured_noise = quality.get("steadyNoiseDbfs")
        noise = measured_noise if measured_noise is not None else -48
        actions["denoise"] = {"method": "fft", "reductionDb": 6, "noiseFloorDbfs": noise,
                              "basis": "measured" if measured_noise is not None else "preset"}
        filters.extend((f"apad=pad_len={delay}", f"afftdn=nr=6:nf={noise}:gs=8",
                        f"atrim=start_sample={delay}", "asetpts=PTS-STARTPTS"))
    if "clarity" in selected:
        actions["equalizer"] = {"lowMidDb": -3, "presenceDb": 4}
        filters.extend(("highpass=f=65", "equalizer=f=340:t=o:w=1.5:g=-3", "equalizer=f=2600:t=o:w=2:g=4"))
    if "quiet-speech" in selected:
        # The curve lifts low speech and tapers back to unity near silence and on loud peaks.
        actions["quietSpeech"] = {"maximumGainDb": 6}
        filters.append("compand=attacks=0.015:decays=0.15:points=-90/-90|-52/-52|-42/-36|-32/-26|-22/-22|0/0"
                       ":soft-knee=2:volume=-90")
    return actions, filters


def restore_dialogue(source, output, spans, *, modes=None, offset_ms=0):
    """Write a timing-compatible WAV only when a correction is warranted."""
    quality = measure_dialogue(source, spans, offset_ms=offset_ms)
    with wave.open(str(source), "rb") as audio:
        rate, frames, channels = audio.getframerate(), audio.getnframes(), audio.getnchannels()
    actions, filters = restoration_filters(quality, rate, modes)
    if not filters:
        return {"status": "unchanged", "quality": quality, "actions": {}}
    filters.extend(("alimiter=limit=0.891251:level=0:latency=1", "apad", f"atrim=end_sample={frames}"))
    ffmpeg(["-i", str(source), "-af", ",".join(filters), "-c:a", "pcm_s16le", str(output)])
    with wave.open(str(output), "rb") as audio:
        if (audio.getnframes(), audio.getframerate(), audio.getnchannels()) != (frames, rate, channels):
            raise ValueError("Очищенная дорожка вышла за границы исходной шкалы.")
    return {"status": "enhanced", "quality": quality, "actions": actions}
