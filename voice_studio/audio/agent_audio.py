"""Export aligned dialogue, music and effects for local MCP clients."""

from voice_studio.mod_runtime import model_provider, analysis_run

import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Literal
from uuid import uuid4
import wave

from voice_studio.audio.audio_processing import denoise_wave
from voice_studio.audio.voice_quality import measure_dialogue, recommended_modes, restore_dialogue
from voice_studio.synthesis.dub_background import PYTHON, RATE
from voice_studio.synthesis.engines import ffmpeg
from voice_paths import model_path, publish_directory, temporary_directory
from voice_studio.graph.media_graph_ingest import probe_source
from voice_studio.core.media_resources import batch_media_job, stage_threads
from voice_studio.core.worker_process import worker_command
from voice_studio.speech.media_transcript import FORMATS, MAX_MEDIA_SECONDS, audio_offset, transcribe_wav

Language = Literal["auto", "ru", "en"]
INSTALL = model_path("separation", "mrx", "default_mrx_pre_trained_weights.pth").parent
WEIGHTS = "default_mrx_pre_trained_weights.pth"
MRX = "merl-mrx-snr-19b3de8"
STEMS = ("dialogue", "music", "effects")
# The current FFmpeg filters introduce one 10 ms RNNoise block or 25 ms of FFT overlap.
DENOISE_DELAY = {"rnnoise": 480, "fft": 1200}


def separation_ready():
    """Report the presence of local model weights and their isolated runtime."""
    return all(path.is_file() for path in (PYTHON, INSTALL / "mrx.py", INSTALL / WEIGHTS))


def media_interval(file_path, start_ms, end_ms):
    """Validate a half-open source interval without copying the complete media."""
    source = Path(file_path).expanduser()
    info = probe_source(source)
    end_ms = info["durationMs"] if end_ms is None else end_ms
    if not info["hasAudio"]:
        raise ValueError("В файле нет звуковой дорожки.")
    if not 0 <= start_ms < end_ms <= info["durationMs"] or not 350 <= end_ms - start_ms <= MAX_MEDIA_SECONDS * 1000:
        raise ValueError("Укажи интервал внутри файла длительностью от 350 мс до 15 минут.")
    return source, start_ms, end_ms


def decode_interval(source, start_ms, end_ms, output, rate, codec="pcm_f32le", *, channels=2):
    """Keep the source clock, including silence before a delayed audio track."""
    offset = audio_offset(source)
    start = start_ms * rate // 1000
    duration = end_ms - start_ms
    stop = start + duration * rate // 1000
    chain = (f"asetpts=PTS-STARTPTS,aresample={rate}:async=1:first_pts=0,adelay={offset}:all=1,"
             f"apad,atrim=start_sample={start}:end_sample={stop},asetpts=N/SR/TB")
    channel_options = ["-ac", str(channels)] if channels is not None else []
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source),
            "-map", "0:a:0", *channel_options, "-af", chain, "-t", f"{duration / 1000:.3f}",
            "-c:a", codec, str(output)])


@model_provider('audio.separation', 'builtin.mrx')
def run_separator(source, destination, duration_ms):
    """Isolate inference and bound the lifetime of a job."""
    if not separation_ready():
        raise ValueError("Установи MRX через scripts/install-mrx-audio.py в каталоге Voice Studio.")
    threads = stage_threads(4)
    try:
        command = worker_command(PYTHON, "voice_studio.workers.mrx_audio_worker",
                                 source, INSTALL, destination, threads)
        subprocess.run(command, capture_output=True, check=True, timeout=max(90, duration_ms / 400))
    except (OSError, subprocess.SubprocessError) as error:
        logging.getLogger(__name__).warning("MRX failed: %s", str(getattr(error, "stderr", None) or error)[-1200:])
        raise ValueError("MRX не смог разделить выбранный фрагмент.") from error


def export_dialogue_cues(source, work, rendered, final, cues, origin):
    """Save the original stereo dialogue slices while keeping source-clock cue bounds."""
    pcm = work / "dialogue-pcm.wav"
    ffmpeg(["-i", str(source), "-c:a", "pcm_s16le", str(pcm)])
    clips = rendered / "cues"
    clips.mkdir()
    with wave.open(str(pcm)) as audio:
        rate = audio.getframerate()
        for index, cue in enumerate(cues, start=1):
            begin = (cue["startMs"] - origin) * rate // 1000
            end = min(audio.getnframes(), (cue["endMs"] - origin) * rate // 1000)
            if end <= begin:
                continue
            name = f"{index:04d}.wav"
            audio.setpos(begin)
            with wave.open(str(clips / name), "wb") as clip:
                clip.setparams(audio.getparams())
                clip.writeframes(audio.readframes(end - begin))
            cue["audioPath"] = str(final / "cues" / name)


def attach_dialogue(result, work, rendered, final, language, speakers, export_cues):
    """Transcribe the dialogue estimate and optionally render individual cue files."""
    dialogue = rendered / "dialogue.wav"
    speech = work / "speech.wav"
    ffmpeg(["-i", str(dialogue), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(speech)])
    start = result["startMs"]
    try:
        cues = transcribe_wav(speech, work, language, start, quality=True, speakers=speakers, audio_layers=False)
    except ValueError as error:
        result.update(cues=[], transcriptionStatus="unavailable", transcriptionWarning=str(error))
        return
    rows = cues["segments"]
    if export_cues:
        export_dialogue_cues(dialogue, work, rendered, final, rows, start)
    (rendered / "dialogue.srt").write_text(cues["srt"], encoding="utf-8")
    result.update(cues=rows, language=cues["language"], srtPath=str(final / "dialogue.srt"),
                  transcriptionStatus="done")
    for field in ("recognitionModel", "recognitionWarning", "speakerCount", "speakerWarning"):
        if field in cues:
            result[field] = cues[field]


@batch_media_job
@analysis_run
def separate_audio(output, file_path, start_ms, end_ms, language, transcription, speakers, export_cues):
    """Publish three aligned WAVs and optional timed cues in an atomic job folder."""
    source, start, end = media_interval(file_path, start_ms, end_ms)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = output / f"stems-{uuid4().hex}"
    with temporary_directory() as scratch:
        work = Path(scratch)
        mix = work / "input.wav"
        rendered = work / "stems"
        rendered.mkdir()
        decode_interval(source, start, end, mix, RATE)
        run_separator(mix, rendered, end - start)
        paths = {stem: str(final / f"{stem}.wav") for stem in STEMS}
        result = {"sourcePath": str(source), "startMs": start, "endMs": end, "sampleRate": RATE,
                  "method": MRX, "stems": paths,
                  "suggestedTool": {"name": "audio_voice_quality", "arguments": {"file_path": paths["dialogue"]},
                                    "when": "Measure the dialogue before choosing restoration."}}
        if transcription:
            attach_dialogue(result, work, rendered, final, language, speakers, export_cues)
        result["manifestPath"] = str(final / "manifest.json")
        (rendered / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        publish_directory(rendered, final)
    return result


@batch_media_job
@analysis_run
def clean_audio(output, file_path, start_ms, end_ms):
    """Save reference and denoised WAVs covering exactly the same time interval."""
    source, start, end = media_interval(file_path, start_ms, end_ms)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = output / f"denoise-{uuid4().hex}"
    rate = 48_000
    with temporary_directory() as scratch:
        work = Path(scratch)
        rendered = work / "ab"
        rendered.mkdir()
        reference = rendered / "reference.wav"
        decode_interval(source, start, end, reference, rate, codec="pcm_s16le")
        padded = work / "padded.wav"
        ffmpeg(["-i", str(reference), "-af", "apad=pad_len=2048", "-c:a", "pcm_s16le", str(padded)])
        filtered = work / "filtered.wav"
        method = denoise_wave(padded, filtered)
        frames = (end - start) * rate // 1000
        if isinstance(method, dict):
            delay = method.get('delaySamples')
            if type(delay) is not int or not 0 <= delay <= rate:
                raise ValueError('Custom denoiser must declare delaySamples between 0 and sampleRate.')
            method = method.get('method', 'custom')
        else:
            delay = DENOISE_DELAY[method]
        align = f"apad,atrim=start_sample={delay}:end_sample={delay + frames},asetpts=N/SR/TB"
        ffmpeg(["-i", str(filtered), "-af", align,
                "-c:a", "pcm_s16le", str(rendered / "cleaned.wav")])
        publish_directory(rendered, final)
    return {"sourcePath": str(source), "startMs": start, "endMs": end, "method": method, "delayCompensatedSamples": delay,
            "referencePath": str(final / "reference.wav"), "cleanedPath": str(final / "cleaned.wav")}


def selected_speech_spans(spans, start, end):
    """Clip and merge supplied speech spans on the media clock for a selected range."""
    if spans is None:
        return [(start, end)]
    if len(spans) > 1000 or any(a < 0 or a >= b for a, b in spans):
        raise ValueError("Укажи до 1000 корректных интервалов речи.")
    merged = []
    for a, b in sorted(spans):
        left, right = max(start, a), min(end, b)
        if left >= right:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


@batch_media_job
def inspect_voice_quality(file_path, start_ms, end_ms, speech_intervals_ms):
    """Measure a local range without writing an artifact or inferring the speaker's identity."""
    source, start, end = media_interval(file_path, start_ms, end_ms)
    spans = selected_speech_spans(speech_intervals_ms, start, end)
    with temporary_directory() as scratch:
        reference = Path(scratch) / "reference.wav"
        decode_interval(source, start, end, reference, 48_000, codec="pcm_s16le", channels=None)
        quality = measure_dialogue(reference, spans, offset_ms=start)
    return {"startMs": start, "endMs": end, "quality": quality,
            "speechTiming": "provided" if speech_intervals_ms is not None else "assumed-dialogue",
            "recommendedModes": recommended_modes(quality)}


@batch_media_job
def render_voice_restoration(output, file_path, start_ms, end_ms, speech_intervals_ms, modes):
    """Publish an aligned reference and a separately processed WAV for the requested presets."""
    source, start, end = media_interval(file_path, start_ms, end_ms)
    spans = selected_speech_spans(speech_intervals_ms, start, end)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = output / f"voice-restoration-{uuid4().hex}"
    with temporary_directory() as scratch:
        rendered = Path(scratch) / "comparison"
        rendered.mkdir()
        reference, cleaned = rendered / "reference.wav", rendered / "cleaned.wav"
        decode_interval(source, start, end, reference, 48_000, codec="pcm_s16le", channels=None)
        result = restore_dialogue(reference, cleaned, spans, modes=modes, offset_ms=start)
        publish_directory(rendered, final)
    return {**result, "startMs": start, "endMs": end, "referencePath": str(final / reference.name),
            "recommendedModes": recommended_modes(result["quality"]),
            "cleanedPath": str(final / cleaned.name) if result["status"] == "enhanced" else None}
