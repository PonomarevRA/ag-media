"""Transcribe a local recording with multilingual Whisper."""

from voice_studio.mod_runtime import model_provider

import json
import os
from pathlib import Path
import subprocess
from threading import Lock
from time import monotonic

from voice_paths import model_path, temporary_directory
from voice_studio.synthesis.engines import bounded_timeout, normalize
from voice_studio.speech.whisper_assets import whisper_base_model, whisper_cli
from voice_studio.audio.pauses import annotate, quiet_spans

CLI = whisper_cli()
MODEL = whisper_base_model()
SMALL_MODEL = Path(os.getenv("VOICE_WHISPER_SMALL_MODEL", model_path("whisper", "ggml-small-q5_1.bin")))
RECOVERY_MODEL = Path(os.getenv("VOICE_WHISPER_RECOVERY_MODEL", model_path("whisper", "ggml-large-v3-turbo-q5_0.bin")))
LOCK = Lock()
TRANSLATION_BATCH_SIZE = 16


def ready(model=MODEL):
    """Report whether the local executable and model are present."""
    return CLI.is_file() and os.access(CLI, os.X_OK) and model.is_file()


def transcribe(root, language, audio, *, deadline=None):
    """Return a draft transcript without saving the input in the corpus."""
    return recognize(root, language, audio, deadline=deadline)["text"]


def recognize_file(wav, output, language, words=False, timeout=90, wait=30, timestamps=True, *,
                   model=None, threads=4, translate=False):
    """Run Whisper on prepared mono 16 kHz PCM and return timed segments."""
    return recognize_result(wav, output, language, words, timeout, wait, timestamps,
                            model=model, threads=threads, translate=translate)["transcription"]


def recognize_result(wav, output, language, words=False, timeout=90, wait=30, timestamps=True, *,
                     model=None, threads=4, translate=False):
    """Read Whisper's timed transcript and native language result from the same pass."""
    options = ["-sow", "-ml", "1"] if words else ([] if timestamps else ["-nt"])
    if translate:
        options.append("-tr")
    return run_whisper([(wav, output)], language, options, model, threads, timeout, wait)[0]


def translate_clips(paths, language):
    """Translate independent short files, loading Whisper Small once per bounded batch."""
    # Batching shares the decoder's random state, which can change draft wording.
    chunk_size = TRANSLATION_BATCH_SIZE if os.getenv("VOICE_DUB_BATCH", "0") == "1" else 1
    translations = []
    for start in range(0, len(paths), chunk_size):
        batch = paths[start:start + chunk_size]
        jobs = [(wav, wav.with_suffix("")) for wav in batch]
        results = run_whisper(jobs, language, ["-nt", "-tr"], SMALL_MODEL, 4, 240, 30)
        translations.extend(" ".join(row["text"] for row in result["transcription"]) for result in results)
    return translations


def read_whisper_output(output):
    """Keep readable cues even when the CLI truncates a multibyte token."""
    result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8", errors="replace"))
    if not isinstance(result, dict) or not isinstance(result.get("transcription"), list):
        raise ValueError("missing transcription")
    return result


def watch_whisper_windows(command, jobs, timeout, on_window):
    """Observe completed JSONs in one CLI run and keep them if the remaining work times out."""
    results = [None] * len(jobs)
    completed = 0
    deadline = monotonic() + timeout

    def collect():
        """A file still being written is retried until it parses."""
        nonlocal completed
        before = completed
        for index, (_, output) in enumerate(jobs):
            if results[index] is not None:
                continue
            try:
                results[index] = read_whisper_output(output)
            except (OSError, ValueError):
                continue
            completed += 1
        if completed != before:
            on_window(completed)

    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        try:
            while process.poll() is None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    process.wait(timeout=min(.5, remaining))
                except subprocess.TimeoutExpired:
                    pass
                collect()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        collect()
    return results


def whisper_mod(jobs, language, options, model, *args, **kwargs):
    """Keep the requested installed Whisper checkpoint as the default provider."""
    if model is not None and Path(model).name == 'ggml-base-q5_1.bin':
        return 'builtin.whisper-live'
    if model is not None and Path(model) == SMALL_MODEL:
        return 'builtin.whisper-small'
    if model is not None and Path(model) == RECOVERY_MODEL:
        return 'builtin.whisper-recovery'
    return 'builtin.whisper-base'


@model_provider('speech.asr', whisper_mod)
def run_whisper(jobs, language, options, model, threads, timeout, wait, *, on_window=None):
    """Run one local process and require one valid result for each supplied output path."""
    model = MODEL if model is None else model
    if not ready(model):
        raise ValueError("Локальная модель Whisper недоступна.")
    if not LOCK.acquire(timeout=wait):
        raise ValueError("Whisper уже обрабатывает запись. Попробуй ещё раз.")
    try:
        command = [str(CLI), "-m", str(model), "-l", language, "-oj", *options, "-ng", "-t", str(threads)]
        for wav, output in jobs:
            command.extend(["-f", str(wav), "-of", str(output)])
        if on_window is not None:
            return watch_whisper_windows(command, jobs, timeout, on_window)
        subprocess.run(command, check=True, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [read_whisper_output(output) for _, output in jobs]
    except (BrokenPipeError, ConnectionResetError):
        raise
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("Whisper не смог расшифровать запись. Текст можно ввести вручную.") from exc
    finally:
        LOCK.release()


def recognize(root, language, audio, words=False, *, deadline=None, word_pause_min_ms=None):
    """Return timed words, optionally with silence measured on the same PCM wave."""
    with temporary_directory() as folder:
        work = Path(folder)
        source, wav = work / "input", work / "input.wav"
        source.write_bytes(audio)
        normalize(source, wav, duration_limit=90 if words else 45, sample_rate=16000, deadline=deadline)
        segments = recognize_file(wav, work / "transcript", language, words=words, wait=30 if words else 0,
                                  timestamps=words, timeout=bounded_timeout(deadline, 90))
        if words:
            if word_pause_min_ms is not None:
                return segments, quiet_spans(wav, word_pause_min_ms)
            return segments
        result = annotate(segments, quiet_spans(wav))
    if not result["text"]:
        raise ValueError("Разборчивой речи не найдено. Текст можно ввести вручную.")
    return result
