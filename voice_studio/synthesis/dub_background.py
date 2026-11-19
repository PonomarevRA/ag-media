"""Separate a continuous source bed and mix it with an aligned dub."""

from voice_studio.mod_runtime import model_provider

import logging
import os
import subprocess

from voice_studio.core.worker_process import worker_command
from voice_studio.synthesis.engines import ROOT, ffmpeg
from voice_studio.speech.media_transcript import FORMATS, audio_offset

MODEL_NAME = "hdemucs-high-musdb-plus"
BACKGROUND_WAV = "background.wav"
RATE = 44_100
from voice_paths import model_path

MODEL = model_path("separation", "hdemucs_high_trained.pt")
PYTHON = ROOT / ".engines" / "coqui" / "bin" / "python"


def timeline_end(duration_ms):
    """Return an audio filter that yields exactly the requested length at RATE."""
    return f"apad,atrim=end_sample={duration_ms * RATE // 1000}"


@model_provider('audio.background', 'builtin.hdemucs')
def prepare_background(source, duration_ms, folder):
    """Extract stereo ambience at source timestamps using an offline model worker."""
    if not MODEL.is_file() or not PYTHON.is_file():
        raise ValueError("Для сохранения музыки и фона нужны локальные веса HDemucs и среда TorchAudio.")
    original = folder / "background-input.wav"
    output = folder / BACKGROUND_WAV
    offset = audio_offset(source)
    timing = f"asetpts=PTS-STARTPTS,aresample={RATE}:async=1:first_pts=0,adelay={offset}:all=1"
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source), "-map", "0:a:0",
            "-ac", "2", "-af", f"{timing},{timeline_end(duration_ms)}", "-c:a", "pcm_f32le", str(original)])
    timeout = max(90, min(1200, 60 + duration_ms // 1000))
    try:
        subprocess.run(worker_command(PYTHON, "voice_studio.workers.dub_background_worker", original, MODEL, output),
                       check=True, capture_output=True, timeout=timeout,
                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    except (OSError, subprocess.SubprocessError) as error:
        details = getattr(error, "stderr", None) or str(error)
        logging.getLogger(__name__).warning("HDemucs separation failed: %s", details[-1200:])
        raise ValueError("HDemucs не смог выделить музыку и фон из исходной дорожки.") from error
    return output


def mix_english(dialogue, background, duration_ms, output):
    """Center the dub on a continuous stereo bed without gating or clock shifts."""
    # The limiter's latency compensation keeps both tracks on the source clock.
    filters = (f"[0:a]aresample={RATE},pan=stereo|c0=0.707*c0|c1=0.707*c0[voice];"
               "[voice][1:a]amix=inputs=2:duration=longest:normalize=0,"
               f"alimiter=limit=0.94:attack=5:release=70:level=0:latency=1,{timeline_end(duration_ms)}")
    ffmpeg(["-i", str(dialogue), "-i", str(background), "-filter_complex", filters,
            "-c:a", "pcm_s24le", str(output)])
