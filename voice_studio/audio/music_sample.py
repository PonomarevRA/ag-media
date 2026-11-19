"""Prepare a short original clip and the signal removed by local denoising."""

from base64 import b64encode
import subprocess
import wave

from voice_studio.audio.audio_processing import DENOISE_MODEL
from voice_studio.synthesis.engines import ffmpeg

LENGTH_SECONDS = 15
MAX_START_MS = 24 * 60 * 60 * 1000
SAMPLE_RATE = "24000"


def music_sample(source, work, start_ms, formats):
    """Return two auditionable WAVs for an explicitly selected source range."""
    if type(start_ms) is not int or not 0 <= start_ms <= MAX_START_MS:
        raise ValueError("Выбери допустимую позицию в записи.")
    original, removed = work / "original.wav", work / "removed.wav"
    try:
        ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", formats,
                "-ss", f"{start_ms / 1000:.3f}", "-i", str(source), "-map", "0:a:0", "-vn",
                "-t", str(LENGTH_SECONDS), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(original)],
               timeout=20)
        with wave.open(str(original)) as audio:
            duration_ms = audio.getnframes() * 1000 // audio.getframerate()
        if duration_ms < 350:
            raise ValueError("В этой позиции слишком мало звука. Выбери момент раньше.")
        method = render_removed(original, removed)
        return {"startMs": start_ms, "durationMs": duration_ms, "method": method,
                "original": encode_wave(original), "removed": encode_wave(removed)}
    except (subprocess.TimeoutExpired, wave.Error, EOFError) as exc:
        raise ValueError("Не удалось подготовить фрагмент звука.") from exc


def render_removed(original, removed):
    """Use the denoiser's native residual output and preserve a simple fallback."""
    if DENOISE_MODEL.is_file():
        model = original.with_name("denoise.rnnn")
        model.symlink_to(DENOISE_MODEL)
        try:
            ffmpeg(["-i", str(original), "-af", "arnndn=m=denoise.rnnn:mix=-1",
                    "-ar", SAMPLE_RATE, "-c:a", "pcm_s16le", str(removed)], cwd=original.parent, timeout=12)
            return "rnnoise"
        except ValueError:
            pass
    ffmpeg(["-i", str(original), "-af", "afftdn=nr=10:nf=-36:om=n",
            "-ar", SAMPLE_RATE, "-c:a", "pcm_s16le", str(removed)], timeout=12)
    return "fft"


def encode_wave(path):
    """Encode a bounded WAV for same-origin transport to the two browser players."""
    return b64encode(path.read_bytes()).decode("ascii")
