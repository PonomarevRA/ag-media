"""Prepare a separate WAV preview with local noise reduction and broad EQ."""

from voice_studio.mod_runtime import model_provider

import re
import subprocess

from voice_studio.synthesis.engines import ffmpeg
from voice_paths import model_path

DENOISE_MODEL = model_path("denoise", "cb.rnnn")
CENTERS = (220, 1000, 3800)
BANDS = ((100, 450), (450, 1800), (1800, 6500))
MAX_EQ = 6
MAX_GAIN = 9


def bounded(value, limit):
    """Validate a finite gain before building an FFmpeg filter expression."""
    if type(value) not in (int, float) or not -limit <= value <= limit:
        raise ValueError(f"Усиление должно быть в пределах ±{limit} дБ.")
    return round(value, 1)


def rms(path, filters=()):
    """Read mean signal power in dBFS from FFmpeg's volume meter."""
    chain = ",".join((*filters, "volumedetect"))
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", chain, "-f", "null", "-"],
        capture_output=True, text=True, timeout=30,
    )
    match = re.search(r"mean_volume: (-?[0-9.]+) dB", result.stderr)
    if result.returncode or not match:
        raise ValueError("Не удалось измерить уровень дорожки.")
    return float(match[1])


def band_levels(path):
    """Measure RMS in matching fixed speech bands."""
    return [rms(path, (f"highpass=f={low}", f"lowpass=f={high}")) for low, high in BANDS]


def eq_filters(gains):
    """Create broad, overlapping bell filters for the three controls."""
    return [f"equalizer=f={hz}:t=o:w=2:g={gain}" for hz, gain in zip(CENTERS, gains) if gain]


def automatic_eq(source, reference):
    """Nudge relative band levels toward the reference with limited boosts."""
    source_level, reference_level = rms(source), rms(reference)
    if min(source_level, reference_level) < -50:
        raise ValueError("Для авто EQ обе дорожки должны содержать слышимый сигнал.")
    balance = []
    for current, target in zip(band_levels(source), band_levels(reference)):
        difference = (target - reference_level) - (current - source_level)
        balance.append(round(max(-MAX_EQ, min(MAX_EQ, difference * 0.65)), 1))
    return balance


def process_wave(source, reference, output, settings):
    """Render a non-destructive mono WAV and return the settings actually applied."""
    automatic, clean = settings.get("auto", False), settings.get("denoise", False)
    if type(automatic) is not bool or type(clean) is not bool:
        raise ValueError("Некорректный режим обработки.")
    if automatic and reference is None:
        raise ValueError("Выбери исходник для авто EQ.")
    manual = settings.get("gains", [0, 0, 0])
    if not isinstance(manual, list) or len(manual) != len(CENTERS):
        raise ValueError("Нужны три полосы эквалайзера.")
    gains = [bounded(value, MAX_EQ) for value in manual]
    gain = bounded(settings.get("gain", 0), MAX_GAIN)
    method = "none"
    if clean:
        prepared = output.with_name("clean.wav")
        method = denoise_wave(source, prepared)
        source = prepared
    if automatic:
        gains = automatic_eq(source, reference)
        difference = rms(reference) - rms(source, eq_filters(gains))
        gain = round(max(-MAX_GAIN, min(MAX_GAIN, difference)), 1)
    filters = eq_filters(gains)
    if gain:
        filters.append(f"volume={gain}dB")
    if filters:
        filters.append("alimiter=limit=0.891251:level=0:latency=1")
    args = ["-i", str(source)]
    if filters:
        args += ["-af", ",".join(filters)]
    ffmpeg([*args, "-ac", "1", "-c:a", "pcm_s16le", str(output)])
    return {"gains": gains, "gain": gain, "denoise": method}


@model_provider('audio.denoise', lambda *args, **kwargs: 'builtin.rnnoise' if DENOISE_MODEL.is_file() else 'builtin.fft-denoise')
def denoise_wave(source, output):
    """Route one complete WAV through the selected noise reduction provider."""
    return denoise_native(source, output)


def denoise_native(source, output, *, provider=None):
    """Preserve automatic RNNoise/FFT behavior; explicit providers fail visibly."""
    if provider != 'fft-denoise' and DENOISE_MODEL.is_file():
        model = output.with_name('denoise.rnnn')
        model.symlink_to(DENOISE_MODEL)
        try:
            ffmpeg(['-i', str(source), '-af', 'aresample=48000,arnndn=m=denoise.rnnn',
                    '-c:a', 'pcm_s16le', str(output)], cwd=output.parent)
            return 'rnnoise'
        except ValueError:
            if provider == 'rnnoise':
                raise
    elif provider == 'rnnoise':
        raise ValueError('RNNoise weights are unavailable.')
    ffmpeg(['-i', str(source), '-af', 'afftdn=nr=10:nf=-36', '-c:a', 'pcm_s16le', str(output)])
    return 'fft'
