"""Local engine adapters and audio preparation for Voice Studio."""

from voice_studio.mod_runtime import model_provider, model_resource, current_snapshot, builtin_provider, is_replacement

import json
import os
from time import monotonic
from pathlib import Path
import subprocess
import urllib.request
import wave

import msgpack

from voice_studio.core.worker_process import worker_command
from voice_studio.synthesis import commercial_voice

from voice_paths import MODELS_DIR as MODEL_DIR, PROJECT_ROOT as ROOT, RUNTIME_DIR, ensure_temporary_root, model_path
RESEARCH_MODELS = {"f5": "F5-TTS", "fish": "Fish Speech", "coqui": "Coqui XTTS"}
MODELS = {**commercial_voice.MODELS, **RESEARCH_MODELS}
SINGLE_REFERENCE_ENGINES = {"f5", *commercial_voice.MODELS}
AUTO_ENGINES = {"f5", "coqui", *commercial_voice.MODELS}


def research_enabled():
    """Expose the previous checkpoints only on explicit operator request."""
    return os.getenv("VOICE_RESEARCH_ENGINES") == "1"
MAX_AUDIO = 24 * 1024 * 1024
F5_ACCENT_WEIGHTS = model_path("f5", "accent-tune.safetensors")


def local_weights(model):
    """Return the configured model path and its tokenizer or config."""
    if model == "f5":
        return (os.getenv("VOICE_F5_CHECKPOINT", str(model_path("f5", "model.safetensors"))),
                os.getenv("VOICE_F5_VOCAB", str(model_path("f5", "vocab.txt"))))
    return (os.getenv("VOICE_COQUI_CHECKPOINT", str(model_path("xtts", "model.pth"))),
            os.getenv("VOICE_COQUI_CONFIG", str(model_path("xtts", "config.json"))))


def python_for(model):
    """Resolve the operator-configured engine interpreter."""
    key = "F5" if model == "f5" else "COQUI"
    return Path(os.getenv(f"VOICE_{key}_PYTHON", ROOT / ".engines" / model / "bin" / "python"))


def fish_request(path, data=None, timeout=2):
    """Call a fixed loopback Fish Speech endpoint without redirects or proxies."""
    class LocalOnly(urllib.request.HTTPRedirectHandler):
        """Keep reference audio on the loopback endpoint."""
        def redirect_request(self, *args, **kwargs):
            return None

    port = int(os.getenv("VOICE_FISH_PORT", "8080"))
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    if data is not None:
        req.add_header("Content-Type", "application/msgpack")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), LocalOnly())
    with opener.open(req, timeout=timeout) as response:
        body = response.read(MAX_AUDIO + 1)
    if len(body) > MAX_AUDIO:
        raise ValueError("Ответ Fish Speech превышает лимит.")
    return body


def native_tts_readiness(model, language, checkpoint=None):
    """Check the actual builtin engine's installed runtime/files without loading weights."""
    if model in commercial_voice.MODELS:
        return (True, 'Можно запустить') if commercial_voice.ready(model) else (False, 'Установите локальные веса MLX')
    if model == 'fish':
        try:
            ready = json.loads(fish_request('/v1/health')).get('status') == 'ok'
        except (OSError, ValueError):
            ready = False
        return ready, 'Можно запустить' if ready else 'Запустите локальный сервер Fish Speech'
    if not python_for(model).is_file():
        return False, 'Подключите окружение модели'
    if model == 'f5' and language == 'ru':
        if not (checkpoint or all(Path(path).is_file() for path in local_weights(model))):
            return False, 'Укажите русский checkpoint F5'
    if model == 'coqui':
        path, config = local_weights(model)
        if not all(Path(item).is_file() for item in (path, config, Path(path).with_name('vocab.json'))):
            return False, 'Установите локальные веса XTTS-v2'
    return True, 'Можно запустить'


def availability(demo, language, checkpoint=None):
    """Describe requested slots using the actual selected provider's readiness."""
    status = {}
    snapshot = current_snapshot()
    override = snapshot.to_wire()['selections'].get('synthesis.tts')
    if override:
        snapshot.selected('synthesis.tts')
    custom = override and not builtin_provider(snapshot, 'synthesis.tts', override)
    catalog = MODELS if research_enabled() else (RESEARCH_MODELS if demo else commercial_voice.MODELS)
    for requested, name in catalog.items():
        actual = override.removeprefix('builtin.') if override and not custom else requested
        if custom:
            ready, reason = True, 'Selected local mod: ' + override
        elif demo:
            ready, reason = True, 'Тестовый тон'
        else:
            compatible_checkpoint = checkpoint if actual == requested else None
            ready, reason = native_tts_readiness(actual, language, compatible_checkpoint)
        status[requested] = {'name': name, 'ready': ready, 'automatic': actual in AUTO_ENGINES,
                             'reason': reason}
        if override:
            status[requested]['actualProvider'] = override
    return status


def ffmpeg(args, timeout=90, cwd=None):
    """Run FFmpeg with bounded execution and actionable failures."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
        capture_output=True, timeout=timeout, cwd=cwd,
    )
    if result.returncode:
        raise ValueError("FFmpeg не смог обработать аудио.")


def bounded_timeout(deadline, maximum):
    """Limit each blocking stage by the remaining automatic-run budget."""
    if deadline is None:
        return maximum
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ValueError("Время автосинтеза истекло.")
    return min(maximum, remaining)


def normalize(source, output, duration_limit=90, sample_rate=24000, *, deadline=None):
    """Decode audio into mono PCM and return the verified duration in seconds."""
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist",
            "wav,matroska,webm,mov,mp3,ogg,flac,aac,aiff",
            "-i", str(source), "-vn", "-t", str(duration_limit + 1), "-ac", "1",
            "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output)],
           timeout=bounded_timeout(deadline, 90))
    with wave.open(str(output)) as audio:
        seconds = audio.getnframes() / audio.getframerate()
    if not 0.35 <= seconds <= duration_limit:
        raise ValueError(f"Длительность аудио должна быть от 0,35 до {duration_limit} секунд.")
    return seconds


@model_provider('synthesis.tts', lambda model, *args, **kwargs: 'builtin.' + model)
def generate(model, job, demo):
    """Write raw candidate audio using the selected local engine."""
    output = Path(job["output"])
    deadline = job.pop("deadline", None)
    if demo:
        frequency = {"f5": 392, "fish": 494, "coqui": 587, "qwen": 440, "chatterbox": 523}[model]
        ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={frequency}:duration=1.5", str(output)],
               timeout=bounded_timeout(deadline, 90))
        return
    if model in commercial_voice.MODELS:
        commercial_voice.generate(model, json.dumps(job), bounded_timeout(deadline, 900))
        return
    if model == "fish":
        refs = [{"audio": Path(item["path"]).read_bytes(), "text": item["text"]}
                for item in job["references"]]
        body = {"text": job["text"], "references": refs, "format": "wav",
                "streaming": False, "seed": job["seed"]}
        output.write_bytes(fish_request("/v1/tts", msgpack.packb(body, use_bin_type=True), 600))
        return
    key = "F5" if model == "f5" else "COQUI"
    checkpoint, companion = local_weights(model)
    if model == "f5" and not Path(checkpoint).is_file():
        checkpoint, companion = "", ""
    vocoder_weights = model_path('vocos', 'pytorch_model.bin')
    selected_vocoder = is_replacement('synthesis.vocoder-resource', 'builtin.vocos')
    if vocoder_weights.is_file() or selected_vocoder:
        vocoder_weights = model_resource('synthesis.vocoder-resource', 'builtin.vocos', vocoder_weights)
    vocoder = vocoder_weights.parent
    if model == 'f5' and job.get('checkpoint') == str(F5_ACCENT_WEIGHTS):
        job['checkpoint'] = str(model_resource('synthesis.accent-resource', 'builtin.f5-accent', F5_ACCENT_WEIGHTS))
    job.update(
        model=model,
        architecture=job.get("architecture") or os.getenv(
            "VOICE_F5_ARCHITECTURE", "F5TTS_Base" if checkpoint else "F5TTS_v1_Base"),
        checkpoint=job.get("checkpoint") or checkpoint,
        config=job.get("config") or companion,
        vocab=companion if model == "f5" else "",
        vocoder=str(vocoder) if vocoder_weights.is_file() else None,
        device=os.getenv(f"VOICE_{key}_DEVICE", "cpu"),
    )
    env = os.environ.copy()
    ensure_temporary_root()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
    env.setdefault("HF_HOME", str(RUNTIME_DIR / "huggingface"))
    env.setdefault("OMP_NUM_THREADS", "4")
    process = subprocess.run(
        worker_command(python_for(model), "voice_studio.workers.model_worker"),
        input=json.dumps(job), text=True, timeout=bounded_timeout(deadline, 900), stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, env=env,
    )
    if process.returncode:
        # Keep local file paths and engine internals out of the browser.
        print(f"{MODELS[model]} failed: {process.stderr[-2000:]}", flush=True)
        raise ValueError(f"{MODELS[model]} не завершила синтез. Подробности в терминале.")
