"""Pinned local MLX checkpoints for reference-based voice synthesis."""

from voice_studio.mod_runtime import model_resource, worker_environment

import json
import os
import subprocess
import sys

from voice_paths import PROJECT_ROOT as ROOT, model_path
from voice_studio.core.worker_process import worker_command
from voice_studio.core.optional_runtime import runtime_environment, runtime_ready

PYTHON = ROOT / ".engines" / "mlx-tts" / "bin" / "python"
MODELS = {"qwen": "Qwen3-TTS 0.6B", "chatterbox": "Chatterbox Multilingual V3"}
REVISIONS = {
    "qwen": ("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit", "0d6bb6fe33f92d47a507e23b9148940e8366ab5b"),
    "chatterbox": ("mlx-community/chatterbox-multilingual-v3", "03565773edd72e949572557597af8063bb49a18a"),
    "s3tokenizer": ("mlx-community/S3TokenizerV2", "e0c9886f0e1c35ae85b1f27277416fb19fc72bec"),
}


def ready(model):
    """Check the files required for local inference before advertising readiness."""
    directory = model_path("commercial", model, "config.json").parent
    if not (PYTHON.is_file() or runtime_ready("mlx")) or not (directory / "config.json").is_file():
        return False
    if model == "qwen":
        index = directory / "model.safetensors.index.json"
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return bool(shards) and all((directory / item).is_file() for item in shards) and (
            directory / "speech_tokenizer" / "model.safetensors").is_file()
    repo, revision = REVISIONS["s3tokenizer"]
    cache = model_path("commercial", "hub") / ("models--" + repo.replace("/", "--"))
    snapshot = cache / "snapshots" / revision
    try:
        pinned = (cache / "refs" / "main").read_text().strip() == revision
    except OSError:
        return False
    return pinned and all(path.is_file() for path in (
        directory / "model.safetensors", directory / "tokenizer.json",
        snapshot / "model.safetensors", snapshot / "config.json",
    ))


def generate(model, job_json, timeout):
    """Render in a separate offline process, bounded by the current job deadline."""
    hub_path = model_path('commercial', 'hub')
    if model == 'chatterbox':
        repo, revision = REVISIONS['s3tokenizer']
        reference = hub_path / ('models--' + repo.replace('/', '--')) / 'refs' / 'main'
        hub_path = model_resource('synthesis.tokenizer-resource', 'builtin.s3tokenizer', reference).parents[2]
    hub = str(hub_path)
    env = dict(os.environ, HF_HOME=hub, HF_HUB_CACHE=hub,
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    interpreter = sys.executable if runtime_ready("mlx") else PYTHON
    process = subprocess.run(
        worker_command(interpreter, "voice_studio.workers.commercial_voice_worker", model),
        input=job_json, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=runtime_environment("mlx", worker_environment(env)), timeout=timeout,
    )
    if process.returncode:
        print(f"{MODELS[model]} failed: {process.stderr[-2500:]}", flush=True)
        raise ValueError(f"{MODELS[model]} не завершила синтез. Подробности в терминале.")
