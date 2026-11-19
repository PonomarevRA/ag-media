"""Render one speech segment with the pinned MLX runtime."""

import json
import sys

from voice_paths import model_path
from voice_studio.synthesis.voice_controls import generation_options


def render(model_name, job):
    """Load a pinned model and write mono PCM from the selected reference."""
    controls = generation_options(model_name, job)
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio
    import numpy as np
    import soundfile as sf

    model = load_model(model_path("commercial", model_name, "config.json").parent)
    reference = job["references"][0]
    mx.random.seed(job["seed"])
    options = {"ref_audio": load_audio(reference["path"], sample_rate=model.sample_rate),
               "lang_code": job["language"], "verbose": False}
    if model_name == "qwen":
        options.update(lang_code={"ru": "russian", "en": "english"}[job["language"]])
    options.update(controls)
    parts = [np.asarray(result.audio, dtype=np.float32).reshape(-1)
             for result in model.generate(job["text"], **options)]
    if not parts or not all(part.size and np.isfinite(part).all() for part in parts):
        raise ValueError("The model returned empty or invalid audio.")
    sf.write(job["output"], np.concatenate(parts), model.sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    render(sys.argv[1], json.load(sys.stdin))
