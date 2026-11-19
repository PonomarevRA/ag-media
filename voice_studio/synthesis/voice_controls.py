"""Explicit pronunciation conditioning supported by the pinned local MLX backends."""

MODES = {"source", "neutral"}


def pronunciation_mode(value, language):
    """Validate a conditioning request; neutral is an English candidate, not an accent guarantee."""
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("Unknown pronunciation mode: use source or neutral.")
    if value == "neutral" and language != "en":
        raise ValueError("Neutral pronunciation is currently available only for English.")
    return value


def generation_options(model, job):
    """Keep legacy conditioning by default; reduce reference-language transfer only on request."""
    if model not in {"qwen", "chatterbox"}:
        raise ValueError("Pronunciation controls require a supported local MLX model.")
    mode = pronunciation_mode(job.get("pronunciationMode", "source"), job["language"])
    options = {}
    if model == "qwen":
        # Pinned MLX chooses speaker-embedding-only generation when ref_text is None.
        options["ref_text"] = job["references"][0]["text"] if mode == "source" else None
    elif mode == "neutral":
        options["cfg_weight"] = 0.0
    if "maxTokens" in job:
        maximum = job["maxTokens"]
        if type(maximum) is not int or not 32 <= maximum <= 2048:
            raise ValueError("maxTokens must be an integer between32 and2048.")
        options["max_tokens"] = maximum
    return options
