"""Explicit pronunciation conditioning supported by the pinned local MLX backends."""

MODES = {"source", "neutral"}
NEUTRAL_LANGUAGES = ("en", "ru")


def pronunciation_mode(value, language):
    """Validate the dubbing language when reduced reference pronunciation is requested."""
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("Unknown pronunciation mode: use source or neutral.")
    if value == "neutral" and language not in NEUTRAL_LANGUAGES:
        raise ValueError("Neutral pronunciation is available for English and Russian.")
    return value


def generation_options(model, job):
    """Keep legacy conditioning by default; reduce reference-language transfer only on request."""
    if model not in {"qwen", "chatterbox"}:
        raise ValueError("Pronunciation controls require a supported local MLX model.")
    mode = pronunciation_mode(job.get("pronunciationMode", "source"), job["language"])
    options = {}
    if model == "qwen":
        # With no reference transcript, MLX Qwen retains the speaker embedding
        # and uses the target language without conditioning on reference speech tokens.
        options["ref_text"] = job["references"][0]["text"] if mode == "source" else None
    elif mode == "neutral":
        options["cfg_weight"] = 0.0
    if "maxTokens" in job:
        maximum = job["maxTokens"]
        if type(maximum) is not int or not 32 <= maximum <= 2048:
            raise ValueError("maxTokens must be an integer between 32 and 2048.")
        options["max_tokens"] = maximum
    return options
