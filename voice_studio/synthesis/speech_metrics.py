"""Measure words and speaker similarity independently for a local WAV pair."""

from voice_studio.mod_runtime import model_resource, worker_environment

import importlib.util
import json
import re
import subprocess
import sys

from voice_paths import model_path
from voice_studio.core.worker_process import worker_command
from voice_studio.synthesis.engines import bounded_timeout, python_for, research_enabled
from voice_studio.speech.recognition import transcribe


ERES2NET = model_path("diarization", "embedding.onnx")


def speaker_encoder():
    """Identify the configured voice encoder, independent of the synth engine."""
    return "xtts" if research_enabled() else "eres2net"


def scoring_version():
    """Identify which speaker scale the scores use."""
    return f"{speaker_encoder()}60-whisper40-v1"


def scorer_ready():
    """Verify the local scorer files and runtime for the selected operating mode."""
    if research_enabled():
        return python_for('coqui').is_file() and model_resource('speech.metric-resource', 'builtin.xtts-speaker-encoder', model_path('xtts', 'model.pth')).is_file()
    return model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', ERES2NET).is_file() and importlib.util.find_spec('sherpa_onnx') is not None


def word_error_rate(expected, actual):
    """Compute word edit distance divided by the number of expected words."""
    reference = re.findall(r"\w+", expected.casefold().replace("ё", "е"))
    hypothesis = re.findall(r"\w+", actual.casefold().replace("ё", "е"))
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, 1):
        row = [index]
        for column, other in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[column] + 1,
                           previous[column - 1] + (word != other)))
        previous = row
    return previous[-1] / len(reference)


def measure(root, language, text, original, candidate, *, deadline=None):
    """Return available measurements without making WAV playback depend on scoring."""
    metrics = {}
    try:
        transcript = transcribe(root, language, candidate.read_bytes(), deadline=deadline)
        rate = word_error_rate(text, transcript)
        if rate is not None:
            metrics["whisper_word_error_rate"] = rate
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Whisper evaluation: {exc}", flush=True)
    try:
        name = f"{speaker_encoder()}_encoder_cosine"
        metrics[name] = speaker_cosine(original, candidate, deadline=deadline)
    except ValueError as exc:
        print(f"Speaker evaluation unavailable: {exc}", flush=True)
    return metrics


def speaker_cosine(original, candidate, *, deadline=None):
    """Compare two PCM recordings with the local speaker encoder in a child process."""
    if not scorer_ready():
        raise ValueError("Локальный энкодер голоса недоступен.")
    if research_enabled():
        command = worker_command(python_for("coqui"), "voice_studio.synthesis.speaker_metric",
                                 model_path("xtts", "model.pth"))
    else:
        command = worker_command(sys.executable, "voice_studio.synthesis.speech_metrics", ERES2NET)
    try:
        output = subprocess.check_output(
            [*command, str(original), str(candidate)],
            env=worker_environment(), text=True, timeout=bounded_timeout(deadline, 120), stderr=subprocess.PIPE,
        )
        value = float(json.loads(output)["cosine"])
        if not -1 <= value <= 1:
            raise ValueError("cosine out of range")
        return value
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise ValueError("Энкодер голоса не смог сравнить дорожки.") from exc


def compare_embeddings(weights, original, candidate):
    """Compute cosine using Sherpa ERes2Net over the first 30 seconds of each recording."""
    from voice_studio.speech.speaker_review import embedding_extractor, infer_embeddings

    model = embedding_extractor(weights)
    vectors = [infer_embeddings(path, {0: [(0, 30000)]}, extractor=model)[0]
               for path in (original, candidate)]
    if not all(vectors):
        raise ValueError("Both tracks need enough speech for an embedding.")
    left, right = (items[0] for items in vectors)
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


if __name__ == "__main__":
    print(json.dumps({"cosine": compare_embeddings(*sys.argv[1:])}))
