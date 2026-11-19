"""Sparse local context model for Russian homographs."""

from voice_studio.mod_runtime import model_resource
from functools import lru_cache
from pathlib import Path
import json
from math import isfinite
import re

from voice_studio.prosody.stress import ROOT, VOWELS
from voice_paths import model_path

MODEL_PATH = model_path("prosody", "context.json", legacy=ROOT / "stress-revolver" / "context.json")
WORDS = re.compile(r"[а-яё]+")
NEGATION = {"не", "ни"}
STOP = {"и", "в", "на", "из", "от", "у", "по", "за", "к", "с", "мы", "я", "он", "она", "это", "для"}
# Major punctuation bounds the context while a hyphen and quotation marks can occur within it.
BOUNDARY = re.compile(r"[.!?…;:\n]")


def context_features(left, right, stage):
    """Emit weighted features for a target's left and right sentence fragments."""
    left = WORDS.findall(BOUNDARY.split(left.casefold())[-1])
    right = WORDS.findall(BOUNDARY.split(right.casefold())[0])
    if NEGATION.intersection(left[-4:] + right[:2]):
        return {}
    features = {}
    if stage >= 2 and left and left[-1] not in STOP:
        features["previous:" + left[-1]] = 2.0
    neighbors = [*left[-4:], *right[:3]] if stage >= 3 else [*left[-2:], *right[:2]]
    for word in neighbors:
        if word in STOP:
            continue
        features["word:" + word] = 1.0
        if stage >= 3 and len(word) >= 5:
            features["prefix:" + word[:4]] = 0.5
    return features


def predict(word, left, right, model, *, include_previous=True):
    """Return a vowel index and score gap when learned context favors one sense."""
    senses = model.get("weights", {}).get(word)
    if not senses or len(senses) < 2:
        return None
    features = context_features(left, right, model["stage"])
    if not include_previous:
        features = {key: value for key, value in features.items() if not key.startswith("previous:")}
    scores = []
    for label, weights in senses.items():
        score = sum(weights.get(feature, 0) * value for feature, value in features.items())
        scores.append((score, int(label)))
    scores.sort(reverse=True)
    best, runner_up = scores[:2]
    margin = best[0] - runner_up[0]
    if best[0] <= 0 or margin < model.get("threshold", 1):
        return None
    return best[1], margin


def load_context_model():
    """Resolve the selected local resource before reading cached context weights."""
    return _load_context_model(str(model_resource('prosody.context', 'builtin.stress-context', MODEL_PATH)))


@lru_cache(maxsize=4)
def _load_context_model(path):
    """Read the selected compact local model, or return an empty model on absence."""
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
        if (model.get("stage") not in (1, 2, 3) or not isinstance(model.get("weights"), dict)
                or not isfinite(model.get("threshold", 1)) or model.get("threshold", 1) <= 0):
            return {}
        for word, senses in model["weights"].items():
            if not WORDS.fullmatch(word) or len(senses) < 2:
                return {}
            for position, features in senses.items():
                if (not position.isdecimal() or int(position) >= len(word)
                        or word[int(position)] not in VOWELS):
                    return {}
                if not all(isfinite(weight) for weight in features.values()):
                    return {}
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        return {}
    return model


load_context_model.cache_clear = _load_context_model.cache_clear
