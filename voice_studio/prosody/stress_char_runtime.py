"""Dependency-free inference for the local Russian character stress model."""

from voice_studio.mod_runtime import model_resource

from array import array
from bisect import bisect_left
from functools import lru_cache
from hashlib import blake2b
from itertools import pairwise
from pathlib import Path
import json
from math import isfinite
import re
import struct
import sys

from voice_paths import PROJECT_DATA, model_path

ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
VOWELS = frozenset("аеёиоуыэюя")
CHAR_IDS = {letter: index for index, letter in enumerate(ALPHABET)}
WORD = re.compile("[а-яё]{1,64}")
MODEL_PATH = PROJECT_DATA / "prosody/stress-revolver/lexical/selected.json.gz"
PACKED_PATH = model_path("prosody", "selected.json.packed", legacy=MODEL_PATH.with_suffix(".packed"))
PACKED_MAGIC = b"VSAFFIX1"
MAX_WEIGHTS = 2_000_000


def feature_id(feature):
    """Map a character-pattern feature to its stable 64-bit runtime ID."""
    digest = blake2b(feature.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class PackedWeights:
    """Read sorted feature IDs and float32 weights without a large Python dictionary."""

    def __init__(self, handle, size):
        """Load fixed-width arrays stored in little-endian order."""
        self.keys = array("Q")
        self.values = array("f")
        self.keys.fromfile(handle, size)
        self.values.fromfile(handle, size)
        if sys.byteorder != "little":
            self.keys.byteswap()
            self.values.byteswap()
        if any(first >= second for first, second in pairwise(self.keys)):
            raise ValueError("Unordered stress feature IDs.")
        if not all(map(isfinite, self.values)):
            raise ValueError("Invalid stress weights.")

    def get(self, feature, default=0):
        """Find a weight by stable hash or return the zero contribution."""
        key = feature_id(feature)
        index = bisect_left(self.keys, key)
        if index < len(self.keys) and self.keys[index] == key:
            return self.values[index]
        return default


def candidates(word):
    """Return valid vowel positions, character windows and normalized positional features."""
    if not WORD.fullmatch(word):
        return []
    letters = [CHAR_IDS[letter] for letter in word]
    positions = [i for i, letter in enumerate(word) if letter in VOWELS]
    fixed = (letters[:3] + [0] * 3)[:3] + ([0] * 5 + letters)[-5:]
    padded = [0] * 4 + letters + [0] * 4
    rows = []
    for rank, index in enumerate(positions):
        features = [min(16, value) / 16 for value in (
            index, len(word) - index - 1, rank, len(positions) - rank - 1, len(word), len(positions),
        )]
        rows.append((index, padded[index:index + 9] + fixed, features))
    return rows


def load_model():
    """Resolve the selected packed artifact before consulting the model cache."""
    return _load_model(str(model_resource('prosody.lexical', 'builtin.stress-lexical', PACKED_PATH)))


@lru_cache(maxsize=4)
def _load_model(path):
    """Load one prepared checkpoint and disable shadow if the weights are unreadable."""
    try:
        with Path(path).open("rb") as handle:
            if handle.read(len(PACKED_MAGIC)) != PACKED_MAGIC:
                return None
            header_size, size = struct.unpack("<II", handle.read(8))
            if not (0 < header_size <= 4096 and 0 < size <= MAX_WEIGHTS):
                return None
            if Path(path).stat().st_size != 16 + header_size + size * 12:
                return None
            model = json.loads(handle.read(header_size))
            if not valid_packed_header(model):
                return None
            model["weights"] = PackedWeights(handle, size)
            return model
    except (OSError, ValueError, EOFError, struct.error):
        return None


load_model.cache_clear = _load_model.cache_clear
load_model.cache_info = _load_model.cache_info


def valid_packed_header(model):
    """Validate metadata bounds before allocating the runtime arrays."""
    return (isinstance(model, dict) and model.get("kind") == "affix"
            and isinstance(model.get("version"), str)
            and type(model.get("width")) is int and 1 <= model["width"] <= 16
            and (model.get("margin") is None or
                 type(model["margin"]) in (float, int) and 0 <= model["margin"] <= 100))


def model_available():
    """Check for prepared weights before prediction without loading their arrays."""
    path = model_resource('prosody.lexical', 'builtin.stress-lexical', PACKED_PATH)
    if _load_model.cache_info().currsize:
        return _load_model(str(path)) is not None
    return path.is_file()


def score_rows(rows, model):
    """Score each candidate from trained embeddings and affine/ReLU layers."""
    ranked = []
    for index, chars, features in rows:
        vector = []
        for char in chars:
            vector.extend(model["embedding"][char])
        vector.extend(features)
        for layer in model["layers"]:
            vector = [max(0.0, bias + sum(a * b for a, b in zip(weights, vector)))
                      for weights, bias in zip(layer["weights"], layer["bias"])]
        score = model["output_bias"] + sum(a * b for a, b in zip(model["output_weights"], vector))
        ranked.append((score, index))
    return sorted(ranked, reverse=True)


def predict_loaded(word):
    """Return a cached proposal and margin from the service checkpoint."""
    path = model_resource('prosody.lexical', 'builtin.stress-lexical', PACKED_PATH)
    return _predict_loaded(str(path), word)


@lru_cache(maxsize=1024)
def _predict_loaded(path, word):
    """Cache word proposals separately for each selected packed checkpoint."""
    model = _load_model(path)
    return predict(word, model) if model else None


def predict_eligible(word):
    """Return a shadow proposal only above the threshold chosen on calibration."""
    normalized = word.casefold()
    if not WORD.fullmatch(normalized) or sum(char in VOWELS for char in normalized) < 2:
        return None
    model = load_model()
    if not model or model["margin"] is None:
        return None
    result = predict_loaded(normalized)
    return result if result and result[1] >= model["margin"] else None


def predict(word, model):
    """Return (character index, top-two logit margin), or None for invalid inputs."""
    normalized = word.casefold()
    if model.get("kind") == "affix":
        if not WORD.fullmatch(normalized):
            return None
        weights, width = model["weights"], model["width"]
        scores = sorted((sum(weights.get(feature, 0) for feature in affix_features(normalized, index, width)), index)
                        for index, char in enumerate(normalized) if char in VOWELS)
        if len(scores) < 2:
            return None
        second, best = scores[-2:]
        return best[1], best[0] - second[0]
    rows = candidates(normalized)
    if len(rows) < 2:
        return None
    best, second, *_ = score_rows(rows, model)
    return best[1], best[0] - second[0]


def affix_features(word, index, width):
    """Describe recurring stressed endings and local character patterns."""
    position = len(word) - index - 1
    count = sum(char in VOWELS for char in word[index + 1:])
    yield f"rank:{sum(char in VOWELS for char in word[:index])}:{count}"
    yield f"vowel:{word[index]}:{count}"
    for size in range(1, min(width, len(word)) + 1):
        yield f"s:{word[-size:]}:{min(15, position)}"
        yield f"r:{word[-size:]}:{count}"
        if size <= 4:
            yield f"p:{word[:size]}:{min(15, index)}"
    padded = "#" * 4 + word + "#" * 4
    center = index + 4
    for before in range(4):
        for after in range(4):
            if before or after:
                local = padded[center - before:center + after + 1]
                yield f"w:{before}:{local}"
                yield f"wc:{before}:{local}:{min(4, count)}"
