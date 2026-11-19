"""Compact character model and dictionary for Russian lexical stress."""
import gzip
import json
from functools import lru_cache
import re

VOWELS = frozenset("аеёиоуыэюя")
WORD = re.compile("[а-яё]+")
from voice_paths import PROJECT_DATA

ROOT = PROJECT_DATA / "prosody"
ACUTE = "\u0301"


def marked_index(marked):
    """Read the RUAccent convention, with a plus immediately before a vowel."""
    index = marked.find("+")
    return index if marked.count("+") == 1 and marked[index + 1:index + 2] in VOWELS else None


def vowel_features(word, index):
    """Describe a candidate vowel using affixes and nearby characters."""
    padded = "###" + word + "###"
    center = index + 3
    yield "vowel:" + word[index]
    for width in (1, 2, 3):
        yield f"left:{padded[center-width:center+1]}"
        yield f"right:{padded[center:center+width+1]}"
        yield f"suffix:{word[-width:]}:{min(5, len(word)-index-1)}"
    yield f"window:{padded[center-1:center+2]}"
    yield f"rank:{sum(c in VOWELS for c in word[:index])}:{sum(c in VOWELS for c in word[index+1:])}"


def stress_scores(word, weights):
    """Rank possible positions by a sum of learned feature weights."""
    return sorted(((sum(weights.get(f, 0) for f in vowel_features(word, i)), i)
                   for i, c in enumerate(word) if c in VOWELS), reverse=True)


@lru_cache(maxsize=1)
def resources():
    """Load compact local data once; retain manual/rule support without assets."""
    def read(name):
        path = ROOT / name
        if not path.exists():
            return {}
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return read("accents_nn.json.gz"), read("omographs.json.gz"), read("stress-model.json")
