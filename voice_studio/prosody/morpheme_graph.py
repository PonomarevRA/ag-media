"""Build a versioned word alignment and optional dictionary morphology layer."""

import re
import unicodedata

from voice_studio.prosody.stress import ACUTE, VOWELS

TOKEN = re.compile(r"(?:[^\W_]\u0301?)+", re.UNICODE)
RUSSIAN_WORD = re.compile("[а-яё]+")
GRAPH_PAUSE_MIN_MS = 120
BOUNDARY_TOLERANCE_MS = 150
# Only explicit segmentations are displayed. Unlisted words remain whole.
MORPHEMES = {
    "шарик": (("шар", "корень"), ("ик", "суффикс")),
    "шарики": (("шар", "корень"), ("ик", "суффикс"), ("и", "окончание")),
    "шариков": (("шар", "корень"), ("ик", "суффикс"), ("ов", "окончание")),
    "домики": (("дом", "корень"), ("ик", "суффикс"), ("и", "окончание")),
    "домиков": (("дом", "корень"), ("ик", "суффикс"), ("ов", "окончание")),
}


def normalized(text):
    """Normalize orthography for alignment while preserving display text."""
    return unicodedata.normalize("NFC", text).casefold().replace("ё", "е").replace("\u0301", "")


def spoken_tokens(segments):
    """Attach Whisper segment offsets to each token without interpolating time."""
    tokens = []
    for segment in segments:
        offsets = segment.get("offsets", {})
        matches = list(TOKEN.finditer(segment["text"]))
        start, end = offsets.get("from"), offsets.get("to")
        timed = len(matches) == 1 and type(start) is int and type(end) is int and 0 <= start < end
        for match in matches:
            tokens.append({"text": match.group(), "startMs": start if timed else None,
                           "endMs": end if timed else None, "timingSource": "whisper" if timed else None})
    return tokens


def align(expected, actual):
    """Return minimum-edit alignment with deterministic exact-first backtracking."""
    left = [normalized(word) for word in expected]
    right = [normalized(word["text"]) for word in actual]
    # Keep the cost matrix bounded for a hallucinated or unusually long transcript.
    if max(len(left), len(right)) > 1000:
        raise ValueError("Слишком длинная расшифровка для сверки слов.")
    costs = [list(range(len(right) + 1))]
    for i, word in enumerate(left, 1):
        previous = costs[-1]
        row = [i]
        for j, heard in enumerate(right, 1):
            row.append(min(previous[j-1] + (word != heard), previous[j] + 1, row[j-1] + 1))
        costs.append(row)
    i, j = len(left), len(right)
    pairs = []
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1] + (left[i-1] != right[j-1]):
            pairs.append((i-1, j-1, "match" if left[i-1] == right[j-1] else "substitution"))
            i, j = i-1, j-1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            pairs.append((i-1, None, "missing"))
            i -= 1
        else:
            pairs.append((None, j-1, "extra"))
            j -= 1
    return list(reversed(pairs))


def word_prosody(word, mark):
    """Group Russian vowel nuclei into approximate text syllables without audio offsets."""
    plain = word.replace(ACUTE, "").casefold()
    if not RUSSIAN_WORD.fullmatch(plain):
        return [], None
    vowels = [i for i, char in enumerate(plain) if char in VOWELS]
    if not vowels:
        return [], None
    accent = mark["index"] if mark else None
    source = mark["source"] if mark else "unmarked"
    if ACUTE in word:
        explicit = word.index(ACUTE) - 1
        accent = explicit if explicit in vowels and word.count(ACUTE) == 1 else None
        source = "manual"
    elif "ё" in plain and plain.count("ё") == 1:
        accent = plain.index("ё")
        source = "yo"
    # Use a simple consonant grouping for display. It models neither reductions nor acoustic boundaries.
    cuts = [0]
    for left, right in zip(vowels, vowels[1:]):
        onset = right - 1
        if plain[onset] in "ьъй" or onset == left:
            onset = right
        cuts.append(onset)
    cuts.append(len(plain))
    display = word.replace(ACUTE, "")
    syllables = [{"text": display[start:end], "vowelIndex": vowel, "stressed": vowel == accent}
                 for (start, end), vowel in zip(zip(cuts, cuts[1:]), vowels)]
    return syllables, {"index": accent, "source": source}


def boundary_pauses(rows, quiet):
    """Associate each measured PCM interval with its closest unambiguous Whisper boundary."""
    pauses = []
    for interval in quiet:
        start = round(interval["start"] * 1000)
        end = round(interval["end"] * 1000)
        pauses.append({"startMs": start, "endMs": end, "durationMs": end - start, "source": "pcm-rms",
                       "afterIndex": None, "beforeIndex": None})
    edges = []
    for left, right in zip(rows, rows[1:]):
        if left["kind"] != "match" or right["kind"] != "match":
            continue
        stop, start = left["heard"]["endMs"], right["heard"]["startMs"]
        if stop is not None and start is not None and stop <= start:
            edges.append((left["index"], right["index"], stop, start))
    for pause in pauses:
        matches = []
        for left, right, stop, start in edges:
            entrance = abs(pause["startMs"] - stop)
            exit_gap = abs(pause["endMs"] - start)
            if max(entrance, exit_gap) <= BOUNDARY_TOLERANCE_MS:
                matches.append((entrance + exit_gap, left, right))
        matches.sort()
        if matches and (len(matches) == 1 or matches[0][0] < matches[1][0]):
            _, pause["afterIndex"], pause["beforeIndex"] = matches[0]
    return pauses


def word_graph(text, segments, *, language="ru", stress_marks=(), quiet=()):
    """Return text annotations, approximate word alignment, and measured PCM pauses."""
    tokens = list(TOKEN.finditer(text))
    expected = [match.group() for match in tokens]
    actual = spoken_tokens(segments)
    alignment = align(expected, actual)
    marks = {mark["start"]: mark for mark in stress_marks}
    rows = []
    for i, j, kind in alignment:
        word = expected[i] if i is not None else None
        parts = MORPHEMES.get(normalized(word), ()) if word else ()
        syllables, stress = word_prosody(word, marks.get(tokens[i].start())) if word and language == "ru" else ([], None)
        rows.append({"index": i, "expected": word, "heard": actual[j] if j is not None else None,
                     "kind": kind, "morphemes": parts, "syllables": syllables, "stress": stress})
    errors = sum(row["kind"] != "match" for row in rows)
    return {"version": 2, "rows": rows, "pauses": boundary_pauses(rows, quiet),
            "wordErrorRate": errors / len(expected) if expected else None}


def fragment(text, index, radius=2):
    """Retain up to radius neighbors on either side and the original punctuation."""
    tokens = list(TOKEN.finditer(text))
    if type(index) is not int or not 0 <= index < len(tokens):
        raise ValueError("Слово для пересинтеза не найдено.")
    left = max(0, index - radius)
    right = min(len(tokens), index + radius + 1)
    start = tokens[left].start()
    end = tokens[right].start() if right < len(tokens) else len(text)
    return text[start:end].strip()
