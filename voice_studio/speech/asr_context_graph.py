"""Suggest Russian ASR corrections using typed local evidence links."""

import re

from voice_studio.prosody.morpheme_graph import TOKEN, normalized

VERSION = "asr-context-1"
HOMOPHONES = {"шеликов": "шариков", "шериков": "шариков", "шаликов": "шариков"}
BOUNDARY = re.compile(r"[.!?…;:\n]+")
WORD_WINDOW = 4

# A link connects an observed context token to a known lexical candidate.
CONTEXT_LINKS = {
    "шариков": {
        "count": frozenset({"пять", "шесть", "5", "6"}),
        "object": frozenset({"мяч", "мячи", "мячик", "маленьких", "воздушных"}),
        "action": frozenset({"надули", "надуть", "надувать", "покатились", "игры", "игра"}),
        "name": frozenset({"фамилия", "фамилии", "подпись", "инженер", "господин", "капитан",
                           "директор", "автор", "врач", "доктор"}),
    },
}


def evidence(words, index, target):
    """Return observed supporting and conflicting links within one clause."""
    links = []
    lo, hi = max(0, index - WORD_WINDOW), min(len(words), index + WORD_WINDOW + 1)
    for neighbor in range(lo, hi):
        if neighbor == index:
            continue
        token = normalized(words[neighbor].group())
        for kind, matches in CONTEXT_LINKS[target].items():
            # A numeral supports this noun only when it quantifies the local phrase.
            if kind == "count" and not 1 <= index - neighbor <= 3:
                continue
            if token in matches:
                links.append({"kind": kind, "word": token})
    return links


def corrections(text, language, contextual=True):
    """Return proposed replacements in Python character offsets, leaving input intact."""
    if language != "ru":
        return []
    suggestions = []
    cursor = 0
    for boundary in BOUNDARY.finditer(text):
        collect(text, cursor, boundary.start(), contextual, suggestions)
        cursor = boundary.end()
    collect(text, cursor, len(text), contextual, suggestions)
    return suggestions


def collect(text, start, end, contextual, suggestions):
    """Collect proposals in one clause, abstaining on conflicting or absent evidence."""
    words = list(TOKEN.finditer(text, start, end))
    for index, match in enumerate(words):
        word = match.group()
        target = HOMOPHONES.get(normalized(word))
        if not target:
            continue
        links = evidence(words, index, target) if contextual else []
        if contextual and (not links or any(link["kind"] == "name" for link in links)):
            continue
        replacement = target.upper() if word.isupper() else target
        if word.istitle():
            replacement = target.capitalize()
        suggestions.append({"word": word, "replacement": replacement, "start": match.start(),
                            "end": match.end(), "evidence": links})


def apply_corrections(text, proposals):
    """Apply a sorted list of nonoverlapping proposals to a copy for evaluation."""
    for fix in reversed(proposals):
        text = text[:fix["start"]] + fix["replacement"] + text[fix["end"]:]
    return text
