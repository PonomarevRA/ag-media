"""Link timed speech to independently observed on-screen text."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import re
from unicodedata import normalize

WORD = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)?", re.UNICODE)
CLICK = re.compile(r"\b(?:click(?:ed)?|tap(?:ped)?|наж(?:ать|мите|ал|али)|клик(?:нуть|ните))\b", re.IGNORECASE)
MAX_DISTANCE_MS = 2200
MAX_EVIDENCE = 3


def tokens(text):
    """Normalize case and decimal separators without erasing visible wording from evidence."""
    return tuple(word.replace(",", ".") for word in WORD.findall(normalize("NFKC", text).casefold()))


def index_terms(words):
    """Use complete multiword phrases and distinctive long word/number anchors."""
    yield from set(word for word in words if len(word) >= 4 and not word[0].isdecimal())
    yield from {f"{left} {right}" for left, right in zip(words, words[1:])
                if len(left) + len(right) >= 7 or left[0].isdecimal() or right[0].isdecimal()}


def add_speech_screen_evidence(graph):
    """Attach a few exact lexical witnesses, preserving both clocks and optional click cues."""
    by_term = defaultdict(list)
    screen = []
    for item in graph.nodes.values():
        if item["kind"] not in ("ui_element", "text_observation"):
            continue
        text = item["data"].get("text") if item["kind"] == "ui_element" else item["label"]
        if isinstance(text, str) and 0 < len(text) <= 180:
            screen.append((item, text))
    for item, text in sorted(screen, key=lambda pair: (pair[0]["startMs"], pair[0]["id"])):
        for term in index_terms(tokens(text)):
            by_term[term].append((item["startMs"], item["id"]))

    for utterance in graph.nodes.values():
        if utterance["kind"] not in ("utterance", "dialogue_draft"):
            continue
        speech_terms = set(index_terms(tokens(utterance["label"])))
        candidates = {}
        first, last = utterance["startMs"], utterance["endMs"]
        middle = (first + last) // 2
        for term in speech_terms:
            timed = by_term.get(term, ())
            start = bisect_left(timed, (first - MAX_DISTANCE_MS, ""))
            end = bisect_right(timed, (last + MAX_DISTANCE_MS, chr(0x10ffff)))
            # Very long cues can repeat the same control on hundreds of frames.
            # Search the bounded samples closest to the cue center.
            center = bisect_left(timed, (middle, ""), start, end)
            for stamp, witness in timed[max(start, center - 6):min(end, center + 6)]:
                candidates.setdefault((stamp, witness), set()).add(term)
        matches = []
        seen = set()
        ordered = sorted(candidates, key=lambda key: (
            -int(any(" " in term for term in candidates[key])),
            0 if first <= key[0] < last else min(abs(first - key[0]), abs(last - key[0])),
            graph.nodes[key[1]]["kind"] != "ui_element", abs(middle - key[0]), key))
        for stamp, witness in ordered:
            evidence = graph.nodes[witness]
            bounds = evidence["data"].get("box")
            caption = evidence["data"].get("text", evidence["label"])[:180]
            canonical = tokens(caption)
            if not bounds or canonical in seen:
                continue
            seen.add(canonical)
            term = max(candidates[(stamp, witness)], key=lambda part: (" " in part, len(part), part))
            row = {"nodeId": witness, "timeMs": stamp, "text": caption, "box": bounds,
                   "match": term, "basis": "matching-visible-text-near-asr"}
            owner = graph.nodes.get(evidence["data"].get("screenId"), {})
            previous = owner.get("data", {}).get("previousStateMs")
            if (previous is not None and first <= stamp <= last + MAX_DISTANCE_MS
                    and evidence["data"].get("role") == "button_candidate"
                    and CLICK.search(utterance["label"])):
                row["actionCandidate"] = "spoken-click-with-screen-state-change"
            matches.append(row)
            graph.connect(utterance["id"], "screen_evidence", witness)
            if len(matches) >= MAX_EVIDENCE:
                break
        if matches:
            utterance["data"]["screenEvidence"] = matches
