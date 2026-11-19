"""Prepare independent synthesis candidates and review the resulting speech."""

from voice_studio.mod_runtime import mod_scope

import re
from uuid import uuid4

from voice_studio.prosody.morpheme_graph import GRAPH_PAUSE_MIN_MS, fragment, word_graph
from voice_studio.prosody.prosody import TOKENS, analyze, phrase_plan
from voice_studio.speech.recognition import recognize
from voice_studio.prosody.stress import ACUTE, VOWELS, resources
from voice_studio.prosody.stress_char_runtime import model_available, predict_eligible
from voice_studio.prosody.stress_context import load_context_model, predict as predict_context

VERSION = "revolver-2"


def mark_lapshi(text):
    """Encode the requested stress for the local accent-tune checkpoint."""
    return re.sub(r"\b(лапш)(и)\b", r"\1+\2", text, flags=re.IGNORECASE)


@mod_scope
def review_prosody(text, language):
    """Attach model proposals to the existing preview without altering synthesis input."""
    draft = analyze(text, language, experimental=False)
    draft["shadowStress"] = shadow_stress(text, draft["stress"]) if language == "ru" else []
    return draft


def shadow_stress(text, baseline):
    """Compare learned context and character predictions with the current preview marks."""
    by_start = {mark["start"]: mark for mark in baseline}
    context = load_context_model()
    lexical = model_available()
    ambiguous = resources()[1] if lexical else {}
    changes = []
    for match in TOKENS.finditer(text):
        word = match.group()
        base = word.casefold()
        mark = by_start.get(match.start(), {})
        if "ё" in base or ACUTE in word or mark.get("source") == "manual":
            continue
        left, right = text[:match.start()], text[match.end():]
        proposal = predict_context(base, left, right, context)
        source = "context-model"
        # A character model cannot distinguish two senses of the same spelling.
        if proposal is None and lexical and mark.get("source") == "abstained" and base not in ambiguous:
            proposal = predict_eligible(word)
            source = "character-model"
        if not proposal:
            continue
        position, margin = proposal
        if not (0 <= position < len(word) and word[position].casefold() in VOWELS):
            continue
        if position == mark.get("index"):
            continue
        changes.append({"word": word, "start": match.start(), "index": position, "source": source,
                        "proposal": word[:position+1] + ACUTE + word[position+1:],
                        "margin": round(margin, 4)})
    return changes


def review_words(root, language, text, audio):
    """Align Whisper words and PCM silence against the unmodified script."""
    segments, quiet = recognize(root, language, audio, words=True, word_pause_min_ms=GRAPH_PAUSE_MIN_MS)
    marks = analyze(text, language, experimental=False)["stress"]
    return word_graph(text, segments, language=language, stress_marks=marks, quiet=quiet)


def fragment_trials(text, index, count, context="short"):
    """Return independently seeded attempts around a selected word."""
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("Выбери от 1 до 3 повторов.")
    if not isinstance(context, str) or context not in {"short", "wide"}:
        raise ValueError("Выбери короткий или расширенный контекст.")
    excerpt = fragment(text, index, fragment_radius(context))
    return excerpt, [(uuid4().int % 2**32, "word-retry", [(excerpt, 0)]) for _ in range(count)]


def fragment_radius(context):
    """Translate the selected context size to a word radius."""
    return 4 if context == "wide" else 2


@mod_scope
def synthesis_variants(text, language, model, settings):
    """Produce one baseline and supported opt-in variants for the same seed."""
    if not settings:
        return [("baseline", [(text, 0)])]
    result = [("baseline", [(text, 0)])]
    if settings.get("stress") and language == "ru" and model == "f5":
        marked = analyze(text, language, experimental=False)["markedText"]
        if marked != text:
            result.append(("stress", [(marked, 0)]))
    tone = settings.get("tone")
    if tone and model in {"f5", "coqui"}:
        # Test the engine's own response to a terminal punctuation change.
        ending = {"question": "?", "exclamation": "!"}[tone]
        candidate = re.sub(r"[.!?…]*$", "", text) + ending
        if candidate != text and not text.endswith(('"', '»', '”', ')', ']', '}')):
            result.append(("tone-" + tone, [(candidate, 0)]))
    if settings.get("pauses") and model in {"f5", "coqui"}:
        phrases = phrase_plan(text)
        # A bounded number of segments limits repeated inference and seam artifacts.
        if 2 <= len(phrases) <= 4 and "".join("".join(p["text"].split()) for p in phrases) == "".join(text.split()):
            result.append(("pauses", [(p["text"], p["pauseMs"]) for p in phrases]))
    return result
