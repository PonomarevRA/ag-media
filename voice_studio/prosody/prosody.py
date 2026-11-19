"""Plan opt-in prosody and separate transcript corrections from canonical text."""
import re
from voice_studio.speech.asr_context_graph import corrections
from voice_studio.prosody.stress import ACUTE, VOWELS, marked_index, resources, stress_scores

VERSION = "ru-prosody-1"
TOKENS = re.compile(r"[А-Яа-яЁё]+(?:\u0301[А-Яа-яЁё]*)?|[A-Za-z][A-Za-z0-9+#]*(?:[.-][A-Za-z0-9+#]+)*")
MANUAL = {"шариков": "ш+ариков"}
TERMS = {
    "api": "API", "tts": "TTS", "sql": "SQL", "http": "HTTP", "https": "HTTPS",
    "f5-tts": "F5-TTS", "python": "Python", "docker": "Docker", "kubernetes": "Kubernetes",
    "github": "GitHub", "javascript": "JavaScript", "coqui": "Coqui", "whisper": "Whisper",
    "mlflow": "MLflow", "json": "JSON", "backend": "backend", "frontend": "frontend",
    "commit": "commit", "debug": "debug", "react": "React", "git": "Git",
}


def context_mark(word, left, right):
    """Resolve common homographs only when an explicit cue is present."""
    neighbors = " ".join((left, right))
    if word == "замок":
        if re.search(r"\b(?:дверной|навесной|ключом|двери)\b", neighbors):
            return "зам+ок"
        if re.search(r"\b(?:старинный|средневековый|рыцарский|крепость)\b", neighbors):
            return "з+амок"
    if word == "мука":
        if re.search(r"\b(?:пшеничная|ржаная|тесто|печь)\b", neighbors):
            return "мук+а"
        if re.search(r"\b(?:невыносимая|страдание|боль)\b", neighbors):
            return "м+ука"
    return None


def lexical_stress(word, left, right, assets, experimental):
    """Prefer manual marks, context, known forms, and finally model predictions."""
    base = word.casefold()
    if ACUTE in base:
        return None, "manual"
    if sum(c in VOWELS for c in base) < 2:
        return None, None
    lexicon, ambiguous, model = assets
    marked = MANUAL.get(base)
    if marked:
        return marked_index(marked), "dictionary"
    if "ё" in base:
        return base.index("ё"), "yo"
    if base in ambiguous:
        marked = context_mark(base, left, right)
        return (marked_index(marked), "context") if marked else (None, "abstained")
    if base in lexicon:
        return marked_index(lexicon[base]), "dictionary"
    if model.get("weights"):
        best, second, *_ = stress_scores(base, model["weights"])
        margin = model.get("margin")
        if margin is not None and best[0] - second[0] >= margin:
            return best[1], "model"
        if experimental:
            return best[1], "model-review"
    return None, "abstained"


def phrase_plan(text):
    """Suggest boundaries and intonation labels from the original punctuation."""
    result = []
    # Only punctuation followed by whitespace and a new sentence can force a cut.
    boundaries = r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ])"
    abbreviations = re.compile(r"(?:\b[А-Яа-яA-Za-z]|\b(?:гг?|им|ул|рис|стр|см|проф|т))\.$", re.I)
    phrase = ""
    for part in re.split(boundaries, text):
        phrase = (phrase + " " + part).strip()
        if not phrase or abbreviations.search(phrase):
            continue
        ending = phrase.rstrip('"»”\')]}').rstrip()
        tone = "question" if ending.endswith("?") else "exclamation" if ending.endswith("!") else "statement"
        result.append({"text": phrase, "pauseMs": 280, "tone": tone})
        phrase = ""
    if phrase:
        result.append({"text": phrase, "pauseMs": 0, "tone": "statement"})
    if result:
        result[-1]["pauseMs"] = 0
    return result


def analyze(text, language, experimental=True):
    """Prepare reviewable marks and narrow, contextual spelling suggestions."""
    assets = resources() if language == "ru" else ({}, {}, {})
    marks, terms = [], []
    pieces = []
    cursor = 0
    lower = text.casefold()
    if language == "ru":
        for match in TOKENS.finditer(text):
            start, end = match.span()
            word = match.group()
            base = word.casefold()
            left = re.split(r"[.!?…]", lower[max(0, start-45):start])[-1]
            right = re.split(r"[.!?…]", lower[end:end+45])[0]
            if base in TERMS:
                terms.append({"word": word, "canonical": TERMS[base], "start": start})
            position, source = lexical_stress(word, left, right, assets, experimental)
            if source:
                marks.append({"word": word, "start": start, "index": position, "source": source})
            pieces.append(text[cursor:start])
            if position is not None:
                word = word[:position+1] + ACUTE + word[position+1:]
            pieces.append(word)
            cursor = end
    pieces.append(text[cursor:])
    marked = "".join(pieces)
    # Commas stay with their surrounding speech; only sentence boundaries are candidates for forced pauses.
    return {"version": VERSION, "modelReady": bool(assets[2]), "markedText": marked,
            "stress": marks, "terms": terms, "corrections": corrections(text, language), "phrases": phrase_plan(text)}

