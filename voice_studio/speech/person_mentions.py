"""Find text positions with the local ag-privacy model and leave cue text intact."""

import json
import logging
import os
from pathlib import Path
import re
import subprocess

from voice_paths import dependency_path

BRIDGE = dependency_path("person-mentions", "bin", "PersonMentions", "release", "PersonMentions.dll")
LOGGER = logging.getLogger(__name__)
LABELS = {"имя", "фамилия", "отчество", "фио"}
NON_PERSON_PHRASES = {"что такое", "понял", "поняла", "понятно", "всё понятно", "все понятно",
                      "хорошо", "отлично", "спасибо", "добрый день", "добрый вечер", "доброе утро"}
BEFORE_GENERATION = re.compile(r"\bу\s+$", re.IGNORECASE)


def clean_mentions(text, spans):
    """Validate ordered UTF-16 intervals against the exact cue seen by the browser."""
    if not isinstance(spans, list):
        return []
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    limit = len(encoded) // 2
    end = 0
    result = []
    for span in spans[:128]:
        if not isinstance(span, dict):
            continue
        left, right = span.get("start"), span.get("end")
        if type(left) is not int or type(right) is not int or not end <= left < right <= limit:
            continue
        try:
            value = encoded[left * 2:right * 2].decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        normalized = value.casefold().strip()
        phrase = re.sub(r"[^\w\s]", "", normalized)
        phrase = " ".join(phrase.split())
        if not normalized or normalized in LABELS or phrase in NON_PERSON_PHRASES:
            continue
        # Whisper can capitalize the common noun in «у зумеров». Keep an explicit full name.
        if normalized in {"зумера", "зумеров"}:
            prefix = encoded[:left * 2].decode("utf-16-le", errors="replace")
            if BEFORE_GENERATION.search(prefix):
                continue
        result.append({"start": left, "end": right, "type": "PERSON"})
        end = right
    return result


def annotate_person_mentions(segments, language):
    """Add spans to Russian cues in one bounded local inference call per batch."""
    if language != "ru":
        return
    # Reimported automatic annotations must pass the current filter too.
    for cue in segments:
        if isinstance(cue, dict) and isinstance(cue.get("text"), str) and "personMentions" in cue:
            cue["personMentions"] = clean_mentions(cue["text"], cue["personMentions"])
    pending = [cue for cue in segments if isinstance(cue, dict) and "personMentions" not in cue
               and isinstance(cue.get("text"), str) and cue["text"]]
    if not pending:
        return
    privacy = Path(os.environ.get("AG_PRIVACY_ROOT") or Path.home() / "Documents/ai/ag-privacy")
    model = Path(os.environ.get("AG_PRIVACY_PERSON_MODEL") or privacy / "models/person-prose.bin")
    bridge = Path(os.environ.get("VOICE_PERSON_MENTIONS_BRIDGE") or BRIDGE)
    if not bridge.is_file() or not model.is_file():
        return
    for offset in range(0, len(pending), 256):
        batch = pending[offset:offset + 256]
        texts = [cue["text"] for cue in batch]
        try:
            completed = subprocess.run(
                ["dotnet", str(bridge), str(model)], input=json.dumps(texts, ensure_ascii=False),
                encoding="utf-8", capture_output=True, check=True, timeout=5,
            )
            rows = json.loads(completed.stdout)["rows"]
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("incomplete person inference")
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            LOGGER.warning("Local person inference is unavailable.")
            return
        for cue, text, spans in zip(batch, texts, rows):
            cue["personMentions"] = clean_mentions(text, spans)
