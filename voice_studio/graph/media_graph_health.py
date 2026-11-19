"""Expose evidence gaps without interpreting detection counts as people or model accuracy."""

import unicodedata

SCRIPTS = {"ru": "CYRILLIC", "en": "LATIN", "ar": "ARABIC", "ur": "ARABIC", "ps": "ARABIC",
           "hi": "DEVANAGARI", "bn": "BENGALI"}
MAX_LABELS = 100


def script_check(language, labels):
    """Flag a dominant incompatible writing system; Latin terms are allowed in non-English speech."""
    expected = SCRIPTS.get(language)
    expected_count = other = latin = 0
    for label in labels:
        for char in label:
            if not char.isalpha():
                continue
            script = unicodedata.name(char, "UNKNOWN").split()[0]
            if script == expected:
                expected_count += 1
            elif script == "LATIN":
                latin += 1
            else:
                other += 1
    total = expected_count + other
    status = "not-enough-evidence"
    if expected and total >= 20:
        status = "unexpected-script" if other / total > .6 else "no-script-conflict"
    return {"status": status, "expectedScript": expected, "expectedLetters": expected_count,
            "otherLetters": other, "allowedLatinLetters": latin,
            "basis": "Unicode writing system only; not language detection or ASR accuracy"}


def data_health(db, language, counts):
    """Read bounded text diagnostics and identity-link counts from the frozen graph."""
    rows = db.execute("SELECT label FROM nodes WHERE kind='utterance' ORDER BY start_ms, rowid LIMIT ?", (MAX_LABELS,))
    labels = [row[0] for row in rows]
    attributed = db.execute("SELECT count(DISTINCT source) FROM edges WHERE relation='spoken_by'").fetchone()[0]
    appearances = db.execute("SELECT count(*) FROM nodes WHERE kind='face' AND json_extract(data, '$.appearance') IS NOT NULL").fetchone()[0]
    speech = counts.get("utterance", 0)
    return {"speechUtterances": speech, "speechWithSpeaker": attributed,
            "faceRecords": counts.get("face", 0), "faceRecordsWithAppearance": appearances,
            "identityNote": "Face IDs are observations/identities from detectors, not a verified count of people.",
            "transcriptScript": {**script_check(language, labels), "inspectedUtterances": len(labels),
                                 "totalUtterances": speech, "sampleLimit": MAX_LABELS, "sampled": speech > len(labels)}}
