"""Editable additive analysis stage using only the Python standard library."""

import re


def run(payload, settings):
    """Count transcript words without mutating the input analysis or writing any files."""
    transcript = payload.get("transcript", {})
    if isinstance(transcript, str):
        text = transcript
    elif isinstance(transcript, dict):
        text = transcript.get("text", "")
        if not text:
            text = " ".join(str(segment.get("text", "")) for segment in transcript.get("segments", []))
    else:
        text = ""
    words = re.findall(r"\w+", str(text), flags=re.UNICODE)
    minimum = settings.get("minimumLength", 1)
    count = 0
    for word in words:
        if len(word) >= minimum:
            count += 1
    return {"wordCount": count, "minimumLength": minimum, "source": "transcript"}
