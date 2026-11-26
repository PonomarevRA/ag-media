"""Explicit delivery styles and reference metadata for saved dub voices."""

import json

STYLES = {"german": "de", "japanese": "ja", "anime": "ja"}
ANIME_PITCH_RATIO = 2 ** (1.5 / 12)


def reference_language(style, language):
    """Accept a native-language sample or a sample already spoken with the accent."""
    if not isinstance(style, str) or style not in STYLES:
        raise ValueError("Выбери немецкий, японский или аниме-профиль.")
    if language not in ("ru", "en", STYLES[style]):
        raise ValueError("Выбери фактический язык записи.")
    return language


def saved_accent(reference_report):
    """Read the accent from a reviewed profile, including rows saved before accents existed."""
    if not reference_report:
        return None
    value = json.loads(reference_report).get("accentStyle")
    return value if isinstance(value, str) and value in STYLES else None
