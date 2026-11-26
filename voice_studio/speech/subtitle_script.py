"""Convert the writing system of Chinese subtitle text with bundled OpenCC dictionaries."""

from functools import lru_cache

MAX_EXPORT_BYTES = 5 * 1024 * 1024


@lru_cache(maxsize=1)
def simplified_converter():
    """Load the dictionaries once when simplified subtitles are first requested."""
    from opencc import OpenCC

    return OpenCC("t2s")


def simplified_subtitles(payload):
    """Convert Han characters while retaining SRT cue numbers, timestamps and separators."""
    if not isinstance(payload, dict) or payload.get("language") != "zh":
        raise ValueError("Упрощённое письмо доступно для китайских субтитров.")
    srt = payload.get("srt")
    if not isinstance(srt, str) or not srt.strip():
        raise ValueError("Для экспорта нужны субтитры.")
    converter = simplified_converter()
    return {"srt": converter.convert(srt)}
