"""Keep clip recognition languages separate from studio languages and live capture."""

from typing import Literal, get_args

# Language tokens shared by multilingual Whisper Base and Small (IDs 0–98).
# Cantonese is a separate token in Large v3, which is only a recovery model here.
ClipLanguage = Literal[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
]
ClipLanguageMode = Literal["auto", ClipLanguage]

STUDIO_LANGUAGES = frozenset(("ru", "en"))
WHISPER_LANGUAGES = get_args(ClipLanguage)
CLIP_LANGUAGES = frozenset(WHISPER_LANGUAGES)
