"""Keep clip recognition languages separate from studio languages and live capture."""

from typing import Literal, get_args

ClipLanguage = Literal["ru", "en", "ar", "ur", "bn", "hi", "ps"]
ClipLanguageMode = Literal["auto", "ru", "en", "ar", "ur", "bn", "hi", "ps"]

STUDIO_LANGUAGES = frozenset(("ru", "en"))
CLIP_LANGUAGES = frozenset(get_args(ClipLanguage))
LANGUAGE_LABELS = {
    "auto": "Автоопределение", "ru": "Русский", "en": "Английский",
    "ar": "Арабский", "ur": "Урду", "bn": "Бенгальский", "hi": "Хинди", "ps": "Пушту",
}
