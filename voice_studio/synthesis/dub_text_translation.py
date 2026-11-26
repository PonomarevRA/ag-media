"""Route transcript text to the selected translation mod and preserve cue order."""

import re

from voice_studio.mod_contract_bridge import validate_provider_output
from voice_studio.mod_runtime import execution_scope
from voice_studio.mods import ModError
from voice_studio.synthesis.apple_translation import CAPABILITY, ERRORS, MOD_ID, request_translation

MAX_CUES = 400
MAX_TEXT = 1000
MAX_REQUEST_BYTES = 5 * 1024 * 1024
LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")


def language_pair(source, target):
    """Validate explicit language tags before passing them to a local provider."""
    if any(not isinstance(code, str) or not LANGUAGE_TAG.fullmatch(code) for code in (source, target)):
        raise ValueError("Укажи исходный язык и язык перевода.")
    return source, target


def text_batch(payload):
    """Validate Studio and MCP requests before entering a model process."""
    if not isinstance(payload, dict):
        raise ValueError("Ожидались языки и реплики для перевода.")
    source, target = language_pair(payload.get("sourceLanguage"), payload.get("targetLanguage"))
    texts = payload.get("texts")
    if (not isinstance(texts, list) or not 1 <= len(texts) <= MAX_CUES
            or any(not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT for text in texts)):
        raise ValueError("Для перевода нужно от 1 до 400 реплик, до 1000 знаков в каждой.")
    return {"sourceLanguage": source, "targetLanguage": target, "texts": texts}


def translation_status(source, target):
    """Report provider availability without doing translation or requesting a download."""
    source, target = language_pair(source, target)
    with execution_scope() as snapshot:
        try:
            provider = snapshot.selected(CAPABILITY, default=MOD_ID)
        except ModError:
            return {"status": "disabled", "message": "Включи мод перевода в разделе «Моды и модели»."}
        if source == target or provider != MOD_ID:
            return {"provider": provider, "status": "ready"}
        result = request_translation({"sourceLanguage": source, "targetLanguage": target, "texts": []})
        status = result.get("status")
        if status == "ready":
            return {"provider": provider, "status": status}
        return {"provider": provider, "status": status if status in ERRORS else "failed",
                "message": ERRORS.get(status, ERRORS["failed"])}


def translate_apple(payload):
    """Run the local Apple mod through the same batch boundary as manual MCP invocations."""
    batch = text_batch(payload)
    if batch["sourceLanguage"] == batch["targetLanguage"]:
        return {"translations": batch["texts"]}
    reply = request_translation(batch)
    if reply.get("status") != "ready":
        raise ValueError(ERRORS.get(reply.get("status"), ERRORS["failed"]))
    return {"translations": reply.get("translations")}


def translate_text(payload):
    """Enforce the same result contract for the system mod and replacements."""
    batch = text_batch(payload)
    if batch["sourceLanguage"] == batch["targetLanguage"]:
        return {"translations": batch["texts"]}
    with execution_scope() as snapshot:
        provider = snapshot.selected(CAPABILITY, default=MOD_ID)
        if "provider" in payload and payload["provider"] != provider:
            raise ValueError("Мод перевода изменился. Открой настройки дубляжа заново.")
        result = snapshot.invoke_mod(provider, CAPABILITY, batch)
    validate_provider_output(CAPABILITY, batch, result)
    return {"translations": result["translations"]}
