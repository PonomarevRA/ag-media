"""Adapt a bounded browser image upload to the same local vision worker used by screenshot MCP."""

from hashlib import file_digest
import json
from pathlib import Path
import subprocess
from time import perf_counter

from voice_studio.core.analysis_stages import ignore_progress, stage
from voice_studio.vision.image_input import IMAGE_SLOT, MAX_SCREENSHOT_BYTES, verify_image
from voice_studio.vision.screen_markdown import screenshot_markdown
from voice_studio.vision.video_revolver import VISION, run_worker
from voice_paths import temporary_directory


def image_options(query):
    """Validate the two independent layers and OCR detail mode before any inference."""
    mode = query.get("mode", ["full"])[0]
    ui, objects = query.get("ui", ["1"])[0], query.get("objects", ["1"])[0]
    if mode not in {"overview", "full"} or ui not in {"0", "1"} or objects not in {"0", "1"}:
        raise ValueError("Некорректные настройки анализа изображения.")
    if ui == objects == "0":
        raise ValueError("Выбери предметы или распознавание интерфейса.")
    language = query.get("markdown", [None])[0]
    if language not in {None, "ru", "en"}:
        raise ValueError("Выбери русский или английский язык Markdown.")
    return {"mode": mode, "include_ui": ui == "1", "include_objects": objects == "1",
            **({"markdown_language": language} if language else {})}


def analyze_image(path, options, *, emit=ignore_progress):
    """Validate PNG/JPEG and return the source hash, evidence and model states without transient file paths."""
    started = perf_counter()
    source = Path(path)
    if not 0 < source.stat().st_size <= MAX_SCREENSHOT_BYTES:
        raise ValueError("Картинка должна занимать не больше 24 МиБ.")
    if not IMAGE_SLOT.acquire(blocking=False):
        raise ValueError("Сейчас анализируется другое изображение. Попробуй ещё раз через несколько секунд.")
    try:
        with stage(emit, "image_validate"):
            verify_image(source)
            with source.open("rb") as handle:
                digest = file_digest(handle, "sha256").hexdigest()
        with stage(emit, "image_vision"), temporary_directory() as directory:
            settings = {key: value for key, value in options.items() if key != "markdown_language"}
            settings["work_dir"] = str(directory)
            try:
                result = run_worker("voice_studio.vision.video_scene", ["--screenshot", source, json.dumps(settings)], [VISION], timeout=110)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise ValueError("Не удалось завершить анализ картинки. Попробуй быстрый режим или повтори запрос.") from exc
        if not options["include_objects"]:
            result.update(objects=[], objectStatus="skipped")
        unavailable = (result.get("objectStatus") in {"unavailable", "failed"}
                       or result.get("uiStatus") in {"partial", "unavailable"})
        if language := options.get("markdown_language"):
            result["markdown"] = screenshot_markdown(result, language)
        return {**result, "schemaVersion": "ag-media-image-analysis/1", "status": "partial" if unavailable else "ok",
                "coordinateSpace": "normalized-oriented-image", "requestElapsedMs": round((perf_counter() - started) * 1000, 1),
                "source": {"sha256": digest, "bytes": source.stat().st_size}}
    finally:
        IMAGE_SLOT.release()
