"""Expose local screenshot OCR and geometry as an editable first-pass description."""

from collections import Counter
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import subprocess
from stat import S_ISREG
from time import perf_counter
from typing import Literal

from voice_paths import temporary_directory
from voice_studio.core.mcp_lanes import AsyncWorkLane
from voice_studio.vision.video_revolver import VISION, run_worker
from voice_studio.vision.screen_session import ScreenSessions, retain
from voice_studio.graph.media_graph_model import box_coordinates

from voice_studio.vision.image_input import (
    MAX_SCREENSHOT_BYTES, MAX_SCREENSHOT_PIXELS, IMAGE_SLOT as SLOT, verify_image,
)
MAX_SCROLL_CAPTURES = 24
MAX_SCROLL_BYTES = 96 * 1024 * 1024
UI_LANE = AsyncWorkLane("screen-ui", workers=1, capacity=1)
SESSIONS = ScreenSessions()
SESSION_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")


def describe_draft(result, language):
    """Summarize visible evidence without inventing control behavior or unreadable text."""
    layout = result.get("interface") or {}
    roles = Counter(element["role"] for element in layout.get("elements", []))
    ocr = result["ocr"]
    labels = list(dict.fromkeys(row["text"] for row in ocr))[:6]
    controls = roles["button_candidate"] + roles["field_candidate"]
    links = len(layout.get("connections", []))
    structure = result.get("structure") or {}
    windows = len(structure.get("windows", []))
    lists = len(structure.get("lists", []))
    clips = len(structure.get("timelineClips", []))
    if language == "en":
        overview = (f"Visible text lines: {len(ocr)}. Candidate buttons and fields: {controls}. "
                    f"Visible connections between blocks: {links}. Windows: {windows}. Lists: {lists}. Timeline clips: {clips}.")
        return overview + (" OCR excerpts: " + " | ".join(labels) if labels else " No text was read.")
    overview = (f"Видимых строк текста — {len(ocr)}. Кандидатов на кнопки и поля — {controls}. "
                f"Видимых связей между блоками — {links}. Окон — {windows}, списков — {lists}, клипов — {clips}.")
    return overview + (" Фрагменты OCR: " + " | ".join(labels) if labels else " Текст не прочитан.")


def screenshot_bytes(path):
    """Copy a bounded regular file before passing any bytes to a decoder."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not S_ISREG(info.st_mode):
                raise ValueError("Нужен обычный файл PNG или JPEG.")
            if info.st_size > MAX_SCREENSHOT_BYTES:
                raise ValueError("Размер скриншота должен быть не больше 24 МиБ.")
            data = stream.read(MAX_SCREENSHOT_BYTES + 1)
    except OSError as exc:
        raise ValueError("Не удалось прочитать локальный скриншот.") from exc
    if not 0 < len(data) <= MAX_SCREENSHOT_BYTES:
        raise ValueError("Размер скриншота должен быть не больше 24 МиБ.")
    return data


def screen_request(file_path, mode, region, session_id):
    """Validate the public screenshot request before reading any selected file."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        raise ValueError("Укажи абсолютный путь к локальному скриншоту PNG или JPEG.")
    if mode not in {"overview", "full"}:
        raise ValueError("Выбери overview или full.")
    box = None if region is None else box_coordinates(region)
    if region is not None and (not box or min(box[2:]) <= 0):
        raise ValueError("Область должна быть нормализованной рамкой [x, y, width, height].")
    if session_id is not None and not SESSION_KEY.fullmatch(session_id):
        raise ValueError("ID сессии должен содержать от 1 до 64 букв ASCII, цифр, дефисов или подчёркиваний.")
    return path, box


def analyze_image_bytes(data, mode, box, session_dir):
    """Probe and recognize the same private staged bytes."""
    with temporary_directory() as directory:
        copy = Path(directory) / "screenshot"
        copy.write_bytes(data)
        verify_image(copy)
        options = {"mode": mode, "region": box, "session_dir": str(session_dir) if session_dir else None,
                   "work_dir": directory}
        try:
            return run_worker("voice_studio.vision.video_scene", ["--screenshot", copy, json.dumps(options)], [VISION], timeout=110)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ValueError("Локальное распознавание скриншота не завершилось.") from exc


def screen_ids(result):
    """Give unchanged visible elements comparable identifiers across calls."""
    def assign(item, role, text=""):
        """Hash one role, visible label and position independently of the screenshot file."""
        box = [round(point, 3) for point in item.get("box") or []]
        signature = json.dumps([role, text.casefold(), box], ensure_ascii=False)
        item["elementId"] = sha256(signature.encode()).hexdigest()[:16]

    for item in (result.get("interface") or {}).get("elements", []):
        assign(item, item["role"], item.get("text") or "")
    structure = result.get("structure") or {}
    for item in structure.get("windows", []):
        assign(item, "window", item.get("title") or "")
    for item in structure.get("lists", []):
        assign(item, "list")
        for row in item["rows"]:
            assign(row, "row", row.get("text") or "")
    for item in structure.get("timelineClips", []):
        assign(item, "timeline-clip", item.get("text") or "")
    return result


def analyze_screenshot(file_path, language, *, mode="full", region=None, session_id=None):
    """Return the local source, draft, and optional cached structure in normalized coordinates."""
    started = perf_counter()
    path, box = screen_request(file_path, mode, region, session_id)
    if not SLOT.acquire(blocking=False):
        raise ValueError("Локальная модель уже анализирует скриншот. Попробуй позже.")
    try:
        path = path.resolve()
        data = screenshot_bytes(path)
        digest = sha256(data).hexdigest()
        session_dir, known = SESSIONS.get(session_id)
        key = digest, mode, tuple(box or ())
        if known is not None and key in known:
            known.move_to_end(key)
            result = {**known[key], "reuse": "exact-image", "ocrJobs": 0, "elapsedMs": 0}
        else:
            result = screen_ids(analyze_image_bytes(data, mode, box, session_dir))
            if known is not None:
                statuses = result["ocrStatuses"]
                if all(statuses.get(f"{prefix}tesseract", statuses.get(f"{prefix}apple-vision", "ok")) == "ok"
                       for prefix in ("", "controls-")):
                    retain(known, key, result)
    finally:
        SLOT.release()
    return {"schemaVersion": "voice-studio-screen-draft/1", "status": "draft",
            "source": {"path": str(path), "sha256": digest},
            "coordinateSpace": "normalized-oriented-image", "draftLanguage": language,
            "draftDescription": describe_draft(result, language), **result,
            "requestElapsedMs": round(1000 * (perf_counter() - started), 1)}


def analyze_scroll_screenshots(file_paths, region=None, timestamps_ms=None, session_id=None):
    """Read one OCR mosaic per verified scroll stretch, reusing an exact session result."""
    if not 2 <= len(file_paths) <= MAX_SCROLL_CAPTURES:
        raise ValueError("Передай от 2 до 24 скриншотов в порядке прокрутки.")
    times = [index * 250 for index in range(len(file_paths))] if timestamps_ms is None else timestamps_ms
    if (len(times) != len(file_paths) or any(type(time) is not int or time < 0 for time in times)
            or any(a >= b for a, b in zip(times, times[1:]))):
        raise ValueError("Таймкоды скриншотов должны строго возрастать.")
    paths = [screen_request(path, "overview", region, session_id)[0] for path in file_paths]
    box = box_coordinates(region) if region is not None else [0, 0, 1, 1]
    if not SLOT.acquire(blocking=False):
        raise ValueError("Локальная модель уже анализирует скриншот. Попробуй позже.")
    try:
        with temporary_directory() as folder:
            root = Path(folder)
            staged, digests, total = [], [], 0
            for index, path in enumerate(paths):
                data = screenshot_bytes(path)
                total += len(data)
                if total > MAX_SCROLL_BYTES:
                    raise ValueError("Серия скриншотов должна занимать до 96 МиБ.")
                digest = sha256(data).hexdigest()
                copy = root / digest
                if not copy.exists():
                    copy.write_bytes(data)
                    verify_image(copy)
                staged.append({"path": str(copy), "timeMs": times[index]})
                digests.append(digest)
            _, known = SESSIONS.get(session_id)
            key = "scroll", tuple(digests), tuple(box), region is not None, tuple(times)
            if known is not None and key in known:
                known.move_to_end(key)
                return {**known[key], "reuse": "exact-sequence", "ocrJobs": 0}
            request = root / "scroll.json"
            request.write_text(json.dumps({"captures": staged, "region": box, "contentOnly": region is not None}),
                               encoding="utf-8")
            result = run_worker("voice_studio.vision.video_scene", ["--scroll-screenshots", request], [VISION], timeout=110)
            result["schemaVersion"] = "voice-studio-screen-scroll/1"
            result["capturedAtMs"] = times
            if known is not None and result["status"] == "draft":
                retain(known, key, result)
            return result
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Локальное распознавание прокрутки не завершилось.") from exc
    finally:
        SLOT.release()


def register_screen_tools(mcp):
    """Register the screen-analysis entry point in the local MCP catalog."""
    @mcp.tool()
    async def screen_analyze(file_path: str, language: Literal["ru", "en"] = "ru",
                             mode: Literal["overview", "full"] = "full",
                             region: list[float] | None = None, session_id: str | None = None) -> dict:
        """Read a local PNG/JPEG screenshot with offline OCR.
        Use overview for one fast OCR pass and a map of visible windows, lists, and timeline clips.
        Use full for detailed OCR of the whole screenshot or a selected normalized [x,y,width,height] region.
        Reuse one session_id across calls to avoid OCR on unchanged pixels. The draft language does not change OCR.
        Uses an independent bounded UI lane while video analysis runs; a simultaneous UI request returns busy.
        """
        return await UI_LANE.run(analyze_screenshot, file_path, language, mode=mode,
                                        region=region, session_id=session_id)

    @mcp.tool()
    async def screen_scroll(file_paths: list[str], region: list[float] | None = None,
                            timestamps_ms: list[int] | None = None, session_id: str | None = None) -> dict:
        """Register overlapping screenshots after small vertical scrolls and OCR each coherent mosaic once.
        Supply two to 24 local PNG/JPEG paths in capture order. Use region [x,y,width,height] for the scrolling content.
        Returned lines point to captureIndex, timeMs and a normalized box on the original screenshot.
        Repeat the same ordered capture sequence with a session_id to reuse its draft without OCR.
        Uses the independent UI lane rather than the long-video analysis queue.
        """
        return await UI_LANE.run(analyze_scroll_screenshots, file_paths, region, timestamps_ms, session_id)
