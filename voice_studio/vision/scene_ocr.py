"""Run local bilingual OCR and project sector coordinates back onto the video."""

from voice_studio.mod_runtime import model_provider

from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from voice_studio.core.media_resources import stage_threads
from voice_studio.vision.scene_objects import iou
from voice_studio.core.storage_cache import use_cache

from voice_paths import PROJECT_ROOT as ROOT, RUNTIME_DIR, ensure_temporary_root, model_path

TESSDATA = model_path("vision", "tessdata", "rus.traineddata").parent
FULL = (0, 0, 1, 1)
BOTTOM = (0, .40, 1, .59)
CENTER = (.02, .23, .96, .54)


def parse_lines(raw, width, height):
    """Group reliable Tesseract TSV words into independently positioned lines."""
    grouped = {}
    for word in csv.DictReader(StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = word.get("text", "").strip()
        if word["level"] != "5" or not any(char.isalnum() for char in text):
            continue
        score = float(word["conf"])
        if score < 70:
            continue
        left, top, w, h = (int(word[key]) for key in ("left", "top", "width", "height"))
        key = tuple(word[k] for k in ("block_num", "par_num", "line_num"))
        grouped.setdefault(key, []).append((text, score, left, top, left + w, top + h))
    lines = []
    for words in grouped.values():
        left, top = min(word[2] for word in words), min(word[3] for word in words)
        right, bottom = max(word[4] for word in words), max(word[5] for word in words)
        lines.append({"text": " ".join(word[0] for word in words),
                      "score": min(word[1] for word in words) / 100,
                      "box": [left / width, top / height, (right - left) / width, (bottom - top) / height]})
    return lines


def pixel_view(image, bounds):
    """Return the crop and its exact normalized origin after rounding to pixels."""
    height, width = image.shape[:2]
    x, y, w, h = bounds
    left, top = round(x * width), round(y * height)
    right, bottom = round((x + w) * width), round((y + h) * height)
    return image[top:bottom, left:right], (left / width, top / height, (right - left) / width, (bottom - top) / height)


def project(box, bounds):
    """Convert coordinates within a crop to the oriented source frame."""
    x, y, w, h = box
    a, b, p, q = bounds
    return [round(value, 4) for value in (a + x * p, b + y * q, w * p, h * q)]


def read_tesseract(command, path, width, height, timeout, *, page_mode=11):
    """Recognize a crop with one Tesseract CPU thread."""
    raw = subprocess.check_output(
        [command, str(path), "stdout", "--tessdata-dir", str(TESSDATA),
         "-l", "rus+eng", "--psm", str(page_mode), "-c", "tessedit_create_tsv=1"],
        env={**os.environ, "OMP_THREAD_LIMIT": "1"}, stderr=subprocess.DEVNULL, timeout=timeout,
    )
    return parse_lines(raw.decode("utf-8"), width, height)


def vision_binary(*, compile_missing=True):
    """Compile the miniature Vision helper once per source revision on macOS."""
    if sys.platform != "darwin" or os.environ.get("VOICE_SCENE_OCR", "auto") == "tesseract":
        return None
    swift = shutil.which("swiftc")
    if not swift:
        return None
    source = ROOT / "scene_ocr.swift"
    ensure_temporary_root()
    cache = RUNTIME_DIR / "scene-ocr"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = sha256(source.read_bytes()).hexdigest()[:16]
    target = cache / f"vision-{key}"
    if not target.is_file():
        if not compile_missing:
            return None
        with use_cache(cache, cache / "swift-modules"), TemporaryDirectory(dir=cache) as temporary:
            artifact = Path(temporary) / "vision"
            subprocess.run([swift, "-O", "-module-cache-path", str(cache / "swift-modules"),
                            str(source), "-o", str(artifact)],
                           check=True, timeout=25, capture_output=True)
            artifact.replace(target)
    return target


def merge_lines(lines, *, line_limit=24, keep_numbers=False):
    """Keep reliable text and remove repeated reads of the same line across tiles."""
    kept = []
    for line in sorted(lines, key=lambda row: -row["score"]):
        text = line["text"].strip()
        minimum = 1 if keep_numbers and text.isdecimal() else 3
        if sum(char.isalnum() for char in text) < minimum or (not keep_numbers and text.isdecimal()) or line["score"] < .49:
            continue
        if any(text.casefold() == row["text"].casefold() and iou(line["box"], row["box"]) >= .35
               for row in kept):
            continue
        kept.append({"text": text[:160], "box": line["box"]})
    return sorted(kept[:line_limit], key=lambda row: (row["box"][1], row["box"][0]))


def vision_batches(binary, jobs, workers):
    """Run one native batch, or two independent batches when explicitly enabled."""
    paths = [str(job[1]) for job in jobs]

    def read(items):
        """Preserve alignment and allow a failed batch to fall back independently."""
        try:
            raw = subprocess.check_output([str(binary)], input=json.dumps(items), text=True,
                                          stderr=subprocess.DEVNULL, timeout=40)
            result = json.loads(raw)
            if isinstance(result, list) and len(result) == len(items):
                return result
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return [None] * len(items)

    if workers == 1 or len(paths) < 12:
        return read(paths)
    middle = len(paths) // 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(read, (paths[:middle], paths[middle:]))
    return first + second


@model_provider('vision.ocr', 'builtin.apple-ocr')
def recognize(jobs, *, line_limit=24, keep_numbers=False, page_mode=11, compile_vision=True, backend=None):
    """Prefer native OCR and retry failed crops with a bounded Tesseract pool."""
    if not jobs:
        return {}, {}
    options = {"line_limit": line_limit, "keep_numbers": keep_numbers}
    cores = stage_threads(os.cpu_count() or 1)
    setting = os.getenv("VOICE_SCENE_OCR_WORKERS")
    workers = 2 if (setting == "2" and cores >= 4) or (setting is None and cores >= 8) else 1
    vision_workers = workers if setting is not None else 1
    batches = [None] * len(jobs)
    stages = {}
    command = shutil.which("tesseract")
    fallback_ready = command and all((TESSDATA / f"{lang}.traineddata").is_file() for lang in ("rus", "eng"))
    try:
        binary = None if backend == 'tesseract' else vision_binary(compile_missing=compile_vision or not fallback_ready)
        if binary:
            batches = vision_batches(binary, jobs, vision_workers)
            stages["apple-vision"] = "partial" if None in batches else "ok"
    except (OSError, ValueError, subprocess.SubprocessError):
        stages["apple-vision"] = "partial"
    missing = [index for index, batch in enumerate(batches) if batch is None or (page_mode == 7 and not batch)]
    if backend == 'apple-ocr' and missing:
        raise RuntimeError('Explicit Apple OCR provider could not recognize all crops.')
    if not missing:
        return collect_lines(jobs, batches, **options), stages
    if not fallback_ready:
        return collect_lines(jobs, batches, **options), {**stages, "tesseract": "unavailable"}

    deadline = monotonic() + 36

    def read(job):
        """Keep individual crop failures from discarding other readings."""
        _, path, _, width, height = job
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        try:
            return read_tesseract(command, path, width, height, min(4, remaining), page_mode=page_mode)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        repairs = pool.map(read, (jobs[index] for index in missing))
        for index, result in zip(missing, repairs):
            batches[index] = result
    return collect_lines(jobs, batches, **options), {**stages, "tesseract": "partial" if None in batches else "ok"}


def collect_lines(jobs, batches, **options):
    """Merge crop detections into a per-timestamp list in source coordinates."""
    readings = {}
    for (stamp, _, bounds, _, _), lines in zip(jobs, batches):
        for line in lines or []:
            box = line["box"]
            if bounds != FULL and (box[0] < .008 or box[0] + box[2] > .992):
                continue
            readings.setdefault(stamp, []).append({**line, "box": project(box, bounds)})
    return {stamp: merge_lines(lines, **options) for stamp, lines in readings.items()}
