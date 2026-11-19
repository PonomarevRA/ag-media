"""Select and execute bounded body, gaze or clothing-mask tasks."""

from voice_studio.mod_runtime import model_provider

from voice_studio.mod_runtime import worker_environment

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
from threading import BoundedSemaphore
from zipfile import ZIP_DEFLATED, ZipFile

from voice_studio.core.worker_process import worker_command
from voice_studio.synthesis.engines import ROOT, ffmpeg
from voice_studio.vision.video_revolver import PYTHON, VISION

from voice_paths import model_path, temporary_directory
MP_PYTHON = ROOT / ".engines/mediapipe/bin/python"
SEGMENT_MODEL = model_path("mediapipe", "selfie_multiclass_256x256.tflite")
EYES_MODEL = model_path("mediapipe", "face_landmarker.task")
MAX_PREVIEW_MS = 60_000
MAX_EXPORT_MS = 6_000
MAX_EXPORT_FRAMES = 360
SLOT = BoundedSemaphore(1)
PTS = re.compile(r"\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:\s*(-?[\d.]+)")


def layer_plan(*, body=False, gaze=False, masks=False):
    """Use NanoDet only for body geometry and MediaPipe only for the requested task."""
    return {"body": bool(body or masks), "gaze": bool(gaze and not masks), "masks": bool(masks)}


def select_frames(source, work, start_ms, end_ms, formats, *, native=False):
    """Decode bounded proxies, retaining the exact decoded presentation times for masks."""
    output = str(work / "%06d.jpg")
    resize = "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2"
    options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", formats]
    if start_ms:
        options += ["-ss", str(start_ms / 1000)]
    options += ["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-filter_threads", "2",
                "-t", str((end_ms - start_ms) / 1000), "-start_number", "0", "-threads:v", "2", "-q:v", "3"]
    if not native:
        ffmpeg([*options, "-vf", f"fps=4:start_time=0,{resize}", "-frames:v", "240", output], timeout=35)
        return [(path.name, index * 250 + start_ms) for index, path in enumerate(sorted(work.glob("*.jpg")))]
    process = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", *options,
                              "-vf", f"{resize},showinfo", "-fps_mode", "passthrough",
                              "-frames:v", str(MAX_EXPORT_FRAMES + 1), output],
                             capture_output=True, text=True, timeout=45, check=True)
    stamps = {int(index): start_ms + round(float(value) * 1000) for index, value in PTS.findall(process.stderr)}
    paths = sorted(work.glob("*.jpg"))
    if len(paths) > MAX_EXPORT_FRAMES:
        raise ValueError("Слишком много кадров в интервале. Сократи его.")
    if not paths or any(int(path.stem) not in stamps for path in paths):
        raise ValueError("Не удалось связать кадры с таймкодами исходника.")
    return [(path.name, stamps[int(path.stem)]) for path in paths if stamps[int(path.stem)] < end_ms]


@model_provider('vision.person-batch', 'builtin.person-batch')
def dispatch_layers(work, frames, timeline, plan, target=None):
    """Run one isolated native worker with only the interpreters and models it needs."""
    interpreter = MP_PYTHON if plan["gaze"] or plan["masks"] else PYTHON
    if not interpreter.is_file():
        raise ValueError("Локальный движок MediaPipe недоступен.")
    request = {"frames": frames, "timeline": timeline, "plan": plan, "target": target,
               "models": {"body": str(model_path("vision", "nanodet.onnx")), "gaze": str(EYES_MODEL),
                          "masks": str(SEGMENT_MODEL)}}
    arguments = work / "request.json"
    arguments.write_text(json.dumps(request), encoding="utf-8")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if interpreter == PYTHON:
        environment["PYTHONPATH"] = str(VISION)
    result = subprocess.run(worker_command(interpreter, "voice_studio.workers.person_layer_worker", arguments),
                            env=worker_environment(environment), capture_output=True, text=True, timeout=75, check=True)
    return json.loads(result.stdout)


@contextmanager
def person_slot():
    """Allow one local person-layer job at a time."""
    if not SLOT.acquire(blocking=False):
        raise ValueError("Анализ фигуры занят другим видео.")
    try:
        yield
    finally:
        SLOT.release()


def preview_people(source, timeline, work, formats, plan):
    """Sample figure geometry and gaze up to the end of the available face timeline."""
    with person_slot(), temporary_directory() as folder:
        folder = Path(folder)
        end = min(MAX_PREVIEW_MS, len(timeline["frames"]) * timeline["stepMs"])
        frames = select_frames(source, folder, 0, end, formats)
        return dispatch_layers(folder, frames, timeline, plan)


def validate_mask_request(spec):
    """Validate the source-clock bounds, normalized face boxes, and requested identity."""
    target, start, end, timeline = (spec.get(field) for field in ("faceId", "startMs", "endMs", "timeline"))
    if (type(target) is not int or not 1 <= target <= 100_000 or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= MAX_PREVIEW_MS or end - start > MAX_EXPORT_MS):
        raise ValueError("Выбери лицо и интервал до 6 секунд в первой минуте.")
    if not isinstance(timeline, dict) or timeline.get("stepMs") != 40:
        raise ValueError("Для масок сначала нужен анализ лиц.")
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not frames or len(frames) > 1500:
        raise ValueError("Некорректный трек лиц.")
    found = False
    for rows in frames:
        if not isinstance(rows, list) or len(rows) > 32:
            raise ValueError("Некорректный кадр трека.")
        for hit in rows:
            if (not isinstance(hit, list) or len(hit) != 7 or type(hit[0]) is not int or type(hit[6]) is not int
                    or not all(type(value) in (int, float) and 0 <= value <= 1 for value in hit[1:5])):
                raise ValueError("Некорректная рамка лица.")
            found |= hit[6] == target
    if not found:
        raise ValueError("Выбранное лицо отсутствует в треке.")
    return target, start, end, timeline


def export_clothing_masks(source, spec, destination, formats):
    """Write a ZIP with binary alpha masks and a per-frame source-time manifest."""
    target, start, end, timeline = validate_mask_request(spec)
    with person_slot(), temporary_directory() as folder:
        work = Path(folder)
        try:
            frames = select_frames(source, work, start, end, formats, native=True)
            result = dispatch_layers(work, frames, timeline, layer_plan(masks=True), target)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Не удалось обработать кадры для масок одежды.") from exc
        count = sum("mask" in row for row in result["frames"])
        if count == 0:
            raise ValueError("В этом интервале не найдено надёжно связанной маски одежды. Попробуй другой план.")
        manifest = {**result, "version": "voice-studio-clothing-mask-1", "faceId": target,
                    "sourceDigest": spec["sourceDigest"], "startMs": start, "endMs": end,
                    "coordinateSpace": "normalized oriented source frame", "maskValues": {"transparent": 0, "clothing": 255}}
        number, profile_id = spec.get("speakerNumber"), spec.get("profileId")
        if type(number) is int and 1 <= number <= 1000:
            manifest["speakerNumber"] = number
        if isinstance(profile_id, str) and re.fullmatch(r"[0-9a-f]{32}", profile_id):
            manifest["profileId"] = profile_id
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for row in result["frames"]:
                if name := row.get("mask"):
                    archive.write(work / name, name)
        return manifest
