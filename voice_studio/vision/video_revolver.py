"""Collect visual proposals without rewriting the audio transcript or speaker labels."""

from voice_studio.mod_runtime import model_provider, is_replacement, current_snapshot

from voice_studio.mod_runtime import worker_environment

import json
import os
from pathlib import Path
from math import isfinite
import subprocess
import sys
from threading import BoundedSemaphore

from voice_studio.core.worker_process import worker_command
from voice_studio.core.optional_runtime import runtime_environment, runtime_ready
from voice_studio.vision.activity_worker_session import ActivityWorkerSession
from voice_studio.vision.facial_expressions import finalize_expressions
from voice_studio.vision.video_activity_chunks import STEP, ActivityTimeline, activity_windows

from voice_studio.core.analysis_stages import MAX_ACTIVITY_CUES, ignore_progress, queued_stage_slot, stage
from voice_studio.synthesis.engines import ROOT, ffmpeg
from voice_paths import dependency_path, model_path, temporary_directory

MODEL = model_path("vision", "yunet.onnx")
SFACE = model_path("vision", "sface-int8.onnx")
VISION = dependency_path("vision")
ASD = dependency_path("light-asd")
ASD_WEIGHTS = model_path("vision", "light-asd", "pretrain_AVA_CVPR.model")
ASD_LEGACY_WEIGHTS = ASD / "upstream/weight/pretrain_AVA_CVPR.model"
if not ASD_WEIGHTS.is_file() and ASD_LEGACY_WEIGHTS.is_file():
    ASD_WEIGHTS = ASD_LEGACY_WEIGHTS
LEGACY_PYTHON = ROOT / ".engines/coqui/bin/python"
PYTHON = LEGACY_PYTHON if LEGACY_PYTHON.is_file() else Path(sys.executable)
MAX_PHRASES = 24
VISUAL_SLOT = BoundedSemaphore(1)


@model_provider('vision.visual-batch', 'builtin.visual-batch')
def review_video(source, wav, phrases, audio_start_ms, formats, *, emit=ignore_progress, duration_ms=None,
                 publish=None, people_count=None, people_origin=None):
    """Run optional visual stages against local files on the video playback clock."""
    result = {"version": "video-revolver-1", "stages": {}, "phrases": {}}
    video, video_end_ms = probe_video(source, formats, timing=True)
    if video is False:
        result["stages"]["yunet"] = "no-video"
        return result
    if not (video_end_ms or duration_ms or phrases):
        result["stages"]["yunet"] = "unavailable"
        return result
    current_snapshot().selected('vision.faces', default='builtin.yunet')
    if (not MODEL.is_file() and not is_replacement('vision.faces', 'builtin.yunet')) or not PYTHON.is_file():
        result["stages"]["yunet"] = "unavailable"
        return result

    with queued_stage_slot(VISUAL_SLOT, emit, "light-asd"):
        return collect_evidence(source, wav, phrases, audio_start_ms, formats, result, emit,
                                duration_ms=duration_ms, video_end_ms=video_end_ms, publish=publish,
                                people_count=people_count, people_origin=people_origin)


def collect_evidence(source, wav, phrases, audio_start_ms, formats, result, emit, *, duration_ms=None,
                     video_end_ms=None, publish=None, people_count=None, people_origin=None):
    """Connect the activity proposal to its exact crop and retain independent YuNet photos."""
    sampled = select_phrases(phrases)
    if len(sampled) < len(phrases):
        result["sampledPhrases"] = len(sampled)
    observations, photos = {}, None
    clock = video_end_ms or max(duration_ms or 0, phrases[-1]["endMs"] if phrases else 0)
    with stage(emit, "light-asd") as progress:
        replacement = is_replacement('vision.active-speaker', 'builtin.light-asd')
        runtime_available = LEGACY_PYTHON.is_file() or runtime_ready("light-asd")
        missing = None
        if not replacement and not ASD_WEIGHTS.is_file():
            missing = "Нет весов Light-ASD. Установите модель в разделе «Моды и модели»."
        elif not replacement and not runtime_available:
            missing = "Нет библиотек Light-ASD. Установите модель в разделе «Моды и модели»."
        weights_ready = missing is None
        try:
            activity_cues = [(row["startMs"], row["endMs"]) for row in phrases] if weights_ready and wav else []
            observations, visual = score_visible_faces(
                source, wav, audio_start_ms, activity_cues, formats, photo_phrases=sampled,
                duration_ms=clock, emit=emit, publish=publish, people_count=people_count, people_origin=people_origin)
            photos = visual.pop("phrasePhotos", None)
            result.update(visual)
            coverage = visual.get("activityCoverage") or {}
            partial = any(row["status"] != "ok" for row in coverage.get("windows", ()))
            if not weights_ready or not activity_cues:
                result["stages"]["light-asd"] = "unavailable" if not weights_ready else "no-phrases"
                reason = missing or "Нет реплик с аудио для оценки говорящего в кадре."
                progress.update(state="skipped", reason=reason)
                result["activityReason"] = reason
            else:
                result["stages"]["light-asd"] = "partial" if partial else "ok"
                if partial:
                    progress["state"] = "partial"
                confirm_profile_face_links(source, formats, phrases, observations, result)
        except (OSError, ValueError, subprocess.SubprocessError):
            result["stages"]["light-asd"] = "failed"
            reason = "Light-ASD не обработал кадры. Проверьте установку модели и повторите анализ."
            progress.update(state="failed", reason=reason)
            result["activityReason"] = reason
    with stage(emit, "yunet") as progress:
        try:
            # The original-source photos are collected by the same worker when ASD succeeds.
            if photos is not None or not sampled:
                result["phrases"] = photos or {}
                progress["reused"] = True
            else:
                result["phrases"] = run_worker("voice_studio.speech.speaker_photos", [source, MODEL, formats, json.dumps(sampled)],
                                               [VISION], timeout=100)
            result["stages"]["yunet"] = "ok"
        except (OSError, ValueError, subprocess.SubprocessError):
            result["stages"]["yunet"] = "failed"
            progress["state"] = "failed"
    for start, observation in observations.items():
        entry = result["phrases"].setdefault(start, {})
        photo = observation.get("sample", {}).pop("jpeg", None)
        if photo:
            entry["activePhoto"] = photo
        reference = observation.get("referenceCandidate")
        if reference:
            photo = reference["sample"].pop("jpeg", None)
            if photo:
                entry["referencePhoto"] = photo
        entry["activity"] = observation
    expression_layer = finalize_expressions(result, phrases, clock)
    result["stages"]["facial-expressions"] = expression_layer["status"]
    return result


def confirm_profile_face_links(source, formats, phrases, observations, visual):
    """Recheck isolated face changes in Assist mode before building voice profiles."""
    from voice_studio.vision.frame_changes import assist_settings
    from voice_studio.vision.speaker_face_recheck import boundary_pairs

    if not assist_settings():
        return
    pairs = boundary_pairs(phrases, observations, visual)
    if not pairs:
        return
    try:
        with temporary_directory() as folder:
            request = Path(folder) / "speaker-face-pairs.json"
            request.write_text(json.dumps({"source": str(source), "formats": formats,
                                           "yunet": str(MODEL), "sface": str(SFACE), "pairs": pairs}), encoding="utf-8")
            review = run_worker("voice_studio.vision.speaker_face_recheck", [request], [VISION], timeout=25)
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    visual["faceTracking"]["profileFaceReview"] = {"compared": len(pairs), "confirmed": len(review["matches"])}
    for match in review["matches"]:
        activity = observations[str(match["cueMs"])]
        activity.update(profileFace=match["face"], faceEvidence="original-head-crops", identitySimilarity=match["similarity"])


def classify_span(span, identity_cuts=()):
    """Require strong coverage from one track and abstain on competing or partial activity."""
    total = (span["end_ms"] + 39) // 40 - (span["start_ms"] + 39) // 40
    candidates = span["candidates"]
    positive = [track for track in candidates if track["positive_frames"] >= 5
                and track["positive_frames"] * 4 >= track["frames"] * 3]
    if len(positive) == 1:
        track = positive[0]
        if track["representative"] and total >= 10 and track["positive_frames"] * 5 >= total * 4:
            return {"state": "visible-active", **activity_metadata(track, total)}
    if positive and any(span["start_ms"] < cut < span["end_ms"] for cut in identity_cuts):
        ranked = sorted(positive, key=lambda row: -row["positive_frames"])
        track, *others = ranked
        if (track["representative"] and total >= 12 and track["positive_frames"] * 4 >= total * 3
                and confirmed_cut_candidates(track, others, total)):
            return {"state": "visible-active", "basis": "verified-track-cut", **activity_metadata(track, total)}
    observation = {"state": "partial-or-ambiguous" if positive else "unconfirmed", "phraseFrames": total}
    # Retain partial evidence for an independently confirmed editorial cue only.
    partial = [track for track in candidates if track["positive_frames"] >= 5]
    if len(partial) == 1:
        track = partial[0]
        if track["representative"] and total >= 12 and track["positive_frames"] * 10 >= total * 7:
            observation["referenceCandidate"] = activity_metadata(track, total)
    return observation


def confirmed_cut_candidates(track, others, total):
    """Allow a short non-concurrent face after a verified cut, never simultaneous speakers."""
    return all(other["positive_frames"] * 5 <= total and
               (track["track_end_ms"] <= other["track_start_ms"]
                or other["track_end_ms"] <= track["track_start_ms"]) for other in others)


def activity_metadata(track, total):
    """Record the exact activity track, its grouped face and a representative frame."""
    result = {"activeFrames": track["positive_frames"], "phraseFrames": total, "track": track["track"],
              "face": track.get("face", track["track"]), "sample": track["representative"]}
    for key in ("faceEvidence", "changeReview"):
        if key in track:
            result[key] = track[key]
    return result


def select_phrases(phrases, limit=MAX_PHRASES):
    """Sample evenly throughout a long transcript and pass only waveform offsets."""
    if len(phrases) > limit:
        step = (len(phrases) - 1) / (limit - 1)
        phrases = [phrases[round(index * step)] for index in range(limit)]
    return [[phrase["startMs"], phrase["endMs"]] for phrase in phrases]


def probe_video(source, formats, *, timing=False):
    """Check the moving video stream and optionally its endpoint on the playback clock."""
    try:
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
             "-format_whitelist", formats, "-select_streams", "V:0", "-show_entries",
             "stream=index,start_time,duration:format=start_time", "-of", "json", str(source)],
            stderr=subprocess.DEVNULL, timeout=10,
        )
        metadata = json.loads(raw)
        streams = metadata["streams"]
        if not timing:
            return bool(streams)
        if not streams:
            return False, None
        stream = streams[0]
        duration = float(stream.get("duration", "nan"))
        start_text = stream.get("start_time")
        origin_text = metadata.get("format", {}).get("start_time")
        offset = float(start_text) - float(origin_text) if start_text and origin_text else 0
        end = duration + max(0, offset)
        return True, round(end * 1000) if isfinite(end) and end > 0 else None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return (None, None) if timing else None


def run_worker(module, arguments, imports, timeout):
    """Keep native dependencies in their own interpreter and parse a single JSON response."""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, imports))}
    raw = subprocess.check_output(
        worker_command(PYTHON, module, *arguments),
        env=runtime_environment("vision", worker_environment(environment)), stderr=subprocess.DEVNULL, timeout=timeout,
    )
    return json.loads(raw)


def window_cues(phrases, start, end, scan_start):
    """Score fragments only once and retain the original cue denominator for merging."""
    fragments = [(max(a, start) - scan_start, min(b, end) - scan_start, (a, b))
                 for a, b in phrases if a < end and b > start]
    if len(fragments) > MAX_ACTIVITY_CUES:
        stride = (len(fragments) - 1) / (MAX_ACTIVITY_CUES - 1)
        fragments = [fragments[round(index * stride)] for index in range(MAX_ACTIVITY_CUES)]
    spans = [{"start_ms": a, "end_ms": b} for a, b, _ in fragments]
    return spans, {a: original for a, _, original in fragments}


def raw_activity_window(source, wav, audio_start_ms, formats, scan_start, scan_end, spans, photo_phrases, worker,
                        *, people_count=None, people_origin=None):
    """Prepare one clock-aligned proxy and cap temporary files to the local window."""
    with temporary_directory() as folder:
        proxy = Path(folder) / "video.mp4"
        aligned = Path(folder) / "audio.wav"
        seconds = str((scan_end - scan_start) / 1000)
        seek = ["-ss", str(scan_start / 1000)] if scan_start else []
        ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", formats, *seek,
                "-i", str(source), "-map", "0:V:0", "-an",
                "-vf", "fps=25:start_time=0,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-t", seconds, "-c:v", "libx264", "-preset", "ultrafast", str(proxy)], timeout=30)
        if spans:
            if scan_start > audio_start_ms:
                audio_seek = ["-ss", str((scan_start - audio_start_ms) / 1000)]
            else:
                audio_seek = []
            delay = max(0, audio_start_ms - scan_start)
            ffmpeg([*audio_seek, "-i", str(wav), "-af", f"adelay={delay}:all=1,apad",
                    "-t", seconds, "-c:a", "pcm_s16le", str(aligned)], timeout=10)
        photo_args = ["--photo-source", source, "--photo-formats", formats,
                      "--photo-phrases", json.dumps(photo_phrases)] if photo_phrases else []
        sampling = os.environ.get("VOICE_YUNET_SAMPLING", "dense")
        pilot_args = []
        enabled = os.environ.get("VOICE_PERSON_FACE_PILOT", "1") == "1" if people_count is None else people_count
        if enabled:
            original, origin_ms = people_origin or (source, 0)
            pilot_args = ["--person-pilot-source", original, "--person-pilot-formats", formats,
                          "--person-pilot-offset", origin_ms + scan_start,
                          "--person-pilot-model", model_path("vision", "nanodet.onnx")]
            if sampling == "adaptive":
                pilot_args.append("--person-pilot-count-only")
        expression_args = ["--expressions-model", model_path("vision", "facial_expression_recognition_mobilefacenet_2022july.onnx")]
        from voice_studio.vision.frame_changes import assist_settings
        settings = assist_settings()
        changes = []
        if spans and settings:
            changes = ["--activity-change-threshold", settings["pixelThreshold"]]
        arguments = ["--video", proxy, "--audio", aligned, "--yunet", MODEL, "--sface", SFACE,
                     "--yunet-sampling", sampling,
                     "--weights", ASD_WEIGHTS, "--spans-ms", json.dumps(spans), *photo_args, *pilot_args,
                     *expression_args, *changes]
        return worker.request(arguments, 100 if photo_args or pilot_args else 60)


def sampled_window_end(start, end, measured_end, *, final):
    """Treat a final sub-frame remainder as EOF quantization on the 25 fps proxy."""
    if final and start < measured_end < end and end - measured_end < STEP:
        return end
    return measured_end


def score_visible_faces(source, wav, audio_start_ms, phrases, formats, *, photo_phrases=None,
                        duration_ms=None, emit=ignore_progress, publish=None, people_count=None, people_origin=None):
    """Accumulate bounded windows, preserving available time ranges across failures."""
    duration = duration_ms or phrases[-1][1]
    plan = list(activity_windows(duration))
    accumulated = ActivityTimeline(duration)
    coverage = []
    photos = None
    published = False
    imports = [VISION]
    command = worker_command(PYTHON, "voice_studio.workers.light_asd_worker")
    with ActivityWorkerSession(command, imports) as worker:
        for index, (start, end, scan_start, scan_end) in enumerate(plan, 1):
            spans, cue_starts = window_cues(phrases, start, end, scan_start)
            try:
                report = raw_activity_window(source, wav, audio_start_ms, formats, scan_start, scan_end,
                                             spans, photo_phrases if len(plan) == 1 else None, worker,
                                             people_count=people_count, people_origin=people_origin)
                photos = report.get("phrasePhotos", photos)
                measured = accumulated.add(report, start, end, scan_start, cue_starts)
                scanned = sampled_window_end(start, end, measured, final=index == len(plan))
                status = "ok" if scanned >= end else "partial" if scanned > start else "no-video"
            except (OSError, ValueError, subprocess.SubprocessError):
                scanned, status = start, "failed"
            coverage.append({"startMs": start, "endMs": end, "coveredEndMs": scanned, "status": status})
            if len(plan) > 1:
                emit({"stage": "light-asd", "state": "running", "chunk": index, "totalChunks": len(plan)})
            if publish is not None and scanned > start and index < len(plan) and (not published or index % 3 == 0):
                view = accumulated.data(duration, available_end=end)
                view["activityCoverage"] = {"durationMs": duration, "windows": list(coverage)}
                publish(view)
                published = True
    if not any(row["coveredEndMs"] > row["startMs"] for row in coverage):
        raise ValueError("No decodable activity windows")
    visual = accumulated.data(duration)
    visual["activityCoverage"] = {"durationMs": duration, "windows": coverage}
    if photos is not None:
        visual["phrasePhotos"] = photos
    identity_cuts = visual["faceTracking"]["identityCutTimesMs"]
    observations = {}
    for span in accumulated.spans.values():
        span["candidates"] = list(span["candidates"].values())
        observations[str(span["start_ms"])] = classify_span(span, identity_cuts)
    return observations, visual
