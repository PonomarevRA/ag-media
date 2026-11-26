"""Extract local media audio and return segment timestamps and SRT subtitles."""

from voice_studio.mod_runtime import analysis_run, submit_with_context

from array import array
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from math import isfinite
import os
from pathlib import Path
from struct import unpack
import subprocess
import sys
import wave
from voice_paths import temporary_directory
from urllib.parse import urlsplit

from voice_studio.core.analysis_stages import ignore_progress, serialized_progress, stage
from voice_studio.speech.asr_context_graph import VERSION, corrections
from voice_studio.speech.asr_recovery import recover_clip
from voice_studio.audio.audio_scene import analyze_audio_scene, select_sound_classifier
from voice_studio.audio.dialogue_pauses import measure_dialogue_pauses
from voice_studio.synthesis.engines import ffmpeg
from voice_studio.vision.person_layers import layer_plan, preview_people
from voice_studio.vision.person_motion import measure_people_motion
from voice_studio.speech.person_mentions import annotate_person_mentions
from voice_studio.vision.face_speakers import bind_faces
from voice_studio.core.media_resources import batch_media_job, pilot_resources, run_batch_stage
from voice_studio.speech.transcription_languages import CLIP_LANGUAGES, STUDIO_LANGUAGES
from voice_studio.speech.recognition import SMALL_MODEL, ready, recognize_file, recognize_result, resolve_whisper_language
from voice_studio.vision.scene_transcript import reconcile_names
from voice_studio.speech.speaker_diarization import detect_turns, label_phrases, label_words
from voice_studio.speech.speaker_review import review_speakers
from voice_studio.speech.speech_overlap import audio_overlaps, clip_overlaps, mixed_label, visual_overlaps
from voice_studio.vision.video_revolver import review_video
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING
from voice_studio.vision.video_scene import review_scene
from voice_studio.vision.shot_moodbook import build_moodbook
from voice_studio.graph.media_graph_model import object_layers

MAX_MEDIA_BYTES = 200 * 1024 * 1024
MAX_MEDIA_SECONDS = 15 * 60
FORMATS = "wav,matroska,webm,mov,mp3,ogg,flac,aac,aiff,avi,mpegts"
CAPTURE_WAV_FORMAT = b"\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00"


def srt_time(milliseconds):
    """Format a non-negative millisecond offset for SubRip."""
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}"


def segments_from_whisper(raw, duration_ms, audio_start_ms=0):
    """Keep nonempty Whisper phrases with ordered offsets inside the uploaded media."""
    segments = []
    previous_end = 0
    for item in raw:
        text = item["text"].strip()
        offsets = item.get("offsets", {})
        start, end = offsets.get("from"), offsets.get("to")
        if not text or type(start) is not int or type(end) is not int:
            continue
        start = max(previous_end, start + audio_start_ms)
        end = min(duration_ms, end + audio_start_ms)
        if start < end:
            segments.append({"startMs": start, "endMs": end, "text": text})
            previous_end = end
    return segments


def audio_offset(source):
    """Read the first audio stream's offset from the container playback origin."""
    try:
        result = subprocess.check_output(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
             "-format_whitelist", FORMATS, "-select_streams", "a:0",
             "-show_entries", "format=start_time:stream=start_time", "-of", "json", str(source)],
            stderr=subprocess.DEVNULL, timeout=30,
        )
        timings = json.loads(result)
        stream = timings["streams"][0]
        start = stream.get("start_time")
        origin = timings.get("format", {}).get("start_time")
        if not start or not origin:
            return 0
        offset = float(start) - float(origin)
        return max(0, round(offset * 1000)) if isfinite(offset) else 0
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось открыть аудиодорожку файла.") from exc


def is_capture_wav(source):
    """Identify the exact 44-byte header and complete PCM payload produced by the browser."""
    with source.open("rb") as stream:
        header = stream.read(44)
    if len(header) != 44:
        return False
    riff, riff_size, wave_fmt, fmt_size, audio_format, data, data_size = unpack("<4sI8sI16s4sI", header)
    return (riff == b"RIFF" and wave_fmt == b"WAVEfmt " and fmt_size == 16
            and audio_format == CAPTURE_WAV_FORMAT and data == b"data"
            and data_size % 2 == 0 and riff_size == data_size + 36
            and source.stat().st_size == data_size + 44)


@analysis_run
def transcribe_media(root, language, source, allow_empty=False, speakers=False, speaker_count=None,
                     *, quality=False, visual=False, scene=False, merge_similar=False, body=False, gaze=False,
                     object_sampling=DEFAULT_OBJECT_SAMPLING, audio_classifier="auto", clip_languages=False, people_count=None,
                     emit=ignore_progress, publish=None, publish_mosaic=None, revolver=None,
                     selected_stages=None, force_stages=None, object_reconciliation=False):
    """Decode media on its source clock, reusing successful independent layer checkpoints.

    selected_stages=None preserves the enabled layer flags. An explicit list adds
    those stages and missing prerequisites, retaining other valid cached layers;
    [] only restores completed work. force_stages bypasses the named checkpoints.
    The optional object review consumes existing scene observations only.
    """
    from voice_studio.graph.media_stage_selection import StageSelection
    selection = StageSelection(selected_stages, force_stages)
    enabled = selection.enabled if selection.enabled is not None else selection.forced
    visual = visual or "visual" in enabled
    scene = scene or "scene" in enabled
    body = body or ("people" in enabled and not gaze)
    speakers = speakers or "voice" in enabled
    if not allow_empty:
        audio_classifier = select_sound_classifier(audio_classifier)
    if allow_empty and is_capture_wav(source):
        with temporary_directory() as folder:
            return transcribe_wav(source, Path(folder), language, allow_empty=True,
                                  speakers=speakers, speaker_count=speaker_count)
    from voice_studio.graph.media_stage_cache import incremental_stages, usable_visual, usable_speech, usable_scene, scene_inputs

    with temporary_directory() as folder, incremental_stages(root, source, revolver, selection) as cache:
        work = Path(folder)
        wav = work / "audio.wav"
        audio_start_ms = audio_offset(source)

        def prepared_audio():
            """Decode PCM once, only when a missing stage actually consumes it."""
            if wav.is_file():
                return wav
            with stage(emit, "audio"):
                ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                    "-i", str(source), "-map", "0:a:0", "-vn",
                    "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
                    "-t", str(11 if allow_empty else MAX_MEDIA_SECONDS + 1), "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], timeout=150)
            return wav

        empty = {"segments": [], "text": "", "srt": "", "language": language or "auto", "durationMs": 0}
        if selection.selective:
            from voice_studio.graph.media_graph_ingest import probe_source
            empty["durationMs"] = probe_source(source)["durationMs"]
            # Restore previously computed optional layers even when their flags are now off.
            visual = visual or "visual" in cache.manifest["layers"]
            body = body or "people" in cache.manifest["layers"]
            scene = scene or "scene" in cache.manifest["layers"]
            speakers = speakers or "voice" in cache.manifest["layers"]
        result = cache.run("speech", {"language": language, "allowEmpty": allow_empty, "speakers": speakers, "speakerCount": speaker_count,
                                      "quality": quality, "mergeSimilar": merge_similar, "clipLanguages": clip_languages},
                           None, lambda: transcribe_wav(prepared_audio(), work, language, audio_start_ms, allow_empty,
                                speakers=speakers, speaker_count=speaker_count, quality=quality,
                                merge_similar=merge_similar, emit=emit, audio_layers=not allow_empty,
                                audio_classifier=audio_classifier, clip_languages=clip_languages, publish=publish, defer_audio=True),
                           usable_speech,
                           publish=(lambda report: publish("base", report)) if publish else None, emit=emit, fallback=empty)
        if publish is not None and scene and selection.selective:
            speech_status = selection.states.get("speech", {})
            if speech_status.get("state") in ("pending", "invalidated"):
                # Scene-only analysis still needs a source clock for subsequent partial layers.
                publish("base", {**result, "analysisStages": {"speech": speech_status}})
        speech_input = deepcopy(result)
        if not allow_empty:
            detail = cache.run("audio", {"classifier": audio_classifier}, speech_input,
                               lambda: run_batch_stage(analyze_audio_scene, prepared_audio(), result["segments"], audio_start_ms,
                                                           result.get("speechOverlaps", []), emit, audio_classifier),
                               lambda report: report.get("soundAnalysisStatus") in ("done", "too-short"), emit=emit)
            result.update(detail)
            if publish is not None:
                publish("audio", detail)
        if visual or body or gaze:
            from voice_studio.speech.speaker_pipeline import measure_speaker_review, usable_speaker_review
            from voice_studio.speech.speaker_review import REVIEW_VERSION as SPEAKER_REVIEW_VERSION

            def measure_visual():
                """Checkpoint measured faces and activity before the independent identity check."""
                visual_result = deepcopy(speech_input)
                run_batch_stage(analyze_visual_track, source, prepared_audio(), audio_start_ms, visual_result,
                                emit=emit, publish=publish, people_count=people_count)
                return visual_result

            visual_result = cache.run("visual", {"peopleCount": people_count, "speakers": speakers,
                                                 "speakerCount": speaker_count, "mergeSimilar": merge_similar,
                                                 "separateSpeakerReview": True},
                                      speech_input, measure_visual, usable_visual, emit=emit)
            if visual_result:
                result.update(visual_result)
                if speakers and speaker_count is None:
                    visual_hash = cache.revolver.counts["layers"]["visual"]["resultHash"]
                    review = cache.run("speaker-review", {"version": SPEAKER_REVIEW_VERSION}, visual_hash,
                                       lambda: measure_speaker_review(visual_result, prepared_audio, audio_start_ms,
                                                                      review_speakers, alternate=not merge_similar, emit=emit),
                                       usable_speaker_review, emit=emit, selection_layer="visual")
                    if review:
                        result["speakerReview"] = review["speakerReview"]
                    if review and "segments" in review:
                        result["segments"] = review["segments"]
                        result["videoEvidence"] = {**result["videoEvidence"], "phrases": review["phrases"]}
                bind_faces(result)
                result["srt"] = render_srt(result["segments"], result["language"])
                if publish is not None:
                    fields = ("videoEvidence", "segments", "speakerReview", "srt")
                    publish("visual", {key: result[key] for key in fields if key in result})
        scene_data = run_batch_stage(analyze_detail_layers, source, work, result.get("videoEvidence"), body=body, gaze=gaze,
                                           scene=scene, object_sampling=object_sampling, emit=emit, publish=publish,
                                           publish_mosaic=publish_mosaic, cache=cache)
        if scene_data is not None:
            if scene_data.get("status") == "ok":
                evidence = result.get("videoEvidence") or {}
                duration = max(result["durationMs"], (evidence.get("editorTimeline") or {}).get("durationMs", 0))
                try:
                    moodbook = build_moodbook(source, FORMATS, scene_data, evidence, duration)
                    if moodbook:
                        scene_data["moodbook"] = moodbook
                except (OSError, ValueError, subprocess.SubprocessError):
                    scene_data["moodbookStatus"] = "unavailable"
            detail = {"sceneEvidence": scene_data, "shotColors": scene_data.pop("shotColors", [])}
            result.update(detail)
            if attach_person_motion(result):
                detail["videoEvidence"] = result["videoEvidence"]
            detail.update(object_layers(result))
            if reconcile_names(result["segments"], scene_data, result["language"]):
                detail.update(transcript_result(result["segments"], result["durationMs"], result["language"]))
            result.update(detail)
            if publish is not None:
                publish("scene", detail)
        elif visual or body or gaze:
            palette = cache.run("scene", {"colorsOnly": True}, scene_inputs(result.get("videoEvidence")),
                                lambda: run_batch_stage(review_scene, source, work, FORMATS, result.get("videoEvidence"), colors_only=True),
                                usable_scene, emit=emit)
            result["shotColors"] = palette.get("shotColors", [])
            if attach_person_motion(result, palette.get("shotStartsMs", ())) and publish is not None:
                publish("visual", {"videoEvidence": result["videoEvidence"]})
        if speakers and not allow_empty and result.get('speechTurns'):
            from voice_studio.speech.speaker_pipeline import enrich_source_speech
            enrich_source_speech(result, source, prepared_audio, audio_start_ms, cache, emit=emit)
            if publish is not None:
                publish('voice', result)
        if selection.selective and "voice" not in selection.states:
            selection.mark("voice", "invalidated" if "voice" in cache.manifest["layers"] else "pending")
            if "voice" in enabled and not result.get("speechTurns"):
                selection.states["voice"]["reason"] = "requires-speech-turns"
        if object_reconciliation or "object_review" in enabled or (selection.selective and "object_review" in cache.manifest["layers"]):
            from voice_studio.graph.media_object_review import review_objects
            review_objects(source, result, cache, emit=emit)
        return selection.attach(result)


def analyze_visual_track(source, wav, audio_start_ms, result, *, emit=ignore_progress, publish=None,
                         people_count=None, people_origin=None):
    """Populate shared visual evidence from the existing word and speaker timecodes."""
    def publish_frames(view):
        """Expose faces as they arrive without changing editable cue identities."""
        publish("visual", {"videoEvidence": {**view, "stages": {"light-asd": "running"}, "phrases": {}}})

    evidence = review_video(source, wav, result["segments"], audio_start_ms, FORMATS, emit=emit,
                            duration_ms=result.get("durationMs"), publish=publish_frames if publish else None,
                            people_count=people_count, people_origin=people_origin)
    annotate_visible_speakers(result["segments"], evidence)
    if evidence.get("timeline"):
        evidence["simultaneousFaces"] = visual_overlaps(evidence["timeline"])
    result["videoEvidence"] = evidence


def attach_person_motion(report, shot_starts=()):
    """Add attributed movement on the source clock after camera evidence is available."""
    evidence = report.get("videoEvidence") or {}
    editor = evidence.get("editorTimeline") or {}
    if not (evidence.get("timeline") or {}).get("frames") or not editor:
        return False
    scene = report.get("sceneEvidence") or {}
    duration = max((value for value in (report.get("durationMs"), editor.get("durationMs"))
                    if type(value) is int), default=0)
    editor["personMotion"] = measure_people_motion(
        evidence, scene.get("cameraMotion") or [], duration,
        shot_starts=tuple(scene.get("shotStartsMs") or ()) + tuple(shot_starts),
    )
    return True


def attach_person_evidence(source, work, evidence, *, body, gaze, emit=ignore_progress):
    """Use the standard worker to enrich the current face tracks with figure and gaze samples."""
    with stage(emit, "person-layer") as progress:
        timeline = evidence.get("timeline") or {}
        if not timeline.get("frames"):
            progress["state"] = "skipped"
            return
        plan = layer_plan(body=body, gaze=gaze)
        try:
            layer = preview_people(source, timeline, work, FORMATS, plan, emit=emit)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception:
            logging.getLogger(__name__).exception("Optional figure and gaze analysis failed")
            reason = "Не удалось измерить фигуру и взгляд. Остальные этапы продолжатся."
            evidence["personLayer"] = {"status": "failed", "plan": plan, "reason": reason}
            progress.update(state="failed", reason=reason)
        else:
            evidence["personLayer"] = layer
            if layer.get("gazeUnavailable"):
                reason = layer.get("reason") or "MediaPipe не смог измерить взгляд. Остальные этапы продолжатся."
                progress.update(state="partial" if layer.get("frames") else "skipped", reason=reason)


def collect_scene(source, work, evidence, object_sampling, emit, publish_mosaic, publish_preview=ignore_progress):
    """Measure one scene pass and forward its optional change geometry preview."""
    with stage(emit, "scene") as progress:
        result = review_scene(source, work, FORMATS, evidence, object_sampling=object_sampling,
                              publish_mosaic=publish_mosaic, emit=emit, publish_preview=publish_preview)
        if result["status"] != "ok":
            progress["state"] = "failed" if result["status"] == "failed" else "skipped"
            if reason := result.get("reason"):
                progress["reason"] = reason
        return result


def analyze_detail_layers(source, work, evidence, *, body, gaze, scene, object_sampling, emit, publish=None,
                          publish_mosaic=None, cache=None):
    """Run figure/gaze and progressive scene analysis from the computed face timeline."""
    people = body or gaze
    evidence = evidence or {}
    has_tracks = bool(((evidence or {}).get("timeline") or {}).get("frames"))
    parallel = scene and people and has_tracks and (os.cpu_count() or 1) >= 8
    resources = pilot_resources()
    if resources is not None:
        parallel = parallel and resources.threads >= 8 and resources.memory_mib >= 8192
    default = "1" if os.getenv("VOICE_SCENE_OCR") == "tesseract" else "0"
    def measure_people(notify):
        """Publish new figure and gaze evidence as soon as it is measured."""
        def operation():
            """Keep the existing worker and its unavailable/failed reporting contract."""
            attach_person_evidence(source, work, evidence, body=body, gaze=gaze, emit=notify)
            return evidence.get("personLayer") or {}

        if cache is None:
            operation()
        elif not has_tracks:
            cache.selection.mark("people", "invalidated" if "people" in cache.manifest["layers"] else "pending")
        else:
            evidence["personLayer"] = cache.run("people", layer_plan(body=body, gaze=gaze), evidence["timeline"],
                                                operation, lambda report: isinstance(report.get("frames"), list)
                                                and not report.get("gazeUnavailable"), emit=notify)
        if publish is not None and evidence.get("personLayer"):
            publish("visual", {"videoEvidence": evidence})

    def measure_scene(view, notify):
        """Cache the raw scene before moodbook/name reconciliation mutates the report."""
        def preview(detail):
            """Send geometry while the scene worker still has recognition ahead."""
            publish("scene-preview", {"sceneEvidence": detail})

        def operation(mosaic):
            callback = preview if publish is not None else ignore_progress
            return collect_scene(source, work, view, object_sampling, notify, mosaic, callback)

        if cache is None:
            return operation(publish_mosaic)
        return cache.scene(view, object_sampling, work, operation, publish_mosaic, notify)

    if parallel and os.getenv("VOICE_VIDEO_PARALLEL", default) == "1":
        notify = serialized_progress(emit)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-analysis") as worker:
            pending = submit_with_context(worker, measure_scene, dict(evidence), notify)
            measure_people(notify)
            return pending.result()
    if people:
        measure_people(emit)
    if scene:
        return measure_scene(evidence, emit)
    return None


def compatibility_preview(source, output):
    """Make a temporary browser-playable WebM covering the supported video duration."""
    try:
        ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "fps=25:start_time=0,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-t", str(MAX_MEDIA_SECONDS), "-c:v", "libvpx", "-deadline", "realtime",
                "-cpu-used", "6", "-b:v", "700k", "-pix_fmt", "yuv420p",
                "-c:a", "libopus", "-b:a", "64k", str(output)], timeout=600)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Не удалось создать совместимый предпросмотр.") from exc


@batch_media_job
@analysis_run
def transcribe_stream(root, language, url, seconds, speakers=False, speaker_count=None, *, audio_classifier="auto"):
    """Capture a bounded HLS or RTSP excerpt and transcribe its audio locally."""
    audio_classifier = select_sound_classifier(audio_classifier)
    if not isinstance(url, str) or len(url) > 4096 or any(char.isspace() for char in url):
        raise ValueError("Укажи ссылку на HLS или RTSP.")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https", "rtsp"} or not parsed.hostname:
        raise ValueError("Укажи ссылку на HLS или RTSP.")
    if type(seconds) is not int or not 1 <= seconds <= 120:
        raise ValueError("Выбери от 1 до 120 секунд захвата.")
    if parsed.scheme == "rtsp":
        options = ["-rtsp_transport", "tcp", "-timeout", "12000000",
                   "-protocol_whitelist", "tcp,rtsp", "-f", "rtsp"]
    else:
        options = ["-rw_timeout", "12000000", "-live_start_index", "-1",
                   "-protocol_whitelist", "http,https,tcp,tls,crypto", "-f", "hls"]
    with temporary_directory() as folder:
        work = Path(folder)
        wav = work / "audio.wav"
        try:
            ffmpeg([*options, "-i", url, "-map", "0:a:0", "-vn", "-t", str(seconds),
                    "-af", f"aresample=16000,atrim=end_sample={seconds * 16000},asetpts=N/SR/TB",
                    "-ac", "1", "-c:a", "pcm_s16le", str(wav)], timeout=seconds + 35)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            raise ValueError("Не удалось получить звук из потока. Проверь ссылку и доступность источника.") from exc
        return transcribe_wav(wav, work, language, speakers=speakers, speaker_count=speaker_count,
                              audio_layers=True, audio_classifier=audio_classifier)


def diarize_audio(wav, speaker_count, merge_similar, emit):
    """Read voice turns and report a recoverable speaker warning independently of ASR."""
    warning = None
    turns = []
    with stage(emit, "diarization") as progress:
        try:
            if merge_similar and speaker_count is None:
                turns = detect_turns(wav, None, merge_similar=True)
            else:
                turns = detect_turns(wav, speaker_count)
        except ValueError as exc:
            warning = str(exc)
            progress["state"] = "failed"
        if not turns and not warning:
            warning = "Модель не нашла уверенных голосовых интервалов."
            progress["state"] = "skipped"
    return turns, warning


def decode_speech(wav, output, language, small, words, emit, threads=None):
    """Read the original Whisper segmentation and native language result."""
    model_options = {"model": SMALL_MODEL} if small else {}
    if threads is not None:
        model_options["threads"] = threads
    with stage(emit, "whisper") as progress:
        progress["detail"] = "Whisper Small" if small else "Whisper Base"
        if language == "auto":
            response = recognize_result(wav, output, language, words=words, timeout=600, **model_options)
            return response["transcription"], (response.get("result") or {}).get("language")
        raw = recognize_file(wav, output, language, words=words, timeout=600, **model_options)
        return raw, language


def recognize_and_diarize(wav, output, language, small, speakers, speaker_count, merge_similar, emit):
    """Overlap independent phrase ASR and diarization where the host has enough CPU."""
    parallel = small and speakers and (os.cpu_count() or 1) >= 6 and os.getenv("VOICE_ASR_PARALLEL", "1") != "0"
    resources = pilot_resources()
    threads = None
    if resources is not None:
        threads, parallel = resources.speech(parallel)
    if parallel:
        notify = serialized_progress(emit)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker-diarization") as worker:
            pending = submit_with_context(worker, diarize_audio, wav, speaker_count, merge_similar, notify)
            raw, detected = decode_speech(wav, output, language, small, False, notify, threads)
            turns, warning = pending.result()
        return raw, detected, turns, warning

    turns, warning = diarize_audio(wav, speaker_count, merge_similar, emit) if speakers else ([], None)
    raw, detected = decode_speech(wav, output, language, small, bool(turns) and not small, emit, threads)
    return raw, detected, turns, warning


def prepare_media_cues(segments, turns, as_phrases, audio_start_ms, overlaps):
    """Attribute speech locally, then shift independent copies onto the video clock."""
    if turns:
        assign = label_phrases if as_phrases else label_words
        segments = assign(segments, turns)
    output = []
    for original in segments:
        item = {**original, "startMs": original["startMs"] + audio_start_ms,
                "endMs": original["endMs"] + audio_start_ms}
        if window := original.get("speakerAudioWindow"):
            item["speakerAudioWindow"] = {
                "startMs": window["startMs"] + audio_start_ms,
                "endMs": window["endMs"] + audio_start_ms,
            }
        output.append(item)
    clip_overlaps(output, overlaps)
    return output


def transcribe_wav(wav, work, language, audio_start_ms=0, allow_empty=False, *,
                   speakers=False, speaker_count=None, quality=False, merge_similar=False, emit=ignore_progress,
                   audio_layers=False, audio_classifier="auto", clip_languages=False, publish=None, defer_audio=False):
    """Publish the speech report before classifying optional audio layers."""
    with wave.open(str(wav)) as audio:
        duration_ms = audio.getnframes() * 1000 // audio.getframerate()
        if allow_empty:
            if duration_ms > 10000:
                raise ValueError("Отрезок захвата должен быть не длиннее 10 секунд.")
            samples = array("h", audio.readframes(audio.getnframes()))
            if sys.byteorder != "little":
                samples.byteswap()
            if not any(abs(sample) >= 32 for sample in samples):
                return transcript_result([], duration_ms, language or "auto")
    if not 350 <= duration_ms <= MAX_MEDIA_SECONDS * 1000:
        if allow_empty and duration_ms < 350:
            return transcript_result([], duration_ms, language or "auto")
        raise ValueError("Длительность звука должна быть от 0,35 секунды до 15 минут.")
    if speaker_count is not None and (type(speaker_count) is not int or not 1 <= speaker_count <= 12):
        raise ValueError("Число говорящих должно быть от 1 до 12.")
    emit({"durationMs": duration_ms})
    small = quality and ready(SMALL_MODEL)
    language = resolve_whisper_language(language, SMALL_MODEL if small else None, clip_languages=clip_languages)
    automatic = language == "auto"
    supported = CLIP_LANGUAGES if clip_languages else STUDIO_LANGUAGES
    raw, detected, turns, warning = run_batch_stage(
        recognize_and_diarize, wav, work / "transcript", language, small, speakers, speaker_count,
        merge_similar, emit, allow_empty=allow_empty)
    segments = segments_from_whisper(raw, duration_ms)
    if not segments and not (allow_empty or audio_layers):
        raise ValueError("В файле не найдено разборчивой речи.")
    if automatic and segments:
        if detected not in supported:
            if clip_languages:
                raise ValueError("Whisper определил язык вне списка. Выбери язык ролика вручную.")
            raise ValueError("Whisper определил другой язык. Выбери русский или английский вручную.")
        language = detected
    elif automatic and not audio_layers:
        return transcript_result([], duration_ms + audio_start_ms, "auto")
    elif automatic:
        language = detected if detected in supported else "ru"
    overlaps = audio_overlaps(turns, audio_start_ms)
    sent_prefix = False

    def publish_prefix(preview):
        """Publish stable repaired cues before waiting for the rest of the retry batch."""
        nonlocal sent_prefix
        preview = prepare_media_cues(preview, turns, small, audio_start_ms, overlaps)
        draft = transcript_result(preview, duration_ms + audio_start_ms, language)
        draft["recognitionModel"] = "whisper-small" if small else "whisper-base"
        draft["recognitionRecovery"] = {
            "state": "running", "coveredEndMs": preview[-1]["endMs"],
            "previewRanges": [{"startMs": cue["startMs"], "endMs": cue["endMs"]} for cue in preview],
        }
        if automatic:
            draft["languageMode"] = "auto"
        publish("base", draft)
        sent_prefix = True

    recovery = None
    if quality and clip_languages and not allow_empty:
        # The word-level diarization fallback can regroup neighboring cues later.
        on_prefix = publish_prefix if publish is not None and (small or not turns) else None
        segments, recovery = recover_clip(wav, work, segments, language, emit=emit, on_prefix=on_prefix)
    segments = prepare_media_cues(segments, turns, small, audio_start_ms, overlaps)
    result = transcript_result(segments, duration_ms + audio_start_ms, language)
    if turns:
        result['speechTurns'] = [{**turn, 'startMs': turn['startMs'] + audio_start_ms,
                                 'endMs': turn['endMs'] + audio_start_ms} for turn in turns]
    if automatic:
        result["languageMode"] = "auto"
    with stage(emit, "dialogue-pauses"):
        result["dialoguePauses"] = measure_dialogue_pauses(wav, segments, audio_start_ms, overlaps, turns)
    if overlaps:
        result["speechOverlaps"] = overlaps
    if quality:
        result["recognitionModel"] = "whisper-small" if small else "whisper-base"
        if not small:
            result["recognitionWarning"] = "Whisper Small не установлен. Использован Whisper Base."
    if recovery is not None:
        for window in recovery["retried"] + recovery["unresolved"]:
            window["startMs"] += audio_start_ms
            window["endMs"] += audio_start_ms
        result["recognitionRecovery"] = recovery
        if recovery["unresolved"]:
            result["recognitionWarning"] = "Некоторые фрагменты речи требуют ручной проверки."
    if speakers:
        result["speakerCount"] = len({row["speaker"] for row in segments if row.get("speaker") is not None})
        if not warning and speaker_count is not None and result["speakerCount"] != speaker_count:
            warning = f"Размечено говорящих: {result['speakerCount']} из {speaker_count}. Проверь метки."
        if warning:
            result["speakerWarning"] = warning
    if publish is not None:
        publish("speech" if sent_prefix else "base", result)
    if audio_layers and not defer_audio:
        detail = run_batch_stage(analyze_audio_scene, wav, segments, audio_start_ms, overlaps, emit, audio_classifier)
        result.update(detail)
        if publish is not None:
            publish("audio", detail)
    return result


def annotate_visible_speakers(cues, evidence):
    """Attach confirmed face tracks, assigning display numbers when audio has no label."""
    by_start = evidence["phrases"]
    confirmed = {}
    for cue in cues:
        activity = by_start.get(str(cue["startMs"]), {}).get("activity", {})
        if activity.get("state") != "visible-active":
            continue
        track = activity["track"]
        cue["visualSpeaker"] = track
        face = activity.get("face", track)
        cue["visualFace"] = face
        if cue.get("speaker") is not None:
            confirmed.setdefault(face, set()).add(cue["speaker"])
    assigned = {track: next(iter(numbers)) for track, numbers in confirmed.items() if len(numbers) == 1}
    next_number = max((cue.get("speaker") or 0 for cue in cues), default=0) + 1
    for cue in cues:
        face = cue.get("visualFace")
        if face is None or cue.get("speaker") is not None or (face in confirmed and face not in assigned):
            continue
        if face not in assigned:
            assigned[face] = next_number
            next_number += 1
        cue["visualSpeakerNumber"] = assigned[face]


def render_srt(segments, language="ru"):
    """Export a downstream name, editorial number, acoustic label or visual fallback."""
    cues = []
    for index, segment in enumerate(segments, 1):
        start = srt_time(segment["startMs"])
        end = srt_time(segment["endMs"])
        label = mixed_label(segment, language) or segment.get("speakerName")
        number = (segment.get("speakerNumber") or segment.get("linkedSpeakerNumber")
                  or segment.get("profileSpeakerNumber") or segment.get("speaker") or segment.get("visualSpeakerNumber"))
        if not label and number is not None:
            label = f"Speaker {number}" if language == "en" else f"Спикер {number}"
        if not label and "speaker" in segment:
            label = "Unknown speaker" if language == "en" else "Спикер не определён"
        prefix = f"{label}: " if label else ""
        cues.append(f"{index}\n{start} --> {end}\n{prefix}{segment['text']}")
    return "\n\n".join(cues) + ("\n" if cues else "")


def transcript_result(segments, duration_ms, language):
    """Serialize timed speech in the common file, stream and capture format."""
    annotate_person_mentions(segments, language)
    text = " ".join(item["text"] for item in segments)
    return {"text": text, "segments": segments, "srt": render_srt(segments, language), "language": language,
            "durationMs": duration_ms, "corrections": corrections(text, language), "contextVersion": VERSION}
