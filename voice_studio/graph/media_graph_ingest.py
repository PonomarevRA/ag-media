"""Index local media on its playback clock using bounded transcription windows."""

from voice_studio.mod_runtime import analysis_run

from decimal import Decimal, InvalidOperation
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
from voice_paths import temporary_directory
from uuid import NAMESPACE_URL, uuid5
import wave

from voice_studio.core.analysis_stages import ignore_progress, stage
from voice_studio.audio.audio_scene import analyze_audio_scene, clean_audio_layers
from voice_studio.audio.dialogue_pauses import measure_dialogue_pauses, shift_dialogue_pauses
from voice_studio.synthesis.engines import ffmpeg
from voice_studio.vision.facial_expressions import finalize_expressions
from voice_studio.vision.face_speakers import bind_faces
from voice_studio.graph.frame_change_graph import change_coverage, shift_change_rows
from voice_studio.vision.person_motion import measure_people_motion
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver, result_fingerprint
from voice_studio.graph.media_stage_selection import StageSelection
from voice_studio.core.media_resources import run_batch_stage
from voice_studio.core.voice_registry_lock import serialize_voice_windows
from voice_studio.speech.media_transcript import (
    FORMATS, analyze_visual_track, attach_person_evidence,
    render_srt, transcribe_media, transcribe_wav,
)
from voice_studio.speech.speaker_diarization import detect_turns
from voice_studio.speech.transcription_languages import CLIP_LANGUAGES
from voice_studio.speech.speaker_profiles import collect_profiles, cue_number
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING
from voice_studio.vision.video_scene import review_scene
from voice_studio.vision.shot_moodbook import build_moodbook

MAX_DURATION_MS = 7_200_000
SPEECH_WINDOW_MS = 120_000
FPS_STEP_MS = 40
ID_BLOCK = 1_000_000
NO_SPEECH = "В файле не найдено разборчивой речи."
LOCAL_FORMATS = ("-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS)
SHOWINFO_TIMES = re.compile(r"\bpts_time:([\d]+(?:\.\d+)?)\b")
SPEAKER_FIELDS = (
    "speaker", "speakerNumber", "linkedSpeakerNumber", "profileSpeakerNumber", "visualSpeakerNumber",
)


def milliseconds(value):
    """Convert finite container seconds to playback milliseconds, without float drift."""
    try:
        number = Decimal(str(value))
        return round(number * 1000) if number.is_finite() else None
    except InvalidOperation:
        return None


def probe_source(source, *, timeout=15):
    """Read the duration and real audio/video streams of an explicit local file."""
    source = Path(source)
    if not source.is_absolute() or not source.is_file():
        raise ValueError("Укажи абсолютный путь к локальному медиафайлу.")
    try:
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", *LOCAL_FORMATS, "-show_entries",
             "format=format_name,start_time,duration:stream=index,codec_type,start_time,duration,width,height:stream_disposition=attached_pic:stream_side_data=rotation",
             "-of", "json", str(source)],
            stderr=subprocess.DEVNULL, timeout=min(15, max(.001, timeout)),
        )
        info = json.loads(raw)
        container = info.get("format", {})
        origin = milliseconds(container.get("start_time")) or 0
        streams = []
        for row in info.get("streams", []):
            kind = row.get("codec_type")
            if kind not in {"audio", "video"} or row.get("disposition", {}).get("attached_pic"):
                continue
            start = milliseconds(row.get("start_time"))
            start = max(0, start - origin) if start is not None else 0
            length = milliseconds(row.get("duration"))
            stream = {"index": row["index"], "kind": kind, "startMs": start, "durationMs": length}
            width, height = row.get("width"), row.get("height")
            if kind == "video" and width and height:
                rotation = next((part.get("rotation", 0) for part in row.get("side_data_list", [])), 0)
                if rotation % 180:
                    width, height = height, width
                stream.update(width=width, height=height)
            streams.append(stream)
        duration = milliseconds(container.get("duration"))
        if duration is not None and 0 < origin < duration and "matroska" in container.get("format_name", "").split(","):
            # Matroska segment durations are on the container clock, including a positive origin.
            duration -= origin
        if duration is None:
            duration = max((row["startMs"] + (row["durationMs"] or 0) for row in streams), default=0)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось прочитать параметры локального медиафайла.") from exc
    audio = next((row for row in streams if row["kind"] == "audio"), None)
    video = next((row for row in streams if row["kind"] == "video"), None)
    if not (audio or video):
        raise ValueError("В файле нет дорожек видео или звука.")
    if not 1 <= duration <= MAX_DURATION_MS:
        raise ValueError("Граф поддерживает локальные файлы длительностью до двух часов.")
    return {"kind": "video" if video else "audio", "durationMs": duration,
            "hasAudio": audio is not None, "streams": streams}


def spans(duration, window_ms=SPEECH_WINDOW_MS):
    """Yield contiguous half-open playback intervals, retaining the final partial interval."""
    return [(start, min(start + window_ms, duration)) for start in range(0, duration, window_ms)]


def interval(start, end, status, **facts):
    """Describe the coverage of one analysis stage on the original clock."""
    return {"startMs": start, "endMs": end, "status": status, **facts}


def input_window(source, start, end):
    """Select an accurate decoded interval on the container playback clock."""
    return [*LOCAL_FORMATS, "-ss", f"{start / 1000:.3f}",
            "-t", f"{(end - start) / 1000:.3f}", "-i", str(source)]


def extract_speech(source, work, start, end, audio):
    """Decode one bounded PCM window and retain any delayed first audio sample."""
    begin = max(start, audio["startMs"])
    if begin >= end:
        return None, 0, 0
    path = work / "window.wav"
    ffmpeg([*input_window(source, begin, end), "-map", f"0:{audio['index']}",
            "-vn", "-sn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)], timeout=90)
    with wave.open(str(path)) as audio_file:
        duration = audio_file.getnframes() * 1000 // audio_file.getframerate()
    return (path if duration >= 350 else None), begin - start, duration


def visual_clip(source, output, start, end):
    """Prepare a bounded video-only proxy, keeping enough resolution for OCR."""
    ffmpeg([*input_window(source, start, end), "-map", "0:V:0", "-an", "-sn",
            "-vf", "scale='min(1920,iw)':'min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-threads:v", "2",
            "-fps_mode", "passthrough", str(output)], timeout=65)


def scan_shots(source, start, end):
    """Measure coarse hard cuts from a 5 fps thumbnail stream over the entire interval."""
    lookbehind = max(0, start - 600)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", *input_window(source, lookbehind, end),
         "-map", "0:V:0", "-an", "-sn", "-filter_threads", "2",
         "-vf", "fps=5:start_time=0,scale=112:112:force_original_aspect_ratio=decrease,select='gt(scene,0.30)',showinfo",
         "-f", "null", "-"],
        stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        check=True, timeout=120, text=True,
    )
    cuts = []
    for match in SHOWINFO_TIMES.finditer(raw.stderr):
        at = lookbehind + milliseconds(match.group(1))
        if start <= at < end and (not cuts or cuts[-1] + 400 <= at):
            cuts.append(at)
    return cuts


def shifted_spans(items, offset, maximum, id_base=0):
    """Offset bounded intervals, retaining only a window's own time and identity evidence."""
    shifted = []
    for item in items:
        start = max(0, item["startMs"]) + offset
        end = min(maximum, item["endMs"] + offset)
        if start >= end:
            continue
        row = {**item, "startMs": start, "endMs": end}
        for key in SPEAKER_FIELDS + ("visualSpeaker", "visualFace"):
            if type(row.get(key)) is int:
                row[key] += id_base
        for key in ("speakers", "tracks", "faces"):
            if key in row:
                row[key] = [number + id_base for number in row[key]]
        if row.get("overlaps"):
            row["overlaps"] = shifted_spans(row["overlaps"], offset, maximum, id_base)
        if window := row.pop("speakerAudioWindow", None):
            audio_start = max(start, window["startMs"] + offset)
            audio_end = min(end, window["endMs"] + offset)
            if audio_start < audio_end:
                row["speakerAudioWindow"] = {"startMs": audio_start, "endMs": audio_end}
        shifted.append(row)
    return shifted


def face_coverage(visual, start, end, **facts):
    """Use measured window coverage so empty or failed frames never mean analyzed."""
    windows = (visual.get("activityCoverage") or {}).get("windows")
    if windows:
        rows = []
        for window in windows:
            left = start + window["startMs"]
            right = min(end, start + window["endMs"])
            if left >= right:
                continue
            covered = min(right, start + window["coveredEndMs"])
            if covered > left:
                rows.append(interval(left, covered, "sampled", **facts))
            if covered < right:
                rows.append(interval(covered, right, window["status"], **facts))
        last = rows[-1]["endMs"] if rows else start
        if last < end:
            rows.append(interval(last, end, "pending" if visual.get("stages", {}).get("light-asd") == "running"
                                 else "unavailable", **facts))
        return rows
    frames = (visual.get("timeline") or {}).get("frames", [])
    covered = min(end, start + len(frames) * FPS_STEP_MS)
    if covered > start:
        return [interval(start, covered, "sampled", **facts)]
    if visual:
        return [interval(start, end, "unavailable", statuses=visual.get("stages", {}), **facts)]
    return []


def apply_visual_evidence(result, visual, start, end, id_base, speech_start):
    """Append face geometry on the original clock, keeping identities scoped to the window."""
    combined = result["videoEvidence"]
    timeline = visual.get("timeline") or {}
    frames = timeline.get("frames", [])
    if frames and timeline.get("stepMs") == FPS_STEP_MS:
        destination = combined["timeline"]["frames"]
        if not destination:
            # A selective run can restore cached faces while the old visual flag is off.
            destination.extend([()] * ((result["durationMs"] + FPS_STEP_MS - 1) // FPS_STEP_MS))
        first = start // FPS_STEP_MS
        stop = min(len(destination), (end + FPS_STEP_MS - 1) // FPS_STEP_MS)
        for index, hits in enumerate(frames[:stop - first], first):
            if hits:
                destination[index] = [[hit[0] + id_base, *hit[1:6],
                                       (hit[6] if len(hit) > 6 else hit[0]) + id_base] for hit in hits]
        if timeline.get("estimatedFrames"):
            combined["timeline"].setdefault("estimatedFrames", []).extend(
                first + index for index in timeline["estimatedFrames"] if 0 <= index < stop - first)
    result["coverage"]["face"].extend(face_coverage(visual, start, end, windowStartMs=speech_start))

    for face, binding in visual.get("faceSpeakers", {}).items():
        combined["faceSpeakers"][int(face) + id_base] = {
            **binding, "speaker": binding["speaker"] + id_base, "windowStartMs": speech_start,
        }
    if layer := visual.get("facialExpressions"):
        target = combined.setdefault("facialExpressions", {"samples": [], "windowStatuses": []})
        target["windowStatuses"].append(layer.get("status", "unavailable"))
        target["samples"].extend({**row, "timeMs": row["timeMs"] + start,
                                  "track": row["track"] + id_base, "faceId": row["faceId"] + id_base}
                                 for row in layer.get("samples", []) if row["timeMs"] + start < end)
    source_editor = visual.get("editorTimeline", {})
    editor = combined["editorTimeline"]
    if pilot := source_editor.get("personPilot"):
        target = editor.setdefault("personPilot", {"samples": [], "windows": []})
        target["samples"].extend({**row, "timeMs": start + row["timeMs"]} for row in pilot["samples"]
                                 if start + row["timeMs"] < end)
        target["windows"].extend(shifted_spans(pilot["windows"], start, end))
    for key in ("faceAttributes", "clothingByFace"):
        for face, facts in source_editor.get(key, {}).items():
            shifted = dict(facts)
            if "timeMs" in shifted:
                shifted["timeMs"] += start
            if "sampleTimesMs" in shifted:
                shifted["sampleTimesMs"] = [sample + start for sample in shifted["sampleTimesMs"]]
            editor[key][int(face) + id_base] = shifted
    combined["simultaneousFaces"].extend(shifted_spans(visual.get("simultaneousFaces", []), start, end, id_base))


def offset_screen_panel(panel, offset):
    """Preserve document boxes and shift only their references to the window clock."""
    shifted = dict(panel)
    for field in ("readingTimeMs", "lastObservedMs", "previousStateMs"):
        if type(panel.get(field)) is int:
            shifted[field] += offset
    if "elements" in panel:
        shifted["elements"] = [
            {**item, "mosaic": {**address, "pageStartMs": address["pageStartMs"] + offset}}
            if isinstance((address := item.get("mosaic")), dict) and type(address.get("pageStartMs")) is int else item
            for item in panel["elements"]
        ]
    return shifted


def apply_scene_evidence(result, scene, start, end):
    """Align window-local scene facts to the source clock without changing checkpoints.

    Sampled instants exclude the window end; complete scroll intervals may end there.
    """
    sampled = [{**row, "timeMs": row["timeMs"] + start} for row in scene.get("frames", [])
               if row["timeMs"] + start < end]
    for row in sampled:
        row["objects"] = [offset_object_matches(item, start) for item in row.get("objects", [])]
        if "interfaces" in row:
            row["interfaces"] = [offset_screen_panel(panel, start) for panel in row["interfaces"]]
    result["sceneEvidence"]["frames"].extend(sampled)
    if activity := scene.get("uiActivity"):
        target = result["sceneEvidence"].setdefault("uiActivity", {"version": activity.get('version', 1), "panels": [], "samples": []})
        target['version'] = max(target.get('version', 1), activity.get('version', 1))
        for key in ('meaning', 'geometryDiscoveryIntervalMs'):
            if key in activity:
                target[key] = activity[key]
        target["truncated"] = target.get("truncated", False) or activity.get("truncated") is True
        target["panels"].extend({**panel, "id": f"{start}:{panel['id']}",
                                 "firstMs": start + panel["firstMs"], "lastMs": start + panel["lastMs"],
                                 **{key: start + panel[key] for key in ("retiredMs", "lastOutlineMs") if type(panel.get(key)) is int}}
                                for panel in activity.get("panels", []))
        target["samples"].extend({**sample, "panelId": f"{start}:{sample['panelId']}",
                                  "timeMs": start + sample["timeMs"]}
                                 for sample in activity.get("samples", []) if start + sample["timeMs"] < end)
    if scene.get("interfaceScrolls"):
        result["sceneEvidence"].setdefault("interfaceScrolls", []).extend(
            {**row, "startMs": row["startMs"] + start, "endMs": row["endMs"] + start,
             **({"controls": [{**item, "timeMs": item["timeMs"] + start} if "timeMs" in item else item
                              for item in row["controls"]]} if "controls" in row else {}),
             **({"mosaicText": [{**item, "timeMs": item["timeMs"] + start} if "timeMs" in item else item
                                for item in row["mosaicText"]]} if "mosaicText" in row else {}),
             "viewports": [{**view, "timeMs": view["timeMs"] + start} for view in row.get("viewports", ())]}
            for row in scene["interfaceScrolls"] if row["endMs"] + start <= end)
    result["sceneEvidence"].setdefault("cameraMotion", []).extend(
        shifted_spans(scene.get("cameraMotion", []), start, end))
    if scene.get("frameChanges"):
        changes = result["sceneEvidence"].setdefault("frameChanges", {"rows": []})
        changes["rows"].extend(shift_change_rows(scene, start, end))
    base = (start // SPEECH_WINDOW_MS) * ID_BLOCK
    result["sceneEvidence"].setdefault("moodbook", []).extend(
        {**row, "representativeMs": row["representativeMs"] + start,
         "characters": [{**item, "faceId": item["faceId"] + base} for item in row["characters"]]}
        for row in shifted_spans(scene.get("moodbook", []), start, end))
    result["coverage"]["scene"].append(scene_coverage(scene, start, end, len(sampled)))
    result.setdefault("shotColors", []).extend(
        {**row, "startMs": start + row["startMs"]} for row in scene.get("shotColors", [])
        if start + row["startMs"] < end)


def offset_object_matches(item, start):
    """Move pair references onto the source clock without changing cached scene data."""
    shifted = item
    for field in ("objectMatches", "objectPartMatches"):
        if not item.get(field):
            continue
        if shifted is item:
            shifted = dict(item)
        shifted[field] = []
        for match in item[field]:
            shifted[field].append({**match, "timeMs": match["timeMs"] + start})
    return shifted


def apply_person_motion(result, evidence, scene, start, end, id_base):
    """Measure within one native face window, then align the intervals and IDs to the source."""
    if not (evidence.get("timeline") or {}).get("frames"):
        return
    rows = measure_people_motion(evidence, scene.get("cameraMotion", []), end - start,
                                 shot_starts=scene.get("shotStartsMs", ()))
    editor = result["videoEvidence"]["editorTimeline"]
    editor.setdefault("personMotion", []).extend(
        {**row, "faceId": row["faceId"] + id_base, "track": row["track"] + id_base}
        for row in shifted_spans(rows, start, end, id_base))


def scene_coverage(scene, start, end, samples):
    """Expose completed and degraded scene evidence with its sampling policy."""
    status = scene.get("status", "unavailable")
    stages = scene.get("stages", {})
    if status == "ok":
        complete = all(part == "ok" for part in stages.values()) and scene.get("interfaceStatus", "ok") == "ok"
        status = "sampled" if complete else "partial"
    return interval(start, end, status, samples=samples, statuses=stages,
                    objectSampling=scene.get("objectSampling"), objectFrames=scene.get("objectFrames"),
                    interfaceFrames=scene.get("interfaceFrames"), interfaceOcrCrops=scene.get("interfaceOcrCrops"),
                    interfaceStatus=scene.get("interfaceStatus"), interfaceOcrStatuses=scene.get("interfaceOcrStatuses"),
                    **({"frameChanges": detail} if (detail := change_coverage(scene)) else {}))


def speech_coverage(start, end, report, status="ok", offset=0):
    """Keep pending and unresolved speech spans explicit on the source clock."""
    if start >= end:
        return []
    recovery = report.get("recognitionRecovery") or {}
    preview = recovery.get("state") == "running"
    marked = recovery.get("previewRanges" if preview else "unresolved") or ()
    base_status, marked_status = ("pending", "partial") if preview else (status, "unresolved")
    facts = {"recognitionModel": report.get("recognitionModel"), "speakerWarning": report.get("speakerWarning")}
    result = []
    cursor = start
    for span in marked:
        left, right = max(cursor, span["startMs"] + offset), min(end, span["endMs"] + offset)
        if left >= right:
            continue
        if left > cursor:
            result.append(interval(cursor, left, base_status, **facts))
        result.append(interval(left, right, marked_status, **facts))
        cursor = right
    if cursor < end:
        result.append(interval(cursor, end, base_status, **facts))
    return result


def attach_coverage(result, metadata, visual, scene):
    """Expose the exact analyzed windows when the original compact path is used."""
    duration = metadata["durationMs"]
    evidence = result.get("videoEvidence", {})
    timeline = evidence.get("timeline") or {}
    scene_data = result.get("sceneEvidence") or {}
    visual = visual or bool(evidence)
    scene = scene or bool(scene_data)
    frames = min(duration, len(timeline.get("frames", [])) * timeline.get("stepMs", FPS_STEP_MS))
    audio_start = next(row["startMs"] for row in metadata["streams"] if row["kind"] == "audio")
    audio_end = min(duration, result["durationMs"])
    coverage = {"speech": [], "face": [], "scene": [], "shots": []}
    if audio_start:
        coverage["speech"].append(interval(0, min(duration, audio_start), "no-audio"))
    speech_state = (result.get("analysisStages", {}).get("speech") or {}).get("state")
    speech_status = "pending" if speech_state in ("pending", "invalidated") else "ok"
    coverage["speech"].extend(speech_coverage(audio_start, audio_end, result, speech_status))
    if audio_end < duration:
        coverage["speech"].append(interval(audio_end, duration, "no-audio"))
    if metadata["kind"] == "video":
        if visual:
            coverage["face"] = face_coverage(evidence, 0, duration)
        if scene:
            coverage["scene"] = [scene_coverage(scene_data, 0, duration, len(scene_data.get("frames", [])))]
        if frames and evidence.get("editorTimeline", {}).get("shots"):
            coverage["shots"] = [interval(0, frames, "ok", method="light-asd")]
    result.update(durationMs=duration, coverage=coverage, streams=metadata["streams"],
                  speakerScope="source")
    return result


def collect_faces(clip, analysis, duration, *, people_count=None, people_origin=None, emit=ignore_progress):
    """Enrich in-window speech cues and return their independent visual evidence."""
    wav, _, _ = analysis["audio_file"]()
    local_cues = []
    if wav:
        for cue in analysis["segments"]:
            if cue["startMs"] < duration:
                local_cues.append({**cue, "endMs": min(cue["endMs"], duration)})
    local = {"segments": local_cues, "durationMs": duration}
    run_batch_stage(analyze_visual_track, clip, wav, analysis["offset"], local, people_count=people_count,
                    people_origin=people_origin, emit=emit)
    bind_faces(local)
    by_start = {cue["startMs"]: cue for cue in local_cues}
    for cue in analysis["segments"]:
        updated = by_start.get(cue["startMs"])
        if updated:
            cue.update({key: value for key, value in updated.items() if key != "endMs"})
    return local["videoEvidence"]


def scene_revolver(stages, evidence, object_sampling):
    """Bind scene reuse to the cut timestamps and occupancy, omitting face thumbnails."""
    editor = evidence.get("editorTimeline", {})
    cuts = [row["startMs"] for row in editor.get("shots", [])]
    return stages.child_for_input("scene-full-window", {"cuts": cuts, "timeline": evidence.get("timeline"),
                                                       "shotColorsVersion": 1, "objectAppearanceVersion": 1, "interfaceVersion": 5,
                                                       "objectSampling": object_sampling})


def collect_visual(source, work, window, analysis, visual, scene, *, object_sampling=DEFAULT_OBJECT_SAMPLING,
                   revolver=None, people_count=None, emit=ignore_progress, body=False, gaze=False):
    """Reuse face and scene checkpoints while scanning objects on the entire original window."""
    if not (visual or scene):
        return {}, {}
    stages = revolver or AnalysisRevolver()
    face = stages.load("face-reid-windowed-1", required=("evidence", "segments")) if visual else None
    if face is not None:
        analysis["segments"][:] = face["segments"]
    evidence = face["evidence"] if face is not None else {}
    face_complete = face is not None or not visual
    if visual and face is None:
        start, end = window
        clip = work / "scene.mp4"
        try:
            visual_clip(source, clip, start, end)
        except (OSError, ValueError, subprocess.SubprocessError):
            evidence = {"stages": {"proxy": "failed"}}
        else:
            evidence = collect_faces(clip, analysis, end - start, people_count=people_count,
                                     people_origin=(source, start), emit=emit)
            face_complete = all(status in {"ok", "no-phrases"} for status in evidence.get("stages", {}).values())
            if face_complete:
                stages.save("face-reid-windowed-1", {"evidence": evidence, "segments": analysis["segments"]})
    if visual and (body or gaze):
        person_stage = stages.child_for_input('person-details', {'timeline': evidence.get('timeline'),
                                                                'body': body, 'gaze': gaze})
        people = person_stage.load('complete', required=('frames',))
        if people is None:
            clip = work / 'scene.mp4'
            if not clip.exists():
                visual_clip(source, clip, *window)
            with stage(emit, 'person-layer'):
                run_batch_stage(attach_person_evidence, clip, work, evidence, body=body, gaze=gaze, emit=emit)
            people = evidence.get('personLayer')
            if isinstance(people, dict) and isinstance(people.get('frames'), list) and not people.get('gazeUnavailable'):
                person_stage.save('complete', people)
        if people is not None:
            evidence['personLayer'] = people
    if not scene:
        return evidence, {}
    scene_stage = scene_revolver(stages, evidence, object_sampling)
    scene_data = scene_stage.load("analysis", required=("status", "frames"))
    if scene_data is None:
        with stage(emit, 'scene'):
            scene_data = run_batch_stage(review_scene, source, work, FORMATS, evidence,
                                         object_sampling=object_sampling, window=window)
        if scene_data.get("status") == "ok":
            try:
                scene_data["moodbook"] = build_moodbook(source, FORMATS, scene_data, evidence,
                                                        window[1] - window[0], window=window)
            except (OSError, ValueError, subprocess.SubprocessError):
                scene_data["moodbookStatus"] = "unavailable"
        scene_complete = scene_data.get("status") == "ok" and all(
            status == "ok" for status in scene_data.get("stages", {}).values())
        scene_complete = scene_complete and scene_data.get("interfaceStatus", "ok") == "ok"
        if face_complete and scene_complete:
            scene_stage.save("analysis", scene_data)
    return evidence, scene_data



def collect_selected_visual(source, work, window, analysis, visual, scene, cache, *,
                            object_sampling, people_count, emit, body, gaze):
    """Use the same bounded workers and shared layer cache for incremental window runs."""
    from voice_studio.graph.media_stage_cache import usable_visual, usable_scene, scene_inputs

    def faces():
        """A forced face pass bypasses its legacy inner checkpoint as well."""
        local = deepcopy(analysis)
        evidence, _ = collect_visual(source, work, window, local, True, False,
                                      revolver=AnalysisRevolver(), people_count=people_count, emit=emit,
                                      object_sampling=object_sampling)
        return {"evidence": evidence, "segments": local["segments"]}

    evidence = {}
    if visual or "visual" in cache.manifest["layers"]:
        face = cache.run("visual", {"peopleCount": people_count}, analysis["segments"], faces,
                         lambda value: usable_visual({"videoEvidence": value.get("evidence") or {}}), emit=emit)
        evidence = face.get("evidence") or {}
        if "segments" in face:
            analysis["segments"][:] = face["segments"]
    if (body or gaze or "people" in cache.manifest["layers"]) and (evidence.get("timeline") or {}).get("frames"):
        def people():
            """Keep the production figure/gaze worker and its source-clock proxy."""
            clip = work / "scene.mp4"
            if not clip.exists():
                visual_clip(source, clip, *window)
            run_batch_stage(attach_person_evidence, clip, work, evidence, body=body, gaze=gaze, emit=emit)
            return evidence.get("personLayer") or {}

        layer = cache.run("people", {"body": body, "gaze": gaze}, evidence["timeline"], people,
                          lambda value: isinstance(value.get("frames"), list) and not value.get("gazeUnavailable"), emit=emit)
        if layer:
            evidence["personLayer"] = layer
    scene_data = {}
    if scene or "scene" in cache.manifest["layers"]:
        def measure_scene():
            """Measure scene inputs independently from optional person details."""
            with stage(emit, "scene"):
                measured = run_batch_stage(review_scene, source, work, FORMATS, evidence,
                                           object_sampling=object_sampling, window=window)
            if measured.get("status") == "ok":
                try:
                    measured["moodbook"] = build_moodbook(source, FORMATS, measured, evidence,
                                                          window[1] - window[0], window=window)
                except (OSError, ValueError, subprocess.SubprocessError):
                    measured["moodbookStatus"] = "unavailable"
            return measured

        scene_data = cache.run("scene", {"sampling": object_sampling}, scene_inputs(evidence), measure_scene,
                               usable_scene, emit=emit)
    return evidence, scene_data


def add_shots(result, source, start, end, evidence, *, scene=None, revolver=None):
    """Prefer native cut evidence where complete, otherwise scan the whole partition."""
    editor = evidence.get("editorTimeline", {})
    native = editor.get("shots", [])
    length = editor.get("durationMs", 0)
    if scene and "shotStartsMs" in scene:
        stamps = [start + stamp for stamp in scene["shotStartsMs"]]
        status, method = "ok", "scene-native"
    elif native and length >= end - start - FPS_STEP_MS:
        stamps = [start + row["startMs"] for row in native]
        status, method = "ok", "light-asd"
    else:
        stages = revolver or AnalysisRevolver()
        cached = stages.load("cuts", required=("stamps",))
        method = "ffmpeg-5fps"
        if cached is not None:
            stamps, status = cached["stamps"], "ok"
        else:
            try:
                stamps, status = scan_shots(source, start, end), "ok"
                stages.save("cuts", {"stamps": stamps})
            except (OSError, ValueError, subprocess.SubprocessError):
                stamps, status = [], "failed"
    bounds = result["coverage"]["shots"]
    if status == "ok" and (not bounds or bounds[-1]["status"] != "ok"):
        stamps.append(start)
    bounds.append(interval(start, end, status, method=method))
    result["videoEvidence"]["editorTimeline"]["shots"].extend(
        {"startMs": stamp, "basis": method} for stamp in sorted(set(stamps)) if start <= stamp < end
    )


def window_profiles(source, start, local, evidence, id_base):
    """Make graph-local profile references without registering cross-window identities."""
    records = collect_profiles({"segments": local, "videoEvidence": evidence})
    for row in records:
        number = row["number"]
        identity = f"{source.as_uri()}#window={start}&speaker={number}"
        row.update(id=uuid5(NAMESPACE_URL, identity).hex, scope="window",
                   windowStartMs=start, localNumber=number, number=number + id_base,
                   anchorsMs=[anchor + start for anchor in row["anchorsMs"]])
        if row["faceId"] is not None:
            row["faceId"] += id_base
        if row.get("clothing"):
            clothing = row["clothing"]
            row["clothing"] = {**clothing,
                               "sampleTimesMs": [time + start for time in clothing.get("sampleTimesMs", [])]}
    return records


def recognize_window(audio_file, language, speaker_count, known_silent, work, *, speakers=True, emit=ignore_progress,
                     quality=True, merge_similar=False):
    """Return the speech result and PCM coverage without saving a temporary WAV path."""
    wav, offset, audio_ms = audio_file()
    local = {"segments": []}
    status = "empty"
    if wav and not known_silent:
        try:
            local = transcribe_wav(wav, work, language, offset, speakers=speakers,
                                   speaker_count=speaker_count, quality=quality, merge_similar=merge_similar,
                                   clip_languages=True, emit=emit)
            status = "ok"
        except ValueError as exc:
            if str(exc) != NO_SPEECH:
                raise
    return {"local": local, "offset": offset, "audioMs": audio_ms, "status": status}


@serialize_voice_windows
def analyze_windows(studio_root, source, metadata, language, visual, scene, speaker_count, *,
                    known_silent=False, object_sampling=DEFAULT_OBJECT_SAMPLING, audio_classifier="auto", revolver=None,
                    publish=None, people_count=None, speakers=True, emit=ignore_progress,
                    quality=True, merge_similar=False, body=False, gaze=False, selection=None, cache_revision=None,
                    object_reconciliation=False):
    """Combine bounded stages, retaining window labels beside source-scoped voice identities."""
    from voice_studio.graph.media_stage_cache import IncrementalStages
    selection = selection or StageSelection()
    duration = metadata["durationMs"]
    audio = next((row for row in metadata["streams"] if row["kind"] == "audio"), None)
    parts = spans(duration)
    stages = revolver or AnalysisRevolver()
    is_video = metadata["kind"] == "video"
    result = {"language": language, "durationMs": duration, "segments": [], "dialoguePauses": [], "audioLayers": [],
              "streams": metadata["streams"],
              "coverage": {"speech": [], "face": [], "scene": [], "shots": []}, "speakerProfileIndex": [],
              "speakerScope": "window" if len(parts) > 1 else "source"}
    result['speechTurns'] = []
    from voice_studio.graph.media_stage_cache import source_digest
    from voice_studio.speech.voice_profiles import VoiceProfileRegistry, VERSION as VOICE_VERSION
    from voice_studio.speech.speaker_pipeline import attach_voice_profiles, project_voice_index, attach_dialogue_audit, profile_visual_evidence
    source_id = source_digest(source)
    result['sourceDigest'] = source_id
    registry_group = stages.directory / 'voice-registry' if stages.directory else None
    if registry_group is not None and cache_revision is not None:
        registry_group = registry_group / cache_revision
    registry_reused = bool(registry_group and registry_group.is_dir())
    speakers = speakers or (selection.selective and registry_reused)
    voice_prefix = result_fingerprint([source_id, cache_revision])
    registry = None
    registry_unavailable = False
    if is_video:
        result["videoEvidence"] = {"timeline": {"stepMs": FPS_STEP_MS,
                                               "frames": [()] * ((duration + FPS_STEP_MS - 1) // FPS_STEP_MS) if visual else []},
                                   "editorTimeline": {"durationMs": duration, "shots": [], "faceAttributes": {},
                                                      "clothingByFace": {}},
                                   "faceSpeakers": {}, "simultaneousFaces": []}
        result["sceneEvidence"] = {"format": "sampled", "frames": []}
    window_states = []
    for part, (start, end) in enumerate(parts):
        selection.states = {}
        with temporary_directory() as folder:
            work = Path(folder)
            window = stages.child(str(start))
            cache = IncrementalStages(window, cache_revision or "legacy", selection)
            decoded_audio = None

            def audio_file():
                """Decode PCM once, when speech or an unfinished face stage needs it."""
                nonlocal decoded_audio
                if decoded_audio is None:
                    decoded_audio = extract_speech(source, work, start, end, audio) if audio else (None, 0, 0)
                return decoded_audio

            def measure_speech():
                """Keep decode lazy and checkpoint only completed recognition."""
                return recognize_window(audio_file, language, speaker_count, known_silent, work,
                                        speakers=speakers, emit=emit, quality=quality, merge_similar=merge_similar)

            speech = cache.run("speech", {"language": language, "speakers": speakers, "speakerCount": speaker_count,
                                          "quality": quality, "mergeSimilar": merge_similar}, None, measure_speech,
                               lambda value: "local" in value and not value["local"].get("speakerWarning"), emit=emit,
                               fallback={"local": {"segments": []}, "offset": 0, "audioMs": 0, "status": "pending"})
            local = speech["local"]
            if local.get('language') in CLIP_LANGUAGES and result['language'] == 'auto':
                result['language'] = local['language']
                result['languageMode'] = 'auto'
            offset, audio_ms, speech_status = speech["offset"], speech["audioMs"], speech["status"]
            speech_rows = result["coverage"]["speech"]
            audio_end = min(end, start + offset + audio_ms)
            if audio_ms:
                if offset:
                    speech_rows.append(interval(start, start + offset, "no-audio"))
                speech_intervals = speech_coverage(start + offset, audio_end, local, speech_status, offset=start)
                speech_rows.extend(speech_intervals)
            if audio_end < end or not audio_ms:
                speech_rows.append(interval(audio_end if audio_ms else start, end,
                                            "pending" if speech_status == "pending" else "no-audio"))
            preview = None
            if publish is not None:
                current = shifted_spans(local["segments"], start, end, part * ID_BLOCK)
                pauses = shift_dialogue_pauses(local.get("dialoguePauses", []), start, end)
                preview = {**result, "segments": [*result["segments"], *current],
                           "dialoguePauses": [*result["dialoguePauses"], *pauses]}
                publish("base", preview)
            if audio_ms:
                def measure_audio():
                    """Classify audio only when the selected layer cannot reuse its checkpoint."""
                    wav, _, _ = audio_file()
                    return run_batch_stage(analyze_audio_scene, wav, local["segments"], offset,
                                           local.get("speechOverlaps", []), emit, audio_classifier)

                layers = cache.run("audio", {"classifier": audio_classifier}, speech, measure_audio,
                                   lambda value: value.get("soundAnalysisStatus") in ("done", "too-short"),
                                   emit=emit, fallback={"audioLayers": []})
                audio_rows = clean_audio_layers(layers["audioLayers"], audio_end - start)
                for row in audio_rows:
                    result["audioLayers"].append({**row, "startMs": row["startMs"] + start,
                                                  "endMs": row["endMs"] + start})
                if preview is not None and audio_rows:
                    publish("audio", preview)
            evidence = {}
            if is_video:
                analysis = {**local, "audio_file": audio_file, "offset": offset}
                if selection.selective:
                    evidence, scene_data = collect_selected_visual(
                        source, work, (start, end), analysis, visual, scene, cache, object_sampling=object_sampling,
                        people_count=people_count, emit=emit, body=body, gaze=gaze)
                else:
                    from voice_studio.graph.media_stage_cache import usable_visual, usable_scene, scene_inputs
                    face_input = deepcopy(local["segments"])
                    child = window.child_for_input("vision", {"revision": cache.revision, "speech": speech,
                                                               "peopleCount": people_count if visual else False})
                    evidence, scene_data = collect_visual(source, work, (start, end), analysis, visual, scene,
                                                          object_sampling=object_sampling, revolver=child,
                                                          people_count=people_count, emit=emit, body=body, gaze=gaze)
                    if visual:
                        cache.run("visual", {"peopleCount": people_count}, face_input,
                                  lambda: {"evidence": {key: value for key, value in evidence.items() if key != "personLayer"},
                                           "segments": analysis["segments"]},
                                  lambda value: usable_visual({"videoEvidence": value.get("evidence") or {}}))
                    if evidence.get("personLayer"):
                        cache.run("people", {"body": body, "gaze": gaze}, evidence.get("timeline"),
                                  lambda: evidence["personLayer"], lambda value: isinstance(value.get("frames"), list))
                    if scene:
                        cache.run("scene", {"sampling": object_sampling}, scene_inputs(evidence),
                                  lambda: scene_data, usable_scene)
                apply_visual_evidence(result, evidence, start, end, part * ID_BLOCK, start)
                if scene_data:
                    apply_scene_evidence(result, scene_data, start, end)
                apply_person_motion(result, evidence, scene_data, start, end, part * ID_BLOCK)
                if evidence or scene_data:
                    add_shots(result, source, start, end, evidence, scene=scene_data, revolver=window)
            id_base = part * ID_BLOCK
            cues = shifted_spans(local["segments"], start, end, id_base)
            result['speechTurns'].extend(shifted_spans(local.get('speechTurns', []), start, end, id_base))
            if speakers and audio_ms and local.get('speechTurns') and not registry_unavailable:
                visual_proofs = profile_visual_evidence(evidence, offset=start, id_base=id_base)
                voice_input = {'revision': cache.revision, 'segments': cues, 'videoProofs': visual_proofs,
                               'overlaps': local.get('speechOverlaps', [])}
                # Each snapshot contains only this exact prefix of measured windows.
                # A changed later face binding cannot contaminate earlier voice identities.
                voice_prefix = result_fingerprint([voice_prefix, voice_input])
                registry_path = registry_group / voice_prefix / (VOICE_VERSION + '.json') if registry_group is not None else None
                voice_stage = window.child_for_input('voice-profiles-' + VOICE_VERSION, {'prefix': voice_prefix})
                saved_voice = voice_stage.load('complete', required=('segments', 'voiceProfiles')) if (
                    registry_path and registry_path.is_file() and 'voice' not in selection.forced) else None
                try:
                    if saved_voice:
                        registry = VoiceProfileRegistry.load(registry_path, source_id)
                    elif registry is None and selection.runs('voice'):
                        registry = VoiceProfileRegistry(source_id)
                except (OSError, ValueError, ImportError):
                    registry_unavailable = True
                    result['voiceProfileWarning'] = 'Голосовые профили недоступны; исходные метки сохранены.'
                    selection.mark('voice', 'failed')
                    saved_voice = None
                if registry_unavailable:
                    registry = None
                if saved_voice:
                    # A public window checkpoint is valid only against its durable
                    # private registry. Older interrupted writes may have published
                    # a voice ID before the registry containing it reached disk.
                    persisted_ids = {row['id'] for row in registry.index()}
                    matched_ids = {cue['voiceProfileId'] for cue in saved_voice['segments'] if cue.get('voiceProfileId')}
                    if not matched_ids <= persisted_ids:
                        saved_voice = None
                if saved_voice:
                    selection.mark("voice", "reused")
                    cues = saved_voice['segments']
                    result['voiceProfiles'] = saved_voice['voiceProfiles']
                    emit({'stage': 'voice-profiles', 'state': 'done', 'cacheHit': True, 'reused': True})
                elif registry is not None and selection.runs('voice'):
                    layer = {'segments': deepcopy(cues), 'speechOverlaps': shifted_spans(local.get('speechOverlaps', []), start, end, id_base),
                             'videoEvidence': visual_proofs}
                    wav, _, _ = audio_file()
                    public = attach_voice_profiles(layer, wav, start + offset, source_id, registry=registry,
                                                   window_id=start, visual_scope=f'{source_id}:window:{start}', emit=emit)
                    cues = layer['segments']
                    selection.mark('voice', 'done' if public is not None else 'failed')
                    if public is not None:
                        result['voiceProfiles'] = public
                        if registry_path:
                            registry.save(registry_path)
                        voice_stage.save('complete', {'segments': cues, 'voiceProfiles': public})
                elif not registry_unavailable:
                    selection.mark('voice', 'invalidated')
            if len(parts) > 1:
                profiles = window_profiles(source, start, local["segments"], evidence, id_base)
                result["speakerProfileIndex"].extend(profiles)
                by_number = {row["number"]: row["id"] for row in profiles}
                for cue in cues:
                    cue["speakerWindowStartMs"] = start
                    number = cue_number(cue)
                    if number is not None:
                        cue["speakerLocalNumber"] = number - id_base
                    if number in by_number:
                        cue["speakerProfileId"] = by_number[number]
            result["segments"].extend(cues)
            if result.get('voiceProfiles'):
                project_voice_index(result)
                if publish is not None:
                    publish('voice', result)
            result["dialoguePauses"].extend(shift_dialogue_pauses(local.get("dialoguePauses", []), start, end))
            if publish is not None and is_video and part + 1 < len(parts):
                if visual:
                    publish("visual", result)
                if scene:
                    publish("scene", result)
        window_states.append(dict(selection.states))
    selection.states = {}
    priority = {"reused": 0, "done": 1, "pending": 2, "invalidated": 3, "failed": 4}
    for states in window_states:
        for name, value in states.items():
            previous = selection.states.get(name)
            if previous is None or priority[value["state"]] > priority[previous["state"]]:
                selection.states[name] = value
    if registry is not None:
        result['voiceProfiles'] = {'version': VOICE_VERSION, 'sourceId': source_id,
                                   'profiles': registry.index()}
        project_voice_index(result)
    result["srt"] = render_srt(result["segments"], result['language'])
    attach_dialogue_audit(result, source_id, emit=emit)
    if visual and is_video:
        finalize_expressions(result["videoEvidence"], result["segments"], duration)
    return selection.attach(result)


def backfill_dialogue_pauses(studio_root, source, metadata, report):
    """Measure the original PCM for old reports while retaining their saved speech and edits."""
    if isinstance(report.get("dialoguePauses"), list):
        return
    report["dialoguePauses"] = []
    audio = next((row for row in metadata["streams"] if row["kind"] == "audio"), None)
    if audio is None or not report["segments"]:
        return
    studio_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    timed = [cue for cue in report["segments"] if isinstance(cue, dict)
             and type(cue.get("startMs")) is int and type(cue.get("endMs")) is int
             and 0 <= cue["startMs"] < cue["endMs"]]
    for start, end in spans(metadata["durationMs"]):
        cues = [cue for cue in timed if cue["startMs"] < end and cue["endMs"] > start]
        if not cues:
            continue
        try:
            with temporary_directory() as folder:
                wav, offset, _ = extract_speech(source, Path(folder), start, end, audio)
                if wav:
                    overlaps = report.get("speechOverlaps") or [span for cue in cues for span in (cue.get("overlaps") or [])]
                    turns = []
                    if report.get("speakerCount"):
                        try:
                            turns = detect_turns(wav, None)
                        except ValueError:
                            pass
                    measured = measure_dialogue_pauses(wav, cues, start + offset, overlaps, turns)
                    report["dialoguePauses"].extend(measured)
        except (OSError, ValueError, wave.Error, subprocess.SubprocessError):
            continue


def restore_legacy_visual_track(studio_root, source, metadata, report, *, people_count=None):
    """Restore the fields stripped by older browser exports using the normal visual stages."""
    legacy = report.get("videoEvidence") or {}
    if ((legacy.get("timeline") or {}).get("frames")
            or legacy.get("stages", {}).get("light-asd") not in {"ok", "duration-limit"}
            or metadata["kind"] != "video" or not metadata["hasAudio"]):
        return None
    limit = metadata["durationMs"]
    # Rebuilt detector IDs must never inherit a face assignment from a previous run.
    derived = {"visualFace", "visualSpeaker", "visualSpeakerNumber", "visualSpeakerBasis", "linkedSpeakerNumber"}
    cues = []
    for cue in report["segments"]:
        if cue["startMs"] >= limit:
            break
        clean = {key: value for key, value in cue.items() if key not in derived}
        clean["endMs"] = min(cue["endMs"], limit)
        cues.append(clean)
    if not cues:
        return "unavailable"
    analysis = {"segments": cues, "durationMs": limit}
    audio = next(row for row in metadata["streams"] if row["kind"] == "audio")
    try:
        with temporary_directory() as folder:
            work = Path(folder)
            wav, offset, _ = extract_speech(source, work, 0, limit, audio)
            if wav is None:
                return "unavailable"
            analyze_visual_track(source, wav, offset, analysis, people_count=people_count)
            detected = analysis["videoEvidence"]
            timeline = detected.get("timeline") or {}
            if not timeline.get("frames"):
                return "unavailable"
            bind_faces(analysis)
            requested = (legacy.get("personLayer") or {}).get("plan") or {}
            if requested.get("body") or requested.get("gaze"):
                attach_person_evidence(source, work, detected, body=bool(requested.get("body")),
                                       gaze=bool(requested.get("gaze")))
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unavailable"
    report["videoEvidence"] = {**legacy, **detected}
    # Old samples refer to the stripped face IDs, so only keep samples rebuilt on the new track.
    if "personLayer" not in detected:
        report["videoEvidence"].pop("personLayer", None)
    for after, before in zip(cues, report["segments"]):
        after["endMs"] = before["endMs"]
    report["segments"] = cues + report["segments"][len(cues):]
    report["srt"] = render_srt(report["segments"], report["language"])
    if isinstance(report.get("coverage"), dict):
        covered = min(limit, len(timeline["frames"]) * timeline["stepMs"])
        report["coverage"]["face"] = face_coverage(detected, 0, limit)
        if detected.get("editorTimeline", {}).get("shots"):
            report["coverage"]["shots"] = [interval(0, covered, "ok", method="light-asd")]
    return "restored"


def restore_imported_person_layer(studio_root, source, report):
    """Recover requested missing head or gaze on the existing exact-source face timeline."""
    evidence = report.get("videoEvidence") or {}
    if not (evidence.get("timeline") or {}).get("frames"):
        return False
    previous = evidence.get("personLayer") or {}
    plan = previous.get("plan") or {}
    if not (plan.get("body") or plan.get("gaze")):
        return False
    people = [person for row in previous.get("frames", []) for person in row.get("people", [])]
    has_pose = any(person.get("headPose") for person in people)
    has_gaze = any(person.get("gaze") for person in people)
    if has_pose and (has_gaze or not plan.get("gaze")):
        return False
    with temporary_directory() as folder:
        attach_person_evidence(source, Path(folder), evidence, body=bool(plan.get("body")), gaze=bool(plan.get("gaze")))
    rebuilt = evidence.get("personLayer") or {}
    frames = {row["timeMs"]: {person["track"]: person for person in row["people"]}
              for row in previous.get("frames", [])}
    changed = False
    for row in rebuilt.get("frames", []):
        for person in row["people"]:
            directions = {key: person[key] for key in ("headPose", "gaze") if person.get(key)}
            if not directions:
                continue
            peers = frames.setdefault(row["timeMs"], {})
            existing = peers.get(person["track"])
            if existing is None:
                peers[person["track"]] = person
                changed = True
                continue
            for key, value in directions.items():
                if not existing.get(key):
                    existing[key] = value
                    changed = True
    if changed:
        evidence["personLayer"] = {**previous, **rebuilt, "frames": [
            {"timeMs": stamp, "people": list(peers.values())} for stamp, peers in sorted(frames.items())
        ]}
    else:
        evidence["personLayer"] = previous
    return changed


def compact_graph_publisher(metadata, visual, scene, publish):
    """Adapt browser-style deltas into cumulative graph input and measured coverage."""
    if publish is None:
        return None
    report = {}

    def update(phase, detail):
        """Publish only the coverage of layers received so far."""
        report.update(detail)
        staged = attach_coverage(dict(report), metadata, visual and "videoEvidence" in report,
                                 scene and "sceneEvidence" in report)
        publish(phase, staged)

    return update


@analysis_run
def analyze_source(studio_root, source, kind, language, *, visual=True, scene=True, speaker_count=None,
                   object_sampling=DEFAULT_OBJECT_SAMPLING, audio_classifier="auto", revolver=None, publish=None,
                   people_count=None, speakers=True, emit=ignore_progress, quality=True,
                   merge_similar=False, body=False, gaze=False, selected_stages=None, force_stages=None,
                   object_reconciliation=False):
    """Analyze a local audio/video source up to two hours, returning timed facts and coverage."""
    selection = StageSelection(selected_stages, force_stages)
    enabled = selection.enabled if selection.enabled is not None else selection.forced
    visual = visual or "visual" in enabled
    scene = scene or "scene" in enabled
    body = body or ("people" in enabled and not gaze)
    speakers = speakers or "voice" in enabled
    source, studio_root = Path(source), Path(studio_root)
    metadata = probe_source(source)
    if kind != metadata["kind"] or language not in (*CLIP_LANGUAGES, 'auto'):
        raise ValueError("Тип источника или язык анализа не соответствует файлу.")
    if speaker_count is not None and (type(speaker_count) is not int or not 1 <= speaker_count <= 12):
        raise ValueError("Число говорящих должно быть от 1 до 12.")
    visual = visual or body or gaze
    duration = metadata["durationMs"]
    known_silent = False
    if metadata["hasAudio"] and 350 <= duration <= SPEECH_WINDOW_MS:
        try:
            direct = transcribe_media(studio_root, language, source, speakers=speakers, speaker_count=speaker_count,
                                      quality=quality, merge_similar=merge_similar, body=body, gaze=gaze,
                                      visual=kind == "video" and visual, scene=kind == "video" and scene,
                                      object_sampling=object_sampling, audio_classifier=audio_classifier,
                                      clip_languages=True, people_count=people_count, revolver=revolver,
                                      publish=compact_graph_publisher(metadata, visual, scene, publish), emit=emit,
                                      selected_stages=selected_stages, force_stages=force_stages,
                                      object_reconciliation=object_reconciliation)
            result = attach_coverage(direct, metadata, visual, scene)
            measured_video = result.get("videoEvidence") or result.get("sceneEvidence")
            needs_shots = result.get("shotColors") or not result["coverage"]["shots"]
            if kind == "video" and (not selection.selective or measured_video) and needs_shots:
                scene_data = result.get("sceneEvidence")
                if result.get("shotColors") and not scene_data:
                    scene_data = {"shotStartsMs": [row["startMs"] for row in result["shotColors"][1:]]}
                evidence = result.setdefault("videoEvidence", {})
                evidence.setdefault("editorTimeline", {}).setdefault("shots", [])
                if scene_data and "shotStartsMs" in scene_data:
                    # The native scene boundaries and the stored color averages refer to the same shots.
                    editor = evidence["editorTimeline"]
                    editor["shots"] = []
                    result["coverage"]["shots"] = []
                    add_shots(result, source, 0, duration, evidence, scene=scene_data)
                elif not result["coverage"]["shots"]:
                    add_shots(result, source, 0, duration, evidence)
            return result
        except ValueError as exc:
            if str(exc) == NO_SPEECH:
                known_silent = True
            elif kind != "video" or str(exc) != "Длительность звука должна быть от 0,35 секунды до 15 минут.":
                raise
    from voice_studio.graph.media_stage_cache import incremental_stages
    with incremental_stages(studio_root, source, revolver, selection) as cache:
        result = analyze_windows(studio_root, source, metadata, language, visual, scene, speaker_count,
                                 known_silent=known_silent, object_sampling=object_sampling,
                                 audio_classifier=audio_classifier, revolver=cache.revolver, publish=publish,
                                 people_count=people_count, speakers=speakers, emit=emit, quality=quality,
                                 merge_similar=merge_similar, body=body, gaze=gaze,
                                 selection=selection, cache_revision=cache.revision)
        if object_reconciliation or "object_review" in enabled or (selection.selective and "object_review" in cache.manifest["layers"]):
            from voice_studio.graph.media_object_review import review_objects
            review_objects(source, result, cache, emit=emit)
        return selection.attach(result)
