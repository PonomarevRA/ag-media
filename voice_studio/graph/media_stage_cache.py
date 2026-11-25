"""Dependency-keyed checkpoints for the common short-media pipeline.

Only settings and completed raw stage outputs are shared. Editorial profile bindings
are reapplied later by the graph publisher. Code/model changes conservatively invalidate
all stages; layer options and environment changes invalidate only their dependants.
"""

from voice_studio.mod_runtime import layer_fingerprint

from contextlib import contextmanager
from base64 import b64decode, b64encode
from copy import deepcopy
from hashlib import sha256
import os
from time import perf_counter

from voice_paths import ensure_temporary_root, workspace_temp
from voice_studio.core.optional_runtime import runtime_path
from voice_studio.core.storage_cache import use_cache
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver, analysis_fingerprint, file_identity, result_fingerprint
from voice_studio.graph.media_stage_selection import StageSelection

LAYER_ENV = {
    "object_review": (),
    "voice": ("VOICE_SPEAKER_EMBEDDING_MODE", "VOICE_SPEAKER_EMBEDDING_TIMEOUT_SECONDS"),
    "speaker-review": ("VOICE_SPEAKER_EMBEDDING_MODE", "VOICE_SPEAKER_EMBEDDING_TIMEOUT_SECONDS"),
    "speech": ("VOICE_ASR_PARALLEL",),
    "audio": ("VOICE_AUDIO_CLASSIFIER", "VOICE_PITCH_BACKEND", "VOICE_SOUND_STREAM_CHUNK_MS"),
    "visual": ("VOICE_YUNET_SAMPLING", "VOICE_PERSON_FACE_PILOT", "VOICE_CLOTHING_PROFILE"),
    "people": ("VOICE_CLOTHING_PROFILE",),
    "scene": ("VOICE_SCENE_OCR", "VOICE_SCENE_OCR_WORKERS", "VOICE_VIDEO_PARALLEL"),
}


def source_digest(source):
    """Stream the source without retaining a second copy of its bytes in RAM."""
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IncrementalStages:
    """Checkpoint successful JSON reports and expose actual read/compute timings."""

    def __init__(self, revolver, revision, selection=None):
        self.revolver = revolver
        self.revision = revision
        self.selection = selection or StageSelection()
        manifest = revolver.load("layer-pointers", required=("layers",))
        if manifest:
            # Metadata reads are not completed worker/checkpoint reuse.
            revolver.counts["reused"] -= 1
        self.manifest = manifest if manifest and isinstance(manifest.get("layers"), dict) else {"layers": {}}

    def run(self, name, settings, upstream, operation, valid, *, publish=None, emit=None, fallback=None,
            selection_layer=None):
        """Load a validated completed result or run the identical production operation.

        The upstream value must contain every measured input consumed by this stage.
        Failed or unavailable results stay usable by callers but never become cache hits.
        An internal stage may share another layer's selection and force policy.
        """
        started = perf_counter()
        environment = {key: os.environ.get(key) for key in LAYER_ENV[name]}
        environment["mods"] = layer_fingerprint(name)
        if name in {"visual", "people"}:
            # Installing MediaPipe also enables hair and appearance sampling in the face pass.
            marker = runtime_path("mediapipe") / ".studio-ready"
            try:
                environment["mediapipe"] = file_identity(marker)
            except FileNotFoundError:
                pass
        key = result_fingerprint([self.revision, settings, environment, upstream])
        requested_key, requested_settings = key, settings
        pointer = self.manifest["layers"].get(name)
        if not isinstance(pointer, dict) or not isinstance(pointer.get("key"), str) or len(pointer["key"]) != 64 or any(character not in "0123456789abcdef" for character in pointer["key"]):
            pointer = None
        layer = selection_layer or name
        active = self.selection.runs(layer)
        input_hash = result_fingerprint([self.revision, environment, upstream])
        dependency_only = self.selection.requested is not None and layer not in self.selection.requested
        if (not active or dependency_only) and pointer and pointer.get("inputHash") == input_hash:
            key = pointer["key"]
            settings = pointer.get("settings", settings)
        checkpoint = self.revolver.child(name).child(key)
        forced = layer in self.selection.forced
        saved = None if forced else checkpoint.load("complete", required=("result", "digest"))
        reused = False
        if saved and isinstance(saved["result"], dict):
            try:
                reused = saved["digest"] == result_fingerprint(saved["result"]) and valid(saved["result"])
            except (TypeError, ValueError):
                reused = False
        if saved and not reused:
            self.revolver.counts["reused"] -= 1
        if reused:
            result = saved["result"]
        elif not active:
            state = "invalidated" if pointer else "pending"
            if layer == name:
                self.selection.mark(name, state)
            if emit is not None:
                emit({"cacheLayer": name, "state": state, "selected": False})
            return deepcopy(fallback) if fallback is not None else {}
        else:
            # A dependency pointer can be corrupt: its old settings describe only
            # a valid reused result, never the current operation closure.
            key, settings = requested_key, requested_settings
            checkpoint = self.revolver.child(name).child(key)
            if emit is not None:
                # Report a cache miss independently of the real worker stage events.
                emit({"cacheLayer": name, "cacheHit": False})
            result = operation()
            if valid(result):
                checkpoint.save("complete", {"result": result, "digest": result_fingerprint(result)})
        complete = valid(result)
        if layer == name:
            self.selection.mark(name, "reused" if reused else ("done" if complete else "failed"))
        if complete:
            self.manifest["layers"][name] = {"key": key, "inputHash": input_hash, "settings": settings}
            self.revolver.save("layer-pointers", self.manifest)
        elapsed = round((perf_counter() - started) * 1000)
        self.revolver.counts.setdefault("layers", {})[name] = {
            "reused": reused, "elapsedMs": elapsed, "key": key, "resultHash": result_fingerprint(result),
        }
        if reused and emit is not None:
            replay_progress(name, settings, result, emit, elapsed)
        if reused and publish is not None:
            publish(result)
        return result

    def scene(self, view, sampling, work, operation, publish_mosaic, emit):
        """Retain mosaic bytes with the scene and republish generation-owned URLs on hits."""
        def measure():
            mosaics = []

            def capture(index, path, markdown=None):
                """Keep the image before the worker removes its temporary directory."""
                mosaics.append({"index": index, "png": b64encode(path.read_bytes()).decode("ascii")})
                return None

            scene = operation(capture if publish_mosaic else None)
            # A cached report must not point at an expired earlier UI generation.
            scene = deepcopy(scene)
            for page in scene.get("interfaceScrolls", []):
                page.pop("previewUrl", None)
            return {"scene": scene, "mosaics": mosaics}

        saved = self.run("scene", {"sampling": sampling, "mosaic": publish_mosaic is not None}, scene_inputs(view),
                         measure, lambda report: usable_scene(report.get("scene") or {}), emit=emit)
        scene = saved.get("scene")
        if scene is None:
            return None
        if publish_mosaic:
            for image in saved["mosaics"]:
                index = image["index"]
                path = work / f"cached-scroll-{index}.png"
                path.write_bytes(b64decode(image["png"], validate=True))
                pages = scene.get("interfaceScrolls", [])
                if not 0 <= index < len(pages):
                    continue
                page = pages[index]
                from voice_studio.vision.video_scene import mosaic_markdown

                markdown = mosaic_markdown(page)
                url = publish_mosaic(index, path, markdown) if markdown else publish_mosaic(index, path)
                if url:
                    page["previewUrl"] = url
        return scene


@contextmanager
def incremental_stages(root, source, revolver=None, selection=None):
    """Share protected external scratch storage between file/UI and MCP callers."""
    if revolver is None:
        ensure_temporary_root()
        directory = workspace_temp(root) / "analysis-checkpoints" / "short-v1" / source_digest(source)
        revolver = AnalysisRevolver(directory)
    # Unknown resource/environment switches are conservative shared dependencies.
    layer_keys = {key for keys in LAYER_ENV.values() for key in keys}
    common_keys = sorted(key for key in os.environ if key.startswith(("VOICE_", "AG_PRIVACY_")) and key not in layer_keys)
    revision = analysis_fingerprint("short-v1", {}, environment_names=common_keys)
    if revolver.directory is None:
        yield IncrementalStages(revolver, revision, selection)
    else:
        group = workspace_temp(root) / "analysis-checkpoints"
        with use_cache(group, revolver.directory):
            yield IncrementalStages(revolver, revision, selection)


def usable_visual(result):
    """Reject interrupted/failed workers; tolerate only a sub-frame terminal tail."""
    evidence = result.get("videoEvidence") or {}
    stages = evidence.get("stages") or {}
    if not stages or any(value in ("failed", "busy", "unavailable", "running") for value in stages.values()):
        return False
    if (result.get("speakerReview") or {}).get("status") == "failed":
        return False
    if stages.get("light-asd") == "partial":
        windows = (evidence.get("activityCoverage") or {}).get("windows", [])
        if not windows:
            return False
        for index, window in enumerate(windows):
            if window.get("status") == "ok":
                continue
            span = window.get("endMs", 0) - window.get("startMs", 0)
            if index != len(windows) - 1 or window.get("status") != "no-video" or not 0 < span < 40:
                return False
    return True


def usable_scene(result):
    """Cache complete detection and OCR, including crops recovered by the fallback."""
    stages = result.get("stages", {})
    ocr_recovered = stages.get("apple-vision") == "partial" and stages.get("tesseract") == "ok"
    workers_complete = all(status == "ok" or (name == "apple-vision" and ocr_recovered)
                           for name, status in stages.items())
    return (result.get("status") == "ok" and result.get("interfaceStatus", "ok") == "ok"
            and workers_complete
            and (result.get("frameChanges") or {}).get("status", "ok") == "ok"
            and (result.get("frameChangeObjects") or {}).get("status", "ok") == "ok")


def usable_speech(result):
    """Reuse completed recovery attempts, preserving explicit uncertainty in their content."""
    recovery = result.get("recognitionRecovery")
    return (isinstance(result.get("segments"), list) and not result.get("speakerWarning")
            and (recovery is None or recovery.get("status") == "done"))


def scene_inputs(evidence):
    """Return the face timeline and shot cuts actually consumed by review_scene.

    Figure/gaze worker timings, portraits and speech bindings are not scene-detector
    inputs. Keep their unrelated changes out of the scene key in both execution modes.
    """
    evidence = evidence or {}
    return {"timeline": evidence.get("timeline") or {},
            "shots": (evidence.get("editorTimeline") or {}).get("shots", [])}


def replay_progress(name, settings, result, emit, elapsed):
    """Replay actual UI stage IDs; a cache read never pretends FFmpeg decoded PCM."""
    names = {"object_review": ["object-review"], "audio": ["audio-layers"], "scene": ["scene"], "people": ["person-layer"], "voice": ["voice-profiles"],
             "speaker-review": ["speaker-review"],
             "visual": ["light-asd", "yunet"],
             "speech": ["whisper", "dialogue-pauses"]}[name]
    if name == "speech":
        emit({"durationMs": result.get("durationMs", 0)})
        if settings.get("speakers"):
            names.append("diarization")
        if result.get("recognitionRecovery"):
            names.append("whisper-recovery")
    if name == "visual" and not settings.get("separateSpeakerReview"):
        names.append("speaker-review")
    for stage_name in names:
        state = "done"
        reason = None
        if name in ("visual", "speaker-review"):
            measured = (result.get("videoEvidence") or {}).get("stages", {})
            if stage_name == "speaker-review":
                review = result.get("speakerReview") or {}
                if not review or review.get("status") == "skipped":
                    state = "skipped"
                    reason = review.get("reason") or "Причина пропуска отсутствует в сохранённом анализе"
            elif measured.get(stage_name) in ("no-phrases", "no-video"):
                state = "skipped"
            elif measured.get(stage_name) == "partial":
                state = "partial"
        emit({"stage": stage_name, "state": state, "reused": True, "cacheHit": True,
              "cacheLayer": name, "elapsedMs": elapsed, **({"reason": reason} if reason else {})})
