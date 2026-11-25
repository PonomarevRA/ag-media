"""Register the Video as Code workflow on the existing local MCP server."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import logging
from pathlib import Path

from voice_paths import ensure_temporary_root, workspace_temp
import sqlite3
from threading import Lock
from time import perf_counter
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from mcp.types import AudioContent, ImageContent, TextContent
from pydantic import StrictBool, StrictInt
from base64 import b64encode

from voice_studio.core.media_event_bus import MediaEventBus, media_event_scope
from voice_studio.core.mcp_lanes import AsyncWorkLane
from voice_studio.graph.media_graph_events import AsyncGraphPublisher
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.audio.audio_scene import SoundClassifier, select_sound_classifier
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING, ObjectSampling
from voice_studio.graph.media_graph_progress import GraphPublication
from voice_studio.graph.media_graph_store import MediaGraphStore, checked_source, encode, identify
from voice_studio.speech.speaker_profiles import attach_ids, collect_profiles, cue_number
from voice_studio.speech.speech_overlap import mixed_label
from voice_studio.speech.transcription_languages import ClipLanguage

LOGGER = logging.getLogger(__name__)
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_EDITOR_MS = 2 * 60 * 60 * 1000


def import_report(path, source, language):
    """Reuse a Studio JSON export only when its language and source digest match."""
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("Укажи абсолютный путь к JSON анализа до 64 МиБ.")
    try:
        report = json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Не удалось прочитать JSON анализа.") from exc
    if not isinstance(report, dict) or report.get("sourceDigest") != source["sha256"]:
        raise ValueError("Анализ относится к другому исходнику или в нём нет SHA-256.")
    if not isinstance(report.get("segments"), list) or report.get("language") != language:
        raise ValueError("Нужен полный JSON транскрипции для выбранного языка.")
    if report.get("analysisComplete") is False:
        raise ValueError("В JSON сохранён промежуточный результат. Дождись окончания анализа и выгрузи его заново.")
    return report


def recover_profile_context(studio_root, source, media, language, report, *, refresh=False):
    """Reuse exact-source editorial evidence and digital profile IDs without modifying Studio history."""
    from voice_studio.app.transcript_editor import TranscriptEditor

    previous, recovered = [], False
    database = studio_root / "studio.db"
    if media["durationMs"] <= MAX_EDITOR_MS and database.is_file():
        def connect():
            """Open existing context without running Studio migrations from MCP."""
            db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            db.row_factory = sqlite3.Row
            return db

        with closing(connect()) as db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE name IN ('editable_transcripts','speaker_profile_snapshots')")}
            if "speaker_profile_snapshots" in tables:
                row = db.execute(
                    "SELECT profiles FROM speaker_profile_snapshots WHERE source_digest=? AND language=?",
                    (source["sha256"], language),
                ).fetchone()
                previous = json.loads(row[0]) if row else []
        if "editable_transcripts" in tables:
            editor = object.__new__(TranscriptEditor)
            editor.connect = connect
            recovered = bool(editor.recover(language, source["sha256"], report)["decisions"])
    imported = report.get("speakerProfileIndex") or []
    if imported and not recovered and not refresh and not report.get("speakerIdentityGroups"):
        return
    profiles = collect_profiles(report)
    attach_ids(profiles, previous or imported)
    for profile in profiles:
        if not profile["reused"] and not profile.get('voiceProfileId'):
            profile["id"] = uuid5(NAMESPACE_URL, f"voice-studio:{source['sha256']}:{language}:"
                                 f"{profile['number']}:{profile['faceId']}").hex
    report["speakerProfileIndex"] = profiles
    profile_ids = {profile["number"]: profile["id"] for profile in profiles}
    for cue in report["segments"]:
        cue.pop("speakerProfileId", None)
        if not mixed_label(cue, language) and cue_number(cue) in profile_ids:
            cue["speakerProfileId"] = profile_ids[cue_number(cue)]


def is_intermediate_ready(phase, report, duration_ms):
    """Keep one useful checkpoint per layer during a graph build."""
    if phase == "base":
        return True
    if phase == "speech":
        return bool(report.get("segments"))
    if phase == "audio":
        return bool(report.get("audioLayers"))
    if phase == "visual":
        return any((report.get("videoEvidence") or {}).get("timeline", {}).get("frames") or ())
    if phase == "scene" and duration_ms > 60_000:
        frames = (report.get("sceneEvidence") or {}).get("frames", [])
        return any(frame.get("text") or frame.get("objects") for frame in frames)
    return False


def intermediate_publisher(publication, store, source, media, language):
    """Publish one useful snapshot per stage and keep later checkpoints bounded.

    Return True from the callback only after readable storage is advertised.
    Malformed intermediate graphs and recoverable SQLite/I/O errors keep their
    existing warning-and-retry behavior; final publication errors are not hidden.
    """
    if publication is None:
        return None
    completed = set()

    from voice_studio.graph.media_graph_model import build_graph

    def publish(phase, report):
        """Serialize a cumulative report at its readiness boundary."""
        if phase in completed or not is_intermediate_ready(phase, report, media["durationMs"]):
            return
        try:
            graph = build_graph(report, kind=media["kind"], duration_ms=media["durationMs"])
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Could not build a partial media graph", exc_info=True)
            return
        provenance = {"mode": "local-analysis", "recognitionModel": report.get("recognitionModel")}
        try:
            publication.publish(phase, lambda progress: store.publish(
                source, media, graph, language, {**provenance, "progress": progress}))
        except (OSError, sqlite3.Error):
            LOGGER.warning("Could not publish a partial media graph", exc_info=True)
            return
        completed.add(phase)
        return True

    return publish


def graph_event_publisher(event_bus, publication, store, source, media, language):
    """Adapt existing readiness and recoverable partial errors to the job bus.
    """
    publish = intermediate_publisher(publication, store, source, media, language)
    if publish is None:
        return None
    return AsyncGraphPublisher(event_bus, publish,
                               lambda phase, report: is_intermediate_ready(phase, report, media["durationMs"]))


def build_snapshot(store, studio_root, file_path, language, analysis_path, visual, scene, speaker_count,
                   reuse_scene_snapshot_id=None, object_sampling=None, audio_classifier="auto",
                   people_count=True, *, publication=None, selected_stages=None, force_stages=None,
                   object_reconciliation=False):
    """Own one source job's bounded event lifecycle through final publication.

    The captured mode defaults to synchronous behavior. Optional asynchronous
    modes affect the MCP graph workflow; Studio browser callbacks stay unchanged.
    """
    job_id = publication.job_id if publication is not None else uuid4().hex
    bus = MediaEventBus(job_id)
    if publication is not None:
        publication.event_bus = bus
    with media_event_scope(bus):
        return _build_snapshot(store, studio_root, file_path, language, analysis_path, visual, scene, speaker_count,
                               reuse_scene_snapshot_id, object_sampling, audio_classifier, people_count,
                               publication=publication, selected_stages=selected_stages, force_stages=force_stages,
                               object_reconciliation=object_reconciliation, event_bus=bus)


def _build_snapshot(store, studio_root, file_path, language, analysis_path, visual, scene, speaker_count,
                   reuse_scene_snapshot_id=None, object_sampling=None, audio_classifier="auto",
                   people_count=True, *, publication=None, selected_stages=None, force_stages=None,
                   object_reconciliation=False, event_bus=None):
    """Analyze a local original or import evidence within the owned event scope.

    The event bus is source-bound before callbacks; its final barrier projects the
    complete returned report after custom stages and all accepted partial writes.
    """
    from voice_studio.graph.media_graph_ingest import (analyze_source, backfill_dialogue_pauses, probe_source,
                                    restore_imported_person_layer, restore_legacy_visual_track)
    from voice_studio.graph.media_graph_model import build_graph, source_coverage, video_shot_cuts
    from voice_studio.core.storage_cache import use_cache

    from voice_studio.graph.media_stage_selection import StageSelection
    selection = StageSelection(selected_stages, force_stages)
    enabled = selection.enabled if selection.enabled is not None else selection.forced
    visual = visual or "visual" in enabled
    scene = scene or "scene" in enabled
    source = identify(file_path)
    event_bus.bind_source(source["sha256"])
    original = checked_source(source)
    media = probe_source(original)
    if reuse_scene_snapshot_id and not scene:
        raise ValueError("Повтор сцены требует включённой сцены.")
    if object_sampling not in (None, "basic") and (analysis_path or reuse_scene_snapshot_id or not scene):
        raise ValueError("Подробный поиск предметов требует нового анализа сцены.")
    if object_sampling is None:
        object_sampling = DEFAULT_OBJECT_SAMPLING if scene and not reuse_scene_snapshot_id else "basic"
    cached = None
    if reuse_scene_snapshot_id:
        started = perf_counter()
        cached = store.scene_evidence(reuse_scene_snapshot_id, source["sha256"], language)
        scene_read_ms = round((perf_counter() - started) * 1000)
    restored = None
    revolver = None
    restored_people = False
    if analysis_path:
        from voice_studio.speech.person_mentions import annotate_person_mentions

        report = import_report(analysis_path, source, language)
        annotate_person_mentions(report["segments"], language)
        backfill_dialogue_pauses(studio_root, original, media, report)
        if visual:
            studio_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            restored = restore_legacy_visual_track(studio_root, original, media, report, people_count=people_count)
            if restored != "restored":
                restored_people = restore_imported_person_layer(studio_root, original, report)
    else:
        audio_classifier = select_sound_classifier(audio_classifier)
        studio_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Source-scoped directories keep unrelated layer settings out of the outer cache key.
        from voice_studio.graph.media_graph_ingest import SPEECH_WINDOW_MS
        key = ("short-v1/" + source["sha256"] if 350 <= media["durationMs"] <= SPEECH_WINDOW_MS
               else "bounded-v2/" + source["sha256"])
        ensure_temporary_root()
        cache = workspace_temp(studio_root) / "analysis-checkpoints"
        revolver = AnalysisRevolver(cache / key)
        with use_cache(cache, revolver.directory):
            report = analyze_source(studio_root, original, media["kind"], language,
                                    visual=visual, scene=scene and cached is None, speaker_count=speaker_count,
                                    object_sampling=object_sampling, audio_classifier=audio_classifier, revolver=revolver,
                                    people_count=people_count, selected_stages=selected_stages, force_stages=force_stages,
                                    object_reconciliation=object_reconciliation,
                                    emit=publication.activity if publication is not None else (lambda event: None),
                                    publish=graph_event_publisher(event_bus, publication, store, source, media, language))
    if cached is not None:
        # Legacy exports infer speech/face/shot coverage from their measured layers. Materialise
        # that shared inference before adding a scene entry, which would otherwise disable it.
        previous_coverage = source_coverage(report, media["durationMs"])
        cuts = video_shot_cuts(report, media["durationMs"])
        if cuts:
            report.setdefault("videoEvidence", {}).setdefault("editorTimeline", {})["shots"] = cuts
        report["sceneEvidence"], coverage = cached
        shot_colors = report["sceneEvidence"].pop("shotColors", [])
        if not report.get("shotColors"):
            report["shotColors"] = shot_colors
        report["coverage"] = {**previous_coverage, "scene": coverage}
    editor = (report.get("videoEvidence") or {}).get("editorTimeline") or {}
    if cached is not None or (analysis_path and "personMotion" not in editor):
        from voice_studio.speech.media_transcript import attach_person_motion

        report.setdefault("durationMs", media["durationMs"])
        attach_person_motion(report, (shot["startMs"] for shot in report.get("shotColors", [])))
    report.setdefault("mediaStreams", media.get("streams", []))
    recover_profile_context(studio_root, source, media, language, report, refresh=restored == "restored" or restored_people)
    if analysis_path and (object_reconciliation or (selection.enabled is not None and "object_review" in selection.enabled)):
        from voice_studio.graph.media_stage_cache import incremental_stages
        from voice_studio.graph.media_object_review import review_objects
        with incremental_stages(studio_root, original, selection=selection) as cache:
            review_objects(original, report, cache, emit=publication.activity if publication else (lambda event: None))
            selection.attach(report)
    provenance = {"mode": "studio-json" if analysis_path else "local-analysis",
                  "recognitionModel": report.get("recognitionModel")}
    for field in ('modProvenance', 'modStages', 'modPipeline'):
        if field in report:
            provenance[field] = report[field]
    # Keep replay conflicts visible to agents after the imported report becomes a graph.
    if report.get("analysisStages"):
        provenance["analysisStages"] = report["analysisStages"]
    if report.get("objectReconciliationEvidence"):
        provenance["objectReconciliationEvidence"] = report["objectReconciliationEvidence"]
    if report.get("speakerIdentityConflicts"):
        provenance["speakerIdentityConflicts"] = report["speakerIdentityConflicts"]
    if revolver and any(revolver.counts.values()):
        provenance["checkpoints"] = revolver.counts
    if restored:
        provenance["visualTrackRecovery"] = restored
    if restored_people:
        provenance["personLayerRecovery"] = "restored"
    if reuse_scene_snapshot_id:
        provenance.update(sceneSnapshotId=reuse_scene_snapshot_id, sceneReadMs=scene_read_ms)
    elif elapsed := (report.get("sceneEvidence") or {}).get("elapsedMs"):
        provenance["sceneStageMs"] = elapsed
    def complete():
        """Project the final mod-enriched report after all accepted partial events.

        Graph construction and immutable storage share the consumer ownership
        boundary. A final error propagates without replacing the last snapshot.
        """
        graph = build_graph(report, kind=media["kind"], duration_ms=media["durationMs"])
        provenance["eventBus"] = event_bus.metrics()
        provenance["eventBus"]["capture"] = "before-final-commit"
        if publication is None:
            return store.publish(source, media, graph, language, provenance)
        return publication.publish("complete", lambda progress: store.publish(
            source, media, graph, language, {**provenance, "progress": progress}))

    return event_bus.finish(complete)


class GraphBuildQueue:
    """Keep long media work off the MCP request thread and bound concurrent analysis."""

    def __init__(self, *, result_key="snapshot", label="графов"):
        self.result_key = result_key
        self.label = label
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="media-jobs")
        self.lock = Lock()
        self.jobs = {}

    def submit(self, operation, *arguments, progressive=False, **options):
        """Start or queue one graph build, returning a pollable local job identifier."""
        with self.lock:
            pending = sum(not future.done() for future, _ in self.jobs.values())
            if pending >= 3:
                raise ValueError(f"Очередь построения {self.label} заполнена.")
            for key in list(self.jobs):
                if len(self.jobs) < 64:
                    break
                if self.jobs[key][0].done():
                    del self.jobs[key]
            job = uuid4().hex
            publication = GraphPublication(job) if progressive else None
            if publication is None:
                future = self.worker.submit(operation, *arguments, **options)
            else:
                future = self.worker.submit(operation, *arguments, publication=publication, **options)
            self.jobs[job] = future, publication
        return {"jobId": job, "status": "queued"}

    def status(self, job):
        """Read the build outcome without blocking a client request."""
        with self.lock:
            item = self.jobs.get(job)
        if item is None:
            raise ValueError("Задание не найдено в этом сеансе MCP.")
        future, publication = item
        latest = publication.state() if publication is not None else {}
        if publication is not None and hasattr(publication, "event_bus"):
            latest["eventBus"] = publication.event_bus.metrics()
        if not future.done():
            active = future.running() or future.done()
            return {"jobId": job, "status": "running" if active else "queued", **latest}
        latest = publication.state() if publication is not None else {}
        if publication is not None and hasattr(publication, "event_bus"):
            latest["eventBus"] = publication.event_bus.metrics()
        try:
            result = future.result()
        except ValueError as exc:
            return {"jobId": job, "status": "failed", "error": str(exc)[:500], **latest}
        except Exception:
            LOGGER.exception("Media job %s failed", job)
            return {"jobId": job, "status": "failed", **latest,
                    "error": "Не удалось выполнить задание. Подробности в журнале MCP."}
        return {"jobId": job, "status": "done", **latest, self.result_key: result}

    def close(self):
        """Retire this session's pending jobs; running native work keeps its own deadline."""
        self.worker.shutdown(wait=False, cancel_futures=True)



def interval_frame_samples(details, frame_count):
    """
    Choose half-open source samples with bounded ASR context around each request.

    details is the bounded interval brief; frame_count is the validated 1–8 sample
    budget. Returns (samples, context metadata) without changing the brief. Context
    retains the whole cue and requested time; it is not aligned to actual pixels.
    """
    start, end = details["interval"]["startMs"], details["interval"]["endMs"]
    samples, context = [], []
    for index in range(frame_count):
        stamp = start + index * (end - start) // frame_count
        sample = {"atMs": stamp}
        match = next((cue for cue in details["sections"]["speech"]["items"]
                      if cue["startMs"] <= stamp < cue["endMs"]), None)
        annotation = {"requestedMs": stamp, "cueId": None, "textTruncated": False,
                      "status": "no-returned-cue-at-request"}
        if match is not None:
            text = match["label"][:120]
            sample["caption"] = "ASR near request (whole cue): " + text
            annotation.update(cueId=match["id"], textTruncated=match["textTruncated"] or len(match["label"]) > 120,
                              status="quoted-whole-cue")
        samples.append(sample)
        context.append(annotation)
    return samples, {"basis": "whole-ASR-cue-overlapping-request; not word-aligned or aligned to actual frame",
                     "samples": context}


def register_media_graph_tools(mcp, root, studio_root):
    """Register graph tools and return their queues/IO lanes to the owning MCP lifespan."""
    store = MediaGraphStore(root)
    queue = GraphBuildQueue()
    read_lane = AsyncWorkLane('graph-read', workers=2, capacity=4)
    render_lane = AsyncWorkLane('graph-render', capacity=2)

    @mcp.tool()
    def media_graph_build(file_path: str, language: ClipLanguage = "ru",
                          analysis_path: str | None = None, visual: bool = True, scene: bool = True,
                          speaker_count: int | None = None,
                          reuse_scene_snapshot_id: str | None = None,
                          object_sampling: ObjectSampling | None = None,
                          progressive: bool = False, audio_classifier: SoundClassifier = "auto",
                          people_count: bool = True, selected_stages: list[str] | None = None,
                          force_stages: list[str] | None = None, object_reconciliation: bool = False) -> dict[str, Any]:
        """Queue local media analysis. An exact-source prior scene can be reused by snapshot ID.
        New scenes use small-detail by default. Choose basic for baseline only, every-third
        for dense checks, or shot-tiles for regional checks on selected shots.
        Set audio_classifier="zipformer" to tag music and background sounds locally with Zipformer.
        Set people_count=false to skip estimated person counts in a new visual analysis.
        Pass analysis_path to import a complete Studio JSON export instead. With progressive=true,
        media_graph_job exposes the most recent readable snapshot even while analysis is running.
        Pair analysis_path with reuse_scene_snapshot_id to retain measured scene/UI/object layers
        from an exact-source snapshot when the imported report has only newer speech/face layers.
        selected_stages accepts speech,audio,visual,people,scene,voice,object_review;
        omit it for defaults or pass [] to retain completed cached work only.
        force_stages reruns named stages; valid unrelated layers remain cached.
        object_reconciliation enables a bounded final review of existing object views.
        Pin each search or paginated read to its returned snapshotId."""
        if speaker_count is not None and not 1 <= speaker_count <= 12:
            raise ValueError("Число говорящих должно быть от 1 до 12.")
        return queue.submit(build_snapshot, store, Path(studio_root), file_path, language,
                            analysis_path, visual, scene, speaker_count, reuse_scene_snapshot_id, object_sampling,
                            audio_classifier, people_count, progressive=progressive,
                            selected_stages=selected_stages, force_stages=force_stages,
                            object_reconciliation=object_reconciliation)

    @mcp.tool()
    def media_graph_job(job_id: str) -> dict[str, Any]:
        """Read graph build progress and the latest immutable snapshot when available."""
        return queue.status(job_id)

    @mcp.tool()
    @read_lane.offload
    def media_graph_list(limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List saved local video and audio graph snapshots, newest first."""
        return store.list(limit, offset)

    @mcp.tool()
    @read_lane.offload
    def media_graph_overview(snapshot_id: str) -> dict[str, Any]:
        """Read the exact local source, track IDs, coverage and counts by node kind and category."""
        return store.overview(snapshot_id, include_source=True)

    @mcp.tool()
    @read_lane.offload
    def media_graph_search(snapshot_id: str, query: str, kind: str | None = None,
                           limit: int = 25, after: int = 0) -> dict[str, Any]:
        """Find a literal phrase in shot descriptions, speech, screen text or object labels.
        Use kind="ui" to search screen regions, controls, UI text, diagram connections and interface profiles.
        Use nextAfter to retrieve the next page for the same search."""
        return store.search(snapshot_id, query, kind, limit, after)

    @mcp.tool()
    @read_lane.offload
    def media_graph_timeline(snapshot_id: str, start_ms: int, end_ms: int, kind: str | None = None,
                             limit: int = 30, after: int = 0) -> dict[str, Any]:
        """Read nodes overlapping a window of up to five minutes on the source clock.
        Use kind="ui" to browse all sampled UI evidence and interface profiles in that window."""
        return store.timeline(snapshot_id, start_ms, end_ms, kind, limit, after)

    @mcp.tool()
    @render_lane.offload
    def media_graph_interval(snapshot_id: str, start_ms: StrictInt, end_ms: StrictInt,
                             detail: str = "compact", frames: StrictBool = True,
                             frame_count: StrictInt = 4,
                             wire: Literal["compact", "expanded"] = "compact") -> list[TextContent | ImageContent]:
        """
        Read a <=60s half-open source interval as bounded facts, coverage gaps and review hints.
        Pin snapshot_id. compact returns up to 8 rows/section; full up to 24, with explicit
        omitted counts and textTruncated flags. Speech is whole-cue quoted evidence, not a
        semantic summary. Frames default to four quarterpoints (max eight); JSON reports
        requested and actual frame timestamps. Frozen JSON remains readable if the original
        is missing/changed; frame status then explains why no grid is returned. No inference.
        Expand selected IDs with media_graph_node or use timeline/search for omitted facts.

        snapshot_id pins one immutable graph; start_ms/end_ms are strict integer
        source milliseconds within its duration. detail selects compact/full;
        frames is a strict boolean, and frame_count is a strict integer 1–8.
        wire selects lossless media-interval/2 compact tables/dictionary or the original
        media-interval/1 expanded JSON; it does not change facts, quotas or images.
        Each compact packet includes named columns and its decoding legend.
        Returns JSON TextContent plus optional PNG ImageContent. Invalid input
        raises ValueError; original rendering failures are recorded in frames.status.
        """
        from voice_studio.graph.media_graph_interval import summarize_interval
        from voice_studio.graph.media_graph_interval_wire import pack_interval

        if wire not in ("compact", "expanded"):
            raise ValueError("wire должен быть compact или expanded.")

        if type(frames) is not bool or type(frame_count) is not int or not 1 <= frame_count <= 8:
            raise ValueError("frames должен быть bool, frame_count — целое число от 1 до 8.")
        details = summarize_interval(store, snapshot_id, start_ms, end_ms, detail)
        grid = None
        frame_started = perf_counter()
        if not frames:
            frame_metadata = {"status": "disabled", "requested": 0, "returned": 0}
        elif details["kind"] != "video":
            frame_metadata = {"status": "not-video", "requested": frame_count, "returned": 0}
        elif details["source"]["status"] != "available":
            frame_metadata = {"status": "source-" + details["source"]["status"],
                              "requested": frame_count, "returned": 0}
        else:
            from voice_studio.graph.media_graph_interval_frames import render_interval_grid

            samples, context = interval_frame_samples(details, frame_count)
            try:
                with store.database(snapshot_id) as db:
                    source_record = store.manifest(db)["source"]
                source = checked_source(source_record)
                grid, frame_metadata = render_interval_grid(source, samples, end_ms=end_ms)
                frame_metadata["context"] = context
                # A concurrent replacement invalidates the entire binary attachment.
                checked_source(source_record)
            except (ValueError, OSError, RuntimeError) as exc:
                LOGGER.warning("Interval grid unavailable: %s", type(exc).__name__)
                grid = None
                frame_metadata = {"status": "unavailable", "requested": frame_count, "returned": 0,
                                  "reason": "Original frame extraction or source verification failed."}
        details["frames"] = frame_metadata
        details["timingMs"]["rendering"] = round((perf_counter() - frame_started) * 1000, 3)
        packet = pack_interval(details, dictionary=True) if wire == "compact" else details
        result = [TextContent(type="text", text=encode(packet))]
        if grid is not None:
            result.append(ImageContent(type="image", mimeType="image/png", data=b64encode(grid).decode("ascii")))
        return result

    @mcp.tool()
    @read_lane.offload
    def media_graph_node(snapshot_id: str, node_id: str, limit: int = 30,
                         offset: int = 0) -> dict[str, Any]:
        """Read a stable node and its paginated references.
        Follow contained_in to its measured frame parent, or incoming contained_in to its children.
        An alias_of edge records an alternate detector label for the same display."""
        return store.node(snapshot_id, node_id, limit, offset)

    @mcp.tool()
    @render_lane.offload
    def media_graph_frame(snapshot_id: str, at_ms: int | None = None, node_id: str | None = None,
                          technical: bool = False, max_side: int = 1024,
                          crop: bool = False) -> list[TextContent | ImageContent]:
        """Return a PNG from the exact original video. Choose timestamp or timed node ID.
        JSON includes the original frame palette and measured lower-third caption contrast.
        Technical mode adds linked and timed boxes, shot label and caption when available.
        crop=True extracts the selected node’s box from the original before scaling."""
        from voice_studio.graph.media_graph_frame import render_frame

        if (at_ms is None) == (node_id is None):
            raise ValueError("Укажи один таймкод или один ID узла.")
        metadata, at, context = store.frame_context(snapshot_id, at_ms, node_id)
        region = None
        if crop:
            if technical or node_id is None or not context.get("targetObservationId"):
                raise ValueError("Для кропа нужен ID узла с рамкой. Техническая разметка включается отдельно.")
            region = context["boxes"][0]["box"]
        image, palette = render_frame(checked_source(metadata["source"]), at, max_side=max_side,
                                      context=context if technical else None, include_palette=True,
                                      **({"region": region} if region is not None else {}))
        checked_source(metadata["source"])
        details = {"snapshotId": snapshot_id, "timeMs": at, "nodeId": node_id,
                   "technical": technical, "palette": palette, "shotColor": context.get("shotColor")}
        if technical:
            details["overlay"] = context
        if region is not None:
            details["cropBox"] = region
        return [TextContent(type="text", text=encode(details)),
                ImageContent(type="image", mimeType="image/png", data=b64encode(image).decode("ascii"))]


    @mcp.tool()
    @render_lane.offload
    def media_graph_audio(snapshot_id: str, at_ms: int | None = None, node_id: str | None = None,
                          duration_ms: int | None = None,
                          track_id: str | None = None) -> list[TextContent | AudioContent]:
        """Return a WAV excerpt from an audio/video graph by timestamp or timed node, up to ten seconds."""
        from voice_studio.graph.media_graph_frame import render_audio

        if (at_ms is None) == (node_id is None):
            raise ValueError("Укажи один таймкод или один ID узла.")
        metadata, at, length, stream = store.audio_context(snapshot_id, at_ms, node_id, duration_ms, track_id)
        wav = render_audio(checked_source(metadata["source"]), at, length, stream)
        checked_source(metadata["source"])
        details = {"snapshotId": snapshot_id, "startMs": at, "endMs": at + length,
                   "nodeId": node_id, "streamIndex": stream}
        return [TextContent(type="text", text=encode(details)),
                AudioContent(type="audio", mimeType="audio/wav", data=b64encode(wav).decode("ascii"))]


    @mcp.tool()
    @render_lane.offload
    def media_graph_object_placement(snapshot_id: str, node_id: str, asset_path: str,
                                     at_ms: int | None = None, height_ratio: float = 1.0,
                                     height_px: int | None = None, anchor_px: list[int] | None = None,
                                     preview: bool = True, max_side: int = 1024) -> list[TextContent | ImageContent]:
        """Fit a local transparent PNG to a measured object by height and bottom-center.
        Use an object observation or a track and an exact sample time. For a clipped box,
        supply the fields listed in required. Preview composites the asset over source pixels."""
        from voice_studio.graph.media_graph_ingest import probe_source
        from voice_studio.graph.media_graph_placement import place_object

        metadata, stamp, observation_id, box = store.placement_context(snapshot_id, node_id, at_ms)
        source = checked_source(metadata["source"])
        frame_size = metadata.get("frameSize")
        if not frame_size:
            video = next(stream for stream in probe_source(source)["streams"] if stream["kind"] == "video")
            if not video.get("width") or not video.get("height"):
                raise ValueError("Не удалось измерить исходный кадр.")
            frame_size = [video["width"], video["height"]]
        plan, image = place_object(source, stamp, box, frame_size, asset_path, height_ratio=height_ratio,
                                   height_px=height_px, anchor_px=anchor_px, preview=preview, max_side=max_side)
        checked_source(metadata["source"])
        details = {"snapshotId": snapshot_id, "timeMs": stamp, "observationId": observation_id, **plan}
        result = [TextContent(type="text", text=encode(details))]
        if image is not None:
            result.append(ImageContent(type="image", mimeType="image/png", data=b64encode(image).decode("ascii")))
        return result

    return queue, read_lane, render_lane
