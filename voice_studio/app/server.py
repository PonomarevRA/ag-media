"""Serve the single-user ag-media-studio on loopback."""

import argparse
from contextlib import contextmanager, nullcontext
from hashlib import file_digest
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from threading import Thread
import webbrowser

from voice_paths import ensure_temporary_root, temporary_directory, workspace_temp
from voice_studio.core.workspace_lock import workspace_lock
from urllib.parse import parse_qs, unquote, urlsplit

from voice_studio.core.analysis_stages import ignore_progress, serialized_progress, stage
from voice_studio.synthesis.engines import MAX_AUDIO, ROOT
from voice_studio.audio.audio_processing import process_wave
from voice_studio.speech.asr_context_graph import VERSION as ASR_CONTEXT_VERSION, corrections
from voice_studio.speech.recognition import recognize
from voice_studio.speech.transcription_languages import CLIP_LANGUAGES, STUDIO_LANGUAGES
from voice_studio.speech.media_transcript import FORMATS, MAX_MEDIA_BYTES, compatibility_preview, transcribe_media, transcribe_stream
from voice_studio.app.http_transport import LoopbackRequestHandler
from voice_studio.app.media_pipeline import transcribe_uploaded_media
from voice_studio.core.media_limits import MAX_UPLOAD_BYTES
from voice_studio.vision.video_export import MAX_METADATA, export_video
from voice_studio.synthesis.video_dub import build_video_dub, suggest_english
from voice_studio.vision.person_layers import export_clothing_masks
from voice_studio.app.live_capture import EXTENSION_ORIGIN, MAX_BYTES as MAX_LIVE_BYTES, LiveBusy, LiveCapture
from voice_studio.audio.music_sample import music_sample
from voice_studio.app.scroll_previews import PREVIEW_PATH, ScrollPreviews
from voice_studio.app.studio import Studio, require_text
from voice_studio.core.storage_cache import cache_lock, mark_cache_used
from voice_studio.app.studio_storage import StudioStorage
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING, OBJECT_SAMPLING
from voice_studio.app.transcript_editor import MAX_DOCUMENT_BYTES
from voice_studio.prosody.revolver import review_prosody
from voice_studio.mcp.web_graph import read_graph
from voice_studio.app.wiki_assets import wiki_asset
from voice_studio.mod_api import mod_operation
from voice_studio.mods import ModError
from voice_studio.app.ui_bridge import UIBridge, handle_bridge
from voice_studio.app.static_assets import STATIC

TRANSCRIPT_ROUTES = frozenset({
    "/api/transcript-materials", "/api/transcript-material", "/api/transcript-material-delete",
    "/api/transcript-cue", "/api/video-export",
})

class StudioServer(ThreadingHTTPServer):
    """Accept a browser's initial burst of module and stylesheet requests."""

    request_queue_size = 64

    def __init__(self, *args, **kwargs):
        """Give native and browser servers the same independent, ephemeral page broker."""
        self.ui_bridge = UIBridge()
        super().__init__(*args, **kwargs)

    def server_close(self):
        """Release long polls before waiting for ordinary HTTP request threads."""
        self.ui_bridge.close()
        super().server_close()


class Handler(LoopbackRequestHandler):
    """Expose same-origin studio operations and authorized local audio."""

    def language(self, query, *, transcript=False):
        """Resolve the language for the workspace or for a clip transcript route."""
        language = query.get("lang", ["ru"])[0]
        supported = CLIP_LANGUAGES if transcript else STUDIO_LANGUAGES
        if language not in supported:
            if transcript:
                raise ValueError("Выбери язык транскрипции из списка.")
            raise ValueError("Выберите русский или английский язык.")
        return language

    def transcription_language(self, query, fallback, *, clip=False):
        """Resolve recognition independently of the recording and synthesis workspace."""
        language = query.get("speech_lang", [fallback])[0]
        supported = CLIP_LANGUAGES if clip else STUDIO_LANGUAGES
        if language != "auto" and language not in supported:
            if clip:
                raise ValueError("Выбери язык транскрипции из списка.")
            raise ValueError("Выбери автоопределение, русский или английский язык.")
        return language

    def do_HEAD(self):
        """Check availability for older browser tabs without transferring the PNG."""
        with self.request_scope() as allowed:
            if not allowed:
                return
            preview = PREVIEW_PATH.fullmatch(urlsplit(self.path).path)
            path = self.server.scroll_previews.path(preview[1], int(preview[2])) if preview else None
            self.send_response(200 if path and path.is_file() else 404)
            self.end_headers()

    def do_GET(self):
        """Serve the studio, model status or a known audio result."""
        with self.request_scope() as allowed:
            if not allowed:
                return
            url = urlsplit(self.path)
            if url.path in {"/wiki", "/wiki/"}:
                locale = parse_qs(url.query).get("ui", ["ru"])[0]
                self.send_response(302)
                self.send_header("Location", "/wiki/Home.html" if locale == "en" else "/wiki/Home-ru.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif url.path.startswith("/wiki/"):
                locale = parse_qs(url.query).get("ui", ["ru"])[0]
                asset = wiki_asset(url.path, locale)
                if asset:
                    self.file(*asset)
                else:
                    self.reply(404, {"error": "Страница документации не найдена."})
            elif url.path in STATIC:
                name, mime = STATIC[url.path]
                self.file(ROOT / "web" / name, mime)
            elif preview := PREVIEW_PATH.fullmatch(url.path):
                image = self.server.scroll_previews.read(preview[1], int(preview[2]))
                if image is None:
                    self.reply(404, {"error": "Временная склейка недоступна."})
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(image)))
                    self.end_headers()
                    self.wfile.write(image)
            elif url.path == "/api/mods/pipeline":
                self.reply(200, mod_operation("pipeline"))
            elif url.path == "/api/mods/contracts":
                self.reply(200, mod_operation("contracts"))
            elif url.path == "/api/mods":
                self.reply(200, mod_operation("list"))
            elif url.path == "/api/storage":
                self.reply(200, self.server.storage.inventory())
            elif url.path == "/api/state":
                self.reply(200, self.server.studio.state(self.language(parse_qs(url.query))))
            elif url.path in {"/api/transcript-materials", "/api/transcript-material"}:
                query = parse_qs(url.query, keep_blank_values=True)
                language = self.language(query, transcript=True)
                editor = self.server.studio.transcripts
                if url.path == "/api/transcript-materials":
                    limit = min(100, max(1, int(query.get("limit", ["100"])[0])))
                    offset = min(2**63 - 1, max(0, int(query.get("offset", ["0"])[0])))
                    items = editor.list(language, limit + 1, offset)
                    self.reply(200, {"items": items[:limit],
                                     "nextOffset": offset + limit if len(items) > limit else None})
                else:
                    self.reply(200, editor.get(language, query.get("id", [None])[0]))
            elif url.path == "/api/voice-job":
                query = parse_qs(url.query, keep_blank_values=True)
                key = query.get("id", [None])[0]
                jobs = self.server.studio.rows(
                    "SELECT id, model, status, seconds, error FROM jobs "
                    "WHERE id=? AND language=? AND demo=? AND kind='synthesis'",
                    (key, self.language(query), int(self.server.studio.demo)))
                if jobs:
                    from voice_studio.builtin_mods.synthesis import synthesis_provenance
                    self.reply(200, {**jobs[0], **synthesis_provenance(self.server.studio.media, key)})
                else:
                    self.reply(404, {"error": "Задание синтеза не найдено."})
            elif re.fullmatch(r"/media/[a-f0-9]{32}\.wav", url.path):
                key = url.path.rsplit("/", 1)[-1][:-4]
                path = self.media_path(key)
                if path:
                    self.file(path, "audio/wav", seekable=True)
                else:
                    self.reply(404, {"error": "Аудио не найдено."})
            else:
                self.reply(404, {"error": "Страница не найдена."})

    def media_path(self, key, language=None):
        """Resolve a visible recording or a finished synthesis in this workspace."""
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key):
            return None
        studio = self.server.studio
        scope, parameters = (" AND language=?", (language,)) if language else ("", ())
        recording = studio.rows("SELECT 1 FROM recordings WHERE id=?" + scope, (key, *parameters))
        job = studio.rows("SELECT 1 FROM jobs WHERE id=? AND status='done' AND demo=? AND kind!='training'" + scope,
                          (key, int(studio.demo), *parameters))
        path = studio.media / f"{key}.wav"
        return path if (recording or job) and path.is_file() else None

    def process_audio(self, language, data):
        """Stream an isolated processed copy of an existing visible recording."""
        source = self.media_path(data.get("id"), language)
        if source is None:
            raise ValueError("Выбранная дорожка недоступна.")
        reference = None
        if data.get("auto") is True:
            reference = self.media_path(data.get("reference"), language)
            if reference is None or reference == source:
                raise ValueError("Выбери другую дорожку как исходник для авто EQ.")
        with temporary_directory() as folder:
            output = Path(folder) / "processed.wav"
            settings = process_wave(source, reference, output, data)
            self.file(output, "audio/wav", metadata=settings)

    @contextmanager
    def uploaded_media(self, limit=MAX_MEDIA_BYTES, *, overlay=False):
        """Stream one bounded upload into a temporary workspace."""
        remaining = int(self.headers.get("Content-Length", "0"))
        if not 0 < remaining <= limit + (MAX_METADATA + 4 if overlay else 0):
            raise ValueError(f"Размер файла должен быть не больше {limit // (1024 * 1024)} МиБ.")
        studio = self.server.studio
        self.connection.settimeout(30)
        layer = None
        if overlay:
            size = int.from_bytes(self.rfile.read(4), "big")
            if not 0 < size <= MAX_METADATA or not 0 < remaining - 4 - size <= limit:
                raise ValueError("Некорректный размер технического слоя.")
            layer = json.loads(self.rfile.read(size))
            if not isinstance(layer, dict):
                raise ValueError("Некорректные данные технического слоя.")
            remaining -= 4 + size
        ensure_temporary_root()
        group = workspace_temp(studio.root) / "temporary-media"
        group.mkdir(mode=0o700, parents=True, exist_ok=True)
        with cache_lock(group), TemporaryDirectory(prefix="upload-", dir=group) as folder:
            directory = Path(folder)
            mark_cache_used(directory)
            path = directory / "media"
            with path.open("wb") as target:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Файл загружен не полностью.")
                    target.write(chunk)
                    remaining -= len(chunk)
            self.connection.settimeout(None)
            yield (path, layer) if overlay else path

    def receive_media(self, language, limit=MAX_UPLOAD_BYTES, allow_empty=False, *,
                      progress=False, context=False, **settings):
        """Transcribe an upload through the JSON or streaming client contract."""
        with self.uploaded_media(limit) as path:
            def analyze(emit=ignore_progress):
                studio = self.server.studio
                digest = None

                def source_digest():
                    """Hash this upload once, when the first editable report needs its identity."""
                    nonlocal digest
                    if digest is None:
                        with path.open("rb") as source:
                            digest = file_digest(source, "sha256").hexdigest()
                    return digest

                def publish(phase, report):
                    """Send a detached report view for the current layer over NDJSON."""
                    if phase == "base":
                        if context:
                            report = {**report, "sourceDigest": source_digest()}
                        emit({"partial": {"phase": "base", "report": report}})
                    else:
                        emit({"partial": {"phase": "detail", "layer": phase, "report": report}})

                previews = self.server.scroll_previews.analysis() if settings.get("scene") else nullcontext()
                with previews as publish_mosaic:
                    result = transcribe_uploaded_media(studio.root, language, path, allow_empty=allow_empty, emit=emit,
                                              publish=publish if progress else None, publish_mosaic=publish_mosaic,
                                              **settings)
                    if context:
                        with stage(emit, "speaker-context") as progress_stage:
                            digest = source_digest()
                            detected = result["language"]
                            graph = studio.transcripts.recover(detected, digest, result)
                            try:
                                studio.transcripts.remember_profiles(detected, digest, result)
                            except sqlite3.Error:
                                result["speakerProfileWarning"] = "Не удалось закрепить профили в локальной истории."
                            if not graph["decisions"] and not result.get("speakerProfileIndex"):
                                progress_stage["state"] = "skipped"
                    return result
            if progress:
                self.stream_analysis(analyze)
            else:
                self.reply(200, analyze())

    def stream_analysis(self, analyze):
        """Flush stage records as NDJSON and finish with one result or error record."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def write_record(event):
            """Flush one complete NDJSON record from any analysis stage."""
            payload = json.dumps(event, ensure_ascii=False).encode()
            self.wfile.write(payload + b"\n")
            self.wfile.flush()

        notify = serialized_progress(write_record)
        notify({"stage": "upload", "state": "done"})
        try:
            result = analyze(emit=notify)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (ValueError, ModError) as exc:
            notify({"error": str(exc)[:300]})
        except Exception as exc:
            print(f"Analysis failure: {exc}", flush=True)
            notify({"error": "Ошибка анализа видео. Подробности в терминале."})
        else:
            notify({"result": result})

    def check_live_origin(self, *, preflight=False):
        """Apply a separate, loopback-only credential policy for the extension route."""
        origin = self.headers.get("Origin", "")
        local = self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"
        if not local or (origin and not EXTENSION_ORIGIN.fullmatch(origin)) or (preflight and not origin):
            self.reply(403, {"error": "Запрос анализа вкладки не разрешён."})
            return False
        self._live_cors_origin = origin
        return True

    def do_OPTIONS(self):
        """Allow the custom authorization header on the extension route only."""
        if urlsplit(self.path).path != "/api/live-chunk":
            self.reply(404, {"error": "Операция не найдена."})
            return
        if not self.check_live_origin(preflight=True):
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Voice-Live-Token")
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def receive_live(self, query):
        """Analyze one authorized tab fragment without persisting source data."""
        if not self.check_live_origin():
            return
        service = self.server.live_capture
        if not service.authorized(self.headers.get("X-Voice-Live-Token", "")):
            self.reply(403, {"error": "Неверный ключ доступа расширения."})
            return
        if self.headers.get_content_type() != "video/webm":
            self.reply(415, {"error": "Ожидался фрагмент WebM."})
            return
        try:
            language = self.language(query)
            start_ms = int(query.get("start_ms", [""])[0])
            session = query.get("session", [""])[0]
            with self.uploaded_media(MAX_LIVE_BYTES) as path:
                self.reply(200, service.process(path, session, start_ms, language))
        except (BrokenPipeError, ConnectionResetError):
            return
        except LiveBusy as exc:
            self.reply(429, {"error": str(exc)})
        except (ValueError, TimeoutError, ModError) as exc:
            self.reply(400, {"error": str(exc)[:300]})

    def do_POST(self):
        """Save a recording or perform a bounded workspace operation."""
        url = urlsplit(self.path)
        if url.path.startswith("/api/ui-bridge/"):
            handle_bridge(self, url.path.rsplit("/", 1)[1])
            return
        if url.path == "/api/live-chunk":
            self.receive_live(parse_qs(url.query))
            return
        with self.request_scope(write=True) as allowed:
            if not allowed:
                return
            query = parse_qs(url.query, keep_blank_values=True)
            language = self.language(query, transcript=url.path in TRANSCRIPT_ROUTES)
            studio = self.server.studio
            if url.path == "/api/analyze-image":
                from voice_studio.app.image_analysis import analyze_image, image_options
                from voice_studio.vision.image_input import MAX_SCREENSHOT_BYTES
                options = image_options(query)
                with self.uploaded_media(MAX_SCREENSHOT_BYTES) as path:
                    def analyze(emit=ignore_progress):
                        """Use the local vision-only path; never call the speech pipeline for a still image."""
                        return analyze_image(path, options, emit=emit)
                    if query.get("progress") == ["1"]:
                        self.stream_analysis(analyze)
                    else:
                        self.reply(200, analyze())
                return
            if url.path.startswith('/api/mods/'):
                try:
                    data = json.loads(self.body(256 * 1024))
                    self.reply(200, mod_operation(url.path.rsplit('/', 1)[1], data))
                except (ModError, ValueError, TypeError) as exc:
                    self.reply(400, {'error': str(exc)[:500]})
                return
            if url.path == '/api/dialogue-audit':
                from voice_studio.app.dialogue_audit_service import audit_media, atomic_json
                with self.uploaded_media(MAX_UPLOAD_BYTES, overlay=True) as (path, report):
                    report_path = path.with_name('audit-report.json')
                    atomic_json(report_path, report)
                    def operation(emit=ignore_progress):
                        """Stream a separate check while preserving the uploaded speech material."""
                        return audit_media(studio.root, path, report_path, rerun=query.get('rerun') == ['1'],
                                           transient_source=True, emit=emit)
                    if query.get('progress') == ['1']:
                        self.stream_analysis(operation)
                    else:
                        self.reply(200, operation())
                return
            if url.path in {"/api/transcribe-media", "/api/transcribe-capture"}:
                clip = url.path == "/api/transcribe-media"
                count = query.get("speaker_count", [None])[0]
                object_sampling = query.get("object_sampling", [DEFAULT_OBJECT_SAMPLING])[0]
                if object_sampling not in OBJECT_SAMPLING:
                    raise ValueError("Неизвестная частота поиска предметов.")
                people_count = query.get("people_count")
                if people_count is not None and people_count not in (["0"], ["1"]):
                    raise ValueError("Для подсчёта людей выбери 0 или 1.")
                from voice_studio.graph.media_stage_selection import query_stage_names
                self.receive_media(self.transcription_language(query, language, clip=clip),
                                   selected_stages=query_stage_names(query, "selected_stages"),
                                   force_stages=query_stage_names(query, "force_stages"),
                                   object_reconciliation=query.get("object_reconciliation") == ["1"],
                                   clip_languages=clip, speakers=query.get("speakers") == ["1"],
                                   speaker_count=int(count) if count is not None else None,
                                   merge_similar=query.get("merge_similar") == ["1"],
                                   quality=query.get("quality") == ["1"], visual=query.get("visual") == ["1"],
                                   body=query.get("body") == ["1"], gaze=query.get("gaze") == ["1"],
                                   scene=query.get("scene") == ["1"], object_sampling=object_sampling,
                                   people_count=None if people_count is None else people_count == ["1"],
                                   audio_classifier=query.get("audio_classifier", ["auto"])[0],
                                   progress=query.get("progress") == ["1"], context=True)
                return
            if url.path == "/api/music-sample":
                start_ms = int(query.get("start_ms", ["0"])[0])
                with self.uploaded_media() as path:
                    self.reply(200, music_sample(path, path.parent, start_ms, FORMATS))
                return
            if url.path == "/api/video-preview":
                with self.uploaded_media() as path:
                    preview = path.with_suffix(".webm")
                    compatibility_preview(path, preview)
                    self.file(preview, "video/webm")
                return
            if url.path == "/api/video-export":
                with self.uploaded_media(overlay=True) as (path, layer):
                    output = path.with_name("technical.mp4")
                    export_video(path, layer, output, language)
                    self.file(output, "video/mp4", attachment="video-technical.mp4")
                return
            if url.path == "/api/video-dub":
                with self.uploaded_media(overlay=True) as (path, dub):
                    output = path.with_name("dub.zip")
                    build_video_dub(path, dub, path.parent, output)
                    self.file(output, "application/zip", attachment="english-dub.zip")
                return
            if url.path == "/api/video-dub-translate":
                with self.uploaded_media(overlay=True) as (path, dub):
                    self.reply(200, {"translations": suggest_english(path, dub, path.parent)})
                return
            if url.path == "/api/person-masks":
                with self.uploaded_media(overlay=True) as (path, spec):
                    with path.open("rb") as source:
                        if file_digest(source, "sha256").hexdigest() != spec.get("sourceDigest"):
                            raise ValueError("Файл отличается от исходника трека лиц.")
                    output = path.with_name("clothing-masks.zip")
                    export_clothing_masks(path, spec, output, FORMATS)
                    self.file(output, "application/zip", attachment="clothing-masks.zip")
                return
            if url.path == "/api/transcribe-chunk":
                self.receive_media(self.transcription_language(query, language), limit=4 * 1024 * 1024, allow_empty=True)
                return
            if url.path == "/api/transcribe":
                result = recognize(studio.root, language, self.body(MAX_AUDIO))
                result["corrections"] = corrections(result["text"], language)
                result["contextVersion"] = ASR_CONTEXT_VERSION
                self.reply(200, result)
                return
            if url.path == "/api/record":
                if query.get("consent") != ["1"]:
                    raise ValueError("Подтвердите право использовать записанный голос.")
                transcript = unquote(self.headers.get("X-Voice-Transcript", ""))
                prompt = query.get("prompt", [None])[0]
                key = studio.record(language, transcript, self.body(MAX_AUDIO), int(prompt) if prompt else None,
                                    evaluation_only=query.get("evaluation") == ["1"])
                self.reply(201, {"id": key})
                return
            limit = MAX_DOCUMENT_BYTES if url.path == "/api/transcript-material" else 64 * 1024
            data = json.loads(self.body(limit))
            if not isinstance(data, dict):
                raise ValueError("Ожидался JSON-объект.")
            if url.path == "/api/storage-delete":
                self.reply(200, self.server.storage.remove(data.get("id"), data.get("revision")))
            elif url.path == "/api/graph-read":
                self.reply(200, read_graph(studio.root / "media-graphs", data))
            elif url.path == "/api/transcribe-stream":
                self.reply(200, transcribe_stream(studio.root, self.transcription_language(query, language),
                                                     data.get("url"), data.get("seconds"),
                                                     speakers=data.get("speakers") is True,
                                                     speaker_count=data.get("speaker_count"),
                                                     audio_classifier=data.get("audio_classifier", "auto")))
            elif url.path == "/api/transcript-material":
                self.reply(201, studio.transcripts.create(
                    language, data.get("title"), data.get("segments"), data.get("sourceDigest"),
                    data.get("dialoguePauses"),
                ))
            elif url.path == "/api/transcript-material-delete":
                studio.transcripts.delete(language, data.get("id"), data.get("revision"))
                self.reply(200, {"id": data["id"]})
            elif url.path == "/api/transcript-cue":
                self.reply(200, studio.transcripts.patch(language, data))
            elif url.path == "/api/process":
                self.process_audio(language, data)
            elif url.path == "/api/word-graph":
                self.reply(200, studio.inspect_words(language, data.get("id")))
            elif url.path == "/api/fragment-score":
                self.reply(200, studio.fragment_score(language, data.get("id"), data.get("index")))
            elif url.path == "/api/retry-words":
                ids = studio.retry_words(language, data.get("id"), data.get("index"), data.get("count"),
                                         data.get("context", "short"))
                self.reply(202, {"ids": ids})
            elif url.path == "/api/prosody":
                self.reply(200, review_prosody(require_text(data.get("text"), 500), language))
            elif url.path == "/api/auto-synthesize":
                self.reply(202, {"id": studio.automatic.start(language, data)})
            elif url.path == "/api/auto-stop":
                studio.automatic.stop(language, data.get("id"))
                self.reply(200, {"ok": True})
            elif url.path == "/api/transcribe-recording":
                path = self.media_path(data.get("id"), language)
                if path is None:
                    raise ValueError("Выбранная запись недоступна.")
                self.reply(200, transcribe_media(studio.root, self.transcription_language(query, language), path,
                                                 speakers=data.get("speakers") is True,
                                                 speaker_count=data.get("speaker_count"),
                                                 audio_classifier=data.get("audio_classifier", "auto")))
            elif url.path == "/api/compare-lapshi":
                self.reply(202, {"ids": studio.compare_lapshi(language, data)})
            elif url.path == "/api/synthesize":
                self.reply(202, {"ids": studio.synthesize(language, data)})
            elif url.path == "/api/voice-sources":
                state = studio.state(language)
                recordings = [{key: item[key] for key in ("id", "seconds", "created")}
                              for item in state["recordings"] if not item["evaluation_only"]]
                self.reply(200, {"demo": state["demo"], "engines": state["engines"], "recordings": recordings})
            elif url.path == "/api/profile":
                self.reply(201, {"version": studio.save_profile(language, data.get("refs"))})
            elif url.path == "/api/rate":
                studio.rate(language, data.get("id"), data.get("score"))
                self.reply(200, {"ok": True})
            elif url.path == "/api/rate-prosody-pair":
                studio.rate_prosody_pair(language, data.get("id"), data.get("choice"))
                self.reply(200, {"ok": True})
            elif url.path == "/api/assemble-blocks":
                self.reply(201, {"id": studio.assemble_blocks(language, data.get("blocks"))})
            elif url.path == "/api/assemble":
                self.reply(201, {"id": studio.assemble(language, data.get("ids"))})
            elif url.path == "/api/train":
                self.reply(202, {"id": studio.train(language, data.get("model"), data.get("profile"))})
            elif url.path == "/api/delete":
                kind, key = data.get("kind"), data.get("id")
                valid = type(key) is int and key > 0 if kind == "profile" else (
                    isinstance(key, str) and re.fullmatch(r"[a-f0-9]{32}", key))
                if not valid:
                    raise ValueError("Некорректный идентификатор.")
                studio.delete(language, kind, key)
                self.reply(200, {"ok": True})
            elif url.path == "/api/export":
                with temporary_directory() as folder:
                    archive = Path(folder) / "voice-corpus.zip"
                    studio.export(language, data.get("refs"), archive)
                    self.file(archive, "application/zip", "voice-corpus.zip")
            else:
                self.reply(404, {"error": "Операция не найдена."})


def main():
    """Start a real or isolated demo workspace on the chosen loopback port."""
    parser = argparse.ArgumentParser(prog="ag-media-studio", description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--demo", action="store_true", help="Generate labeled test tones in a separate workspace.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local Studio URL.")
    options = parser.parse_args()
    root = options.data or ROOT / "data" / ("demo" if options.demo else "studio")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with workspace_lock(root):
        with StudioServer(("127.0.0.1", options.port), Handler) as server:
            server.studio = Studio(root, options.demo)
            server.scroll_previews = ScrollPreviews(workspace_temp(root) / "scroll-previews")
            server.live_capture = LiveCapture(root)
            production_data = ROOT / "data"
            project_data = production_data if not options.demo and root.resolve() == (production_data / "studio").resolve() else root
            server.storage = StudioStorage(root, project_data)
            stop_storage = server.storage.start()
            address = f"http://127.0.0.1:{server.server_port}"
            print(f"ag-media-studio: {address}", flush=True)
            if options.open_browser:
                Thread(target=webbrowser.open, args=(address,), daemon=True).start()
            try:
                server.serve_forever()
            finally:
                stop_storage.set()
                server.live_capture.close()


if __name__ == "__main__":
    main()
