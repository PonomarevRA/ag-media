"""ag-media-studio MCP over stdio, backed by the running local studio."""

import argparse
from contextlib import asynccontextmanager, ExitStack
import http.client
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from voice_paths import PROJECT_DATA
from typing import Any, Literal
from urllib.parse import urlencode
import urllib.request
import urllib.error

from mcp.server.fastmcp import FastMCP

from voice_studio.mcp.agent_audio_mcp import register_audio_tools
from voice_studio.mcp.media_graph_mcp import MAX_REPORT_BYTES, register_media_graph_tools
from voice_studio.speech.media_transcript import MAX_MEDIA_BYTES
from voice_studio.core.media_limits import MAX_UPLOAD_BYTES
from voice_studio.core.mcp_lanes import AsyncWorkLane
from voice_studio.speech.transcription_languages import ClipLanguage, ClipLanguageMode, STUDIO_LANGUAGES
from voice_studio.vision.scene_sampling import DEFAULT_OBJECT_SAMPLING, ObjectSampling
from voice_studio.audio.audio_scene import SoundClassifier
from voice_studio.vision.video_export import MAX_METADATA
from voice_studio.mcp.voice_catalog import register_catalog
from voice_studio.mod_api import register_mod_tools
from voice_studio.mcp.screen_mcp import register_screen_tools
from voice_studio.mcp.dialogue_audit_mcp import register_dialogue_tools
from voice_studio.mcp.ui_bridge_mcp import register_ui_bridge_tools

Language = Literal["ru", "en"]
TranscriptionLanguage = Literal["auto", "ru", "en"]
Engine = Literal["qwen", "chatterbox", "f5", "coqui", "fish"]


class StudioClient:
    """Use one configured loopback port and the studio's existing queue and corpus."""

    def __init__(self, port, output):
        self.port = port
        self.output = output.resolve()
        self.origin = f"http://127.0.0.1:{port}"
        self.headers = {"Origin": self.origin, "X-Voice-Studio": "1"}

    def request(self, route, language, data=None, *, timeout=90, **query):
        """Read JSON from the trusted studio origin."""
        url = self.origin + "/api/" + route + "?" + urlencode({"lang": language, **query})
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers={**self.headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(json.load(exc)["error"]) from exc
        except OSError as exc:
            raise ValueError("Запусти локальную ag-media-studio на настроенном порту.") from exc

    def transcript(self, path, language, speakers=False, speaker_count=None, *, quality=False, visual=False,
                   editable=False, body=False, gaze=False, scene=False, object_sampling=None,
                   audio_classifier="auto", people_count=True, selected_stages=None, force_stages=None,
                   object_reconciliation=False):
        """Stream an explicitly supplied local file and save subtitles to the configured output."""
        source = Path(path).expanduser()
        if not source.is_absolute() or not source.is_file():
            raise ValueError("Укажи абсолютный путь к локальному файлу.")
        size = source.stat().st_size
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise ValueError("Размер файла должен быть не больше 2 ГиБ.")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1020)
        try:
            with source.open("rb") as stream:
                query = {"lang": language if language in STUDIO_LANGUAGES else "ru", "speech_lang": language or "default",
                         "speakers": int(speakers or speaker_count is not None),
                         "quality": int(quality), "visual": int(visual or body or gaze),
                         "body": int(body), "gaze": int(gaze), "scene": int(scene or object_sampling not in (None, "basic")),
                         "object_sampling": object_sampling or DEFAULT_OBJECT_SAMPLING, "audio_classifier": audio_classifier,
                         "people_count": int(people_count)}
                from voice_studio.graph.media_stage_selection import StageSelection
                StageSelection(selected_stages, force_stages)
                for key, names in (("selected_stages", selected_stages), ("force_stages", force_stages)):
                    if names is not None:
                        query[key] = ",".join(names)
                query["object_reconciliation"] = int(object_reconciliation)
                if speaker_count is not None:
                    query["speaker_count"] = speaker_count
                connection.request("POST", "/api/transcribe-media?" + urlencode(query), body=stream,
                                   headers={**self.headers, "Content-Length": str(size),
                                            "Content-Type": "application/octet-stream"})
                response = connection.getresponse()
                result = json.load(response)
                if response.status != 200:
                    raise ValueError(result["error"])
        except OSError as exc:
            raise ValueError("Не удалось передать файл в локальную ag-media-studio.") from exc
        finally:
            connection.close()
        if editable:
            document = self.request("transcript-material", result["language"], {"title": source.name, "segments": result["segments"],
                                                                       "sourceDigest": result.get("sourceDigest"),
                                                                       "dialoguePauses": result.get("dialoguePauses")})
            result.update(materialId=document["id"], revision=document["revision"])
        evidence = result.get("videoEvidence", {})
        if evidence.get("timeline") or result.get("sceneEvidence") or result.get('speechTurns') or result.get('voiceProfiles') or result.get('modProvenance'):
            self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output, prefix="media-analysis-",
                                    suffix=".json", delete=False) as document:
                json.dump(result, document, ensure_ascii=False, separators=(",", ":"))
                result["analysisPath"] = document.name
        return self.save_subtitles(result)

    def clothing_masks(self, file_path, analysis_path, face_id, start_ms, end_ms, language):
        """Stream an exact source and its reviewed face track, then save the returned ZIP."""
        source, analysis = [Path(value).expanduser() for value in (file_path, analysis_path)]
        if not all(path.is_absolute() and path.is_file() for path in (source, analysis)):
            raise ValueError("Укажи абсолютные пути к видео и JSON анализа.")
        size = source.stat().st_size
        if not 0 < size <= MAX_MEDIA_BYTES or not 0 < analysis.stat().st_size <= MAX_REPORT_BYTES:
            raise ValueError("Размер видео или JSON анализа превышает лимит.")
        data = json.loads(analysis.read_text(encoding="utf-8"))
        metadata = json.dumps(clothing_request(data, face_id, start_ms, end_ms)).encode()
        if len(metadata) > MAX_METADATA:
            raise ValueError("JSON трека превышает лимит.")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=150)
        try:
            connection.putrequest("POST", "/api/person-masks?" + urlencode({"lang": language}))
            for name, value in self.headers.items():
                connection.putheader(name, value)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(4 + len(metadata) + size))
            connection.endheaders(len(metadata).to_bytes(4, "big") + metadata)
            with source.open("rb") as stream:
                connection.send(stream)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError(json.load(response)["error"])
                self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=self.output, prefix="clothing-", suffix=".zip", delete=False) as archive:
                    try:
                        shutil.copyfileobj(response, archive)
                    except (OSError, http.client.HTTPException):
                        Path(archive.name).unlink(missing_ok=True)
                        raise
                    return {"zipPath": archive.name, "faceId": face_id, "startMs": start_ms, "endMs": end_ms}
        finally:
            connection.close()

    def save_subtitles(self, result):
        """Persist the SRT from a completed transcript under the configured output directory."""
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output,
                                prefix="transcript-", suffix=".srt", delete=False) as subtitles:
            subtitles.write(result.pop("srt"))
            result["srtPath"] = subtitles.name
        return result


def clothing_request(data, face_id, start_ms, end_ms):
    """Prepare the bounded mask input from a full report or an older track sidecar."""
    timeline = (data.get("videoEvidence") or {}).get("timeline") or data.get("timeline")
    if not timeline:
        raise ValueError("В JSON нет покадрового трека лиц.")
    profiles = data.get("speakerProfileIndex") or data.get("speakerProfiles", [])
    profile = next((entry for entry in profiles if entry.get("faceId") == face_id), {})
    return {"faceId": face_id, "startMs": start_ms, "endMs": end_ms, "sourceDigest": data.get("sourceDigest"),
            "timeline": timeline, "profileId": profile.get("id"), "speakerNumber": profile.get("number")}


def build_server(studio, *, studio_root=None, graph_root=None):
    """Register canonical media/UI tools with their input and output schemas."""
    resources = []

    @asynccontextmanager
    async def lifespan(server):
        """Close only this server's queues and IO lanes when its stdio session ends."""
        with ExitStack() as cleanup:
            for resource in resources:
                cleanup.callback(resource.close)
            yield

    mcp = FastMCP("ag-media-studio", lifespan=lifespan)
    transcription_lane = AsyncWorkLane('transcription-http', capacity=2)
    http_lane = AsyncWorkLane('studio-http', workers=2, capacity=4)
    masks_lane = AsyncWorkLane('studio-masks', capacity=2)
    resources.extend((transcription_lane, http_lane, masks_lane))
    studio_root = studio_root or PROJECT_DATA / "studio"
    resources.extend(register_media_graph_tools(mcp, graph_root or studio_root / "media-graphs", studio_root))
    resources.append(register_audio_tools(mcp, studio.output))
    register_screen_tools(mcp)
    resources.extend(register_dialogue_tools(mcp, studio_root))
    register_catalog(mcp)
    resources.append(register_mod_tools(mcp))
    register_ui_bridge_tools(mcp, studio)

    @mcp.tool()
    def browser_ml_capabilities() -> dict[str, Any]:
        """Discover browser-local WebNN tools for agents. Run the probe through WebMCP in the open Studio tab.
        Browser device availability cannot be inferred from the stdio Python process.
        """
        return {
            "scope": "browser", "browserCapability": "unmeasured",
            "entrypoint": studio.origin + "/tools",
            "tools": [
                {"name": "browser_ml_probe", "arguments": {"powerPreference": "default"}},
                {"name": "browser_ml_cosine",
                 "arguments": {"left": [1, 0], "right": [1, 0], "runs": 3}},
            ],
            "execution": "Compile and validate float32 mul/reduceSum in a bounded local worker; JS fallback is explicit.",
            "comparison": "Use vectors from the same embedding model/version. Cosine is not an identity probability.",
            "modelRouting": "Existing Python models keep their own runtimes. WebNN model compatibility needs per-model validation.",
            "sources": [
                "https://webmachinelearning.github.io/webnn-samples-intro/",
                "https://webmachinelearning.github.io/webnn/",
                "https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html",
            ],
        }

    @mcp.tool()
    @http_lane.offload
    def speech_sources(language: Language = "ru") -> dict[str, Any]:
        """List local engines and voice profiles with measurements and stable version IDs for generation and dubbing."""
        return studio.request("voice-sources", language, {})

    @mcp.tool()
    @http_lane.offload
    def speech_profile_check(reference_ids: list[str], language: Language = "ru",
                             save: bool = False, name: str | None = None) -> dict[str, Any]:
        """Compare up to eight authorized local takes and optionally save their verified order."""
        return studio.request("voice-profile-check", language, {"refs": reference_ids, "save": save, "name": name})

    @mcp.tool()
    def speech_reference_script(language: Language = "ru") -> dict[str, Any]:
        """Return spoken prompts, separate delivery directions and the held-out control phrase."""
        from voice_studio.prosody.reference_script import reference_script

        return reference_script(language)

    @mcp.tool()
    @http_lane.offload
    def speech_generate(text: str, reference_ids: list[str] | None = None, engine: Engine = "qwen",
                       language: Language = "ru", checkpoint_id: str | None = None,
                       profile_version: int = 0, smooth_sequence: bool = False) -> dict[str, Any]:
        """Queue local speech from authorized samples or a saved profile. With smooth_sequence, add a matched Qwen variant for two to four sentences."""
        checkpoints = {engine: checkpoint_id} if checkpoint_id else {}
        return studio.request("synthesize", language, {"text": text, "models": [engine],
                              "refs": reference_ids, "profile": profile_version, "checkpoints": checkpoints,
                              "prosody": {"sequence": smooth_sequence}})

    @mcp.tool()
    @http_lane.offload
    def speech_job(job_id: str, language: Language = "ru", download: bool = False) -> dict[str, Any]:
        """Read synthesis status and optionally save the completed WAV in MCP outputs."""
        job = studio.request("voice-job", language, id=job_id)
        if job["status"] == "done":
            job["wavUrl"] = f"{studio.origin}/media/{job['id']}.wav"
            if download:
                studio.output.mkdir(mode=0o700, parents=True, exist_ok=True)
                with urllib.request.urlopen(job["wavUrl"], timeout=30) as response:
                    with NamedTemporaryFile(dir=studio.output, prefix="voice-", suffix=".wav", delete=False) as output:
                        shutil.copyfileobj(response, output)
                        job["wavPath"] = output.name
        return job

    @mcp.tool()
    @http_lane.offload
    def media_pipeline_status(run_id: str | None = None) -> dict[str, Any]:
        """Read Studio upload stages, elapsed milliseconds, retry groups and stage errors. Omit run_id for the latest Studio run; for graph builds use media_graph_job."""
        options = {"id": run_id} if run_id is not None else {}
        return studio.request("analysis-status", "ru", **options)

    @mcp.tool()
    async def media_transcribe(file_path: str, language: ClipLanguageMode | None = None,
                         speakers: bool = False, speaker_count: int | None = None,
                         quality: bool = False, visual: bool = False, editable: bool = False,
                         body: bool = False, gaze: bool = False, scene: bool = False,
                         object_sampling: ObjectSampling | None = None,
                         audio_classifier: SoundClassifier = "auto", people_count: bool = True,
                         selected_stages: list[str] | None = None, force_stages: list[str] | None = None,
                         object_reconciliation: bool = False) -> dict[str, Any]:
        """Transcribe a local file and save SRT plus an importable analysis JSON.
        Omit language to use the active Whisper mod's setting. Pass a language code such as bn or en
        to override it for this file, or pass "auto" to explicitly detect the language.
        Set scene=True for the default small-detail pass.
        Choose basic for baseline only, every-third for dense checks, or shot-tiles for regional checks on selected shots.
        Set audio_classifier="zipformer" to tag music and background sounds with Zipformer.
        With visual=true, people_count estimates visible people by default. Set it false to skip this pass.
        selected_stages accepts speech,audio,visual,people,scene,voice,object_review.
        Omit it for defaults or pass [] to restore cached layers without new work.
        force_stages reruns named workers; other valid layers are preserved.
        object_reconciliation reviews existing object views without rerunning scene detection.
        """
        return await transcription_lane.run(studio.transcript, file_path, language, speakers, speaker_count, quality=quality,
                                 visual=visual, editable=editable, body=body, gaze=gaze, scene=scene,
                                 object_sampling=object_sampling, audio_classifier=audio_classifier,
                                 people_count=people_count, selected_stages=selected_stages, force_stages=force_stages,
                                 object_reconciliation=object_reconciliation)

    @mcp.tool()
    @masks_lane.offload
    def video_clothing_masks(file_path: str, analysis_path: str, face_id: int,
                             start_ms: int, end_ms: int, language: Language = "ru") -> dict[str, Any]:
        """Export timed clothing PNG masks for a face over at most 6 seconds. Use analysisPath from media_transcribe."""
        return studio.clothing_masks(file_path, analysis_path, face_id, start_ms, end_ms, language)

    @mcp.tool()
    @http_lane.offload
    def transcript_materials(language: ClipLanguage = "ru") -> dict[str, Any]:
        """List locally saved transcripts that the user or an agent can edit."""
        return studio.request("transcript-materials", language)

    @mcp.tool()
    @http_lane.offload
    def transcript_get(material_id: str, language: ClipLanguage = "ru") -> dict[str, Any]:
        """Read timed cues, named speakers, visible tracks, evidence and the current revision."""
        return studio.request("transcript-material", language, id=material_id)

    @mcp.tool()
    @http_lane.offload
    def transcript_edit(material_id: str, revision: int, start_ms: int,
                              speaker_number: int | None = None, clear_speaker_number: bool = False,
                              speaker_name: str | None = None, context_start_ms: int | None = None,
                              text: str | None = None, language: ClipLanguage = "ru") -> dict[str, Any]:
        """Edit one cue in an editable transcript; use a numbered speaker in the studio. A name is optional downstream LLM metadata. Pass a preceding cue start as evidence."""
        if clear_speaker_number and speaker_number is not None:
            raise ValueError("Выбери назначение или сброс номера.")
        body = {"id": material_id, "revision": revision, "startMs": start_ms, "source": "agent"}
        if clear_speaker_number or speaker_number is not None:
            body["speakerNumber"] = speaker_number
        if speaker_name is not None:
            body["speakerName"] = speaker_name
        if text is not None:
            body["text"] = text
        if not any(field in body for field in ("speakerNumber", "speakerName", "text")):
            raise ValueError("Укажи номер спикера или исправленный текст.")
        body["contextStartMs"] = context_start_ms
        return studio.request("transcript-cue", language, body)

    @mcp.tool()
    async def media_transcribe_stream(url: str, capture_seconds: int = 30, language: TranscriptionLanguage | None = None,
                                speakers: bool = False, speaker_count: int | None = None,
                                audio_classifier: SoundClassifier = "auto") -> dict[str, Any]:
        """Capture 1–120 seconds of HLS or RTSP and return timed speech and SRT.

        Omit language to use Whisper settings or pass ru, en, or auto.
        Set speakers to label turns within the excerpt.
        """
        result = await transcription_lane.run(studio.request, "transcribe-stream", "en" if language == "en" else "ru", {
            "url": url, "seconds": capture_seconds, "speakers": speakers or speaker_count is not None,
            "speaker_count": speaker_count, "audio_classifier": audio_classifier,
        }, timeout=1020, speech_lang=language or "default")
        return studio.save_subtitles(result)

    return mcp


def main():
    """Connect the stdio server to a running loopback ag-media-studio."""
    parser = argparse.ArgumentParser(prog="ag-media-studio-mcp", description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", type=Path, default=PROJECT_DATA / "mcp-output")
    parser.add_argument("--studio-data", type=Path)
    parser.add_argument("--graph-root", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    build_server(StudioClient(args.port, args.output), studio_root=args.studio_data,
                 graph_root=args.graph_root).run(transport="stdio")


if __name__ == "__main__":
    main()
