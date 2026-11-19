"""Bounded local analysis for explicitly selected browser tab fragments."""

from voice_studio.mod_runtime import model_resource, worker_environment, current_snapshot, analysis_run, submit_with_context, is_replacement

from array import array
from concurrent.futures import ThreadPoolExecutor
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import select
import subprocess
from threading import Lock, Timer
from time import monotonic, perf_counter
import wave

from voice_studio.core.worker_process import worker_command
from voice_studio.synthesis.engines import bounded_timeout, ffmpeg
from voice_studio.speech.media_transcript import segments_from_whisper
from voice_studio.speech.recognition import MODEL, recognize_file
from voice_paths import model_path
from voice_studio.vision.video_revolver import PYTHON, VISION, MODEL as YUNET
from voice_studio.core.optional_runtime import runtime_environment

MAX_BYTES = 5 * 1024 * 1024
MAX_SECONDS = 6
FRAME_STEP_MS = 250
IDLE_WORKER_SECONDS = 20
ANALYSIS_BUDGET = 10
RECENT_SESSIONS = 8
LIVE_MODEL = (MODEL.with_name("ggml-base-q5_1.bin") if os.getenv("VOICE_WHISPER_MODEL")
              else model_path("whisper", "ggml-base-q5_1.bin"))
WEBM_FORMATS = "matroska,webm"
EXTENSION_ORIGIN = re.compile(r"chrome-extension://[a-p]{32}")


class LiveBusy(ValueError):
    """Report a dropped concurrent or unordered fragment without queueing it."""


def live_token(root):
    """Create or read the extension's local secret, keeping it out of URLs and logs."""
    configured = os.environ.get("VOICE_LIVE_TOKEN")
    if configured is not None:
        if len(configured) < 32 or not configured.isascii():
            raise ValueError("VOICE_LIVE_TOKEN должен содержать не менее 32 ASCII-символов.")
        return configured
    path = root / "live-token.txt"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="ascii").strip()
    value = secrets.token_hex(32)
    with os.fdopen(descriptor, "w", encoding="ascii") as file:
        file.write(value + "\n")
    return value


class FaceProcess:
    """Keep an OpenCV child warm and restart it if a request fails or times out."""

    def __init__(self):
        self.process = None
        self.next_numbers = {}
        self.mod_revision = None

    def run(self, session, folder, start_ms, deadline):
        """Return a bounded JSON response for this batch without copying image bytes."""
        timeout = bounded_timeout(deadline, 5)
        revision = current_snapshot().fingerprint(('vision.faces', 'vision.face-embedding'))
        if self.mod_revision != revision:
            self.close()
            self.mod_revision = revision
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.process = subprocess.Popen(
                worker_command(PYTHON, "voice_studio.workers.live_face_worker", unbuffered=True),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=runtime_environment("vision", worker_environment({**os.environ, "PYTHONPATH": str(VISION)})), text=True,
            )
        process = self.process
        request = {
            "session": session, "folder": str(folder), "startMs": start_ms, "stepMs": FRAME_STEP_MS,
            "firstNumber": self.next_numbers.get(session, 1),
        }
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            if not select.select([process.stdout], [], [], timeout)[0]:
                raise ValueError("Превышено время стадии лиц.")
            result = json.loads(process.stdout.readline())
            if "error" in result:
                raise ValueError(result["error"])
            self.next_numbers.pop(session, None)
            self.next_numbers[session] = result["nextNumber"]
            if len(self.next_numbers) > RECENT_SESSIONS:
                del self.next_numbers[next(iter(self.next_numbers))]
            return result
        except (ValueError, OSError):
            self.close()
            raise

    def close(self):
        """Terminate the private worker and close its local pipes."""
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdin.close()
        process.stdout.close()


class LiveCapture:
    """Admit only one fragment at a time and keep recent session clocks in memory."""

    def __init__(self, root):
        self.token = live_token(root)
        self.lock = Lock()
        self.positions = {}
        self.faces = FaceProcess()
        self.idle = None
        self.last_activity = monotonic()

    def authorized(self, candidate):
        """Check an ASCII bearer token without exposing its value."""
        return (bool(self.token) and candidate.isascii()
                and hmac.compare_digest(candidate, self.token))

    def close(self):
        """Release the warm model when the studio stops."""
        if self.idle is not None:
            self.idle.cancel()
        with self.lock:
            self.faces.close()

    def release_idle_worker(self):
        """Release camera inference memory when the extension stops sending fragments."""
        with self.lock:
            if monotonic() - self.last_activity >= IDLE_WORKER_SECONDS:
                self.faces.close()

    @analysis_run
    def process(self, path, session, start_ms, language):
        """Return capture-clock speech and visible faces, with no persistent recording."""
        if not re.fullmatch(r"[0-9a-f]{32}", session) or not 0 <= start_ms <= 24 * 3600 * 1000:
            raise ValueError("Некорректная сессия или позиция.")
        if not self.lock.acquire(blocking=False):
            raise LiveBusy("Предыдущий фрагмент ещё обрабатывается.")
        try:
            previous = self.positions.get(session)
            if previous is not None and start_ms <= previous:
                raise LiveBusy("Этот фрагмент уже прошёл или устарел.")
            result = self.analyze(path, session, start_ms, language)
            self.positions.pop(session, None)
            self.positions[session] = start_ms
            if len(self.positions) > RECENT_SESSIONS:
                del self.positions[next(iter(self.positions))]
            return result
        finally:
            self.last_activity = monotonic()
            if self.idle is not None:
                self.idle.cancel()
            self.idle = Timer(IDLE_WORKER_SECONDS, self.release_idle_worker)
            self.idle.daemon = True
            self.idle.start()
            self.lock.release()

    def analyze(self, path, session, start_ms, language):
        """Decode once, then run bounded visual sampling and local Whisper independently."""
        started = perf_counter()
        deadline = monotonic() + ANALYSIS_BUDGET
        wav, frames = decode_fragment(path, deadline)
        decoded = perf_counter()
        has_faces = frames and (YUNET.is_file() or is_replacement('vision.faces', 'builtin.yunet')) and PYTHON.is_file()
        face_frames, face_warnings, face_ms = [], [], 0
        segments, duration, speech_warnings, speech_ms = [], 0, [], 0
        if has_faces and wav:
            # Join before the temporary directory and the single-fragment lock are released.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-faces") as worker:
                faces = submit_with_context(worker, self.track_faces, path.parent, session, start_ms, deadline)
                segments, duration, speech_warnings, speech_ms = transcribe_chunk(wav, start_ms, language, deadline)
                face_frames, face_warnings, face_ms = faces.result()
        elif has_faces:
            face_frames, face_warnings, face_ms = self.track_faces(path.parent, session, start_ms, deadline)
        elif wav:
            segments, duration, speech_warnings, speech_ms = transcribe_chunk(wav, start_ms, language, deadline)
        finished = perf_counter()
        return {
            "segments": segments,
            "faceFrames": face_frames,
            "warnings": face_warnings + speech_warnings,
            "durationMs": max(duration, frames * FRAME_STEP_MS),
            "timingsMs": {"decode": round((decoded - started) * 1000),
                          "faces": face_ms, "speech": speech_ms,
                          "total": round((finished - started) * 1000)},
        }

    def track_faces(self, folder, session, start_ms, deadline):
        """Return face observations, recoverable warnings and elapsed milliseconds."""
        started = perf_counter()
        warnings = []
        frames = []
        try:
            seen = self.faces.run(session, folder, start_ms, deadline)
            frames = seen["faceFrames"]
            if not seen["sface"]:
                warnings.append("Сопоставление лиц после смены кадра недоступно.")
            if seen["trackingReset"]:
                warnings.append("Трекинг лиц возобновился. Обнаруженные лица получат новые номера.")
        except (OSError, ValueError):
            warnings.append("Лица в этом фрагменте не обработаны.")
        return frames, warnings, round((perf_counter() - started) * 1000)


def transcribe_chunk(wav, start_ms, language, deadline):
    """Return capture-clock speech, audio duration, warnings and elapsed milliseconds."""
    started = perf_counter()
    duration = 0
    segments, warnings = [], []
    try:
        duration, audible = audio_info(wav)
        if audible:
            model = LIVE_MODEL if LIVE_MODEL.is_file() else MODEL
            raw = recognize_file(wav, wav.parent / "speech", language, model=model,
                                 threads=2, timeout=bounded_timeout(deadline, 10), wait=0)
            segments = segments_from_whisper(raw, start_ms + duration, audio_start_ms=start_ms)
    except (OSError, ValueError, wave.Error):
        warnings.append("Речь в этом фрагменте не распознана.")
    return segments, duration, warnings, round((perf_counter() - started) * 1000)


def stream_types(path, deadline):
    """Inspect local WebM streams before requesting optional outputs."""
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", WEBM_FORMATS,
         "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        stderr=subprocess.DEVNULL, timeout=bounded_timeout(deadline, 4),
    )
    return {item.get("codec_type") for item in json.loads(raw).get("streams", ())}


def decode_fragment(path, deadline):
    """Extract 16 kHz mono audio and at most 24 small video frames in one FFmpeg pass."""
    try:
        kinds = stream_types(path, deadline)
        wav = path.with_name("live.wav") if "audio" in kinds else None
        options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", WEBM_FORMATS,
                   "-threads", "2", "-filter_threads", "1", "-i", str(path)]
        if wav:
            options += ["-map", "0:a:0", "-t", str(MAX_SECONDS), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(wav)]
        if "video" in kinds:
            options += ["-map", "0:v:0", "-t", str(MAX_SECONDS), "-vf",
                        "fps=4:start_time=0,scale=480:480:force_original_aspect_ratio=decrease",
                        "-threads", "1", "-q:v", "4", "-start_number", "0", str(path.with_name("frame-%03d.jpg"))]
        if not wav and "video" not in kinds:
            raise ValueError("В WebM нет звука или видео.")
        ffmpeg(options, timeout=bounded_timeout(deadline, 8))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось открыть фрагмент WebM.") from exc
    return wav, len(list(path.parent.glob("frame-*.jpg")))


def audio_info(wav):
    """Skip obvious silence before launching an external recognizer."""
    with wave.open(str(wav)) as pcm:
        duration_ms = pcm.getnframes() * 1000 // pcm.getframerate()
        samples = array("h", pcm.readframes(pcm.getnframes()))
        return duration_ms, duration_ms >= 350 and any(abs(value) >= 120 for value in samples)
