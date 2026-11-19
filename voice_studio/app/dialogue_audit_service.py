"""Validate and serialize source-bound dialogue audits for UI and local agents.

The same monotonic deadline covers hashing, native admission, PCM preparation
and the runner. Cached PCM requires matching source/clock receipt and bytes.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic, sleep
import os
import re
import wave

from voice_paths import ensure_temporary_root, workspace_temp
from voice_studio.core.media_limits import MAX_UPLOAD_BYTES, MAX_SOURCE_MS
from voice_studio.core.media_resources import run_bounded_batch_stage
from voice_studio.core.storage_cache import use_cache
from voice_studio.graph.media_graph_ingest import probe_source, FORMATS
from voice_studio.graph.media_graph_store import signature
from voice_studio.speech.dialogue_audit import AuditOptions, stable_hash
from voice_studio.speech.native_turn_bounds import audit_native_dialogue
from voice_studio.speech.dialogue_audit_runner import AuditCancelled, run_audit, atomic_json, file_hash
from voice_studio.synthesis.engines import ffmpeg

try:
    import fcntl
except ImportError:
    fcntl = None

MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_CUE_TIME_MS = 4 * MAX_SOURCE_MS
PCM_VERSION = 'original-clock-mono16k-v2'
SERVICE_VERSION = 'dialogue-audit-service-v3'
LOCAL_CACHE_LOCK = Lock()


def read_document(path):
    """Capture one bounded absolute JSON file before any native media work."""
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Нужен абсолютный путь к JSON до 64 МиБ.')
    before = path.stat()
    if not 0 < before.st_size <= MAX_REPORT_BYTES:
        raise ValueError('Нужен непустой JSON до 64 МиБ.')
    with path.open('rb') as stream:
        contents = stream.read(MAX_REPORT_BYTES + 1)
    after = path.stat()
    if (len(contents) > MAX_REPORT_BYTES or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns
            or before.st_ino != after.st_ino):
        raise ValueError('JSON изменился во время чтения или превышает лимит.')
    try:
        return json.loads(contents)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError('Не удалось прочитать JSON анализа.') from error


def read_report(path):
    """Read a transcript object; detailed source-clock validation follows separately."""
    value = read_document(path)
    if not isinstance(value, dict) or not isinstance(value.get('segments'), list):
        raise ValueError('В JSON нет расшифровки.')
    return value


def validate_budgets(rerun, max_windows, max_audio_ms, timeout_seconds):
    """Reject invalid native budgets before reading JSON or hashing source media."""
    if type(rerun) is not bool:
        raise ValueError('rerun должен быть логическим значением.')
    if type(max_windows) is not int or not 0 <= max_windows <= 100:
        raise ValueError('max_windows must be 0..100')
    if type(max_audio_ms) is not int or not 0 <= max_audio_ms <= 600000:
        raise ValueError('max_audio_ms must be 0..600000')
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 3600:
        raise ValueError('timeout_seconds must be in (0, 3600]')


def validate_inputs(report, turns, rerun, max_windows, max_audio_ms, timeout_seconds):
    """Fail closed on malformed source-clock facts before native decoding."""
    validate_budgets(rerun, max_windows, max_audio_ms, timeout_seconds)
    duration = report.get('durationMs')
    if type(duration) is not int or not 0 < duration <= MAX_SOURCE_MS:
        raise ValueError('Нужна длительность анализа до двух часов в миллисекундах.')
    if not isinstance(turns, list):
        raise ValueError('JSON голосовых интервалов должен содержать список turns.')
    language = report.get('language', 'auto')
    if not isinstance(language, str) or not re.fullmatch(r'[A-Za-z][A-Za-z_-]{0,15}', language):
        raise ValueError('Некорректный язык распознавания.')
    # Reuse the detector's interval/row/candidate bounds. The declared hash is a
    # placeholder here; it is checked against actual source bytes before native work.
    declared_source = report.get('source') or {}
    if not isinstance(declared_source, dict):
        raise ValueError('Некорректные метаданные исходника.')
    declared = report.get('sourceDigest') or report.get('sourceSha256') or declared_source.get('sha256')
    identity = {'sha256': declared or '0' * 64, 'durationMs': duration}
    try:
        cue_time = 0
        for row in report['segments']:
            if not isinstance(row, dict) or not isinstance(row.get('text', ''), str):
                raise ValueError('Некорректная фраза расшифровки.')
            start, end = row.get('startMs'), row.get('endMs')
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= duration:
                raise ValueError('Некорректные границы фразы расшифровки.')
            cue_time += end - start
            if cue_time > MAX_CUE_TIME_MS:
                raise ValueError('Суммарная длительность фраз превышает бюджет перекрытий.')
        for row in turns:
            if not isinstance(row, dict) or type(row.get('speaker')) is bool:
                raise ValueError('Некорректный голосовой интервал.')
        audit_native_dialogue(report, turns, source=identity, options=AuditOptions())
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError('Некорректные поля анализа или голосовых интервалов.') from error


def validate_bindings(*documents):
    """Reject contradictory source digests or nonzero imported clock offsets."""
    hashes = []
    for document in documents:
        source = document.get('source') or {}
        if not isinstance(source, dict):
            raise ValueError('Некорректные метаданные исходника.')
        for value in (document.get('clockOffsetMs', 0), source.get('clockOffsetMs', 0)):
            if type(value) is not int or value != 0:
                raise ValueError('Интервалы должны быть на исходной шкале времени: clockOffsetMs=0.')
        for value in (document.get('sourceDigest'), document.get('sourceSha256'), source.get('sha256')):
            if value is None:
                continue
            if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
                raise ValueError('Некорректный SHA-256 исходника.')
            hashes.append(value)
    if len(set(hashes)) > 1:
        raise ValueError('JSON анализа и голосовых интервалов относятся к разным исходникам.')
    return hashes


def identify_source(path, check):
    """Preserve the graph identity contract while bounding source hashing by deadline."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError('Укажи абсолютный путь к локальному медиафайлу.')
    path = path.resolve()
    before = signature(path)
    if before[2] > MAX_UPLOAD_BYTES:
        raise ValueError('Размер исходника должен быть не больше 2 ГиБ.')
    digest = file_hash(path, check)
    if signature(path) != before:
        raise ValueError('Исходник изменился во время чтения.')
    check()
    return {'path': str(path), 'signature': before, 'sha256': digest}


def stable_source(record, check):
    """Require the request's exact source snapshot without an unbudgeted rehash.

    Unlike reopening a durable graph snapshot, one in-flight decode must abort
    even on metadata-only replacement. A new request can verify it afresh.
    """
    check()
    path = Path(record['path'])
    if signature(path) != record['signature']:
        raise ValueError('Исходник изменился во время аудита.')
    return path


def audio_stream(media):
    """Validate the audio stream's measured offset on the original media clock."""
    duration = media.get('durationMs')
    if type(duration) is not int or not 0 < duration <= MAX_SOURCE_MS:
        raise ValueError('Некорректная длительность исходника.')
    audio = next((row for row in media.get('streams', []) if row.get('kind') == 'audio'), None)
    if audio is None:
        raise ValueError('В исходнике нет звуковой дорожки.')
    start = audio.get('startMs', 0)
    if type(start) is not int or not 0 <= start < duration or type(audio.get('index')) is not int or audio['index'] < 0:
        raise ValueError('Некорректное смещение звуковой дорожки.')
    return audio


def inspect_pcm(path, duration):
    """Reject wrong-format, wrong-length and truncated PCM before accepting a cache."""
    with wave.open(str(path), 'rb') as audio:
        frames = audio.getnframes()
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('Некорректный формат кэшированного PCM.')
        if abs(frames - duration * 16) > 16 or frames <= 0:
            raise ValueError('Некорректная длительность кэшированного PCM.')
        audio.setpos(frames - 1)
        if len(audio.readframes(1)) != 2:
            raise ValueError('Кэшированный PCM обрезан.')
    return frames


def prepare_pcm(source, media, output, *, timeout=180, check=lambda: None):
    """Atomically decode full-clock PCM with measured leading silence and trailing pad.

    Cancellation is observed before/after FFmpeg; its blocking subprocess itself
    is bounded by the remaining caller timeout. Existing PCM survives failures.
    """
    audio = audio_stream(media)
    filters = ['asetpts=PTS-STARTPTS', 'aresample=async=1:first_pts=0']
    if audio.get('startMs', 0):
        filters.append(f"adelay={audio['startMs']}:all=1")
    filters.extend(['apad', f"atrim=duration={media['durationMs'] / 1000}"])
    output = Path(output)
    with NamedTemporaryFile(dir=output.parent, suffix='.wav', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        check()
        ffmpeg(['-protocol_whitelist', 'file,pipe', '-format_whitelist', FORMATS,
                '-i', str(source), '-map', f"0:{audio['index']}", '-vn', '-af', ','.join(filters),
                '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(temporary)], timeout=timeout)
        check()
        inspect_pcm(temporary, media['durationMs'])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def cache_writer(directory, check):
    """Serialize same-source cache writers across processes with bounded lock polling.

    Without POSIX flock, the fallback serializes this process only. The shared
    storage-cache lease independently prevents cleanup while artifacts are used.
    """
    if fcntl is None:
        while not LOCAL_CACHE_LOCK.acquire(timeout=.05):
            check()
        try:
            check()
            yield
        finally:
            LOCAL_CACHE_LOCK.release()
        return
    with (directory / '.dialogue-audit.lock').open('a') as lock:
        while True:
            check()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reusable_pcm(pcm, receipt, expected, duration, check):
    """Require a source/clock-matched receipt, valid WAV and matching byte digest."""
    try:
        if pcm.is_symlink() or receipt.is_symlink() or not pcm.is_file() or not receipt.is_file() or receipt.stat().st_size > 4096:
            return False
        value = json.loads(receipt.read_bytes())
        if not isinstance(value, dict) or value.get('inputKey') != expected or value.get('durationMs') != duration:
            return False
        frames = inspect_pcm(pcm, value['durationMs'])
        return frames == value.get('frames') and file_hash(pcm, check) == value.get('pcmSha256')
    except (OSError, ValueError, KeyError, TypeError, EOFError, RecursionError, wave.Error):
        return False


def audit_media(root, source_path, report_path, *, turns_path=None, rerun=False,
                max_windows=6, max_audio_ms=30000, timeout_seconds=180,
                transient_source=False, emit=None, cancelled=None):
    """Validate first, then run within bounded shared native admission and cache locks.

    Historical reports without measured turns return unavailable. Reports/turns
    are captured once: external edits cannot change the in-flight invocation.
    Source bytes and every declared source hash must agree. Up to 100 ms of
    historical duration rounding is accepted; PCM pads to the greater duration.
    JSON is bounded to 64 MiB per document; source duration to two hours,
    aggregate cue-time to eight hours, including overlapping/duplicate cues.
    On non-POSIX hosts writer serialization is process-local. Cancellation
    during PCM decoding is observed when its deadline-bounded FFmpeg returns.
    """
    started = monotonic()
    # Budget validation precedes reading large JSON or hashing original media.
    validate_budgets(rerun, max_windows, max_audio_ms, timeout_seconds)
    deadline = started + timeout_seconds

    def check():
        """Bound every phase with the same deadline and optional cancellation hook."""
        if cancelled is not None and cancelled():
            raise AuditCancelled('dialogue audit cancelled')
        if monotonic() >= deadline:
            raise TimeoutError('dialogue audit timeout')

    def remaining():
        """Pass only the unused budget to each native blocking phase."""
        check()
        return deadline - monotonic()

    check()
    report = read_report(report_path)
    if turns_path is None:
        turns = report.get('speechTurns')
        unavailable = turns is None or turns == []
        turns = [] if unavailable else turns
        turns_document = {'turns': turns}
    else:
        captured = read_document(turns_path)
        turns = captured.get('turns') if isinstance(captured, dict) else captured
        unavailable = turns == []
        turns_document = captured if isinstance(captured, dict) else {'turns': turns}
    declared_hashes = validate_bindings(report, turns_document)
    validate_inputs(report, turns, rerun, max_windows, max_audio_ms, timeout_seconds)
    check()
    if rerun and not unavailable:
        from voice_studio.speech.recognition import SMALL_MODEL, ready
        if not ready(SMALL_MODEL):
            raise ValueError('Установленная локальная модель Whisper недоступна.')
    source = identify_source(source_path, check)
    original = stable_source(source, check)
    for declared in declared_hashes:
        if declared and declared != source['sha256']:
            raise ValueError('Расшифровка относится к другому исходнику.')
    media = probe_source(original, timeout=remaining())
    check()
    audio = audio_stream(media)
    if abs(report['durationMs'] - media['durationMs']) > 100:
        raise ValueError('Длительность исходника и расшифровки не совпадает.')
    if unavailable:
        stable_source(source, check)
        check()
        return {'status': 'unavailable', 'reason': 'native-voice-turns-unavailable',
                'message': 'Перезапусти речевой анализ для измерения границ голосов.', 'sourceSha256': source['sha256']}
    ensure_temporary_root()
    group = workspace_temp(root) / 'analysis-checkpoints'
    directory = group / ('dialogue-audit-' + source['sha256'])
    metadata = {'sha256': source['sha256'], 'sizeBytes': source['signature'][2],
                'durationMs': report['durationMs'], 'clockOffsetMs': 0,
                'reportSourceBinding': 'verified-source-hash' if validate_bindings(report) else 'caller-supplied-duration-checked',
                'path': 'sha256:' + source['sha256'] if transient_source else str(original)}
    # Reports historically round duration upward; pad rather than letting a
    # valid source-clock turn run beyond decoded EOF. Never trim actual media.
    pcm_media = {**media, 'durationMs': max(media['durationMs'], report['durationMs'])}
    pcm_identity = {'version': PCM_VERSION, 'sourceSha256': source['sha256'],
                    'durationMs': pcm_media['durationMs'], 'sourceDurationMs': media['durationMs'], 'audioIndex': audio['index'], 'audioStartMs': audio.get('startMs', 0)}
    pcm_key = stable_hash(pcm_identity)
    invocation_key = stable_hash({'version': SERVICE_VERSION, 'report': report, 'turns': turns_document,
                                  'rerun': rerun, 'maxWindows': max_windows, 'maxAudioMs': max_audio_ms,
                                  'timeoutSeconds': timeout_seconds, 'sourceMetadata': metadata})

    def runner_cancelled():
        """Keep the requested runner/cache budget stable while enforcing the shared deadline."""
        check()
        return False

    def execute():
        """Own all same-source cache mutations only after shared native admission."""
        with use_cache(group, directory, deadline=deadline), cache_writer(directory, check):
            pcm = directory / (PCM_VERSION + '.wav')
            receipt = directory / (PCM_VERSION + '.json')
            if not reusable_pcm(pcm, receipt, pcm_key, pcm_media['durationMs'], check):
                prepare_pcm(original, pcm_media, pcm, timeout=remaining(), check=check)
                stable_source(source, check)
                atomic_json(receipt, {'inputKey': pcm_key, **pcm_identity,
                                     'frames': inspect_pcm(pcm, pcm_media['durationMs']), 'pcmSha256': file_hash(pcm, check)})
            stable_source(source, check)
            invocation = directory / invocation_key
            invocation.mkdir(parents=True, exist_ok=True)
            normalized_report = invocation / 'input-report.json'
            normalized_turns = invocation / 'input-turns.json'
            protected = {Path(report_path).expanduser().resolve(), original}
            if turns_path is not None:
                protected.add(Path(turns_path).expanduser().resolve())
            if normalized_report in protected or normalized_turns in protected:
                raise ValueError('Путь кэша совпадает с входным JSON.')
            atomic_json(normalized_report, report)
            atomic_json(normalized_turns, turns_document)
            result = run_audit(normalized_report, normalized_turns, pcm, invocation / 'results',
                               rerun=rerun, max_windows=max_windows, max_audio_ms=max_audio_ms,
                               timeout_seconds=timeout_seconds, source_metadata=metadata, emit=emit, cancelled=runner_cancelled)
            stable_source(source, check)
            check()
            return result

    return run_bounded_batch_stage(execute, deadline=deadline)
