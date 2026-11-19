"""Run a local, cancellable dialogue audit with content-keyed atomic outputs.

The detector is pure; this module owns files, PCM crops and optional local ASR.
Partial/cancelled runs have their own artifacts and never replace audit.json.
"""

from array import array
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic, sleep
from uuid import uuid4
import wave

from voice_studio.core.media_resources import stage_threads
from voice_studio.speech.dialogue_audit import AuditOptions, stable_hash
from voice_studio.speech.native_turn_bounds import audit_native_dialogue

RUNNER_VERSION = '1.1.2'


class AuditCancelled(Exception):
    """Caller requested cancellation; the prior successful artifact remains intact."""


def file_hash(path, check=lambda: None):
    """Hash source bytes incrementally without reading large recordings into RAM."""
    digest = sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            check()
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """Publish a complete JSON file through same-directory rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    """Read an explicit input or a previously complete local cache entry."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snippet_plan(inventory, max_windows, max_audio_ms):
    """Choose deterministic high-priority clean windows within hard audio/count caps."""
    candidates = sorted(inventory['candidates'], key=lambda row: (
        row['priority'] != 'high',
        'sparse-text-over-voiced-audio' not in row['reasons'],
        row['evidence'].get('wordsPerVoicedSecond', 0), row['startMs'], row['id']))
    proposals = {}
    for candidate in candidates:
        rows = candidate['reviewWindows']
        if 'sparse-text-over-voiced-audio' in candidate['reasons']:
            # A short question often precedes the missing answer. Prioritize the
            # longest clean supported span, rather than decoding the known text.
            rows = sorted(rows, key=lambda row: (row['startMs'] - row['endMs'], row['startMs']))
        proposals[candidate['id']] = rows
    windows, seen, spent = [], set(), 0
    # Round robin gives both sides of a mixed phrase a chance before a long cue
    # consumes the entire budget. Overlapping proposals are decoded only once.
    depth = 0
    while len(windows) < max_windows:
        found = False
        for candidate in candidates:
            if depth >= len(proposals[candidate['id']]):
                continue
            found = True
            row = proposals[candidate['id']][depth]
            start, end = row['startMs'], row['endMs']
            if any(start < right and end > left for left, right in seen):
                continue
            length = end - start
            if spent + length > max_audio_ms:
                continue
            windows.append({**row, 'candidateId': candidate['id'],
                            'originalCueIndices': candidate['evidence']['cueIndices']})
            seen.add((start, end))
            spent += length
            if len(windows) >= max_windows:
                break
        if not found:
            break
        depth += 1
    return windows


def crop_pcm(source, target, start, end):
    """Copy exact source-clock PCM samples and measure energy for silence pruning."""
    with wave.open(str(source), 'rb') as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000:
            raise ValueError('rerun requires mono 16 kHz 16-bit PCM WAV')
        audio.setpos(start * 16)
        pcm = audio.readframes((end - start) * 16)
        if len(pcm) != (end - start) * 32:
            raise ValueError('snippet exceeds source WAV')
        parameters = audio.getparams()
    values = array('h', pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    energy = sum(value * value for value in values) / max(1, len(values))
    rms = energy ** .5
    if rms < 80:
        return rms  # Silence does not become a hallucinated ASR repair.
    with wave.open(str(target), 'wb') as output:
        output.setparams(parameters)
        output.writeframes(pcm)
    return rms


def decode_local(wav, output, language, model, check, *, threads=None):
    """Run installed Whisper only, killing the child promptly on timeout/cancel."""
    from voice_studio.speech.recognition import LOCK, current_cli
    threads = stage_threads(4) if threads is None else threads
    command = [str(current_cli()), '-m', str(model), '-l', language, '-ojf', '-mc', '0',
               '-bs', '1', '-bo', '1', '-nf', '-ng', '-t', str(threads), '-f', str(wav), '-of', str(output)]
    while not LOCK.acquire(timeout=.1):
        check()
    try:
        check()
        # A file avoids a full stderr pipe blocking an otherwise bounded child.
        with TemporaryFile() as errors, subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors) as process:
            try:
                while process.poll() is None:
                    check()
                    sleep(.1)
                process.wait()
                if process.returncode:
                    errors.seek(0, 2)
                    errors.seek(max(0, errors.tell() - 500))
                    stderr = errors.read(500).decode(errors='replace')
                    raise ValueError(f'local Whisper exited {process.returncode}: {stderr}')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        check()
        return read_json(output.with_suffix('.json'))
    finally:
        LOCK.release()


def projected_suggestions(raw, window):
    """Keep raw model output separate; project segment offsets without alignment claims."""
    suggestions = []
    width = window['endMs'] - window['startMs']
    for row in raw.get('transcription', []):
        offsets = row.get('offsets') or {}
        left, right = offsets.get('from'), offsets.get('to')
        if type(left) is not int or type(right) is not int:
            continue
        left, right = max(0, left), min(width, right)
        text = row.get('text', '').strip()
        if left >= right or not text:
            continue
        suggestions.append({'candidateId': window['candidateId'],
                            'startMs': window['startMs'] + left, 'endMs': window['startMs'] + right,
                            'text': text, 'speaker': window['speaker'], 'identity': 'unresolved',
                            'timingBasis': 'asr-segment-offsets', 'forceAligned': False,
                            'originalCueIndices': window['originalCueIndices'],
                            'acceptance': 'review-required'})
    return suggestions


def retry_inventory(inventory, source, work, model, language, max_windows, max_audio_ms, check, emit, *, threads=None):
    """Decode budgeted single-label snippets; silence and failure remain explicit."""
    windows = snippet_plan(inventory, max_windows, max_audio_ms)
    repairs = {'status': 'complete', 'suggestions': [], 'windows': [],
               'budget': {'maxWindows': max_windows, 'maxAudioMs': max_audio_ms,
                          'plannedWindows': len(windows),
                          'plannedAudioMs': sum(row['endMs'] - row['startMs'] for row in windows),
                          'decodedWindows': 0, 'decodedAudioMs': 0}}
    inventory['repairs'] = repairs
    for index, window in enumerate(windows):
        check()
        wav, output = work / f'{index}.wav', work / f'{index}-asr'
        rms = crop_pcm(source, wav, window['startMs'], window['endMs'])
        evidence = {**window, 'pcmRms': round(rms, 2)}
        repairs['windows'].append(evidence)
        if rms < 80:
            evidence['status'] = 'silence-pruned'
            continue
        raw = decode_local(wav, output, language, model, check, threads=threads)
        # Store the native result independently of editable suggestions. In
        # particular t_dtw=-1 is retained as-is and never promoted to alignment.
        evidence.update(status='decoded', rawASR=raw)
        repairs['suggestions'].extend(projected_suggestions(raw, window))
        repairs['budget']['decodedWindows'] += 1
        repairs['budget']['decodedAudioMs'] += window['endMs'] - window['startMs']
        emit({'stage': 'dialogue-audit-asr', 'windowsDone': index + 1, 'windowsTotal': len(windows)})
    return repairs


def run_audit(report_path, turns_path, source_path, output_dir, *, visual_path=None, options=None,
              rerun=False, max_windows=6, max_audio_ms=30000, timeout_seconds=120,
              cancelled=None, emit=None, model_path=None, source_metadata=None):
    """Audit a full-source PCM WAV and return JSON plus artifactPath/cacheHit.

    Inputs are read-only. Each key binds source bytes, report/turn/AV JSON values,
    detector version/options and rerun model/options. Cancellation and timeout
    cover hashing and child lifetime; incomplete attempts never replace the last
    complete ``audit.json`` or enter the successful-result cache. Optional
    ``source_metadata`` binds a caller-verified original MOV identity to decoded
    PCM; both original sha256 and pcmSha256 participate in the cache identity.
    """
    if type(max_windows) is not int or not 0 <= max_windows <= 100:
        raise ValueError('max_windows must be 0..100')
    if type(max_audio_ms) is not int or not 0 <= max_audio_ms <= 600000:
        raise ValueError('max_audio_ms must be 0..600000')
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 3600:
        raise ValueError('timeout_seconds must be in (0, 3600]')
    started = monotonic()
    deadline = started + timeout_seconds
    emit = emit or (lambda event: None)

    def check():
        """Bound all expensive phases by the same caller deadline and cancel hook."""
        if cancelled is not None and cancelled():
            raise AuditCancelled('dialogue audit cancelled')
        if monotonic() >= deadline:
            raise TimeoutError('dialogue audit timeout')

    check()
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    protected = [Path(report_path).resolve(), Path(turns_path).resolve(), source_path]
    if visual_path:
        protected.append(Path(visual_path).resolve())
    if output_dir / 'audit.json' in protected:
        raise ValueError('output would overwrite an input')
    report, turns = read_json(report_path), read_json(turns_path)
    visual = read_json(visual_path) if visual_path else None
    with wave.open(str(source_path), 'rb') as audio:
        actual_ms = round(audio.getnframes() * 1000 / audio.getframerate())
    # Historical reports round media duration to tenths of a second; decoded
    # PCM remains sample-exact and crops are still checked against actual EOF.
    if abs(actual_ms - report.get('durationMs', -1000)) > 100:
        raise ValueError('full-source WAV/report duration mismatch')
    pcm_hash = file_hash(source_path, check)
    source = {'path': str(source_path), 'sha256': pcm_hash,
              'sizeBytes': source_path.stat().st_size, 'durationMs': report['durationMs'],
              'clock': 'original-media-ms', 'clockOffsetMs': 0,
              'reportBinding': 'verified-hash' if report.get('sourceSha256') or report.get('sourceDigest') or (report.get('source') or {}).get('sha256')
              else 'caller-supplied-source-duration-checked'}
    if source_metadata is not None:
        if source_metadata.get('durationMs') != report['durationMs']:
            raise ValueError('original source/report duration mismatch')
        if source_metadata.get('clockOffsetMs', 0) != 0:
            raise ValueError('only full-source PCM with clockOffsetMs=0 is supported')
        if not source_metadata.get('sha256') or not source_metadata.get('path'):
            raise ValueError('source_metadata requires original sha256 and path')
        source = dict(source_metadata)
        source.update(pcmPath=str(source_path), pcmSha256=pcm_hash, pcmSizeBytes=source_path.stat().st_size,
                      clock='original-media-ms', clockOffsetMs=0, reportBinding='caller-verified-original-source')
    inventory = audit_native_dialogue(report, turns, source=source, visual_facts=visual, options=options)
    check()
    run_options = {'rerun': rerun, 'maxWindows': max_windows, 'maxAudioMs': max_audio_ms,
                   'timeoutSeconds': timeout_seconds, 'runnerVersion': RUNNER_VERSION,
                   'effectiveThreads': stage_threads(4) if rerun else None}
    model = None
    if rerun:
        from voice_studio.speech.recognition import SMALL_MODEL, current_cli, ready
        model = Path(model_path) if model_path else SMALL_MODEL
        if not ready(model):
            raise ValueError('installed local Whisper executable/model unavailable; no download attempted')
        run_options['model'] = {'path': str(model.resolve()), 'sha256': file_hash(model, check)}
        run_options['executableSha256'] = file_hash(current_cli(), check)
        run_options['decodeOptions'] = ['-ojf', '-mc', '0', '-bs', '1', '-bo', '1', '-nf', '-ng', '-t', str(run_options['effectiveThreads'])]
    run_key = stable_hash({'inventoryKey': inventory['cacheKey'], 'runOptions': run_options})
    inventory.update(runKey=run_key, runOptions=run_options)
    cache_path = output_dir / 'cache' / f'{run_key}.json'
    if cache_path in protected:
        raise ValueError('cache output would overwrite an input')
    if cache_path.is_file():
        try:
            cached = read_json(cache_path)
            if (isinstance(cached, dict) and cached.get('runKey') == run_key
                    and cached.get('status') == 'complete' and isinstance(cached.get('elapsedSeconds'), (int, float))):
                cached.update(cacheHit=True, artifactPath=str(output_dir / 'audit.json'),
                              cachedResultElapsedSeconds=cached['elapsedSeconds'],
                              elapsedSeconds=round(monotonic() - started, 4))
                check()
                atomic_json(output_dir / 'audit.json', cached)
                emit({'stage': 'dialogue-audit', 'state': 'cached'})
                return cached
        except (OSError, ValueError):
            pass  # A corrupt cache is not a working result and can be rebuilt.
    emit({'stage': 'dialogue-audit', 'state': 'inventory', **inventory['summary']})
    status = 'complete'
    try:
        if rerun:
            with TemporaryDirectory(prefix='dialogue-audit-') as folder:
                retry_inventory(inventory, source_path, Path(folder), model, report.get('language', 'auto'),
                                max_windows, max_audio_ms, check, emit, threads=run_options['effectiveThreads'])
        check()
    except (AuditCancelled, TimeoutError, ValueError, OSError, subprocess.SubprocessError) as error:
        status = 'failed'
        if isinstance(error, AuditCancelled):
            status = 'cancelled'
        elif isinstance(error, TimeoutError):
            status = 'timeout'
        inventory['repairs']['status'] = status
        inventory['error'] = str(error)
    artifact = output_dir / ('audit.json' if status == 'complete' else f'attempt-{uuid4().hex}.json')
    inventory.update(status=status, cacheHit=False, artifactPath=str(artifact),
                     elapsedSeconds=round(monotonic() - started, 4))
    if status == 'complete':
        atomic_json(cache_path, inventory)
    atomic_json(artifact, inventory)
    emit({'stage': 'dialogue-audit', 'state': status, 'artifactPath': str(artifact)})
    return inventory
