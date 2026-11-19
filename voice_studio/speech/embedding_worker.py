"""Own one lazy native speaker process, bounded serial admission and its deadline."""

import atexit
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import sys
from threading import BoundedSemaphore, Lock, Thread, Timer
from time import monotonic

from voice_paths import PROJECT_ROOT
from voice_studio.speech.embedding_protocol import (
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_STDERR_BYTES, validate_spans, validate_vectors,
)


class EmbeddingWorkerError(subprocess.SubprocessError):
    """A native request failed; only a later request may start a replacement process."""


@dataclass
class _Child:
    process: subprocess.Popen
    responses: Queue = field(default_factory=lambda: Queue(maxsize=1))
    stderr: bytearray = field(default_factory=bytearray)
    stderr_lock: Lock = field(default_factory=Lock)


def _stdout_reader(child):
    """Read one bounded JSON line at a time without growing a producer-side queue."""
    try:
        while True:
            line = child.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
            if not line:
                child.responses.put_nowait(EmbeddingWorkerError('Embedding worker closed stdout'))
                return
            if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b'\n'):
                child.responses.put_nowait(EmbeddingWorkerError('Embedding worker response exceeds limit'))
                return
            child.responses.put_nowait(line)
    except (OSError, ValueError, Full) as error:
        try:
            child.responses.put_nowait(EmbeddingWorkerError(str(error)))
        except Full:
            pass


def _stderr_reader(child):
    """Drain native diagnostics while retaining only a fixed-size tail."""
    try:
        while chunk := child.process.stderr.read(4096):
            with child.stderr_lock:
                child.stderr.extend(chunk)
                del child.stderr[:-MAX_STDERR_BYTES]
    except (OSError, ValueError):
        pass


class EmbeddingWorker:
    """Serialize native calls with a bounded waiting queue and one overall deadline.

    command is an optional process launcher for protocol/failure tests. No inference
    retry occurs after a timeout, malformed reply or crash. close() also interrupts
    a request already running; idle retirement never races an active request.
    """

    def __init__(self, command=None, *, idle_seconds=60, queue_limit=8):
        """Create lazy ownership; queue_limit bounds waiting callers, excluding the active call."""
        if not math.isfinite(idle_seconds) or idle_seconds <= 0 or type(queue_limit) is not int or queue_limit < 0:
            raise ValueError('Invalid embedding worker limits')
        self.command = command or [sys.executable, str(PROJECT_ROOT / 'speaker_review.py'), '--embedding-worker']
        self.idle_seconds = idle_seconds
        self._admission = BoundedSemaphore(queue_limit + 1)
        self._serial = Lock()
        self._state = Lock()
        self._child = None
        self._timer = None
        self._active = False
        self._sequence = 0
        self._starts = 0
        self._requests = 0
        self._model_loads = 0
        self._identity = None
        self._stderr_tail = ''

    def _start_locked(self):
        """Spawn a clean native-only child; every model choice arrives in its request."""
        environment = dict(os.environ)
        environment.pop('VOICE_MOD_SNAPSHOT', None)
        environment['VOICE_EMBEDDING_PARENT_PID'] = str(os.getpid())
        # Buffered reads avoid a syscall per byte when a full reference batch emits JSON.
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment, bufsize=-1)
        child = _Child(process)
        self._child = child
        self._starts += 1
        Thread(target=_stdout_reader, args=(child,), daemon=True).start()
        Thread(target=_stderr_reader, args=(child,), daemon=True).start()
        return child

    def _retire_locked(self, child):
        """Reap this exact owned process with bounded termination, then retain diagnostics."""
        if self._child is not child:
            return
        self._child = None
        if child.process.poll() is None:
            child.process.terminate()
            try:
                child.process.wait(timeout=.4)
            except subprocess.TimeoutExpired:
                child.process.kill()
                child.process.wait(timeout=.4)
        for stream in (child.process.stdin, child.process.stdout, child.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        with child.stderr_lock:
            self._stderr_tail = child.stderr.decode('utf-8', errors='replace')

    def _idle_close(self, child):
        """Release idle native memory only if no request has claimed this child."""
        with self._state:
            if not self._active and self._child is child:
                self._retire_locked(child)

    def close(self):
        """Interrupt and reap the owned child; the next explicit request may start anew."""
        with self._state:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._child:
                self._retire_locked(self._child)

    def stats(self):
        """Return diagnostic counters without exposing or caching voice vectors."""
        with self._state:
            child = self._child
            return {'pid': child.process.pid if child and child.process.poll() is None else None,
                    'starts': self._starts, 'requests': self._requests, 'model_loads': self._model_loads,
                    'identity': self._identity, 'stderr_tail': self._stderr_tail}

    def extract(self, wav, spans, model_path, identity, *, timeout=35):
        """Compute requested references; timeout seconds includes admission and serial wait."""
        started = monotonic()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Embedding timeout must be finite and positive')
        validate_spans(spans)
        request = {'wav': str(Path(wav).resolve()), 'spans': spans,
                   'model_path': str(model_path), 'identity': identity}
        payload_size = len(json.dumps(request, allow_nan=False).encode('utf-8'))
        if payload_size > MAX_REQUEST_BYTES - 128:
            raise ValueError('Embedding request exceeds limit')
        if not self._admission.acquire(blocking=False):
            raise EmbeddingWorkerError('Embedding worker queue is full')
        acquired = False
        try:
            remaining = timeout - (monotonic() - started)
            acquired = self._serial.acquire(timeout=max(0, remaining))
            if not acquired:
                raise subprocess.TimeoutExpired(self.command, timeout)
            remaining = timeout - (monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return self._request(request, spans, started + timeout, timeout)
        finally:
            if acquired:
                self._serial.release()
            self._admission.release()

    def _request(self, request, spans, deadline, timeout):
        """Exchange exactly one correlated request; never transparently replay failed work."""
        child = None
        try:
            with self._state:
                self._active = True
                if self._timer:
                    self._timer.cancel()
                child = self._child
                if child and child.process.poll() is not None:
                    self._retire_locked(child)
                    child = None
                child = child or self._start_locked()
                self._sequence += 1
                request['id'] = self._sequence
            line = json.dumps(request, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
            # A stalled child must not block the caller inside a full stdin pipe.
            Thread(target=self._write, args=(child, line), daemon=True).start()
            try:
                response = child.responses.get(timeout=max(0, deadline - monotonic()))
            except Empty as error:
                raise subprocess.TimeoutExpired(self.command, timeout) from error
            if isinstance(response, Exception):
                raise response
            value = json.loads(response)
            if not isinstance(value, dict) or value.get('id') != request['id']:
                raise EmbeddingWorkerError('Uncorrelated embedding worker response')
            if 'error' in value:
                raise EmbeddingWorkerError(str(value['error'])[:2000])
            vectors = validate_vectors(value.get('vectors'), spans)
            with self._state:
                if self._child is not child:
                    raise EmbeddingWorkerError('Embedding worker was closed during request')
                self._requests += 1
                self._model_loads = value.get('model_loads', 0)
                self._identity = request['identity']
            return vectors
        except BaseException:
            with self._state:
                if child:
                    self._retire_locked(child)
            raise
        finally:
            with self._state:
                self._active = False
                if child and self._child is child:
                    self._timer = Timer(self.idle_seconds, self._idle_close, args=(child,))
                    self._timer.daemon = True
                    self._timer.start()

    @staticmethod
    def _write(child, line):
        """Complete a bounded write off the deadline-owning caller thread."""
        try:
            remaining = memoryview(line)
            while remaining:
                written = child.process.stdin.write(remaining)
                if not written:
                    raise BrokenPipeError('Embedding worker stdin closed')
                remaining = remaining[written:]
            child.process.stdin.flush()
        except (OSError, ValueError) as error:
            try:
                child.responses.put_nowait(EmbeddingWorkerError(str(error)))
            except Full:
                pass


_WORKER = None
_OWNER_PID = os.getpid()
_WORKER_LOCK = Lock()


def get_worker():
    """Get the lazy process-local transport, with operator-configured idle/queue bounds."""
    global _WORKER, _OWNER_PID
    with _WORKER_LOCK:
        if _OWNER_PID != os.getpid():
            # An inherited Popen belongs to the parent host; never terminate it here.
            _WORKER = None
            _OWNER_PID = os.getpid()
        if _WORKER is None:
            _WORKER = EmbeddingWorker(idle_seconds=float(os.environ.get('VOICE_SPEAKER_EMBEDDING_IDLE_SECONDS', '60')),
                                      queue_limit=int(os.environ.get('VOICE_SPEAKER_EMBEDDING_QUEUE_LIMIT', '8')))
        return _WORKER


def close_worker():
    """Release the existing process without creating a worker as a shutdown side effect."""
    if _WORKER is not None and _OWNER_PID == os.getpid():
        _WORKER.close()


atexit.register(close_worker)
