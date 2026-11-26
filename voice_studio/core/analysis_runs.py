"""Collect bounded pipeline progress for the local Studio and graph MCP."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from time import monotonic
from uuid import uuid4


RETRY_GROUP = {
    "audio": "speech", "diarization": "speech", "whisper": "speech",
    "whisper-recovery": "speech", "dialogue-pauses": "speech",
    "audio-layers": "audio", "light-asd": "visual", "yunet": "visual",
    "speaker-review": "visual", "voice-profiles": "voice", "dialogue-audit": "voice", "person-layer": "people",
    "object-review": "object_review", "scene": "scene",
}
STATES = {"queued", "running", "done", "partial", "skipped", "failed", "pending", "invalidated"}
NUMBERS = ("elapsedMs", "chunk", "totalChunks", "windowStartMs", "windowsDone", "windowsTotal")


def stage_error(error):
    """Keep a readable worker error without exposing subprocess commands or tracebacks."""
    if isinstance(error, ValueError) or type(error).__name__ == "ModError":
        message = " ".join(str(error).split())[:500]
    elif isinstance(error, FileNotFoundError):
        message = "Не найден необходимый файл или исполняемый модуль."
    elif isinstance(error, TimeoutError) or type(error).__name__ == "TimeoutExpired":
        message = "Этап превысил время ожидания."
    else:
        message = "Этап завершился с ошибкой. Подробности в локальном журнале."
    return {"type": type(error).__name__, "message": message or "Этап завершился с ошибкой."}


class AnalysisRun:
    """Accumulate the latest stage states and retain a failure after a later chunk recovers."""

    def __init__(self, kind, run_id=None):
        self.run_id = run_id or uuid4().hex
        self.kind = kind
        self.lock = Lock()
        self.started = None
        self.ended = None
        self.status = "queued"
        self.error = None
        self.rows = {}
        self.stage_starts = {}
        self.layers = {}

    def start(self):
        """Start measuring the run's wall time when a worker or upload is ready."""
        with self.lock:
            self.started = monotonic()
            self.status = "running"

    def observe(self, event):
        """Copy status and bounded diagnostic fields, leaving partial reports on their own channel."""
        name, state = event.get("stage"), event.get("state")
        if not isinstance(name, str) or not 1 <= len(name) <= 64 or not isinstance(state, str) or state not in STATES:
            return
        with self.lock:
            if self.ended is not None or (name not in self.rows and len(self.rows) >= 64):
                return
            previous = self.rows.get(name, {})
            row = {"stage": name, "state": state}
            if name in RETRY_GROUP:
                row["retryGroup"] = RETRY_GROUP[name]
            for field in ("detail", "reason", "recoveryPass", "cacheLayer"):
                if isinstance(event.get(field), str):
                    row[field] = event[field][:500]
            if "detail" not in row and "detail" in previous:
                row["detail"] = previous["detail"]
            for field in NUMBERS:
                if type(event.get(field)) is int and event[field] >= 0:
                    row[field] = event[field]
            for field in ("reused", "cacheHit"):
                if type(event.get(field)) is bool:
                    row[field] = event[field]
            if state == "running":
                self.stage_starts.setdefault(name, monotonic())
            elif state != "queued":
                started = self.stage_starts.pop(name, None)
                if started is not None:
                    row.setdefault("elapsedMs", round((monotonic() - started) * 1000))
            if "lastFailure" in previous:
                row.update(lastFailure=previous["lastFailure"], failureCount=previous["failureCount"])
            if state == "failed":
                diagnostic = event.get("error")
                message = diagnostic.get("message") if isinstance(diagnostic, dict) else None
                if not isinstance(message, str) or not message:
                    message = row.get("reason") or "Этап завершился с ошибкой."
                failure = {"message": message[:500]}
                if isinstance(diagnostic, dict) and isinstance(diagnostic.get("type"), str):
                    failure["type"] = diagnostic["type"][:80]
                row.update(error=failure, lastFailure=failure, failureCount=row.get("failureCount", 0) + 1)
            self.rows[name] = row

    @contextmanager
    def execution(self):
        """Finish status on every exit while preserving the caller's original error."""
        self.start()
        try:
            yield
        except Exception as error:
            self.finish(error)
            raise
        else:
            self.finish()

    def set_layers(self, layers):
        """Attach the selective-run manifest while retaining separate worker outcomes."""
        if layers:
            with self.lock:
                self.layers = deepcopy(layers)

    def finish(self, error=None, *, layers=None):
        """Freeze timings and distinguish incomplete work from measured failures."""
        self.set_layers(layers)
        with self.lock:
            if self.ended is not None:
                return
            self.ended = monotonic()
            self.error = stage_error(error) if error is not None else None
            self.status = "failed" if error is not None else "done"
            for name, row in self.rows.items():
                if row["state"] == "running" and name in self.stage_starts:
                    row["elapsedMs"] = round((self.ended - self.stage_starts[name]) * 1000)
                if row["state"] in ("queued", "running"):
                    row["state"] = "pending"

    def snapshot(self):
        """Return a consistent status without holding a model or media lock."""
        with self.lock:
            now = self.ended if self.ended is not None else monotonic()
            stages = deepcopy(list(self.rows.values()))
            for row in stages:
                if row["state"] == "running" and row["stage"] in self.stage_starts:
                    row["elapsedMs"] = round((now - self.stage_starts[row["stage"]]) * 1000)
            result = {"runId": self.run_id, "kind": self.kind, "status": self.status,
                      "elapsedMs": round((now - self.started) * 1000) if self.started is not None else None,
                      "stages": stages, "failedStages": [row["stage"] for row in stages if row["state"] == "failed"]}
            if self.error:
                result["error"] = dict(self.error)
            if self.layers:
                result["layers"] = deepcopy(self.layers)
            return result


class AnalysisRuns:
    """Retain recent local uploads without writing transcripts, logs or media to disk."""

    def __init__(self):
        self.lock = Lock()
        self.runs = OrderedDict()

    @contextmanager
    def track(self):
        """Register a Studio run before reading the body, including errors during upload."""
        run = AnalysisRun("studio")
        with self.lock:
            for key, existing in list(self.runs.items()):
                if len(self.runs) < 32:
                    break
                if existing.snapshot()["status"] in ("done", "failed"):
                    del self.runs[key]
            if len(self.runs) >= 32:
                raise ValueError("Слишком много одновременных запусков анализа.")
            self.runs[run.run_id] = run
        with run.execution():
            yield run

    def status(self, run_id=None):
        """Return an addressed run or the latest local Studio analysis."""
        with self.lock:
            run = self.runs.get(run_id) if run_id else next(reversed(self.runs.values()), None)
        if run is None:
            if run_id:
                raise ValueError("Запуск анализа не найден в текущем сеансе студии.")
            return {"status": "idle", "stages": []}
        return run.snapshot()
