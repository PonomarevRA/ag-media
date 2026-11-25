"""Report real analysis boundaries to optionally subscribed media clients."""

from contextlib import contextmanager
from threading import Lock
from time import monotonic

MAX_ACTIVITY_CUES = 64


def ignore_progress(event):
    """Keep internal and existing JSON callers independent of the progress transport."""


def serialized_progress(emit):
    """Keep progress records intact when independent stages overlap."""
    lock = Lock()

    def notify(event):
        """Deliver one complete event under the same lock as sibling stages."""
        with lock:
            emit(event)

    return notify


@contextmanager
def queued_stage_slot(slot, emit, name, *, already_started=False):
    """Wait for a bounded video worker to finish and expose contention as progress."""
    queued = not slot.acquire(blocking=False)
    if queued:
        emit({"stage": name, "state": "queued",
              "reason": "Ждём завершения другого этапа анализа видео."})
        slot.acquire()
    try:
        if queued and already_started:
            emit({"stage": name, "state": "running"})
        yield
    finally:
        slot.release()


@contextmanager
def stage(emit, name):
    """Emit the start and outcome of one stage, including its measured wall time."""
    started = monotonic()
    status = {"state": "done"}
    emit({"stage": name, "state": "running"})
    try:
        yield status
    except (BrokenPipeError, ConnectionResetError):
        status["state"] = "running"
        raise
    except Exception:
        status["state"] = "failed"
        raise
    finally:
        if status["state"] != "running":
            emit({"stage": name, **status, "elapsedMs": round((monotonic() - started) * 1000)})

