"""Plan CPU and memory for concurrent local media inference."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
from voice_paths import RUNTIME_DIR, ensure_temporary_root
from threading import Lock
from time import monotonic, sleep

try:
    import fcntl
except ImportError:
    fcntl = None

SLOT = RUNTIME_DIR / ".media-analysis-slot"
LOCAL_SLOT = Lock()
IN_BATCH = ContextVar("media_analysis_budget_held", default=False)


@dataclass(frozen=True)
class MediaResources:
    """Describe the available CPU threads and RAM for one media analysis."""

    threads: int
    memory_mib: int

    def speech(self, parallel):
        """Reserve two Sherpa workers during parallel ASR and space for live work."""
        parallel = parallel and self.threads >= 6 and self.memory_mib >= 4096
        reserve = 2 if self.threads >= 6 else 0
        threads = min(8, max(1, self.threads - reserve - (2 if parallel else 0)))
        return threads, parallel


def resource_override(name, detected):
    """Apply a positive ceiling without requesting more than the detected resources."""
    value = os.environ.get(name, "")
    if len(value) > 8 or not value.isascii() or not value.isdecimal():
        return detected
    return min(detected, int(value)) if int(value) else detected


def host_threads():
    """Respect process CPU affinity where the operating system exposes it."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def host_memory_mib():
    """Read physical memory, using a conservative allocation if discovery fails."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except (AttributeError, OSError, ValueError):
        return 4096


def pilot_resources():
    """Read the opt-in resource plan used by offline analysis stages."""
    if os.environ.get("VOICE_RESOURCE_PROFILE") != "throughput":
        return None
    return MediaResources(resource_override("VOICE_CPU_BUDGET", host_threads()),
                          resource_override("VOICE_MEMORY_MIB", max(1, host_memory_mib())))


def stage_threads(default):
    """Use the stage preset within a configured pilot CPU allowance."""
    resources = pilot_resources()
    return min(default, resources.threads) if resources is not None else default


@contextmanager
def batch_slot(*, deadline=None):
    """Share a batch slot across Studio and MCP when both opt into the pilot."""
    deadline = min(monotonic() + 120, deadline) if deadline is not None else monotonic() + 120
    if fcntl is None:
        if not LOCAL_SLOT.acquire(timeout=max(0, deadline - monotonic())):
            raise ValueError("Анализ медиа занят другой задачей.")
        try:
            yield
        finally:
            LOCAL_SLOT.release()
        return
    ensure_temporary_root()
    SLOT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with SLOT.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise ValueError("Анализ медиа занят другой задачей.") from None
                sleep(min(.1, max(0, deadline - monotonic())))
        yield


def batch_media_job(operation):
    """Keep one heavy pilot batch active while allowing nested stages in that job."""
    @wraps(operation)
    def run(*args, **kwargs):
        """Reuse the same allocation for nested stage entry points."""
        if IN_BATCH.get() or pilot_resources() is None or kwargs.get("allow_empty") is True:
            return operation(*args, **kwargs)
        with batch_slot():
            token = IN_BATCH.set(True)
            try:
                return operation(*args, **kwargs)
            finally:
                IN_BATCH.reset(token)
    return run


@batch_media_job
def run_batch_stage(operation, *args, allow_empty=False, **kwargs):
    """Guard one heavy stage rather than holding the shared slot through unrelated enrichment.

    Small live-capture calls retain the existing allow_empty bypass. Nested stages reuse
    the same resource allocation; native operations keep their existing CPU/model options.
    """
    return operation(*args, **kwargs)


def run_bounded_batch_stage(operation, *args, deadline, **kwargs):
    """Admit native work within the caller's monotonic wall-clock budget.

    The operation must enforce the remaining budget itself after admission. UI OCR
    uses its independent lane; heavy audio/video stages share this pilot slot.
    """
    if monotonic() >= deadline:
        raise TimeoutError('Истёк бюджет проверки диалога.')
    if IN_BATCH.get() or pilot_resources() is None:
        return operation(*args, **kwargs)
    with batch_slot(deadline=deadline):
        if monotonic() >= deadline:
            raise TimeoutError('Истёк бюджет проверки диалога.')
        token = IN_BATCH.set(True)
        try:
            return operation(*args, **kwargs)
        finally:
            IN_BATCH.reset(token)
