"""Serialize writers to one source's private voice registry without locking UI parsing."""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from threading import Lock
from time import monotonic, sleep

try:
    import fcntl
except ImportError:
    fcntl = None

FALLBACK_LOCK = Lock()


@contextmanager
def registry_writer(path, *, timeout=120):
    """Hold a separate bounded writer lease; retention reader locks do not serialize updates."""
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + timeout
    if fcntl is None:
        if not FALLBACK_LOCK.acquire(timeout=timeout):
            raise TimeoutError('Голосовой профиль занят другой задачей.')
        try:
            yield
        finally:
            FALLBACK_LOCK.release()
        return
    with path.with_name(path.name + '.writer.lock').open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise TimeoutError('Голосовой профиль занят другой задачей.') from None
                sleep(min(.05, max(0, deadline - monotonic())))
        yield


def serialize_voice_windows(operation):
    """Protect the complete window/registry checkpoint sequence for one durable analysis."""
    @wraps(operation)
    def run(*args, **kwargs):
        revolver = kwargs.get('revolver')
        directory = getattr(revolver, 'directory', None)
        path = directory / 'voice-registry' if directory is not None and kwargs.get('speakers', True) else None
        with registry_writer(path):
            return operation(*args, **kwargs)
    return run


def serialize_voice_file(operation):
    """Protect short-file registry read/modify/write while leaving in-memory registries owned by callers."""
    @wraps(operation)
    def run(*args, **kwargs):
        with registry_writer(kwargs.get('cache_path')):
            return operation(*args, **kwargs)
    return run
