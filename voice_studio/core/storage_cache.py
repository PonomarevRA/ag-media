"""Coordinate regenerable storage with its readers across Studio and MCP processes."""

from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep

try:
    import fcntl
except ImportError:
    fcntl = None

LAST_USED = ".studio-cache-used"


@contextmanager
def cache_lock(group, *, exclusive=False, deadline=None):
    """Share one stable lock inode per cache group, outside the entries it protects."""
    if fcntl is None:
        yield not exclusive
        return
    group = Path(group)
    group.parent.mkdir(parents=True, exist_ok=True)
    with group.with_name(f".{group.name}.storage.lock").open("a") as handle:
        operation = fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH
        if deadline is not None:
            operation |= fcntl.LOCK_NB
        while True:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError('Истёк бюджет ожидания кэша.')
            try:
                fcntl.flock(handle, operation)
                break
            except BlockingIOError:
                if exclusive:
                    yield False
                    return
                if deadline is None:
                    raise
                sleep(min(.05, max(0, deadline - monotonic())))
        yield True


@contextmanager
def use_cache(group, entry, *, deadline=None):
    """Record cache activity and keep the entry alive throughout its use."""
    with cache_lock(group, deadline=deadline):
        if not entry.is_symlink():
            entry.mkdir(parents=True, exist_ok=True)
        mark_cache_used(entry)
        try:
            yield
        finally:
            mark_cache_used(entry)


def mark_cache_used(entry):
    """Record an actual use even if the operation never reaches normal teardown."""
    marker = entry / LAST_USED
    if entry.is_dir() and not entry.is_symlink() and not marker.is_symlink():
        marker.touch()
