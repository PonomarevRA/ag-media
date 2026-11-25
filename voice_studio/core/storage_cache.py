"""Coordinate regenerable storage with its readers across Studio and MCP processes."""

from contextlib import ExitStack, contextmanager
from os import name as platform
from pathlib import Path
from time import monotonic, sleep

try:
    import fcntl
except ImportError:
    fcntl = None

if platform == "nt":
    from voice_studio.core.storage_winlock import windows_lease
else:
    windows_lease = None

LAST_USED = ".studio-cache-used"


@contextmanager
def platform_lease(handle, *, exclusive, nonblocking):
    """Use the operating system's shared and exclusive lease implementation."""
    if windows_lease is not None:
        with windows_lease(handle, exclusive=exclusive, nonblocking=nonblocking):
            yield
        return
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if nonblocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(handle, operation)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def cache_lock(group, *, exclusive=False, deadline=None):
    """Share one stable lock inode per cache group, outside the entries it protects."""
    if fcntl is None and windows_lease is None:
        yield not exclusive
        return
    group = Path(group)
    group.parent.mkdir(parents=True, exist_ok=True)
    with group.with_name(f".{group.name}.storage.lock").open("a") as handle, ExitStack() as lease:
        # Only retry acquisition. A failure in the caller must propagate as-is.
        while True:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError('Истёк бюджет ожидания кэша.')
            try:
                lease.enter_context(platform_lease(handle, exclusive=exclusive,
                                                   nonblocking=exclusive or deadline is not None))
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
