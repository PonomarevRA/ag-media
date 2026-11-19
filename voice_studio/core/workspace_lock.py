"""Cross-platform, process-lifetime ownership for one writable Studio workspace."""

from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def workspace_lock(root):
    """Hold a nonblocking OS lock in root until the caller finishes.

    root is an existing workspace directory. BlockingIOError means another process
    owns it. The lock file stays in place so a second inode cannot bypass ownership.
    """
    path = Path(root) / "studio.lock"
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise BlockingIOError("Workspace already open") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
