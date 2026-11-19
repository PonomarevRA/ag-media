"""Atomic, process-locked media mod configuration outside the application bundle."""

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading

from .errors import ModError
from .schema import identifier, json_bytes

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def empty_state():
    """Return an independent v1 state with implicit bundled defaults."""
    return {"apiVersion": 1, "revision": 0, "enabled": {}, "selections": {}, "settings": {}, "extensions": {}}


class StateStore:
    """Read every request freshly and serialize read-modify-write across processes."""

    def __init__(self, path):
        """Bind a local state path; construction does not create files."""
        self.path = Path(path)
        with _LOCKS_GUARD:
            self.lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())

    def read(self):
        """Return validated state, raising ModError rather than activating from corrupt JSON."""
        try:
            if self.path.stat().st_size > 262144:
                raise ModError("Mod state exceeds 256 KiB")
            with self.path.open("rb") as handle:
                payload = handle.read(262145)
            if len(payload) > 262144:
                raise ModError("Mod state exceeds 256 KiB")
            state = json.loads(payload)
        except FileNotFoundError:
            return empty_state()
        except (OSError, ValueError, RecursionError) as error:
            raise ModError(f"Cannot read mod state: {error}") from error
        if isinstance(state, dict):
            state.setdefault("extensions", {})  # Existing v1 configurations predate extension metadata.
        if (not isinstance(state, dict) or set(state) != set(empty_state())
                or type(state["apiVersion"]) is not int or state["apiVersion"] != 1
                or type(state["revision"]) is not int or state["revision"] < 0):
            raise ModError("Invalid mod state contract")
        if not isinstance(state["extensions"], dict) or len(json_bytes(state["extensions"])) > 65536:
            raise ModError("State extensions must be a JSON object no larger than 64 KiB")
        for key in ("enabled", "selections", "settings"):
            if not isinstance(state[key], dict) or len(state[key]) > 128:
                raise ModError("Invalid mod state collection")
            for mod_id, value in state[key].items():
                identifier(mod_id)
                if key == "enabled" and not (value is False or isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)):
                    raise ModError("Invalid enabled package fingerprint")
                if key == "selections":
                    identifier(value, "selected mod id")
                if key == "settings" and not isinstance(value, dict):
                    raise ModError("Invalid persisted settings")
        json_bytes(state)
        return state

    @contextmanager
    def transaction(self):
        """Lock a mutation until its atomic replacement completes; exceptions leave old state."""
        with self.lock:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(".lock")
            with lock_path.open("a+b") as handle:
                if os.name == "nt":
                    import msvcrt
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    state = deepcopy(self.read())
                    yield state
                    state["revision"] += 1
                    self._write(state)
                finally:
                    if os.name == "nt":
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, state):
        """Flush a unique sibling temporary file before replacing the complete state."""
        extensions = state.get("extensions", {})
        if not isinstance(extensions, dict) or len(json_bytes(extensions)) > 65536:
            raise ModError("State extensions must be a JSON object no larger than 64 KiB")
        payload = json_bytes(state)
        if len(payload) > 262144:
            raise ModError("Mod state exceeds 256 KiB")
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".mod-state-", delete=False) as handle:
                staged = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, self.path)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
