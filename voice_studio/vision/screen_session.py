"""Bound private screenshot results by both session age and generation count."""

from collections import OrderedDict
import json
from pathlib import Path
import shutil
from threading import Lock, Timer
from time import monotonic, time

from voice_paths import temporary_directory

MAX_SESSIONS = 8
MAX_GENERATIONS = 6
SESSION_TTL_SECONDS = 300
MAX_LAYOUT_BYTES = 512 * 1024


class ScreenSessions:
    """Own temporary per-session results in one MCP process."""

    def __init__(self):
        self.sessions = OrderedDict()
        self.lock = Lock()
        self.timer = None
        self.scratch = None
        self.next_sweep = None

    def _prune(self, now):
        """Remove all sessions whose most recent request has expired."""
        for name, (used, folder, _) in list(self.sessions.items()):
            if now - used >= SESSION_TTL_SECONDS:
                folder.cleanup()
                del self.sessions[name]

    def _schedule(self, now):
        """Schedule the oldest expiry rather than polling or keeping one timer per session."""
        if self.timer:
            self.timer.cancel()
        self.timer = None
        deadlines = [] if self.next_sweep is None else [self.next_sweep]
        if self.sessions:
            deadlines.append(next(iter(self.sessions.values()))[0] + SESSION_TTL_SECONDS)
        if deadlines:
            self.timer = Timer(max(.01, min(deadlines) - now), self.expire)
            self.timer.daemon = True
            self.timer.start()

    def expire(self):
        """Delete idle temporary artifacts even when no more requests arrive."""
        with self.lock:
            now = monotonic()
            self._prune(now)
            self._scavenge(now)
            self._schedule(now)

    def _scavenge(self, now):
        """Periodically remove private directories abandoned by stopped MCP processes."""
        if self.scratch is None or (self.next_sweep is not None and now < self.next_sweep):
            return
        self.next_sweep = now + SESSION_TTL_SECONDS
        cutoff = time() - 2 * SESSION_TTL_SECONDS
        owned = {Path(folder.name) for _, folder, _ in self.sessions.values()}
        for entry in self.scratch.glob("screen-session-*"):
            try:
                if entry not in owned and not entry.is_symlink() and entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry)
            except OSError:
                continue

    def get(self, key, now=None, *, create=True):
        """Refresh a session and create it on demand unless this is a read-only lookup."""
        live_clock = now is None
        now = monotonic() if live_clock else now
        with self.lock:
            self._prune(now)
            if key is None:
                return None, None
            if key in self.sessions:
                _, folder, results = self.sessions.pop(key)
            else:
                if not create:
                    return None, None
                if len(self.sessions) >= MAX_SESSIONS:
                    _, (_, previous, _) = self.sessions.popitem(last=False)
                    previous.cleanup()
                folder, results = temporary_directory(prefix="screen-session-"), OrderedDict()
            path = Path(folder.name)
            path.touch()
            self.sessions[key] = now, folder, results
            if live_clock:
                self.scratch = path.parent
                self._scavenge(now)
                self._schedule(now)
            return path, results


def retain(results, key, value):
    """Keep at most the most recent exact-image generation keys in memory."""
    results[key] = value
    results.move_to_end(key)
    while len(results) > MAX_GENERATIONS:
        results.popitem(last=False)


def cached_layout(path):
    """Return a previously analyzed pixel scope and refresh generation order."""
    try:
        if path.stat().st_size > MAX_LAYOUT_BYTES:
            return None
        value = json.loads(path.read_bytes())
        if isinstance(value, dict) and "structure" in value:
            path.touch()
            return value
    except (OSError, ValueError):
        return None
    return None


def save_layout(path, value):
    """Publish only JSON results and evict older generations of the same session."""
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_LAYOUT_BYTES:
        return
    staged = path.with_suffix(".part")
    staged.write_bytes(data)
    staged.replace(path)
    previous = sorted(path.parent.glob("*.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    for item in previous[MAX_GENERATIONS:]:
        item.unlink(missing_ok=True)
