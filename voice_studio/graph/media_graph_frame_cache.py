"""Bounded process-local reuse of undecorated source PNGs and color measurements."""

from collections import OrderedDict
import json
from threading import Lock


class FrameCache:
    """
    Keep at most eight MiB of PNG/palette payload across eight frames.

    Python object overhead is outside this payload budget. Graph overlays are excluded.
    """

    def __init__(self, max_entries=8, max_bytes=8 * 1024 * 1024):
        """
        Create an empty bounded cache with no disk I/O.

        Inputs max_entries/max_bytes must be positive integers; result is empty state.
        Invalid limits raise ValueError.
        """
        if type(max_entries) is not int or type(max_bytes) is not int or min(max_entries, max_bytes) < 1:
            raise ValueError("Frame cache limits must be positive integers.")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes_used = 0
        self.lock = Lock()

    def get(self, key):
        """
        Read an entry and promote its recent-use position.

        Input key is a hashable source/frame identity; result is PNG/palette or None.
        Palette copies isolate callers. Unhashable keys raise TypeError; no I/O occurs.
        """
        with self.lock:
            item = self.entries.get(key)
            if item is None:
                return None
            self.entries.move_to_end(key)
            png, palette_json = item
        palette = json.loads(palette_json) if palette_json is not None else None
        return png, palette

    def put(self, key, png, palette):
        """
        Store key's immutable PNG and JSON palette; oversized entries are skipped.

        Inputs are source-signature keys, PNG bytes and optional JSON-safe palette.
        Result is None; serialization errors propagate and no disk files are written.
        """
        palette_json = json.dumps(palette, ensure_ascii=False).encode() if palette is not None else None
        size = len(png) + (len(palette_json) if palette_json is not None else 0)
        if size > self.max_bytes:
            return
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.bytes_used -= self._size(previous)
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes_used + size > self.max_bytes):
                _, removed = self.entries.popitem(last=False)
                self.bytes_used -= self._size(removed)
            self.entries[key] = (png, palette_json)
            self.bytes_used += size

    @staticmethod
    def _size(item):
        """
        Count one stored item's payload bytes.

        Input item is (PNG bytes, optional palette JSON bytes); result is an integer.
        Internal callers supply validated tuples; no I/O or expected errors occur.
        """
        png, palette_json = item
        return len(png) + (len(palette_json) if palette_json is not None else 0)
