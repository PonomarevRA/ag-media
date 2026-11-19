"""Keep the latest successful set of scroll mosaics in the Studio's private cache."""

from contextlib import contextmanager
from pathlib import Path
import re
import shutil
from threading import Lock
from uuid import uuid4

MAX_PREVIEWS = 16
MAX_PREVIEW_BYTES = 4 * 1024 * 1024
PREVIEW_KEY = re.compile(r"[a-f0-9]{32}")
PREVIEW_PATH = re.compile(r"/api/scene-scroll-preview/([a-f0-9]{32})/([0-9]{1,2})\.png")
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


class ScrollPreviews:
    """Keep the last working analysis and in-flight writes without relying on browser timers."""

    def __init__(self, root):
        self.root = Path(root)
        self.active = set()
        self.lock = Lock()
        self.started = 0
        self.latest = 0

    @contextmanager
    def analysis(self):
        """Publish one generation, replacing older images only after a successful analysis with mosaics."""
        key = uuid4().hex
        folder = self.root / key
        with self.lock:
            self.active.add(key)
            self.started += 1
            order = self.started
        published = completed = False

        def publish(index, source):
            """Copy one bounded worker PNG before the worker's scratch directory is removed."""
            nonlocal published
            if not 0 <= index < MAX_PREVIEWS:
                return None
            try:
                with source.open("rb") as stream:
                    image = stream.read(MAX_PREVIEW_BYTES + 1)
                if len(image) > MAX_PREVIEW_BYTES or not image.startswith(PNG_HEADER):
                    return None
                folder.mkdir(mode=0o700, parents=True, exist_ok=True)
                target = folder / f"{index}.png"
                staged = target.with_suffix(".part")
                staged.write_bytes(image)
                staged.replace(target)
            except OSError:
                return None
            published = True
            return f"/api/scene-scroll-preview/{key}/{index}.png"

        try:
            yield publish
            completed = True
        finally:
            with self.lock:
                self.active.discard(key)
                if not (completed and published):
                    shutil.rmtree(folder, ignore_errors=True)
                elif order > self.latest:
                    # A late older request may keep its own images but must not replace a newer result.
                    self.latest = order
                    self._remove_previous(key)

    def _remove_previous(self, newest):
        """After a successful replacement, clear older and abandoned private generations."""
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return
        for folder in entries:
            if (PREVIEW_KEY.fullmatch(folder.name) and folder.name != newest
                    and folder.name not in self.active and not folder.is_symlink()):
                shutil.rmtree(folder, ignore_errors=True)

    def path(self, key, index):
        """Look up a slot by an opaque token, including results from the previous server process."""
        if not PREVIEW_KEY.fullmatch(key) or not 0 <= index < MAX_PREVIEWS:
            return None
        return self.root / key / f"{index}.png"

    def read(self, key, index):
        """Read the requested page if it still belongs to the current or in-flight generation."""
        path = self.path(key, index)
        try:
            return path.read_bytes() if path is not None else None
        except OSError:
            return None
