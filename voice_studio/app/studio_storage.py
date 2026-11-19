"""Inspect this Studio workspace and expire its explicitly registered disposable caches."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
from time import time
import logging
import re
import shutil
import sqlite3
import stat

from voice_studio.graph.media_graph_store import IDENTIFIER, MediaGraphStore
from voice_studio.core.storage_cache import LAST_USED, cache_lock
from voice_paths import DEPENDENCIES_DIR, MODELS_DIR, PROJECT_DATA, RUNTIME_DIR, workspace_temp

DAY = 86400
CACHE_DAYS = 14
TEMP_DAYS = 1
EXPORT_DAYS = 30
SWEEP_SECONDS = 3600
MAX_VISIBLE = 300
LOGGER = logging.getLogger(__name__)
WORKSPACE_GROUPS = ("media", "checkpoints", "media-graphs")
PROJECT_GROUPS = ("models", "runtime", "test-assets", "exports")
KEY = re.compile(r"[a-f0-9]{64}")
TEMP_NAME = re.compile(r"upload-[a-z0-9_]{8}")
CACHE_GROUPS = (("analysis-checkpoints", "analysis-cache", KEY, CACHE_DAYS),
                ("temporary-media", "temporary-media", TEMP_NAME, TEMP_DAYS))


@dataclass(frozen=True)
class Entry:
    """One explicitly registered cache or reviewable Studio export."""

    key: str
    path: Path
    group: Path
    kind: str
    days: int
    automatic: bool


def file_details(path):
    """Sum regular files without following links and fingerprint their metadata."""
    size = 0
    activity = 0
    signature = sha256()
    pending = [path]
    while pending:
        current = pending.pop()
        info = current.lstat()
        signature.update(repr((str(current.relative_to(path)), info.st_dev, info.st_ino, info.st_mode,
                               info.st_size, info.st_mtime_ns, info.st_ctime_ns)).encode())
        if stat.S_ISDIR(info.st_mode):
            activity = max(activity, info.st_mtime)
            pending.extend(sorted(current.iterdir(), reverse=True))
        elif stat.S_ISREG(info.st_mode):
            activity = max(activity, info.st_atime, info.st_mtime)
            if current.name != LAST_USED:
                size += info.st_size
    return size, activity, signature.hexdigest()


def regular_entry(path):
    """Accept only real files and directories under their configured canonical path."""
    return path.exists() and not path.is_symlink() and path.resolve() == path


class StudioStorage:
    """Scope inspection and cleanup to the current Studio data locations."""

    def __init__(self, workspace, project_data):
        self.workspace = Path(workspace).resolve()
        self.project_data = Path(project_data).resolve()

    def cache_entries(self):
        """Yield the reserved directories with an explicit regeneration and lease contract."""
        external = workspace_temp(self.workspace)
        for root, prefix in ((external, ""), (self.workspace, "legacy-")):
            for directory, kind, names, days in CACHE_GROUPS:
                group = root / directory
                if regular_entry(group) and group.is_dir():
                    for path in group.iterdir():
                        if names.fullmatch(path.name) and regular_entry(path) and path.is_dir():
                            yield Entry(f"{prefix}{kind}/{path.name}", path, group, kind, days, True)
        runtimes = [(self.project_data / "runtime", "legacy-")]
        if self.project_data == PROJECT_DATA:
            runtimes.insert(0, (RUNTIME_DIR, ""))
        for root, prefix in runtimes:
            for name in ("audio-scene", "scene-ocr"):
                group = root / name
                path = group / "swift-modules"
                if regular_entry(path) and path.is_dir():
                    yield Entry(f"{prefix}swift-cache/{name}", path, group, "swift-cache", CACHE_DAYS, True)

    def usage_groups(self):
        """Count each configured storage location under its existing UI category."""
        paths = [(kind, self.workspace / kind) for kind in WORKSPACE_GROUPS]
        paths.extend((kind, self.project_data / kind) for kind in PROJECT_GROUPS)
        external = workspace_temp(self.workspace)
        for directory, _, _, _ in CACHE_GROUPS:
            paths.extend(((directory, self.workspace / directory), (directory, external / directory)))
        if self.project_data == PROJECT_DATA:
            paths.extend((("models", MODELS_DIR), ("runtime", DEPENDENCIES_DIR), ("runtime", RUNTIME_DIR)))
        totals = {}
        seen = set()
        for kind, path in paths:
            if path not in seen and regular_entry(path):
                seen.add(path)
                try:
                    size, _, _ = file_details(path)
                except OSError:
                    continue
                totals[kind] = totals.get(kind, 0) + size
        return [{"kind": kind, "bytes": size} for kind, size in sorted(totals.items(), key=lambda item: -item[1])]

    def entries(self):
        """Add user-selected exports to the catalog without adding them to automatic cleanup."""
        yield from self.cache_entries()
        group = self.project_data / "exports"
        if regular_entry(group) and group.is_dir():
            for path in group.iterdir():
                if not path.name.startswith(".") and regular_entry(path) and (path.is_file() or path.is_dir()):
                    key = sha256(path.name.encode("utf-8", errors="surrogateescape")).hexdigest()
                    yield Entry(f"export/{key}", path, group, "export", EXPORT_DAYS, False)

    def graph_sources(self):
        """Read saved original paths, including sources referenced by partial snapshots."""
        store = MediaGraphStore(self.workspace / "media-graphs")
        sources = set()
        for path in store.root.glob("*.sqlite"):
            if not IDENTIFIER.fullmatch(path.stem) or not regular_entry(path):
                continue
            with store.database(path.stem) as database:
                record = store.manifest(database).get("source")
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError("The saved graph has an invalid original path.")
            sources.add(Path(record["path"]))
        return sources

    def export_references(self):
        """Fail closed for export deletion if a saved graph cannot be inspected."""
        try:
            return self.graph_sources()
        except (OSError, ValueError, sqlite3.Error):
            return None

    def describe(self, entry, now, sources=()):
        """Compute age, activity source and a token for the exact inspected generation."""
        size, activity, revision = file_details(entry.path)
        marker = entry.path / LAST_USED
        tracked = entry.automatic and regular_entry(marker) and marker.is_file()
        if tracked:
            activity = max(activity, marker.stat().st_mtime)
        reference_unknown = entry.kind == "export" and sources is None
        referenced = entry.kind == "export" and sources is not None and any(
            source.is_relative_to(entry.path) for source in sources)
        return {"id": entry.key, "referenced": referenced, "referenceUnknown": reference_unknown,
                "label": entry.path.name if entry.kind != "swift-cache" else entry.group.name,
                "kind": entry.kind, "bytes": size, "lastUsed": round(activity),
                "activitySource": "studio" if tracked else "filesystem",
                "candidate": not (referenced or reference_unknown) and activity <= now - entry.days * DAY,
                "automatic": entry.automatic and tracked, "revision": revision}

    def inventory(self):
        """Return workspace usage and candidates ordered by measured activity."""
        now = time()
        items = []
        sources = self.export_references()
        for entry in self.entries():
            try:
                row = self.describe(entry, now, sources)
                if row["bytes"]:
                    items.append(row)
            except OSError:
                continue
        items.sort(key=lambda row: (not row["candidate"], row["lastUsed"], row["id"]))
        return {"groups": self.usage_groups(),
                "items": items[:MAX_VISIBLE], "totalItems": len(items),
                "freeBytes": shutil.disk_usage(self.workspace).free,
                "policy": {"cacheDays": CACHE_DAYS, "temporaryDays": TEMP_DAYS, "exportDays": EXPORT_DAYS}}

    def remove(self, key, revision):
        """Delete a selected stale entry only if its inspected generation still matches."""
        if not isinstance(key, str) or not isinstance(revision, str):
            raise ValueError("Выберите строку в хранилище.")
        entry = next((item for item in self.entries() if item.key == key), None)
        if entry is None:
            raise ValueError("Элемент исчез. Обновите список хранилища.")
        return self.remove_entry(entry, revision)

    def remove_entry(self, entry, revision):
        """Recheck age and activity under an exclusive lease before removing data."""
        with cache_lock(entry.group, exclusive=True) as acquired:
            if not acquired:
                raise ValueError("Файлы сейчас используются. Повторите позже.")
            if not regular_entry(entry.path):
                raise ValueError("Элемент исчез. Обновите список хранилища.")
            sources = self.export_references() if entry.kind == "export" else ()
            current = self.describe(entry, time(), sources)
            if current["referenced"] or current["referenceUnknown"]:
                raise ValueError("Экспорт используется сохранённым графом или его связь не удалось проверить.")
            if current["revision"] != revision or not current["candidate"]:
                raise ValueError("Данные изменились. Обновите список хранилища.")
            if entry.path.is_dir():
                shutil.rmtree(entry.path)
            else:
                entry.path.unlink()
            return {"id": entry.key, "bytes": current["bytes"]}

    def sweep(self):
        """Expire leased cache entries by age. Ignore legacy entries until adopted on use."""
        for entry in self.cache_entries():
            try:
                row = self.describe(entry, time())
                if row["automatic"] and row["candidate"]:
                    self.remove_entry(entry, row["revision"])
            except (OSError, ValueError):
                continue

    def start(self):
        """Start a low-frequency background sweep and return its stop event."""
        stop = Event()

        def run():
            """Sweep soon after startup, then once per hour while Studio runs."""
            if stop.wait(10):
                return
            while not stop.is_set():
                try:
                    self.sweep()
                except OSError:
                    LOGGER.warning("Voice Studio storage sweep failed", exc_info=True)
                if stop.wait(SWEEP_SECONDS):
                    break

        Thread(target=run, name="studio-storage", daemon=True).start()
        return stop
