"""Verify storage expiry, live leases, reviewed export revisions and retained originals."""

from os import utime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_graph_store import MediaGraphStore, identify
from voice_studio.core.storage_cache import LAST_USED, cache_lock, use_cache
from voice_studio.app.studio_storage import DAY, StudioStorage


class StudioStorageTests(TestCase):
    def setUp(self):
        """Create a disposable workspace with the same layout as Studio."""
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.workspace = self.root / "studio"
        self.workspace.mkdir()
        self.storage = StudioStorage(self.workspace, self.root)

    def age(self, path, days):
        """Set both file timestamps without reading media contents."""
        timestamp = time() - days * DAY
        for child in path.rglob("*") if path.is_dir() else []:
            utime(child, (timestamp, timestamp))
        utime(path, (timestamp, timestamp))

    def make_entry(self, relative, days):
        """Create an aged small file that stands in for real media."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"material")
        self.age(path, days)
        return path

    def test_cleanup_requires_platform_lock_support(self):
        group = self.workspace / "analysis-checkpoints"
        with patch('voice_studio.core.storage_cache.fcntl', None):
            with cache_lock(group) as reader:
                self.assertTrue(reader)
            with cache_lock(group, exclusive=True) as cleaner:
                self.assertFalse(cleaner)

    def test_sweep_expires_only_registered_inactive_caches(self):
        swift = self.root / "runtime/audio-scene/swift-modules"
        swift.mkdir(parents=True)
        with use_cache(swift.parent, swift):
            (swift / "module").write_bytes(b"cache")
        self.assertTrue((swift / LAST_USED).exists())
        self.age(swift, 20)
        temp = self.make_entry("studio/temporary-media/upload-abcdefgh/media", 2).parent
        (temp / LAST_USED).touch()
        self.age(temp, 2)
        legacy = self.make_entry("studio/analysis-checkpoints/" + "a" * 64 + "/legacy.json", 40).parent
        self.age(legacy, 40)
        export = self.make_entry("exports/keep.mov", 45)
        model = self.make_entry("models/model.bin", 90)
        with cache_lock(swift.parent):
            self.storage.sweep()
            self.assertTrue(swift.is_dir())
            self.assertFalse(temp.exists())
        self.storage.sweep()
        self.assertFalse(swift.exists())
        for path in (legacy, export, model):
            self.assertTrue(path.exists())

    def test_recent_reads_and_saved_graphs_prevent_deleting_exports(self):
        old = self.make_entry("exports/old.mov", 45)
        recent = self.make_entry("exports/read.mov", 45)
        utime(recent, (time(), recent.stat().st_mtime))
        linked = self.make_entry("exports/linked.mov", 45)
        store = MediaGraphStore(self.workspace / "media-graphs")
        store.publish(identify(linked), {"kind": "video", "durationMs": 1000},
                      {"nodes": [], "edges": []}, "ru", {})
        self.age(linked, 45)
        (old.parent / "external.mov").symlink_to(linked)
        rows = {row["label"]: row for row in self.storage.inventory()["items"]}
        self.assertTrue(rows["old.mov"]["candidate"])
        self.assertFalse(rows["read.mov"]["candidate"])
        self.assertTrue(rows["linked.mov"]["referenced"])
        self.assertNotIn("external.mov", rows)
        with self.assertRaisesRegex(ValueError, "сохранённым графом"):
            self.storage.remove(rows["linked.mov"]["id"], rows["linked.mov"]["revision"])
        with self.assertRaises(ValueError):
            self.storage.remove("../studio", "arbitrary")
        chosen = rows["old.mov"]
        old.write_bytes(b"replaced")
        self.age(old, 45)
        with self.assertRaisesRegex(ValueError, "изменились"):
            self.storage.remove(chosen["id"], chosen["revision"])
        chosen = next(row for row in self.storage.inventory()["items"] if row["label"] == "old.mov")
        self.assertEqual(self.storage.remove(chosen["id"], chosen["revision"])["bytes"], 8)
        self.assertFalse(old.exists())
        self.assertTrue(linked.exists())

    def test_analysis_checkpoints_compress_new_entries_and_read_older_ones(self):
        path = self.workspace / "cache"
        path.mkdir()
        (path / "legacy.json").write_text('{"frames":[1,2]}')
        revolver = AnalysisRevolver(path)
        self.assertEqual(revolver.load("legacy"), {"frames": [1, 2]})
        payload = {"frames": [{"startMs": i, "text": "repeated label"} for i in range(1000)]}
        revolver.save("scene", payload)
        saved = path / "scene.json.gz"
        self.assertLess(saved.stat().st_size, len(str(payload)) // 4)
        self.assertEqual(revolver.load("scene"), payload)
        saved.write_bytes(b"corrupt")
        self.assertIsNone(revolver.load("scene"))
