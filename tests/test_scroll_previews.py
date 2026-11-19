"""Verify that the last working mosaics survive idle clients and failed replacement."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.app.scroll_previews import PNG_HEADER, PREVIEW_PATH, ScrollPreviews


class ScrollPreviewTests(TestCase):
    def test_last_working_page_survives_idle_restart_and_failed_analysis(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mosaic.png"
            source.write_bytes(PNG_HEADER + b"first")
            cache = root / "previews"
            store = ScrollPreviews(cache)
            with store.analysis() as publish:
                first_key, _ = PREVIEW_PATH.fullmatch(publish(0, source)).groups()
            os.utime(store.path(first_key, 0), (1, 1))
            store = ScrollPreviews(cache)
            self.assertEqual(store.read(first_key, 0), PNG_HEADER + b"first")

            with self.assertRaises(RuntimeError), store.analysis() as publish:
                failed_key, _ = PREVIEW_PATH.fullmatch(publish(1, source)).groups()
                self.assertIsNotNone(store.read(first_key, 0))
                raise RuntimeError("analysis failed")
            self.assertIsNone(store.read(failed_key, 1))
            self.assertEqual(store.read(first_key, 0), PNG_HEADER + b"first")

            with store.analysis() as publish:
                self.assertIsNone(publish(0, root / "missing.png"))
            self.assertIsNotNone(store.read(first_key, 0))

            source.write_bytes(PNG_HEADER + b"second")
            with store.analysis() as publish:
                second_key, _ = PREVIEW_PATH.fullmatch(publish(1, source)).groups()
                self.assertIsNotNone(store.read(first_key, 0))
                self.assertIsNone(store.read(second_key, 0))
            self.assertIsNone(store.read(first_key, 0))
            self.assertEqual(store.read(second_key, 1), PNG_HEADER + b"second")
            self.assertIsNone(store.read("f" * 32, 0))

    def test_concurrent_processing_keeps_its_own_files_during_replacement(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mosaic.png"
            source.write_bytes(PNG_HEADER + b"pixels")
            store = ScrollPreviews(root / "previews")
            with store.analysis() as earlier:
                in_flight, _ = PREVIEW_PATH.fullmatch(earlier(0, source)).groups()
                with store.analysis() as later:
                    latest, _ = PREVIEW_PATH.fullmatch(later(0, source)).groups()
                self.assertIsNotNone(store.read(in_flight, 0))
                self.assertIsNotNone(store.read(latest, 0))
            self.assertIsNotNone(store.read(latest, 0))
            with store.analysis() as replace:
                replacement, _ = PREVIEW_PATH.fullmatch(replace(0, source)).groups()
            self.assertIsNone(store.read(latest, 0))
            self.assertIsNone(store.read(in_flight, 0))
            self.assertIsNotNone(store.read(replacement, 0))
