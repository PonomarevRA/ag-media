"""Exercise independent temporary/model roots and cross-volume publication."""

from errno import EXDEV
import os
from pathlib import Path
from time import time
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from voice_studio.core.storage_cache import use_cache
from voice_studio.app.studio_storage import DAY, StudioStorage
from voice_paths import model_path, publish_directory, publish_file, temporary_directory, workspace_temp


class VoicePathTests(TestCase):
    def setUp(self):
        """Isolate model lookup and scratch work from the running studio."""
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.workspace = self.root / "project" / "studio"
        self.workspace.mkdir(parents=True)
        self.scratch = self.root / "external-scratch"
        settings = patch("voice_paths.TEMP_DIR", self.scratch)
        settings.start()
        self.addCleanup(settings.stop)

    def test_model_lookup_keeps_default_legacy_and_honors_explicit_override(self):
        legacy = self.root / "project" / "data" / "models"
        installed = self.root / "installed"
        relative = Path("whisper") / "weights.bin"
        (legacy / relative).parent.mkdir(parents=True)
        (legacy / relative).write_bytes(b"legacy")
        with (patch("voice_paths.MODELS_DIR", installed),
              patch("voice_paths.LEGACY_MODELS_DIR", legacy),
              patch.dict("os.environ", {"VOICE_MODELS_DIR": ""})):
            self.assertEqual(model_path(*relative.parts), legacy / relative)
            with patch.dict("os.environ", {"VOICE_MODELS_DIR": str(installed)}):
                self.assertEqual(model_path(*relative.parts), installed / relative)
            (installed / relative).parent.mkdir(parents=True)
            (installed / relative).write_bytes(b"new")
            self.assertEqual(model_path(*relative.parts), installed / relative)

    def test_scratch_isolated_from_workspace_and_removed_on_exit(self):
        base = workspace_temp(self.workspace)
        self.assertTrue(base.is_relative_to(self.scratch))
        self.assertNotEqual(base, workspace_temp(self.workspace.with_name("another-studio")))
        with temporary_directory() as folder:
            scratch = Path(folder)
            self.assertTrue(scratch.is_relative_to(self.scratch))
            self.assertFalse(scratch.is_relative_to(self.workspace))
        self.assertFalse(scratch.exists())
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_external_cache_is_visible_and_expired_with_an_active_workspace(self):
        group = workspace_temp(self.workspace) / "analysis-checkpoints"
        entry = group / ("a" * 64)
        with use_cache(group, entry):
            (entry / "stage.json").write_text("{}")
        stale = time() - 20 * DAY
        for path in (*entry.iterdir(), entry):
            os.utime(path, (stale, stale))
        storage = StudioStorage(self.workspace, self.workspace.parent / "data")
        rows = storage.inventory()["items"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["automatic"])
        self.assertTrue(rows[0]["candidate"])
        storage.sweep()
        self.assertFalse(entry.exists())
        self.assertFalse((self.workspace / group.name).exists())

    @skipUnless(hasattr(os, "getuid"), "POSIX directory ownership and permissions")
    def test_default_scratch_rejects_precreated_symlink_and_public_directory(self):
        target = self.root / "target"
        target.mkdir()
        self.scratch.symlink_to(target, target_is_directory=True)
        with patch("voice_paths.TEMP_OVERRIDE", None):
            with self.assertRaises(PermissionError):
                temporary_directory()
            self.assertEqual(list(target.iterdir()), [])
            self.scratch.unlink()
            self.scratch.mkdir()
            self.scratch.chmod(0o755)
            with self.assertRaises(PermissionError):
                temporary_directory()
            self.assertEqual(list(self.scratch.iterdir()), [])

    def test_cross_volume_publication_retains_atomic_artifact_swap(self):
        replace = Path.replace
        with temporary_directory() as folder:
            source = Path(folder)
            wave = source / "result.wav"
            wave.write_bytes(b"wave")
            bundle = source / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text("{}")
            output = self.workspace / wave.name
            output.write_bytes(b"previous")

            def cross_device(path, destination):
                """Require staging on the output filesystem before replacing it."""
                if path.is_relative_to(source):
                    raise OSError(EXDEV, "Cross-device link")
                return replace(path, destination)

            with patch.object(Path, "replace", cross_device):
                publish_file(wave, output)
                publish_directory(bundle, self.workspace / bundle.name)
            self.assertEqual(output.read_bytes(), b"wave")
            self.assertEqual((self.workspace / bundle.name / "manifest.json").read_text(), "{}")
            self.assertEqual(sorted(child.name for child in self.workspace.iterdir()), ["bundle", "result.wav"])
