"""Model installation integrity and the background settings contract."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from voice_studio.provision import assets, jobs
from voice_studio.core import optional_runtime
from voice_studio.core.worker_process import worker_command


class InstallAssetTests(TestCase):
    """Keep partial and incompatible files out of the published model path."""

    def test_verified_download_becomes_available_and_is_reused(self):
        """A retry leaves the exact verified file and does not fetch it a second time."""
        payload = b"model fixture"
        artifact = assets.Asset("weights/test.bin", "https://example.invalid/model", len(payload), sha256(payload).hexdigest())
        with TemporaryDirectory() as folder, patch.object(assets, "model_path", side_effect=Path(folder).joinpath):
            with patch.object(assets, "urlopen", return_value=BytesIO(payload)) as fetch:
                assets.install_assets((artifact,), lambda *_: None)
                assets.install_assets((artifact,), lambda *_: None)
                fetch.assert_called_once()
            self.assertEqual(artifact.path.read_bytes(), payload)
            self.assertEqual(list(artifact.path.parent.iterdir()), [artifact.path])

    def test_hash_failure_and_existing_custom_weight(self):
        """A failed transfer is discarded and a custom file is preserved."""
        artifact = assets.Asset("weights/test.bin", "https://example.invalid/model", 4, sha256(b"good").hexdigest())
        with TemporaryDirectory() as folder, patch.object(assets, "model_path", side_effect=Path(folder).joinpath):
            with patch.object(assets, "urlopen", return_value=BytesIO(b"bad!")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    assets.install_assets((artifact,), lambda *_: None)
            self.assertFalse(artifact.path.exists())
            self.assertEqual(list(artifact.path.parent.iterdir()), [])
            artifact.path.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "Existing file"):
                assets.install_assets((artifact,), lambda *_: None)
            self.assertEqual(artifact.path.read_bytes(), b"keep")


class InstallJobTests(TestCase):
    """Exercise single-flight background work and readiness without external downloads."""

    def test_job_publishes_progress_and_validates_group_ids(self):
        """A repeated action reuses the same install, and a second group waits."""
        entered, release, ready = Event(), Event(), Event()

        def install(key, group, progress):
            progress("weights", 3, 7, "test.bin")
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Fixture release timed out")
            ready.set()

        with TemporaryDirectory() as folder, patch.object(jobs, "dependency_path", return_value=Path(folder)), \
                patch.object(jobs, "group_ready", side_effect=lambda key, group: key == "speakers" and ready.is_set()), \
                patch.object(jobs, "install_group", side_effect=install) as call:
            service = jobs.ModelInstalls()
            with self.assertRaises(ValueError):
                service.start({"command": "anything"})
            service.start("speakers")
            self.assertTrue(entered.wait(3))
            try:
                service.start("speakers")
                with self.assertRaises(ValueError):
                    service.start("sounds")
                report = service.inventory()
                self.assertEqual(report["active"], "speakers")
                speaker = next(item for item in report["items"] if item["id"] == "speakers")
                self.assertEqual((speaker["stage"], speaker["received"], speaker["total"]), ("weights", 3, 7))
            finally:
                release.set()
            deadline = monotonic() + 3
            while service.inventory()["active"] and monotonic() < deadline:
                sleep(.01)
            self.assertIsNone(service.inventory()["active"])
            call.assert_called_once()


class RuntimeLaunchTests(TestCase):
    """Managed wheels are found even when ordinary PYTHONPATH is ignored."""

    def test_worker_loads_managed_runtime_in_isolated_python(self):
        """A worker sees the staged runtime on platforms with isolated interpreters."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".studio-ready").touch()
            (root / "studio_optional_fixture.py").write_text('print("managed runtime imported")\n')
            with patch.object(optional_runtime, "runtime_path", return_value=root):
                env = optional_runtime.runtime_environment("vision")
            command = worker_command(sys.executable, "studio_optional_fixture")
            command.insert(1, "-I")
            output = subprocess.check_output(command, env=env, text=True, timeout=10)
            self.assertEqual(output.strip(), "managed runtime imported")

    def test_activity_worker_combines_managed_runtimes_in_order(self):
        """Vision wheels take priority and the isolated worker can also import activity wheels."""
        from voice_studio.vision.activity_worker_session import ActivityWorkerSession

        with TemporaryDirectory() as folder:
            root = Path(folder)
            vision = root / "vision"
            activity = root / "light-asd"
            for group in (vision, activity):
                group.mkdir()
                (group / ".studio-ready").touch()
                (group / "studio_shared_fixture.py").write_text(f"source = {group.name!r}\n")
            (activity / "studio_activity_fixture.py").write_text(
                'from studio_shared_fixture import source\nprint(source)\n')
            command = worker_command(sys.executable, "studio_activity_fixture")
            command.insert(1, "-I")
            with patch.object(optional_runtime, "runtime_path", side_effect=lambda name: root / name):
                with ActivityWorkerSession(command, []) as worker:
                    output = subprocess.check_output(worker.command, env=worker.env, text=True, timeout=10)
            self.assertEqual(output.strip(), "vision")
            self.assertFalse((vision / "__pycache__").exists())
