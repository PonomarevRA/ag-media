"""Exercise native service ownership, explicit discovery and portable workspace locking."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from voice_studio.core.workspace_lock import workspace_lock


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('voice_studio.app.desktop_runtime',run_name='__main__')"


class DesktopRuntimeTests(unittest.TestCase):
    """Use independent child processes and isolated data, never the running developer service."""

    def test_lock_rejects_another_process_and_releases(self):
        """A second interpreter cannot own a live workspace, but succeeds after the owner exits."""
        with tempfile.TemporaryDirectory(prefix="native-lock-") as folder:
            command = [sys.executable, "-c", "from voice_studio.core.workspace_lock import workspace_lock;from pathlib import Path;import sys;\nwith workspace_lock(Path(sys.argv[1])): print('locked')", folder]
            with workspace_lock(folder):
                denied = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(denied.returncode, 0)
            accepted = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(accepted.stdout.strip(), "locked")

    def test_linux_mcp_config_finds_vendored_packages_in_isolated_client(self):
        """Execute the printed launch command against a staged Debian source and vendor layout."""
        from voice_studio.app import desktop_config

        with tempfile.TemporaryDirectory(prefix="native-linux-config-") as folder:
            root = Path(folder).resolve()
            source = root / "opt/ag-media-studio/studio"
            vendor = source.parent / "python"
            vendor.mkdir(parents=True)
            (vendor / "ag_media_vendor_probe.py").write_text('RESULT = "vendored import"\n')
            module = source / "voice_studio/app/desktop_config.py"
            module.parent.mkdir(parents=True)
            module.write_text("import sys\nfrom ag_media_vendor_probe import RESULT\n"
                              "print(RESULT)\nprint(sys.argv[1:])\n")
            live = {"schema": "synthetic", "pid": 1, "port": 1, "origin": "synthetic", "instanceId": "synthetic"}
            workspace = root / "workspace"
            with patch.object(desktop_config, "__file__", str(module)), patch.object(sys, "platform", "linux"):
                dev = desktop_config.configuration(workspace, live)["mcpServers"]["ag-media-studio"]
                self.assertNotIn(str(vendor), dev["args"])
                with patch.object(desktop_config, "LINUX_PACKAGE_SOURCE", source):
                    package = desktop_config.configuration(workspace, live)["mcpServers"]["ag-media-studio"]

            child = subprocess.run([package["command"], *package["args"]], cwd=root,
                                   capture_output=True, text=True, timeout=5)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.splitlines(), ["vendored import", str(["--mcp", "--data", str(workspace)])])

    def test_owned_port_metadata_and_config_expire(self):
        """The native service falls back from an occupied port and configuration verifies its nonce."""
        with tempfile.TemporaryDirectory(prefix="native-runtime-") as folder, socket.socket() as occupied:
            root = Path(folder)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            environment = dict(os.environ)
            environment["VOICE_RELEASE_REPOSITORY"] = "unconfigured"
            environment.update({"VOICE_TEMP_DIR": str(root / "temp"), "VOICE_MODELS_DIR": str(root / "models"),
                                "VOICE_DEPENDENCIES_DIR": str(root / "dependencies"), "PYTHONDONTWRITEBYTECODE": "1"})
            child = subprocess.Popen([sys.executable, "-I", "-B", "-c", BOOTSTRAP, str(ROOT), "--data", str(root / "workspace"),
                                      "--port", str(occupied.getsockname()[1])], cwd=root, env=environment,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                status = json.loads(child.stdout.readline())
                self.assertEqual(status["schema"], "ag-media-desktop-ready-v1")
                self.assertEqual(status["pid"], child.pid)
                self.assertNotEqual(status["port"], occupied.getsockname()[1])
                self.assertFalse(status["capabilities"]["automaticDownloads"])
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(status["origin"] + "/api/desktop-instance", timeout=5) as response:
                    self.assertEqual(json.load(response)["instanceId"], status["instanceId"])
                update_url = status["origin"] + "/api/desktop-update"
                without_origin = urllib.request.Request(update_url, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    opener.open(without_origin, timeout=5)
                self.assertEqual(denied.exception.code, 403)
                request = urllib.request.Request(update_url, method="POST",
                    headers={"Origin": status["origin"], "X-Voice-Studio": "1"})
                with opener.open(request, timeout=5) as response:
                    checked = json.load(response)
                self.assertEqual(checked["status"], "unconfigured")
                self.assertRegex(checked["installedVersion"], r"^\d+\.\d+\.\d+$")
                from voice_studio.app.desktop_config import configuration, instance
                workspace = root / "workspace"
                live = instance(workspace)
                config = configuration(workspace, live)
                self.assertIn(str(workspace), config["mcpServers"]["ag-media-studio"]["args"])
                metadata = workspace / "desktop-instance.json"
                tampered = dict(status, instanceId="wrong")
                metadata.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, "identity"):
                    instance(workspace)
                metadata.write_text(json.dumps(status))
                child.stdin.write("quit\n")
                child.stdin.flush()
                child.wait(timeout=10)
                self.assertEqual(child.returncode, 0, child.stderr.read())
                self.assertFalse(metadata.exists())
                with self.assertRaises(ValueError):
                    instance(workspace)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()

    def test_owner_pipe_eof_stops_owned_service(self):
        """Losing the private host pipe releases the service instead of leaving an orphan."""
        with tempfile.TemporaryDirectory(prefix="native-eof-") as folder:
            root = Path(folder)
            environment = dict(os.environ, VOICE_TEMP_DIR=str(root / "temp"),
                               VOICE_MODELS_DIR=str(root / "models"), VOICE_DEPENDENCIES_DIR=str(root / "dependencies"))
            child = subprocess.Popen([sys.executable, "-I", "-B", "-c", BOOTSTRAP, str(ROOT), "--data", str(root / "workspace"), "--port", "0"],
                                     cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                status = json.loads(child.stdout.readline())
                self.assertEqual(status["pid"], child.pid)
                child.stdin.close()
                child.wait(timeout=10)
                self.assertEqual(child.returncode, 0, child.stderr.read())
                self.assertFalse((root / "workspace/desktop-instance.json").exists())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()


if __name__ == "__main__":
    unittest.main()
