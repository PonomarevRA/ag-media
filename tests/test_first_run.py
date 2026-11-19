"""Verify first-run provisioning without changing installed weights or tools."""

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.app import first_run
from voice_studio.speech import whisper_assets


class FirstRunModelTests(TestCase):
    """Exercise model reuse, integrity and the publication boundary."""

    model_data = b"synthetic Whisper payload"

    def setUp(self):
        """Use synthetic weights in an owned model directory."""
        directory = self.enterContext(TemporaryDirectory())
        self.target = Path(directory) / "whisper" / whisper_assets.BASE_NAME
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(whisper_assets, "whisper_base_model", return_value=self.target))
        self.enterContext(patch.object(whisper_assets.ModelRelease, "copy", return_value=False))
        self.enterContext(patch.object(whisper_assets, "BASE_BYTES", len(self.model_data)))
        self.enterContext(patch.object(whisper_assets, "BASE_SHA256", sha256(self.model_data).hexdigest()))

    def test_install_then_reuse_offline(self):
        """A verified download becomes observable once and is reused without HTTP."""
        progress = []
        def report(received, total):
            progress.append((received, total))

        with patch.object(whisper_assets, "urlopen", return_value=BytesIO(self.model_data)) as fetch:
            self.assertEqual(whisper_assets.install_base(report), (self.target, True))
            self.assertEqual(whisper_assets.install_base(report), (self.target, False))
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0].full_url, whisper_assets.BASE_URL)
        self.assertEqual(progress, [(len(self.model_data), len(self.model_data))])
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertEqual(self.target.read_bytes(), self.model_data)

    def test_invalid_download_leaves_no_partial_weight(self):
        """Truncation, excess bytes and incorrect content never publish a model."""
        samples = (self.model_data[:-1], self.model_data + b"extra", b"x" * len(self.model_data))
        for sample in samples:
            with self.subTest(size=len(sample)), patch.object(whisper_assets, "urlopen") as fetch:
                def stream(request, **_):
                    """Return an empty tail after a short upstream response."""
                    if request.get_header("Range"):
                        response = BytesIO()
                        response.status = 206
                        offset = len(sample)
                        response.headers = {"Content-Range": f"bytes {offset}-{len(self.model_data) - 1}/{len(self.model_data)}"}
                        return response
                    return BytesIO(sample)

                fetch.side_effect = stream
                with self.assertRaises(ValueError):
                    whisper_assets.install_base(lambda *_: None)
                self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_resume_interrupted_response(self):
        """Continue from the last written byte and verify hash before publishing."""
        offset = 5
        first = BytesIO(self.model_data[:offset])
        remainder = BytesIO(self.model_data[offset:])
        remainder.status = 206
        remainder.headers = {"Content-Range": f"bytes {offset}-{len(self.model_data) - 1}/{len(self.model_data)}"}
        with patch.object(whisper_assets, "urlopen", side_effect=[first, remainder]) as fetch:
            self.assertEqual(whisper_assets.install_base(lambda *_: None), (self.target, True))
        request = fetch.call_args.args[0]
        self.assertEqual(request.get_header("Range"), f"bytes={offset}-")
        self.assertEqual(self.target.read_bytes(), self.model_data)

    def test_preserve_existing_different_weights(self):
        """A mismatched local file is preserved instead of being replaced silently."""
        self.target.parent.mkdir()
        self.target.write_bytes(b"operator weights")
        with self.assertRaises(ValueError):
            whisper_assets.install_base(lambda *_: None)
        self.assertEqual(self.target.read_bytes(), b"operator weights")
        with patch.dict(os.environ, {"VOICE_WHISPER_MODEL": str(self.target)}):
            self.assertEqual(whisper_assets.install_base(lambda *_: None), (self.target, False))

    def test_other_installer_can_publish_while_downloading(self):
        """Concurrent completion keeps the first verified inode."""
        def publish_existing_model(*_):
            self.target.write_bytes(self.model_data)

        with patch.object(whisper_assets, "urlopen", return_value=BytesIO(self.model_data)):
            self.assertEqual(whisper_assets.install_base(publish_existing_model), (self.target, False))
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])


class FirstRunToolsTests(TestCase):
    """Check that setup delegates only missing tools to an available platform manager."""

    def test_macos_homebrew_packages_are_selected_once(self):
        """FFmpeg provides both tools, and the correct Whisper formula is selected."""
        with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True), \
                patch.object(first_run.sys, "platform", "darwin"), \
                patch.object(first_run, "missing_tools", side_effect=[["ffmpeg", "ffprobe", "whisper-cli"], []]), \
                patch.object(first_run, "which", return_value="/opt/homebrew/bin/brew"), \
                patch.object(first_run.subprocess, "run") as run:
            first_run.prepare_tools()
        run.assert_called_once_with(["/opt/homebrew/bin/brew", "install", "ffmpeg", "whisper.cpp"], check=True)

    def test_linux_missing_tools_stop_with_an_actionable_error(self):
        """Setup reports tools that have to be available on the current machine."""
        with patch.object(first_run.sys, "platform", "linux"), \
                patch.object(first_run, "missing_tools", return_value=["whisper-cli"]), \
                patch.object(first_run.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "whisper-cli"):
                first_run.prepare_tools()
        run.assert_not_called()
