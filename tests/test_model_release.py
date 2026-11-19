"""Exercise release-first model installs without the network or user model cache."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError
from zipfile import ZipFile

import voice_paths
from voice_studio.provision import assets, model_release
from voice_studio.speech import whisper_assets


def bundle(files):
    """Build a miniature release with the same member paths as production."""
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, payload in files.items():
            archive.writestr(f"Models/{name}", payload)
    return output.getvalue()


def ignore_progress(*_):
    """Discard progress values in the installation fixtures."""


class ModelReleaseTests(TestCase):
    """Verify source priority, shared caching and recovery from an unusable mirror."""

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(voice_paths, "TEMP_DIR", self.root / "tmp"))
        self.enterContext(patch.object(assets, "model_path", side_effect=(self.root / "models").joinpath))
        self.whisper_target = self.root / "models" / "whisper" / whisper_assets.BASE_NAME
        self.whisper_target.parent.mkdir(parents=True)
        self.yunet = assets.Asset("vision/yunet.onnx", "https://example.invalid/yunet",
                                  4, sha256(b"face").hexdigest())
        self.enterContext(patch.object(whisper_assets, "BASE_BYTES", 5))
        self.enterContext(patch.object(whisper_assets, "BASE_SHA256", sha256(b"voice").hexdigest()))

    def mock_mirror(self, files, **response):
        """Pin a small test archive and replace only its HTTP transfer."""
        expected = bundle(files)
        self.enterContext(patch.object(model_release, "ARCHIVE_SIZE", len(expected)))
        self.enterContext(patch.object(model_release, "ARCHIVE_SHA256", sha256(expected).hexdigest()))
        return self.enterContext(patch.object(model_release, "urlopen", **(
            response or {"return_value": BytesIO(expected)})))

    def test_both_installers_share_one_download_and_reuse_installed_weights(self):
        """Whisper and vision obtain exact weights from one ZIP without upstream."""
        fetch = self.mock_mirror({"whisper/ggml-base.bin": b"voice", self.yunet.name: b"face"})
        with patch.object(whisper_assets, "fetch_base") as whisper_upstream, \
                patch.object(assets, "fetch_file") as vision_upstream:
            self.assertTrue(whisper_assets.download_base(self.whisper_target, ignore_progress))
            assets.install_assets((self.yunet,), ignore_progress)
            assets.install_assets((self.yunet,), ignore_progress)
        fetch.assert_called_once_with(model_release.RELEASE_URL, timeout=10)
        whisper_upstream.assert_not_called()
        vision_upstream.assert_not_called()
        self.assertEqual(self.whisper_target.read_bytes(), b"voice")
        self.assertEqual(self.yunet.path.read_bytes(), b"face")

    def test_404_downloads_original_whisper(self):
        """Pass a clean staged file to Whisper's original resumable downloader."""
        error = HTTPError(model_release.RELEASE_URL, 404, "Missing release", {}, None)
        fetch = self.mock_mirror({}, side_effect=error)

        def original(stream, progress):
            self.assertEqual(stream.tell(), 0)
            stream.write(b"voice")

        with patch.object(whisper_assets, "fetch_base", side_effect=original) as upstream:
            whisper_assets.download_base(self.whisper_target, ignore_progress)
        fetch.assert_called_once()
        upstream.assert_called_once()
        self.assertEqual(self.whisper_target.read_bytes(), b"voice")

    def test_damaged_archive_uses_original_and_never_enters_cache(self):
        """Reject a downloaded ZIP whose digest differs from the pinned release."""
        expected = bundle({self.yunet.name: b"face"})
        fetch = self.mock_mirror({self.yunet.name: b"face"}, return_value=BytesIO(b"x" + expected[1:]))
        with patch.object(assets, "urlopen", return_value=BytesIO(b"face")) as upstream:
            assets.install_assets((self.yunet,), ignore_progress)
        fetch.assert_called_once()
        upstream.assert_called_once()
        self.assertEqual(list((self.root / "tmp" / "model-releases").iterdir()), [])

    def test_wrong_member_does_not_prepend_bytes_to_original(self):
        """A valid ZIP with a different member hash leaves no stale staged bytes."""
        fetch = self.mock_mirror({self.yunet.name: b"fail"})
        with patch.object(assets, "urlopen", return_value=BytesIO(b"face")) as upstream:
            assets.install_assets((self.yunet,), ignore_progress)
        fetch.assert_called_once()
        upstream.assert_called_once()
        self.assertEqual(self.yunet.path.read_bytes(), b"face")
