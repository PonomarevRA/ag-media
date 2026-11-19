"""Verify native update decisions without depending on a live release or user credentials."""

import os
import unittest
from unittest.mock import patch

from voice_studio.app import desktop_updates


class DesktopUpdateTests(unittest.TestCase):
    def setUp(self):
        """Pin the edition and native package name while leaving manual lookup explicit."""
        product = patch.object(desktop_updates, "load_product", return_value={"version": "0.1.9"})
        self.addCleanup(product.stop)
        product.start()
        repository = patch.dict(os.environ, {"VOICE_RELEASE_REPOSITORY": "example/studio"})
        self.addCleanup(repository.stop)
        repository.start()
        suffix = patch.object(desktop_updates, "package_suffix", return_value="macos-arm64.dmg")
        self.addCleanup(suffix.stop)
        suffix.start()

    def check(self, tag="v0.1.10", assets=(), **fields):
        """Pass a synthetic stable release record through the production comparison."""
        release = {"tag_name": tag, "draft": False, "prerelease": False,
                   "assets": [{"name": name, "state": "uploaded"} for name in assets], **fields}
        with patch.object(desktop_updates, "fetch_release", return_value=release) as fetch:
            result = desktop_updates.check_desktop_update()
            fetch.assert_called_once_with("example/studio")
        return result

    def test_unconfigured_edition_needs_no_network(self):
        """Reject a non-repository override before creating a network request."""
        with patch.dict(os.environ, {"VOICE_RELEASE_REPOSITORY": "https://example.test/feed"}):
            with patch.object(desktop_updates, "fetch_release") as fetch:
                result = desktop_updates.check_desktop_update()
        fetch.assert_not_called()
        self.assertEqual(result["status"], "unconfigured")
        self.assertEqual(result["installedVersion"], "0.1.9")

    def test_update_requires_matching_package_and_checksum(self):
        """Only a matching native package and companion checksum qualify the update."""
        package = "ag-media-studio-0.1.10-macos-arm64.dmg"
        available = self.check(assets=[package, package + ".sha256"])
        self.assertEqual(available["status"], "available")
        self.assertEqual(available["releasesUrl"], "https://github.com/example/studio/releases")
        self.assertEqual(available["latestVersion"], "0.1.10")
        self.assertEqual(self.check(assets=[package])["status"], "no-installer")

    def test_stable_comparison_does_not_offer_an_older_release(self):
        """Compare numeric versions and keep the configured page independent of remote links."""
        self.assertEqual(self.check(tag="v0.1.8")["status"], "current")
        self.assertEqual(self.check(tag="v0.1.9")["status"], "current")
        self.assertEqual(self.check(prerelease=True)["status"], "unavailable")
        self.assertEqual(self.check(tag="v0.1.10-rc1")["status"], "unavailable")

    def test_distinguishes_missing_release_from_connection_failure(self):
        """Tell apart no public stable release and a temporary fetch error."""
        with patch.object(desktop_updates, "fetch_release", return_value=None):
            self.assertEqual(desktop_updates.check_desktop_update()["status"], "no-release")
        with patch.object(desktop_updates, "fetch_release", side_effect=TimeoutError):
            self.assertEqual(desktop_updates.check_desktop_update()["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
