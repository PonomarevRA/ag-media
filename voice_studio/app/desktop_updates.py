"""Read stable native release metadata on demand and report matching packages."""

import json
from http.client import HTTPException
import os
import platform
import re
import sys
import urllib.error
import urllib.request

from voice_paths import PROJECT_ROOT


VERSION = re.compile(r"v?((?:0|[1-9][0-9]{0,7})\.(?:0|[1-9][0-9]{0,7})\.(?:0|[1-9][0-9]{0,7}))")
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")
MAX_RESPONSE = 256 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep release lookups at the configured GitHub API endpoint."""

    def redirect_request(self, *args, **kwargs):
        """Decline redirects, including to another GitHub repository."""
        return None


def version(value):
    """Extract a stable version for display and numeric comparison."""
    match = VERSION.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("Invalid stable version")
    normalized = match[1]
    return normalized, tuple(map(int, normalized.split(".")))


def package_suffix(system, machine):
    """Return the asset suffix produced by the native builder for this host."""
    return {
        ("darwin", "arm64"): "macos-arm64.dmg",
        ("darwin", "x86_64"): "macos-x86_64.dmg",
        ("linux", "aarch64"): "linux-arm64.deb",
        ("linux", "x86_64"): "linux-amd64.deb",
        ("win32", "amd64"): "windows-x86_64.zip",
    }.get((system, machine.lower()))


def fetch_release(repository):
    """Load a bounded public release record without cookies, proxy or credentials."""
    url = f"https://api.github.com/repos/{repository}/releases/latest"
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "ag-media-studio-update-check",
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=4) as response:
            data = response.read(MAX_RESPONSE + 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    if len(data) > MAX_RESPONSE:
        raise ValueError("Release metadata exceeds size limit")
    return json.loads(data)


def load_product():
    """Read the packaging version from a source checkout or installed app."""
    for path in (PROJECT_ROOT / "packaging/desktop-product.json",
                 PROJECT_ROOT.parent / "desktop-product.json"):
        if path.is_file():
            product = json.loads(path.read_text(encoding="utf-8"))
            if product.get("appId") != "ag-media-studio":
                break
            return product
    raise ValueError("Studio product metadata unavailable")


def check_desktop_update():
    """Compare stable versions and offer the releases page for manual installation."""
    try:
        product = load_product()
        current, installed = version(product.get("version"))
    except (OSError, AttributeError, ValueError):
        return {"status": "unavailable", "message": "The installed version could not be read."}
    repository = os.environ.get("VOICE_RELEASE_REPOSITORY") or product.get("releaseRepository")
    result = {"installedVersion": current}
    if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
        return {**result, "status": "unconfigured",
                "message": f"Installed version {current}. No release repository is configured."}
    result["releasesUrl"] = f"https://github.com/{repository}/releases"
    try:
        release = fetch_release(repository)
        if release is None:
            return {**result, "status": "no-release",
                    "message": f"Installed version {current}. No public stable release was found."}
        if not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False:
            raise ValueError("Expected a stable release")
        latest, published = version(release.get("tag_name"))
        result["latestVersion"] = latest
        if published <= installed:
            return {**result, "status": "current", "message": f"Installed version {current} is up to date."}
        suffix = package_suffix(sys.platform, platform.machine())
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise ValueError("Invalid release assets")
        uploaded = {row["name"] for row in assets if isinstance(row, dict)
                    and row.get("state") == "uploaded" and isinstance(row.get("name"), str)}
    except (OSError, ValueError, TypeError, RecursionError, HTTPException):
        return {**result, "status": "unavailable",
                "message": f"Installed version {current}. Release metadata is unavailable."}
    package = f"ag-media-studio-{latest}-{suffix}"
    if suffix and {package, package + ".sha256"} <= uploaded:
        return {**result, "status": "available",
                "message": f"Version {latest} is available. Installed version {current}."}
    return {**result, "status": "no-installer",
            "message": f"Version {latest} is published. A package for this device is unavailable."}
