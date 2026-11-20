"""Assemble a Windows x64 app from explicit local, verified build inputs; never install or publish it."""

import argparse
import getpass
from email.parser import BytesParser
from hashlib import file_digest
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.tags import cpython_tags, compatible_tags

from build_native import copy_sources

PYTHON_VERSION = "3.13.11"
PYTHON_SHA256 = "1ec066fb61ba5e8c73e29e048cd07c26850f74585e3a116005135b31b8004890"
PYTHON_URL = "https://www.python.org/ftp/python/3.13.11/python-3.13.11-embed-amd64.zip"


def sha256(path):
    """Hash a local file without loading the whole runtime into memory."""
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def checked_member(name):
    """Reject absolute, traversal and Windows-drive ZIP members before any filesystem write."""
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("Unsafe archive member: " + name)
    return path


def extract_archive(archive, destination, wheel=False):
    """Unpack safe CPython/wheel members; relocate wheel data payloads to their defined library roots."""
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            relative = checked_member(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are forbidden in the runtime inputs")
            if item.is_dir():
                continue
            parts = relative.parts
            if wheel and parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in {"purelib", "platlib", "data", "scripts", "headers"}:
                    raise ValueError("Unknown wheel data scheme: " + item.filename)
                if parts[1] in {"scripts", "headers"}:
                    continue  # The app invokes modules through its real interpreter, not wheel launch scripts.
                parts = parts[2:]
            target = destination.joinpath(*parts)
            if target.exists() and target.read_bytes() != package.read(item):
                raise ValueError("Conflicting runtime files: " + str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def wheel_metadata(path):
    """Read package identity and dependency declarations without importing untrusted runtime code."""
    with zipfile.ZipFile(path) as package:
        candidates = [name for name in package.namelist() if name.endswith(".dist-info/METADATA")]
        if len(candidates) != 1:
            raise ValueError("A wheel must have exactly one metadata document")
        return BytesParser().parsebytes(package.read(candidates[0]))


def validate_wheels(folder):
    """Check Python 3.13/Windows markers, including dependencies pip may miss when cross-downloading."""
    environment = default_environment()
    environment.update(sys_platform="win32", os_name="nt", platform_system="Windows", platform_machine="AMD64",
                       python_version="3.13", python_full_version=PYTHON_VERSION, implementation_name="cpython", extra="")
    supported_tags = set(cpython_tags((3, 13), abis=["cp313"], platforms=["win_amd64"]))
    supported_tags.update(compatible_tags((3, 13), interpreter="cp313", platforms=["win_amd64"]))
    wheels = sorted(folder.glob("*.whl"))
    if not wheels:
        raise ValueError("The Windows wheel directory is empty")
    packages = {}
    for wheel in wheels:
        if not wheel.name.endswith(("-win_amd64.whl", "-any.whl")):
            raise ValueError("Non-Windows wheel: " + wheel.name)
        if not parse_wheel_filename(wheel.name)[3].intersection(supported_tags):
            raise ValueError("Wheel cannot run on CPython 3.13 Windows x64: " + wheel.name)
        metadata = wheel_metadata(wheel)
        name = canonicalize_name(metadata["Name"])
        if name in packages:
            raise ValueError("Duplicate package version: " + name)
        packages[name] = (metadata, wheel)
    for metadata, wheel in packages.values():
        for declaration in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = packages.get(canonicalize_name(requirement.name))
            if dependency is None or dependency[0]["Version"] not in requirement.specifier:
                raise ValueError(f"Unsatisfied Windows dependency in {wheel.name}: {declaration}")
    return packages


def pe_machine(path):
    """Read a PE machine field to reject an accidentally copied macOS or ARM executable."""
    with path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError("Not a Windows executable: " + path.name)
        handle.seek(0x3C)
        offset = struct.unpack("<I", handle.read(4))[0]
        handle.seek(offset)
        if handle.read(4) != b"PE\x00\x00":
            raise ValueError("Invalid PE header: " + path.name)
        return struct.unpack("<H", handle.read(2))[0]


def copy_licenses(nuget, destination):
    """Preserve official Microsoft redistribution licenses and notices alongside package metadata."""
    destination.mkdir(parents=True)
    count = 0
    for path in sorted(nuget.rglob("*")):
        if not path.is_file() or not ("license" in path.name.lower() or "thirdpartynotices" in path.name.lower()):
            continue
        target = destination / path.relative_to(nuget)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    if not count:
        raise ValueError("Microsoft dependency licenses are missing from the NuGet cache")


def private_build_markers():
    """Identify this build user's home paths and sufficiently specific account name without persisting them."""
    home = str(Path.home())
    account = getpass.getuser()
    github_runner = (os.environ.get("GITHUB_ACTIONS") == "true"
                     and account.casefold() == "runneradmin")
    paths = [home]
    # The hosted Windows account is shared by public upstream builds. Only its
    # standard home is generic; a local user's home or a custom home is still private.
    if github_runner and home.replace("/", "\\").casefold() == r"c:\users\runneradmin":
        paths.clear()
    if github_runner and (workspace := os.environ.get("GITHUB_WORKSPACE")):
        paths.append(workspace)
    values = set()
    for path in paths:
        values.update({path, path.replace("\\", "/"), path.replace("/", "\\")})
    if len(account) >= 8 and not github_runner:
        values.add(account)
    markers = []
    for value in values:
        if not value or value in {"/", "\\"}:
            continue
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            markers.append(value.lower().encode(encoding))
    return markers


def check_payload_privacy(folder, markers=None):
    """Reject local build identity bytes in every staged file, including native debug metadata."""
    markers = private_build_markers() if markers is None else markers
    if not markers:
        return 0
    overlap = max(len(value) for value in markers) - 1
    inspected = 0
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        tail = b""
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                content = tail + block.lower()
                if any(marker in content for marker in markers):
                    raise ValueError("Private build identity in staged payload: " + path.relative_to(folder).as_posix())
                tail = content[-overlap:] if overlap else b""
        if path.name == "AgMediaStudio.dll":
            content = path.read_bytes()
            position = content.find(b"MPDB")
            if position >= 0:
                # Embedded portable PDBs are deflate-compressed, so a byte-only
                # file scan cannot detect compiler paths inside their documents.
                symbols = zlib.decompress(content[position + 8:], -15).lower()
                if any(marker in symbols for marker in markers):
                    raise ValueError("Private build identity in embedded debug metadata: " + path.name)
        inspected += 1
    return inspected


def build(options):
    """Create a complete versioned ZIP plus integrity evidence, with Windows execution explicitly unknown."""
    source = options.source.resolve(strict=True)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    product = json.loads((source / "packaging/desktop-product.json").read_text())
    stem = f"ag-media-studio-{product['version']}-windows-x86_64"
    stage = output / stem
    if stage.exists() or (output / (stem + ".zip")).exists():
        raise FileExistsError("Choose a fresh Windows output directory; existing packages are never overwritten")
    if sha256(options.python_zip) != PYTHON_SHA256:
        raise ValueError("The official CPython embeddable archive checksum does not match the pinned release")
    packages = validate_wheels(options.wheels)
    for line in (source / "requirements-core.txt").read_text().splitlines():
        declaration = line.split("#", 1)[0].strip()
        if not declaration:
            continue
        requirement = Requirement(declaration)
        supplied = packages.get(canonicalize_name(requirement.name))
        if supplied is None or supplied[0]["Version"] not in requirement.specifier:
            raise ValueError("Pinned core dependency is missing: " + declaration)
    for executable in ("AgMediaStudio.exe", "WebView2Loader.dll"):
        if pe_machine(options.publish / executable) != 0x8664:
            raise ValueError("Windows x64 executable required: " + executable)
    shutil.copytree(options.publish, stage, ignore=shutil.ignore_patterns("*.pdb", "Microsoft.Web.WebView2.Wpf.*"))
    runtime = stage / "runtime/python"
    runtime.mkdir(parents=True)
    extract_archive(options.python_zip, runtime)
    if pe_machine(runtime / "python.exe") != 0x8664:
        raise ValueError("CPython is not Windows x64")
    libraries = runtime / "Lib/site-packages"
    libraries.mkdir(parents=True)
    wheels_manifest = []
    for name, (metadata, wheel) in sorted(packages.items()):
        extract_archive(wheel, libraries, wheel=True)
        wheels_manifest.append({"name": metadata["Name"], "version": metadata["Version"], "wheel": wheel.name, "sha256": sha256(wheel)})
    # Keep embedded isolation. Explicit paths support pywin32 without executing arbitrary .pth files.
    (runtime / "python313._pth").write_text("python313.zip\n.\n../../source\nLib/site-packages\nLib/site-packages/win32\nLib/site-packages/win32/lib\nLib/site-packages/Pythonwin\nLib/site-packages/pywin32_system32\n", encoding="utf-8")
    for dll in (libraries / "pywin32_system32").glob("*.dll"):
        # Portable pywin32 imports locate their support DLLs beside python.exe; this
        # works for both the desktop server and real-interpreter worker subprocesses.
        shutil.copy2(dll, runtime / dll.name)
    copy_sources(source, stage / "source")
    (stage / "desktop-product.json").write_text(json.dumps(product, indent=2) + "\n", encoding="utf-8")
    for filename in ("install.ps1", "uninstall.ps1"):
        shutil.copy2(source / "packaging/windows" / filename, stage / filename)
    shutil.copy2(source / "docs/WINDOWS-DISTRIBUTION.md", stage / "WINDOWS-DISTRIBUTION.md")
    for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md", "THIRD-PARTY.md"):
        if not (source / filename).is_file():
            raise FileNotFoundError("Required project notice: " + filename)
        shutil.copy2(source / filename, stage / filename)
    copy_licenses(options.nuget, stage / "licenses/microsoft")
    runtime_config = json.loads((stage / "AgMediaStudio.runtimeconfig.json").read_text())
    frameworks = runtime_config["runtimeOptions"]["includedFrameworks"]
    dotnet_version = next(item["version"] for item in frameworks if item["name"] == "Microsoft.NETCore.App")
    dependencies = json.loads((stage / "AgMediaStudio.deps.json").read_text())
    webview_version = next(name.split("/", 1)[1] for name in dependencies["libraries"] if name.startswith("Microsoft.Web.WebView2/"))
    (stage / "python-wheels.json").write_text(json.dumps(wheels_manifest, indent=2) + "\n")
    release = {"schema": "ag-media-windows-package-v1", "appId": product["appId"], "version": product["version"],
               "platform": "windows", "architecture": "x86_64", "signing": "unsigned", "windowsRuntimeVerified": False,
               "python": {"version": PYTHON_VERSION, "url": PYTHON_URL, "archiveSha256": PYTHON_SHA256},
               "dotnet": dotnet_version, "webview2Sdk": webview_version, "webview2Runtime": "external-evergreen-required",
               "modelsBundled": False, "automaticDownloads": False, "optionalExternalTools": ["ffmpeg", "ffprobe", "model engines"]}
    (stage / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    manifest = {**release, "files": []}
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            manifest["files"].append({"path": path.relative_to(stage).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    (stage / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    privacy_files_checked = check_payload_privacy(stage)
    archive = output / (stem + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(output).as_posix())
    result = {**release, "asset": archive.name, "sha256": sha256(archive), "files": len(manifest["files"]), "wheelCount": len(packages), "privacyFilesChecked": privacy_files_checked}
    (output / (archive.name + ".sha256")).write_text(result["sha256"] + "  " + archive.name + "\n")
    (output / "windows-release.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    """Require explicit runtime, publish, cache and output directories; no credential or model discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    for name in ("publish", "python-zip", "wheels", "nuget", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
