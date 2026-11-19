"""Build an independent macOS AppKit application with a real bundled Python interpreter."""

import argparse
from hashlib import file_digest
import importlib.metadata
import json
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def run(arguments):
    """Run fixed local build tools with literal path boundaries and visible failures."""
    subprocess.run([str(value) for value in arguments], check=True)


def copy_sources(source, destination):
    """Copy the finite application source/assets, excluding every data, engine and model directory."""
    destination.mkdir(parents=True)
    for name in ("ag_media_studio.py", "ag_media_studio_mcp.py", "server.py", "voice_mcp.py", "voice_paths.py"):
        shutil.copy2(source / name, destination / name)
    for directory in ("voice_studio", "web", "docs/wiki"):
        root = source / directory
        for item in root.rglob("*"):
            if item.is_file() and not item.is_symlink() and "__pycache__" not in item.parts and item.suffix in {".py", ".json", ".js", ".css", ".html", ".svg", ".md", ".png", ".ico", ".swift"}:
                target = destination / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    for name in ("project.json", "THIRD-PARTY.md", "THIRD_PARTY_NOTICES.md", "LICENSE"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)


def normalize_runtime(destination, version, installed_prefix):
    """Remove installation metadata using loader/configuration APIs, preserving relocatable behavior."""
    library = destination / "lib" / ("libpython" + version + ".dylib")
    run(["/usr/bin/install_name_tool", "-id", "@rpath/" + library.name, library])
    marker = "__AG_BUNDLED_PYTHON_PREFIX__"
    for config in (destination / "lib" / ("python" + version)).glob("_sysconfigdata_*.py"):
        text = config.read_text()
        if installed_prefix not in text:
            continue
        text = text.replace(installed_prefix, marker)
        # uv's generated installation configuration records its current prefix.
        # Restore those values from this interpreter at import, after relocation.
        text += "\n# ag-media-studio: resolve copied installation paths after bundle relocation.\nimport sys as _ag_sys\nfor _ag_key, _ag_value in tuple(build_time_vars.items()):\n    if isinstance(_ag_value, str):\n        build_time_vars[_ag_key] = _ag_value.replace('" + marker + "', _ag_sys.base_prefix)\ndel _ag_sys, _ag_key, _ag_value\n"
        config.write_text(text)


def copy_runtime(python, site_packages, destination):
    """Preserve the standalone interpreter's relative dylib paths and licensed package metadata."""
    description = subprocess.check_output([str(python), "-I", "-c", "import sys,json; print(json.dumps({'prefix':sys.base_prefix,'version':f'{sys.version_info.major}.{sys.version_info.minor}'}))"], text=True)
    runtime = json.loads(description)
    base = Path(runtime["prefix"]).resolve()
    if not (base / "lib" / ("libpython" + runtime["version"] + ".dylib")).is_file():
        raise ValueError("Use a relocatable standalone CPython distribution, not a framework or system installation")
    shutil.copytree(base, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "include", "pkgconfig"))
    libraries = destination / "lib" / ("python" + runtime["version"]) / "site-packages"
    libraries.mkdir(parents=True, exist_ok=True)
    for item in site_packages.iterdir():
        if item.name == "__pycache__" or item.name.startswith("_virtualenv") or item.suffix == ".pth":
            continue
        target = libraries / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=False, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif item.is_file():
            shutil.copy2(item, target)
    normalize_runtime(destination, runtime["version"], str(base))
    packages = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        packages.append({"name": distribution.metadata["Name"], "version": distribution.version})
    return {"python": runtime["version"], "packages": sorted(packages, key=lambda value: value["name"].lower())}


def audit_macos_minimum(bundle, declared="13.0"):
    """Check every bundled Mach-O slice against the declared OS floor and retain exact evidence."""
    declared_parts = (tuple(map(int, declared.split("."))) + (0, 0, 0))[:3]
    records = []
    for path in bundle.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic not in {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"}:
            continue
        commands = subprocess.check_output(["/usr/bin/otool", "-l", str(path)], text=True)
        versions = re.findall(r"\bminos\s+(\d+(?:\.\d+)+)", commands)
        versions += re.findall(r"cmd LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+version (\d+(?:\.\d+)+)", commands)
        if not versions:
            raise ValueError("Mach-O minimum OS could not be read: " + str(path.relative_to(bundle)))
        for version in versions:
            parts = (tuple(map(int, version.split("."))) + (0, 0, 0))[:3]
            if parts > declared_parts:
                raise ValueError("Bundled binary exceeds declared minimum macOS: " + str(path.relative_to(bundle)) + " " + version)
        records.append({"path": str(path.relative_to(bundle)), "minimumVersions": versions})
    return {"declaredMinimumMacOS": declared, "allBundledMachOWithinFloor": True, "records": records,
            "limit": "Load-command audit; execution on earlier OS releases is not verified"}


def sign_bundle(bundle):
    """Ad-hoc sign each Mach-O payload before sealing the containing application."""
    for path in bundle.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic in {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"}:
            run(["/usr/bin/codesign", "--force", "--sign", "-", path])
    run(["/usr/bin/codesign", "--force", "--sign", "-", bundle])
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", bundle])


def build(options):
    """Create a tested native app plus DMG and checksum metadata without installing it."""
    if sys.platform != "darwin":
        raise RuntimeError("The AppKit build must run on macOS")
    source = options.source.resolve(strict=True)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "packaging/desktop-product.json").read_text())
    bundle = output / (manifest["name"] + ".app")
    if bundle.exists():
        raise FileExistsError("Choose an empty output directory; existing applications are never replaced")
    contents = bundle / "Contents"
    resources = contents / "Resources"
    executable = contents / "MacOS/AgMediaStudio"
    executable.parent.mkdir(parents=True)
    resources.mkdir(parents=True)
    shutil.copy2(source / "packaging/icons/AgMediaStudio.icns", resources / "AgMediaStudio.icns")
    copy_sources(source, resources / "studio")
    runtime = copy_runtime(options.python.resolve(), options.site_packages.resolve(), resources / "runtime")
    info = {"CFBundleExecutable": "AgMediaStudio", "CFBundleIdentifier": manifest["bundleId"], "CFBundleName": manifest["name"],
            "CFBundleDisplayName": manifest["name"], "CFBundlePackageType": "APPL", "CFBundleShortVersionString": manifest["version"],
            "CFBundleVersion": manifest["version"], "CFBundleIconFile": "AgMediaStudio.icns", "LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Record audio only when you choose the Studio recording action.",
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True}}
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    (resources / "desktop-product.json").write_text(json.dumps({**manifest, "runtime": runtime, "modelsBundled": False,
        "externalEngines": ["ffmpeg", "ffprobe", "Whisper CLI and weights", "MLX/TTS runtimes"], "automaticDownloads": False}, indent=2) + "\n")
    swift = sorted((source / "packaging/native").glob("*.swift"))
    run(["xcrun", "swiftc", "-O", "-parse-as-library", "-target", platform.machine() + "-apple-macosx13.0", "-module-cache-path", options.work / "swift-cache", *swift, "-o", executable])
    minimum_audit = audit_macos_minimum(bundle)
    (output / "macos-minimum-os-audit.json").write_text(json.dumps(minimum_audit, indent=2) + "\n")
    # Import through a real standalone interpreter: its prefix and worker executable
    # must resolve inside the app even with isolated Python and an unrelated cwd.
    bootstrap = "import sys,sysconfig,pathlib;sys.path.insert(0,sys.argv[1]);import voice_studio.app.desktop_runtime;assert pathlib.Path(sysconfig.get_config_var('BINDIR')).parent.resolve()==pathlib.Path(sys.base_prefix).resolve();print(sys.executable)"
    run([resources / "runtime/bin/python3", "-I", "-B", "-c", bootstrap, resources / "studio"])
    sign_bundle(bundle)
    stem = f"ag-media-studio-{manifest['version']}-macos-{platform.machine()}"
    with tempfile.TemporaryDirectory(prefix="media-dmg-", dir=options.work) as folder:
        image = Path(folder) / "image"
        image.mkdir()
        run(["/usr/bin/ditto", bundle, image / bundle.name])
        (image / "Applications").symlink_to("/Applications")
        (image / "READ ME.txt").write_text("Drag ag-media-studio.app to Applications or another folder.\nAd-hoc signed; not notarized.\nCore Python is bundled. FFmpeg and model/engine installations are optional external dependencies; no models are downloaded automatically.\nData stays outside this application in Application Support/Voice Studio.\n")
        asset = output / (stem + ".dmg")
        run(["/usr/bin/hdiutil", "create", "-quiet", "-volname", "ag-media-studio", "-srcfolder", image, "-format", "UDZO", asset])
    with asset.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    (output / (asset.name + ".sha256")).write_text(f"{digest}  {asset.name}\n")
    result = {"appId": manifest["appId"], "version": manifest["version"], "platform": "macos", "architecture": platform.machine(),
              "asset": asset.name, "sha256": digest, "signing": "ad-hoc", "notarized": False, "modelsBundled": False, "externalEngines": True, "minimumMacOS": "13.0", "earlierOSRuntimeVerified": False}
    (output / "release.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    """Read explicit build inputs; never discover user data, signing keys or model directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    options = parser.parse_args()
    options.work.mkdir(parents=True, exist_ok=True)
    build(options)


if __name__ == "__main__":
    main()
