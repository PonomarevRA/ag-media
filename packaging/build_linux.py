"""Build a native GTK/WebKit Debian package on a Linux build runner, without bundling models."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from build_native import copy_sources


def run(arguments):
    """Run one fixed build tool with literal arguments and stop on failure."""
    subprocess.run([str(value) for value in arguments], check=True)


def build(options):
    """Create a real .deb with vendored core wheels and explicit distro GUI/Python requirements."""
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise RuntimeError("Build on Linux with Python 3.12 or newer")
    source = options.source.resolve(strict=True)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "packaging/desktop-product.json").read_text())
    architecture = {"aarch64": "arm64", "x86_64": "amd64"}.get(platform.machine())
    if architecture is None:
        raise RuntimeError("Unsupported Debian build architecture")
    python_major, python_minor = sys.version_info[:2]
    python_range = f"python3 (>= {python_major}.{python_minor}), python3 (<< {python_major}.{python_minor + 1})"
    with tempfile.TemporaryDirectory(prefix="ag-media-deb-") as directory:
        package = Path(directory) / "package"
        app = package / "opt/ag-media-studio"
        copy_sources(source, app / "studio")
        libraries = app / "python"
        if options.site_packages:
            shutil.copytree(options.site_packages.resolve(strict=True), libraries, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-compile", "--target", libraries,
                 "-r", source / "requirements-core.txt"])
        # Preserve the exact wheel metadata/licenses in the payload and emit a readable inventory.
        import importlib.metadata
        packages = [{"name": item.metadata["Name"], "version": item.version} for item in importlib.metadata.distributions(path=[str(libraries)])]
        (app / "desktop-product.json").write_text(json.dumps({**manifest, "host": "GTK3/WebKit2GTK", "python": platform.python_version(),
            "corePackages": sorted(packages, key=lambda value: value["name"].lower()), "modelsBundled": False,
            "externalEngines": ["ffmpeg", "ffprobe", "Whisper CLI and weights", "MLX/TTS runtimes"], "automaticDownloads": False}, indent=2) + "\n")
        launcher = package / "usr/bin/ag-media-studio"
        launcher.parent.mkdir(parents=True)
        launcher.write_text('#!/usr/bin/python3\n"""Native ag-media-studio launcher using explicit installed library paths."""\nimport sys\nsys.path.insert(0,"/opt/ag-media-studio/python")\nsys.path.insert(0,"/opt/ag-media-studio/studio")\nfrom voice_studio.app.desktop_linux import main\nmain()\n')
        launcher.chmod(0o755)
        desktop = package / "usr/share/applications/ag-media-studio.desktop"
        desktop.parent.mkdir(parents=True)
        desktop.write_text('[Desktop Entry]\nType=Application\nName=ag-media-studio\nComment=Local media analysis and editing\nExec=/usr/bin/ag-media-studio\nIcon=ag-media-studio\nTerminal=false\nCategories=AudioVideo;Audio;Video;\nStartupWMClass=ag-media-studio\n')
        icon = package / "usr/share/icons/hicolor/scalable/apps/ag-media-studio.svg"
        icon.parent.mkdir(parents=True)
        shutil.copy2(source / "web/brand/studio.svg", icon)
        notices = package / "usr/share/doc/ag-media-studio"
        notices.mkdir(parents=True)
        for name in ("LICENSE", "THIRD-PARTY.md", "THIRD_PARTY_NOTICES.md"):
            if (source / name).is_file():
                shutil.copy2(source / name, notices / name)
        (notices / "README").write_text('Native GTK3/WebKit2GTK application. Core Python packages are vendored; system Python3.12+ and distro GUI libraries are required. No models or FFmpeg are bundled. Provision optional engines explicitly. Data stays in the external XDG data directory (or VOICE_STUDIO_DATA).\n')
        control = package / "DEBIAN/control"
        control.parent.mkdir()
        size = sum(path.stat().st_size for path in package.rglob("*") if path.is_file()) // 1024
        control.write_text(f'Package: ag-media-studio\nVersion: {manifest["version"]}\nSection: sound\nPriority: optional\nArchitecture: {architecture}\nMaintainer: ag-media-studio contributors\nInstalled-Size: {size}\nDepends: {python_range}, python3-gi, gir1.2-gtk-3.0, gir1.2-webkit2-4.1, xdg-utils\nDescription: Local media studio with a native GTK desktop host\n Vendored core packages; optional media engines and weights remain external.\n')
        asset = output / f'ag-media-studio-{manifest["version"]}-linux-{architecture}.deb'
        if asset.exists():
            raise FileExistsError("Choose a new output directory; release artifacts are immutable")
        run(["dpkg-deb", "--root-owner-group", "--build", package, asset])
    with asset.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    result = {"appId": manifest["appId"], "version": manifest["version"], "platform": "linux", "architecture": architecture,
              "asset": asset.name, "sha256": digest, "signing": "unsigned", "modelsBundled": False, "externalSystemDependencies": True,
              "python": platform.python_version(), "nativeHost": "GTK3/WebKit2GTK"}
    (output / (asset.name + ".sha256")).write_text(f"{digest}  {asset.name}\n")
    (output / "release-linux.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    """Accept an explicit source export and artifact destination; never access user data or models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, help="Reuse an explicit already-qualified native core library tree")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
