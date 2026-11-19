"""Read bounded package metadata and code bytes without importing optional providers."""

from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
from itertools import islice
import json
from pathlib import Path
import stat
import site
import sys

from .errors import ModError
from .schema import json_bytes, manifest

MAX_PACKAGES = 64
MAX_FILES = 64
MAX_PACKAGE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Package:
    """Captured JSON metadata and source bytes; no live user module objects are retained."""
    manifest: dict
    fingerprint: str
    origin: str
    root: Path | None
    assets: dict


def read_package(root):
    """Capture regular nonsymlink assets, binding all executable package contents to one hash."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ModError("Package root must be a regular directory")
    files = {}
    total = 0
    count = 0
    for path in sorted(islice(root.rglob("*"), 129)):
        count += 1
        if count > 128:
            raise ModError("Package contains too many filesystem entries")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ModError("Package symlinks are not supported")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ModError("Package assets must be regular files")
        if len(files) >= MAX_FILES or path.stat().st_size > MAX_PACKAGE_BYTES:
            raise ModError("Package asset budget exceeded")
        with path.open("rb") as source:
            data = source.read(MAX_PACKAGE_BYTES + 1)
        total += len(data)
        if total > MAX_PACKAGE_BYTES:
            raise ModError("Package exceeds 4 MiB; store model weights externally")
        files[path.relative_to(root).as_posix()] = data
    raw = files.get("mod.json")
    if raw is None or len(raw) > 65536:
        raise ModError("Missing mod.json or manifest exceeds 64 KiB")
    try:
        declared = manifest(json.loads(raw))
    except (ValueError, UnicodeError, TypeError, RecursionError) as error:
        raise ModError(f"Invalid manifest JSON: {error}") from error
    if root.name != declared["id"]:
        raise ModError("Package folder name must equal mod id")
    for capability in declared["capabilities"]:
        for key in ("entrypoint", "prepare_input", "convert_output"):
            if key in capability and capability[key].split(":")[0] not in files:
                raise ModError(f"Declared {key} file is missing")
    digest = sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return Package(declared, digest.hexdigest(), "user", root, files)


def builtin_package(value):
    """Hash host metadata without transient availability; the host registers callbacks."""
    declared = manifest(value, builtin=True)
    identity = {key: item for key, item in declared.items() if key != "availability"}
    return Package(declared, sha256(json_bytes(identity)).hexdigest(), "builtin", None, {})


def unmet_requirements(package):
    """Inspect declared installed prerequisites only; never install dependencies or download models."""
    required = package.manifest["requirements"]
    missing = []
    minimum = required.get("python")
    if minimum:
        version = tuple(int(part) for part in minimum[2:].split("."))
        if sys.version_info[:2] < version:
            missing.append(f"Python {minimum}")
    platforms = required.get("platforms", [])
    if platforms and sys.platform not in platforms:
        missing.append(f"platform {sys.platform} (supported: {', '.join(platforms)})")
    for distribution in required.get("packages", []):
        # -I workers omit user-site and PYTHONPATH; inspect only interpreter-owned sites.
        normalized = distribution.lower().replace("-", "_").replace(".", "_")
        installed = False
        for item in importlib.metadata.distributions(path=site.getsitepackages()):
            name = (item.metadata.get("Name") or "").lower().replace("-", "_").replace(".", "_")
            if name == normalized:
                installed = True
                break
        if not installed:
            missing.append(f"Python distribution {distribution}")
    for filename in required.get("files", []):
        path = Path(filename).expanduser()
        if not path.is_absolute():
            if package.root is None:
                missing.append(f"absolute required file {filename}")
                continue
            path = package.root / path
        if not path.is_file():
            missing.append(f"file {filename}")
    return missing
