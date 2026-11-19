"""Filesystem locations shared by Voice Studio, model workers and installers."""

from errno import EACCES, EXDEV
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
import getpass
import os
import shutil
import stat
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA = PROJECT_ROOT / "data"


def configured_path(key, default):
    """Resolve an optional operator override and expand a leading home directory."""
    return Path(os.environ.get(key) or default).expanduser().resolve()


def application_data():
    """Return a per-user location for installed Voice Studio assets."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Voice Studio"
    if sys.platform == "win32":
        return configured_path("LOCALAPPDATA", Path.home() / "AppData" / "Local") / "Voice Studio"
    return configured_path("XDG_DATA_HOME", Path.home() / ".local" / "share") / "voice-studio"


USER_SCOPE = str(os.getuid()) if hasattr(os, "getuid") else sha256(getpass.getuser().encode()).hexdigest()[:12]
MODELS_DIR = configured_path("VOICE_MODELS_DIR", application_data() / "Models")
DEPENDENCIES_DIR = configured_path("VOICE_DEPENDENCIES_DIR", application_data() / "Dependencies")
TEMP_OVERRIDE = os.environ.get("VOICE_TEMP_DIR")
if TEMP_OVERRIDE:
    TEMP_DIR = Path(TEMP_OVERRIDE).expanduser().resolve()
else:
    # Resolve the system temp directory without following a preexisting symlink at our own leaf.
    TEMP_DIR = Path(gettempdir()).resolve() / f"voice-studio-{USER_SCOPE}"
RUNTIME_DIR = TEMP_DIR / "runtime"
LEGACY_MODELS_DIR = PROJECT_DATA / "models"
LEGACY_DEPENDENCIES_DIR = PROJECT_DATA / "runtime"


def model_path(*parts, legacy=None):
    """Find model weights, reading a legacy installation during default migration."""
    path = MODELS_DIR.joinpath(*parts)
    if legacy is None:
        legacy = LEGACY_MODELS_DIR.joinpath(*parts)
    if not os.environ.get("VOICE_MODELS_DIR") and not path.exists() and legacy.exists():
        return legacy
    return path


def dependency_path(*parts):
    """Find a provisioned dependency, reading the legacy runtime during migration."""
    path = DEPENDENCIES_DIR.joinpath(*parts)
    legacy = LEGACY_DEPENDENCIES_DIR.joinpath(*parts)
    if not os.environ.get("VOICE_DEPENDENCIES_DIR") and not path.exists() and legacy.exists():
        return legacy
    return path


def workspace_temp(workspace):
    """Isolate persistent temporary caches by the canonical workspace path."""
    key = sha256(os.fsencode(Path(workspace).resolve())).hexdigest()
    return TEMP_DIR / "workspaces" / key


def ensure_temporary_root():
    """Create the application's private parent directory before any cache writes."""
    TEMP_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not TEMP_OVERRIDE and hasattr(os, "getuid"):
        info = TEMP_DIR.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError(EACCES, "Voice Studio temporary directory must be private", TEMP_DIR)


def temporary_directory(*, prefix="voice-"):
    """Create disposable processing space on the configured scratch volume."""
    ensure_temporary_root()
    scratch = TEMP_DIR / "scratch"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    return TemporaryDirectory(prefix=prefix, dir=scratch)


def publish_file(source, destination):
    """Publish a finished scratch file through an atomic replacement."""
    try:
        source.replace(destination)
    except OSError as error:
        if error.errno != EXDEV:
            raise
        with NamedTemporaryFile(prefix=f".{destination.name}-", dir=destination.parent, delete=False) as staged:
            staging_path = Path(staged.name)
        try:
            shutil.copyfile(source, staging_path)
            staging_path.replace(destination)
        finally:
            staging_path.unlink(missing_ok=True)


def publish_directory(source, destination):
    """Publish a finished output bundle atomically, including on another volume."""
    try:
        source.replace(destination)
    except OSError as error:
        if error.errno != EXDEV:
            raise
        with TemporaryDirectory(prefix=".publish-", dir=destination.parent) as folder:
            staged = Path(folder) / destination.name
            shutil.copytree(source, staged)
            staged.replace(destination)


if __name__ == "__main__":
    if sys.argv[1] == "runtime":
        ensure_temporary_root()
    print({"models": MODELS_DIR, "dependencies": DEPENDENCIES_DIR, "runtime": RUNTIME_DIR}[sys.argv[1]])
