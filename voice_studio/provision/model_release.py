"""Install known model weights from a verified GitHub Release archive."""

from hashlib import sha256
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile
from zlib import error as DecompressionError

import voice_paths


ARCHIVE_NAME = "AG-Media-Models-v1.zip"
RELEASE_URL = f"https://github.com/PonomarevRA/ag-media/releases/download/models-v1/{ARCHIVE_NAME}"
ARCHIVE_SIZE = 156_341_488
ARCHIVE_SHA256 = "4aa0d2e4408656ebe9cec795823d8475fdd1e953af6568073d398fe99ee6155c"
MEMBERS = frozenset({
    "whisper/ggml-base.bin",
    "vision/yunet.onnx",
    "audio-tagging/zipformer-small-int8/model.int8.onnx",
    "audio-tagging/zipformer-small-int8/class_labels_indices.csv",
})
_cache_lock = Lock()


def copy_bounded(source, output, limit, progress):
    """Stream bounded chunks and return their size and SHA-256."""
    received = 0
    digest = sha256()
    while chunk := source.read(1024 * 1024):
        received += len(chunk)
        if received > limit:
            raise ValueError("Downloaded file exceeds its size limit")
        output.write(chunk)
        digest.update(chunk)
        progress(received)
    return received, digest.hexdigest()


def cached_archive(progress):
    """Share one verified transfer across installs on the same temporary volume."""
    with _cache_lock:
        voice_paths.ensure_temporary_root()
        directory = voice_paths.TEMP_DIR / "model-releases"
        directory.mkdir(mode=0o700, exist_ok=True)
        archive = directory / ARCHIVE_NAME
        if archive.is_file() and archive.stat().st_size == ARCHIVE_SIZE:
            return archive

        with TemporaryDirectory(dir=directory, prefix=".model-release-") as staging:
            temporary = Path(staging) / ARCHIVE_NAME
            with urlopen(RELEASE_URL, timeout=10) as response, temporary.open("wb") as output:
                identity = copy_bounded(response, output, ARCHIVE_SIZE,
                                        lambda received: progress(received, ARCHIVE_SIZE))
            if identity != (ARCHIVE_SIZE, ARCHIVE_SHA256):
                raise ValueError("Model release archive failed integrity verification")
            temporary.replace(archive)
        return archive


class ModelRelease:
    """Attempt the mirror once per installation when the archive is unavailable."""

    def __init__(self):
        self.unavailable = False

    def copy(self, name, output, size, sha, progress):
        """Copy a verified member or leave the staged output empty for upstream."""
        if self.unavailable or name not in MEMBERS:
            return False
        try:
            with ZipFile(cached_archive(progress)) as archive:
                member = archive.getinfo(f"Models/{name}")
                if member.is_dir() or member.file_size != size:
                    raise ValueError("Unexpected model release member size")
                with archive.open(member) as source:
                    identity = copy_bounded(source, output, size, lambda _: None)
                if identity != (size, sha):
                    raise ValueError("Model release member failed integrity verification")
            return True
        except (OSError, HTTPException, ValueError, KeyError, BadZipFile, NotImplementedError, DecompressionError):
            self.unavailable = True
            output.seek(0)
            output.truncate()
            return False
