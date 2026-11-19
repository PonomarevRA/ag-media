"""Persist successful media-analysis stages so a local graph build can resume."""

import gzip
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from shutil import which
import sys
import zlib
from tempfile import NamedTemporaryFile

from voice_studio.synthesis.engines import ROOT
from voice_studio.speech.person_mentions import BRIDGE
from voice_studio.speech.recognition import CLI, MODEL, RECOVERY_MODEL, SMALL_MODEL
from voice_studio.vision.video_revolver import ASD_WEIGHTS
from voice_paths import LEGACY_MODELS_DIR, MODELS_DIR

LOGGER = logging.getLogger(__name__)


def file_identity(path):
    """Return local file identity for model weights and native executables."""
    info = path.stat()
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def result_fingerprint(result):
    """Hash the exact serialized stage input used by downstream children."""
    return sha256(json.dumps(result, ensure_ascii=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def analysis_fingerprint(source_digest, settings, *, environment_names=None):
    """Bind checkpoints to the source, analysis settings, code and installed model files."""
    environment = {name: os.environ.get(name) for name in (
        "VOICE_SCENE_OCR", "VOICE_SCENE_OCR_WORKERS", "VOICE_RESOURCE_PROFILE", "VOICE_CPU_BUDGET",
        "VOICE_MEMORY_MIB", "VOICE_PITCH_BACKEND", "VOICE_SOUND_STREAM_CHUNK_MS",
        "VOICE_ASR_PARALLEL", "VOICE_VIDEO_PARALLEL", "VOICE_YUNET_SAMPLING",
        "VOICE_CLOTHING_PROFILE", "VOICE_AUDIO_CLASSIFIER",
    )}
    environment["VOICE_PERSON_FACE_PILOT"] = os.environ.get("VOICE_PERSON_FACE_PILOT", "1") == "1"
    if environment_names is not None:
        environment = {name: os.environ.get(name) for name in environment_names}
    digest = sha256(json.dumps([source_digest, settings, sys.version, environment], sort_keys=True).encode())
    sources = [
        *ROOT.glob("*.py"), *(ROOT / "voice_studio").rglob("*.py"),
        ROOT / "scene_ocr.swift", ROOT / "audio_scene.swift",
    ]
    for path in sorted(sources):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    weights = list(MODELS_DIR.rglob("*"))
    if not os.environ.get("VOICE_MODELS_DIR") and LEGACY_MODELS_DIR != MODELS_DIR:
        weights.extend(LEGACY_MODELS_DIR.rglob("*"))
    privacy = Path(os.environ.get("AG_PRIVACY_ROOT") or Path.home() / "Documents/ai/ag-privacy")
    external = [
        CLI, MODEL, SMALL_MODEL, RECOVERY_MODEL, ASD_WEIGHTS,
        Path(os.environ.get("AG_PRIVACY_PERSON_MODEL") or privacy / "models/person-prose.bin"),
        Path(os.environ.get("VOICE_PERSON_MENTIONS_BRIDGE") or BRIDGE),
    ]
    programs = ("ffmpeg", "ffprobe", "tesseract", "swiftc", "dotnet")
    executables = [Path(path) for name in programs if (path := which(name))]
    for path in sorted(weights + external + executables):
        identity = file_identity(path) if path.is_file() else None
        if path in external or identity is not None:
            digest.update(json.dumps([str(path), identity]).encode())
    return digest.hexdigest()


class AnalysisRevolver:
    """Run a stage or a nested group, sharing one exact-input checkpoint directory."""

    def __init__(self, directory=None, counts=None):
        self.directory = directory
        self.counts = counts if counts is not None else {"reused": 0, "saved": 0, "writeFailures": 0}

    def child(self, name):
        """Give a subtask a separate namespace and share the parent's counters."""
        return AnalysisRevolver(self.directory / name if self.directory else None, self.counts)

    def child_for_input(self, name, value):
        """Namespace a child by its upstream result so different results cannot share state."""
        group = self.child(name)
        return group.child(result_fingerprint(value)) if group.directory is not None else group

    def load(self, stage, *, required=()):
        """Read a complete checkpoint; a missing or corrupt file is a cache miss."""
        if self.directory is None:
            return None
        compressed = self.directory / (stage + ".json.gz")
        try:
            if compressed.is_file():
                with gzip.open(compressed, "rb") as handle:
                    value = json.load(handle)
            else:
                value = json.loads((self.directory / (stage + ".json")).read_bytes())
        except (OSError, EOFError, ValueError, RecursionError, zlib.error):
            return None
        if not isinstance(value, dict) or not all(key in value for key in required):
            return None
        self.counts["reused"] += 1
        return value

    def save(self, stage, value):
        """Publish the successful stage atomically, leaving analysis usable on disk errors."""
        if self.directory is None:
            return
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with NamedTemporaryFile(dir=self.directory, delete=False) as handle:
                temporary = Path(handle.name)
                payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                handle.write(gzip.compress(payload.encode(), compresslevel=3, mtime=0))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.directory / (stage + ".json.gz"))
            self.counts["saved"] += 1
        except (OSError, ValueError):
            self.counts["writeFailures"] += 1
            LOGGER.warning("Could not checkpoint media stage %s", stage, exc_info=True)
        finally:
            if temporary:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove incomplete media checkpoint", exc_info=True)
