"""Verified model artifacts from fixed upstream locations."""

from dataclasses import dataclass
from hashlib import file_digest
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory, NamedTemporaryFile
from urllib.request import urlopen

from voice_paths import model_path
from voice_studio.provision.model_release import ARCHIVE_NAME, ModelRelease, copy_bounded
from voice_studio.vision.facial_expressions import MODEL_SHA256


SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
ZIPFORMER = "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
PYANNOTE = "sherpa-onnx-pyannote-segmentation-3-0"
EXPRESSION = "facial_expression_recognition_mobilefacenet_2022july.onnx"
ARCHIVE_HASHES = {
    f"{SHERPA}/speaker-segmentation-models/{PYANNOTE}.tar.bz2":
        "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    f"{SHERPA}/audio-tagging-models/{ZIPFORMER}.tar.bz2":
        "07e2fafcdcbc461f2816188d9b0bbafced12584030cf67d5652e549ef256a2c6",
}



@dataclass(frozen=True)
class Asset:
    """An exact file to publish under the normal external model directory."""

    name: str
    url: str
    size: int
    sha: str
    member: str | None = None
    compatible: tuple = ()

    def fingerprints(self):
        """Include recognized earlier local formats when checking an existing file."""
        return ((self.size, self.sha), *self.compatible)

    @property
    def path(self):
        """Use the same external/legacy resolution as inference."""
        return model_path(*self.name.split("/"))

    def present(self):
        """Provide a cheap presence check without rereading weights during polling."""
        try:
            path = self.path
            if not path.is_file():
                return False
            size = path.stat().st_size
            return any(size == expected for expected, _ in self.fingerprints())
        except OSError:
            return False


SPEAKERS = (
    Asset("diarization/segmentation.onnx", f"{SHERPA}/speaker-segmentation-models/{PYANNOTE}.tar.bz2",
          5992913, "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079", f"{PYANNOTE}/model.onnx"),
    Asset("diarization/embedding.onnx", f"{SHERPA}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
          39593762, "f1ee61be04000b50392a63442e5dd9feaea7e81dddd494d3647cf1c3d88a1498",
          compatible=((39593761, "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"),)),
)
SOUNDS = (
    Asset("audio-tagging/zipformer-small-int8/model.int8.onnx", f"{SHERPA}/audio-tagging-models/{ZIPFORMER}.tar.bz2",
          27038066, "69304b8a1b96bbe6b7d16c24079f0732c65faf3568a14cb82a8238c709afe76c", f"{ZIPFORMER}/model.int8.onnx"),
    Asset("audio-tagging/zipformer-small-int8/class_labels_indices.csv", f"{SHERPA}/audio-tagging-models/{ZIPFORMER}.tar.bz2",
          14675, "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429", f"{ZIPFORMER}/class_labels_indices.csv"),
)
VISION = (
    Asset("vision/yunet.onnx", f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
          232589, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    Asset("vision/sface-int8.onnx", f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
          9896933, "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a"),
    Asset("vision/nanodet.onnx", f"{ZOO}/object_detection_nanodet/object_detection_nanodet_2022nov.onnx",
          3800954, "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"),
    Asset(f"vision/{EXPRESSION}", f"https://huggingface.co/opencv/facial_expression_recognition/resolve/b473270576eabddaacffd08adb4bed27b3fc54e6/{EXPRESSION}",
          4791892, MODEL_SHA256),
)


def fetch_file(url, output, limit, progress):
    """Download from a fixed source into a caller-owned staging file."""
    with urlopen(url, timeout=45) as response:
        return copy_bounded(response, output, limit, progress)


def check_existing(path, asset):
    """Protect custom and concurrent files from silent replacement."""
    if asset.present():
        with path.open("rb") as stream:
            identity = path.stat().st_size, file_digest(stream, "sha256").hexdigest()
            if identity in asset.fingerprints():
                return
    raise ValueError(f"Existing file differs from the supported model: {asset.name}")


def install_assets(assets, progress):
    """Verify every model and share upstream archives within the installation."""
    mirror = ModelRelease()
    with TemporaryDirectory(prefix="studio-models-") as directory:
        archives = {}
        for asset in assets:
            path = asset.path
            if path.exists() or path.is_symlink():
                check_existing(path, asset)
                continue
            publish_asset(asset, path, mirror, Path(directory), archives, progress)


def copy_upstream(asset, output, directory, archives, progress):
    """Use the pinned original file or extract it from a verified upstream archive."""
    def report(received):
        progress(asset.name, received, asset.size)

    if asset.member is None:
        return fetch_file(asset.url, output, asset.size, report)
    archive = archives.get(asset.url)
    if archive is None:
        archive = directory / str(len(archives))
        with archive.open("wb") as bundle:
            _, digest = fetch_file(asset.url, bundle, 128 * 1024 * 1024,
                                   lambda count: progress(asset.name, count, None))
        if digest != ARCHIVE_HASHES.get(asset.url):
            raise ValueError(f"Archive SHA-256 does not match: {asset.name}")
        archives[asset.url] = archive
    with tarfile.open(archive, "r:bz2") as bundle:
        member = bundle.getmember(asset.member)
        if not member.isfile() or member.size != asset.size:
            raise ValueError(f"Unexpected archive contents: {asset.name}")
        with bundle.extractfile(member) as source:
            return copy_bounded(source, output, asset.size, report)


def publish_asset(asset, path, mirror, directory, archives, progress):
    """Atomically expose a single verified payload without overwriting a local file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = NamedTemporaryFile(dir=path.parent, prefix=".studio-model-", delete=False)
    temporary = Path(staging.name)
    try:
        with staging:
            copied = mirror.copy(asset.name, staging, asset.size, asset.sha,
                                 lambda count, total: progress(ARCHIVE_NAME, count, total))
            if not copied:
                identity = copy_upstream(asset, staging, directory, archives, progress)
                if identity != (asset.size, asset.sha):
                    raise ValueError(f"SHA-256 or size does not match: {asset.name}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            check_existing(path, asset)
    finally:
        temporary.unlink(missing_ok=True)
