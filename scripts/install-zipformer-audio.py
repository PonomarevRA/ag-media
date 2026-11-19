"""Install the pinned Zipformer audio tagging weights from the official Sherpa release."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_paths import MODELS_DIR, publish_file, temporary_directory

RELEASE = "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/{RELEASE}.tar.bz2"
SHA256 = "07e2fafcdcbc461f2816188d9b0bbafced12584030cf67d5652e549ef256a2c6"
FILES = ("class_labels_indices.csv", "model.int8.onnx")


def install(target, local_archive):
    """Verify the archive, then publish only the INT8 model and its class labels."""
    target.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="voice-install-zipformer-") as folder:
        work = Path(folder)
        archive = local_archive if local_archive else work / "release.tar.bz2"
        if local_archive is None:
            with urlopen(URL, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            if file_digest(source, "sha256").hexdigest() != SHA256:
                raise ValueError("ZIPFORMER release checksum mismatch.")
        with tarfile.open(archive, "r:bz2") as bundle:
            for name in FILES:
                with bundle.extractfile(f"{RELEASE}/{name}") as source, (work / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        for name in FILES:
            publish_file(work / name, target / name)


def main():
    """Download the release or reuse an already downloaded archive."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the official release.")
    args = parser.parse_args()
    target = MODELS_DIR / "audio-tagging/zipformer-small-int8"
    install(target, args.archive)
    print(target)


if __name__ == "__main__":
    main()
