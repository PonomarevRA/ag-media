"""Install and verify the pinned MERL three-stem model and its MIT notice."""

import argparse
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_paths import MODELS_DIR, publish_file, temporary_directory

REVISION = "19b3de827ebc4bfb014570cf92dd32b4ee3b6921"
REPO = "merlresearch/cocktail-fork-separation"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}"
WEIGHTS = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}/checkpoints"
FILES = {
    "mrx.py": (RAW, "c6cd543c73c0cf0f8b55628dd3966aa0474ebb00d39d4ff3784c7f4e51b1aa1c"),
    "LICENSE.md": (RAW, "30bc570d294d0a703a0ebe37d7bb37c360d4f9d1dbe707eead3c805b13fca759"),
    "default_mrx_pre_trained_weights.pth": (
        WEIGHTS, "9cecc071c9e58afda34d1a98609a1540f61320e718bafecc4c9c1c5e3ed20a0c"),
}


def install(target, local):
    """Validate every file before publishing a local model installation."""
    target.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="voice-install-mrx-") as scratch:
        folder = Path(scratch)
        for name, (base, expected) in FILES.items():
            source = (local / name).open("rb") if local else urlopen(f"{base}/{name}", timeout=90)
            with source, (folder / name).open("wb") as output:
                copyfileobj(source, output)
            with (folder / name).open("rb") as staged:
                if file_digest(staged, "sha256").hexdigest() != expected:
                    raise ValueError(f"MRX checksum mismatch: {name}")
        for name in FILES:
            publish_file(folder / name, target / name)


def main():
    """Use the upstream release or a locally downloaded copy of its files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    target = MODELS_DIR / "separation/mrx"
    install(target, args.source_dir)
    print(target)


if __name__ == "__main__":
    main()
