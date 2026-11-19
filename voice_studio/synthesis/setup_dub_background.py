"""Install the fixed TorchAudio accompaniment checkpoint for local dubbing."""

from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

from voice_paths import MODELS_DIR, publish_file, temporary_directory

MODEL = MODELS_DIR / "separation" / "hdemucs_high_trained.pt"

URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"


def digest(path):
    """Compute SHA-256 without loading all model weights into memory."""
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def install():
    """Download to a private temporary directory and publish only verified weights."""
    if MODEL.is_file() and digest(MODEL) == SHA256:
        print("HDemucs already installed.")
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="voice-install-hdemucs-") as scratch:
        staged = Path(scratch) / MODEL.name
        with urlopen(URL, timeout=60) as source, staged.open("wb") as output:
            copyfileobj(source, output)
        if digest(staged) != SHA256:
            raise ValueError("The HDemucs checkpoint failed SHA-256 verification.")
        staged.chmod(0o600)
        publish_file(staged, MODEL)
    print("HDemucs installed.")


if __name__ == "__main__":
    install()
