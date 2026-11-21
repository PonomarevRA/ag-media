"""Install the fixed TorchAudio accompaniment checkpoint for local dubbing."""

from voice_studio.provision.assets import DUB_BACKGROUND, install_assets


def install():
    """Download to a private temporary directory and publish only verified weights."""
    install_assets(DUB_BACKGROUND, lambda name, received, total: None)
    print("HDemucs installed.")


if __name__ == "__main__":
    install()
