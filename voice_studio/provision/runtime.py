"""Install compatible wheel runtimes without modifying the bundled interpreter."""

from contextlib import suppress
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from voice_studio.core.optional_runtime import runtime_path, runtime_ready
from voice_studio.provision.assets import fetch_file


PIP = "https://files.pythonhosted.org/packages/c9/bc/b7db44f5f39f9d0494071bddae6880eb645970366d0a200022a1a93d57f5/pip-25.0.1-py3-none-any.whl"
PIP_INTEGRITY = (1841526, "c46efd13b6aa8279f33f2864459c8ce587ea6a1a59ee20de055868d8f7688f7f")
PACKAGES = {
    "sherpa": (("sherpa-onnx==1.13.8",), "import sherpa_onnx"),
    "vision": (("opencv-python-headless==4.13.0.92",), "import cv2; assert hasattr(cv2, 'FaceDetectorYN')"),
    "mediapipe": (("mediapipe==0.10.32",),
                  "import cv2, mediapipe as mp; assert mp.tasks.vision.FaceLandmarker and mp.tasks.vision.ImageSegmenter"),
    "light-asd": (("scipy==1.18.1",), "import torch; from scipy.io import wavfile; assert torch.version.cuda is None"),
    "dub-background": (("torchaudio==2.8.0", "numpy==2.5.5", "soundfile==0.14.0", "cffi==2.0.0", "pycparser==2.23"),
                       "import soundfile; from torchaudio.models import hdemucs_high; assert callable(hdemucs_high)"),
    "mlx": (("mlx-audio==0.6.5", "mlx==0.32.4", "soundfile"),
            "import mlx.core; from mlx_audio.tts.utils import load_model; import soundfile"),
}
PIP_BOOTSTRAP = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')"


def install_runtime(group):
    """Verify a staged native import before publishing the whole runtime."""
    if runtime_ready(group):
        return
    target = runtime_path(group)
    requirements, probe = PACKAGES[group]
    options = []
    probe_runtimes = []
    if group == "light-asd":
        if sys.platform == "darwin":
            requirements = (*requirements, "torch==2.8.0")
        else:
            requirements = (*requirements, "torch==2.8.0+cpu")
            options = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    elif group == "dub-background":
        # Audio separation shares Torch with Light-ASD. Older Light-ASD installs relied on vision for NumPy.
        install_runtime("light-asd")
        options = ["--no-deps"]
        probe_runtimes = [str(runtime_path("light-asd"))]
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix=".studio-install-") as folder:
        work = Path(folder)
        archive = work / Path(urlparse(PIP).path).name
        with archive.open("wb") as stream:
            if fetch_file(PIP, stream, PIP_INTEGRITY[0], lambda count: None) != PIP_INTEGRITY:
                raise ValueError("pip bootstrap SHA-256 does not match")
        staged = work / "runtime"
        command = [sys.executable, "-I", "-B", "-c", PIP_BOOTSTRAP, str(archive),
                   "--isolated", "--disable-pip-version-check", "install", "--no-input",
                   "--only-binary=:all:", "--ignore-installed", "--no-compile", "--index-url", "https://pypi.org/simple",
                   *options, "--target", str(staged), *requirements]
        env = {**os.environ, "PIP_CONFIG_FILE": os.devnull}
        subprocess.run(command, env=env, check=True, timeout=900)
        subprocess.run([sys.executable, "-I", "-B", "-c", "import sys; sys.path[:0] = sys.argv[1:]; " + probe,
                        str(staged), *probe_runtimes], check=True, timeout=60)
        (staged / ".studio-ready").touch()
        with suppress(FileExistsError):
            if not target.exists():
                staged.rename(target)
        if not runtime_ready(group):
            raise ValueError("Runtime directory already exists without a verified installation")
