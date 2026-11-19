"""Install the pinned small OpenCV Zoo expression checkpoint and its Apache 2.0 provenance."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_paths import model_path, publish_file, temporary_directory
from voice_studio.vision.facial_expressions import MODEL_ID, MODEL_SHA256

REVISION = "b473270576eabddaacffd08adb4bed27b3fc54e6"
FILENAME = "facial_expression_recognition_mobilefacenet_2022july.onnx"
BASE = f"https://huggingface.co/opencv/facial_expression_recognition/resolve/{REVISION}"


def install():
    """Verify the weight before publication and retain the publisher's license declaration."""
    destination = model_path("vision", FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {FILENAME: f"{BASE}/{FILENAME}",
             "facial-expression-README.md": f"{BASE}/README.md",
             "facial-expression-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/LICENSE"}
    with temporary_directory() as folder:
        staging = Path(folder)
        for name, url in files.items():
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if name == FILENAME and sha256(data).hexdigest() != MODEL_SHA256:
                raise ValueError("Downloaded expression checkpoint does not match its pinned SHA-256")
            (staging / name).write_bytes(data)
        receipt = {"model": MODEL_ID, "revision": REVISION, "sha256": MODEL_SHA256,
                   "bytes": (staging / FILENAME).stat().st_size, "sources": files, "license": "Apache-2.0"}
        receipt_name = "facial-expression-provenance.json"
        (staging / receipt_name).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        for name in (*files, receipt_name):
            publish_file(staging / name, destination.parent / name)
    print(json.dumps({"model": str(destination), **receipt}, ensure_ascii=False))


if __name__ == "__main__":
    install()
