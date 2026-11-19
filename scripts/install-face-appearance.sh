#!/bin/sh
# Install the pinned optional adult appearance model in the configured model and dependency directories.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
model_root="$(python3 voice_paths.py models)/vision/openvino-age-gender"
runtime="$(python3 voice_paths.py dependencies)/face-attributes"
mkdir -p "$model_root"
staging=$(mktemp -d "$model_root/.download.XXXXXX")
trap 'rm -rf "$staging"' EXIT
source_url='https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16'
for extension in xml bin; do
  curl --fail --silent --show-error --location --retry 2 \
    "$source_url/age-gender-recognition-retail-0013.$extension" -o "$staging/model.$extension"
done
(
  cd "$staging"
  shasum -a 384 -c <<'HASHES'
a3c4c149f644a8cbcf81fb9fe5c9eb2500c5a102c6dea8ed9ef748eb60e07a0205371d5a839f26592848433971ea8547  model.xml
667b71d543f455e5683c70de8ae6db303e8c2362b1e25d3b2539a2ce80bb689ccedefb79b2111b0b8cb6dafcbb2cc5b4  model.bin
HASHES
)
uv pip install --python .engines/mediapipe/bin/python --no-deps --target "$runtime" 'openvino==2025.4.1'
mv "$staging/model.bin" "$model_root/model.bin"
mv "$staging/model.xml" "$model_root/model.xml"
