#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
base=https://github.com/k2-fsa/sherpa-onnx/releases/download
bundle=sherpa-onnx-pyannote-segmentation-3-0
curl -fLsS "$base/speaker-segmentation-models/$bundle.tar.bz2" -o "$work/segmentation.tar.bz2"
tar -xOf "$work/segmentation.tar.bz2" "$bundle/model.onnx" > "$work/segmentation.onnx"
curl -fLsS "$base/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" -o "$work/embedding.onnx"
target="$(python3 voice_paths.py models)/diarization"
mkdir -p "$target"
mv "$work/segmentation.onnx" "$work/embedding.onnx" "$target/"
