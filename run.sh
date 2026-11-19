#!/bin/sh
set -eu
umask 077
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
voice_runtime="$(.venv/bin/python voice_paths.py runtime)"
mkdir -p "$voice_runtime/matplotlib" "$voice_runtime/huggingface"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$voice_runtime/matplotlib}"
export HF_HOME="${HF_HOME:-$voice_runtime/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export VOICE_F5_TRAINER="${VOICE_F5_TRAINER:-$PWD/trainers/f5}"
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
  export VOICE_F5_DEVICE="${VOICE_F5_DEVICE:-mps}"
fi
exec .venv/bin/python ag_media_studio.py "$@"
