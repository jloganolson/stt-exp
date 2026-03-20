#!/usr/bin/env bash
set -euo pipefail

PARAKEET_PYTHON="${PARAKEET_PYTHON:-/home/logan/Projects/parakeet-exp/.venv/bin/python}"
PARAKEET_DEVICE="${PARAKEET_DEVICE:-cuda}"
SHERPA_MODEL_DIR="${SHERPA_MODEL_DIR:-/home/logan/Projects/sherpa-exp/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14}"

if [ "$#" -eq 0 ]; then
  ARGS=(--providers deepgram sherpa parakeet --chunk-ms 40)
else
  ARGS=("$@")
fi

export PARAKEET_PYTHON
export PARAKEET_DEVICE
export SHERPA_MODEL_DIR

uv run stt-exp live --parakeet-device "${PARAKEET_DEVICE}" "${ARGS[@]}"
