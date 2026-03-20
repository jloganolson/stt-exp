#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-models/voxtral-optimized}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

uv run --extra voxtral --frozen vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager
