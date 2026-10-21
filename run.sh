#!/usr/bin/env bash
# Start Flint (NVFP4 GGUF) on vLLM: ./run.sh [port]. Settings via env: FLINT_KERNELS (moe,attn,dense on sm_121), GPU_UTIL,
# MAX_LEN, MAX_SEQS, NAME, IMAGE, FOREGROUND=1 (stay attached; stops the container on exit).
set -euo pipefail
PORT=${1:-${PORT:-8000}}
NAME=${NAME:-flint}; CONTAINER=${CONTAINER:-flint-vllm}; IMAGE=${IMAGE:-flint-vllm-spark:pr62938}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}; CACHE=${CACHE:-$HOME/.cache/flint-vllm}
mkdir -p "$CACHE"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --gpus all --network host --ipc host \
  -v "$HF_CACHE":/root/.cache/huggingface -v "$CACHE":/cache \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_FLINT_KERNELS="${FLINT_KERNELS:-}" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton -e CUTE_DSL_CACHE_DIR=/cache/cutedsl \
  "$IMAGE" RecursiveSI/flint-nvfp4-gguf --hf-config-path RecursiveSI/flint --tokenizer RecursiveSI/flint \
  --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" \
  --max-model-len "${MAX_LEN:-262144}" --max-num-seqs "${MAX_SEQS:-8}" --gpu-memory-utilization "${GPU_UTIL:-0.40}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 --exclude-tools-when-tool-choice-none \
  ${EXTRA_ARGS:-} >/dev/null
if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap 'docker stop -t 30 "$CONTAINER" >/dev/null 2>&1; docker rm "$CONTAINER" >/dev/null 2>&1' INT TERM EXIT
  docker logs -f "$CONTAINER"
fi
