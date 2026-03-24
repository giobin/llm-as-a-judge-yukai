#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF_INNER'
Usage:
  run_vllm_server_in_container.sh [image.sif] <model-name> [extra vllm args...]

Examples:
  LLM_JUDGE_SIF=/path/to/shared-vllm.sif \
    ./container/run_vllm_server_in_container.sh google/gemma-3-12b-it --dtype bfloat16

  ./container/run_vllm_server_in_container.sh \
    /path/to/shared-vllm.sif \
    google/gemma-3-12b-it \
    --dtype bfloat16
EOF_INNER
  exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity was not found in PATH." >&2
  exit 1
fi

if [[ $# -ge 2 && ( "$1" == *.sif || -f "$1" ) ]]; then
  IMAGE_PATH="$1"
  MODEL_NAME="$2"
  shift 2
else
  IMAGE_PATH="${LLM_JUDGE_SIF:?Set LLM_JUDGE_SIF or pass <image.sif> explicitly}"
  MODEL_NAME="$1"
  shift 1
fi

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
HF_CACHE_DIR="${HF_HOME:-/leonardo_work/FBKLM_prj1/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_DIR}}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "${HF_CACHE_DIR}" "${HF_DATASETS_CACHE}"

echo "==> Engine: singularity"
echo "==> Image: ${IMAGE_PATH}"
echo "==> HF_HOME: ${HF_CACHE_DIR}"
echo "==> Model: ${MODEL_NAME}"
echo "==> HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "==> TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"

BIND_ARGS=(
  --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}"
  --bind "${PWD}:${PWD}"
)

exec singularity exec --nv \
  "${BIND_ARGS[@]}" \
  "${IMAGE_PATH}" \
  bash -c 'export HF_HOME="$1"; export TRANSFORMERS_CACHE="$2"; export HF_DATASETS_CACHE="$3"; export HF_HUB_OFFLINE="$4"; export TRANSFORMERS_OFFLINE="$5"; export LD_LIBRARY_PATH=$(printf "%s" "$LD_LIBRARY_PATH" | awk -v RS=: '\''$0 !~ /cuda/ { out = out ? out ":" $0 : $0 } END { print out }'\''); shift 5; exec "$@"' _ \
  "${HF_CACHE_DIR}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_OFFLINE}" \
  "${TRANSFORMERS_OFFLINE}" \
  vllm serve "${MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  "$@"
