#!/bin/bash
# Run llm-as-a-judge-yukai inside the shared vLLM Singularity container.
# Usage:
#   ./container/run_in_container.sh --setup
#   ./container/run_in_container.sh <python args...>
# Example:
#   ./container/run_in_container.sh --setup
#   ./container/run_in_container.sh -m llm_judge_pipeline.cli --help

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${LLM_JUDGE_SIF:?Set LLM_JUDGE_SIF to the path of your .sif container}"
VENV_DIR="${PROJECT_DIR}/.venv"
INSTALLED_MARKER="${VENV_DIR}/.installed"
export HF_HOME="${HF_HOME:-/leonardo_work/FBKLM_prj1/hf_cache}"

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_ENGINE="${CONTAINER_ENGINE:-apptainer}"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_ENGINE="${CONTAINER_ENGINE:-singularity}"
else
  echo "ERROR: neither apptainer nor singularity was found in PATH." >&2
  exit 1
fi

HOST_BINDS=()
if [ -d /etc/pki ]; then
  HOST_BINDS+=(--bind /etc/pki:/etc/pki)
fi

CONTAINER_PYTHON=$("${CONTAINER_ENGINE}" exec "${CONTAINER}" which python3)

install_dependencies() {
  echo "==> Installing project dependencies via uv sync..."
  "${CONTAINER_ENGINE}" exec \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    "${HOST_BINDS[@]}" \
    --pwd "${PROJECT_DIR}" \
    "${CONTAINER}" \
    bash -c "uv venv --python ${CONTAINER_PYTHON} --system-site-packages && uv sync --no-install-package torch --no-install-package vllm"
  touch "${INSTALLED_MARKER}"
  echo "==> Dependencies installed."
}

if [[ "${1:-}" == "--setup" ]]; then
  shift
  install_dependencies
  if [[ $# -eq 0 ]]; then
    exit 0
  fi
fi

if [[ ! -f "${INSTALLED_MARKER}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cat >&2 <<EOF_INNER
ERROR: .venv is not initialized yet, and this compute node has no internet access.

Run this once on a login node first:
  ./container/run_in_container.sh --setup

Then retry the same command on the GPU node.
EOF_INNER
    exit 1
  fi

  echo "==> First run detected on login node. Bootstrapping .venv..."
  install_dependencies
fi

echo "==> Running: $*"
"${CONTAINER_ENGINE}" exec \
  --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  "${HOST_BINDS[@]}" \
  --pwd "${PROJECT_DIR}" \
  "${CONTAINER}" \
  bash -c 'export LD_LIBRARY_PATH=$(printf "%s" "$LD_LIBRARY_PATH" | awk -v RS=: '\''$0 !~ /cuda/ { out = out ? out ":" $0 : $0 } END { print out }'\''); exec "$@"' _ "${VENV_DIR}/bin/python" "$@"
