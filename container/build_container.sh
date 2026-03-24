#!/bin/bash
# Pull the same base Singularity container used by trustworthiness.
# Usage: sbatch build_container.sh <output_path.sif>
# Example: sbatch build_container.sh /leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif

#SBATCH --job-name=singularity_build
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --mem=30000
#SBATCH --output=singularity_build_%j.log

set -euo pipefail

OUTPUT_SIF="${1:?Usage: sbatch build_container.sh <output_path.sif>}"
OUTPUT_DIR="$(dirname "${OUTPUT_SIF}")"
VLLM_VERSION="${VLLM_VERSION:-v0.17.0}"

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_ENGINE="${CONTAINER_ENGINE:-apptainer}"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_ENGINE="${CONTAINER_ENGINE:-singularity}"
else
  echo "ERROR: neither apptainer nor singularity was found in PATH." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

TMPDIR_BUILD="/scratch_local/${USER}_singularity_build_$$"
mkdir -p "${TMPDIR_BUILD}"
trap 'rm -rf "${TMPDIR_BUILD}"' EXIT

if [[ "${CONTAINER_ENGINE}" == "apptainer" ]]; then
  export APPTAINER_TMPDIR="${TMPDIR_BUILD}"
  export APPTAINER_CACHEDIR="${TMPDIR_BUILD}/cache"
else
  export SINGULARITY_TMPDIR="${TMPDIR_BUILD}"
  export SINGULARITY_CACHEDIR="${TMPDIR_BUILD}/cache"
fi

echo "==> Container engine: ${CONTAINER_ENGINE}"
echo "==> Starting container pull at $(date)"
"${CONTAINER_ENGINE}" pull "${OUTPUT_SIF}" "docker://vllm/vllm-openai:${VLLM_VERSION}"

echo "==> Pull completed at $(date)"
ls -lh "${OUTPUT_SIF}"
