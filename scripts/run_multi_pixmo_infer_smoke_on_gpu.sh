#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_SCRIPT="${ROOT_DIR}/scripts/run_multi_pixmo_infer_smoke_worker.sh"

NUM_GPUS="${NUM_GPUS:-4}"
PARTITION="${PARTITION:-boost_usr_prod}"
WALLTIME="${WALLTIME:-03:00:00}"
ACCOUNT="${ACCOUNT:-FBKLM_prj1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

if [[ ! -x "${WORKER_SCRIPT}" ]]; then
  echo "ERROR: worker script non eseguibile o mancante: ${WORKER_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${LLM_JUDGE_SIF:-}" ]]; then
  echo "ERROR: LLM_JUDGE_SIF non impostata." >&2
  exit 1
fi

if [[ -z "${HF_HOME:-}" ]]; then
  echo "ERROR: HF_HOME non impostata." >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "==> Launching GPU smoke benchmark"
echo "==> ROOT_DIR: ${ROOT_DIR}"
echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"
echo "==> HF_HOME: ${HF_HOME}"
echo "==> NUM_GPUS: ${NUM_GPUS}"
echo "==> PARTITION: ${PARTITION}"
echo "==> WALLTIME: ${WALLTIME}"
echo "==> ACCOUNT: ${ACCOUNT}"

SRUN_CMD=(
  srun
  -N1
  "--gres=gpu:${NUM_GPUS}"
  -p "${PARTITION}"
  -t "${WALLTIME}"
  --account="${ACCOUNT}"
  bash -lc "cd '${ROOT_DIR}' && '${WORKER_SCRIPT}'"
)

if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${SRUN_EXTRA_ARGS} )
  SRUN_CMD=(srun "${EXTRA_ARGS[@]}" -N1 "--gres=gpu:${NUM_GPUS}" -p "${PARTITION}" -t "${WALLTIME}" --account="${ACCOUNT}" bash -lc "cd '${ROOT_DIR}' && '${WORKER_SCRIPT}'")
fi

"${SRUN_CMD[@]}"
