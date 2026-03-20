#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/model_infer_results/gemma3_27b_it"
LOGS_DIR="${ROOT_DIR}/logs"

BACKEND="vllm"
MODEL_NAME="google/gemma-3-27b-it"
DATASET_SUBSET="gen"
SPLIT="test"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_IMAGE_DIMENSION="${MAX_IMAGE_DIMENSION:-900}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"

VLLM_PID=""

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

stop_vllm_server() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
  VLLM_PID=""
}

cleanup() {
  stop_vllm_server
}

trap cleanup EXIT INT TERM

wait_for_vllm_server() {
  local log_file="$1"
  local elapsed=0

  while (( elapsed < VLLM_TIMEOUT_SECONDS )); do
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM si e' fermato prima di essere pronto. Log: ${log_file}"
      tail -n 100 "${log_file}" || true
      return 1
    fi

    if curl -fsS "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
      return 0
    fi

    sleep "${VLLM_POLL_INTERVAL}"
    elapsed=$((elapsed + VLLM_POLL_INTERVAL))
  done

  echo "ERROR: timeout in attesa di vLLM (${VLLM_TIMEOUT_SECONDS}s). Log: ${log_file}"
  tail -n 100 "${log_file}" || true
  return 1
}

start_vllm_server() {
  local model_slug="${MODEL_NAME//[^a-zA-Z0-9]/_}"
  local log_file="${LOGS_DIR}/vllm_model_infer_${model_slug}.log"
  local tp_size

  tp_size=$(awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}")

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per modello: ${MODEL_NAME}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tp_size}" \
      --disable-custom-all-reduce \
      --enforce-eager \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per: ${MODEL_NAME} (pid: ${VLLM_PID})"
}

run_infer() {
  local lang="$1"
  local dataset="$2"
  local prompt_file="$3"

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  local dataset_slug model_slug output_file
  dataset_slug="${dataset##*/}"
  model_slug="${MODEL_NAME//[^a-zA-Z0-9]/_}"
  output_file="${RESULTS_DIR}/infer_${lang}_simple_${dataset_slug}_${model_slug}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Model: ${MODEL_NAME}"
  echo "Lang: ${lang}"
  echo "Dataset: ${dataset} (subset: ${DATASET_SUBSET}, split: ${SPLIT})"
  echo "Prompt file: ${prompt_file}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Num frames: ${NUM_FRAMES}"
  echo "Output: ${output_file}"

  model-infer \
    --backend "${BACKEND}" \
    --model-name "${MODEL_NAME}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-api-key "${VLLM_API_KEY}" \
    --dataset-name "${dataset}" \
    --dataset-subset "${DATASET_SUBSET}" \
    --split "${SPLIT}" \
    --offset-samples "${OFFSET_SAMPLES}" \
    --max-samples "${MAX_SAMPLES}" \
    --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
    --prompt-file "${prompt_file}" \
    --num-frames "${NUM_FRAMES}" \
    --max-image-dimension "${MAX_IMAGE_DIMENSION}" \
    --output-file "${output_file}"
}

start_vllm_server
run_infer "ita" "caput/MAIA_ita" "${ROOT_DIR}/prompts/generation_prompt_ita.txt"
run_infer "eng" "caput/MAIA_eng" "${ROOT_DIR}/prompts/generation_prompt_en.txt"
run_infer "spa" "caput/MAIA_spa" "${ROOT_DIR}/prompts/generation_prompt_es.txt"
stop_vllm_server

echo "------------------------------------------------------------"
echo "Completato. Report disponibili in: ${RESULTS_DIR}"
