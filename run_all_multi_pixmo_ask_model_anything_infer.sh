#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/data01/gbonetta/llm-as-a-judge/.llm_as_a_judge_venv}"
LOGS_DIR="${ROOT_DIR}/logs"
RESULTS_BASE_DIR="${ROOT_DIR}/model_infer_results"
DATASET_NAME="${DATASET_NAME:-VillanovaAI/multi-pixmo-ask-model-anything}"
SPLIT="${SPLIT:-train}"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
MAX_IMAGE_DIMENSION="${MAX_IMAGE_DIMENSION:-900}"

GPT_BACKEND="openai"
GPT_MODEL_NAME="${GPT_MODEL_NAME:-gpt-5.2}"
GPT_RESULTS_DIR="${RESULTS_BASE_DIR}/gpt52_multi_pixmo_ask_model_anything"

GEMMA_BACKEND="${GEMMA_BACKEND:-vllm}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-3-27b-it}"
GEMMA_RESULTS_DIR="${RESULTS_BASE_DIR}/gemma3_27b_it_multi_pixmo_ask_model_anything"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"

SUBSETS=("it" "en" "es")
VLLM_PID=""

mkdir -p "${LOGS_DIR}" "${GPT_RESULTS_DIR}" "${GEMMA_RESULTS_DIR}"

activate_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: virtualenv non trovato: ${VENV_DIR}"
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

gemma_tp_size() {
  awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}"
}

validate_backend() {
  case "${GEMMA_BACKEND}" in
    vllm|vllm_in_process) ;;
    *)
      echo "ERROR: GEMMA_BACKEND deve essere 'vllm' oppure 'vllm_in_process'. Valore attuale: ${GEMMA_BACKEND}"
      exit 1
      ;;
  esac
}

check_requirements() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY non impostata."
    exit 1
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python non disponibile nel venv attivo."
    exit 1
  fi

  if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: package vllm non disponibile. Verifica il venv: ${VENV_DIR}"
    exit 1
  fi

  if [[ "${GEMMA_BACKEND}" == "vllm" ]]; then
    if ! command -v vllm >/dev/null 2>&1; then
      echo "ERROR: comando vllm non trovato. Verifica il venv: ${VENV_DIR}"
      exit 1
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "ERROR: curl non trovato."
      exit 1
    fi
  fi
}

prompt_file_for_subset() {
  local subset="$1"
  case "${subset}" in
    it) echo "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_ita.txt" ;;
    en) echo "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_en.txt" ;;
    es) echo "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_es.txt" ;;
    *)
      echo "ERROR: subset non supportato: ${subset}" >&2
      return 1
      ;;
  esac
}

run_model_infer() {
  python -m model_inference.cli "$@"
}

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

start_vllm_server() {
  local model_slug log_file tp_size
  model_slug="${GEMMA_MODEL_NAME//[^a-zA-Z0-9]/_}"
  log_file="${LOGS_DIR}/vllm_model_infer_${model_slug}_multi_pixmo_ask_model_anything.log"
  tp_size="$(gemma_tp_size)"

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM server per modello: ${GEMMA_MODEL_NAME}"
  echo "Dataset: ${DATASET_NAME}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${GEMMA_MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tp_size}" \
      --disable-custom-all-reduce \
      --enforce-eager \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per: ${GEMMA_MODEL_NAME} (pid: ${VLLM_PID})"
}

run_gpt_infer() {
  local subset="$1"
  local prompt_file output_file
  prompt_file="$(prompt_file_for_subset "${subset}")"

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  output_file="${GPT_RESULTS_DIR}/infer_${subset}_simple_multi_pixmo_ask_model_anything_gpt_5_2_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Model: ${GPT_MODEL_NAME}"
  echo "Subset: ${subset}"
  echo "Dataset: ${DATASET_NAME} (subset: ${subset}, split: ${SPLIT})"
  echo "Prompt file: ${prompt_file}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Max image dimension: ${MAX_IMAGE_DIMENSION}"
  echo "Output: ${output_file}"

  run_model_infer \
    --backend "${GPT_BACKEND}" \
    --model-name "${GPT_MODEL_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-subset "${subset}" \
    --split "${SPLIT}" \
    --offset-samples "${OFFSET_SAMPLES}" \
    --max-samples "${MAX_SAMPLES}" \
    --media-mode image \
    --prompt-file "${prompt_file}" \
    --max-image-dimension "${MAX_IMAGE_DIMENSION}" \
    --output-file "${output_file}"
}

run_gemma_infer() {
  local subset="$1"
  local prompt_file output_file tp_size
  local -a cmd env_prefix
  prompt_file="$(prompt_file_for_subset "${subset}")"
  tp_size="$(gemma_tp_size)"

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  output_file="${GEMMA_RESULTS_DIR}/infer_${subset}_simple_multi_pixmo_ask_model_anything_google_gemma_3_27b_it_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Model: ${GEMMA_MODEL_NAME}"
  echo "Backend: ${GEMMA_BACKEND}"
  echo "Subset: ${subset}"
  echo "Dataset: ${DATASET_NAME} (subset: ${subset}, split: ${SPLIT})"
  echo "Prompt file: ${prompt_file}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Max image dimension: ${MAX_IMAGE_DIMENSION}"
  echo "Output: ${output_file}"

  cmd=(
    python
    -m
    model_inference.cli
    --backend "${GEMMA_BACKEND}"
    --model-name "${GEMMA_MODEL_NAME}"
    --dataset-name "${DATASET_NAME}"
    --dataset-subset "${subset}"
    --split "${SPLIT}"
    --offset-samples "${OFFSET_SAMPLES}"
    --max-samples "${MAX_SAMPLES}"
    --media-mode image
    --prompt-file "${prompt_file}"
    --max-image-dimension "${MAX_IMAGE_DIMENSION}"
    --output-file "${output_file}"
  )

  if [[ "${GEMMA_BACKEND}" == "vllm" ]]; then
    cmd+=(
      --vllm-base-url "${VLLM_BASE_URL}"
      --vllm-api-key "${VLLM_API_KEY}"
    )
    "${cmd[@]}"
    return
  fi

  cmd+=(
    --vllm-tensor-parallel-size "${tp_size}"
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-enforce-eager
    --vllm-disable-custom-all-reduce
  )

  if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    cmd+=(--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi

  if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--vllm-trust-remote-code)
  fi

  env_prefix=(env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}")
  "${env_prefix[@]}" "${cmd[@]}"
}

main() {
  local subset

  activate_venv
  validate_backend
  check_requirements

  if [[ "${GEMMA_BACKEND}" == "vllm" ]]; then
    start_vllm_server
  else
    echo "------------------------------------------------------------"
    echo "Uso vLLM in-process per Gemma"
    echo "Model: ${GEMMA_MODEL_NAME}"
    echo "CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"
    echo "Tensor parallel size: $(gemma_tp_size)"
  fi

  for subset in "${SUBSETS[@]}"; do
    run_gemma_infer "${subset}"
  done

  if [[ "${GEMMA_BACKEND}" == "vllm" ]]; then
    stop_vllm_server
  fi

  for subset in "${SUBSETS[@]}"; do
    run_gpt_infer "${subset}"
  done

  echo "------------------------------------------------------------"
  echo "Completato."
  echo "Report Gemma: ${GEMMA_RESULTS_DIR}"
  echo "Report GPT-5.2: ${GPT_RESULTS_DIR}"
}

main "$@"
