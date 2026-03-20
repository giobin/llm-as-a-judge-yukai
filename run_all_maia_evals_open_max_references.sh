#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/judge_results/open_models_caput_max_references"
LOGS_DIR="${ROOT_DIR}/logs"
BACKEND="vllm"
DATASET_SUBSET="gen"
SPLIT="test"
NUM_REFERENCES_LIST=(7 1)
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"
SMALL_MODELS_GPU="${SMALL_MODELS_GPU:-${GEMMA27_GPUS%%,*}}"

MODELS=(
  "google/gemma-3-27b-it"
  "google/gemma-3-12b-it"
  "openai/gpt-oss-20b"
)

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
  local model_name="$1"
  local model_slug="$2"
  local log_file="${LOGS_DIR}/vllm_${model_slug}.log"
  local -a extra_args=()
  local -a model_env=()
  local -a model_args=()
  local tp_size=1

  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${VLLM_EXTRA_ARGS})
  fi

  case "${model_name}" in
    "google/gemma-3-27b-it")
      tp_size=$(awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}")
      model_env=("NCCL_P2P_DISABLE=1" "CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}")
      model_args=(
        "--tensor-parallel-size" "${tp_size}"
        "--disable-custom-all-reduce"
        "--enforce-eager"
      )
      ;;
    "google/gemma-3-12b-it"|"openai/gpt-oss-20b")
      model_env=("CUDA_VISIBLE_DEVICES=${SMALL_MODELS_GPU}")
      model_args=()
      ;;
    *)
      model_env=()
      model_args=()
      ;;
  esac

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per modello: ${model_name}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  if [[ ${#model_env[@]} -gt 0 ]]; then
    echo "Env modello: ${model_env[*]}"
  fi

  env "${model_env[@]}" vllm serve "${model_name}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    "${model_args[@]}" \
    "${extra_args[@]}" \
    >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per: ${model_name} (pid: ${VLLM_PID})"
}

run_eval() {
  local model_name="$1"
  local lang="$2"
  local dataset="$3"
  local prompt_kind="$4"
  local prompt_file="$5"
  local num_references="$6"

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  local dataset_slug model_slug output_file
  dataset_slug="${dataset##*/}"
  model_slug="${model_name//[^a-zA-Z0-9]/_}"
  output_file="${RESULTS_DIR}/eval_${lang}_${prompt_kind}_${dataset_slug}_${model_slug}_num_refs_${num_references}.json"

  echo "------------------------------------------------------------"
  echo "Model: ${model_name}"
  echo "Lang: ${lang} | Prompt: ${prompt_kind}"
  echo "Dataset: ${dataset} (subset: ${DATASET_SUBSET}, split: ${SPLIT})"
  echo "Num references: ${num_references}"
  echo "Prompt file: ${prompt_file}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Output: ${output_file}"

  llm-judge \
    --backend "${BACKEND}" \
    --model-name "${model_name}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-api-key "${VLLM_API_KEY}" \
    --dataset-name "${dataset}" \
    --dataset-subset "${DATASET_SUBSET}" \
    --split "${SPLIT}" \
    --num-references "${num_references}" \
    --offset-samples "${OFFSET_SAMPLES}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-file "${prompt_file}" \
    --output-file "${output_file}"
}

for model in "${MODELS[@]}"; do
  model_slug="${model//[^a-zA-Z0-9]/_}"
  start_vllm_server "${model}" "${model_slug}"

  for num_refs in "${NUM_REFERENCES_LIST[@]}"; do
    run_eval "${model}" "ita" "caput/MAIA_ita" "simple" "${ROOT_DIR}/prompts/judge_prompt_ita.txt" "${num_refs}"
    run_eval "${model}" "eng" "caput/MAIA_eng" "simple" "${ROOT_DIR}/prompts/judge_prompt_en.txt" "${num_refs}"
    run_eval "${model}" "spa" "caput/MAIA_spa" "simple" "${ROOT_DIR}/prompts/judge_prompt_es.txt" "${num_refs}"
  done

  stop_vllm_server
done

echo "------------------------------------------------------------"
echo "Completato. Report disponibili in: ${RESULTS_DIR}"
