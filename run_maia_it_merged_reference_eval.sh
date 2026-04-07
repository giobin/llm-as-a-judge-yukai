#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.llm_as_a_judge_venv}"
RESULTS_DIR="${ROOT_DIR}/judge_results/maia_it_merged_reference"
LOGS_DIR="${ROOT_DIR}/logs"

DATASET_NAME="${DATASET_NAME:-caput/MAIA_ita}"
DATASET_SUBSET="${DATASET_SUBSET:-gen}"
SPLIT="${SPLIT:-test}"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-300}"
TEMPERATURE="${TEMPERATURE:-0.0}"

REFERENCE_OVERRIDES_JSON="${REFERENCE_OVERRIDES_JSON:-${ROOT_DIR}/data/references_merged.json}"
REFERENCE_OVERRIDES_ID_FIELD="${REFERENCE_OVERRIDES_ID_FIELD:-ID}"
REFERENCE_OVERRIDES_VALUE_FIELD="${REFERENCE_OVERRIDES_VALUE_FIELD:-merged_reference}"

PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/prompts/image_prompts/judge_prompt_ita.txt}"

RUN_GPT52="${RUN_GPT52:-1}"
RUN_GEMMA27="${RUN_GEMMA27:-1}"

GPT_JUDGE_MODEL="${GPT_JUDGE_MODEL:-gpt-5.2}"
GEMMA_JUDGE_MODEL="${GEMMA_JUDGE_MODEL:-google/gemma-3-27b-it}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
GEMMA27_GPUS="${GEMMA27_GPUS:-2,3}"

VLLM_PID=""

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

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

check_requirements() {
  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "ERROR: prompt file non trovato: ${PROMPT_FILE}"
    exit 1
  fi

  if [[ ! -f "${REFERENCE_OVERRIDES_JSON}" ]]; then
    echo "ERROR: file reference overrides non trovato: ${REFERENCE_OVERRIDES_JSON}"
    exit 1
  fi

  if [[ "${RUN_GPT52}" == "1" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY non impostata."
    exit 1
  fi

  if [[ "${RUN_GEMMA27}" == "1" ]]; then
    if ! command -v vllm >/dev/null 2>&1; then
      echo "ERROR: comando vllm non trovato. Verifica il venv: ${VENV_DIR}"
      exit 1
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "ERROR: curl non trovato."
      exit 1
    fi
  fi

  if [[ "${RUN_GPT52}" != "1" && "${RUN_GEMMA27}" != "1" ]]; then
    echo "ERROR: almeno uno tra RUN_GPT52 e RUN_GEMMA27 deve essere impostato a 1."
    exit 1
  fi

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

start_vllm_server_for_gemma_judge() {
  local model_slug log_file tp_size
  model_slug="${GEMMA_JUDGE_MODEL//[^a-zA-Z0-9]/_}"
  log_file="${LOGS_DIR}/vllm_maia_it_merged_reference_${model_slug}.log"
  tp_size="$(gemma_tp_size)"

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per judge model: ${GEMMA_JUDGE_MODEL}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"
  echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
  echo "vLLM eager mode: disabled"

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${GEMMA_JUDGE_MODEL}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tp_size}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --disable-custom-all-reduce \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per judge: ${GEMMA_JUDGE_MODEL} (pid: ${VLLM_PID})"
}

output_file() {
  local judge_key="$1"
  local model_name="$2"
  local model_slug dataset_slug subset_slug split_slug
  model_slug="${model_name//[^a-zA-Z0-9]/_}"
  dataset_slug="${DATASET_NAME//[^a-zA-Z0-9]/_}"
  subset_slug="${DATASET_SUBSET:-default}"
  subset_slug="${subset_slug//[^a-zA-Z0-9]/_}"
  split_slug="${SPLIT//[^a-zA-Z0-9]/_}"
  echo "${RESULTS_DIR}/eval_maia_it_merged_reference_${dataset_slug}_subset_${subset_slug}_split_${split_slug}_${judge_key}_${model_slug}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
}

print_experiment_header() {
  echo "------------------------------------------------------------"
  echo "Esperimento MAIA it con merged references"
  echo "Dataset: ${DATASET_NAME} (subset: ${DATASET_SUBSET}, split: ${SPLIT})"
  echo "Reference overrides: ${REFERENCE_OVERRIDES_JSON}"
  echo "Setup judge:"
  echo "  reference = ${REFERENCE_OVERRIDES_VALUE_FIELD}"
  echo "  positivi = answer1..7"
  echo "  negativi = wrong_answer1/wrong_answer2 se presenti, altrimenti fallback a risposte di altri sample"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Prompt file: ${PROMPT_FILE}"
}

run_judge_eval() {
  local judge_key="$1"
  local backend="$2"
  local judge_model="$3"
  local output_path
  local -a cmd

  output_path="$(output_file "${judge_key}" "${judge_model}")"

  echo "------------------------------------------------------------"
  echo "Judge key: ${judge_key}"
  echo "Backend: ${backend}"
  echo "Model: ${judge_model}"
  echo "Output: ${output_path}"

  cmd=(
    llm-judge
    --backend "${backend}"
    --model-name "${judge_model}"
    --dataset-name "${DATASET_NAME}"
    --split "${SPLIT}"
    --num-references 1
    --offset-samples "${OFFSET_SAMPLES}"
    --max-samples "${MAX_SAMPLES}"
    --random-seed "${RANDOM_SEED}"
    --temperature "${TEMPERATURE}"
    --max-tokens "${MAX_TOKENS}"
    --prompt-file "${PROMPT_FILE}"
    --reference-overrides-json "${REFERENCE_OVERRIDES_JSON}"
    --reference-overrides-id-field "${REFERENCE_OVERRIDES_ID_FIELD}"
    --reference-overrides-value-field "${REFERENCE_OVERRIDES_VALUE_FIELD}"
    --output-file "${output_path}"
  )

  if [[ -n "${DATASET_SUBSET}" ]]; then
    cmd+=(--dataset-subset "${DATASET_SUBSET}")
  fi

  if [[ "${backend}" == "vllm" ]]; then
    cmd+=(
      --vllm-base-url "${VLLM_BASE_URL}"
      --vllm-api-key "${VLLM_API_KEY}"
    )
  fi

  "${cmd[@]}"
}

main() {
  activate_venv
  check_requirements
  print_experiment_header

  if [[ "${RUN_GEMMA27}" == "1" ]]; then
    start_vllm_server_for_gemma_judge
    run_judge_eval "gemma3_27b_it" "vllm" "${GEMMA_JUDGE_MODEL}"
    stop_vllm_server
  fi

  if [[ "${RUN_GPT52}" == "1" ]]; then
    run_judge_eval "gpt52" "openai" "${GPT_JUDGE_MODEL}"
  fi

  echo "------------------------------------------------------------"
  echo "Completato. Report disponibili in: ${RESULTS_DIR}"
}

main "$@"
