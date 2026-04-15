#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.llm_as_a_judge_venv}"
INPUT_JSON="${INPUT_JSON:-${ROOT_DIR}/model_infer_results/gemma3_27b_it/infer_ita_simple_MAIA_ita_google_gemma_3_27b_it_offset_0_max_240.json}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${ROOT_DIR}/judge_results/maia_it_generated_gemma_cross_judge_merged_reference}"
LOGS_DIR="${LOGS_DIR:-${ROOT_DIR}/logs}"

OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
NUM_REFERENCES="${NUM_REFERENCES:-1}"
GENERATED_SAMPLE_MODE="${GENERATED_SAMPLE_MODE:-generated_field}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/prompts/image_prompts/judge_prompt_ita.txt}"
REFERENCE_OVERRIDES_JSON="${REFERENCE_OVERRIDES_JSON:-${ROOT_DIR}/data/references_merged.json}"
REFERENCE_OVERRIDES_ID_FIELD="${REFERENCE_OVERRIDES_ID_FIELD:-ID}"
REFERENCE_OVERRIDES_VALUE_FIELD="${REFERENCE_OVERRIDES_VALUE_FIELD:-merged_reference}"

GEMMA_JUDGE_MODEL="${GEMMA_JUDGE_MODEL:-google/gemma-3-27b-it}"
GPT_JUDGE_MODEL="${GPT_JUDGE_MODEL:-gpt-5.2}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"

CANDIDATE_SPECS=(
  "generated_answer1:yes"
  "wrong_answer1:no"
  "wrong_answer2:no"
)

VLLM_PID=""

mkdir -p "${RESULTS_BASE_DIR}" "${LOGS_DIR}"

activate_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: virtualenv non trovato: ${VENV_DIR}"
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

run_llm_judge() {
  python -m llm_judge_pipeline.cli "$@"
}

check_requirements() {
  if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "ERROR: input JSON non trovato: ${INPUT_JSON}"
    exit 1
  fi

  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "ERROR: prompt file non trovato: ${PROMPT_FILE}"
    exit 1
  fi

  if [[ ! -f "${REFERENCE_OVERRIDES_JSON}" ]]; then
    echo "ERROR: file reference overrides non trovato: ${REFERENCE_OVERRIDES_JSON}"
    exit 1
  fi

  if [[ "${GENERATED_SAMPLE_MODE}" != "generated_field" ]]; then
    echo "ERROR: questo script supporta solo GENERATED_SAMPLE_MODE=generated_field."
    exit 1
  fi

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY non impostata."
    exit 1
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python non disponibile nel venv attivo."
    exit 1
  fi

  if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: comando vllm non trovato. Verifica il venv: ${VENV_DIR}"
    exit 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl non trovato."
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
  local -a vllm_args
  model_slug="${GEMMA_JUDGE_MODEL//[^a-zA-Z0-9]/_}"
  log_file="${LOGS_DIR}/vllm_cross_judge_${model_slug}_maia_it_generated_gemma.log"
  tp_size=$(awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}")

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per judge model: ${GEMMA_JUDGE_MODEL}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"
  echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
  echo "vLLM enforce eager: ${VLLM_ENFORCE_EAGER}"

  vllm_args=(
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${tp_size}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --disable-custom-all-reduce
  )

  if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    vllm_args+=(--enforce-eager)
  fi

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${GEMMA_JUDGE_MODEL}" \
      "${vllm_args[@]}" \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per judge: ${GEMMA_JUDGE_MODEL} (pid: ${VLLM_PID})"
}

output_dir_for_judge() {
  local judge_key="$1"
  echo "${RESULTS_BASE_DIR}/judge_${judge_key}"
}

run_judge_eval() {
  local judge_key="$1"
  local backend="$2"
  local judge_model="$3"
  local generated_field="$4"
  local expected_label="$5"

  local judge_slug output_dir output_file
  judge_slug="${judge_model//[^a-zA-Z0-9]/_}"
  output_dir="$(output_dir_for_judge "${judge_key}")"
  mkdir -p "${output_dir}"
  output_file="${output_dir}/eval_ita_generated_from_google_gemma_3_27b_it_field_${generated_field}_expected_${expected_label}_judged_by_${judge_slug}_num_refs_${NUM_REFERENCES}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Judge model: ${judge_model} (backend: ${backend})"
  echo "Input JSON: ${INPUT_JSON}"
  echo "Candidate field: ${generated_field}"
  echo "Expected label: ${expected_label}"
  echo "Prompt file: ${PROMPT_FILE}"
  echo "Reference overrides: ${REFERENCE_OVERRIDES_JSON}"
  echo "Num references: ${NUM_REFERENCES}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Output: ${output_file}"

  if [[ "${backend}" == "vllm" ]]; then
    run_llm_judge \
      --backend "${backend}" \
      --model-name "${judge_model}" \
      --vllm-base-url "${VLLM_BASE_URL}" \
      --vllm-api-key "${VLLM_API_KEY}" \
      --candidate-source generated \
      --input-json "${INPUT_JSON}" \
      --generated-field "${generated_field}" \
      --generated-expected-label "${expected_label}" \
      --generated-sample-mode "${GENERATED_SAMPLE_MODE}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${PROMPT_FILE}" \
      --reference-overrides-json "${REFERENCE_OVERRIDES_JSON}" \
      --reference-overrides-id-field "${REFERENCE_OVERRIDES_ID_FIELD}" \
      --reference-overrides-value-field "${REFERENCE_OVERRIDES_VALUE_FIELD}" \
      --output-file "${output_file}"
  else
    run_llm_judge \
      --backend "${backend}" \
      --model-name "${judge_model}" \
      --candidate-source generated \
      --input-json "${INPUT_JSON}" \
      --generated-field "${generated_field}" \
      --generated-expected-label "${expected_label}" \
      --generated-sample-mode "${GENERATED_SAMPLE_MODE}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${PROMPT_FILE}" \
      --reference-overrides-json "${REFERENCE_OVERRIDES_JSON}" \
      --reference-overrides-id-field "${REFERENCE_OVERRIDES_ID_FIELD}" \
      --reference-overrides-value-field "${REFERENCE_OVERRIDES_VALUE_FIELD}" \
      --output-file "${output_file}"
  fi
}

main() {
  local spec generated_field expected_label

  activate_venv
  check_requirements

  start_vllm_server_for_gemma_judge
  for spec in "${CANDIDATE_SPECS[@]}"; do
    generated_field="${spec%%:*}"
    expected_label="${spec##*:}"
    run_judge_eval "gemma3_27b_it" "vllm" "${GEMMA_JUDGE_MODEL}" "${generated_field}" "${expected_label}"
  done
  stop_vllm_server

  for spec in "${CANDIDATE_SPECS[@]}"; do
    generated_field="${spec%%:*}"
    expected_label="${spec##*:}"
    run_judge_eval "gpt52" "openai" "${GPT_JUDGE_MODEL}" "${generated_field}" "${expected_label}"
  done

  echo "------------------------------------------------------------"
  echo "Completato. Report disponibili in: ${RESULTS_BASE_DIR}"
}

main "$@"
