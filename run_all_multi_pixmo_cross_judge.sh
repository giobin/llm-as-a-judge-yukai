#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/data01/gbonetta/llm-as-a-judge/.llm_as_a_judge_venv}"
INFER_BASE_DIR="${ROOT_DIR}/model_infer_results"
RESULTS_BASE_DIR="${ROOT_DIR}/judge_results/multi_pixmo_cross_judge"
LOGS_DIR="${ROOT_DIR}/logs"

OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
NUM_REFERENCES="${NUM_REFERENCES:-1}"
GENERATED_FIELD="${GENERATED_FIELD:-generated_answer1}"
GENERATED_SAMPLE_MODE="${GENERATED_SAMPLE_MODE:-generated_field}"

GEMMA_JUDGE_MODEL="${GEMMA_JUDGE_MODEL:-google/gemma-3-27b-it}"
GPT_JUDGE_MODEL="${GPT_JUDGE_MODEL:-gpt-5.2}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"

SUBSETS=("it" "en" "es")
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

check_requirements() {
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

prompt_file_for_subset() {
  local subset="$1"
  case "${subset}" in
    it) echo "${ROOT_DIR}/prompts/image_prompts/judge_prompt_ita.txt" ;;
    en) echo "${ROOT_DIR}/prompts/image_prompts/judge_prompt_en.txt" ;;
    es) echo "${ROOT_DIR}/prompts/image_prompts/judge_prompt_es.txt" ;;
    *)
      echo "ERROR: subset non supportato: ${subset}" >&2
      return 1
      ;;
  esac
}

source_json_for_model_and_subset() {
  local source_dir="$1"
  local source_suffix="$2"
  local subset="$3"
  echo "${INFER_BASE_DIR}/${source_dir}/infer_${subset}_simple_multi_pixmo_cap_${source_suffix}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
}

run_llm_judge() {
  python -m llm_judge_pipeline.cli "$@"
}

validate_generated_sample_mode() {
  case "${GENERATED_SAMPLE_MODE}" in
    generated_field|transcript_pair) ;;
    *)
      echo "ERROR: GENERATED_SAMPLE_MODE deve essere 'generated_field' oppure 'transcript_pair'. Valore attuale: ${GENERATED_SAMPLE_MODE}"
      exit 1
      ;;
  esac
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
  log_file="${LOGS_DIR}/vllm_cross_judge_${model_slug}_multi_pixmo_cap.log"
  tp_size=$(awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}")

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per judge model: ${GEMMA_JUDGE_MODEL}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"
  echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${GEMMA_JUDGE_MODEL}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tp_size}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --disable-custom-all-reduce \
      --enforce-eager \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per judge: ${GEMMA_JUDGE_MODEL} (pid: ${VLLM_PID})"
}

run_judge_eval() {
  local judge_key="$1"
  local backend="$2"
  local judge_model="$3"
  local source_dir="$4"
  local source_suffix="$5"
  local subset="$6"

  local input_file prompt_file output_dir output_file judge_slug source_slug mode_suffix
  input_file="$(source_json_for_model_and_subset "${source_dir}" "${source_suffix}" "${subset}")"
  prompt_file="$(prompt_file_for_subset "${subset}")"

  if [[ ! -f "${input_file}" ]]; then
    echo "ERROR: input JSON non trovato: ${input_file}"
    exit 1
  fi

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  judge_slug="${judge_model//[^a-zA-Z0-9]/_}"
  source_slug="${source_suffix}"
  output_dir="${RESULTS_BASE_DIR}/judge_${judge_key}"
  mkdir -p "${output_dir}"
  mode_suffix=""
  if [[ "${GENERATED_SAMPLE_MODE}" != "generated_field" ]]; then
    mode_suffix="_sample_mode_${GENERATED_SAMPLE_MODE}"
  fi
  output_file="${output_dir}/eval_${subset}_generated_from_${source_slug}_judged_by_${judge_slug}_num_refs_${NUM_REFERENCES}${mode_suffix}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Judge model: ${judge_model} (backend: ${backend})"
  echo "Input generations: ${input_file}"
  echo "Subset: ${subset}"
  echo "Prompt file: ${prompt_file}"
  echo "Num references: ${NUM_REFERENCES}"
  echo "Generated sample mode: ${GENERATED_SAMPLE_MODE}"
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
      --input-json "${input_file}" \
      --generated-field "${GENERATED_FIELD}" \
      --generated-sample-mode "${GENERATED_SAMPLE_MODE}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${prompt_file}" \
      --output-file "${output_file}"
  else
    run_llm_judge \
      --backend "${backend}" \
      --model-name "${judge_model}" \
      --candidate-source generated \
      --input-json "${input_file}" \
      --generated-field "${GENERATED_FIELD}" \
      --generated-sample-mode "${GENERATED_SAMPLE_MODE}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${prompt_file}" \
      --output-file "${output_file}"
  fi
}

main() {
  local source_dir source_suffix subset source

  activate_venv
  validate_generated_sample_mode
  check_requirements

  start_vllm_server_for_gemma_judge
  for source in \
    "gemma3_27b_it_multi_pixmo_cap:google_gemma_3_27b_it" \
    "gpt52_multi_pixmo_cap:gpt_5_2"; do
    source_dir="${source%%:*}"
    source_suffix="${source##*:}"
    for subset in "${SUBSETS[@]}"; do
      run_judge_eval "gemma3_27b_it" "vllm" "${GEMMA_JUDGE_MODEL}" "${source_dir}" "${source_suffix}" "${subset}"
    done
  done
  stop_vllm_server

  for source in \
    "gemma3_27b_it_multi_pixmo_cap:google_gemma_3_27b_it" \
    "gpt52_multi_pixmo_cap:gpt_5_2"; do
    source_dir="${source%%:*}"
    source_suffix="${source##*:}"
    for subset in "${SUBSETS[@]}"; do
      run_judge_eval "gpt52" "openai" "${GPT_JUDGE_MODEL}" "${source_dir}" "${source_suffix}" "${subset}"
    done
  done

  echo "------------------------------------------------------------"
  echo "Completato. Report disponibili in: ${RESULTS_BASE_DIR}"
}

main "$@"
