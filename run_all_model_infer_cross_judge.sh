#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_BASE_DIR="${ROOT_DIR}/model_infer_results"
RESULTS_BASE_DIR="${ROOT_DIR}/judge_results/model_infer_cross_judge"
LOGS_DIR="${ROOT_DIR}/logs"

OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
NUM_REFERENCES="${NUM_REFERENCES:-4}"
GENERATED_FIELD="${GENERATED_FIELD:-generated_answer1}"

GEMMA_JUDGE_MODEL="google/gemma-3-27b-it"
GPT_JUDGE_MODEL="gpt-5.2"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT_SECONDS="${VLLM_TIMEOUT_SECONDS:-600}"
VLLM_POLL_INTERVAL="${VLLM_POLL_INTERVAL:-5}"
GEMMA27_GPUS="${GEMMA27_GPUS:-4,5}"

VLLM_PID=""

mkdir -p "${RESULTS_BASE_DIR}" "${LOGS_DIR}"

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

start_vllm_server_for_gemma_judge() {
  local model_slug="${GEMMA_JUDGE_MODEL//[^a-zA-Z0-9]/_}"
  local log_file="${LOGS_DIR}/vllm_cross_judge_${model_slug}.log"
  local tp_size
  tp_size=$(awk -F',' '{print NF}' <<< "${GEMMA27_GPUS}")

  stop_vllm_server

  echo "------------------------------------------------------------"
  echo "Avvio vLLM per judge model: ${GEMMA_JUDGE_MODEL}"
  echo "Base URL: ${VLLM_BASE_URL}"
  echo "Log: ${log_file}"
  echo "Env modello: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GEMMA27_GPUS}"

  env NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GEMMA27_GPUS}" \
    vllm serve "${GEMMA_JUDGE_MODEL}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tp_size}" \
      --disable-custom-all-reduce \
      --enforce-eager \
      >"${log_file}" 2>&1 &
  VLLM_PID=$!

  wait_for_vllm_server "${log_file}"
  echo "vLLM pronto per judge: ${GEMMA_JUDGE_MODEL} (pid: ${VLLM_PID})"
}

prompt_file_for_lang() {
  local lang="$1"
  case "${lang}" in
    ita) echo "${ROOT_DIR}/prompts/judge_prompt_ita.txt" ;;
    eng) echo "${ROOT_DIR}/prompts/judge_prompt_en.txt" ;;
    spa) echo "${ROOT_DIR}/prompts/judge_prompt_es.txt" ;;
    *) echo "ERROR: lingua non supportata: ${lang}" >&2; exit 1 ;;
  esac
}

run_judge_eval() {
  local judge_key="$1"          # gemma | gpt
  local backend="$2"            # vllm | openai
  local judge_model="$3"
  local source_dir="$4"         # gemma3_27b_it | gpt52
  local source_suffix="$5"      # google_gemma_3_27b_it | gpt_5_2
  local lang="$6"               # ita | eng | spa

  local dataset_slug="MAIA_${lang}"
  local input_file="${INFER_BASE_DIR}/${source_dir}/infer_${lang}_simple_${dataset_slug}_${source_suffix}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
  local prompt_file
  prompt_file="$(prompt_file_for_lang "${lang}")"

  if [[ ! -f "${input_file}" ]]; then
    echo "ERROR: input JSON non trovato: ${input_file}"
    exit 1
  fi
  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  local judge_slug source_slug output_dir output_file
  judge_slug="${judge_model//[^a-zA-Z0-9]/_}"
  source_slug="${source_suffix}"
  output_dir="${RESULTS_BASE_DIR}/judge_${judge_key}"
  mkdir -p "${output_dir}"
  output_file="${output_dir}/eval_${lang}_generated_from_${source_slug}_judged_by_${judge_slug}_num_refs_${NUM_REFERENCES}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"

  echo "------------------------------------------------------------"
  echo "Judge model: ${judge_model} (backend: ${backend})"
  echo "Input generations: ${input_file}"
  echo "Lang: ${lang}"
  echo "Prompt file: ${prompt_file}"
  echo "Num references: ${NUM_REFERENCES}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Output: ${output_file}"

  if [[ "${backend}" == "vllm" ]]; then
    llm-judge \
      --backend "${backend}" \
      --model-name "${judge_model}" \
      --vllm-base-url "${VLLM_BASE_URL}" \
      --vllm-api-key "${VLLM_API_KEY}" \
      --candidate-source generated \
      --input-json "${input_file}" \
      --generated-field "${GENERATED_FIELD}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${prompt_file}" \
      --output-file "${output_file}"
  else
    llm-judge \
      --backend "${backend}" \
      --model-name "${judge_model}" \
      --candidate-source generated \
      --input-json "${input_file}" \
      --generated-field "${GENERATED_FIELD}" \
      --num-references "${NUM_REFERENCES}" \
      --offset-samples "${OFFSET_SAMPLES}" \
      --max-samples "${MAX_SAMPLES}" \
      --prompt-file "${prompt_file}" \
      --output-file "${output_file}"
  fi
}

# 1) Gemma as Judge (via vLLM) on generations from Gemma + GPT
start_vllm_server_for_gemma_judge
for source in "gemma3_27b_it:google_gemma_3_27b_it" "gpt52:gpt_5_2"; do
  source_dir="${source%%:*}"
  source_suffix="${source##*:}"
  for lang in ita eng spa; do
    run_judge_eval "gemma3_27b_it" "vllm" "${GEMMA_JUDGE_MODEL}" "${source_dir}" "${source_suffix}" "${lang}"
  done
done
stop_vllm_server

# 2) GPT-5.2 as Judge (OpenAI) on generations from Gemma + GPT
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY non impostata (necessaria per judge ${GPT_JUDGE_MODEL})."
  exit 1
fi

for source in "gemma3_27b_it:google_gemma_3_27b_it" "gpt52:gpt_5_2"; do
  source_dir="${source%%:*}"
  source_suffix="${source##*:}"
  for lang in ita eng spa; do
    run_judge_eval "gpt52" "openai" "${GPT_JUDGE_MODEL}" "${source_dir}" "${source_suffix}" "${lang}"
  done
done

echo "------------------------------------------------------------"
echo "Completato. Report disponibili in: ${RESULTS_BASE_DIR}"
