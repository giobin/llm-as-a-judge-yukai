#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE_DIR="${ROOT_DIR}/judge_results"
MODEL_NAME="gpt-5.2"
BACKEND="openai"
DATASET_SUBSET="gen"
SPLIT="test"
CHUNK_MAX_SAMPLES="${CHUNK_MAX_SAMPLES:-240}"
CHUNK_OFFSETS=(0 240)
NUM_REFERENCES_LIST=(7 1)

mkdir -p "${RESULTS_BASE_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY non impostata."
  exit 1
fi

run_eval() {
  local lang="$1"
  local dataset="$2"
  local prompt_kind="$3"
  local prompt_file="$4"
  local num_references="$5"
  local offset_samples="$6"
  local max_samples="$7"
  local results_dir

  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}"
    exit 1
  fi

  local dataset_slug model_slug output_file
  dataset_slug="${dataset##*/}"
  model_slug="${MODEL_NAME//[^a-zA-Z0-9]/_}"
  if [[ "${offset_samples}" == "0" ]]; then
    results_dir="${RESULTS_BASE_DIR}/gpt52_caput"
  elif [[ "${offset_samples}" == "240" ]]; then
    results_dir="${RESULTS_BASE_DIR}/gpt52_caput_offset"
  else
    results_dir="${RESULTS_BASE_DIR}/gpt52_caput_offset_${offset_samples}"
  fi
  mkdir -p "${results_dir}"
  output_file="${results_dir}/eval_${lang}_${prompt_kind}_${dataset_slug}_${model_slug}_num_refs_${num_references}_offset_${offset_samples}_max_${max_samples}.json"

  echo "------------------------------------------------------------"
  echo "Lang: ${lang} | Prompt: ${prompt_kind}"
  echo "Dataset: ${dataset} (subset: ${DATASET_SUBSET}, split: ${SPLIT})"
  echo "Num references: ${num_references}"
  echo "Prompt file: ${prompt_file}"
  echo "Offset samples: ${offset_samples}"
  echo "Max samples: ${max_samples}"
  echo "Output: ${output_file}"

  llm-judge \
    --backend "${BACKEND}" \
    --model-name "${MODEL_NAME}" \
    --dataset-name "${dataset}" \
    --dataset-subset "${DATASET_SUBSET}" \
    --split "${SPLIT}" \
    --num-references "${num_references}" \
    --offset-samples "${offset_samples}" \
    --max-samples "${max_samples}" \
    --prompt-file "${prompt_file}" \
    --output-file "${output_file}"
}

for offset in "${CHUNK_OFFSETS[@]}"; do
  for num_refs in "${NUM_REFERENCES_LIST[@]}"; do
    run_eval "ita" "caput/MAIA_ita" "simple" "${ROOT_DIR}/prompts/judge_prompt_ita.txt" "${num_refs}" "${offset}" "${CHUNK_MAX_SAMPLES}"
    run_eval "eng" "caput/MAIA_eng" "simple" "${ROOT_DIR}/prompts/judge_prompt_en.txt" "${num_refs}" "${offset}" "${CHUNK_MAX_SAMPLES}"
    run_eval "spa" "caput/MAIA_spa" "simple" "${ROOT_DIR}/prompts/judge_prompt_es.txt" "${num_refs}" "${offset}" "${CHUNK_MAX_SAMPLES}"
  done
done

echo "------------------------------------------------------------"
echo "Completato. Report disponibili in:"
echo "  - ${RESULTS_BASE_DIR}/gpt52_caput (offset 0)"
echo "  - ${RESULTS_BASE_DIR}/gpt52_caput_offset (offset 240)"
