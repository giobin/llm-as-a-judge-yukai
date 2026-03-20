#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/model_infer_results/gpt52"

BACKEND="openai"
MODEL_NAME="gpt-5.2"
DATASET_SUBSET="gen"
SPLIT="test"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_IMAGE_DIMENSION="${MAX_IMAGE_DIMENSION:-900}"

mkdir -p "${RESULTS_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY non impostata."
  exit 1
fi

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

run_infer "ita" "caput/MAIA_ita" "${ROOT_DIR}/prompts/generation_prompt_ita.txt"
run_infer "eng" "caput/MAIA_eng" "${ROOT_DIR}/prompts/generation_prompt_en.txt"
run_infer "spa" "caput/MAIA_spa" "${ROOT_DIR}/prompts/generation_prompt_es.txt"

echo "------------------------------------------------------------"
echo "Completato. Report disponibili in: ${RESULTS_DIR}"
