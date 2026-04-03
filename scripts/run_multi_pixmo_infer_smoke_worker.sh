#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="${ROOT_DIR}/logs/multi_pixmo_smoke"
RESULTS_DIR="${ROOT_DIR}/model_infer_results/smoke"
SUMMARY_FILE="${LOGS_DIR}/summary.tsv"
CONTAINER_ENTRY="${ROOT_DIR}/scripts/container_python_entry.py"
CONTAINER_IMAGE="${LLM_JUDGE_SIF:-/leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif}"

MAX_SAMPLES="${MAX_SAMPLES:-10}"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_IMAGE_DIMENSION="${MAX_IMAGE_DIMENSION:-900}"
NUM_GPUS="${NUM_GPUS:-4}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
REQUIRE_LOCAL_IMAGES="${REQUIRE_LOCAL_IMAGES:-1}"
IMAGE_CACHE_ROOT="${IMAGE_CACHE_ROOT:-}"
SUBSETS_CSV="${SUBSETS:-it}"
DATASET_NAMES_CSV="${DATASET_NAMES:-VillanovaAI/multi-pixmo-cap,VillanovaAI/multi-pixmo-ask-model-anything}"
MODELS_CSV="${MODELS:-models--google--gemma-3-12b-it,models--google--gemma-3-27b-it,models--google--gemma-3-4b-it,models--mistralai--Mistral-Small-3.2-24B-Instruct-2506,models--mistral-community--pixtral-12b,models--m-polignano--ANITA-NEXT-24B-Magistral-2506-VISION-ITA,models--openbmb--MiniCPM-o-2_6,models--OpenGVLab--InternVL2-40B,models--OpenGVLab--InternVL2_5-78B,models--OpenGVLab--InternVL2-Llama3-76B,models--OpenGVLab--InternVL3-14B-Instruct,models--OpenGVLab--InternVL3_5-38B,models--OpenGVLab--InternVL3_5-4B,models--OpenGVLab--InternVL3_5-8B,models--OpenGVLab--InternVL3-78B,models--OpenGVLab--InternVL3-78B-hf,models--OpenGVLab--InternVL3-8B,models--OpenGVLab--InternVL3-8B-hf,models--Qwen--Qwen2.5-VL-32B-Instruct,models--Qwen--Qwen2.5-VL-72B-Instruct,models--Qwen--Qwen2.5-VL-7B-Instruct,models--Qwen--Qwen3.5-0.8B,models--Qwen--Qwen3-VL-30B-A3B-Instruct,models--Qwen--Qwen3-VL-32B-Instruct,models--Qwen--Qwen3-VL-4B-Instruct,models--Qwen--Qwen3-VL-8B-Instruct}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
: > "${SUMMARY_FILE}"
printf 'model\tdataset\tsubset\tstatus\telapsed_seconds\testimated_240_seconds\toutput_file\terror_log\n' >> "${SUMMARY_FILE}"

if [[ -z "${LLM_JUDGE_SIF:-}" ]]; then
  echo "ERROR: LLM_JUDGE_SIF non impostata." >&2
  exit 1
fi
if [[ -z "${HF_HOME:-}" ]]; then
  echo "ERROR: HF_HOME non impostata." >&2
  exit 1
fi
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "ERROR: container non trovato: ${CONTAINER_IMAGE}" >&2
  exit 1
fi
if [[ ! -f "${CONTAINER_ENTRY}" ]]; then
  echo "ERROR: entry Python non trovata: ${CONTAINER_ENTRY}" >&2
  exit 1
fi

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
IFS=',' read -r -a DATASETS <<< "${DATASET_NAMES_CSV}"
IFS=',' read -r -a SUBSETS <<< "${SUBSETS_CSV}"

normalize_model_name() {
  local raw="$1"
  raw="${raw#models--}"
  printf '%s\n' "${raw//--//}"
}

slugify() {
  printf '%s\n' "$1" | tr '/:' '__' | tr -c '[:alnum:]_.-' '_'
}

prompt_file_for_dataset_subset() {
  local dataset_name="$1"
  local subset="$2"
  case "${dataset_name}" in
    VillanovaAI/multi-pixmo-cap)
      case "${subset}" in
        it) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/generation_prompt_ita.txt" ;;
        en) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/generation_prompt_en.txt" ;;
        es) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/generation_prompt_es.txt" ;;
        *) return 1 ;;
      esac
      ;;
    VillanovaAI/multi-pixmo-ask-model-anything)
      case "${subset}" in
        it) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_ita.txt" ;;
        en) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_en.txt" ;;
        es) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/qa_generation_prompt_es.txt" ;;
        *) return 1 ;;
      esac
      ;;
    *) return 1 ;;
  esac
}

run_inference_once() {
  local model_name="$1"
  local dataset_name="$2"
  local subset="$3"
  local model_slug dataset_slug prompt_file output_file error_log start_ts end_ts elapsed estimate240 status

  model_slug="$(slugify "${model_name}")"
  dataset_slug="$(slugify "${dataset_name}")"
  prompt_file="$(prompt_file_for_dataset_subset "${dataset_name}" "${subset}")"
  output_file="${RESULTS_DIR}/infer_${dataset_slug}_${subset}_${model_slug}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
  error_log="${LOGS_DIR}/infer_${dataset_slug}_${subset}_${model_slug}.log"

  local inner_cmd=(
    python3 "${CONTAINER_ENTRY}"
    model_inference.cli
    --backend vllm_in_process
    --model-name "${model_name}"
    --dataset-name "${dataset_name}"
    --dataset-subset "${subset}"
    --split train
    --offset-samples "${OFFSET_SAMPLES}"
    --max-samples "${MAX_SAMPLES}"
    --media-mode image
    --prompt-file "${prompt_file}"
    --max-image-dimension "${MAX_IMAGE_DIMENSION}"
    --vllm-tensor-parallel-size "${NUM_GPUS}"
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --output-file "${output_file}"
  )

  if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    inner_cmd+=(--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi
  if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
    inner_cmd+=(--vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}")
  fi
  if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
    inner_cmd+=(--vllm-trust-remote-code)
  fi
  if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    inner_cmd+=(--vllm-enforce-eager)
  fi
  if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    inner_cmd+=(--vllm-disable-custom-all-reduce)
  fi
  if [[ "${HF_LOCAL_FILES_ONLY}" == "1" ]]; then
    inner_cmd+=(--hf-local-files-only)
  fi
  if [[ "${REQUIRE_LOCAL_IMAGES}" == "1" ]]; then
    inner_cmd+=(--require-local-images)
  fi
  if [[ -n "${IMAGE_CACHE_ROOT}" ]]; then
    inner_cmd+=(--image-cache-root "${IMAGE_CACHE_ROOT}")
  fi

  local cmd=(
    singularity exec
    --nv
    --bind "${ROOT_DIR}:${ROOT_DIR}"
    --bind "${HF_HOME}:${HF_HOME}"
    --pwd "${ROOT_DIR}"
    "${CONTAINER_IMAGE}"
    env
    HF_HOME="${HF_HOME}"
    TRANSFORMERS_CACHE="${HF_HOME}"
    HF_DATASETS_CACHE="${HF_HOME}/datasets"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    "${inner_cmd[@]}"
  )

  start_ts=$(date +%s)

  echo "------------------------------------------------------------"
  echo "Model: ${model_name}"
  echo "Dataset: ${dataset_name}"
  echo "Subset: ${subset}"
  echo "Prompt: ${prompt_file}"
  echo "Output: ${output_file}"
  echo "Log: ${error_log}"
  echo "Backend: vllm_in_process"
  echo "Python env: container python + repo src + .venv site-packages appended"

  if "${cmd[@]}" >"${error_log}" 2>&1; then
    status="ok"
  else
    status="error"
  fi

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  estimate240=$((elapsed * 24))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model_name}" "${dataset_name}" "${subset}" "${status}" "${elapsed}" "${estimate240}" "${output_file}" "${error_log}" \
    >> "${SUMMARY_FILE}"

  if [[ "${status}" == "ok" ]]; then
    echo "OK: ${model_name} | ${dataset_name} | ${subset} | ${elapsed}s | estimate_240=${estimate240}s"
    return 0
  fi

  echo "ERROR: ${model_name} | ${dataset_name} | ${subset} | ${elapsed}s | estimate_240=${estimate240}s"
  tail -n 40 "${error_log}" || true
  return 1
}

print_final_summary() {
  echo "============================================================"
  echo "Smoke benchmark completed"
  echo "Summary TSV: ${SUMMARY_FILE}"
  echo
  column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
  echo
  echo "Models with at least one successful run:"
  awk -F '\t' 'NR>1 && $4=="ok" {print $1}' "${SUMMARY_FILE}" | sort -u | sed 's/^/  - /' || true
  echo
  echo "Models with only failures:"
  awk -F '\t' 'NR>1 {seen[$1]=1; if ($4=="ok") ok[$1]=1} END {for (m in seen) if (!ok[m]) print "  - " m}' "${SUMMARY_FILE}" | sort || true
}

main() {
  local raw_model model_name overall_failures=0

  echo "==> ROOT_DIR: ${ROOT_DIR}"
  echo "==> HF_HOME: ${HF_HOME}"
  echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"
  echo "==> NUM_GPUS: ${NUM_GPUS}"
  echo "==> MAX_SAMPLES: ${MAX_SAMPLES}"
  echo "==> SUBSETS: ${SUBSETS_CSV}"
  echo "==> DATASETS: ${DATASET_NAMES_CSV}"
  echo "==> BACKEND: vllm_in_process"
  echo "==> RESULTS_DIR: ${RESULTS_DIR}"
  echo "==> VLLM_ENFORCE_EAGER: ${VLLM_ENFORCE_EAGER}"
  echo "==> VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS}"
  echo "==> HF_HUB_OFFLINE: 1"
  echo "==> TRANSFORMERS_OFFLINE: 1"
  echo "==> HF_DATASETS_OFFLINE: 1"

  for raw_model in "${MODELS[@]}"; do
    model_name="$(normalize_model_name "${raw_model}")"
    echo "============================================================"
    echo "Running model: ${model_name}"
    for dataset_name in "${DATASETS[@]}"; do
      for subset in "${SUBSETS[@]}"; do
        if ! run_inference_once "${model_name}" "${dataset_name}" "${subset}"; then
          overall_failures=1
        fi
      done
    done
  done

  print_final_summary
  return "${overall_failures}"
}

main "$@"
