#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONTAINER_ENTRY="${ROOT_DIR}/scripts/container_python_entry.py"
RESULTS_DIR="${ROOT_DIR}/judge_results/maia_it_merged_reference_mistral"
LOGS_DIR="${ROOT_DIR}/logs/maia_it_merged_reference_mistral"
SUMMARY_FILE="${LOGS_DIR}/summary.tsv"
VENV_MARKER="${ROOT_DIR}/.venv/.installed"

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

MISTRAL_JUDGE_MODEL="${MISTRAL_JUDGE_MODEL:-mistralai/Mistral-Large-Instruct-2407}"

NUM_GPUS="${NUM_GPUS:-4}"
PARTITION="${PARTITION:-boost_usr_prod}"
QOS="${QOS:-}"
WALLTIME="${WALLTIME:-08:00:00}"
ACCOUNT="${ACCOUNT:-FBKLM_prj1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
MISTRAL_VLLM_GPU_MEMORY_UTILIZATION="${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
MISTRAL_VLLM_MAX_MODEL_LEN="${MISTRAL_VLLM_MAX_MODEL_LEN:-8192}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

slugify() {
  printf '%s\n' "$1" | tr '/:' '__' | tr -c '[:alnum:]_.-' '_'
}

repo_cache_key_for_model() {
  local model_name="$1"
  printf 'models--%s\n' "${model_name//\//--}"
}

resolve_cached_model_source() {
  local model_name="$1"
  local repo_key base_dir snapshots_dir latest_snapshot
  local -a candidate_roots snapshot_dirs

  if [[ "${model_name}" == /* ]] && [[ -d "${model_name}" ]]; then
    printf '%s\n' "${model_name}"
    return 0
  fi

  repo_key="$(repo_cache_key_for_model "${model_name}")"
  candidate_roots=(
    "${HF_HOME}/${repo_key}"
    "${HF_HOME}/hub/${repo_key}"
  )

  for base_dir in "${candidate_roots[@]}"; do
    snapshots_dir="${base_dir}/snapshots"
    if [[ ! -d "${snapshots_dir}" ]]; then
      continue
    fi

    snapshot_dirs=("${snapshots_dir}"/*)
    if [[ ${#snapshot_dirs[@]} -eq 0 ]] || [[ ! -d "${snapshot_dirs[0]}" ]]; then
      continue
    fi

    latest_snapshot="$(printf '%s\n' "${snapshot_dirs[@]}" | sort | tail -n 1)"
    if [[ -n "${latest_snapshot}" ]] && [[ -d "${latest_snapshot}" ]]; then
      printf '%s\n' "${latest_snapshot}"
      return 0
    fi
  done

  printf '%s\n' "${model_name}"
}

detect_container_engine() {
  if [[ -n "${CONTAINER_ENGINE:-}" ]]; then
    if command -v "${CONTAINER_ENGINE}" >/dev/null 2>&1; then
      printf '%s\n' "${CONTAINER_ENGINE}"
      return 0
    fi
    echo "ERROR: CONTAINER_ENGINE=${CONTAINER_ENGINE} ma il comando non è disponibile in PATH." >&2
    exit 1
  fi
  if command -v apptainer >/dev/null 2>&1; then
    printf '%s\n' "apptainer"
    return 0
  fi
  if command -v singularity >/dev/null 2>&1; then
    printf '%s\n' "singularity"
    return 0
  fi
  echo "ERROR: né apptainer né singularity sono disponibili in PATH." >&2
  exit 1
}

validate_common_requirements() {
  if [[ -z "${LLM_JUDGE_SIF:-}" ]]; then
    echo "ERROR: LLM_JUDGE_SIF non impostata." >&2
    exit 1
  fi
  if [[ -z "${HF_HOME:-}" ]]; then
    echo "ERROR: HF_HOME non impostata." >&2
    exit 1
  fi
  if [[ ! -f "${LLM_JUDGE_SIF}" ]]; then
    echo "ERROR: container non trovato: ${LLM_JUDGE_SIF}" >&2
    exit 1
  fi
  if [[ ! -f "${CONTAINER_ENTRY}" ]]; then
    echo "ERROR: entry Python non trovata: ${CONTAINER_ENTRY}" >&2
    exit 1
  fi
  if [[ ! -f "${VENV_MARKER}" ]]; then
    cat >&2 <<EOF_INNER
ERROR: .venv non inizializzata per il workflow container.

Esegui prima sul login node:
  ./container/run_in_container.sh --setup
EOF_INNER
    exit 1
  fi
  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "ERROR: prompt file non trovato: ${PROMPT_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${REFERENCE_OVERRIDES_JSON}" ]]; then
    echo "ERROR: file reference overrides non trovato: ${REFERENCE_OVERRIDES_JSON}" >&2
    exit 1
  fi
}

output_file() {
  local judge_model="$1"
  local model_slug dataset_slug subset_slug split_slug
  model_slug="$(slugify "${judge_model}")"
  dataset_slug="$(slugify "${DATASET_NAME}")"
  subset_slug="$(slugify "${DATASET_SUBSET:-default}")"
  split_slug="$(slugify "${SPLIT}")"
  printf '%s\n' "${RESULTS_DIR}/eval_maia_it_merged_reference_mistral_${dataset_slug}_subset_${subset_slug}_split_${split_slug}_${model_slug}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
}

submit_to_gpu_node() {
  local -a srun_cmd

  echo "==> Launching MAIA merged-reference eval on GPU node"
  echo "==> ROOT_DIR: ${ROOT_DIR}"
  echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"
  echo "==> HF_HOME: ${HF_HOME}"
  echo "==> NUM_GPUS: ${NUM_GPUS}"
  echo "==> PARTITION: ${PARTITION}"
  if [[ -n "${QOS}" ]]; then
    echo "==> QOS: ${QOS}"
  fi
  echo "==> WALLTIME: ${WALLTIME}"
  echo "==> ACCOUNT: ${ACCOUNT}"
  echo "==> Judge model: ${MISTRAL_JUDGE_MODEL}"
  echo "==> Backend: vllm_in_process"
  echo "==> VLLM_ENFORCE_EAGER: ${VLLM_ENFORCE_EAGER}"
  echo "==> VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS}"

  srun_cmd=(
    srun
    -N1
    "--gres=gpu:${NUM_GPUS}"
    -p "${PARTITION}"
    -t "${WALLTIME}"
    --account="${ACCOUNT}"
  )

  if [[ -n "${QOS}" ]]; then
    srun_cmd+=(--qos="${QOS}")
  fi

  srun_cmd+=(bash -lc "cd '${ROOT_DIR}' && RUN_ON_COMPUTE_NODE=1 '${SCRIPT_PATH}'")

  if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( ${SRUN_EXTRA_ARGS} )
    srun_cmd=(
      srun
      "${extra_args[@]}"
      -N1
      "--gres=gpu:${NUM_GPUS}"
      -p "${PARTITION}"
      -t "${WALLTIME}"
      --account="${ACCOUNT}"
    )
    if [[ -n "${QOS}" ]]; then
      srun_cmd+=(--qos="${QOS}")
    fi
    srun_cmd+=(bash -lc "cd '${ROOT_DIR}' && RUN_ON_COMPUTE_NODE=1 '${SCRIPT_PATH}'")
  fi

  "${srun_cmd[@]}"
}

run_eval() {
  local container_engine="$1"
  local judge_slug model_source output_path error_log
  local -a inner_cmd cmd

  judge_slug="$(slugify "${MISTRAL_JUDGE_MODEL}")"
  model_source="$(resolve_cached_model_source "${MISTRAL_JUDGE_MODEL}")"
  output_path="$(output_file "${MISTRAL_JUDGE_MODEL}")"
  error_log="${LOGS_DIR}/judge_maia_it_merged_reference_mistral_${judge_slug}.log"

  inner_cmd=(
    python3 "${CONTAINER_ENTRY}"
    llm_judge_pipeline.cli
    --backend vllm_in_process
    --model-name "${model_source}"
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
    --vllm-tensor-parallel-size "${NUM_GPUS}"
    --vllm-gpu-memory-utilization "${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --output-file "${output_path}"
  )

  if [[ -n "${DATASET_SUBSET}" ]]; then
    inner_cmd+=(--dataset-subset "${DATASET_SUBSET}")
  fi
  if [[ -n "${MISTRAL_VLLM_MAX_MODEL_LEN}" ]]; then
    inner_cmd+=(--vllm-max-model-len "${MISTRAL_VLLM_MAX_MODEL_LEN}")
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

  cmd=(
    "${container_engine}" exec
    --nv
    --bind "${ROOT_DIR}:${ROOT_DIR}"
    --bind "${HF_HOME}:${HF_HOME}"
    --pwd "${ROOT_DIR}"
    "${LLM_JUDGE_SIF}"
    env
    HF_HOME="${HF_HOME}"
    TRANSFORMERS_CACHE="${HF_HOME}"
    HF_DATASETS_CACHE="${HF_HOME}/datasets"
    HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}"
    TRANSFORMERS_OFFLINE="${HF_LOCAL_FILES_ONLY}"
    HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    "${inner_cmd[@]}"
  )

  echo "------------------------------------------------------------"
  echo "Esperimento MAIA it con merged references"
  echo "Dataset: ${DATASET_NAME} (subset: ${DATASET_SUBSET}, split: ${SPLIT})"
  echo "Reference overrides: ${REFERENCE_OVERRIDES_JSON}"
  echo "Judge model: ${MISTRAL_JUDGE_MODEL}"
  echo "Resolved model source: ${model_source}"
  echo "Prompt file: ${PROMPT_FILE}"
  echo "Offset samples: ${OFFSET_SAMPLES}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "Temperature: ${TEMPERATURE}"
  echo "Max tokens: ${MAX_TOKENS}"
  echo "vLLM gpu_memory_utilization: ${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION}"
  if [[ -n "${MISTRAL_VLLM_MAX_MODEL_LEN}" ]]; then
    echo "vLLM max_model_len: ${MISTRAL_VLLM_MAX_MODEL_LEN}"
  fi
  echo "vLLM enforce eager: ${VLLM_ENFORCE_EAGER}"
  echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
  echo "Output: ${output_path}"
  echo "Log: ${error_log}"

  : > "${SUMMARY_FILE}"
  printf 'judge_key\tmodel\tstatus\tartifact\n' >> "${SUMMARY_FILE}"

  if "${cmd[@]}" >"${error_log}" 2>&1; then
    printf '%s\t%s\t%s\t%s\n' "mistral_large_2407" "${MISTRAL_JUDGE_MODEL}" "ok" "${output_path}" >> "${SUMMARY_FILE}"
    echo "OK: ${MISTRAL_JUDGE_MODEL}"
    return 0
  fi

  printf '%s\t%s\t%s\t%s\n' "mistral_large_2407" "${MISTRAL_JUDGE_MODEL}" "error" "${error_log}" >> "${SUMMARY_FILE}"
  echo "ERROR: ${MISTRAL_JUDGE_MODEL}"
  tail -n 40 "${error_log}" || true
  return 1
}

run_on_compute_node() {
  local container_engine
  validate_common_requirements
  container_engine="$(detect_container_engine)"

  echo "==> Running on compute node"
  echo "==> Container engine: ${container_engine}"
  echo "==> HF_HOME: ${HF_HOME}"
  echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"

  run_eval "${container_engine}"

  echo "============================================================"
  echo "MAIA merged-reference eval completed"
  echo "Summary TSV: ${SUMMARY_FILE}"
  column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
}

main() {
  if [[ "${RUN_ON_COMPUTE_NODE:-0}" == "1" ]]; then
    run_on_compute_node
    return
  fi

  validate_common_requirements
  submit_to_gpu_node
}

main "$@"
