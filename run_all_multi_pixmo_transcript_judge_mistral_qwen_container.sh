#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONTAINER_ENTRY="${ROOT_DIR}/scripts/container_python_entry.py"
RESULTS_BASE_DIR="${ROOT_DIR}/judge_results/multi_pixmo_transcript_judge_mistral_qwen_container"
LOGS_DIR="${ROOT_DIR}/logs/multi_pixmo_transcript_judge_mistral_qwen_container"
SUMMARY_FILE="${LOGS_DIR}/summary.tsv"
VENV_MARKER="${ROOT_DIR}/.venv/.installed"

DATASET_NAME="${DATASET_NAME:-VillanovaAI/multi-pixmo-cap}"
SPLIT="${SPLIT:-train}"
OFFSET_SAMPLES="${OFFSET_SAMPLES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-240}"
PIXMO_TRANSCRIPT_POSITIVE_RATIO="${PIXMO_TRANSCRIPT_POSITIVE_RATIO:-0.5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SUBSETS_CSV="${SUBSETS:-it,en,es}"

MISTRAL_JUDGE_MODEL="${MISTRAL_JUDGE_MODEL:-mistralai/Mistral-Large-Instruct-2407}"
QWEN_JUDGE_MODEL="${QWEN_JUDGE_MODEL:-Qwen/Qwen3.5-27B}"
RUN_MISTRAL="${RUN_MISTRAL:-1}"
RUN_QWEN="${RUN_QWEN:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
PARTITION="${PARTITION:-boost_usr_prod}"
WALLTIME="${WALLTIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-FBKLM_prj1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
MISTRAL_VLLM_GPU_MEMORY_UTILIZATION="${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
MISTRAL_VLLM_MAX_MODEL_LEN="${MISTRAL_VLLM_MAX_MODEL_LEN:-8192}"
QWEN_VLLM_GPU_MEMORY_UTILIZATION="${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
QWEN_VLLM_MAX_MODEL_LEN="${QWEN_VLLM_MAX_MODEL_LEN:-8192}"
QWEN_VLLM_ENFORCE_EAGER="${QWEN_VLLM_ENFORCE_EAGER:-1}"

mkdir -p "${RESULTS_BASE_DIR}" "${LOGS_DIR}"

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

gpu_memory_utilization_for_judge() {
  local judge_key="$1"
  case "${judge_key}" in
    mistral_large_2407) printf '%s\n' "${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION}" ;;
    qwen3_5_27b) printf '%s\n' "${QWEN_VLLM_GPU_MEMORY_UTILIZATION}" ;;
    *) printf '%s\n' "${VLLM_GPU_MEMORY_UTILIZATION}" ;;
  esac
}

max_model_len_for_judge() {
  local judge_key="$1"
  case "${judge_key}" in
    mistral_large_2407) printf '%s\n' "${MISTRAL_VLLM_MAX_MODEL_LEN}" ;;
    qwen3_5_27b) printf '%s\n' "${QWEN_VLLM_MAX_MODEL_LEN}" ;;
    *) printf '%s\n' "${VLLM_MAX_MODEL_LEN}" ;;
  esac
}

enforce_eager_for_judge() {
  local judge_key="$1"
  case "${judge_key}" in
    qwen3_5_27b) printf '%s\n' "${QWEN_VLLM_ENFORCE_EAGER}" ;;
    *) printf '%s\n' "${VLLM_ENFORCE_EAGER}" ;;
  esac
}

prompt_file_for_subset() {
  local subset="$1"
  case "${subset}" in
    it) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/judge_prompt_ita.txt" ;;
    en) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/judge_prompt_en.txt" ;;
    es) printf '%s\n' "${ROOT_DIR}/prompts/image_prompts/judge_prompt_es.txt" ;;
    *)
      echo "ERROR: subset non supportato: ${subset}" >&2
      return 1
      ;;
  esac
}

validate_positive_ratio() {
  python3 - "${PIXMO_TRANSCRIPT_POSITIVE_RATIO}" <<'PY'
import sys
ratio = float(sys.argv[1])
if not 0.0 <= ratio <= 1.0:
    raise SystemExit("ERROR: PIXMO_TRANSCRIPT_POSITIVE_RATIO deve essere compreso tra 0.0 e 1.0.")
PY
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
  if [[ "${RUN_MISTRAL}" != "1" ]] && [[ "${RUN_QWEN}" != "1" ]]; then
    echo "ERROR: almeno uno tra RUN_MISTRAL e RUN_QWEN deve essere impostato a 1." >&2
    exit 1
  fi
}

submit_to_gpu_node() {
  local -a srun_cmd

  echo "==> Launching multi-pixmo transcript judge on GPU node"
  echo "==> ROOT_DIR: ${ROOT_DIR}"
  echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"
  echo "==> HF_HOME: ${HF_HOME}"
  echo "==> NUM_GPUS: ${NUM_GPUS}"
  echo "==> PARTITION: ${PARTITION}"
  echo "==> WALLTIME: ${WALLTIME}"
  echo "==> ACCOUNT: ${ACCOUNT}"
  echo "==> Judge models: ${MISTRAL_JUDGE_MODEL} | ${QWEN_JUDGE_MODEL}"
  echo "==> RUN_MISTRAL: ${RUN_MISTRAL}"
  echo "==> RUN_QWEN: ${RUN_QWEN}"
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
    bash -lc "cd '${ROOT_DIR}' && RUN_ON_COMPUTE_NODE=1 '${SCRIPT_PATH}'"
  )

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
      bash -lc "cd '${ROOT_DIR}' && RUN_ON_COMPUTE_NODE=1 '${SCRIPT_PATH}'"
    )
  fi

  "${srun_cmd[@]}"
}

run_judge_eval() {
  local container_engine="$1"
  local judge_key="$2"
  local judge_model="$3"
  local subset="$4"
  local prompt_file judge_slug output_dir output_file error_log
  local model_source effective_gpu_memory_utilization effective_max_model_len effective_enforce_eager
  local -a inner_cmd cmd

  prompt_file="$(prompt_file_for_subset "${subset}")"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "ERROR: prompt file non trovato: ${prompt_file}" >&2
    return 1
  fi

  judge_slug="$(slugify "${judge_model}")"
  model_source="$(resolve_cached_model_source "${judge_model}")"
  effective_gpu_memory_utilization="$(gpu_memory_utilization_for_judge "${judge_key}")"
  effective_max_model_len="$(max_model_len_for_judge "${judge_key}")"
  effective_enforce_eager="$(enforce_eager_for_judge "${judge_key}")"
  output_dir="${RESULTS_BASE_DIR}/judge_${judge_key}"
  output_file="${output_dir}/eval_${subset}_pixmo_transcripts_judged_by_${judge_slug}_positive_ratio_${PIXMO_TRANSCRIPT_POSITIVE_RATIO//./_}_offset_${OFFSET_SAMPLES}_max_${MAX_SAMPLES}.json"
  error_log="${LOGS_DIR}/judge_${judge_key}_${subset}_${judge_slug}.log"
  mkdir -p "${output_dir}"

  inner_cmd=(
    python3 "${CONTAINER_ENTRY}"
    llm_judge_pipeline.cli
    --backend vllm_in_process
    --model-name "${model_source}"
    --dataset-name "${DATASET_NAME}"
    --dataset-subset "${subset}"
    --split "${SPLIT}"
    --candidate-source pixmo_transcripts
    --pixmo-transcript-positive-ratio "${PIXMO_TRANSCRIPT_POSITIVE_RATIO}"
    --random-seed "${RANDOM_SEED}"
    --offset-samples "${OFFSET_SAMPLES}"
    --max-samples "${MAX_SAMPLES}"
    --prompt-file "${prompt_file}"
    --vllm-tensor-parallel-size "${NUM_GPUS}"
    --vllm-gpu-memory-utilization "${effective_gpu_memory_utilization}"
    --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --output-file "${output_file}"
  )

  if [[ -n "${effective_max_model_len}" ]]; then
    inner_cmd+=(--vllm-max-model-len "${effective_max_model_len}")
  fi
  if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
    inner_cmd+=(--vllm-trust-remote-code)
  fi
  if [[ "${effective_enforce_eager}" == "1" ]]; then
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
  echo "Judge model: ${judge_model}"
  echo "Resolved model source: ${model_source}"
  echo "Subset: ${subset}"
  echo "Prompt file: ${prompt_file}"
  echo "Positive ratio: ${PIXMO_TRANSCRIPT_POSITIVE_RATIO}"
  echo "Backend: vllm_in_process"
  echo "Tensor parallel size: ${NUM_GPUS}"
  echo "vLLM gpu_memory_utilization: ${effective_gpu_memory_utilization}"
  if [[ -n "${effective_max_model_len}" ]]; then
    echo "vLLM max_model_len: ${effective_max_model_len}"
  fi
  echo "vLLM enforce eager: ${effective_enforce_eager}"
  echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
  echo "Output: ${output_file}"
  echo "Log: ${error_log}"

  if "${cmd[@]}" >"${error_log}" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${judge_key}" "${judge_model}" "${subset}" "ok" "${output_file}" >> "${SUMMARY_FILE}"
    echo "OK: ${judge_model} | ${subset}"
    return 0
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "${judge_key}" "${judge_model}" "${subset}" "error" "${error_log}" >> "${SUMMARY_FILE}"
  echo "ERROR: ${judge_model} | ${subset}"
  tail -n 40 "${error_log}" || true
  return 1
}

run_on_compute_node() {
  local container_engine overall_failures=0 judge_key judge_model subset
  local -a subsets

  validate_common_requirements
  validate_positive_ratio
  container_engine="$(detect_container_engine)"

  IFS=',' read -r -a subsets <<< "${SUBSETS_CSV}"
  : > "${SUMMARY_FILE}"
  printf 'judge_key\tmodel\tsubset\tstatus\tartifact\n' >> "${SUMMARY_FILE}"

  echo "==> Running on compute node"
  echo "==> Container engine: ${container_engine}"
  echo "==> DATASET_NAME: ${DATASET_NAME}"
  echo "==> SUBSETS: ${SUBSETS_CSV}"
  echo "==> HF_HOME: ${HF_HOME}"
  echo "==> LLM_JUDGE_SIF: ${LLM_JUDGE_SIF}"
  echo "==> RUN_MISTRAL: ${RUN_MISTRAL}"
  echo "==> RUN_QWEN: ${RUN_QWEN}"
  echo "==> VLLM_ENFORCE_EAGER: ${VLLM_ENFORCE_EAGER}"
  echo "==> VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS}"

  local -a judge_specs=()
  if [[ "${RUN_MISTRAL}" == "1" ]]; then
    judge_specs+=("mistral_large_2407:${MISTRAL_JUDGE_MODEL}")
  fi
  if [[ "${RUN_QWEN}" == "1" ]]; then
    judge_specs+=("qwen3_5_27b:${QWEN_JUDGE_MODEL}")
  fi

  for judge_spec in "${judge_specs[@]}"; do
    judge_key="${judge_spec%%:*}"
    judge_model="${judge_spec#*:}"
    for subset in "${subsets[@]}"; do
      if ! run_judge_eval "${container_engine}" "${judge_key}" "${judge_model}" "${subset}"; then
        overall_failures=1
      fi
    done
  done

  echo "============================================================"
  echo "Transcript judge completed"
  echo "Summary TSV: ${SUMMARY_FILE}"
  column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
  return "${overall_failures}"
}

main() {
  if [[ "${RUN_ON_COMPUTE_NODE:-0}" == "1" ]] || [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    run_on_compute_node
    return
  fi

  validate_common_requirements
  submit_to_gpu_node
}

main "$@"
