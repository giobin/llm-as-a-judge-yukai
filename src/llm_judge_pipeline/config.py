from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    backend: str
    model_name: str
    dataset_name: str
    dataset_subset: str | None
    split: str
    prompt_file: Path
    num_references: int
    offset_samples: int
    max_samples: int
    random_seed: int
    temperature: float
    max_tokens: int
    verbose: bool
    vllm_base_url: str | None
    vllm_api_key: str | None
    vllm_tensor_parallel_size: int
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: int | None
    vllm_trust_remote_code: bool
    vllm_enforce_eager: bool
    vllm_disable_custom_all_reduce: bool
    input_json: Path | None
    candidate_source: str
    generated_field: str
    generated_sample_mode: str
