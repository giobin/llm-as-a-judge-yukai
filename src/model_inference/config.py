from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    backend: str
    model_name: str
    dataset_name: str
    dataset_subset: str | None
    split: str
    offset_samples: int
    max_samples: int
    media_mode: str
    videos_dir: Path | None
    prompt_file: Path
    num_frames: int
    max_image_dimension: int
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
