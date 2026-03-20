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
