from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class VLLMInProcessConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False


@lru_cache(maxsize=1)
def import_vllm() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "The `vllm` package is required for --backend vllm_in_process. "
            "Install it in the current Python environment."
        ) from exc

    return LLM, SamplingParams


def build_llm(model_name: str, cfg: VLLMInProcessConfig) -> Any:
    LLM, _ = import_vllm()

    init_kwargs: dict[str, Any] = {
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.max_model_len is not None:
        init_kwargs["max_model_len"] = cfg.max_model_len
    if cfg.enforce_eager:
        init_kwargs["enforce_eager"] = True
    if cfg.disable_custom_all_reduce:
        init_kwargs["disable_custom_all_reduce"] = True

    return LLM(model=model_name, **init_kwargs)


def build_sampling_params(temperature: float, max_tokens: int) -> Any:
    _, SamplingParams = import_vllm()
    return SamplingParams(temperature=temperature, max_tokens=max_tokens)


def extract_generated_text(request_output: Any) -> str:
    outputs = getattr(request_output, "outputs", None) or []
    if not outputs:
        return ""

    text = getattr(outputs[0], "text", "")
    return text if isinstance(text, str) else ""
