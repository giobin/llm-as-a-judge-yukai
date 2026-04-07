from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, OpenAI

from common.vllm_in_process import (
    VLLMInProcessConfig,
    build_llm,
    build_sampling_params,
    extract_generated_text,
)


@dataclass
class JudgeModelClient:
    model_name: str
    client: OpenAI | None = None
    llm: Any | None = None

    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if self.llm is not None:
            sampling_params = build_sampling_params(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outputs = self.llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
            if not outputs:
                return ""
            return extract_generated_text(outputs[0])

        if self.client is None:
            raise RuntimeError("JudgeModelClient is not initialized.")

        request_kwargs = {
            "model": self.model_name,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            response = self.client.responses.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "temperature" not in message:
                raise

            # Retry path for models that reject temperature=0 or temperature param in general.
            request_no_temp = dict(request_kwargs)
            request_no_temp.pop("temperature", None)
            try:
                response = self.client.responses.create(**request_no_temp)
            except BadRequestError:
                request_with_default_temp = dict(request_kwargs)
                request_with_default_temp["temperature"] = 1
                response = self.client.responses.create(**request_with_default_temp)

        return response.output_text or ""


def build_model_client(
    backend: str,
    model_name: str,
    vllm_base_url: str | None,
    vllm_api_key: str | None,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int | None,
    vllm_max_num_seqs: int | None,
    vllm_trust_remote_code: bool,
    vllm_enforce_eager: bool,
    vllm_disable_custom_all_reduce: bool,
) -> JudgeModelClient:
    backend = backend.lower().strip()

    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY non trovata nell'ambiente.")
        client = OpenAI()
        return JudgeModelClient(model_name=model_name, client=client)

    if backend == "vllm":
        if not vllm_base_url:
            raise ValueError("Per backend vllm devi passare --vllm-base-url.")
        client = OpenAI(
            base_url=vllm_base_url.rstrip("/") + "/v1",
            api_key=vllm_api_key or "EMPTY",
        )
        return JudgeModelClient(model_name=model_name, client=client)

    if backend == "vllm_in_process":
        llm = build_llm(
            model_name=model_name,
            cfg=VLLMInProcessConfig(
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=vllm_max_model_len,
                max_num_seqs=vllm_max_num_seqs,
                trust_remote_code=vllm_trust_remote_code,
                enforce_eager=vllm_enforce_eager,
                disable_custom_all_reduce=vllm_disable_custom_all_reduce,
            ),
        )
        return JudgeModelClient(model_name=model_name, llm=llm)

    raise ValueError("Backend non supportato. Usa 'openai', 'vllm' o 'vllm_in_process'.")
