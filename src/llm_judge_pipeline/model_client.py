from __future__ import annotations

import os
from dataclasses import dataclass

from openai import BadRequestError, OpenAI


@dataclass
class JudgeModelClient:
    model_name: str
    client: OpenAI

    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
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

    raise ValueError("Backend non supportato. Usa 'openai' o 'vllm'.")
