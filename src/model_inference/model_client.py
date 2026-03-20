from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, OpenAI


@dataclass
class MultimodalModelClient:
    model_name: str
    client: OpenAI

    def generate(self, content: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self._create_with_temperature_fallback(request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "max_tokens" in message and "max_completion_tokens" in message:
                # Compatibility path for models that only accept max_completion_tokens.
                request_completion_tokens = dict(request_kwargs)
                request_completion_tokens.pop("max_tokens", None)
                request_completion_tokens["max_completion_tokens"] = max_tokens
                response = self._create_with_temperature_fallback(request_completion_tokens)
            else:
                raise

        return _extract_message_text(response)

    def _create_with_temperature_fallback(self, request_kwargs: dict[str, Any]):
        try:
            return self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "temperature" not in message:
                raise

            # Retry path for models that reject temperature=0 or temperature param in general.
            request_no_temp = dict(request_kwargs)
            request_no_temp.pop("temperature", None)
            try:
                return self.client.chat.completions.create(**request_no_temp)
            except BadRequestError:
                request_default_temp = dict(request_kwargs)
                request_default_temp["temperature"] = 1
                return self.client.chat.completions.create(**request_default_temp)


def _extract_message_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        return ""

    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                continue

            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))

    return "\n".join(parts).strip()


def build_model_client(
    backend: str,
    model_name: str,
    vllm_base_url: str | None,
    vllm_api_key: str | None,
) -> MultimodalModelClient:
    backend = backend.lower().strip()

    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY non trovata nell'ambiente.")
        client = OpenAI()
        return MultimodalModelClient(model_name=model_name, client=client)

    if backend == "vllm":
        if not vllm_base_url:
            raise ValueError("Per backend vllm devi passare --vllm-base-url.")
        client = OpenAI(
            base_url=vllm_base_url.rstrip("/") + "/v1",
            api_key=vllm_api_key or "EMPTY",
        )
        return MultimodalModelClient(model_name=model_name, client=client)

    raise ValueError("Backend non supportato. Usa 'openai' o 'vllm'.")
