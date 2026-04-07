from __future__ import annotations

from model_inference.model_client import build_model_client


class _FakeSamplingParams:
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens


class _FakeGeneratedChunk:
    def __init__(self, text: str):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text: str):
        self.outputs = [_FakeGeneratedChunk(text)]


class _FakeLLM:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, sampling_params, use_tqdm, chat_template_content_format):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "chat_template_content_format": chat_template_content_format,
            }
        )
        return [_FakeRequestOutput("generated-answer")]


def test_build_model_client_vllm_in_process_chats_with_openai_content(monkeypatch) -> None:
    fake_llm = _FakeLLM()

    monkeypatch.setattr("model_inference.model_client.build_llm", lambda model_name, cfg: fake_llm)
    monkeypatch.setattr(
        "model_inference.model_client.build_sampling_params",
        lambda temperature, max_tokens: _FakeSamplingParams(temperature=temperature, max_tokens=max_tokens),
    )

    client = build_model_client(
        backend="vllm_in_process",
        model_name="google/gemma-3-27b-it",
        vllm_base_url=None,
        vllm_api_key=None,
        vllm_tensor_parallel_size=2,
        vllm_gpu_memory_utilization=0.85,
        vllm_max_model_len=4096,
        vllm_max_num_seqs=8,
        vllm_trust_remote_code=False,
        vllm_enforce_eager=True,
        vllm_disable_custom_all_reduce=True,
    )

    output = client.generate(
        content=[{"type": "text", "text": "Describe the image."}],
        temperature=0.1,
        max_tokens=48,
    )

    assert output == "generated-answer"
    assert fake_llm.calls == [
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe the image."}]}],
            "sampling_params": fake_llm.calls[0]["sampling_params"],
            "use_tqdm": False,
            "chat_template_content_format": "openai",
        }
    ]
    sampling_params = fake_llm.calls[0]["sampling_params"]
    assert isinstance(sampling_params, _FakeSamplingParams)
    assert sampling_params.temperature == 0.1
    assert sampling_params.max_tokens == 48
