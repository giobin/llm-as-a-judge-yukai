from __future__ import annotations

from llm_judge_pipeline.model_client import build_model_client


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

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        return [_FakeRequestOutput("judge-answer")]


def test_build_model_client_vllm_in_process_generates_text(monkeypatch) -> None:
    fake_llm = _FakeLLM()

    monkeypatch.setattr("llm_judge_pipeline.model_client.build_llm", lambda model_name, cfg: fake_llm)
    monkeypatch.setattr(
        "llm_judge_pipeline.model_client.build_sampling_params",
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
        vllm_trust_remote_code=False,
        vllm_enforce_eager=True,
        vllm_disable_custom_all_reduce=True,
    )

    output = client.complete("Prompt text", temperature=0.2, max_tokens=64)

    assert output == "judge-answer"
    assert fake_llm.calls == [
        {
            "prompts": ["Prompt text"],
            "sampling_params": fake_llm.calls[0]["sampling_params"],
            "use_tqdm": False,
        }
    ]
    sampling_params = fake_llm.calls[0]["sampling_params"]
    assert isinstance(sampling_params, _FakeSamplingParams)
    assert sampling_params.temperature == 0.2
    assert sampling_params.max_tokens == 64
