from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .dataset import (
    load_generated_samples_from_json,
    load_maia_gen_samples,
    load_pixmo_transcript_pair_samples,
)
from .evaluation import evaluate_judge
from .model_client import build_model_client
from .prompting import load_prompt_template, render_prompt
from .schemas import JudgeSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation pipeline")

    parser.add_argument("--backend", choices=["openai", "vllm", "vllm_in_process"], required=True)
    parser.add_argument("--model-name", required=True)

    parser.add_argument("--dataset-name", default="caput/MAIA_dev_set_eng")
    parser.add_argument(
        "--dataset-subset",
        default=None,
        help="Dataset config/subset (e.g. gen). Omit for datasets with no config.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--num-references",
        type=int,
        choices=range(1, 8),
        default=4,
        help="Numero di ground-truth references da mostrare al judge (1-7).",
    )
    parser.add_argument("--offset-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/judge_prompt.txt"))

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-api-key", default=None)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-trust-remote-code", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-disable-custom-all-reduce", action="store_true")

    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument(
        "--candidate-source",
        choices=["synthetic", "generated", "pixmo_transcripts"],
        default="synthetic",
        help="Source of candidate answers: synthetic positive/negative samples or generated answers from model-infer JSON.",
    )
    parser.add_argument(
        "--generated-field",
        default="generated_answer1",
        help="Field name to read generated candidate answers from when --candidate-source=generated and --generated-sample-mode=generated_field.",
    )
    parser.add_argument(
        "--generated-sample-mode",
        choices=["generated_field", "transcript_pair"],
        default="generated_field",
        help="How to build judge samples from --input-json when --candidate-source=generated.",
    )
    parser.add_argument(
        "--pixmo-transcript-positive-ratio",
        type=float,
        default=0.5,
        help=(
            "Quota di candidate answers positive per --candidate-source=pixmo_transcripts. "
            "0.0 = tutte negative, 1.0 = tutte positive."
        ),
    )

    parser.add_argument("--output-file", type=Path, default=Path("judge_eval_report.json"))

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        backend=args.backend,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        split=args.split,
        prompt_file=args.prompt_file,
        num_references=args.num_references,
        offset_samples=args.offset_samples,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_trust_remote_code=args.vllm_trust_remote_code,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_disable_custom_all_reduce=args.vllm_disable_custom_all_reduce,
        input_json=args.input_json,
        candidate_source=args.candidate_source,
        generated_field=args.generated_field,
        pixmo_transcript_positive_ratio=args.pixmo_transcript_positive_ratio,
        generated_sample_mode=args.generated_sample_mode,
    )


def write_sample_prompt_file(
    output_file: Path,
    prompt_template: str,
    samples: list[JudgeSample],
) -> Path | None:
    if not samples:
        return None

    prompt_sample = render_prompt(prompt_template, samples[0])
    prompt_sample_path = output_file.with_name(f"{output_file.stem}_sample_prompt.txt")
    prompt_sample_path.write_text(prompt_sample, encoding="utf-8")
    return prompt_sample_path


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    prompt_template = load_prompt_template(cfg.prompt_file)

    if cfg.candidate_source == "generated":
        if cfg.input_json is None:
            raise ValueError("--input-json is required when --candidate-source=generated.")

        samples = load_generated_samples_from_json(
            input_json=cfg.input_json,
            generated_field=cfg.generated_field,
            generated_sample_mode=cfg.generated_sample_mode,
            num_references=cfg.num_references,
            offset_samples=cfg.offset_samples,
            max_samples=cfg.max_samples,
        )
    elif cfg.candidate_source == "pixmo_transcripts":
        samples = load_pixmo_transcript_pair_samples(
            dataset_name=cfg.dataset_name,
            subset=cfg.dataset_subset,
            split=cfg.split,
            offset_samples=cfg.offset_samples,
            max_samples=cfg.max_samples,
            positive_ratio=cfg.pixmo_transcript_positive_ratio,
            seed=cfg.random_seed,
        )
    else:
        samples = load_maia_gen_samples(
            dataset_name=cfg.dataset_name,
            subset=cfg.dataset_subset,
            split=cfg.split,
            num_references=cfg.num_references,
            offset_samples=cfg.offset_samples,
            max_samples=cfg.max_samples,
            seed=cfg.random_seed,
        )

    if not samples:
        raise RuntimeError("Nessun sample valido estratto dal dataset/input JSON.")

    prompt_sample_path = write_sample_prompt_file(
        output_file=args.output_file,
        prompt_template=prompt_template,
        samples=samples,
    )

    model_client = build_model_client(
        backend=cfg.backend,
        model_name=cfg.model_name,
        vllm_base_url=cfg.vllm_base_url,
        vllm_api_key=cfg.vllm_api_key,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_len=cfg.vllm_max_model_len,
        vllm_trust_remote_code=cfg.vllm_trust_remote_code,
        vllm_enforce_eager=cfg.vllm_enforce_eager,
        vllm_disable_custom_all_reduce=cfg.vllm_disable_custom_all_reduce,
    )

    report = evaluate_judge(
        model_client=model_client,
        samples=samples,
        prompt_template=prompt_template,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        verbose=cfg.verbose,
    )

    payload = {
        "config": {
            "backend": cfg.backend,
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "dataset_subset": cfg.dataset_subset,
            "split": cfg.split,
            "num_references": cfg.num_references,
            "effective_num_references": 1 if cfg.candidate_source == "pixmo_transcripts" else cfg.num_references,
            "offset_samples": cfg.offset_samples,
            "max_samples": cfg.max_samples,
            "random_seed": cfg.random_seed,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "verbose": cfg.verbose,
            "vllm_base_url": cfg.vllm_base_url,
            "vllm_tensor_parallel_size": cfg.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
            "vllm_max_model_len": cfg.vllm_max_model_len,
            "vllm_trust_remote_code": cfg.vllm_trust_remote_code,
            "vllm_enforce_eager": cfg.vllm_enforce_eager,
            "vllm_disable_custom_all_reduce": cfg.vllm_disable_custom_all_reduce,
            "candidate_source": cfg.candidate_source,
            "generated_field": cfg.generated_field,
            "pixmo_transcript_positive_ratio": cfg.pixmo_transcript_positive_ratio,
            "generated_sample_mode": cfg.generated_sample_mode,
            "input_json": str(cfg.input_json) if cfg.input_json else None,
            "sample_prompt_file": str(prompt_sample_path) if prompt_sample_path else None,
        },
        "metrics": {
            "total": report.total,
            "correct": report.correct,
            "accuracy": report.accuracy,
            "precision_yes": report.precision_yes,
            "recall_yes": report.recall_yes,
            "f1_yes": report.f1_yes,
        },
        "examples": [
            {
                "question_id": row.question_id,
                "expected": row.expected,
                "predicted": row.predicted,
                "score": row.score,
                "question": sample.question,
                "candidate_answer": sample.candidate_answer,
                "ground_truth_answers": sample.ground_truth_answers,
                "metadata": sample.metadata,
            }
            for row, sample in zip(report.results, samples)
        ],
    }

    args.output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluation completed on {report.total} samples.")
    print(f"Accuracy: {report.accuracy:.4f}")
    print(f"Precision(yes): {report.precision_yes:.4f}")
    print(f"Recall(yes): {report.recall_yes:.4f}")
    print(f"F1(yes): {report.f1_yes:.4f}")
    print(f"Report saved to: {args.output_file}")
    if prompt_sample_path:
        print(f"Sample prompt saved to: {prompt_sample_path}")


if __name__ == "__main__":
    main()
