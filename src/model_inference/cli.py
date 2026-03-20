from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import InferenceConfig
from .pipeline import run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal model inference pipeline for video or image datasets")

    parser.add_argument("--backend", choices=["openai", "vllm"], required=True)
    parser.add_argument("--model-name", required=True)

    parser.add_argument("--dataset-name", default="caput/MAIA_eng")
    parser.add_argument("--dataset-subset", default="gen")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--media-mode", choices=["auto", "video", "image"], default="auto")
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, required=True)

    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-image-dimension", type=int, default=900)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-api-key", default=None)

    parser.add_argument("--output-file", type=Path, default=Path("model_inference_report.json"))

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        backend=args.backend,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        split=args.split,
        offset_samples=args.offset_samples,
        max_samples=args.max_samples,
        media_mode=args.media_mode,
        videos_dir=args.videos_dir,
        prompt_file=args.prompt_file,
        num_frames=args.num_frames,
        max_image_dimension=args.max_image_dimension,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    report = run_generation(cfg)

    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]
    print(f"Generation completed on {summary['total_rows']} rows.")
    print(f"Successful rows: {summary['generated_ok']}")
    print(f"Errored rows: {summary['generated_errors']}")
    print(f"Report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
