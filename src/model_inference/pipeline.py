from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from .config import InferenceConfig
from .dataset import load_dataset_rows, prepare_inference_row
from .model_client import build_model_client
from .prompting import build_user_content, load_prompt_template, render_prompt
from .schemas import GenerationReport, GenerationSummary
from .video import (
    format_frames_as_data_uris,
    format_image_list_as_data_uris,
    load_image_from_url,
    resolve_video_path,
    sample_video_frames,
)


def _build_public_config(cfg: InferenceConfig) -> dict[str, Any]:
    return {
        "backend": cfg.backend,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "dataset_subset": cfg.dataset_subset,
        "split": cfg.split,
        "offset_samples": cfg.offset_samples,
        "max_samples": cfg.max_samples,
        "media_mode": cfg.media_mode,
        "videos_dir": str(cfg.videos_dir) if cfg.videos_dir else None,
        "prompt_file": str(cfg.prompt_file),
        "num_frames": cfg.num_frames,
        "max_image_dimension": cfg.max_image_dimension,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "verbose": cfg.verbose,
        "vllm_base_url": cfg.vllm_base_url,
        "has_vllm_api_key": bool(cfg.vllm_api_key),
    }


def run_generation(cfg: InferenceConfig) -> dict[str, Any]:
    prompt_template = load_prompt_template(cfg.prompt_file)

    rows = load_dataset_rows(
        dataset_name=cfg.dataset_name,
        subset=cfg.dataset_subset,
        split=cfg.split,
        offset_samples=cfg.offset_samples,
        max_samples=cfg.max_samples,
    )

    if not rows:
        raise RuntimeError("No rows available for generation after applying offset/max filters.")

    model_client = build_model_client(
        backend=cfg.backend,
        model_name=cfg.model_name,
        vllm_base_url=cfg.vllm_base_url,
        vllm_api_key=cfg.vllm_api_key,
    )

    media_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []

    generated_ok = 0
    generated_errors = 0

    iterator = tqdm(rows, desc="Generating answers", unit="row")

    for index, row in enumerate(iterator, start=1):
        result_row = dict(row)
        generation_error: str | None = None
        generated_answer: str | None = None

        cached_meta: dict[str, Any] | None = None
        prepared_row = None

        try:
            prepared_row = prepare_inference_row(
                row=row,
                dataset_name=cfg.dataset_name,
                media_mode=cfg.media_mode,
                row_index=index,
            )

            cache_key = prepared_row.media_cache_key
            if cache_key not in media_cache:
                if prepared_row.media_type == "video":
                    if cfg.videos_dir is None:
                        raise ValueError("Video datasets require --videos-dir.")
                    resolved_path = resolve_video_path(row=row, videos_dir=cfg.videos_dir)
                    frames, frame_indices = sample_video_frames(
                        video_path=resolved_path,
                        num_frames=cfg.num_frames,
                        max_image_dimension=cfg.max_image_dimension,
                    )
                    media_cache[cache_key] = {
                        "media_type": "video",
                        "video_path": str(resolved_path),
                        "image_url": None,
                        "frame_indices": frame_indices,
                        "media_data_uris": format_frames_as_data_uris(frames),
                    }
                else:
                    image = load_image_from_url(
                        image_url=prepared_row.media_locator,
                        max_image_dimension=cfg.max_image_dimension,
                    )
                    media_cache[cache_key] = {
                        "media_type": "image",
                        "video_path": None,
                        "image_url": prepared_row.media_locator,
                        "frame_indices": [0],
                        "media_data_uris": format_image_list_as_data_uris([image]),
                    }

            cached_meta = media_cache[cache_key]
            prompt = render_prompt(prompt_template, prepared_row.prompt_variables)
            user_content = build_user_content(
                prompt=prompt,
                media_data_uris=cached_meta["media_data_uris"],
            )

            if cfg.verbose:
                tqdm.write(f"\n[Row {index}/{len(rows)}] id={row.get('id')}")
                tqdm.write("[PROMPT]")
                tqdm.write(prompt)

            generated_answer = model_client.generate(
                content=user_content,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

            if cfg.verbose:
                tqdm.write("[MODEL ANSWER]")
                tqdm.write(generated_answer)

            generated_ok += 1
        except Exception as exc:
            generation_error = str(exc)
            generated_errors += 1

        result_row["generated_answer1"] = generated_answer
        result_row["generation_error"] = generation_error
        result_row["generation_meta"] = {
            "media_type": cached_meta["media_type"] if cached_meta else None,
            "media_id": prepared_row.media_id if prepared_row else str(row.get("id", "")),
            "video_id": str(row.get("video_id", "")),
            "video_path": cached_meta["video_path"] if cached_meta else None,
            "image_url": cached_meta["image_url"] if cached_meta else None,
            "num_frames_used": len(cached_meta["frame_indices"]) if cached_meta else 0,
            "frame_indices": cached_meta["frame_indices"] if cached_meta else [],
        }
        output_rows.append(result_row)

    report = GenerationReport(
        schema_version="model_inference/v1",
        config=_build_public_config(cfg),
        summary=GenerationSummary(
            total_rows=len(output_rows),
            generated_ok=generated_ok,
            generated_errors=generated_errors,
        ),
        rows=output_rows,
    )

    return report.to_dict()
