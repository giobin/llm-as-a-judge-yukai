from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MULTI_PIXMO_CAP_DATASET = "VillanovaAI/multi-pixmo-cap"
MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET = "VillanovaAI/multi-pixmo-ask-model-anything"


@dataclass(frozen=True)
class PreparedInferenceRow:
    source_row: dict[str, Any]
    media_type: str
    media_cache_key: str
    prompt_variables: dict[str, str]
    reference_answers: list[str]
    media_locator: str
    media_id: str


def _load_split(dataset_name: str, subset: str | None, split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `datasets` package is required to load Hugging Face datasets. Install project dependencies first."
        ) from exc

    if subset:
        try:
            return load_dataset(dataset_name, subset, split=split)
        except ValueError:
            # Fallback for datasets published without config/subset.
            return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, split=split)


def load_maia_rows(
    dataset_name: str,
    subset: str | None,
    split: str,
    offset_samples: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    if offset_samples < 0:
        raise ValueError("offset_samples must be >= 0.")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0.")

    dataset = _load_split(dataset_name=dataset_name, subset=subset, split=split)
    rows = [dict(dataset[idx]) for idx in range(len(dataset))]

    selected = rows[offset_samples:]
    if max_samples > 0:
        selected = selected[:max_samples]

    return selected


def load_dataset_rows(
    dataset_name: str,
    subset: str | None,
    split: str,
    offset_samples: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    return load_maia_rows(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        offset_samples=offset_samples,
        max_samples=max_samples,
    )


def prepare_inference_row(
    row: dict[str, Any],
    dataset_name: str,
    media_mode: str,
    row_index: int,
) -> PreparedInferenceRow:
    detected_media_type = _detect_media_type(row=row, dataset_name=dataset_name)
    effective_media_type = detected_media_type if media_mode == "auto" else media_mode

    if effective_media_type == "image":
        return _prepare_image_row(
            row=row,
            dataset_name=dataset_name,
            detected_media_type=detected_media_type,
            row_index=row_index,
        )
    if effective_media_type == "video":
        return _prepare_video_row(
            row=row,
            detected_media_type=detected_media_type,
            row_index=row_index,
        )
    raise ValueError(f"Unsupported media mode: {media_mode}")


def _detect_media_type(row: dict[str, Any], dataset_name: str) -> str:
    dataset_name_normalized = dataset_name.strip().lower()
    if dataset_name_normalized in {
        MULTI_PIXMO_CAP_DATASET.lower(),
        MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET.lower(),
    }:
        return "image"
    if row.get("image_url"):
        return "image"
    return "video"


def _prepare_video_row(
    row: dict[str, Any],
    detected_media_type: str,
    row_index: int,
) -> PreparedInferenceRow:
    if detected_media_type != "video":
        raise ValueError("Requested video mode for a row that looks like an image example.")

    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("Row missing non-empty 'question'.")

    media_id = str(row.get("video_id") or row.get("file_name") or row.get("id") or row_index)
    return PreparedInferenceRow(
        source_row=dict(row),
        media_type="video",
        media_cache_key=media_id,
        prompt_variables={"question": question},
        reference_answers=_collect_reference_answers(row),
        media_locator="",
        media_id=media_id,
    )


def _prepare_image_row(
    row: dict[str, Any],
    dataset_name: str,
    detected_media_type: str,
    row_index: int,
) -> PreparedInferenceRow:
    if detected_media_type != "image":
        raise ValueError("Requested image mode for a row that does not expose an image.")

    image_url = str(row.get("image_url", "")).strip()
    if not image_url:
        raise ValueError("Row missing non-empty 'image_url'.")

    dataset_name_normalized = dataset_name.strip().lower()

    if dataset_name_normalized == MULTI_PIXMO_CAP_DATASET.lower():
        transcript = _extract_first_transcript(row.get("transcripts"))
        if not transcript:
            raise ValueError("Villanova row missing a non-empty first transcript in 'transcripts'.")
        prompt_variables = {"transcript": transcript}
    elif dataset_name_normalized == MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET.lower():
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError("Villanova row missing non-empty 'question'.")
        prompt_variables = {"question": question}
    elif str(row.get("question", "")).strip():
        prompt_variables = {"question": str(row.get("question", "")).strip()}
    else:
        prompt_variables = {"caption_hint": str(row.get("caption_hint", "")).strip()}

    media_id = str(row.get("id") or image_url or row_index)
    return PreparedInferenceRow(
        source_row=dict(row),
        media_type="image",
        media_cache_key=image_url,
        prompt_variables=prompt_variables,
        reference_answers=_collect_reference_answers(row),
        media_locator=image_url,
        media_id=media_id,
    )


def _extract_first_transcript(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            transcript = str(item).strip()
            if transcript:
                return transcript
        return ""
    return str(value or "").strip()


def _collect_reference_answers(row: dict[str, Any]) -> list[str]:
    references: list[str] = []

    if isinstance(row.get("answer"), str):
        answer = str(row["answer"]).strip()
        if answer:
            references.append(answer)

    if isinstance(row.get("caption"), str):
        caption = str(row["caption"]).strip()
        if caption:
            references.append(caption)

    for key in ("answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7"):
        value = str(row.get(key, "")).strip()
        if value:
            references.append(value)

    return references
