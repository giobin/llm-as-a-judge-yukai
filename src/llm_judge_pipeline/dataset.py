from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .schemas import JudgeSample


def _find_answer_columns(row: dict[str, Any]) -> list[str]:
    candidates = []
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        if key.lower().startswith("answer") and value.strip():
            candidates.append(key)
    return sorted(candidates)


def _collect_answers(row: dict[str, Any], max_answers: int = 4) -> list[str]:
    answer_keys = _find_answer_columns(row)
    answers = [row[key].strip() for key in answer_keys if row[key].strip()]
    return answers[:max_answers]


def _collect_all_answers(row: dict[str, Any]) -> list[str]:
    answer_keys = _find_answer_columns(row)
    return [row[key].strip() for key in answer_keys if row[key].strip()]


def _collect_reference_answers(row: dict[str, Any]) -> list[str]:
    answers = _collect_all_answers(row)
    if answers:
        return answers

    caption = str(row.get("caption", "")).strip()
    if caption:
        return [caption]

    return []


def _extract_first_transcript(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _build_task_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    if question:
        return question

    generation_meta = row.get("generation_meta")
    media_type = ""
    if isinstance(generation_meta, dict):
        media_type = str(generation_meta.get("media_type", "")).strip().lower()
    if media_type == "image" or str(row.get("image_url", "")).strip():
        return "Write a caption for the image."

    return ""


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


def load_maia_gen_samples(
    dataset_name: str,
    subset: str | None,
    split: str,
    num_references: int = 4,
    offset_samples: int = 0,
    max_samples: int = 50,
    seed: int = 42,
) -> list[JudgeSample]:
    """Build positive+negative judge samples from MAIA-like rows.

    Positive sample: question + one valid answer not used in references -> expected yes.
    Negative sample: question + random answer from different row -> expected no.
    """
    dataset = _load_split(dataset_name=dataset_name, subset=subset, split=split)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 rows to create negative samples.")
    if not 1 <= num_references <= 7:
        raise ValueError("num_references must be between 1 and 7.")
    if offset_samples < 0:
        raise ValueError("offset_samples must be >= 0.")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))

    selected = indices[offset_samples:]
    if max_samples > 0:
        selected = selected[:max_samples]
    samples: list[JudgeSample] = []

    for idx in selected:
        row = dataset[idx]
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        all_answers = _collect_all_answers(row)
        if len(all_answers) < num_references + 1:
            # Need num_references references + at least 1 different candidate positive answer.
            continue
        gt_answers = all_answers[:num_references]
        candidate_pool = all_answers[num_references:]

        question_id = str(row.get("id", idx))

        # Positive sample: good answer from the pool not used as references.
        pos_answer = rng.choice(candidate_pool)
        samples.append(
            JudgeSample(
                question_id=question_id,
                question=question,
                candidate_answer=pos_answer,
                ground_truth_answers=gt_answers,
                expected_label="yes",
            )
        )

        # Negative sample:
        # 1) prefer explicit foils from augmented datasets
        # 2) fallback to answer from another random row.
        neg_foil_candidates = [
            str(row.get("wrong_answer1", "")).strip(),
            str(row.get("wrong_answer2", "")).strip(),
        ]
        neg_foil_candidates = [candidate for candidate in neg_foil_candidates if candidate]

        neg_answer: str | None = None
        if neg_foil_candidates:
            neg_answer = rng.choice(neg_foil_candidates)
        else:
            neg_idx = idx
            while neg_idx == idx:
                neg_idx = rng.choice(indices)

            neg_row = dataset[neg_idx]
            neg_answers = _collect_all_answers(neg_row)
            if not neg_answers:
                continue
            neg_answer = rng.choice(neg_answers)

        if neg_answer in gt_answers:
            continue

        samples.append(
            JudgeSample(
                question_id=question_id,
                question=question,
                candidate_answer=neg_answer,
                ground_truth_answers=gt_answers,
                expected_label="no",
            )
        )

    return samples


def load_generated_samples_from_json(
    input_json: Path,
    generated_field: str,
    num_references: int,
    offset_samples: int,
    max_samples: int,
) -> list[JudgeSample]:
    if not 1 <= num_references <= 7:
        raise ValueError("num_references must be between 1 and 7.")
    if offset_samples < 0:
        raise ValueError("offset_samples must be >= 0.")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0.")

    payload = json.loads(input_json.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Invalid input JSON: expected a top-level 'rows' list.")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Invalid input JSON: expected object or array.")

    selected_rows = rows[offset_samples:]
    if max_samples > 0:
        selected_rows = selected_rows[:max_samples]

    samples: list[JudgeSample] = []
    for idx, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            continue

        question = _build_task_prompt(row)
        if not question:
            continue

        generated_answer = str(row.get(generated_field, "")).strip()
        if not generated_answer:
            continue

        all_answers = _collect_reference_answers(row)
        if not all_answers:
            continue

        ground_truth_answers = all_answers[: min(num_references, len(all_answers))]
        question_id = str(row.get("id", row.get("question_id", idx)))

        samples.append(
            JudgeSample(
                question_id=question_id,
                question=question,
                candidate_answer=generated_answer,
                ground_truth_answers=ground_truth_answers,
                expected_label="yes",
            )
        )

    return samples
