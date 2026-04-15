from __future__ import annotations

import json
import math
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


def _normalize_override_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_reference_overrides(
    reference_overrides_json: Path,
    id_field: str,
    value_field: str,
) -> dict[str, str]:
    payload = json.loads(reference_overrides_json.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(
                "Invalid reference overrides JSON: expected a top-level list or an object with a 'rows' list."
            )
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
            "Invalid reference overrides JSON: expected a top-level list or an object with a 'rows' list."
        )

    overrides: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid reference overrides JSON row at index {index}: expected object.")

        override_id = _normalize_override_key(row.get(id_field))
        if not override_id:
            raise ValueError(
                f"Invalid reference overrides JSON row at index {index}: missing non-empty id field {id_field!r}."
            )

        override_value = str(row.get(value_field, "")).strip()
        if not override_value:
            raise ValueError(
                f"Invalid reference overrides JSON row at index {index}: missing non-empty value field {value_field!r}."
            )

        previous = overrides.get(override_id)
        if previous is not None and previous != override_value:
            raise ValueError(f"Duplicate override id with conflicting values: {override_id!r}.")

        overrides[override_id] = override_value

    return overrides


def _extract_transcripts(value: Any) -> list[str]:
    if isinstance(value, list):
        transcripts: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                transcripts.append(text)
        return transcripts

    text = str(value or "").strip()
    return [text] if text else []


def _extract_transcripts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


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


def _half_up_round(value: float) -> int:
    return math.floor(value + 0.5)


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


def load_pixmo_transcript_pair_samples(
    dataset_name: str,
    subset: str | None,
    split: str,
    offset_samples: int = 0,
    max_samples: int = 50,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> list[JudgeSample]:
    if offset_samples < 0:
        raise ValueError("offset_samples must be >= 0.")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0.")
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("positive_ratio must be between 0.0 and 1.0.")

    dataset = _load_split(dataset_name=dataset_name, subset=subset, split=split)

    eligible_rows: list[dict[str, Any]] = []
    transcripts_by_row: list[list[str]] = []
    question_ids: list[str] = []

    for idx in range(len(dataset)):
        row = dict(dataset[idx])
        transcripts = _extract_transcripts(row.get("transcripts"))
        if len(transcripts) < 2:
            continue

        eligible_rows.append(row)
        transcripts_by_row.append(transcripts)
        question_ids.append(str(row.get("id", idx)))

    eligible_rows = eligible_rows[offset_samples:]
    transcripts_by_row = transcripts_by_row[offset_samples:]
    question_ids = question_ids[offset_samples:]
    if max_samples > 0:
        eligible_rows = eligible_rows[:max_samples]
        transcripts_by_row = transcripts_by_row[:max_samples]
        question_ids = question_ids[:max_samples]

    if not eligible_rows:
        return []
    if len(eligible_rows) < 2 and positive_ratio < 1.0:
        raise ValueError("Need at least 2 eligible rows with 2+ transcripts to create negative samples.")

    rng = random.Random(seed)
    positive_count = _half_up_round(len(eligible_rows) * positive_ratio)
    shuffled_positions = list(range(len(eligible_rows)))
    rng.shuffle(shuffled_positions)
    positive_positions = set(shuffled_positions[:positive_count])

    samples: list[JudgeSample] = []
    for position, row in enumerate(eligible_rows):
        transcripts = transcripts_by_row[position]
        reference_answer = transcripts[1]
        question_id = question_ids[position]
        question = _build_task_prompt(row)
        if not question:
            continue

        if position in positive_positions:
            candidate_answer = transcripts[0]
            expected_label = "yes"
            metadata = {
                "experiment": "pixmo_transcript_pairs",
                "candidate_source": "same_sample_first_transcript",
                "source_row_id": question_id,
                "candidate_row_id": question_id,
                "candidate_transcript_index": 0,
                "reference_row_id": question_id,
                "reference_transcript_index": 1,
            }
        else:
            donor_positions = [idx for idx in range(len(eligible_rows)) if idx != position]
            donor_position = rng.choice(donor_positions)
            donor_transcripts = transcripts_by_row[donor_position]
            donor_transcript_index = rng.randrange(len(donor_transcripts))
            candidate_answer = donor_transcripts[donor_transcript_index]
            expected_label = "no"
            metadata = {
                "experiment": "pixmo_transcript_pairs",
                "candidate_source": "other_sample_random_transcript",
                "source_row_id": question_id,
                "candidate_row_id": question_ids[donor_position],
                "candidate_transcript_index": donor_transcript_index,
                "reference_row_id": question_id,
                "reference_transcript_index": 1,
            }

        samples.append(
            JudgeSample(
                question_id=question_id,
                question=question,
                candidate_answer=candidate_answer,
                ground_truth_answers=[reference_answer],
                expected_label=expected_label,
                metadata=metadata,
            )
        )

    return samples


def load_maia_gen_samples(
    dataset_name: str,
    subset: str | None,
    split: str,
    num_references: int = 4,
    offset_samples: int = 0,
    max_samples: int = 50,
    seed: int = 42,
    reference_overrides_json: Path | None = None,
    reference_overrides_id_field: str = "ID",
    reference_overrides_value_field: str = "merged_reference",
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
    reference_overrides = (
        _load_reference_overrides(
            reference_overrides_json=reference_overrides_json,
            id_field=reference_overrides_id_field,
            value_field=reference_overrides_value_field,
        )
        if reference_overrides_json is not None
        else None
    )

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
        question_id = str(row.get("id", idx))

        if reference_overrides is not None:
            merged_reference = reference_overrides.get(question_id)
            if merged_reference is None:
                raise ValueError(
                    f"Missing reference override for selected MAIA sample id={question_id!r} "
                    f"in {reference_overrides_json}."
                )
            if not all_answers:
                continue
            gt_answers = [merged_reference]
            candidate_pool = list(all_answers)
        else:
            if len(all_answers) < num_references + 1:
                # Need num_references references + at least 1 different candidate positive answer.
                continue
            gt_answers = all_answers[:num_references]
            candidate_pool = all_answers[num_references:]

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
    generated_expected_label: str,
    generated_sample_mode: str,
    num_references: int,
    offset_samples: int,
    max_samples: int,
    reference_overrides_json: Path | None = None,
    reference_overrides_id_field: str = "ID",
    reference_overrides_value_field: str = "merged_reference",
) -> list[JudgeSample]:
    if generated_expected_label not in {"yes", "no"}:
        raise ValueError("generated_expected_label must be 'yes' or 'no'.")
    if generated_sample_mode not in {"generated_field", "transcript_pair"}:
        raise ValueError("generated_sample_mode must be 'generated_field' or 'transcript_pair'.")
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

    reference_overrides = (
        _load_reference_overrides(
            reference_overrides_json=reference_overrides_json,
            id_field=reference_overrides_id_field,
            value_field=reference_overrides_value_field,
        )
        if reference_overrides_json is not None
        else None
    )

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

        if generated_sample_mode == "transcript_pair":
            transcripts = _extract_transcripts(row.get("transcripts"))
            if len(transcripts) < 2:
                continue
            candidate_answer = transcripts[0]
            ground_truth_answers = [transcripts[1]]
        else:
            candidate_answer = str(row.get(generated_field, "")).strip()
            if not candidate_answer:
                continue

        question_id = str(row.get("id", row.get("question_id", idx)))

        if generated_sample_mode != "transcript_pair":
            if reference_overrides is not None:
                merged_reference = reference_overrides.get(question_id)
                if merged_reference is None:
                    raise ValueError(
                        f"Missing reference override for selected generated sample id={question_id!r} "
                        f"in {reference_overrides_json}."
                    )
                ground_truth_answers = [merged_reference]
            else:
                all_answers = _collect_reference_answers(row)
                if not all_answers:
                    continue

                ground_truth_answers = all_answers[: min(num_references, len(all_answers))]

        samples.append(
            JudgeSample(
                question_id=question_id,
                question=question,
                candidate_answer=candidate_answer,
                ground_truth_answers=ground_truth_answers,
                expected_label=generated_expected_label,
            )
        )

    return samples
