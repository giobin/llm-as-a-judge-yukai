from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_dataset_builder


def _find_answer_columns(row: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        if key.lower().startswith("answer") and value.strip():
            columns.append(key)
    return sorted(columns)


def _collect_answers(row: dict[str, Any]) -> list[str]:
    return [str(row[k]).strip() for k in _find_answer_columns(row) if str(row[k]).strip()]


def _render_question(question: str, candidate_answer: str, references: list[str]) -> str:
    refs_block = "\n".join(f"- {answer}" for answer in references)
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Candidate Answer]\n"
        f"{candidate_answer}\n\n"
        "[Reference Answers]\n"
        f"{refs_block}"
    )


def build_judge_examples(
    dataset_name: str,
    split: str,
    seed: int,
    dataset_subset: str | None = None,
    max_rows: int = 0,
) -> list[dict[str, str]]:
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=split)
    else:
        try:
            dataset = load_dataset(dataset_name, split=split)
        except ValueError as exc:
            message = str(exc).lower()
            if "config name is missing" not in message:
                raise

            builder = load_dataset_builder(dataset_name)
            config_names = list(builder.builder_configs.keys())
            if not config_names:
                raise

            preferred = "gen" if "gen" in config_names else config_names[0]
            dataset = load_dataset(dataset_name, preferred, split=split)

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if max_rows > 0:
        indices = indices[:max_rows]

    examples: list[dict[str, str]] = []

    for idx in indices:
        row = dataset[idx]
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        all_answers = _collect_answers(row)
        if len(all_answers) < 5:
            continue

        references = all_answers[:4]
        positive_candidates = all_answers[4:]
        if not positive_candidates:
            continue

        positive_answer = rng.choice(positive_candidates)
        examples.append(
            {
                "question": _render_question(question, positive_answer, references),
                "answer": "yes",
                "final_answer": "yes",
            }
        )

        wrong_answer1 = str(row.get("wrong_answer1", "")).strip()
        wrong_answer2 = str(row.get("wrong_answer2", "")).strip()
        negative_candidates = [a for a in [wrong_answer1, wrong_answer2] if a]

        if not negative_candidates:
            continue

        negative_answer = rng.choice(negative_candidates)
        examples.append(
            {
                "question": _render_question(question, negative_answer, references),
                "answer": "no",
                "final_answer": "no",
            }
        )

    rng.shuffle(examples)
    return examples


def split_train_test(examples: list[dict[str, str]], train_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not examples:
        return [], []

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    train_size = int(len(shuffled) * train_ratio)
    train_size = max(1, min(train_size, len(shuffled) - 1))

    return shuffled[:train_size], shuffled[train_size:]


def save_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
