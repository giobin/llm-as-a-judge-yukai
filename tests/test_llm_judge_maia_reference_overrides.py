from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge_pipeline import dataset as dataset_module


def test_load_maia_gen_samples_uses_reference_override_as_single_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "id": 1,
            "question": "What is happening?",
            "answer1": "A person is cooking.",
            "answer2": "Someone cooks a meal.",
            "answer3": "A person prepares food.",
            "answer4": "Someone is in the kitchen cooking.",
            "answer5": "A cook is making food.",
            "answer6": "Food is being prepared.",
            "answer7": "The person is cooking dinner.",
            "wrong_answer1": "A person is sleeping.",
            "wrong_answer2": "Someone is driving a car.",
        },
        {
            "id": 2,
            "question": "What animal is visible?",
            "answer1": "A dog.",
            "answer2": "There is a dog.",
            "answer3": "A canine is visible.",
            "answer4": "The image shows a dog.",
            "answer5": "A dog is present.",
            "answer6": "You can see a dog.",
            "answer7": "The visible animal is a dog.",
            "wrong_answer1": "A cat.",
            "wrong_answer2": "A horse.",
        },
    ]
    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    override_path = tmp_path / "references_merged.json"
    override_path.write_text(
        json.dumps(
            [
                {"ID": 1, "merged_reference": "A person is cooking food in a kitchen."},
                {"ID": 2, "merged_reference": "A dog is visible in the scene."},
            ]
        ),
        encoding="utf-8",
    )

    samples = dataset_module.load_maia_gen_samples(
        dataset_name="caput/MAIA_ita",
        subset="gen",
        split="test",
        num_references=7,
        offset_samples=0,
        max_samples=2,
        seed=3,
        reference_overrides_json=override_path,
        reference_overrides_id_field="ID",
        reference_overrides_value_field="merged_reference",
    )

    assert len(samples) == 4
    sample_groups = {}
    for sample in samples:
        sample_groups.setdefault(sample.question_id, []).append(sample)

    assert set(sample_groups) == {"1", "2"}
    assert all(len(group) == 2 for group in sample_groups.values())

    positive_candidates_by_id = {str(row["id"]): {row[f"answer{i}"] for i in range(1, 8)} for row in rows}
    negative_candidates_by_id = {
        str(row["id"]): {row["wrong_answer1"], row["wrong_answer2"]}
        for row in rows
    }
    merged_reference_by_id = {
        "1": "A person is cooking food in a kitchen.",
        "2": "A dog is visible in the scene.",
    }

    for question_id, grouped_samples in sample_groups.items():
        positives = [sample for sample in grouped_samples if sample.expected_label == "yes"]
        negatives = [sample for sample in grouped_samples if sample.expected_label == "no"]

        assert len(positives) == 1
        assert len(negatives) == 1

        positive = positives[0]
        negative = negatives[0]

        assert positive.ground_truth_answers == [merged_reference_by_id[question_id]]
        assert negative.ground_truth_answers == [merged_reference_by_id[question_id]]
        assert positive.candidate_answer in positive_candidates_by_id[question_id]
        assert negative.candidate_answer in negative_candidates_by_id[question_id]


def test_load_maia_gen_samples_requires_override_for_every_selected_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "id": 1,
            "question": "What is happening?",
            "answer1": "A person is cooking.",
            "answer2": "Someone cooks a meal.",
        },
        {
            "id": 2,
            "question": "What animal is visible?",
            "answer1": "A dog.",
            "answer2": "There is a dog.",
        }
    ]
    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    override_path = tmp_path / "references_merged.json"
    override_path.write_text(
        json.dumps([{"ID": 999, "merged_reference": "Unrelated reference."}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing reference override"):
        dataset_module.load_maia_gen_samples(
            dataset_name="caput/MAIA_ita",
            subset="gen",
            split="test",
            num_references=1,
            offset_samples=0,
            max_samples=1,
            seed=1,
            reference_overrides_json=override_path,
            reference_overrides_id_field="ID",
            reference_overrides_value_field="merged_reference",
        )
