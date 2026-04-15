from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge_pipeline.dataset import load_generated_samples_from_json


def test_load_generated_samples_from_json_rows_object(tmp_path: Path) -> None:
    payload = {
        "schema_version": "model_inference/v1",
        "rows": [
            {
                "id": 1,
                "question": "What is shown?",
                "answer1": "A cat",
                "answer2": "A feline",
                "answer3": "Cat",
                "answer4": "It is a cat",
                "generated_answer1": "A cat on the table",
            },
            {
                "id": 2,
                "question": "Missing generation",
                "answer1": "x",
                "answer2": "y",
                "answer3": "z",
                "answer4": "w",
                "generated_answer1": "",
            },
        ],
    }

    path = tmp_path / "generated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=path,
        generated_field="generated_answer1",
        generated_expected_label="yes",
        generated_sample_mode="generated_field",
        num_references=4,
        offset_samples=0,
        max_samples=0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "1"
    assert sample.expected_label == "yes"
    assert sample.candidate_answer == "A cat on the table"
    assert sample.ground_truth_answers == ["A cat", "A feline", "Cat", "It is a cat"]


def test_load_generated_samples_from_json_pixmo_caption_rows(tmp_path: Path) -> None:
    payload = {
        "schema_version": "model_inference/v1",
        "rows": [
            {
                "id": "pixmo-1",
                "image_url": "https://example.com/image.png",
                "transcripts": ["A child in a yellow raincoat jumps in a puddle."],
                "caption": "A child in a yellow raincoat jumps into a puddle.",
                "generated_answer1": "A kid wearing a yellow raincoat splashes in a puddle.",
                "generation_meta": {"media_type": "image"},
            }
        ],
    }

    path = tmp_path / "generated_pixmo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=path,
        generated_field="generated_answer1",
        generated_expected_label="yes",
        generated_sample_mode="generated_field",
        num_references=4,
        offset_samples=0,
        max_samples=0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "pixmo-1"
    assert sample.expected_label == "yes"
    assert sample.candidate_answer == "A kid wearing a yellow raincoat splashes in a puddle."
    assert sample.ground_truth_answers == ["A child in a yellow raincoat jumps into a puddle."]
    assert sample.question == "Write a caption for the image."


def test_load_generated_samples_from_json_pixmo_caption_transcript_pair_mode(tmp_path: Path) -> None:
    payload = {
        "schema_version": "model_inference/v1",
        "rows": [
            {
                "id": "pixmo-1",
                "image_url": "https://example.com/image.png",
                "transcripts": [
                    "Candidate transcript.",
                    "Reference transcript.",
                    "Unused third transcript.",
                ],
                "caption": "This caption is ignored in transcript_pair mode.",
                "generated_answer1": "This generated answer is ignored too.",
                "generation_meta": {"media_type": "image"},
            },
            {
                "id": "pixmo-2",
                "image_url": "https://example.com/image2.png",
                "transcripts": ["Only one transcript."],
                "generation_meta": {"media_type": "image"},
            },
        ],
    }

    path = tmp_path / "generated_pixmo_transcripts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=path,
        generated_field="generated_answer1",
        generated_expected_label="yes",
        generated_sample_mode="transcript_pair",
        num_references=4,
        offset_samples=0,
        max_samples=0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "pixmo-1"
    assert sample.expected_label == "yes"
    assert sample.question == "Write a caption for the image."
    assert sample.candidate_answer == "Candidate transcript."
    assert sample.ground_truth_answers == ["Reference transcript."]


def test_load_generated_samples_from_json_pixmo_ask_model_anything_rows(tmp_path: Path) -> None:
    payload = {
        "schema_version": "model_inference/v1",
        "rows": [
            {
                "id": "ask-1",
                "image_url": "https://example.com/question-image.png",
                "question": "What is the person holding?",
                "answer": "A red umbrella.",
                "generated_answer1": "The person is holding a red umbrella.",
                "generation_meta": {"media_type": "image"},
            }
        ],
    }

    path = tmp_path / "generated_pixmo_ask.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=path,
        generated_field="generated_answer1",
        generated_expected_label="yes",
        generated_sample_mode="generated_field",
        num_references=4,
        offset_samples=0,
        max_samples=0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "ask-1"
    assert sample.expected_label == "yes"
    assert sample.question == "What is the person holding?"
    assert sample.candidate_answer == "The person is holding a red umbrella."
    assert sample.ground_truth_answers == ["A red umbrella."]


def test_load_generated_samples_from_json_supports_negative_expected_label(tmp_path: Path) -> None:
    payload = {
        "schema_version": "model_inference/v1",
        "rows": [
            {
                "id": 7,
                "question": "What is shown?",
                "answer1": "A cat",
                "answer2": "A feline",
                "wrong_answer1": "A dog",
            }
        ],
    }

    path = tmp_path / "generated_negative.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=path,
        generated_field="wrong_answer1",
        generated_expected_label="no",
        generated_sample_mode="generated_field",
        num_references=4,
        offset_samples=0,
        max_samples=0,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "7"
    assert sample.candidate_answer == "A dog"
    assert sample.expected_label == "no"


def test_load_generated_samples_from_json_rejects_invalid_expected_label(tmp_path: Path) -> None:
    payload = {
        "rows": [
            {
                "id": 1,
                "question": "What is shown?",
                "answer1": "A cat",
                "generated_answer1": "A cat",
            }
        ]
    }

    path = tmp_path / "generated_invalid_label.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generated_expected_label"):
        load_generated_samples_from_json(
            input_json=path,
            generated_field="generated_answer1",
            generated_expected_label="maybe",
            generated_sample_mode="generated_field",
            num_references=4,
            offset_samples=0,
            max_samples=0,
        )


def test_load_generated_samples_from_json_uses_reference_override_as_single_ground_truth(tmp_path: Path) -> None:
    payload = {
        "rows": [
            {
                "id": 1,
                "question": "What is happening?",
                "answer1": "A person is cooking.",
                "answer2": "Someone cooks a meal.",
                "generated_answer1": "A person cooks in a kitchen.",
            }
        ]
    }
    overrides = [{"ID": 1, "merged_reference": "A person is cooking food in a kitchen."}]

    input_path = tmp_path / "generated_with_override.json"
    override_path = tmp_path / "references_merged.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    override_path.write_text(json.dumps(overrides), encoding="utf-8")

    samples = load_generated_samples_from_json(
        input_json=input_path,
        generated_field="generated_answer1",
        generated_expected_label="yes",
        generated_sample_mode="generated_field",
        num_references=4,
        offset_samples=0,
        max_samples=0,
        reference_overrides_json=override_path,
        reference_overrides_id_field="ID",
        reference_overrides_value_field="merged_reference",
    )

    assert len(samples) == 1
    assert samples[0].question_id == "1"
    assert samples[0].ground_truth_answers == ["A person is cooking food in a kitchen."]


def test_load_generated_samples_from_json_requires_override_for_every_selected_sample(tmp_path: Path) -> None:
    payload = {
        "rows": [
            {
                "id": 1,
                "question": "What is happening?",
                "answer1": "A person is cooking.",
                "generated_answer1": "A person cooks in a kitchen.",
            }
        ]
    }
    overrides = [{"ID": 999, "merged_reference": "Unrelated reference."}]

    input_path = tmp_path / "generated_missing_override.json"
    override_path = tmp_path / "references_merged.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    override_path.write_text(json.dumps(overrides), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing reference override"):
        load_generated_samples_from_json(
            input_json=input_path,
            generated_field="generated_answer1",
            generated_expected_label="yes",
            generated_sample_mode="generated_field",
            num_references=4,
            offset_samples=0,
            max_samples=0,
            reference_overrides_json=override_path,
            reference_overrides_id_field="ID",
            reference_overrides_value_field="merged_reference",
        )
