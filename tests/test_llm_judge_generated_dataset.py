from __future__ import annotations

import json
from pathlib import Path

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
