from __future__ import annotations

import pytest

from model_inference.dataset import (
    MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET,
    MULTI_PIXMO_CAP_DATASET,
    prepare_inference_row,
)


def test_prepare_maia_row_uses_video_and_question() -> None:
    row = {
        "id": 3,
        "video_id": "Video3",
        "question": "What is happening?",
        "answer1": "A person is cooking.",
        "answer2": "Someone cooks.",
    }

    prepared = prepare_inference_row(
        row=row,
        dataset_name="caput/MAIA_eng",
        media_mode="auto",
        row_index=1,
    )

    assert prepared.media_type == "video"
    assert prepared.media_cache_key == "Video3"
    assert prepared.prompt_variables == {"question": "What is happening?"}
    assert prepared.reference_answers == ["A person is cooking.", "Someone cooks."]


def test_prepare_villanova_row_uses_first_transcript_and_caption() -> None:
    row = {
        "image_url": "https://example.com/sample.png",
        "transcripts": ["A dog jumping over a log.", "Unused second transcript."],
        "caption": "A dog jumps over a fallen tree.",
    }

    prepared = prepare_inference_row(
        row=row,
        dataset_name=MULTI_PIXMO_CAP_DATASET,
        media_mode="auto",
        row_index=4,
    )

    assert prepared.media_type == "image"
    assert prepared.media_cache_key == "https://example.com/sample.png"
    assert prepared.prompt_variables == {"transcript": "A dog jumping over a log."}
    assert prepared.reference_answers == ["A dog jumps over a fallen tree."]
    assert prepared.media_locator == "https://example.com/sample.png"


def test_prepare_villanova_row_requires_non_empty_first_transcript() -> None:
    row = {
        "image_url": "https://example.com/sample.png",
        "transcripts": ["   ", ""],
        "caption": "Caption",
    }

    with pytest.raises(ValueError, match="first transcript"):
        prepare_inference_row(
            row=row,
            dataset_name=MULTI_PIXMO_CAP_DATASET,
            media_mode="auto",
            row_index=1,
        )


def test_prepare_villanova_ask_model_anything_row_uses_question_and_answer() -> None:
    row = {
        "image_url": "https://example.com/question-image.png",
        "question": "What color is the car?",
        "answer": "Red.",
    }

    prepared = prepare_inference_row(
        row=row,
        dataset_name=MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET,
        media_mode="auto",
        row_index=2,
    )

    assert prepared.media_type == "image"
    assert prepared.media_cache_key == "https://example.com/question-image.png"
    assert prepared.prompt_variables == {"question": "What color is the car?"}
    assert prepared.reference_answers == ["Red."]
    assert prepared.media_locator == "https://example.com/question-image.png"


def test_prepare_villanova_ask_model_anything_requires_question() -> None:
    row = {
        "image_url": "https://example.com/question-image.png",
        "question": "   ",
        "answer": "Red.",
    }

    with pytest.raises(ValueError, match="non-empty 'question'"):
        prepare_inference_row(
            row=row,
            dataset_name=MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET,
            media_mode="auto",
            row_index=2,
        )
