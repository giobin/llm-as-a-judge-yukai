from __future__ import annotations

import pytest

from llm_judge_pipeline import dataset as dataset_module


def test_load_pixmo_transcript_pair_samples_uses_second_transcript_as_reference_and_seeded_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "row-1",
            "image_url": "https://example.com/1.png",
            "transcripts": ["row1 transcript0", "row1 transcript1", "row1 transcript2"],
        },
        {
            "id": "row-2",
            "image_url": "https://example.com/2.png",
            "transcripts": ["row2 transcript0", "row2 transcript1"],
        },
        {
            "id": "row-3",
            "image_url": "https://example.com/3.png",
            "transcripts": ["row3 transcript0", "row3 transcript1", "row3 transcript2"],
        },
        {
            "id": "row-4",
            "image_url": "https://example.com/4.png",
            "transcripts": ["row4 transcript0", "row4 transcript1"],
        },
        {
            "id": "ignored",
            "image_url": "https://example.com/5.png",
            "transcripts": ["only one transcript"],
        },
    ]

    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    samples = dataset_module.load_pixmo_transcript_pair_samples(
        dataset_name="VillanovaAI/multi-pixmo-cap",
        subset="it",
        split="train",
        offset_samples=0,
        max_samples=0,
        positive_ratio=0.5,
        seed=7,
    )

    repeated_samples = dataset_module.load_pixmo_transcript_pair_samples(
        dataset_name="VillanovaAI/multi-pixmo-cap",
        subset="it",
        split="train",
        offset_samples=0,
        max_samples=0,
        positive_ratio=0.5,
        seed=7,
    )

    assert samples == repeated_samples
    assert len(samples) == 4
    assert [sample.question_id for sample in samples] == ["row-1", "row-2", "row-3", "row-4"]
    assert [sample.expected_label for sample in samples].count("yes") == 2
    assert [sample.expected_label for sample in samples].count("no") == 2

    row_lookup = {row["id"]: row for row in rows}

    for sample in samples:
        source_transcripts = row_lookup[sample.question_id]["transcripts"]
        assert sample.question == "Write a caption for the image."
        assert sample.ground_truth_answers == [source_transcripts[1]]
        assert sample.metadata is not None
        assert sample.metadata["reference_row_id"] == sample.question_id
        assert sample.metadata["reference_transcript_index"] == 1

        if sample.expected_label == "yes":
            assert sample.candidate_answer == source_transcripts[0]
            assert sample.metadata["candidate_source"] == "same_sample_first_transcript"
            assert sample.metadata["candidate_row_id"] == sample.question_id
            assert sample.metadata["candidate_transcript_index"] == 0
        else:
            donor_row_id = sample.metadata["candidate_row_id"]
            assert donor_row_id != sample.question_id
            donor_transcripts = row_lookup[donor_row_id]["transcripts"]
            assert sample.candidate_answer in donor_transcripts
            assert sample.metadata["candidate_source"] == "other_sample_random_transcript"


@pytest.mark.parametrize(
    ("positive_ratio", "expected_yes", "expected_no"),
    [
        (0.0, 0, 3),
        (1.0, 3, 0),
    ],
)
def test_load_pixmo_transcript_pair_samples_supports_ratio_extremes(
    monkeypatch: pytest.MonkeyPatch,
    positive_ratio: float,
    expected_yes: int,
    expected_no: int,
) -> None:
    rows = [
        {"id": "row-1", "image_url": "https://example.com/1.png", "transcripts": ["a0", "a1"]},
        {"id": "row-2", "image_url": "https://example.com/2.png", "transcripts": ["b0", "b1"]},
        {"id": "row-3", "image_url": "https://example.com/3.png", "transcripts": ["c0", "c1"]},
    ]

    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    samples = dataset_module.load_pixmo_transcript_pair_samples(
        dataset_name="VillanovaAI/multi-pixmo-cap",
        subset="it",
        split="train",
        offset_samples=0,
        max_samples=0,
        positive_ratio=positive_ratio,
        seed=5,
    )

    assert [sample.expected_label for sample in samples].count("yes") == expected_yes
    assert [sample.expected_label for sample in samples].count("no") == expected_no


def test_load_pixmo_transcript_pair_samples_rejects_invalid_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: [])

    with pytest.raises(ValueError, match="positive_ratio"):
        dataset_module.load_pixmo_transcript_pair_samples(
            dataset_name="VillanovaAI/multi-pixmo-cap",
            subset="it",
            split="train",
            positive_ratio=1.1,
        )


def test_load_pixmo_transcript_pair_samples_requires_two_rows_for_negative_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "row-1",
            "image_url": "https://example.com/1.png",
            "transcripts": ["row1 transcript0", "row1 transcript1"],
        }
    ]

    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    with pytest.raises(ValueError, match="at least 2 eligible rows"):
        dataset_module.load_pixmo_transcript_pair_samples(
            dataset_name="VillanovaAI/multi-pixmo-cap",
            subset="it",
            split="train",
            positive_ratio=0.0,
            seed=3,
        )


def test_load_pixmo_transcript_pair_samples_applies_offset_and_max_after_eligibility_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"id": "ignored-1", "image_url": "https://example.com/1.png", "transcripts": ["only one"]},
        {"id": "keep-1", "image_url": "https://example.com/2.png", "transcripts": ["a0", "a1"]},
        {"id": "ignored-2", "image_url": "https://example.com/3.png", "transcripts": []},
        {"id": "keep-2", "image_url": "https://example.com/4.png", "transcripts": ["b0", "b1"]},
        {"id": "keep-3", "image_url": "https://example.com/5.png", "transcripts": ["c0", "c1"]},
    ]

    monkeypatch.setattr(dataset_module, "_load_split", lambda dataset_name, subset, split: rows)

    samples = dataset_module.load_pixmo_transcript_pair_samples(
        dataset_name="VillanovaAI/multi-pixmo-cap",
        subset="it",
        split="train",
        offset_samples=1,
        max_samples=1,
        positive_ratio=1.0,
        seed=1,
    )

    assert len(samples) == 1
    assert samples[0].question_id == "keep-2"
