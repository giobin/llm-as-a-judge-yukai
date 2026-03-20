from __future__ import annotations

from pathlib import Path

from llm_judge_pipeline.cli import write_sample_prompt_file
from llm_judge_pipeline.schemas import JudgeSample


def test_write_sample_prompt_file_writes_rendered_first_prompt(tmp_path: Path) -> None:
    output_file = tmp_path / "judge_eval_report.json"
    prompt_template = "Question:\n{question}\n\nCandidate:\n{candidate_answer}\n\nRefs:\n{ground_truth_answers}\n"
    samples = [
        JudgeSample(
            question_id="42",
            question="What is shown?",
            candidate_answer="A cat on a chair.",
            ground_truth_answers=["A cat.", "A feline on a chair."],
            expected_label="yes",
        )
    ]

    path = write_sample_prompt_file(output_file=output_file, prompt_template=prompt_template, samples=samples)

    assert path == tmp_path / "judge_eval_report_sample_prompt.txt"
    assert path.read_text(encoding="utf-8") == (
        "Question:\nWhat is shown?\n\n"
        "Candidate:\nA cat on a chair.\n\n"
        "Refs:\n- A cat.\n- A feline on a chair.\n"
    )
