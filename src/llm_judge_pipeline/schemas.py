from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JudgeSample:
    """Single evaluation example fed to the judge model."""

    question_id: str
    question: str
    candidate_answer: str
    ground_truth_answers: list[str]
    expected_label: str  # "yes" if candidate_answer should be considered correct, else "no"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class JudgePrediction:
    """Parsed judge model prediction."""

    verdict: str  # "yes" or "no"
    score: float | None
    raw_response: str
