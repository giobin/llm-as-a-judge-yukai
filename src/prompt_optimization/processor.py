from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing


class MaiaJudgeProcessing(DatasetSpecificProcessing):
    """Dataset adapter for PromptWizard on MAIA LLM-as-a-Judge yes/no task."""

    _POSITIVE_LABELS = {
        "yes",
        "y",
        "correct",
        "correcta",
        "corretta",
        "vero",
        "vera",
        "true",
    }
    _NEGATIVE_LABELS = {
        "no",
        "n",
        "incorrect",
        "scorretta",
        "sbagliata",
        "falso",
        "falsa",
        "false",
    }

    @classmethod
    def _normalize_label(cls, text: str) -> str | None:
        token = (text or "").strip().lower()
        token = re.sub(r"[`\"'\\[\\]\\(\\)\\{\\}:;,.!?]", "", token)
        token = re.sub(r"\\s+", " ", token)
        if token in cls._POSITIVE_LABELS:
            return "yes"
        if token in cls._NEGATIVE_LABELS:
            return "no"
        return None

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        dataset = kwargs["dataset"]
        path = Path(dataset_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for sample in dataset:
                row = {
                    DatasetSpecificProcessing.QUESTION_LITERAL: sample["question"],
                    DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample["answer"],
                    DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: sample["final_answer"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def extract_final_answer(self, answer: str) -> str:
        if not answer:
            return self.INVALID_ANS

        content = answer.strip()

        # Prefer answers inside tags when present. Consider all tags and return first valid label.
        tag_matches = re.findall(
            rf"{re.escape(self.ANSWER_START)}(.*?){re.escape(self.ANSWER_END)}",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for candidate in tag_matches:
            normalized = self._normalize_label(candidate)
            if normalized:
                return normalized

        # Fallback: try JSON-like verdict field.
        verdict_match = re.search(
            r'"verdict"\s*:\s*"([^"]+)"',
            content,
            flags=re.IGNORECASE,
        )
        if verdict_match:
            normalized = self._normalize_label(verdict_match.group(1))
            if normalized:
                return normalized

        lowered = content.lower()
        # Fallback: scan last occurrence of label-like tokens in free text.
        tokens = re.findall(
            r"\b(yes|no|corretta|scorretta|correct|incorrect|vero|falso|true|false)\b",
            lowered,
        )
        for token in reversed(tokens):
            normalized = self._normalize_label(token)
            if normalized:
                return normalized

        return self.INVALID_ANS

    def access_answer(self, llm_output: str, gt_answer: str) -> tuple[bool, str]:
        predicted = self.extract_final_answer(llm_output)
        is_correct = predicted == gt_answer.strip().lower()
        return is_correct, predicted
