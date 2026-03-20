from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class FoilPair:
    wrong_answer1: str
    wrong_answer2: str


def _extract_json_block(text: str) -> dict[str, str] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    return {
        "wrong_answer1": str(payload.get("wrong_answer1", "")).strip(),
        "wrong_answer2": str(payload.get("wrong_answer2", "")).strip(),
    }


def build_foil_prompt(question: str, answer1: str, answer2: str) -> str:
    return (
        "You are creating foil answers for an evaluation dataset.\n"
        "Given one question and two correct answers, produce two minimally wrong answers. Use the language the original answers were written in.\n"
        "Constraints:\n"
        "1) wrong_answer1 must be based on answer1 and contain a subtle factual error.\n"
        "2) wrong_answer2 must be based on answer2 and contain a subtle factual error.\n"
        "3) Keep each wrong answer concise and plausible.\n"
        "4) Do not output the original correct answer verbatim.\n"
        "5) Output ONLY valid JSON with keys: wrong_answer1, wrong_answer2.\n\n"
        f"Question:\n{question}\n\n"
        f"answer1 (correct):\n{answer1}\n\n"
        f"answer2 (correct):\n{answer2}\n"
    )


def generate_foils(
    client: OpenAI,
    model_name: str,
    question: str,
    answer1: str,
    answer2: str,
    max_output_tokens: int = 220,
    max_retries: int = 2,
) -> FoilPair:
    prompt = build_foil_prompt(question=question, answer1=answer1, answer2=answer2)

    last_text = ""
    for _ in range(max_retries + 1):
        response = client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        text = (response.output_text or "").strip()
        last_text = text

        parsed = _extract_json_block(text)
        if parsed and parsed["wrong_answer1"] and parsed["wrong_answer2"]:
            return FoilPair(
                wrong_answer1=parsed["wrong_answer1"],
                wrong_answer2=parsed["wrong_answer2"],
            )

        # Retry with stricter instruction when JSON is invalid.
        prompt = (
            prompt
            + "\nIMPORTANT: previous output was invalid. Return ONLY JSON with "
            + '{"wrong_answer1":"...","wrong_answer2":"..."}'
        )

    raise RuntimeError(f"Unable to parse foil JSON from model output: {last_text}")
