from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import JudgePrediction, JudgeSample


DEFAULT_PROMPT_TEMPLATE = """Sei un LLM as a Judge.
Valuta se la candidate answer risponde correttamente alla domanda, confrontandola con le ground-truth answers.

Regole:
1) Rispondi "yes" se la candidate answer e' semanticamente corretta rispetto ad almeno una ground-truth.
2) Rispondi "no" se e' errata, non pertinente, o contraddice le ground-truth.
3) Assegna score tra 0 e 1 (1 = totalmente corretta).

Restituisci SOLO JSON valido nel formato:
{"verdict":"yes|no","score":0.0,"reason":"breve motivazione"}

Question:
{question}

Candidate answer:
{candidate_answer}

Ground truth answers:
{ground_truth_answers}
"""


def load_prompt_template(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return DEFAULT_PROMPT_TEMPLATE
    return prompt_file.read_text(encoding="utf-8")


def render_prompt(template: str, sample: JudgeSample) -> str:
    ground_truth = "\n".join(f"- {ans}" for ans in sample.ground_truth_answers)
    # Use explicit token replacement to avoid clashes with literal JSON braces
    # inside prompt templates (e.g. {"verdict":"yes|no"}).
    rendered = template
    rendered = rendered.replace("{question}", sample.question)
    rendered = rendered.replace("{candidate_answer}", sample.candidate_answer)
    rendered = rendered.replace("{ground_truth_answers}", ground_truth)
    return rendered


def parse_judge_output(raw_text: str) -> JudgePrediction:
    """Parse JSON-like judge output with safe fallbacks."""
    cleaned = raw_text.strip()

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    parsed = None
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = None

    if isinstance(parsed, dict):
        verdict_raw = str(parsed.get("verdict", "")).strip().lower()
        verdict = "yes" if verdict_raw == "yes" else "no"

        score_raw = parsed.get("score")
        score = None
        if isinstance(score_raw, (int, float)):
            score = float(max(0.0, min(1.0, score_raw)))

        return JudgePrediction(verdict=verdict, score=score, raw_response=cleaned)

    # Fallback: naive keyword parse.
    lowered = cleaned.lower()
    verdict = "yes" if re.search(r"\byes\b", lowered) else "no"
    return JudgePrediction(verdict=verdict, score=None, raw_response=cleaned)
