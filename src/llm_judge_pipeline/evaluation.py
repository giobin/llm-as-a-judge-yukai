from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm

from .model_client import JudgeModelClient
from .prompting import parse_judge_output, render_prompt
from .schemas import JudgeSample


@dataclass(frozen=True)
class ExampleResult:
    question_id: str
    expected: str
    predicted: str
    score: float | None


@dataclass(frozen=True)
class EvalReport:
    total: int
    correct: int
    accuracy: float
    precision_yes: float
    recall_yes: float
    f1_yes: float
    results: list[ExampleResult]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_judge(
    model_client: JudgeModelClient,
    samples: list[JudgeSample],
    prompt_template: str,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> EvalReport:
    results: list[ExampleResult] = []

    tp = fp = fn = tn = 0

    iterator = tqdm(samples, desc="Evaluating judge", unit="sample")

    for index, sample in enumerate(iterator, start=1):
        prompt = render_prompt(prompt_template, sample)
        if verbose:
            tqdm.write(f"\n[Sample {index}/{len(samples)}] question_id={sample.question_id}")
            tqdm.write("[PROMPT]")
            tqdm.write(prompt)

        raw_output = model_client.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        pred = parse_judge_output(raw_output)
        if verbose:
            tqdm.write("[RAW ANSWER]")
            tqdm.write(raw_output)
            tqdm.write("[PARSED ANSWER]")
            tqdm.write(f'verdict="{pred.verdict}" score={pred.score}')
            tqdm.write(f'[EXPECTED] "{sample.expected_label}"\n')

        expected = sample.expected_label

        if expected == "yes" and pred.verdict == "yes":
            tp += 1
        elif expected == "no" and pred.verdict == "yes":
            fp += 1
        elif expected == "yes" and pred.verdict == "no":
            fn += 1
        else:
            tn += 1

        results.append(
            ExampleResult(
                question_id=sample.question_id,
                expected=expected,
                predicted=pred.verdict,
                score=pred.score,
            )
        )

    total = len(results)
    correct = tp + tn
    accuracy = _safe_div(correct, total)
    precision_yes = _safe_div(tp, tp + fp)
    recall_yes = _safe_div(tp, tp + fn)
    f1_yes = _safe_div(2 * precision_yes * recall_yes, precision_yes + recall_yes)

    return EvalReport(
        total=total,
        correct=correct,
        accuracy=accuracy,
        precision_yes=precision_yes,
        recall_yes=recall_yes,
        f1_yes=f1_yes,
        results=results,
    )
