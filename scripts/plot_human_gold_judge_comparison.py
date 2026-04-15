#!/usr/bin/env python3
"""Compare judge predictions against human gold labels and plot summary metrics."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = ROOT_DIR / "data" / "human_gold_240.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "judge_results" / "maia_it_generated_gemma_cross_judge_merged_reference" / "human_gold_comparison"

METRICS = ("accuracy", "precision", "recall", "f1")
METRIC_KEYS = {
    "accuracy": "accuracy",
    "precision": "precision_yes",
    "recall": "recall_yes",
    "f1": "f1_yes",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
MODEL_LABELS = {
    "google/gemma-3-27b-it": "Gemma 3 27B IT",
    "gpt-5.2": "GPT-5.2",
}
HUMAN_LABEL_MAP = {
    "correct": "yes",
    "wrong": "no",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute judge metrics against human gold labels and render a comparison plot."
    )
    parser.add_argument(
        "eval_files",
        nargs="+",
        type=Path,
        help="Evaluation JSON files containing examples with question_id and predicted.",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Human gold JSONL file. Default: {DEFAULT_GOLD_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for metrics and plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_gold_labels(path: Path) -> dict[int, str]:
    gold: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                item_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid id at {path}:{line_number}") from exc

            raw_label = str(row.get("human_judge", "")).strip().lower()
            mapped = HUMAN_LABEL_MAP.get(raw_label)
            if mapped is None:
                raise ValueError(f"Unsupported human_judge={raw_label!r} at {path}:{line_number}")
            gold[item_id] = mapped
    return gold


def prettify_model_name(model_name: str, file_path: Path) -> str:
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    if model_name.strip():
        return model_name.split("/")[-1].replace("_", "-")
    return file_path.stem


def compute_metrics(eval_path: Path, gold: dict[int, str]) -> dict[str, object]:
    with eval_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Missing or invalid examples in {eval_path}")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing or invalid config in {eval_path}")

    predictions: dict[int, str] = {}
    for idx, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"Invalid example entry #{idx} in {eval_path}")
        try:
            question_id = int(example["question_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid question_id in {eval_path} example #{idx}") from exc

        predicted = str(example.get("predicted", "")).strip().lower()
        if predicted not in {"yes", "no"}:
            raise ValueError(f"Unsupported predicted label {predicted!r} in {eval_path} example #{idx}")
        predictions[question_id] = predicted

    common_ids = sorted(set(predictions) & set(gold))
    missing_in_gold = sorted(set(predictions) - set(gold))
    missing_in_eval = sorted(set(gold) - set(predictions))
    if not common_ids:
        raise ValueError(f"No overlapping ids between {eval_path} and gold labels")

    tp = sum(1 for item_id in common_ids if gold[item_id] == "yes" and predictions[item_id] == "yes")
    tn = sum(1 for item_id in common_ids if gold[item_id] == "no" and predictions[item_id] == "no")
    fp = sum(1 for item_id in common_ids if gold[item_id] == "no" and predictions[item_id] == "yes")
    fn = sum(1 for item_id in common_ids if gold[item_id] == "yes" and predictions[item_id] == "no")

    total = len(common_ids)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    model_name = str(config.get("model_name", "")).strip()
    return {
        "label": prettify_model_name(model_name, eval_path),
        "model_name": model_name,
        "eval_file": str(eval_path),
        "n_common": total,
        "n_predictions": len(predictions),
        "n_gold": len(gold),
        "missing_in_gold": missing_in_gold,
        "missing_in_eval": missing_in_eval,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
    }


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_str = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{escape(text)}</text>'


def render_svg(results: list[dict[str, object]]) -> str:
    width = 920
    height = 560
    margin_left = 86
    margin_right = 24
    margin_top = 74
    margin_bottom = 118
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    chart_bottom = margin_top + chart_height
    group_width = chart_width / len(METRICS)
    bar_width = min(54.0, (group_width * 0.78) / max(len(results), 1))
    colors = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Judge performance vs human gold labels">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            width / 2,
            34,
            "Judge performance vs human gold labels",
            fill="#111111",
            font_size="24",
            font_weight="700",
            text_anchor="middle",
            font_family="sans-serif",
        ),
    ]

    for tick in range(0, 11):
        value = tick / 10
        y = chart_bottom - value * chart_height
        stroke = "#cbd5e1" if tick in (0, 10) else "#e5e7eb"
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                margin_left - 12,
                y + 4,
                f"{value:.1f}",
                fill="#475569",
                font_size="12",
                text_anchor="end",
                font_family="sans-serif",
            )
        )

    parts.append(
        svg_text(
            26,
            margin_top + chart_height / 2,
            "Score",
            fill="#111111",
            font_size="14",
            font_weight="600",
            text_anchor="middle",
            transform=f"rotate(-90 26 {margin_top + chart_height / 2:.2f})",
            font_family="sans-serif",
        )
    )

    for metric_index, metric in enumerate(METRICS):
        group_center = margin_left + group_width * (metric_index + 0.5)
        total_bars_width = len(results) * bar_width
        group_start = group_center - total_bars_width / 2
        for result_index, result in enumerate(results):
            value = float(result[METRIC_KEYS[metric]])
            bar_height = value * chart_height
            x = group_start + result_index * bar_width
            y = chart_bottom - bar_height
            parts.append(
                f'<rect x="{x + 2:.2f}" y="{y:.2f}" width="{bar_width - 4:.2f}" height="{bar_height:.2f}" fill="{colors[result_index % len(colors)]}" rx="4"/>'
            )
            parts.append(
                svg_text(
                    x + bar_width / 2,
                    max(y - 8, margin_top - 6),
                    f"{value:.3f}",
                    fill="#111111",
                    font_size="11",
                    text_anchor="middle",
                    font_family="sans-serif",
                )
            )

        parts.append(
            svg_text(
                group_center,
                chart_bottom + 28,
                METRIC_LABELS[metric],
                fill="#111111",
                font_size="13",
                font_weight="600",
                text_anchor="middle",
                font_family="sans-serif",
            )
        )

    legend_x = width - margin_right - 220
    legend_y = height - 48
    for idx, result in enumerate(results):
        x = legend_x + idx * 110
        parts.append(f'<rect x="{x:.2f}" y="{legend_y:.2f}" width="16" height="16" fill="{colors[idx % len(colors)]}" rx="3"/>')
        parts.append(
            svg_text(
                x + 24,
                legend_y + 13,
                str(result["label"]),
                fill="#111111",
                font_size="12",
                text_anchor="start",
                font_family="sans-serif",
            )
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    gold = load_gold_labels(args.gold_file)

    results = [compute_metrics(path, gold) for path in args.eval_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = args.output_dir / "judge_vs_human_gold_metrics.json"
    metrics_output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    output_svg = args.output_dir / "judge_vs_human_gold_metrics.svg"
    output_svg.write_text(render_svg(results), encoding="utf-8")

    print(f"Saved metrics: {metrics_output}")
    print(f"Saved plot: {output_svg}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
