#!/usr/bin/env python3
"""Render evaluation metrics as dependency-free SVG grouped bar charts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

METRICS = ("accuracy", "precision_yes", "recall_yes", "f1_yes")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_yes": "Precision",
    "recall_yes": "Recall",
    "f1_yes": "F1",
}
METRIC_COLORS = {
    "accuracy": "#4C78A8",
    "precision_yes": "#F58518",
    "recall_yes": "#54A24B",
    "f1_yes": "#E45756",
}
MODEL_LABELS = {
    "google/gemma-3-27b-it": "Gemma 3 27B IT",
    "gpt-5.2": "GPT-5.2",
}
LANGUAGE_LABELS = {
    "it": "IT",
    "ita": "IT",
    "en": "EN",
    "eng": "EN",
    "es": "ES",
    "spa": "ES",
}
LANGUAGE_ORDER = {"it": 0, "ita": 0, "en": 1, "eng": 1, "es": 2, "spa": 2}


@dataclass(frozen=True)
class Entry:
    label_lines: tuple[str, ...]
    metrics: dict[str, float]
    model: str
    language: str | None
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG grouped bar charts for evaluation metrics in JSON result directories."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing evaluation JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to each input directory.",
    )
    return parser.parse_args()


def prettify_model_name(model_name: str) -> str:
    known_label = MODEL_LABELS.get(model_name)
    if known_label:
        return known_label

    hf_cache_match = re.search(r"models--([^/]+)--([^/]+)/snapshots/", model_name)
    if hf_cache_match:
        org, model = hf_cache_match.groups()
        return f"{org}/{model}".replace("_", "-")

    return model_name.split("/")[-1].replace("_", "-")


def detect_language(config: dict[str, object], file_name: str) -> str | None:
    subset = config.get("dataset_subset")
    if isinstance(subset, str) and subset.strip():
        return subset.strip().lower()

    dataset_name = str(config.get("dataset_name", "")).lower()
    for token in ("ita", "eng", "spa", "it", "en", "es"):
        if re.search(rf"(^|[^a-z0-9]){token}([^a-z0-9]|$)", dataset_name):
            return token

    file_name_lower = file_name.lower()
    for token in ("_ita_", "_eng_", "_spa_", "_it_", "_en_", "_es_"):
        if token in file_name_lower:
            return token.strip("_")

    return None


def build_label_lines(model_name: str, language: str | None) -> tuple[str, ...]:
    model_label = prettify_model_name(model_name)
    if language:
        return (LANGUAGE_LABELS.get(language, language.upper()), model_label)
    return (model_label,)


def load_entries(input_dir: Path) -> list[Entry]:
    entries: list[Entry] = []

    for path in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue

        config = payload.get("config")
        metrics = payload.get("metrics")
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            continue

        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            continue

        metric_values: dict[str, float] = {}
        missing_metric = False
        for metric in METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)):
                missing_metric = True
                break
            metric_values[metric] = float(value)
        if missing_metric:
            continue

        language = detect_language(config, path.name)
        entries.append(
            Entry(
                label_lines=build_label_lines(model_name, language),
                metrics=metric_values,
                model=model_name,
                language=language,
                source_path=path,
            )
        )

    entries.sort(
        key=lambda item: (
            LANGUAGE_ORDER.get(item.language or "", 99),
            prettify_model_name(item.model).lower(),
            item.source_path.name.lower(),
        )
    )
    return entries


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_str = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{escape(text)}</text>'


def svg_multiline_text(x: float, y: float, lines: Iterable[str], **attrs: object) -> str:
    attr_str = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    tspans = []
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else "14"
        tspans.append(f'<tspan x="{x:.2f}" dy="{dy}">{escape(line)}</tspan>')
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{"".join(tspans)}</text>'


def render_svg(entries: list[Entry], title: str) -> str:
    if not entries:
        raise ValueError("No valid evaluation JSON files found.")

    width = max(900, 180 + len(entries) * 150)
    height = 620
    margin_left = 80
    margin_right = 30
    margin_top = 70
    margin_bottom = 160
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    chart_bottom = margin_top + chart_height
    group_width = chart_width / len(entries)
    bar_width = min(22.0, max(12.0, group_width / (len(METRICS) + 1.8)))
    group_inner_width = bar_width * len(METRICS)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, title, fill="#111111", font_size="24", font_weight="700", text_anchor="middle"),
    ]

    for tick in range(0, 11):
        value = tick / 10
        y = chart_bottom - value * chart_height
        stroke = "#c9d1d9" if tick in (0, 10) else "#e5e7eb"
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                margin_left - 10,
                y + 4,
                f"{value:.1f}",
                fill="#4b5563",
                font_size="12",
                text_anchor="end",
                font_family="sans-serif",
            )
        )

    parts.append(
        svg_text(
            24,
            margin_top + chart_height / 2,
            "Score",
            fill="#111111",
            font_size="14",
            font_weight="600",
            text_anchor="middle",
            font_family="sans-serif",
            transform=f"rotate(-90 24 {margin_top + chart_height / 2:.2f})",
        )
    )

    for index, entry in enumerate(entries):
        group_left = margin_left + index * group_width + (group_width - group_inner_width) / 2
        center_x = margin_left + index * group_width + group_width / 2

        for metric_index, metric in enumerate(METRICS):
            value = entry.metrics[metric]
            bar_height = value * chart_height
            x = group_left + metric_index * bar_width
            y = chart_bottom - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 1.5:.2f}" height="{bar_height:.2f}" fill="{METRIC_COLORS[metric]}" rx="3"/>'
            )
            parts.append(
                svg_text(
                    x + (bar_width - 1.5) / 2,
                    max(y - 6, margin_top + 10),
                    f"{value:.3f}",
                    fill="#111111",
                    font_size="11",
                    text_anchor="middle",
                    font_family="sans-serif",
                )
            )

        parts.append(
            svg_multiline_text(
                center_x,
                chart_bottom + 24,
                entry.label_lines,
                fill="#111111",
                font_size="12",
                font_weight="600",
                text_anchor="middle",
                font_family="sans-serif",
            )
        )

    legend_x = margin_left
    legend_y = height - 40
    for metric in METRICS:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{METRIC_COLORS[metric]}" rx="2"/>'
        )
        parts.append(
            svg_text(
                legend_x + 20,
                legend_y + 1,
                METRIC_LABELS[metric],
                fill="#111111",
                font_size="13",
                font_family="sans-serif",
            )
        )
        legend_x += 140

    parts.append("</svg>")
    return "\n".join(parts)


def output_path_for(input_dir: Path, explicit_output_dir: Path | None) -> Path:
    base_dir = explicit_output_dir if explicit_output_dir else input_dir
    return base_dir / f"{input_dir.name}_metrics.svg"


def main() -> None:
    args = parse_args()

    for input_dir in args.input_dirs:
        if not input_dir.exists() or not input_dir.is_dir():
            raise SystemExit(f"Invalid input directory: {input_dir}")

        entries = load_entries(input_dir)
        if not entries:
            raise SystemExit(f"No valid evaluation JSON files found in {input_dir}")

        output_path = output_path_for(input_dir, args.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        title = f"{input_dir.name.replace('_', ' ')} metrics"
        output_path.write_text(render_svg(entries, title), encoding="utf-8")
        print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
