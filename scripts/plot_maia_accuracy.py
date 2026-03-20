#!/usr/bin/env python3
"""Plot grouped bar charts for MAIA metrics by language and prompt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "judge_results"
METRICS = [
    ("accuracy", "accuracy", "MAIA Accuracy by Language and Prompt"),
    ("precision_yes", "precision", "MAIA Precision by Language and Prompt"),
    ("recall_yes", "recall", "MAIA Recall by Language and Prompt"),
    ("f1_yes", "f1", "MAIA F1 by Language and Prompt"),
]

LANGUAGES = ["ita", "eng", "spa"]
PROMPTS = ["simple", "wizard", "wizard_few_shots"]
PROMPT_LABELS = {
    "simple": "simple",
    "wizard": "wizard",
    "wizard_few_shots": "wizard_few_shot",
}


def find_result_file(lang: str, prompt: str) -> Path:
    pattern = f"eval_{lang}_{prompt}_*.json"
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Nessun file trovato per pattern: {pattern}")
    return matches[0]


def read_metric(path: Path, metric_key: str) -> float:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["metrics"][metric_key])


def plot_metric(metric_key: str, metric_slug: str, title: str) -> None:
    results: dict[str, list[float]] = {prompt: [] for prompt in PROMPTS}
    for prompt in PROMPTS:
        for lang in LANGUAGES:
            result_file = find_result_file(lang, prompt)
            results[prompt].append(read_metric(result_file, metric_key))

    x = np.arange(len(LANGUAGES))
    bar_w = 0.24
    offsets = [-bar_w, 0.0, bar_w]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx_prompt, prompt in enumerate(PROMPTS):
        values = results[prompt]
        bars = ax.bar(
            x + offsets[idx_prompt],
            values,
            width=bar_w * 0.95,
            color=colors[idx_prompt],
            label=PROMPT_LABELS[prompt],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Language")
    ax.set_ylabel(metric_slug.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Prompt")

    fig.tight_layout()
    OUTPUT_PNG = RESULTS_DIR / f"maia_{metric_slug}_by_language_prompt.png"
    OUTPUT_SVG = RESULTS_DIR / f"maia_{metric_slug}_by_language_prompt.svg"
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200)
    fig.savefig(OUTPUT_SVG)
    plt.close(fig)
    print(f"Plot salvato in: {OUTPUT_PNG}")
    print(f"Plot salvato in: {OUTPUT_SVG}")


def main() -> None:
    for metric_key, metric_slug, title in METRICS:
        plot_metric(metric_key, metric_slug, title)


if __name__ == "__main__":
    main()
