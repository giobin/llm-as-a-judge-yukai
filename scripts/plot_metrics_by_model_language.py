#!/usr/bin/env python3
"""Generate MAIA metric heatmaps from evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LANGUAGES = ("ita", "eng", "spa")
METRICS = ("accuracy", "f1_yes", "precision_yes", "recall_yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Legge file JSON di valutazione e genera heatmap PNG delle metriche "
            "accuracy e f1_yes (assi: x=lingua dataset, y=modello)."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Cartella contenente i file .json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella di output dei PNG (default: input_dir)",
    )
    return parser.parse_args()


def detect_language(dataset_name: str) -> str | None:
    tokens = re.split(r"[^a-z0-9]+", dataset_name.lower())
    for lang in LANGUAGES:
        if lang in tokens:
            return lang
    for lang in LANGUAGES:
        if lang in dataset_name.lower():
            return lang
    return None


def load_entries(input_dir: Path) -> list[dict[str, str | float]]:
    entries: list[dict[str, str | float]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skip {path.name}: {exc}")
            continue

        config = data.get("config", {})
        metrics = data.get("metrics", {})
        model = config.get("model_name")
        dataset_name = config.get("dataset_name")

        if not isinstance(model, str) or not isinstance(dataset_name, str):
            print(f"[WARN] Skip {path.name}: config.model_name/config.dataset_name mancanti")
            continue

        language = detect_language(dataset_name)
        if language is None:
            print(f"[WARN] Skip {path.name}: lingua non trovata in dataset_name={dataset_name!r}")
            continue

        row: dict[str, str | float] = {"model": model, "language": language}
        missing_metric = False
        for metric in METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (float, int)):
                print(f"[WARN] Skip {path.name}: metrica {metric!r} mancante o non numerica")
                missing_metric = True
                break
            row[metric] = float(value)

        if not missing_metric:
            entries.append(row)

    return entries


def plot_metric(
    entries: list[dict[str, str | float]],
    metric: str,
    output_dir: Path,
) -> None:
    models = sorted({str(item["model"]) for item in entries})
    if not models:
        raise ValueError("Nessun modello disponibile dopo il parsing dei JSON.")

    values = {
        (str(item["model"]), str(item["language"])): float(item[metric])
        for item in entries
    }

    matrix = np.full((len(models), len(LANGUAGES)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, lang in enumerate(LANGUAGES):
            if (model, lang) in values:
                matrix[i, j] = values[(model, lang)]

    fig_h = max(4.2, 0.55 * len(models) + 1.8)
    fig_w = 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    colorbar.set_label(metric)

    ax.set_xticks(np.arange(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Lingua dataset")
    ax.set_ylabel("Modello")
    ax.set_title(f"{metric} by dataset language and model")

    for i in range(len(models)):
        for j in range(len(LANGUAGES)):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "-", ha="center", va="center", color="#666666", fontsize=9)
            else:
                text_color = "white" if value > 0.6 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.tight_layout()
    output_path = output_dir / f"maia_{metric}_by_model_language.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Plot salvato: {output_path}")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir if args.output_dir else input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input non valido: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(input_dir)
    if not entries:
        raise SystemExit("Nessun file JSON valido trovato.")

    for metric in METRICS:
        plot_metric(entries, metric, output_dir)


if __name__ == "__main__":
    main()
