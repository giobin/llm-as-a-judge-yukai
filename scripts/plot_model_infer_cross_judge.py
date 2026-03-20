#!/usr/bin/env python3
"""Plot yes/no judge frequencies and export samples predicted as no."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = (
    ROOT_DIR / "judge_results" / "multi_pixmo_cross_judge",
    ROOT_DIR / "judge_results" / "multi_pixmo_ask_model_anything_cross_judge",
)

LANGUAGE_ORDER = ("it", "en", "es")
METRICS = ("yes_frequency", "no_frequency")

FILE_PATTERN = re.compile(
    r"^eval_(?P<lang>ita|eng|spa|it|en|es)_generated_from_(?P<source>.+?)_judged_by_(?P<judge>.+?)_num_refs_.*\.json$"
)

KNOWN_SLUG_LABELS = {
    "google_gemma_3_27b_it": "google/gemma-3-27b-it",
    "gpt_5_2": "gpt-5.2",
}

LANGUAGE_NORMALIZATION = {
    "ita": "it",
    "eng": "en",
    "spa": "es",
    "it": "it",
    "en": "en",
    "es": "es",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera plot della frequenza yes/no del judge e salva i sample "
            "con prediction 'no' per ispezione."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        type=Path,
        default=[path for path in DEFAULT_INPUT_DIRS if path.exists()],
        help="Una o piu' cartelle con report JSON del judge.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Cartella output radice. Se omessa, i file vengono scritti "
            "direttamente dentro ciascuna input dir."
        ),
    )
    return parser.parse_args()


def slug_to_label(slug: str) -> str:
    return KNOWN_SLUG_LABELS.get(slug, slug.replace("_", "-"))


def normalize_language(lang: str) -> str:
    normalized = LANGUAGE_NORMALIZATION.get(lang.lower())
    if normalized is None:
        raise ValueError(f"Unsupported language slug: {lang}")
    return normalized


def load_entries_and_no_samples(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    predicted_no_samples: list[dict[str, Any]] = []
    input_json_cache: dict[Path, list[dict[str, Any]]] = {}

    for path in sorted(input_dir.rglob("*.json")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skip {path}: {exc}")
            continue

        examples = payload.get("examples")
        if not isinstance(examples, list):
            print(f"[WARN] Skip {path}: campo examples mancante/non lista")
            continue

        config = payload.get("config")
        if not isinstance(config, dict):
            print(f"[WARN] Skip {path}: config mancante/non dict")
            continue

        language = normalize_language(match.group("lang"))
        input_json_raw = config.get("input_json")
        if not input_json_raw:
            print(f"[WARN] Skip {path}: input_json mancante in config")
            continue

        input_json_path = Path(str(input_json_raw))
        try:
            input_rows = input_json_cache.setdefault(input_json_path, load_input_rows(input_json_path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[WARN] Skip {path}: impossibile caricare input_json {input_json_path}: {exc}")
            continue

        row_lookup = build_row_lookup(input_rows)
        generated_field = str(config.get("generated_field", "generated_answer1")).strip() or "generated_answer1"

        row: dict[str, Any] = {
            "dataset_group": input_dir.name,
            "file": str(path),
            "language": language,
            "source_slug": match.group("source"),
            "source_label": slug_to_label(match.group("source")),
            "judge_slug": match.group("judge"),
            "judge_label": slug_to_label(match.group("judge")),
        }

        yes_count = 0
        no_count = 0
        for item in examples:
            if not isinstance(item, dict):
                continue

            question_id = str(item.get("question_id", ""))
            original_row = row_lookup.get(question_id)
            if not is_successful_generation(original_row, generated_field):
                continue

            predicted = str(item.get("predicted", "")).strip().lower()
            if predicted == "yes":
                yes_count += 1
            elif predicted == "no":
                no_count += 1
                predicted_no_samples.append(
                    build_predicted_no_record(
                        report_path=path,
                        input_json_path=input_json_path,
                        source_slug=match.group("source"),
                        judge_slug=match.group("judge"),
                        language=language,
                        question_id=question_id,
                        item=item,
                        original_row=original_row,
                        dataset_group=input_dir.name,
                    )
                )

        total = yes_count + no_count
        if total == 0:
            print(f"[WARN] Skip {path}: nessuna prediction yes/no valida")
            continue

        row["yes_count"] = float(yes_count)
        row["no_count"] = float(no_count)
        row["total_count"] = float(total)
        row["yes_frequency"] = float(yes_count) / float(total)
        row["no_frequency"] = float(no_count) / float(total)
        entries.append(row)

    return entries, predicted_no_samples


def build_predicted_no_record(
    report_path: Path,
    input_json_path: Path,
    source_slug: str,
    judge_slug: str,
    language: str,
    question_id: str,
    item: dict[str, Any],
    original_row: dict[str, Any] | None,
    dataset_group: str,
) -> dict[str, Any]:
    return {
        "dataset_group": dataset_group,
        "report_file": str(report_path),
        "input_json": str(input_json_path),
        "source_model": slug_to_label(source_slug),
        "judge_model": slug_to_label(judge_slug),
        "language": language,
        "question_id": question_id,
        "expected": item.get("expected"),
        "predicted": item.get("predicted"),
        "score": item.get("score"),
        "original_row": original_row,
    }


def load_input_rows(input_json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid input JSON structure in {input_json_path}")
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Invalid input JSON structure in {input_json_path}")


def build_row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        question_id = str(row.get("id", row.get("question_id", idx)))
        lookup[question_id] = row
    return lookup


def is_successful_generation(row: dict[str, Any] | None, generated_field: str) -> bool:
    if not isinstance(row, dict):
        return False

    generated_value = row.get(generated_field)
    error_value = row.get("generation_error")

    generated_answer = str(generated_value).strip() if generated_value is not None else ""
    generation_error = str(error_value).strip() if error_value is not None else ""
    return bool(generated_answer) and not generation_error


def plot_metric_by_judge(
    entries: list[dict[str, Any]],
    metric: str,
    output_dir: Path,
    dataset_group: str,
) -> list[Path]:
    outputs: list[Path] = []

    judges = sorted({str(item["judge_label"]) for item in entries})
    sources = sorted({str(item["source_label"]) for item in entries})

    values = {
        (str(item["judge_label"]), str(item["source_label"]), str(item["language"])): float(item[metric])
        for item in entries
    }
    yes_counts = {
        (str(item["judge_label"]), str(item["source_label"]), str(item["language"])): float(item["yes_count"])
        for item in entries
    }
    no_counts = {
        (str(item["judge_label"]), str(item["source_label"]), str(item["language"])): float(item["no_count"])
        for item in entries
    }
    totals = {
        (str(item["judge_label"]), str(item["source_label"]), str(item["language"])): float(item["total_count"])
        for item in entries
    }

    for judge in judges:
        matrix = np.full((len(sources), len(LANGUAGE_ORDER)), np.nan, dtype=float)
        for i, source in enumerate(sources):
            for j, lang in enumerate(LANGUAGE_ORDER):
                value = values.get((judge, source, lang))
                if value is not None:
                    matrix[i, j] = value

        fig_h = max(4.0, 0.7 * len(sources) + 1.5)
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        cmap = "YlGnBu" if metric == "yes_frequency" else "YlOrRd"
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label(f"{metric} (0-1)")

        ax.set_xticks(np.arange(len(LANGUAGE_ORDER)))
        ax.set_xticklabels(LANGUAGE_ORDER)
        ax.set_yticks(np.arange(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel("Lingua")
        ax.set_ylabel("Modello che ha generato")
        ax.set_title(f"{dataset_group} | {metric} | Judge: {judge}")

        for i in range(len(sources)):
            for j in range(len(LANGUAGE_ORDER)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "-", ha="center", va="center", color="#666666", fontsize=9)
                else:
                    txt_color = "white" if val > 0.6 else "black"
                    key = (judge, sources[i], LANGUAGE_ORDER[j])
                    count = yes_counts[key] if metric == "yes_frequency" else no_counts[key]
                    total = totals[key]
                    label = f"{val:.1%}\n({int(count)}/{int(total)})"
                    ax.text(j, i, label, ha="center", va="center", color=txt_color, fontsize=8)

        fig.tight_layout()
        judge_slug = judge.replace("/", "_").replace(".", "_").replace("-", "_")
        png_path = output_dir / f"{dataset_group}_{metric}_judge_{judge_slug}.png"
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
        outputs.append(png_path)

    return outputs


def write_summary_files(entries: list[dict[str, Any]], output_dir: Path, dataset_group: str) -> list[Path]:
    outputs: list[Path] = []

    summary_json = output_dir / f"{dataset_group}_yes_no_summary.json"
    summary_csv = output_dir / f"{dataset_group}_yes_no_summary.csv"

    summary_json.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(summary_json)

    fieldnames = [
        "dataset_group",
        "judge_label",
        "source_label",
        "language",
        "yes_count",
        "no_count",
        "total_count",
        "yes_frequency",
        "no_frequency",
        "file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entries:
            writer.writerow({key: row.get(key) for key in fieldnames})
    outputs.append(summary_csv)

    return outputs


def write_no_sample_files(
    predicted_no_samples: list[dict[str, Any]],
    output_dir: Path,
    dataset_group: str,
) -> list[Path]:
    outputs: list[Path] = []
    no_json = output_dir / f"{dataset_group}_predicted_no_samples.json"

    no_json.write_text(json.dumps(predicted_no_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(no_json)

    return outputs


def resolve_output_dir(input_dir: Path, output_root: Path | None, multiple_inputs: bool) -> Path:
    if output_root is None:
        return input_dir
    if multiple_inputs:
        return output_root / input_dir.name
    return output_root


def main() -> None:
    args = parse_args()
    input_dirs = [path for path in args.input_dirs if path.exists() and path.is_dir()]
    if not input_dirs:
        raise SystemExit("Nessuna input dir valida trovata.")

    multiple_inputs = len(input_dirs) > 1

    for input_dir in input_dirs:
        output_dir = resolve_output_dir(input_dir, args.output_dir, multiple_inputs)
        output_dir.mkdir(parents=True, exist_ok=True)

        entries, predicted_no_samples = load_entries_and_no_samples(input_dir)
        if not entries:
            print(f"[WARN] Nessun file JSON valido trovato in {input_dir}")
            continue

        generated_files: list[Path] = []
        generated_files.extend(write_summary_files(entries, output_dir, input_dir.name))
        generated_files.extend(write_no_sample_files(predicted_no_samples, output_dir, input_dir.name))
        for metric in METRICS:
            generated_files.extend(plot_metric_by_judge(entries, metric, output_dir, input_dir.name))

        for out in generated_files:
            print(f"[OK] {out}")


if __name__ == "__main__":
    main()
