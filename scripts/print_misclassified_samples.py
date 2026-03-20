#!/usr/bin/env python3
"""Print misclassified LLM-as-a-judge samples from an evaluation JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stampa i sample con errore (expected != predicted) da un file JSON di "
            "valutazione LLM-as-a-judge."
        )
    )
    parser.add_argument("results_file", type=Path, help="Path del file JSON risultati")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Numero massimo di errori da stampare (0 = tutti)",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Non ricaricare il dataset; stampa solo metadati dell'errore.",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"File non trovato: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON non valido ({path}): {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit("Formato JSON non valido: root deve essere un oggetto.")

    examples = data.get("examples")
    if not isinstance(examples, list):
        raise SystemExit("Formato JSON non valido: campo 'examples' mancante o non lista.")

    return data


def reconstruct_samples(config: dict[str, Any]) -> list[Any]:
    from llm_judge_pipeline.dataset import load_maia_gen_samples

    dataset_name = config.get("dataset_name")
    split = config.get("split")

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("config.dataset_name mancante o non valido")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("config.split mancante o non valido")

    subset = config.get("dataset_subset")
    if subset is not None and not isinstance(subset, str):
        subset = None

    offset_samples = int(config.get("offset_samples", 0))
    max_samples = int(config.get("max_samples", 0))
    random_seed = int(config.get("random_seed", 42))

    return load_maia_gen_samples(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        offset_samples=offset_samples,
        max_samples=max_samples,
        seed=random_seed,
    )


def build_sample_lookup(samples: list[Any]) -> dict[tuple[str, str], list[Any]]:
    lookup: dict[tuple[str, str], list[Any]] = {}
    for sample in samples:
        key = (str(sample.question_id), str(sample.expected_label))
        lookup.setdefault(key, []).append(sample)
    return lookup


def main() -> None:
    args = parse_args()
    data = load_results(args.results_file)

    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}

    examples = data["examples"]
    mismatches: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(examples):
        if not isinstance(row, dict):
            continue
        expected = row.get("expected")
        predicted = row.get("predicted")
        if expected != predicted:
            mismatches.append((idx, row))

    print(f"File: {args.results_file}")
    print(f"Totale esempi: {len(examples)}")
    print(f"Errori (expected != predicted): {len(mismatches)}")

    if not mismatches:
        return

    samples_by_index: list[Any] | None = None
    sample_lookup: dict[tuple[str, str], list[Any]] | None = None
    if not args.no_dataset:
        try:
            samples = reconstruct_samples(config)
            if len(samples) == len(examples):
                samples_by_index = samples
            else:
                print(
                    "[WARN] Numero sample ricostruiti diverso da examples "
                    f"({len(samples)} vs {len(examples)})."
                )
                sample_lookup = build_sample_lookup(samples)
        except Exception as exc:
            print(f"[WARN] Impossibile ricostruire i sample dal dataset: {exc}")
            print("[WARN] Procedo stampando solo i metadati presenti nel JSON.")

    limit = args.limit if args.limit and args.limit > 0 else len(mismatches)

    for out_idx, (index, row) in enumerate(mismatches[:limit], start=1):
        question_id = str(row.get("question_id", "?"))
        expected = str(row.get("expected", "?"))
        predicted = str(row.get("predicted", "?"))
        score = row.get("score")

        print("\n" + "=" * 80)
        print(f"Errore {out_idx}/{len(mismatches)} (example_index={index})")
        print(f"question_id: {question_id}")
        print(f"expected: {expected} | predicted: {predicted} | score: {score}")

        sample = None
        if samples_by_index is not None and index < len(samples_by_index):
            sample = samples_by_index[index]
        elif sample_lookup is not None:
            key = (question_id, expected)
            candidates = sample_lookup.get(key) or []
            if candidates:
                sample = candidates.pop(0)

        if sample is None:
            continue

        print("-" * 80)
        print("QUESTION:")
        print(sample.question)
        print("\nCANDIDATE ANSWER:")
        print(sample.candidate_answer)
        print("\nGROUND TRUTH ANSWERS:")
        for i, ans in enumerate(sample.ground_truth_answers, start=1):
            print(f"  {i}. {ans}")


if __name__ == "__main__":
    main()
