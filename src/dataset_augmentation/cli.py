from __future__ import annotations

import argparse
import os
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder
from openai import OpenAI
from tqdm.auto import tqdm

from .foils import generate_foils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a Hugging Face dataset with wrong_answer1 and wrong_answer2 using OpenAI."
    )
    parser.add_argument("--source-dataset", default="caput/MAIA_dev_set_eng")
    parser.add_argument("--target-owner", default="giobin")
    parser.add_argument("--target-dataset", default=None)
    parser.add_argument("--subset", default=None, help="Config/subset name. If omitted, process all configs.")

    parser.add_argument("--openai-model", default="gpt-5.2")
    parser.add_argument("--openai-api-key", default=None)

    parser.add_argument("--max-rows", type=int, default=0, help="If >0, limit rows per split (debug only).")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-push", action="store_true", help="Generate locally but do not push to Hub.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def derive_target_repo_id(source_dataset: str, owner: str, explicit_target: str | None) -> str:
    if explicit_target:
        return f"{owner}/{explicit_target}"

    dataset_name = source_dataset.split("/", 1)[-1]
    return f"{owner}/{dataset_name}_2wrong"


def iter_configs(source_dataset: str, subset: str | None) -> list[str | None]:
    if subset:
        return [subset]

    builder = load_dataset_builder(source_dataset)
    config_names = list(builder.builder_configs.keys())
    if not config_names:
        return [None]
    return config_names


def _prepare_rows(split_ds: Dataset, max_rows: int) -> Iterable[dict]:
    if max_rows > 0:
        return split_ds.select(range(min(max_rows, len(split_ds))))
    return split_ds


def augment_split(
    split_ds: Dataset,
    client: OpenAI,
    model_name: str,
    max_rows: int,
    max_output_tokens: int,
    max_retries: int,
    verbose: bool,
) -> Dataset:
    rows = _prepare_rows(split_ds, max_rows=max_rows)
    wrong_answer1_values: list[str] = []
    wrong_answer2_values: list[str] = []

    progress = tqdm(rows, desc="Generating foils", unit="row")
    for idx, row in enumerate(progress, start=1):
        question = str(row.get("question", "")).strip()
        answer1 = str(row.get("answer1", "")).strip()
        answer2 = str(row.get("answer2", "")).strip()

        if not question or not answer1 or not answer2:
            wrong_answer1_values.append("")
            wrong_answer2_values.append("")
            continue

        foils = generate_foils(
            client=client,
            model_name=model_name,
            question=question,
            answer1=answer1,
            answer2=answer2,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )
        wrong_answer1_values.append(foils.wrong_answer1)
        wrong_answer2_values.append(foils.wrong_answer2)

        if verbose:
            tqdm.write(f"\n[Row {idx}]")
            tqdm.write(f"Q: {question}")
            tqdm.write(f"A1: {answer1}")
            tqdm.write(f"A2: {answer2}")
            tqdm.write(f"wrong_answer1: {foils.wrong_answer1}")
            tqdm.write(f"wrong_answer2: {foils.wrong_answer2}")

    if "wrong_answer1" in rows.column_names:
        rows = rows.remove_columns(["wrong_answer1"])
    if "wrong_answer2" in rows.column_names:
        rows = rows.remove_columns(["wrong_answer2"])

    rows = rows.add_column("wrong_answer1", wrong_answer1_values)
    rows = rows.add_column("wrong_answer2", wrong_answer2_values)
    return rows


def load_datasetdict(source_dataset: str, config_name: str | None) -> DatasetDict:
    if config_name is None:
        loaded = load_dataset(source_dataset)
    else:
        loaded = load_dataset(source_dataset, config_name)

    if not isinstance(loaded, DatasetDict):
        raise TypeError("Expected a DatasetDict from load_dataset.")
    return loaded


def main() -> None:
    args = parse_args()

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set env var or pass --openai-api-key.")

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    target_repo_id = derive_target_repo_id(
        source_dataset=args.source_dataset,
        owner=args.target_owner,
        explicit_target=args.target_dataset,
    )

    client = OpenAI(api_key=api_key)

    configs = iter_configs(source_dataset=args.source_dataset, subset=args.subset)
    if args.verbose:
        print(f"Source dataset: {args.source_dataset}")
        print(f"Configs to process: {configs}")
        print(f"Target repo: {target_repo_id}")

    for config_index, config_name in enumerate(configs):
        cfg_label = config_name if config_name is not None else "default"
        print(f"\nProcessing config: {cfg_label}")
        dataset_dict = load_datasetdict(source_dataset=args.source_dataset, config_name=config_name)

        augmented = DatasetDict()
        for split_name, split_ds in dataset_dict.items():
            print(f"- split: {split_name} ({len(split_ds)} rows)")
            augmented_split = augment_split(
                split_ds=split_ds,
                client=client,
                model_name=args.openai_model,
                max_rows=args.max_rows,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                verbose=args.verbose,
            )
            augmented[split_name] = augmented_split

        if args.no_push:
            print(f"Skipping push for config '{cfg_label}' due to --no-push")
            continue

        if not hf_token:
            raise ValueError(
                "HF token missing. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, "
                "or pass --hf-token."
            )

        augmented.push_to_hub(
            target_repo_id,
            config_name=config_name if config_name is not None else "default",
            set_default=(config_index == 0),
            token=hf_token,
            private=args.private,
        )
        print(f"Pushed config '{cfg_label}' to {target_repo_id}")

    print("Dataset augmentation completed.")


if __name__ == "__main__":
    main()
