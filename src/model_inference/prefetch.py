from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .dataset import (
    MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET,
    MULTI_PIXMO_CAP_DATASET,
)
from .offline_assets import build_cached_image_path, default_hf_home, default_image_cache_root

DEFAULT_DATASETS = [
    MULTI_PIXMO_CAP_DATASET,
    MULTI_PIXMO_ASK_MODEL_ANYTHING_DATASET,
]
DEFAULT_SUBSETS = ["it", "en", "es"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch multi-pixmo datasets in HF_HOME and download the first N images per subset for offline GPU jobs."
    )
    parser.add_argument("--dataset-name", action="append", dest="dataset_names", default=[])
    parser.add_argument("--subset", action="append", dest="subsets", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--image-cache-root", type=Path, default=None)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser.parse_args()


def _load_split(dataset_name: str, subset: str, split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `datasets` package is required to prefetch Hugging Face datasets. Install project dependencies first."
        ) from exc

    return load_dataset(dataset_name, subset, split=split)


def _download_image(image_url: str, destination: Path, timeout: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return "cached"

    request = Request(image_url, headers={"User-Agent": "llm-as-a-judge/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to download image from {image_url}: {exc}") from exc

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(destination)
    return "downloaded"


def _write_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    dataset_names = args.dataset_names or list(DEFAULT_DATASETS)
    subsets = args.subsets or list(DEFAULT_SUBSETS)
    image_cache_root = (args.image_cache_root or default_image_cache_root()).expanduser()
    hf_home = default_hf_home()

    print(f"HF_HOME: {hf_home}")
    print(f"Image cache root: {image_cache_root}")
    print(f"Split: {args.split}")
    print(f"Max images per subset: {args.max_images}")

    for dataset_name in dataset_names:
        for subset in subsets:
            print("-" * 60)
            print(f"Prefetch dataset={dataset_name} subset={subset} split={args.split}")
            dataset = _load_split(dataset_name=dataset_name, subset=subset, split=args.split)
            total_rows = len(dataset)
            target_rows = min(args.max_images, total_rows)
            print(f"Cached dataset rows available: {total_rows}")
            print(f"Downloading images for first {target_rows} rows")

            downloaded = 0
            cached = 0
            errors: list[dict[str, str | int]] = []
            manifest_rows: list[dict[str, str | int]] = []

            for index in range(target_rows):
                row = dict(dataset[index])
                image_url = str(row.get("image_url", "")).strip()
                if not image_url:
                    errors.append({"row_index": index, "error": "missing image_url"})
                    continue

                destination = build_cached_image_path(
                    dataset_name=dataset_name,
                    subset=subset,
                    split=args.split,
                    image_url=image_url,
                    image_cache_root=image_cache_root,
                )

                try:
                    status = _download_image(image_url=image_url, destination=destination, timeout=args.request_timeout)
                    if status == "downloaded":
                        downloaded += 1
                    else:
                        cached += 1
                    manifest_rows.append(
                        {
                            "row_index": index,
                            "id": str(row.get("id", index)),
                            "image_url": image_url,
                            "image_path": str(destination),
                            "status": status,
                        }
                    )
                except Exception as exc:
                    errors.append({"row_index": index, "image_url": image_url, "error": str(exc)})

            manifest_path = image_cache_root / dataset_name.replace("/", "__") / subset / args.split / args.manifest_name
            _write_manifest(
                manifest_path,
                {
                    "dataset_name": dataset_name,
                    "subset": subset,
                    "split": args.split,
                    "hf_home": str(hf_home),
                    "image_cache_root": str(image_cache_root),
                    "max_images": args.max_images,
                    "downloaded": downloaded,
                    "cached": cached,
                    "errors": errors,
                    "rows": manifest_rows,
                },
            )
            print(f"Completed: downloaded={downloaded}, cached={cached}, errors={len(errors)}")
            print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
