from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

_ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def default_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    return Path.home() / ".cache" / "huggingface"


def default_image_cache_root() -> Path:
    return default_hf_home() / "llm_as_a_judge_assets" / "images"


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.strip().replace("/", "__")


def infer_image_extension(image_url: str) -> str:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in _ALLOWED_EXTENSIONS:
        return suffix
    return ".img"


def build_cached_image_path(
    dataset_name: str,
    subset: str | None,
    split: str,
    image_url: str,
    image_cache_root: Path | None = None,
) -> Path:
    root = (image_cache_root or default_image_cache_root()).expanduser()
    subset_slug = (subset or "default").strip() or "default"
    digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    filename = f"{digest}{infer_image_extension(image_url)}"
    return root / dataset_slug(dataset_name) / subset_slug / split / filename


def is_remote_url(value: str) -> bool:
    scheme = urlparse(value).scheme.lower()
    return scheme in {"http", "https"}
