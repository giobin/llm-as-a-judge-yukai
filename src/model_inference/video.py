from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .offline_assets import is_remote_url

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - handled at runtime with a clear error.
    VideoReader = None
    cpu = None


def resize_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    if max_dim <= 0:
        raise ValueError("max_dim must be > 0.")

    width, height = img.size
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    if scale == 1.0:
        return img

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.BICUBIC)


def compute_uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0.")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0.")

    k = max(1, min(num_frames, total_frames))
    return np.linspace(0, total_frames - 1, num=k, dtype=np.int64).tolist()


def resolve_video_path(row: dict[str, Any], videos_dir: Path) -> Path:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos_dir is not a directory: {videos_dir}")

    candidates: list[Path] = []

    video_id = str(row.get("video_id", "")).strip()
    if video_id:
        candidates.append(videos_dir / f"{video_id}.mp4")
        candidates.append(videos_dir / f"{video_id}.MP4")

    file_name = str(row.get("file_name", "")).strip()
    if file_name:
        basename = Path(file_name).name
        if basename:
            candidates.append(videos_dir / basename)
            if not basename.lower().endswith(".mp4"):
                candidates.append(videos_dir / f"{basename}.mp4")

    seen: set[str] = set()
    deduped_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(path) for path in deduped_candidates) or "<none>"
    raise FileNotFoundError(
        f"Unable to resolve local video for row id={row.get('id')} video_id={video_id!r}. Tried: {tried}"
    )


def sample_video_frames(
    video_path: Path,
    num_frames: int,
    max_image_dimension: int,
) -> tuple[list[Image.Image], list[int]]:
    if VideoReader is None or cpu is None:
        raise RuntimeError(
            "decord is required for video frame extraction. Install with `pip install decord`."
        )

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    indices = compute_uniform_indices(total_frames=total_frames, num_frames=num_frames)
    batch = vr.get_batch(indices).asnumpy()

    frames: list[Image.Image] = []
    for frame_array in batch:
        image = Image.fromarray(frame_array, mode="RGB")
        image = resize_max_dim(image, max_dim=max_image_dimension)
        frames.append(image)

    return frames, indices


def format_image_as_data_uri(img: Image.Image) -> str:
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def format_frames_as_data_uris(frames: list[Image.Image]) -> list[str]:
    return [format_image_as_data_uri(frame) for frame in frames]


def load_image(image_locator: str, max_image_dimension: int) -> Image.Image:
    if is_remote_url(image_locator):
        return load_image_from_url(image_locator, max_image_dimension=max_image_dimension)
    return load_image_from_file(Path(image_locator), max_image_dimension=max_image_dimension)


def load_image_from_file(image_path: Path, max_image_dimension: int) -> Image.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"Local image file not found: {image_path}")

    try:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unable to decode image from local file {image_path}: {exc}") from exc

    return resize_max_dim(image, max_dim=max_image_dimension)


def load_image_from_url(image_url: str, max_image_dimension: int) -> Image.Image:
    request = Request(
        image_url,
        headers={
            "User-Agent": "llm-as-a-judge/0.1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to download image from {image_url}: {exc}") from exc

    try:
        with Image.open(io.BytesIO(payload)) as raw_image:
            image = raw_image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unable to decode image downloaded from {image_url}: {exc}") from exc

    return resize_max_dim(image, max_dim=max_image_dimension)


def format_image_list_as_data_uris(images: list[Image.Image]) -> list[str]:
    return [format_image_as_data_uri(image) for image in images]
