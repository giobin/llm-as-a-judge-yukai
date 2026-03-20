from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from model_inference.video import compute_uniform_indices, resolve_video_path, resize_max_dim


def test_compute_uniform_indices() -> None:
    indices = compute_uniform_indices(total_frames=10, num_frames=4)
    assert indices == [0, 3, 6, 9]


def test_resize_max_dim_preserves_ratio() -> None:
    img = Image.new("RGB", (400, 200))
    out = resize_max_dim(img, max_dim=100)
    assert out.size == (100, 50)


def test_resolve_video_path_from_video_id(tmp_path: Path) -> None:
    video = tmp_path / "Video12.mp4"
    video.write_bytes(b"fake")

    row = {"video_id": "Video12", "file_name": "Videos/Other.mp4"}
    resolved = resolve_video_path(row=row, videos_dir=tmp_path)
    assert resolved == video


def test_resolve_video_path_fallback_to_file_name(tmp_path: Path) -> None:
    video = tmp_path / "Video5.mp4"
    video.write_bytes(b"fake")

    row = {"video_id": "MissingVideo", "file_name": "Videos/Video5.mp4"}
    resolved = resolve_video_path(row=row, videos_dir=tmp_path)
    assert resolved == video


def test_resolve_video_path_raises_when_missing(tmp_path: Path) -> None:
    row = {"id": 7, "video_id": "Video404", "file_name": "Videos/Video404.mp4"}
    with pytest.raises(FileNotFoundError):
        resolve_video_path(row=row, videos_dir=tmp_path)
