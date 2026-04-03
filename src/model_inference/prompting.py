from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any

from PIL import Image

VIDEO_PLACEHOLDER = "<video>"
IMAGE_PLACEHOLDER = "<image>"


def load_prompt_template(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    field_names = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    if not field_names:
        raise ValueError("Prompt template must include at least one placeholder.")

    missing = sorted(field_name for field_name in field_names if field_name not in variables)
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Prompt template missing values for placeholders: {missing_list}")

    stringified_variables = {key: str(value) for key, value in variables.items()}
    return template.format_map(stringified_variables)


def _build_content_with_media_parts(
    prompt: str,
    media_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    for placeholder in (VIDEO_PLACEHOLDER, IMAGE_PLACEHOLDER):
        if placeholder in prompt:
            segments = prompt.split(placeholder)
            for index, segment in enumerate(segments):
                if segment:
                    content.append({"type": "text", "text": segment})
                if index < len(segments) - 1:
                    content.extend(media_parts)
            return content

    content.extend(media_parts)
    content.append({"type": "text", "text": prompt})
    return content


def build_user_content(prompt: str, media_data_uris: list[str]) -> list[dict[str, Any]]:
    media_parts = [
        {
            "type": "image_url",
            "image_url": {"url": media_uri},
        }
        for media_uri in media_data_uris
    ]
    return _build_content_with_media_parts(prompt=prompt, media_parts=media_parts)


def build_vllm_in_process_user_content(
    prompt: str,
    media_images: list[Image.Image],
) -> list[dict[str, Any]]:
    media_parts = [
        {
            "type": "image_pil",
            "image_pil": image,
        }
        for image in media_images
    ]
    return _build_content_with_media_parts(prompt=prompt, media_parts=media_parts)
