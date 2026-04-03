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


def build_user_content(
    prompt: str,
    media_items: list[Any],
    media_part_type: str = "image_url",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    for placeholder in (VIDEO_PLACEHOLDER, IMAGE_PLACEHOLDER):
        if placeholder in prompt:
            segments = prompt.split(placeholder)
            for index, segment in enumerate(segments):
                if segment:
                    content.append({"type": "text", "text": segment})
                if index < len(segments) - 1:
                    content.extend(_build_media_parts(media_items, media_part_type))
            return content

    content.extend(_build_media_parts(media_items, media_part_type))
    content.append({"type": "text", "text": prompt})
    return content


def _build_media_parts(media_items: list[Any], media_part_type: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for item in media_items:
        if media_part_type == "image_url":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(item)},
                }
            )
        elif media_part_type == "image_pil":
            parts.append({"type": "image_pil", "image_pil": item})
        else:
            raise ValueError(f"Unsupported media_part_type: {media_part_type}")
    return parts


def build_vllm_in_process_user_content(
    prompt: str,
    media_images: list[Image.Image],
) -> list[dict[str, Any]]:
    return build_user_content(
        prompt=prompt,
        media_items=media_images,
        media_part_type="image_pil",
    )
