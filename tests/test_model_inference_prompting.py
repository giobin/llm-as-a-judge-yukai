from __future__ import annotations

import pytest

from model_inference.prompting import build_user_content, render_prompt


def test_render_prompt_requires_question_placeholder() -> None:
    with pytest.raises(ValueError):
        render_prompt("No placeholder here", {"question": "Q?"})


def test_render_prompt_replaces_question() -> None:
    rendered = render_prompt("Question: {question}", {"question": "Where?"})
    assert rendered == "Question: Where?"


def test_render_prompt_replaces_transcript() -> None:
    rendered = render_prompt("Transcript: {transcript}", {"transcript": "A red bus"})
    assert rendered == "Transcript: A red bus"


def test_build_user_content_with_video_placeholder() -> None:
    content = build_user_content(
        prompt="Intro <video> Outro",
        media_data_uris=["data:image/png;base64,AAA", "data:image/png;base64,BBB"],
    )

    assert content[0] == {"type": "text", "text": "Intro "}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].endswith("AAA")
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].endswith("BBB")
    assert content[3] == {"type": "text", "text": " Outro"}


def test_build_user_content_without_video_placeholder_prepends_images() -> None:
    content = build_user_content(
        prompt="Only text",
        media_data_uris=["data:image/png;base64,CCC"],
    )

    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "Only text"}


def test_build_user_content_with_image_placeholder() -> None:
    content = build_user_content(
        prompt="Observe <image> Caption it",
        media_data_uris=["data:image/png;base64,DDD"],
    )

    assert content[0] == {"type": "text", "text": "Observe "}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].endswith("DDD")
    assert content[2] == {"type": "text", "text": " Caption it"}
