from __future__ import annotations

from io import BytesIO

from PIL import Image

from model_inference.video import load_image_from_url


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_load_image_from_url_resizes_downloaded_image(monkeypatch) -> None:
    image = Image.new("RGB", (800, 400), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.com/image.png"
        assert timeout == 30
        return _FakeResponse(buffer.getvalue())

    monkeypatch.setattr("model_inference.video.urlopen", fake_urlopen)

    loaded = load_image_from_url("https://example.com/image.png", max_image_dimension=200)

    assert loaded.size == (200, 100)
