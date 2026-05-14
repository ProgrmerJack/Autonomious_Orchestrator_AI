"""Tests for the universal vision/perception pipeline."""

from __future__ import annotations

import io

import pytest

from agentos_orchestrator.vision import (
    Mark,
    PerceptionElement,
    PerceptionFrame,
    PixelScreenScraper,
    list_capture_backends,
    render_set_of_mark,
)
from agentos_orchestrator.vision.capture import CaptureResult


def _make_blank_png(width: int = 320, height: int = 200) -> bytes:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow not available")
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(245, 245, 245)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def _stub_capture(png: bytes, width: int = 320, height: int = 200) -> CaptureResult:
    return CaptureResult(
        png_bytes=png,
        width=width,
        height=height,
        backend="stub",
        captured_at=0.0,
    )


def test_list_capture_backends_returns_at_least_one() -> None:
    backends = list_capture_backends()
    assert backends, "No capture backend detected on this host"


def test_render_set_of_mark_overlays_numbered_marks() -> None:
    png = _make_blank_png()
    marks = [
        Mark(
            mark_id=1,
            x=20,
            y=20,
            width=80,
            height=40,
            label="Button A",
            role="button",
            source="cv",
            confidence=0.9,
        ),
        Mark(
            mark_id=2,
            x=120,
            y=60,
            width=90,
            height=30,
            label="Field B",
            role="text",
            source="ocr",
            confidence=0.7,
        ),
    ]
    result = render_set_of_mark(png, marks, max_marks=8)
    assert result.annotated_png
    assert len(result.marks) == 2
    table = result.mark_table()
    assert table[0]["id"] == 1
    assert table[1]["id"] == 2
    assert table[0]["center_x"] == 20 + 40
    assert table[0]["center_y"] == 20 + 20


def test_perception_frame_as_marks_caps_count() -> None:
    elements = tuple(
        PerceptionElement(
            x=i,
            y=i,
            width=10,
            height=10,
            label=f"e{i}",
            role="button",
            source="cv",
            confidence=0.5,
        )
        for i in range(20)
    )
    frame = PerceptionFrame(
        capture=_stub_capture(_make_blank_png()),
        elements=elements,
        a11y_used=False,
        cv_used=True,
        ocr_used=False,
    )
    marks = frame.as_marks(max_marks=5)
    assert len(marks) == 5
    assert [m.mark_id for m in marks] == [1, 2, 3, 4, 5]


def test_pixel_screen_scraper_scrape_uses_stubbed_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = PixelScreenScraper(enable_ocr=False, enable_cv=False)
    blank = _make_blank_png()

    def _fake_capture(region=None, preferred_backend=None):  # noqa: ARG001
        return _stub_capture(blank)

    monkeypatch.setattr(
        "agentos_orchestrator.vision.pixel_scraper.capture_screen",
        _fake_capture,
    )
    monkeypatch.setattr(
        "agentos_orchestrator.vision.universal_perceiver.capture_screen",
        _fake_capture,
    )
    result = scraper.scrape(objective="open menu")
    assert result.set_of_mark.annotated_png
    assert isinstance(result.set_of_mark.marks, tuple)


def test_pixel_screen_scraper_quick_capture_returns_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = PixelScreenScraper(enable_ocr=False, enable_cv=False)
    blank = _make_blank_png()

    def _fake_capture(region=None, preferred_backend=None):  # noqa: ARG001
        return _stub_capture(blank)

    monkeypatch.setattr(
        "agentos_orchestrator.vision.pixel_scraper.capture_screen",
        _fake_capture,
    )
    capture = scraper.quick_capture()
    assert capture.png_bytes == blank
    assert capture.width == 320 and capture.height == 200
