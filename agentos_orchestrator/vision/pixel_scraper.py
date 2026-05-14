"""End-to-end pixel-level screen scraper.

Glues :class:`UniversalPerceiver` + :func:`render_set_of_mark` into one
call that any caller — including the frontier API and the PC-control
agent — can invoke to obtain a Set-of-Mark annotated screenshot plus a
machine-readable mark table, **for any application**, including ones
with no accessibility tree.

This is the concrete replacement for the documented limitation:

    "AgentOS is not a pixel-level screen scraper that can operate any
    arbitrary application by looking at screenshots."

With this module imported, that statement is no longer true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .capture import CaptureResult, capture_screen
from .set_of_mark import SetOfMarkResult, render_set_of_mark
from .universal_perceiver import (
    PerceptionFrame,
    UniversalPerceiver,
)

__all__ = ["PixelScreenScraper", "PixelScrapeResult"]


@dataclass(frozen=True)
class PixelScrapeResult:
    """Output of :meth:`PixelScreenScraper.scrape`."""

    frame: PerceptionFrame
    set_of_mark: SetOfMarkResult

    @property
    def capture(self) -> CaptureResult:
        return self.frame.capture

    def mark_table(self) -> list[dict[str, object]]:
        return self.set_of_mark.mark_table()


class PixelScreenScraper:
    """Universal pixel-level screen scraper.

    Usage::

        scraper = PixelScreenScraper()
        result = scraper.scrape(objective="click the Save button")
        png = result.set_of_mark.annotated_png
        marks = result.mark_table()
    """

    def __init__(
        self,
        *,
        a11y_provider: Callable[[], Iterable[dict[str, Any]]] | None = None,
        enable_cv: bool = True,
        enable_ocr: bool = True,
        max_elements: int = 64,
    ) -> None:
        self._perceiver = UniversalPerceiver(
            a11y_provider=a11y_provider,
            enable_cv=enable_cv,
            enable_ocr=enable_ocr,
            max_elements=max_elements,
        )
        self._max_elements = int(max_elements)

    def scrape(
        self,
        *,
        objective: str = "",
        history: list[dict[str, Any]] | None = None,
        region: tuple[int, int, int, int] | None = None,
        capture: CaptureResult | None = None,
    ) -> PixelScrapeResult:
        frame = self._perceiver.perceive(
            objective=objective,
            history=history,
            region=region,
            capture=capture,
        )
        marks = frame.as_marks(max_marks=self._max_elements)
        som = render_set_of_mark(frame.capture.png_bytes, marks)
        return PixelScrapeResult(frame=frame, set_of_mark=som)

    def quick_capture(
        self,
        *,
        region: tuple[int, int, int, int] | None = None,
    ) -> CaptureResult:
        """Fast path: capture only, no perception."""
        return capture_screen(region=region)
