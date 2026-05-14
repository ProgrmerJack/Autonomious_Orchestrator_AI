"""Cross-platform screenshot capture.

Backend resolution order:

1. ``mss``  — fast, zero-copy, works on Windows / macOS / Linux / Wayland-XWayland.
2. ``PIL.ImageGrab`` — fallback on Windows and macOS when mss is missing.
3. ``pyscreeze`` — last-resort fallback.
4. ``CaptureUnavailable`` — raised only when *no* backend exists.

All backends return a :class:`CaptureResult` with PNG bytes + (w, h)
so downstream code never has to know which library succeeded.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "CaptureBackend",
    "CaptureResult",
    "CaptureUnavailable",
    "capture_screen",
    "list_capture_backends",
]


class CaptureUnavailable(RuntimeError):
    """Raised only when no screenshot backend is installed."""


@dataclass(frozen=True)
class CaptureResult:
    """A captured screen region.

    Attributes
    ----------
    png_bytes:
        PNG-encoded image bytes (so callers can persist / hash / send
        to a VLM without re-encoding).
    width / height:
        Pixel dimensions of the capture.
    backend:
        Name of the backend that produced this capture.
    captured_at:
        Monotonic timestamp (seconds) when capture finished.
    """

    png_bytes: bytes
    width: int
    height: int
    backend: str
    captured_at: float


@dataclass(frozen=True)
class CaptureBackend:
    """Describes a capture backend that is available on this host."""

    name: str
    capture: Callable[[tuple[int, int, int, int] | None], CaptureResult]


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #


def _try_mss() -> CaptureBackend | None:
    try:
        import mss  # type: ignore[import-not-found]
        import mss.tools  # type: ignore[import-not-found]
    except Exception:
        return None

    def _capture(
        region: tuple[int, int, int, int] | None,
    ) -> CaptureResult:
        with mss.mss() as sct:
            if region is None:
                monitor = sct.monitors[0]
            else:
                left, top, width, height = region
                monitor = {
                    "left": int(left),
                    "top": int(top),
                    "width": int(width),
                    "height": int(height),
                }
            raw = sct.grab(monitor)
            png = mss.tools.to_png(raw.rgb, raw.size)
            return CaptureResult(
                png_bytes=png,
                width=int(raw.size[0]),
                height=int(raw.size[1]),
                backend="mss",
                captured_at=time.monotonic(),
            )

    return CaptureBackend(name="mss", capture=_capture)


def _try_pil_imagegrab() -> CaptureBackend | None:
    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]
    except Exception:
        return None

    def _capture(
        region: tuple[int, int, int, int] | None,
    ) -> CaptureResult:
        bbox = None
        if region is not None:
            left, top, width, height = region
            bbox = (int(left), int(top), int(left + width), int(top + height))
        img = ImageGrab.grab(bbox=bbox, all_screens=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return CaptureResult(
            png_bytes=buf.getvalue(),
            width=img.width,
            height=img.height,
            backend="pillow.ImageGrab",
            captured_at=time.monotonic(),
        )

    return CaptureBackend(name="pillow.ImageGrab", capture=_capture)


def _try_pyscreeze() -> CaptureBackend | None:
    try:
        import pyscreeze  # type: ignore[import-not-found]
    except Exception:
        return None

    def _capture(
        region: tuple[int, int, int, int] | None,
    ) -> CaptureResult:
        img = pyscreeze.screenshot(region=region)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return CaptureResult(
            png_bytes=buf.getvalue(),
            width=img.width,
            height=img.height,
            backend="pyscreeze",
            captured_at=time.monotonic(),
        )

    return CaptureBackend(name="pyscreeze", capture=_capture)


_BACKEND_FACTORIES: tuple[Callable[[], CaptureBackend | None], ...] = (
    _try_mss,
    _try_pil_imagegrab,
    _try_pyscreeze,
)


def list_capture_backends() -> list[CaptureBackend]:
    """Return every capture backend that imports successfully *now*."""
    found: list[CaptureBackend] = []
    for factory in _BACKEND_FACTORIES:
        try:
            backend = factory()
        except Exception:
            backend = None
        if backend is not None:
            found.append(backend)
    return found


def capture_screen(
    region: tuple[int, int, int, int] | None = None,
    *,
    preferred_backend: str | None = None,
) -> CaptureResult:
    """Capture a screenshot of *region* (or the full virtual desktop).

    Parameters
    ----------
    region:
        ``(left, top, width, height)`` in screen-space pixels.  ``None``
        captures the full virtual desktop (spanning all monitors).
    preferred_backend:
        Optional explicit backend name (``"mss"``, ``"pillow.ImageGrab"``,
        ``"pyscreeze"``).  Falls back to the standard resolution order
        when the preferred backend is unavailable or fails.

    Raises
    ------
    CaptureUnavailable
        Only when *no* capture backend is importable.
    """
    backends = list_capture_backends()
    if not backends:
        raise CaptureUnavailable(
            "No screenshot backend available; install one of: mss, Pillow, pyscreeze."
        )
    if preferred_backend:
        backends.sort(key=lambda b: 0 if b.name == preferred_backend else 1)
    last_error: Exception | None = None
    for backend in backends:
        try:
            return backend.capture(region)
        except Exception as exc:
            last_error = exc
            continue
    raise CaptureUnavailable(f"All capture backends failed; last error: {last_error!r}")
