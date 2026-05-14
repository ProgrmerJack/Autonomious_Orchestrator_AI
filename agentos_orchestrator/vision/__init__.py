"""AgentOS vision pipeline.

Pixel-level screen perception that makes AgentOS able to operate ANY
application — even ones with no accessibility tree and no registered
app-family adapter — by combining:

* Cross-platform screenshot capture (``capture.py``)
* Set-of-Mark numbered overlay rendering (``set_of_mark.py``)
* Universal perception that fuses UIA/AT-SPI/AX trees with adaptive
  computer-vision element detection and OCR (``universal_perceiver.py``)
* A thin frontier-API adapter (``pixel_scraper.py``) that turns any
  on-screen pixel region into a target a vision LLM can act on.

Design tenets:

1. *Degrade, never refuse.*  Every layer is optional; missing libs
   (mss, opencv, pillow, pytesseract, rapidocr-onnxruntime) reduce
   detail but never raise.
2. *Zero net-new heavy deps.*  We bring in only what is already
   available in the existing environment; everything else is detected
   at import time and the feature self-disables with a clear status.
3. *Bound RAM.*  Screenshots are produced as ``bytes`` (PNG) and
   marks are scored/clipped to a hard cap so we never hold giant
   numpy arrays after a frame is consumed.
"""

from __future__ import annotations

from .capture import (
    CaptureBackend,
    CaptureResult,
    CaptureUnavailable,
    capture_screen,
    list_capture_backends,
)
from .set_of_mark import (
    Mark,
    SetOfMarkResult,
    render_set_of_mark,
)
from .universal_perceiver import (
    UniversalPerceiver,
    PerceptionFrame,
    PerceptionElement,
)
from .pixel_scraper import PixelScreenScraper

__all__ = [
    "CaptureBackend",
    "CaptureResult",
    "CaptureUnavailable",
    "Mark",
    "PerceptionElement",
    "PerceptionFrame",
    "PixelScreenScraper",
    "SetOfMarkResult",
    "UniversalPerceiver",
    "capture_screen",
    "list_capture_backends",
    "render_set_of_mark",
]
