"""Universal multi-source perception.

Fuses (in priority order):

1. **Accessibility tree** (UIA on Windows, AT-SPI on Linux, AX API on
   macOS).  Most reliable when the target app exposes a real tree.
2. **Adaptive computer-vision detection**
   (:mod:`agentos_orchestrator.cognition.adaptive_perception`).  Works
   on ANY app — including games, canvas apps, remote desktops, kiosk
   modes — because it operates on pure pixels.
3. **OCR** (RapidOCR if available, else pytesseract, else heuristic
   text-region detection).  Adds text labels to CV-detected boxes so
   the vision LLM can ground each mark.

The output is a :class:`PerceptionFrame` of :class:`PerceptionElement`
records that are directly compatible with
:class:`agentos_orchestrator.vision.set_of_mark.Mark`.

The class is intentionally tolerant: every fusion source is optional
and any failure falls back to the next-lower tier without raising.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .capture import CaptureResult, capture_screen
from .set_of_mark import Mark

__all__ = [
    "PerceptionElement",
    "PerceptionFrame",
    "UniversalPerceiver",
]


@dataclass(frozen=True)
class PerceptionElement:
    """A single perceived UI element."""

    x: int
    y: int
    width: int
    height: int
    label: str = ""
    role: str = ""
    source: str = "cv"
    confidence: float = 0.5
    semantic_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptionFrame:
    """A perception frame: capture + fused elements."""

    capture: CaptureResult
    elements: tuple[PerceptionElement, ...]
    a11y_used: bool
    cv_used: bool
    ocr_used: bool

    def as_marks(self, *, max_marks: int = 64) -> list[Mark]:
        out: list[Mark] = []
        for idx, elem in enumerate(self.elements[:max_marks], start=1):
            out.append(
                Mark(
                    mark_id=idx,
                    x=elem.x,
                    y=elem.y,
                    width=elem.width,
                    height=elem.height,
                    label=elem.label,
                    role=elem.role,
                    source=elem.source,
                    confidence=elem.confidence,
                    metadata={k: str(v) for k, v in elem.metadata.items()},
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Helpers — every importer is wrapped so missing libs degrade gracefully.
# --------------------------------------------------------------------------- #


def _load_adaptive_engine() -> Any | None:
    try:
        from agentos_orchestrator.cognition.adaptive_perception import (
            AdaptivePerceptionEngine,
        )
    except Exception:
        return None
    try:
        return AdaptivePerceptionEngine()
    except Exception:
        return None


def _ocr_with_rapidocr(
    png_bytes: bytes,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)
        engine = RapidOCR()
        result, _ = engine(arr)
        out: list[tuple[tuple[int, int, int, int], str, float]] = []
        if not result:
            return out
        for entry in result:
            try:
                box, text, conf = entry
            except Exception:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            out.append(
                ((x0, y0, max(1, x1 - x0), max(1, y1 - y0)), str(text), float(conf)),
            )
        return out
    except Exception:
        return []


def _ocr_with_tesseract(
    png_bytes: bytes,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    out: list[tuple[tuple[int, int, int, int], str, float]] = []
    texts = data.get("text") or []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        try:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            conf_raw = data["conf"][i]
            conf = float(conf_raw) / 100.0 if conf_raw not in ("", -1) else 0.0
        except Exception:
            continue
        out.append(((x, y, w, h), text.strip(), conf))
    return out


def _ocr_extract(
    png_bytes: bytes,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Best-available OCR; empty list if nothing installed."""
    results = _ocr_with_rapidocr(png_bytes)
    if results:
        return results
    return _ocr_with_tesseract(png_bytes)


def _a11y_collect(
    a11y_provider: Callable[[], Iterable[dict[str, Any]]] | None,
) -> list[PerceptionElement]:
    if a11y_provider is None:
        return []
    try:
        raw = list(a11y_provider())
    except Exception:
        return []
    out: list[PerceptionElement] = []
    for entry in raw:
        try:
            bounds = entry.get("bounds") or entry.get("bounding_rect") or {}
            x = int(bounds.get("x", 0))
            y = int(bounds.get("y", 0))
            w = int(bounds.get("width", 0))
            h = int(bounds.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            label = str(
                entry.get("name") or entry.get("label") or entry.get("text") or ""
            )
            role = str(entry.get("role") or entry.get("control_type") or "")
            out.append(
                PerceptionElement(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    label=label,
                    role=role,
                    source="a11y",
                    confidence=1.0,
                    metadata={"automation_id": str(entry.get("automation_id") or "")},
                ),
            )
        except Exception:
            continue
    return out


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _merge_elements(
    a11y: Sequence[PerceptionElement],
    cv: Sequence[PerceptionElement],
    ocr_hits: Sequence[tuple[tuple[int, int, int, int], str, float]],
    *,
    iou_threshold: float = 0.45,
) -> list[PerceptionElement]:
    """Fuse a11y + CV + OCR into a single non-overlapping mark list.

    Strategy:
    * A11y elements always win (highest fidelity, real semantics).
    * CV elements are kept only if they do not heavily overlap an a11y box.
    * OCR text is attached to whichever element overlaps it the most;
      orphan OCR boxes become their own marks (covers canvas/games).
    """
    fused: list[PerceptionElement] = list(a11y)
    for c in cv:
        cbox = (c.x, c.y, c.width, c.height)
        if any(
            _box_iou(cbox, (e.x, e.y, e.width, e.height)) >= iou_threshold
            for e in fused
        ):
            continue
        fused.append(c)
    # Attach OCR
    for box, text, conf in ocr_hits:
        attached = False
        for idx, e in enumerate(fused):
            ebox = (e.x, e.y, e.width, e.height)
            if _box_iou(box, ebox) >= 0.2:
                if not e.label:
                    fused[idx] = PerceptionElement(
                        x=e.x,
                        y=e.y,
                        width=e.width,
                        height=e.height,
                        label=text,
                        role=e.role,
                        source=e.source,
                        confidence=max(e.confidence, conf),
                        semantic_score=e.semantic_score,
                        metadata={**e.metadata, "ocr_text": text},
                    )
                attached = True
                break
        if not attached:
            x, y, w, h = box
            fused.append(
                PerceptionElement(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    label=text,
                    role="text",
                    source="ocr",
                    confidence=conf,
                ),
            )
    # De-duplicate by IoU 0.85 (same element from two sources)
    deduped: list[PerceptionElement] = []
    for e in fused:
        if any(
            _box_iou(
                (e.x, e.y, e.width, e.height),
                (f.x, f.y, f.width, f.height),
            )
            >= 0.85
            for f in deduped
        ):
            continue
        deduped.append(e)
    # Stable order: a11y first, then by confidence
    deduped.sort(
        key=lambda e: (0 if e.source == "a11y" else 1, -e.confidence),
    )
    return deduped


# --------------------------------------------------------------------------- #
# Universal perceiver
# --------------------------------------------------------------------------- #


class UniversalPerceiver:
    """Fuses a11y tree + adaptive CV + OCR into a single mark list.

    Parameters
    ----------
    a11y_provider:
        Optional callable returning a11y elements as dicts (UIA/AT-SPI/AX).
    enable_cv:
        Run :class:`AdaptivePerceptionEngine` on the captured pixels.
    enable_ocr:
        Run RapidOCR/Tesseract on the captured pixels.
    max_elements:
        Hard cap on returned marks (prevents giant mark tables).
    """

    def __init__(
        self,
        *,
        a11y_provider: Callable[[], Iterable[dict[str, Any]]] | None = None,
        enable_cv: bool = True,
        enable_ocr: bool = True,
        max_elements: int = 64,
    ) -> None:
        self._a11y_provider = a11y_provider
        self._cv_engine = _load_adaptive_engine() if enable_cv else None
        self._enable_ocr = enable_ocr
        self._max_elements = int(max_elements)

    def perceive(
        self,
        *,
        objective: str = "",
        history: list[dict[str, Any]] | None = None,
        region: tuple[int, int, int, int] | None = None,
        capture: CaptureResult | None = None,
    ) -> PerceptionFrame:
        """Capture the screen and return a fused perception frame."""
        cap = capture or capture_screen(region=region)
        png = cap.png_bytes

        a11y_elements = _a11y_collect(self._a11y_provider)
        a11y_used = bool(a11y_elements)

        cv_elements: list[PerceptionElement] = []
        cv_used = False
        if self._cv_engine is not None:
            try:
                perceived = self._cv_engine.perceive(
                    png,
                    objective or "interact",
                    history or [],
                )
            except Exception:
                perceived = []
            for p in perceived:
                try:
                    cv_elements.append(
                        PerceptionElement(
                            x=int(p.x),
                            y=int(p.y),
                            width=int(p.width),
                            height=int(p.height),
                            label=str(getattr(p, "text", "") or ""),
                            role=str(getattr(p, "element_type", "") or ""),
                            source="cv",
                            confidence=float(getattr(p, "confidence", 0.5) or 0.5),
                            semantic_score=float(
                                getattr(p, "semantic_score", 0.0) or 0.0
                            ),
                        ),
                    )
                except Exception:
                    continue
            cv_used = True

        ocr_hits: list[tuple[tuple[int, int, int, int], str, float]] = []
        if self._enable_ocr:
            ocr_hits = _ocr_extract(png)
        ocr_used = bool(ocr_hits)

        fused = _merge_elements(a11y_elements, cv_elements, ocr_hits)
        fused = fused[: self._max_elements]
        return PerceptionFrame(
            capture=cap,
            elements=tuple(fused),
            a11y_used=a11y_used,
            cv_used=cv_used,
            ocr_used=ocr_used,
        )
