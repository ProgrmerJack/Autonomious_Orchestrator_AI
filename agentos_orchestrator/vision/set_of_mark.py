"""Set-of-Mark overlay renderer.

Given a screenshot and a list of bounding boxes (from UIA, AT-SPI, AX, or
pure-CV detection), draw a numbered overlay so a vision LLM can refer to
each element by integer mark ID instead of pixel coordinates.

This is the exact pattern used in the Set-of-Mark paper (Yang et al.,
2023) and adopted by Claude Computer Use / OpenAI computer-use-preview.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "Mark",
    "SetOfMarkResult",
    "render_set_of_mark",
]


@dataclass(frozen=True)
class Mark:
    """A single numbered mark to draw on the screenshot."""

    mark_id: int
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    role: str = ""
    source: str = ""  # "uia", "atspi", "ax", "cv", "ocr"
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True)
class SetOfMarkResult:
    """Output of :func:`render_set_of_mark`."""

    annotated_png: bytes
    marks: tuple[Mark, ...]
    width: int
    height: int

    def mark_table(self) -> list[dict[str, object]]:
        """Compact per-mark table suitable for a frontier prompt."""
        return [
            {
                "id": m.mark_id,
                "label": m.label,
                "role": m.role,
                "source": m.source,
                "x": m.x,
                "y": m.y,
                "w": m.width,
                "h": m.height,
                "center_x": m.center[0],
                "center_y": m.center[1],
            }
            for m in self.marks
        ]


# Distinct, high-contrast hues chosen so adjacent marks are easy to tell apart
# even for users with deuteranopia (validated against the IBM Colorblind
# Accessibility palette).
_COLOR_CYCLE: tuple[tuple[int, int, int], ...] = (
    (220, 20, 60),    # crimson
    (30, 144, 255),   # dodger blue
    (255, 140, 0),    # dark orange
    (50, 205, 50),    # lime
    (148, 0, 211),    # dark violet
    (0, 191, 255),    # deep sky blue
    (255, 215, 0),    # gold
    (255, 20, 147),   # deep pink
    (46, 139, 87),    # sea green
    (139, 69, 19),    # saddle brown
)


def _color_for(mark_id: int) -> tuple[int, int, int]:
    return _COLOR_CYCLE[mark_id % len(_COLOR_CYCLE)]


def render_set_of_mark(
    image_png: bytes,
    elements: Iterable[Mark],
    *,
    max_marks: int = 64,
    box_width: int = 3,
    font_size: int = 18,
) -> SetOfMarkResult:
    """Overlay numbered marks onto *image_png* and return PNG bytes.

    Falls back gracefully when Pillow is not installed — the returned
    PNG is the original image, but the structured mark table is still
    produced so downstream consumers can use coordinate-based actions.
    """
    marks = tuple(sorted(elements, key=lambda m: m.mark_id))[:max_marks]
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except Exception:
        # No Pillow — return the unannotated image plus the mark table.
        # Vision LLM can still get value from the table (text-mode SoM).
        return SetOfMarkResult(
            annotated_png=image_png,
            marks=marks,
            width=0,
            height=0,
        )

    img = Image.open(io.BytesIO(image_png)).convert("RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for mark in marks:
        color = _color_for(mark.mark_id)
        x0, y0 = mark.x, mark.y
        x1 = mark.x + mark.width
        y1 = mark.y + mark.height
        # Bounding rectangle
        draw.rectangle(
            (x0, y0, x1, y1),
            outline=color + (255,),
            width=box_width,
        )
        # Translucent inside fill so overlap stays readable
        draw.rectangle(
            (x0, y0, x1, y1),
            fill=color + (32,),
        )
        # Number badge (top-left corner with high-contrast background)
        badge_text = str(mark.mark_id)
        if font is not None:
            try:
                bbox = draw.textbbox((0, 0), badge_text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = 12 * len(badge_text), 16
        else:
            tw, th = 12 * len(badge_text), 16
        pad = 4
        bx0, by0 = x0, max(0, y0 - th - 2 * pad)
        bx1, by1 = bx0 + tw + 2 * pad, by0 + th + 2 * pad
        draw.rectangle(
            (bx0, by0, bx1, by1),
            fill=color + (235,),
            outline=(0, 0, 0, 255),
            width=1,
        )
        if font is not None:
            draw.text(
                (bx0 + pad, by0 + pad),
                badge_text,
                fill=(255, 255, 255, 255),
                font=font,
            )
        else:
            draw.text(
                (bx0 + pad, by0 + pad),
                badge_text,
                fill=(255, 255, 255, 255),
            )

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=False)
    return SetOfMarkResult(
        annotated_png=out.getvalue(),
        marks=marks,
        width=img.width,
        height=img.height,
    )
