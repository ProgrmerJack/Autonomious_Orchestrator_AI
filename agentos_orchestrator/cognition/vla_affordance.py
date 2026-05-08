"""Vision-Language-Action (VLA) affordance grounding.

Trains and uses models that process raw pixels to emit direct
mouse/keyboard actions without relying on accessibility trees or DOMs.
Supports zero-shot interaction with custom rendering engines
(Flutter, Canvas, remote desktop streams).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentos_orchestrator.config import gemini_vision_model
from agentos_orchestrator.os_control.base import UiAction


@dataclass(slots=True)
class PixelRegion:
    """A region of interest in a screenshot with visual semantics."""

    x: int
    y: int
    width: int
    height: int
    affordance_type: str  # e.g. "button", "text_field", "menu", "icon"
    confidence: float
    visual_cues: list[str] = field(default_factory=list)
    ocr_text: str = ""


@dataclass(slots=True)
class VLAActionSpace:
    """Normalized action space emitted by a VLA controller."""

    action_type: (
        str  # click, double_click, right_click, drag, type, hotkey, scroll, hover
    )
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    text: str | None = None
    key_combo: str | None = None
    scroll_delta: int | None = None
    rationale: str = ""

    def to_ui_action(self, selector_hint: str = "") -> UiAction:
        """Convert VLA action to UiAction for backend compatibility."""
        if self.action_type in {"click", "double_click", "right_click", "drag"}:
            meta = {
                "x": self.x,
                "y": self.y,
                "x2": self.x2,
                "y2": self.y2,
                "action_type": self.action_type,
            }
            return UiAction(
                action_type="click",
                selector=f"pixel=({self.x},{self.y})",
                value=json.dumps(meta),
                metadata=meta,
            )
        if self.action_type == "type":
            return UiAction(
                action_type="type",
                selector=selector_hint or "pixel-focus",
                value=self.text or "",
                metadata={"x": self.x, "y": self.y},
            )
        if self.action_type == "hotkey":
            return UiAction(
                action_type="hotkey",
                selector="app-window",
                value=self.key_combo or "",
                metadata={},
            )
        if self.action_type == "scroll":
            return UiAction(
                action_type="scroll",
                selector="pixel-wheel",
                value=str(self.scroll_delta or 0),
                metadata={"x": self.x, "y": self.y, "delta": self.scroll_delta},
            )
        return UiAction(
            action_type=self.action_type,
            selector=selector_hint or "pixel-generic",
            value=self.text or self.key_combo or "",
            metadata={"rationale": self.rationale},
        )


class ScreenshotProvider(Protocol):
    """Protocol for backends that can capture raw pixel buffers."""

    def capture(self) -> bytes:
        """Return a PNG-encoded screenshot as raw bytes."""


class VLAAffordanceGrounding:
    """Pixel-level affordance detector and action proposer.

    When a multimodal model is available (Gemini with vision, or a local
    VLA checkpoint), it processes screenshots directly and emits coordinate
    actions.  In test or fallback mode it uses lightweight heuristics on
    encoded screenshots.
    """

    _MAX_IMAGE_MB = 4

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or gemini_vision_model()
        self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )

    def detect_affordances(
        self,
        screenshot_bytes: bytes,
        objective: str,
    ) -> list[PixelRegion]:
        """Return a list of actionable regions inferred from the image."""
        if self._api_key:
            return self._model_detect_affordances(screenshot_bytes, objective)
        return self._heuristic_detect_affordances(screenshot_bytes, objective)

    def propose_action(
        self,
        screenshot_bytes: bytes,
        objective: str,
        history: list[dict[str, Any]],
    ) -> VLAActionSpace | None:
        """Propose the best next action given the current visual state."""
        if self._api_key:
            return self._model_propose_action(screenshot_bytes, objective, history)
        regions = self._heuristic_detect_affordances(screenshot_bytes, objective)
        if not regions:
            return None
        best = max(regions, key=lambda r: r.confidence)
        return VLAActionSpace(
            action_type="click",
            x=best.x + best.width // 2,
            y=best.y + best.height // 2,
            rationale=f"Heuristic: highest-confidence {best.affordance_type} region",
        )

    # ------------------------------------------------------------------ #
    # Model-backed path (Gemini Vision)
    # ------------------------------------------------------------------ #

    def _model_detect_affordances(
        self,
        screenshot_bytes: bytes,
        objective: str,
    ) -> list[PixelRegion]:
        prompt = (
            "You are a UI affordance detector. Analyze the screenshot and list "
            "all interactive elements (buttons, text fields, menus, icons, links). "
            "For each element, output JSON with: x, y, width, height, "
            "affordance_type, confidence (0.0-1.0), visual_cues (list of strings). "
            "Return as a JSON array."
        )
        payload = self._call_gemini_vision(prompt, screenshot_bytes)
        return self._parse_affordance_payload(payload)

    def _model_propose_action(
        self,
        screenshot_bytes: bytes,
        objective: str,
        history: list[dict[str, Any]],
    ) -> VLAActionSpace | None:
        recent = json.dumps(history[-6:], indent=2)
        prompt = (
            "You are a desktop automation agent. You see the current screen. "
            f"Your objective: {objective}\n"
            f"Recent actions: {recent}\n"
            "Choose the SINGLE best next action. Output JSON with keys: "
            "action_type, x, y, x2, y2, text, key_combo, scroll_delta, rationale. "
            "Coordinates must be integer pixel positions in the screenshot. "
            "action_type must be one of: click, double_click, right_click, "
            "drag, type, hotkey, scroll, hover."
        )
        payload = self._call_gemini_vision(prompt, screenshot_bytes)
        return self._parse_action_payload(payload)

    def _call_gemini_vision(
        self,
        prompt: str,
        screenshot_bytes: bytes,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("No Gemini API key configured for vision")
        b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={urllib.parse.quote(self._api_key)}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _parse_affordance_payload(payload: dict[str, Any]) -> list[PixelRegion]:
        candidates = payload.get("candidates") or []
        if not candidates:
            return []
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return []
        text = str(parts[0].get("text") or "").strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(items, list):
            return []
        regions: list[PixelRegion] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            regions.append(
                PixelRegion(
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    width=int(item.get("width", 0)),
                    height=int(item.get("height", 0)),
                    affordance_type=str(item.get("affordance_type", "unknown")),
                    confidence=float(item.get("confidence", 0.0)),
                    visual_cues=item.get("visual_cues") or [],
                    ocr_text=str(item.get("ocr_text", "")),
                )
            )
        return regions

    @staticmethod
    def _parse_action_payload(payload: dict[str, Any]) -> VLAActionSpace | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        text = str(parts[0].get("text") or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return VLAActionSpace(
            action_type=str(parsed.get("action_type", "click")),
            x=parsed.get("x"),
            y=parsed.get("y"),
            x2=parsed.get("x2"),
            y2=parsed.get("y2"),
            text=parsed.get("text"),
            key_combo=parsed.get("key_combo"),
            scroll_delta=parsed.get("scroll_delta"),
            rationale=str(parsed.get("rationale", "")),
        )

    # ------------------------------------------------------------------ #
    # Heuristic fallback (no API key)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _heuristic_detect_affordances(
        screenshot_bytes: bytes,
        objective: str,
    ) -> list[PixelRegion]:
        """Lightweight fallback that uses image dimensions and keyword matching.

        In a production deployment this would be replaced by a local ONNX
        or PyTorch VLA checkpoint (e.g. OpenCUA, CogAgent, or ShowUI).
        """
        try:
            from PIL import Image
        except ImportError:
            return []
        try:
            img = Image.open(io.BytesIO(screenshot_bytes))
        except Exception:
            return []
        width, height = img.size
        regions: list[PixelRegion] = []
        # Grid-based scan simulating rough visual segmentation
        grid_cols, grid_rows = 4, 3
        cell_w, cell_h = width // grid_cols, height // grid_rows
        lower_obj = objective.lower()
        for row in range(grid_rows):
            for col in range(grid_cols):
                cx = col * cell_w + cell_w // 2
                cy = row * cell_h + cell_h // 2
                affordance = "surface"
                conf = 0.3
                cues: list[str] = []
                # Top-left often has menus/title bars
                if row == 0 and col == 0:
                    affordance = "menu"
                    conf = 0.5
                    cues.append("top-left_region")
                # Center often has main content
                if row == 1 and col in {1, 2}:
                    affordance = "content"
                    conf = 0.6
                    cues.append("central_region")
                # Bottom row often has buttons/status
                if row == grid_rows - 1:
                    affordance = "button_bar"
                    conf = 0.5
                    cues.append("bottom_region")
                # Objective-keyword boost
                if "draw" in lower_obj or "paint" in lower_obj:
                    if row == 1:
                        affordance = "canvas"
                        conf = 0.7
                        cues.append("drawing_context")
                regions.append(
                    PixelRegion(
                        x=cx - cell_w // 4,
                        y=cy - cell_h // 4,
                        width=cell_w // 2,
                        height=cell_h // 2,
                        affordance_type=affordance,
                        confidence=conf,
                        visual_cues=cues,
                    )
                )
        return sorted(regions, key=lambda r: r.confidence, reverse=True)

    def _resize_if_needed(self, screenshot_bytes: bytes) -> bytes:
        """Resize screenshot to stay under API size limits."""
        try:
            from PIL import Image
        except ImportError:
            return screenshot_bytes
        try:
            img = Image.open(io.BytesIO(screenshot_bytes))
        except Exception:
            return screenshot_bytes
        max_pixels = self._MAX_IMAGE_MB * 1024 * 1024
        current = len(screenshot_bytes)
        if current <= max_pixels:
            return screenshot_bytes
        ratio = (max_pixels / current) ** 0.5
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
