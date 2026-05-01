"""VLM Adapter — Vision-Language Model abstraction layer.

Phase 1: Upgrading the "Eyes" and the "Brain".

This module defines a clean protocol (VLMAdapter) that all downstream
cognition code calls.  Right now the default implementation is a
ClassicalVLMAdapter that uses the existing local_vla.py + adaptive_perception.py
pipeline.  When you are ready to swap in a real VLM (Qwen-VL-2B locally, or
GPT-4o via API), you implement the same three methods and register the new
backend — zero changes to planning or memory code.

Hierarchy
─────────
    VLMAdapter (abstract protocol)
        ├─ ClassicalVLMAdapter      ← today: Random Forest + classical CV
        ├─ QwenVLAdapter            ← next:  local Qwen-VL-2B via transformers
        └─ GPT4VisionAdapter        ← later: cloud GPT-4o / Claude-3.5 Sonnet

The key VLM outputs this architecture needs:
1. scene_description(screenshot) → str
      "A dark-themed code editor with an open Python file and a Run button in
       the top-right corner."
2. locate_elements(screenshot, query) → list[VLMElement]
      Returns bounding boxes WITH semantic labels, not just pixel coords.
3. extract_text(screenshot, region) → str
      Full OCR output for a bounding box — better than our Otsu-OCR for
      complex font rendering (anti-aliased, emoji, non-Latin scripts).

These three primitives replace:
    local_vla.py        →  locate_elements + scene_description
    adaptive_perception →  extract_text + locate_elements
"""

from __future__ import annotations

import abc
import io
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from .local_vla import LocalFastVLA, DetectedElement
    from .adaptive_perception import AdaptivePerceptionEngine, PerceivedElement


# ─────────────────────────────────────────────────────────────────────────── #
# Data structures shared by all adapters                                      #
# ─────────────────────────────────────────────────────────────────────────── #


@dataclass
class VLMElement:
    """A UI element as perceived by the VLM."""

    x: int
    y: int
    width: int
    height: int
    # Semantic label from the VLM — NOT hallucinated, grounded in pixels
    label: str
    # Element category: button | text_field | checkbox | icon | text_block |
    #                   dropdown | slider | image | canvas | unknown
    element_type: str
    # Confidence in [0, 1] — higher = VLM is more certain
    confidence: float = 0.5
    # Natural-language description of the element's purpose/affordance
    affordance: str = ""
    # Raw OCR text if available
    ocr_text: str = ""

    @property
    def cx(self) -> int:
        return self.x + self.width // 2

    @property
    def cy(self) -> int:
        return self.y + self.height // 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.width,
            "h": self.height,
            "label": self.label,
            "type": self.element_type,
            "confidence": self.confidence,
            "affordance": self.affordance,
        }


@dataclass
class SceneUnderstanding:
    """Rich semantic description of a UI screenshot."""

    # Top-level natural-language description
    description: str = ""
    # Inferred application type
    app_type: str = (
        "unknown"  # browser | code_editor | terminal | media | office | unknown
    )
    # Detected UI theme
    theme: str = "unknown"  # light | dark | high_contrast | unknown
    # Semantic elements
    elements: list[VLMElement] = field(default_factory=list)
    # Confidence in description [0, 1]
    confidence: float = 0.0
    # How long the VLM call took
    latency_ms: float = 0.0
    # Which adapter produced this
    adapter_name: str = "unknown"

    @property
    def has_modal(self) -> bool:
        return any(
            e.element_type in {"modal", "dialog"} or "dialog" in e.label.lower()
            for e in self.elements
        )

    @property
    def interactive_elements(self) -> list[VLMElement]:
        return [
            e
            for e in self.elements
            if e.element_type
            in {"button", "text_field", "checkbox", "dropdown", "slider"}
        ]


# ─────────────────────────────────────────────────────────────────────────── #
# Abstract protocol                                                           #
# ─────────────────────────────────────────────────────────────────────────── #


class VLMAdapter(abc.ABC):
    """Protocol all VLM backends must implement.

    All methods accept a PIL Image and return structured data.
    Implementations must be thread-safe (they may be called from multiple
    threads if the agent uses async perception).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifier for logging / debugging."""

    @abc.abstractmethod
    def understand_scene(
        self,
        screenshot: Image.Image,
        objective: str = "",
    ) -> SceneUnderstanding:
        """Return a full semantic scene understanding.

        This is the primary method.  Adapters should return as much detail
        as they can — the calling code will degrade gracefully on missing fields.
        """

    @abc.abstractmethod
    def locate_elements(
        self,
        screenshot: Image.Image,
        query: str,
    ) -> list[VLMElement]:
        """Return elements relevant to `query`, sorted by relevance.

        Examples:
            query = "submit button"        → list with submit button first
            query = "price input field"    → list with price fields first
            query = "login / sign in"      → list with login button first
        """

    @abc.abstractmethod
    def extract_text(
        self,
        screenshot: Image.Image,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """OCR the screenshot (or a crop of it) and return plain text."""


# ─────────────────────────────────────────────────────────────────────────── #
# Default: classical CV adapter (wraps existing local_vla + perception)      #
# ─────────────────────────────────────────────────────────────────────────── #


class ClassicalVLMAdapter(VLMAdapter):
    """Thin wrapper around LocalFastVLA + AdaptivePerceptionEngine.

    This is the 'today' implementation — it uses Random Forests, Otsu-OCR,
    and semantic embedding scoring.  No GPU required.  Swap it for
    QwenVLAdapter when you have the hardware / API key.
    """

    def __init__(self) -> None:
        # Lazy imports to avoid circular dependency at module load
        from .local_vla import LocalFastVLA
        from .adaptive_perception import AdaptivePerceptionEngine
        from .semantic_memory import SemanticEmbedder

        self._vla = LocalFastVLA()
        self._perception = AdaptivePerceptionEngine(
            semantic_embedder=SemanticEmbedder(n_components=64)
        )

    @property
    def name(self) -> str:
        return "classical_cv"

    def understand_scene(
        self,
        screenshot: Image.Image,
        objective: str = "",
    ) -> SceneUnderstanding:
        t0 = time.perf_counter()

        # perception engine expects bytes
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        screenshot_bytes = buf.getvalue()

        # Use perception engine for element detection
        elements_raw = self._perception.perceive(screenshot_bytes, objective, [])
        # Convert to VLMElement
        vlm_elements = [
            VLMElement(
                x=e.x,
                y=e.y,
                width=e.width,
                height=e.height,
                label=e.text or e.element_type,
                element_type=e.element_type,
                confidence=e.confidence,
                affordance=e.element_type,
                ocr_text=e.text,
            )
            for e in elements_raw
        ]

        # Infer theme from mean brightness
        import numpy as np

        arr = np.array(screenshot.convert("L"), dtype=np.float32)
        theme = "dark" if arr.mean() < 100 else "light"

        # Rough app type from element counts
        type_counts: dict[str, int] = {}
        for e in vlm_elements:
            type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1
        if type_counts.get("text_field", 0) > 2:
            app_type = "office"
        elif type_counts.get("button", 0) > 3:
            app_type = "browser"
        else:
            app_type = "unknown"

        description = (
            f"{theme.capitalize()} UI ({app_type}), "
            f"{len(vlm_elements)} elements detected"
            + (f" for objective: {objective}" if objective else "")
        )

        return SceneUnderstanding(
            description=description,
            app_type=app_type,
            theme=theme,
            elements=vlm_elements,
            confidence=0.55,
            latency_ms=(time.perf_counter() - t0) * 1000,
            adapter_name=self.name,
        )

    def locate_elements(
        self,
        screenshot: Image.Image,
        query: str,
    ) -> list[VLMElement]:
        elements = self.understand_scene(screenshot, query).elements
        # Sort by confidence descending (already semantic-scored in perception)
        return sorted(elements, key=lambda e: e.confidence, reverse=True)

    def extract_text(
        self,
        screenshot: Image.Image,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        if region:
            screenshot = screenshot.crop(region)
        text = self._perception._simple_ocr(screenshot)
        return text if text else ""


# ─────────────────────────────────────────────────────────────────────────── #
# Stub: Qwen-VL local adapter (real model, activate when hardware is ready)  #
# ─────────────────────────────────────────────────────────────────────────── #


class QwenVLAdapter(VLMAdapter):
    """7B/2B Qwen-VL running locally via HuggingFace Transformers.

    Requirements:
        pip install transformers accelerate bitsandbytes pillow
        ~8 GB VRAM for 2B int4, ~16 GB for 7B int8

    This adapter is NOT instantiated until you call QwenVLAdapter(model_id=...)
    explicitly.  The weight download happens on first construction.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-VL-Chat",
        device: str = "cuda",
        load_in_4bit: bool = True,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for QwenVLAdapter: pip install transformers accelerate"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self._load_in_4bit:
            kwargs["load_in_4bit"] = True
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
        self._model.eval()
        self._loaded = True

    @property
    def name(self) -> str:
        return f"qwen_vl:{self._model_id}"

    def understand_scene(
        self,
        screenshot: Image.Image,
        objective: str = "",
    ) -> SceneUnderstanding:
        self._lazy_load()
        t0 = time.perf_counter()

        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            screenshot.save(tmp.name)
            tmp_path = tmp.name

        try:
            prompt = (
                f"Describe this UI screenshot in detail. "
                f"List all interactive elements (buttons, inputs, checkboxes) "
                f"with their approximate positions."
                + (f" The user wants to: {objective}" if objective else "")
            )
            query = self._tokenizer.from_list_format(
                [{"image": tmp_path}, {"text": prompt}]
            )
            response, _ = self._model.chat(self._tokenizer, query=query, history=None)
        finally:
            os.unlink(tmp_path)

        return SceneUnderstanding(
            description=response,
            confidence=0.85,
            latency_ms=(time.perf_counter() - t0) * 1000,
            adapter_name=self.name,
        )

    def locate_elements(
        self,
        screenshot: Image.Image,
        query: str,
    ) -> list[VLMElement]:
        # For now delegate to scene understanding + filter
        scene = self.understand_scene(screenshot, query)
        return scene.elements

    def extract_text(
        self,
        screenshot: Image.Image,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        self._lazy_load()
        if region:
            screenshot = screenshot.crop(region)

        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            screenshot.save(tmp.name)
            tmp_path = tmp.name

        try:
            query = self._tokenizer.from_list_format(
                [
                    {"image": tmp_path},
                    {"text": "Extract all text from this image verbatim."},
                ]
            )
            response, _ = self._model.chat(self._tokenizer, query=query, history=None)
        finally:
            os.unlink(tmp_path)

        return response


# ─────────────────────────────────────────────────────────────────────────── #
# Factory / registry                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

_ADAPTER_REGISTRY: dict[str, type[VLMAdapter]] = {
    "classical": ClassicalVLMAdapter,
    "qwen_vl": QwenVLAdapter,
}


def register_adapter(name: str, cls: type[VLMAdapter]) -> None:
    """Register a custom VLM adapter under `name`."""
    _ADAPTER_REGISTRY[name] = cls


def create_adapter(name: str = "classical", **kwargs: Any) -> VLMAdapter:
    """Instantiate a VLM adapter by name.

    Examples
    ────────
        adapter = create_adapter()                        # classical CV
        adapter = create_adapter("qwen_vl", load_in_4bit=True)
        adapter = create_adapter("my_custom_adapter")
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY)
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)
