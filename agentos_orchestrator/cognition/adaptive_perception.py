"""Adaptive Perception Engine for Robust UI Understanding.

Addresses the "Classical CV vs Modern UI Clash" by providing:
1. Multi-scale, contrast-adaptive element detection (handles dark mode, flat UIs)
2. Lightweight local OCR for reading button labels, text fields
3. Semantic text matching to understand what elements mean
4. UI mode detection (dark/light/high-contrast) to adapt thresholds

Key insight: Instead of one fixed threshold, we adapt dynamically based on
local contrast statistics and detected UI theme.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .semantic_memory import SemanticEmbedder


@dataclass(slots=True)
class PerceivedElement:
    """A UI element detected and understood by the perception engine."""

    x: int
    y: int
    width: int
    height: int
    element_type: str  # button, text_field, checkbox, icon, text_block, unknown
    text: str = ""
    confidence: float = 0.0
    # Semantic score: how well this element matches current objective
    semantic_score: float = 0.0
    # Visual salience: how much it stands out from background
    salience: float = 0.0


@dataclass
class UIMode:
    """Detected UI theme and contrast characteristics."""

    is_dark_mode: bool
    avg_brightness: float
    contrast_ratio: float
    dominant_hue: float
    edge_density: float


class AdaptivePerceptionEngine:
    """Robust pixel-based UI perception with adaptive thresholds and local OCR.

    Solves the classical-CV/modern-UI clash by:
    - Detecting UI theme (dark/light) and adjusting thresholds
    - Using multi-scale analysis (small icons vs large panels)
    - Applying lightweight OCR to read element text
    - Scoring elements semantically against the task objective
    """

    def __init__(
        self,
        semantic_embedder: SemanticEmbedder | None = None,
        min_element_size: int = 8,
        max_element_size_ratio: float = 0.6,
    ) -> None:
        self.embedder = semantic_embedder or SemanticEmbedder(n_components=64)
        self.min_element_size = min_element_size
        self.max_element_size_ratio = max_element_size_ratio
        # Cached UI mode from last frame
        self.last_ui_mode: UIMode | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def perceive(
        self,
        screenshot_bytes: bytes,
        objective: str,
        history: list[dict[str, Any]],
    ) -> list[PerceivedElement]:
        """Full perception pipeline: detect → OCR → score → rank.

        Returns elements ranked by relevance to the objective.
        """
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        arr = np.array(img)

        # 1. Detect UI theme
        ui_mode = self._detect_ui_mode(arr)
        self.last_ui_mode = ui_mode

        # 2. Multi-scale element detection
        raw_elements = self._detect_elements_adaptive(arr, ui_mode)

        # 3. OCR on each element to read text
        elements_with_text = self._ocr_elements(img, raw_elements)

        # 4. Classify element type + semantic scoring
        scored_elements = self._classify_and_score(
            elements_with_text,
            objective,
            history,
            ui_mode,
        )

        # 5. Non-max suppression + sort by relevance
        scored_elements = self._non_max_suppression(scored_elements)
        scored_elements.sort(key=lambda e: e.semantic_score, reverse=True)

        return scored_elements

    def quick_detect(self, screenshot_bytes: bytes) -> list[PerceivedElement]:
        """Fast detection without OCR/semantic scoring (<20ms).

        Used when latency is critical (e.g. real-time tracking).
        """
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        arr = np.array(img)
        ui_mode = self._detect_ui_mode(arr)
        raw = self._detect_elements_adaptive(arr, ui_mode)
        return [
            PerceivedElement(
                x=e["x"],
                y=e["y"],
                width=e["w"],
                height=e["h"],
                element_type="unknown",
                text="",
                confidence=e["conf"],
                semantic_score=0.0,
                salience=e["salience"],
            )
            for e in raw
        ]

    # ------------------------------------------------------------------ #
    # UI Mode Detection
    # ------------------------------------------------------------------ #

    def _detect_ui_mode(self, arr: np.ndarray) -> UIMode:
        """Analyze global image statistics to detect dark mode, contrast, etc."""
        gray = np.mean(arr, axis=2)
        avg_brightness = float(np.mean(gray)) / 255.0
        is_dark = avg_brightness < 0.45

        # Contrast ratio (max - min)
        contrast = float(np.max(gray) - np.min(gray)) / 255.0

        # Dominant hue
        hsv = self._rgb_to_hsv(arr)
        dominant_hue = float(np.median(hsv[:, :, 0]))

        # Edge density
        from scipy import ndimage

        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(dx**2 + dy**2)
        edge_density = float(np.mean(edge_mag > 20.0))

        return UIMode(
            is_dark_mode=is_dark,
            avg_brightness=avg_brightness,
            contrast_ratio=contrast,
            dominant_hue=dominant_hue,
            edge_density=edge_density,
        )

    # ------------------------------------------------------------------ #
    # Multi-Scale Adaptive Element Detection
    # ------------------------------------------------------------------ #

    def _detect_elements_adaptive(
        self,
        arr: np.ndarray,
        ui_mode: UIMode,
    ) -> list[dict[str, Any]]:
        """Find UI elements using thresholds adapted to UI theme."""
        gray = np.mean(arr, axis=2)

        # Adaptive threshold based on brightness
        if ui_mode.is_dark_mode:
            # In dark mode, elements are often lighter than background
            base_thresh = np.percentile(gray, 75)
        else:
            # In light mode, elements are often darker
            base_thresh = np.percentile(gray, 25)

        elements: list[dict[str, Any]] = []

        # Scale 1: Fine detail (small buttons, icons)
        fine = self._detect_at_scale(arr, gray, base_thresh, ui_mode, scale=1.0)
        elements.extend(fine)

        # Scale 2: Medium panels, text fields
        medium = self._detect_at_scale(arr, gray, base_thresh, ui_mode, scale=2.0)
        elements.extend(medium)

        # Scale 3: Large containers, dialogs
        large = self._detect_at_scale(arr, gray, base_thresh, ui_mode, scale=4.0)
        elements.extend(large)

        return elements

    def _detect_at_scale(
        self,
        arr: np.ndarray,
        gray: np.ndarray,
        base_thresh: float,
        ui_mode: UIMode,
        scale: float,
    ) -> list[dict[str, Any]]:
        """Element detection at a specific Gaussian blur scale."""
        from scipy import ndimage

        # Blur to suppress fine texture at this scale
        sigma = scale
        blurred = ndimage.gaussian_filter(gray, sigma=sigma)

        # Fast local contrast using uniform filter (much faster than generic_filter)
        size = max(3, int(8 * scale))
        local_mean = ndimage.uniform_filter(blurred, size=size)
        # Approximate std using mean(abs(x - mean))
        local_diff = ndimage.uniform_filter(np.abs(blurred - local_mean), size=size)

        # Adaptive threshold: regions that stand out from local mean
        if ui_mode.is_dark_mode:
            salient = blurred > local_mean + local_diff * 0.8
        else:
            salient = blurred < local_mean - local_diff * 0.8

        # Label connected components
        labels, num = ndimage.label(salient)
        if num == 0:
            return []

        h, w = arr.shape[:2]
        elements: list[dict[str, Any]] = []
        for i in range(1, num + 1):
            mask = labels == i
            ys, xs = np.where(mask)
            if len(xs) < 25:  # Too small
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            ew, eh = x1 - x0, y1 - y0
            if ew < self.min_element_size or eh < self.min_element_size:
                continue
            if (
                ew > w * self.max_element_size_ratio
                or eh > h * self.max_element_size_ratio
            ):
                continue

            # Salience = how much this region deviates from surroundings
            region_mean = float(np.mean(gray[mask]))
            bg = gray[~mask]
            bg_mean = float(np.mean(bg)) if bg.size > 0 else region_mean
            salience = abs(region_mean - bg_mean) / 255.0

            elements.append(
                {
                    "x": x0,
                    "y": y0,
                    "w": ew,
                    "h": eh,
                    "conf": salience,
                    "salience": salience,
                    "scale": scale,
                }
            )

        return elements

    # ------------------------------------------------------------------ #
    # Lightweight Local OCR
    # ------------------------------------------------------------------ #

    def _ocr_elements(
        self,
        img: Image.Image,
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract text from each element region using simple heuristics.

        Full OCR (tesseract) is too heavy. We use:
        - Detect text-like regions by aspect ratio and edge density
        - Simple connected-component analysis for character blobs
        - Template matching for common UI text patterns
        """
        for elem in elements:
            x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]
            # Skip very small or very large regions
            if w < 20 or h < 10 or w > img.width * 0.5 or h > img.height * 0.3:
                elem["text"] = ""
                continue

            # Crop the region
            region = img.crop((x, y, x + w, y + h))
            text = self._simple_ocr(region)
            elem["text"] = text

        return elements

    def _simple_ocr(self, region: Image.Image) -> str:
        """Lightweight text detection on a cropped region.

        Uses horizontal projection profiles to detect text lines,
        then vertical profiles to detect character spacing.
        Returns empty string if no text detected.

        Threshold: Otsu's method (optimal bimodal split) so dark-mode UIs
        (light text on dark BG) and flat light UIs both binarize correctly.
        """
        gray = ImageOps.grayscale(region)
        arr = np.array(gray)

        # Otsu's threshold: maximize inter-class variance between BG and FG
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        hist_f = hist.astype(np.float64) / max(hist.sum(), 1)
        cum = np.cumsum(hist_f)
        cum_mean = np.cumsum(hist_f * np.arange(256, dtype=np.float64))
        total_mean = cum_mean[-1]
        var = np.zeros(256, dtype=np.float64)
        for t in range(1, 255):
            w0, w1 = cum[t], 1.0 - cum[t]
            if w0 < 1e-9 or w1 < 1e-9:
                continue
            m0 = cum_mean[t] / w0
            m1 = (total_mean - cum_mean[t]) / w1
            var[t] = w0 * w1 * (m0 - m1) ** 2
        otsu_thresh = int(np.argmax(var))

        # Dark-mode polarity: if average brightness < threshold, BG is dark
        # → text is LIGHTER than background; invert the mask direction.
        avg_brightness = float(np.mean(arr))
        if avg_brightness < otsu_thresh:
            binary = arr > otsu_thresh  # Light text on dark background
        else:
            binary = arr < otsu_thresh  # Dark text on light background

        # Horizontal projection: sum of dark pixels per row
        h_proj = np.sum(binary, axis=1)
        # Find text lines (consecutive rows with significant dark pixels)
        text_rows = h_proj > (np.max(h_proj) * 0.2)

        if not np.any(text_rows):
            return ""

        # Check for character-like structure: periodic vertical variation
        v_proj = np.sum(binary, axis=0)
        # Text has alternating high/low in vertical projection
        if len(v_proj) < 10:
            return ""

        # Compute local variance as proxy for "character-ness"
        window = max(3, len(v_proj) // 20)
        local_var = np.array(
            [
                np.var(v_proj[max(0, i - window) : min(len(v_proj), i + window)])
                for i in range(len(v_proj))
            ]
        )
        char_score = np.mean(local_var) / max(np.var(v_proj), 1)

        # Aspect ratio check: text regions are wide and short
        aspect = region.width / max(region.height, 1)
        is_text_like = (
            aspect > 1.5 and char_score > 0.1 and region.height < region.width * 0.5
        )

        if not is_text_like:
            return ""

        # Try to recognize common patterns via pixel statistics
        return self._pattern_guess(region, binary)

    def _pattern_guess(self, region: Image.Image, binary: np.ndarray) -> str:
        """Return a structural description of text content based on pixel geometry.

        We deliberately avoid hallucinating specific button labels like "OK" or
        "Cancel" from character counts alone — that caused false semantic matches.
        Instead we return honest structural tags:
          "text:N"   → N character glyphs detected (used by semantic embedder)
          "label"    → short single-word label geometry
          "input"    → wide, shallow text-field geometry
          "icon"     → near-square, small
        The semantic scorer treats these structurally, not as literal text.
        """
        from scipy import ndimage

        labels, num = ndimage.label(binary)
        if num == 0:
            return ""

        sizes = ndimage.sum(binary, labels, range(1, num + 1))
        valid = sizes[sizes > 5]
        if len(valid) == 0:
            return ""

        n_chars = len(valid)
        w = region.width
        h = region.height

        # Near-square small region → icon/checkbox, not text
        if w < 30 and h < 30 and abs(w - h) < 5:
            return "icon"

        # Very wide, shallow geometry → input field
        if w > 150 and h < 40 and w / max(h, 1) > 4.0:
            return "input"

        # Text blob: encode glyph count so semantic scorer can reason about length
        # (short labels = actions; long labels = descriptions)
        if n_chars >= 1:
            return f"text:{n_chars}"

        return ""

    # ------------------------------------------------------------------ #
    # Semantic Classification and Scoring
    # ------------------------------------------------------------------ #

    def _classify_and_score(
        self,
        elements: list[dict[str, Any]],
        objective: str,
        history: list[dict[str, Any]],
        ui_mode: UIMode,
    ) -> list[PerceivedElement]:
        """Classify each element and score its relevance to the objective."""
        # Embed the objective once
        obj_emb = self.embedder.embed(objective)

        perceived: list[PerceivedElement] = []
        for elem in elements:
            text = elem.get("text", "")
            w, h = elem["w"], elem["h"]
            aspect = w / max(h, 1)

            # Classify by geometry + text
            element_type = self._classify_element_type(text, aspect, w, h, ui_mode)

            # Semantic score: how well does text match objective?
            semantic_score = 0.0
            if text:
                text_emb = self.embedder.embed(text)
                semantic_score = float(np.dot(obj_emb, text_emb))

            # Boost score for elements that match known action patterns
            semantic_score += self._action_pattern_boost(text, objective)

            perceived.append(
                PerceivedElement(
                    x=elem["x"],
                    y=elem["y"],
                    width=w,
                    height=h,
                    element_type=element_type,
                    text=text,
                    confidence=elem["conf"],
                    semantic_score=semantic_score,
                    salience=elem["salience"],
                )
            )

        return perceived

    @staticmethod
    def _classify_element_type(
        text: str,
        aspect: float,
        w: int,
        h: int,
        ui_mode: UIMode,
    ) -> str:
        """Classify element by geometry and OCR structural tag.

        The 'text' field now contains structural tags from _pattern_guess:
          "text:N" → N character glyphs
          "input"  → text-field geometry
          "icon"   → near-square small region
        We classify based on geometry + tag, never on hallucinated button labels.
        """
        if text == "input" or (aspect > 4 and h < 40 and w > 100):
            return "text_field"
        if text == "icon" or (w < 30 and h < 30 and abs(w - h) < 5):
            return "checkbox"
        if text.startswith("text:"):
            try:
                n = int(text.split(":")[1])
            except (IndexError, ValueError):
                n = 0
            if n <= 3 and w < 80 and h < 35:
                return "button"       # Short label, button-sized
            if n <= 10 and w < 200 and h < 40:
                return "button"       # Medium label, still button
            return "text_block"       # Longer → paragraph/label
        if aspect > 5 and h < 25:
            return "text_block"
        if aspect > 0.8 and aspect < 1.3 and w > 40 and h > 40:
            return "icon"
        if w > 200 and h > 150:
            return "panel"
        # Geometry fallback for unrecognised tags (e.g. raw OCR words like "OK"):
        # Small, bounded regions with a non-empty label are almost always buttons.
        if text and w < 120 and h < 50:
            return "button"
        return "unknown"

    @staticmethod
    def _action_pattern_boost(text: str, objective: str) -> float:
        """Boost semantic score when structural tag matches objective action type.

        We use structural geometry tags ("text:N", "input", "icon") rather than
        hallucinated labels, so we boost based on element geometry vs. intent.
        """
        obj_lower = objective.lower()
        boost = 0.0

        # "input" tag = text field → boost for type/search/enter intents
        if text == "input":
            if any(kw in obj_lower for kw in ("type", "enter", "write", "search", "fill")):
                boost += 0.35

        # Short text glyph count → likely an interactive button/label
        if text.startswith("text:"):
            try:
                n = int(text.split(":")[1])
            except (IndexError, ValueError):
                n = 99
            if n <= 10:
                # Short label = probably an action button
                if any(kw in obj_lower for kw in ("click", "press", "open", "submit", "save", "close")):
                    boost += 0.25

        # "icon" tag → relevant for navigate / launch / toggle actions
        if text == "icon":
            if any(kw in obj_lower for kw in ("open", "launch", "navigate", "toggle")):
                boost += 0.15

        # Literal OCR match: if the detected text word appears verbatim in the
        # objective (e.g. "Submit" found in "click submit button"), it is very
        # likely the correct target — small bump to signal relevance.
        if text and len(text) >= 2 and text.lower() in obj_lower:
            boost += 0.15

        return min(boost, 1.0)

    # ------------------------------------------------------------------ #
    # Non-Maximum Suppression
    # ------------------------------------------------------------------ #

    @staticmethod
    def _non_max_suppression(
        elements: list[PerceivedElement],
        iou_threshold: float = 0.5,
    ) -> list[PerceivedElement]:
        """Remove overlapping detections, keeping highest-confidence ones."""
        if not elements:
            return []
        sorted_elems = sorted(elements, key=lambda e: e.confidence, reverse=True)
        kept: list[PerceivedElement] = []
        for elem in sorted_elems:
            overlap = False
            for kept_elem in kept:
                iou = _compute_iou(
                    (elem.x, elem.y, elem.width, elem.height),
                    (kept_elem.x, kept_elem.y, kept_elem.width, kept_elem.height),
                )
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(elem)
        return kept

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
        """Convert RGB array to HSV."""
        img_f = arr.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        df = mx - mn
        h = np.zeros_like(mx)
        nonzero = df > 1e-6
        safe_df = np.where(nonzero, df, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hr = np.where(mx == r, (60 * ((g - b) / safe_df) + 360) % 360, 0)
            hg = np.where(mx == g, (60 * ((b - r) / safe_df) + 120) % 360, 0)
            hb = np.where(mx == b, (60 * ((r - g) / safe_df) + 240) % 360, 0)
        h = hr + hg + hb
        s = np.zeros_like(mx)
        s[mx != 0] = df[mx != 0] / mx[mx != 0]
        v = mx
        # Stack as [H/360, S, V]
        return np.stack([h / 360.0, s, v], axis=2)


def _compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute intersection-over-union of two boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / max(union_area, 1)
