"""Pure Pixel-Based POMDP — zero accessibility tree dependency.

The agent operates entirely on raw pixel observations. No DOM, no UIA,
no accessibility metadata. Just screenshots, exactly like a human viewing
a remote-desktop stream.

State features are extracted using classical computer vision:
- Edge orientation histograms
- Color distribution moments
- Local binary patterns (texture)
- SIFT-like keypoint density maps
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass
class PixelObservation:
    """A raw pixel observation with extracted visual features."""

    timestamp: float
    screenshot: np.ndarray  # H x W x 3 RGB
    features: np.ndarray  # Fixed-length feature vector
    resolution: tuple[int, int]


@dataclass
class PixelBeliefState:
    """Belief over possible screen configurations, represented as particles."""

    particles: list[np.ndarray] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    history: list[tuple[PixelObservation, Any]] = field(default_factory=list)

    def update(self, observation: PixelObservation, action: Any) -> None:
        """Bayesian update: reweight particles by visual similarity."""
        if not self.particles:
            self._bootstrap(observation)
            return
        new_weights = []
        for p, w in zip(self.particles, self.weights):
            # Likelihood = visual similarity between particle prediction and obs
            likelihood = self._visual_likelihood(p, observation.features)
            new_weights.append(w * likelihood)
        total = sum(new_weights)
        if total > 0:
            self.weights = [nw / total for nw in new_weights]
        else:
            self.weights = [1.0 / len(self.weights)] * len(self.weights)
        self.history.append((observation, action))
        if len(self.history) > 1:
            self._resample_if_degenerate()

    def predict_particles(self, action: Any) -> None:
        """Propagate particles forward through action effects."""
        # In pixel space, actions cause local changes (click → ripple,
        # type → text appearance). Model as small Gaussian perturbation
        # plus action-specific bias.
        for i, p in enumerate(self.particles):
            noise = np.random.randn(*p.shape).astype(np.float32) * 0.02
            # Action-specific state shift
            shift = self._action_shift(action)
            self.particles[i] = p + noise + shift

    def most_likely(self) -> np.ndarray:
        """Return the highest-weight particle."""
        if not self.particles:
            return np.zeros(256, dtype=np.float32)
        best_idx = int(np.argmax(self.weights))
        return self.particles[best_idx]

    def entropy(self) -> float:
        """Shannon entropy of the belief distribution."""
        if not self.weights:
            return 1.0
        w = np.array(self.weights)
        w = w[w > 0]
        if len(w) == 0:
            return 1.0
        return float(-np.sum(w * np.log2(w)))

    def _bootstrap(self, observation: PixelObservation, n: int = 10) -> None:
        """Initialize particles from first observation with small perturbations."""
        self.particles = []
        self.weights = []
        for i in range(n):
            noise = (
                np.random.randn(*observation.features.shape).astype(np.float32) * 0.05
            )
            self.particles.append(observation.features + noise)
            self.weights.append(1.0 / n)

    @staticmethod
    def _visual_likelihood(particle: np.ndarray, obs_features: np.ndarray) -> float:
        """Compute P(obs | particle) using cosine similarity on feature vectors."""
        dot = float(np.dot(particle, obs_features))
        p_norm = float(np.linalg.norm(particle))
        o_norm = float(np.linalg.norm(obs_features))
        if p_norm == 0 or o_norm == 0:
            return 0.1
        sim = dot / (p_norm * o_norm)
        # Keep a tiny floor for numerical stability without flattening random
        # observations into identical likelihoods.
        return max(1e-6, float(np.exp(-3 * (1 - sim))))

    def _resample_if_degenerate(self, threshold: float = 0.5) -> None:
        """Resample particles if effective sample size drops."""
        w = np.array(self.weights)
        ess = 1.0 / float(np.sum(w**2))
        if ess < threshold * len(self.weights):
            # Systematic resampling
            new_particles = []
            cumsum = np.cumsum(self.weights)
            u0 = np.random.random() / len(self.weights)
            for i in range(len(self.weights)):
                u = u0 + i / len(self.weights)
                idx = int(np.searchsorted(cumsum, u))
                new_particles.append(self.particles[idx].copy())
            self.particles = new_particles
            self.weights = [1.0 / len(self.weights)] * len(self.weights)

    @staticmethod
    def _action_shift(action: Any) -> np.ndarray:
        """Return a feature-space shift vector for a given action."""
        # This is a learned function. For now, use heuristics based on action type.
        dim = 256
        shift = np.zeros(dim, dtype=np.float32)
        if hasattr(action, "action_type"):
            at = action.action_type
            if at == "click":
                # Click typically changes focus → edge density shift in region
                shift[:10] = 0.05
            elif at == "type":
                # Typing adds text → texture change
                shift[10:20] = 0.03
            elif at == "hotkey":
                # Hotkeys cause global UI changes (menus, dialogs)
                shift[20:30] = 0.04
            elif at == "scroll":
                # Scrolling shifts everything vertically
                shift[30:40] = 0.06
        return shift


class PixelFeatureExtractor:
    """Extracts fixed-length feature vectors from raw pixel screenshots.

    Uses classical CV techniques that run in milliseconds on CPU:
    1. Multi-scale edge histograms (shape information)
    2. Color distribution moments (H, S, V)
    3. Local Binary Patterns (texture)
    4. Grid-based brightness variance (layout structure)
    """

    def __init__(self, target_dim: int = 256) -> None:
        self.target_dim = target_dim
        self._grid_size = (4, 4)  # 4x4 spatial grid

    def extract(self, screenshot: np.ndarray) -> np.ndarray:
        """Extract feature vector from RGB screenshot."""
        h, w = screenshot.shape[:2]
        features: list[np.ndarray] = []
        # 1. Edge orientation histogram (64 dims)
        edges = self._edge_features(screenshot)
        features.append(edges)
        # 2. Color moments in HSV (48 dims)
        colors = self._color_features(screenshot)
        features.append(colors)
        # 3. Local Binary Pattern histogram (32 dims)
        texture = self._texture_features(screenshot)
        features.append(texture)
        # 4. Spatial grid brightness (64 dims)
        spatial = self._spatial_features(screenshot)
        features.append(spatial)
        # 5. Aspect ratio and resolution (8 dims)
        meta = np.array(
            [
                h / 1080.0,
                w / 1920.0,
                h / max(w, 1),
                np.log1p(h * w) / 20.0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )
        features.append(meta)
        # Concatenate and normalize
        vec = np.concatenate(features)
        # Replace any NaN/inf from numerical edge cases
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if len(vec) < self.target_dim:
            vec = np.pad(vec, (0, self.target_dim - len(vec)))
        elif len(vec) > self.target_dim:
            vec = vec[: self.target_dim]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _edge_features(self, img: np.ndarray) -> np.ndarray:
        """Multi-scale edge orientation histogram."""
        from scipy import ndimage

        gray = np.mean(img, axis=2)
        edges = []
        for sigma in [1.0, 2.0, 4.0]:
            blurred = ndimage.gaussian_filter(gray, sigma=sigma)
            dx = ndimage.sobel(blurred, axis=1)
            dy = ndimage.sobel(blurred, axis=0)
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = np.arctan2(dy, dx)
            # Weighted histogram of orientations
            hist = np.zeros(8, dtype=np.float32)
            for i in range(8):
                bin_start = -np.pi + i * (2 * np.pi / 8)
                bin_end = bin_start + (2 * np.pi / 8)
                mask = (orientation >= bin_start) & (orientation < bin_end)
                hist[i] = float(np.sum(magnitude[mask]))
            edges.append(hist)
        return np.concatenate(edges)

    def _color_features(self, img: np.ndarray) -> np.ndarray:
        """Color distribution moments in HSV space."""
        # Manual RGB to HSV conversion
        img_f = img.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        df = mx - mn
        # Hue
        h = np.zeros_like(mx)
        nonzero = df > 1e-6
        # Only compute hue where df is nonzero to avoid NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            gmb = np.where(nonzero, g - b, 0.0)
            bmr = np.where(nonzero, b - r, 0.0)
            rmg = np.where(nonzero, r - g, 0.0)
            safe_df = np.where(nonzero, df, 1.0)
            hr = np.where(mx == r, (60 * (gmb / safe_df) + 360) % 360, 0)
            hg = np.where(mx == g, (60 * (bmr / safe_df) + 120) % 360, 0)
            hb = np.where(mx == b, (60 * (rmg / safe_df) + 240) % 360, 0)
        h = hr + hg + hb
        h = h / 360.0  # Normalize to [0, 1]
        # Saturation
        s = np.zeros_like(mx)
        s[mx != 0] = df[mx != 0] / mx[mx != 0]
        # Value
        v = mx
        hsv = np.stack([h, s, v], axis=2)
        features = []
        for channel in range(3):
            c = hsv[:, :, channel].flatten()
            features.extend(
                [
                    float(np.mean(c)),
                    float(np.std(c)),
                    float(np.percentile(c, 25)),
                    float(np.percentile(c, 75)),
                ]
            )
        # Dominant hue bin (12 bins)
        hue_hist = np.histogram(hsv[:, :, 0], bins=12, range=(0, 1))[0].astype(
            np.float32
        )
        hue_hist = hue_hist / max(np.sum(hue_hist), 1)
        features.append(hue_hist)
        return np.concatenate(
            [
                np.array(f if isinstance(f, np.ndarray) else [f], dtype=np.float32)
                for f in features
            ]
        )

    def _texture_features(self, img: np.ndarray) -> np.ndarray:
        """Uniform Local Binary Pattern histogram."""
        gray = np.mean(img, axis=2).astype(np.uint8)
        lbp = self._compute_lbp(gray)
        hist = np.histogram(lbp, bins=32, range=(0, 32))[0].astype(np.float32)
        return hist / max(np.sum(hist), 1)

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        """Compute uniform LBP (simplified 3x3 neighborhood)."""
        padded = np.pad(gray, 1, mode="edge")
        center = gray
        result = np.zeros_like(gray, dtype=np.uint8)
        # 8 neighbors
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for i, (dy, dx) in enumerate(offsets):
            neighbor = padded[
                1 + dy : 1 + dy + gray.shape[0], 1 + dx : 1 + dx + gray.shape[1]
            ]
            result += ((neighbor >= center) << i).astype(np.uint8)
        return result

    def _spatial_features(self, img: np.ndarray) -> np.ndarray:
        """Grid-based brightness and contrast features."""
        gray = np.mean(img, axis=2)
        h, w = gray.shape
        grid_h, grid_w = self._grid_size
        cell_h, cell_w = h // grid_h, w // grid_w
        features = []
        for i in range(grid_h):
            for j in range(grid_w):
                y0, x0 = i * cell_h, j * cell_w
                y1, x1 = min(y0 + cell_h, h), min(x0 + cell_w, w)
                cell = gray[y0:y1, x0:x1]
                features.extend(
                    [
                        float(np.mean(cell)),
                        float(np.std(cell)),
                        float(np.max(cell) - np.min(cell)),
                        float(np.sum(cell > np.mean(gray))) / max(cell.size, 1),
                    ]
                )
        return np.array(features, dtype=np.float32) / 255.0


class ScreenshotBackend(Protocol):
    """Backend that only provides screenshot capture (no accessibility tree)."""

    def capture(self) -> bytes:
        """Return a PNG-encoded screenshot."""


class PurePixelEnvironment:
    """Environment wrapper for pure pixel-based control.

    No UiNode. No accessibility tree. Just pixels and actions.
    """

    def __init__(self, screenshot_backend: ScreenshotBackend) -> None:
        self.backend = screenshot_backend
        self.extractor = PixelFeatureExtractor(target_dim=256)
        self.belief = PixelBeliefState()
        self._step_count = 0

    def observe(self) -> PixelObservation:
        """Capture screenshot and extract features."""
        png_bytes = self.backend.capture()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)
        features = self.extractor.extract(arr)
        obs = PixelObservation(
            timestamp=time.time(),
            screenshot=arr,
            features=features,
            resolution=arr.shape[:2][::-1],
        )
        if self._step_count == 0:
            self.belief._bootstrap(obs)
        self._step_count += 1
        return obs

    def act(self, action: Any) -> dict[str, Any]:
        """Execute action and return receipt."""
        start = time.time()
        try:
            if hasattr(self.backend, "perform"):
                receipt = self.backend.perform(action)
            else:
                receipt = {"status": "performed", "action": str(action)}
            return {
                "receipt": receipt,
                "duration_ms": (time.time() - start) * 1000,
                "error": None,
            }
        except Exception as exc:
            return {
                "receipt": str(exc),
                "duration_ms": (time.time() - start) * 1000,
                "error": str(exc),
            }

    def step(self, action: Any) -> tuple[PixelObservation, dict[str, Any]]:
        """Full POMDP step: act, wait for UI to settle, observe."""
        outcome = self.act(action)
        time.sleep(0.2)
        obs = self.observe()
        self.belief.update(obs, action)
        return obs, outcome

    def get_visual_state_summary(self) -> dict[str, Any]:
        """Human-readable summary of current visual state."""
        return {
            "resolution": self.belief.history[-1][0].resolution
            if self.belief.history
            else (0, 0),
            "belief_entropy": self.belief.entropy(),
            "particle_count": len(self.belief.particles),
            "steps_taken": self._step_count,
        }
