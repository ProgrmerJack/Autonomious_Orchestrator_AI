from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .base import BackendUnavailable, OsControlBackend, UiAction


@dataclass(slots=True)
class Point:
    x: int
    y: int


@dataclass(slots=True)
class VisualObservation:
    width: int
    height: int
    label: str = "screen"
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class TargetEstimate:
    point: Point
    confidence: float
    rationale: str = ""


@dataclass(slots=True)
class VisualProbeResult:
    success: bool
    error_dx: int = 0
    error_dy: int = 0
    message: str = ""


class VisualRefinementBackend(Protocol):
    def capture(self) -> VisualObservation:
        """Capture or describe the current rendered viewport."""

    def estimate_target(
        self,
        observation: VisualObservation,
        target: str,
    ) -> TargetEstimate:
        """Estimate where the target appears in the viewport."""

    def probe(self, point: Point, target: str) -> VisualProbeResult:
        """Probe a point and measure displacement error."""


class SeePointRefineController:
    """Multi-turn visual correction for inaccessible UI surfaces."""

    def __init__(
        self,
        backend: VisualRefinementBackend,
        max_turns: int = 5,
        tolerance_px: int = 4,
    ) -> None:
        self.backend = backend
        self.max_turns = max_turns
        self.tolerance_px = tolerance_px

    def locate(self, target: str) -> TargetEstimate:
        correction = Point(0, 0)
        last_estimate: TargetEstimate | None = None
        for _turn in range(self.max_turns):
            observation = self.backend.capture()
            estimate = self.backend.estimate_target(observation, target)
            point = Point(
                estimate.point.x + correction.x,
                estimate.point.y + correction.y,
            )
            probe = self.backend.probe(point, target)
            last_estimate = TargetEstimate(
                point=point,
                confidence=estimate.confidence,
                rationale=probe.message or estimate.rationale,
            )
            if probe.success or self._within_tolerance(probe):
                return TargetEstimate(
                    point=point,
                    confidence=max(estimate.confidence, 0.8),
                    rationale="visual target refined within tolerance",
                )
            correction = Point(
                correction.x - probe.error_dx,
                correction.y - probe.error_dy,
            )
        if last_estimate is None:
            raise BackendUnavailable("visual backend produced no estimate")
        return last_estimate

    def _within_tolerance(self, probe: VisualProbeResult) -> bool:
        return (
            abs(probe.error_dx) <= self.tolerance_px
            and abs(probe.error_dy) <= self.tolerance_px
        )


class HybridControlBackend:
    """Uses accessibility first and visual refinement only as fallback."""

    def __init__(
        self,
        primary: OsControlBackend,
        fallback: SeePointRefineController,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"hybrid:{primary.name}"

    def available(self) -> bool:
        return self.primary.available()

    def perform(self, action: UiAction) -> str:
        try:
            nodes = self.primary.snapshot()
        except BackendUnavailable:
            nodes = []
        if nodes:
            return self.primary.perform(action)
        estimate = self.fallback.locate(action.selector)
        return f"visual:{estimate.point.x},{estimate.point.y}"
