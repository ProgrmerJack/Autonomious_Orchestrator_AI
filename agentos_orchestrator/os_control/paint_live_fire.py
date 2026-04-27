from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentos_orchestrator.cognition.safety_gates import (
    FormalSafetyVerifier,
    SafetyPolicy,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode
from agentos_orchestrator.os_control.image_verification import analyze_png
from agentos_orchestrator.os_control.live_fire import (
    PollObservation,
    SnapshotPoller,
    SnapshotPollTimeout,
    _elapsed_ms,
    _has_role,
    _json_or_text,
    _node_label,
    _node_text,
    _save_as_dialog_observation,
    enable_windows_dpi_awareness,
)


DEFAULT_PAINT_FILE_NAME = "paint_smoke.png"
DEFAULT_PAINT_STROKE_POINTS = (
    (0.35, 0.58),
    (0.65, 0.58),
    (0.65, 0.72),
    (0.35, 0.72),
    (0.35, 0.58),
)


@dataclass(slots=True)
class PaintLiveFireConfig:
    file_name: str = DEFAULT_PAINT_FILE_NAME
    dialog_timeout_seconds: float = 12.0
    file_timeout_seconds: float = 8.0
    poll_interval_seconds: float = 0.25
    stable_snapshot_count: int = 2
    paint_selector: str = "name=Paint"
    draw_selector: str = "name=Paint"
    filename_selector: str = "automation_id=1001&&class_name=Edit"
    save_button_selector: str = "automation_id=1&&class_name=Button&&name=Save"
    save_hotkey: str = "^s"
    stroke_points: tuple[
        tuple[float, float], ...
    ] = DEFAULT_PAINT_STROKE_POINTS
    min_non_background_pixels: int = 1
    min_non_background_width: int = 40
    min_non_background_height: int = 40


@dataclass(slots=True)
class PaintLiveFireResult:
    success: bool
    target_path: str
    actual_sha256: str = ""
    dpi_awareness: str = "not-attempted"
    image_width: int = 0
    image_height: int = 0
    distinct_pixel_count: int = 0
    non_background_pixel_count: int = 0
    non_background_bounds: tuple[int, int, int, int] | None = None
    receipts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    safety_reason: str = ""
    error: str = ""
    elapsed_ms: float = 0.0


class PaintLiveFireTrial:
    """Guarded Windows Paint live-fire trial with image evidence."""

    def __init__(
        self,
        backend: Any,
        workspace_root: str | Path,
        safety_verifier: FormalSafetyVerifier | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.live_fire_root = self.workspace_root / "artifacts" / "live_fire"
        self.safety_verifier = safety_verifier or FormalSafetyVerifier(
            SafetyPolicy(allowed_roots=[self.live_fire_root])
        )
        self.poller = SnapshotPoller(backend, sleep_fn=sleep_fn)

    def run(
        self,
        config: PaintLiveFireConfig | None = None,
    ) -> PaintLiveFireResult:
        config = config or _default_paint_config()
        start = time.perf_counter()
        target_path = self._target_path(config)
        result = PaintLiveFireResult(
            success=False,
            target_path=str(target_path),
        )
        safety_action = self._target_path_action(config, target_path)
        if not self._verify_target_path(safety_action, result, start):
            return result

        self.live_fire_root.mkdir(parents=True, exist_ok=True)
        result.dpi_awareness = enable_windows_dpi_awareness()

        try:
            self._execute_paint_save(
                config,
                target_path,
                safety_action,
                result,
            )
        except (
            SnapshotPollTimeout,
            FileNotFoundError,
            RuntimeError,
            OSError,
            ValueError,
            zlib.error,
        ) as exc:
            result.error = str(exc)

        result.elapsed_ms = _elapsed_ms(start)
        return result

    def _target_path(self, config: PaintLiveFireConfig) -> Path:
        return (self.live_fire_root / config.file_name).resolve(strict=False)

    def _target_path_action(
        self,
        config: PaintLiveFireConfig,
        target_path: Path,
    ) -> UiAction:
        return UiAction(
            action_type="set_text",
            selector=config.filename_selector,
            value=str(target_path),
            metadata={"target_path": str(target_path)},
        )

    def _verify_target_path(
        self,
        action: UiAction,
        result: PaintLiveFireResult,
        start: float,
    ) -> bool:
        safety = self.safety_verifier.verify_action(
            action,
            objective="paint live-fire save target",
        )
        result.safety_reason = safety.reason
        if safety.allowed:
            return True
        result.error = f"Safety gate blocked target path: {safety.reason}"
        result.elapsed_ms = _elapsed_ms(start)
        return False

    def _execute_paint_save(
        self,
        config: PaintLiveFireConfig,
        target_path: Path,
        safety_action: UiAction,
        result: PaintLiveFireResult,
    ) -> None:
        self._perform(
            UiAction("launch_app", "mspaint.exe", "mspaint.exe"),
            result,
        )
        self._poll("paint_window", _paint_window_observation, config, result)
        self._perform(
            UiAction(
                "draw_path",
                config.draw_selector,
                json.dumps({"points": config.stroke_points}),
            ),
            result,
        )
        self._perform(
            UiAction("hotkey", "app-window", config.save_hotkey),
            result,
        )
        self._poll(
            "save_as_dialog",
            _save_as_dialog_observation,
            config,
            result,
        )
        self._perform(safety_action, result)
        self._perform(UiAction("invoke", config.save_button_selector), result)
        self._verify_image(target_path, config, result)

    def _perform(
        self,
        action: UiAction,
        result: PaintLiveFireResult,
    ) -> None:
        receipt = self.backend.perform(action)
        result.receipts.append(
            {
                "action_type": action.action_type,
                "selector": action.selector,
                "value": action.value,
                "metadata": dict(action.metadata),
                "receipt": _json_or_text(receipt),
            }
        )

    def _poll(
        self,
        label: str,
        predicate: Callable[[list[UiNode]], PollObservation],
        config: PaintLiveFireConfig,
        result: PaintLiveFireResult,
    ) -> None:
        observation = self.poller.until(
            predicate,
            timeout_seconds=config.dialog_timeout_seconds,
            interval_seconds=config.poll_interval_seconds,
            stable_count=config.stable_snapshot_count,
        )
        result.observations.append(
            {
                "label": label,
                "matched": observation.matched,
                "reason": observation.reason,
                "node_count": observation.node_count,
                "matched_nodes": observation.matched_nodes,
            }
        )

    def _verify_image(
        self,
        target_path: Path,
        config: PaintLiveFireConfig,
        result: PaintLiveFireResult,
    ) -> None:
        summary = self._wait_for_image_summary(target_path, config)
        result.actual_sha256 = summary.sha256
        result.image_width = summary.width
        result.image_height = summary.height
        result.distinct_pixel_count = summary.distinct_pixel_count
        result.non_background_pixel_count = summary.non_background_pixel_count
        result.non_background_bounds = summary.non_background_bounds
        footprint_width, footprint_height = _foreground_footprint(
            summary.non_background_bounds
        )
        result.success = _paint_image_matches(
            config,
            summary,
            footprint_width,
            footprint_height,
        )
        if not result.success:
            result.error = (
                "Saved Paint image did not contain the expected 2D stroke "
                f"footprint: pixels={summary.non_background_pixel_count} "
                f"width={footprint_width} height={footprint_height}"
            )

    def _wait_for_image_summary(
        self,
        target_path: Path,
        config: PaintLiveFireConfig,
    ):
        attempts = max(
            1,
            int(
                config.file_timeout_seconds
                / max(config.poll_interval_seconds, 0.001)
            )
            + 1,
        )
        last_error = ""
        for attempt in range(attempts):
            if target_path.exists():
                try:
                    return analyze_png(target_path.read_bytes())
                except ValueError as exc:
                    last_error = str(exc)
            if attempt < attempts - 1:
                self.poller.sleep_fn(config.poll_interval_seconds)
        if last_error:
            raise ValueError(last_error)
        raise FileNotFoundError(
            f"Paint output file was not created: {target_path}"
        )


def _paint_window_observation(nodes: list[UiNode]) -> PollObservation:
    matches = [
        node
        for node in nodes
        if "paint" in _node_text(node) and _has_role(node, "window")
    ]
    return PollObservation(
        matched=bool(matches),
        reason=(
            "paint window observed"
            if matches
            else "paint window not observed"
        ),
        node_count=len(nodes),
        matched_nodes=[_node_label(node) for node in matches[:5]],
    )


def _default_paint_config() -> PaintLiveFireConfig:
    return PaintLiveFireConfig(
        file_name=f"paint_smoke_{int(time.time())}.png"
    )


def _paint_image_matches(
    config: PaintLiveFireConfig,
    summary: Any,
    footprint_width: int,
    footprint_height: int,
) -> bool:
    return (
        summary.non_background_pixel_count >= config.min_non_background_pixels
        and footprint_width >= config.min_non_background_width
        and footprint_height >= config.min_non_background_height
    )


def _foreground_footprint(
    bounds: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    if bounds is None:
        return 0, 0
    left, top, right, bottom = bounds
    return right - left + 1, bottom - top + 1
