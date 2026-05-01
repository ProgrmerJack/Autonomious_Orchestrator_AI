"""Build train/eval examples from universal OS-agent trajectory JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TRAINING_HEADS = (
    "perception",
    "affordance_ranker",
    "option_policy",
    "world_model",
    "outcome_critic",
)
SHADOW_HEAD_ORDER = (
    "outcome_critic",
    "option_policy",
    "affordance_ranker",
)


@dataclass(slots=True)
class TrainingExample:
    head: str
    input: dict[str, Any]
    target: dict[str, Any]
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryTrainingBundle:
    examples: list[TrainingExample] = field(default_factory=list)

    def by_head(self) -> dict[str, list[TrainingExample]]:
        grouped: dict[str, list[TrainingExample]] = {
            head: [] for head in TRAINING_HEADS
        }
        for example in self.examples:
            grouped.setdefault(example.head, []).append(example)
        return grouped

    def summary(self) -> dict[str, Any]:
        grouped = self.by_head()
        return {
            "schema_version": 1,
            "total_examples": len(self.examples),
            "heads": {head: len(items) for head, items in grouped.items()},
            "ready_for_training": all(grouped[head] for head in TRAINING_HEADS),
        }

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for example in self.examples:
                line = json.dumps(example.asdict(), sort_keys=True)
                handle.write(line + "\n")
        return target


class TrajectoryTrainingBuilder:
    """Convert recorded trajectory steps into supervised learning targets."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.trajectory_root = self.workspace_root / ".agentos" / "trajectories"

    def build(
        self,
        paths: list[str | Path] | None = None,
    ) -> TrajectoryTrainingBundle:
        bundle = TrajectoryTrainingBundle()
        for event in self.iter_step_events(paths):
            bundle.examples.extend(_examples_from_step(event))
        return bundle

    def iter_step_events(
        self,
        paths: list[str | Path] | None = None,
    ) -> list[dict[str, Any]]:
        source_paths = [Path(path) for path in paths] if paths else self._paths()
        events: list[dict[str, Any]] = []
        for path in source_paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "step":
                    payload.setdefault("trajectory_path", str(path))
                    events.append(payload)
        return events

    def write_dataset(
        self,
        output_path: str | Path | None = None,
        paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        bundle = self.build(paths)
        target = Path(output_path) if output_path else self._default_dataset_path()
        bundle.write_jsonl(target)
        summary = bundle.summary()
        summary["path"] = str(target)
        return summary

    def write_shadow_head_datasets(
        self,
        output_dir: str | Path | None = None,
        paths: list[str | Path] | None = None,
        head_order: tuple[str, ...] = SHADOW_HEAD_ORDER,
    ) -> dict[str, Any]:
        bundle = self.build(paths)
        grouped = bundle.by_head()
        target_dir = Path(output_dir) if output_dir else self._shadow_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        heads: dict[str, dict[str, Any]] = {}
        for index, head in enumerate(head_order, start=1):
            examples = grouped.get(head, [])
            path = target_dir / f"{index:02d}_{head}.jsonl"
            TrajectoryTrainingBundle(examples).write_jsonl(path)
            heads[head] = {
                "path": str(path),
                "examples": len(examples),
                "ready": bool(examples),
                "advisory_only": True,
            }
        return {
            "schema_version": 1,
            "advisory_only": True,
            "baseline_required": True,
            "head_order": list(head_order),
            "output_dir": str(target_dir),
            "total_examples": sum(item["examples"] for item in heads.values()),
            "heads": heads,
            "ready_for_shadow_training": all(item["ready"] for item in heads.values()),
        }

    def _paths(self) -> list[Path]:
        if not self.trajectory_root.exists():
            return []
        return sorted(self.trajectory_root.glob("*.jsonl"))

    def _default_dataset_path(self) -> Path:
        return self.workspace_root / ".agentos" / "training" / "trajectory.jsonl"

    def _shadow_dir(self) -> Path:
        return self.workspace_root / ".agentos" / "training" / "shadow_heads"


def _examples_from_step(event: dict[str, Any]) -> list[TrainingExample]:
    before = _dict(event.get("before"))
    after = _dict(event.get("after"))
    action = _dict(event.get("action"))
    outcome = _dict(event.get("outcome_evaluation"))
    mode = _dict(event.get("mode_decision"))
    metadata = {
        "run_id": event.get("run_id"),
        "objective": event.get("objective"),
        "option": event.get("option"),
        "trajectory_path": event.get("trajectory_path"),
    }
    return [
        TrainingExample(
            head="perception",
            input={"ui_summary": before, "objective": event.get("objective")},
            target={
                "element_count": before.get("element_count", 0),
                "interactive_count": before.get("interactive_count", 0),
                "active_modal": before.get("active_modal"),
            },
            metadata=metadata,
        ),
        TrainingExample(
            head="affordance_ranker",
            input={"objective": event.get("objective"), "ui_summary": before},
            target={
                "selector": action.get("selector"),
                "action_type": action.get("action_type"),
                "matched": bool(outcome.get("matched")),
            },
            weight=1.0 if outcome.get("matched") else 0.4,
            metadata=metadata,
        ),
        TrainingExample(
            head="option_policy",
            input={
                "objective": event.get("objective"),
                "option": event.get("option"),
                "before": before,
            },
            target={
                "mode": mode.get("mode") or _mode_from_action(action),
                "confidence": mode.get("confidence", 0.0),
            },
            metadata=metadata,
        ),
        TrainingExample(
            head="world_model",
            input={"before": before, "action": action},
            target={"after": after, "diff": _dict(event.get("diff"))},
            metadata=metadata,
        ),
        TrainingExample(
            head="outcome_critic",
            input={
                "expected_observation": event.get("expected_observation"),
                "receipt": event.get("receipt"),
                "diff": _dict(event.get("diff")),
            },
            target={
                "matched": bool(outcome.get("matched")),
                "failure_reason": outcome.get("failure_reason"),
                "new_blocker": outcome.get("new_blocker"),
                "suggested_repair": outcome.get("suggested_repair"),
            },
            weight=1.2 if outcome.get("failure_reason") else 1.0,
            metadata=metadata,
        ),
    ]


def _mode_from_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type in {"tool", "explore"}:
        return action_type
    return "ui"


def _dict(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return payload
