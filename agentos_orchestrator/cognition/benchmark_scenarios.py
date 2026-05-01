"""Golden-trace benchmark helpers for universal OS-agent failure modes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FAILURE_MODES = {
    "surprise_modal",
    "invalid_path",
    "focus_theft",
    "delayed_dialog",
    "stale_screenshot",
    "tool_vs_ui_routing",
}


@dataclass(slots=True)
class GoldenTraceFinding:
    trace_id: str
    passed: bool
    reason: str
    severity: str = "info"
    context: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GoldenTraceReplay:
    trace_id: str
    passed: bool
    expectations_checked: int
    step_count: int
    failure_modes: list[str]
    findings: list[GoldenTraceFinding] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "passed": self.passed,
            "expectations_checked": self.expectations_checked,
            "step_count": self.step_count,
            "failure_modes": list(self.failure_modes),
            "findings": [finding.asdict() for finding in self.findings],
        }


def load_golden_traces(workspace_root: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace_root)
    trace_dir = root / "benchmarks" / "golden_traces"
    traces: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "trace_id": path.stem,
                "status": "invalid",
                "error": str(exc),
            }
        payload.setdefault("trace_id", path.stem)
        payload.setdefault("path", str(path.relative_to(root)))
        traces.append(payload)
    return traces


def replay_golden_traces(
    workspace_root: str | Path,
    trace_id: str = "",
) -> dict[str, Any]:
    selected = _select_traces(load_golden_traces(workspace_root), trace_id)
    results = [_replay_trace(trace) for trace in selected]
    covered_modes = _covered_failure_modes(results)
    missing_modes = sorted(REQUIRED_FAILURE_MODES.difference(covered_modes))
    suite_findings = _suite_findings(trace_id, missing_modes)
    return {
        "passed": _suite_passed(results, trace_id, missing_modes),
        "trace_count": len(results),
        "covered_failure_modes": covered_modes,
        "missing_failure_modes": missing_modes,
        "suite_findings": suite_findings,
        "results": [result.asdict() for result in results],
    }


def _replay_trace(trace: dict[str, Any]) -> GoldenTraceReplay:
    trace_id = str(trace.get("trace_id") or "unknown")
    findings: list[GoldenTraceFinding] = []
    _require(trace, "objective", findings, trace_id)
    steps = _list_field(trace, "steps", findings, trace_id)
    expectations = _list_field(trace, "expectations", findings, trace_id)
    failure_modes = _failure_modes(trace)
    findings.extend(_invalid_payload_findings(trace, trace_id))
    findings.extend(_step_findings(steps, trace_id))
    findings.extend(_expectation_findings(expectations, trace_id))
    findings.extend(_failure_mode_findings(trace_id, failure_modes))
    return GoldenTraceReplay(
        trace_id=trace_id,
        passed=not any(not finding.passed for finding in findings),
        expectations_checked=len(expectations),
        step_count=len(steps),
        failure_modes=failure_modes,
        findings=findings,
    )


def _select_traces(
    traces: list[dict[str, Any]],
    trace_id: str,
) -> list[dict[str, Any]]:
    if not trace_id:
        return traces
    return [trace for trace in traces if trace.get("trace_id") == trace_id]


def _covered_failure_modes(results: list[GoldenTraceReplay]) -> list[str]:
    modes = {
        mode
        for result in results
        for mode in result.failure_modes
        if mode in REQUIRED_FAILURE_MODES
    }
    return sorted(modes)


def _suite_findings(
    trace_id: str,
    missing_modes: list[str],
) -> list[dict[str, Any]]:
    if trace_id or not missing_modes:
        return []
    return [
        {
            "passed": False,
            "severity": "error",
            "reason": "golden trace corpus misses required failure modes",
            "context": {"missing_failure_modes": missing_modes},
        }
    ]


def _suite_passed(
    results: list[GoldenTraceReplay],
    trace_id: str,
    missing_modes: list[str],
) -> bool:
    if not all(result.passed for result in results):
        return False
    return bool(trace_id) or not missing_modes


def _invalid_payload_findings(
    trace: dict[str, Any],
    trace_id: str,
) -> list[GoldenTraceFinding]:
    if trace.get("status") != "invalid":
        return []
    return [
        GoldenTraceFinding(
            trace_id=trace_id,
            passed=False,
            reason=str(trace.get("error") or "invalid trace payload"),
            severity="error",
        )
    ]


def _step_findings(
    steps: list[Any],
    trace_id: str,
) -> list[GoldenTraceFinding]:
    findings: list[GoldenTraceFinding] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            findings.append(_step_type_finding(trace_id, index))
            continue
        _require(step, "kind", findings, trace_id, step_index=index)
        _require(step, "expect", findings, trace_id, step_index=index)
    return findings


def _step_type_finding(trace_id: str, index: int) -> GoldenTraceFinding:
    return GoldenTraceFinding(
        trace_id=trace_id,
        passed=False,
        reason="step must be an object",
        severity="error",
        context={"step_index": index},
    )


def _expectation_findings(
    expectations: list[Any],
    trace_id: str,
) -> list[GoldenTraceFinding]:
    if expectations:
        return []
    return [
        GoldenTraceFinding(
            trace_id=trace_id,
            passed=False,
            reason="trace must declare at least one expectation",
            severity="error",
        )
    ]


def _failure_mode_findings(
    trace_id: str,
    failure_modes: list[str],
) -> list[GoldenTraceFinding]:
    if not trace_id.startswith("universal_os_") or failure_modes:
        return []
    return [
        GoldenTraceFinding(
            trace_id=trace_id,
            passed=False,
            reason="universal OS traces must declare failure_modes",
            severity="error",
        )
    ]


def _require(
    payload: dict[str, Any],
    key: str,
    findings: list[GoldenTraceFinding],
    trace_id: str,
    step_index: int | None = None,
) -> None:
    if payload.get(key):
        return
    context = {} if step_index is None else {"step_index": step_index}
    findings.append(
        GoldenTraceFinding(
            trace_id=trace_id,
            passed=False,
            reason=f"missing required field: {key}",
            severity="error",
            context=context,
        )
    )


def _list_field(
    payload: dict[str, Any],
    key: str,
    findings: list[GoldenTraceFinding],
    trace_id: str,
) -> list[Any]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    findings.append(
        GoldenTraceFinding(
            trace_id=trace_id,
            passed=False,
            reason=f"{key} must be a list",
            severity="error",
        )
    )
    return []


def _failure_modes(trace: dict[str, Any]) -> list[str]:
    raw = trace.get("failure_modes")
    if not isinstance(raw, list):
        raw = trace.get("coverage", {}).get("failure_modes", [])
    if not isinstance(raw, list):
        return []
    modes = sorted({str(item) for item in raw if str(item)})
    return modes
