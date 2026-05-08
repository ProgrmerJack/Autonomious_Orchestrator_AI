from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
from typing import Any

from agentos_orchestrator.cognition.tool_executor import (
    QuantAnalysisRequest,
    ToolExecutor,
)
from agentos_orchestrator.os_control.base import UiAction

from .models import DesktopWorkflowPlan, DesktopWorkflowStep


_PROGRAMMER_KINDS = {
    "report",
    "presentation-outline",
    "drawing-concept",
    "script",
}


@dataclass(slots=True)
class ProgrammerOutput:
    path: str
    kind: str
    description: str
    sandbox_name: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProgrammerTask:
    objective: str
    mode: str
    selector: str
    description: str
    code: str
    outputs: list[ProgrammerOutput]
    allow_network: bool = False
    allowed_packages: list[str] = field(default_factory=list)
    expose_env_keys: list[str] = field(default_factory=list)
    timeout_seconds: int = 45

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [output.asdict() for output in self.outputs]
        return payload

    @classmethod
    def fromdict(cls, raw: dict[str, Any]) -> "ProgrammerTask":
        outputs = [
            ProgrammerOutput(**item)
            for item in list(raw.get("outputs") or [])
            if isinstance(item, dict)
        ]
        return cls(
            objective=str(raw.get("objective") or ""),
            mode=str(raw.get("mode") or "app-task"),
            selector=str(
                raw.get("selector") or "tool_executor:workflow_programmer"
            ),
            description=str(raw.get("description") or ""),
            code=str(raw.get("code") or ""),
            outputs=outputs,
            allow_network=bool(raw.get("allow_network", False)),
            allowed_packages=[
                str(item) for item in raw.get("allowed_packages", [])
            ],
            expose_env_keys=[
                str(item) for item in raw.get("expose_env_keys", [])
            ],
            timeout_seconds=int(raw.get("timeout_seconds", 45)),
        )


class ProgrammerLane:
    """Generate workflow artefacts through an explicit sandboxed code lane."""

    def __init__(
        self,
        workspace_root: str | Path,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.tool_executor = tool_executor or ToolExecutor(
            self.workspace_root / ".agentos",
        )

    def augment_plan(self, plan: DesktopWorkflowPlan) -> DesktopWorkflowPlan:
        if any(
            step.action_type == "tool"
            and step.selector == "tool_executor:workflow_programmer"
            for step in plan.steps
        ):
            return plan
        task = self.build_task(plan)
        if task is None:
            return plan
        first_output = task.outputs[0]
        first_output_path = str(
            (self.workspace_root / first_output.path).resolve(),
        )
        plan.steps = [
            DesktopWorkflowStep(
                action_type="tool",
                selector=task.selector,
                description=task.description,
                metadata={
                    "tool_request": task.asdict(),
                    "programmer_lane": {
                        "mode": task.mode,
                        "output_count": len(task.outputs),
                    },
                    "path": first_output_path,
                    "verification_contract": {
                        "kind": "file_exists",
                        "expected": f"The file exists at {first_output_path}.",
                        "path": first_output_path,
                        "required": True,
                    },
                },
            ),
            *plan.steps,
        ]
        return plan

    def reserved_paths(self, plan: DesktopWorkflowPlan) -> set[str]:
        task = self.build_task(plan)
        if task is None:
            return set()
        return {output.path for output in task.outputs}

    def build_task(self, plan: DesktopWorkflowPlan) -> ProgrammerTask | None:
        outputs = [
            ProgrammerOutput(
                path=artifact.path,
                kind=artifact.kind,
                description=artifact.description,
                sandbox_name=Path(artifact.path).name,
            )
            for artifact in plan.artifacts
            if artifact.kind in _PROGRAMMER_KINDS
        ]
        if not outputs:
            return None
        code = self._programmer_code(plan.objective, plan.mode, outputs)
        return ProgrammerTask(
            objective=plan.objective,
            mode=plan.mode,
            selector="tool_executor:workflow_programmer",
            description=(
                "Generate workflow artefacts in the sandboxed programmer "
                "lane."
            ),
            code=code,
            outputs=outputs,
        )

    def execute_action(self, action: UiAction) -> str:
        request_data = action.metadata.get("tool_request")
        if not isinstance(request_data, dict):
            return json.dumps(
                {
                    "status": "invalid-tool-request",
                    "success": False,
                    "error": "Missing programmer tool request metadata.",
                },
                sort_keys=True,
            )
        task = ProgrammerTask.fromdict(request_data)
        result = self.tool_executor.run(
            QuantAnalysisRequest(
                objective=task.objective,
                code=task.code,
                allow_network=task.allow_network,
                allowed_packages=list(task.allowed_packages),
                timeout_seconds=task.timeout_seconds,
                expose_env_keys=list(task.expose_env_keys),
            )
        )
        sandbox_files = {path.name: path for path in result.artefacts}
        generated_outputs: list[dict[str, Any]] = []
        missing_outputs: list[str] = []
        for output in task.outputs:
            source = sandbox_files.get(output.sandbox_name)
            if source is None:
                missing_outputs.append(output.path)
                continue
            target = self.workspace_root / output.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            generated_outputs.append(
                {
                    "path": output.path,
                    "kind": output.kind,
                    "description": output.description,
                    "bytes": target.stat().st_size,
                }
            )
        success = bool(result.success and not missing_outputs)
        status = "success" if success else "missing_outputs"
        if not result.success:
            status = "tool_error"
        payload = {
            "status": status,
            "success": success,
            "selector": task.selector,
            "objective": task.objective,
            "mode": task.mode,
            "generated_outputs": generated_outputs,
            "missing_outputs": missing_outputs,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "parsed_results": result.parsed_results,
            "artefacts": [str(path) for path in result.artefacts],
            "elapsed_ms": result.elapsed_ms,
        }
        return json.dumps(payload, sort_keys=True)

    def _programmer_code(
        self,
        objective: str,
        mode: str,
        outputs: list[ProgrammerOutput],
    ) -> str:
        contents = {
            output.sandbox_name: self._content_for_output(
                objective,
                mode,
                output,
            )
            for output in outputs
        }
        payload = json.dumps(contents, sort_keys=True)
        return "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                f"outputs = {payload}",
                "manifest = {'files': [], 'bytes': {}}",
                "for name, text in outputs.items():",
                "    path = Path(name)",
                "    path.write_text(text, encoding='utf-8')",
                "    manifest['files'].append(name)",
                "    manifest['bytes'][name] = len(text.encode('utf-8'))",
                (
                    "print('RESULT: generated=' + json.dumps(manifest, "
                    "sort_keys=True))"
                ),
                "",
            ]
        )

    def _content_for_output(
        self,
        objective: str,
        mode: str,
        output: ProgrammerOutput,
    ) -> str:
        if output.kind == "report":
            return _report_content(objective)
        if output.kind == "presentation-outline":
            return _presentation_outline(objective)
        if output.kind == "drawing-concept":
            return _drawing_concept(objective)
        if output.kind == "script":
            return _script_content(objective, output.path)
        return json.dumps(
            {
                "objective": objective,
                "mode": mode,
                "output": output.asdict(),
            },
            indent=2,
        )


def _report_content(objective: str) -> str:
    title = _title(objective)
    tickers = _tickers(objective)
    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        (
            "This report was generated through the AgentOS programmer lane "
            "before any GUI handoff, so the workflow already has a concrete "
            "draft to refine."
        ),
        "",
        "## Objective",
        objective.strip() or title,
        "",
        "## Operating Plan",
        "1. Validate the requested outcome and artefacts.",
        (
            "2. Use the safest available control route for the current app "
            "family."
        ),
        "3. Re-observe after each committed step before continuing.",
        "",
    ]
    if tickers or _looks_like_market_work(objective):
        lines.extend(
            [
                "## Market Focus",
                (
                    "The universal OS agent should prefer the programmer lane "
                    "for data shaping, chart preparation, and reusable notes "
                    "before opening a browser or trading surface."
                ),
            ]
        )
        if tickers:
            lines.extend([f"- Monitor ticker: {ticker}" for ticker in tickers])
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "- Approval remains required for destructive or external actions.",
            "- Outputs should be reproducible from the sandbox recipe.",
            (
                "- GUI work should only start once the generated artefacts "
                "verify."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _presentation_outline(objective: str) -> str:
    title = _title(objective)
    return "\n".join(
        [
            "# Presentation Outline",
            "",
            f"Objective: {title}",
            "",
            "## Slide 1: Title",
            f"- {title}",
            "## Slide 2: Situation",
            "- Current state",
            "- Constraint surface",
            "## Slide 3: Strategy",
            "- Programmer lane output",
            "- AppAgent handoff",
            "## Slide 4: Risks",
            "- Goal lock",
            "- Approval boundaries",
            "## Slide 5: Commit Plan",
            "- Safe prefix commit",
            "- Re-observe before the next step",
            "",
        ]
    )


def _drawing_concept(objective: str) -> str:
    label = _xml_escape(_title(objective))
    return "\n".join(
        [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="960" '
                'height="540" viewBox="0 0 960 540">'
            ),
            '  <rect width="960" height="540" fill="#f4efe6" />',
            (
                '  <rect x="72" y="72" width="816" height="396" '
                'rx="28" fill="#fffaf0" stroke="#18363d" '
                'stroke-width="6" />'
            ),
            (
                '  <path d="M 120 390 C 230 210, 340 210, 450 390 S '
                '670 570, 820 250" fill="none" stroke="#1f6f78" '
                'stroke-width="14" />'
            ),
            '  <circle cx="250" cy="210" r="34" fill="#f28f3b" />',
            (
                f'  <text x="120" y="470" fill="#18363d" '
                f'font-size="30" font-family="Georgia">{label}</text>'
            ),
            "</svg>",
            "",
        ]
    )


def _script_content(objective: str, output_path: str) -> str:
    script_name = Path(output_path).stem.replace("-", "_")
    goal = objective.replace('"', "'")
    return "\n".join(
        [
            '"""Workflow starter generated by the AgentOS programmer lane."""',
            "",
            "from __future__ import annotations",
            "",
            "from pathlib import Path",
            "",
            "",
            f"def {script_name}_workflow() -> None:",
            f'    """Goal: {goal}."""',
            "    workspace = Path.cwd()",
            "    print(f'Workspace: {workspace}')",
            "    print('Replace this starter with domain-specific logic.')",
            "",
            "",
            "def main() -> None:",
            f"    {script_name}_workflow()",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )


def _title(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.rstrip(".") or "AgentOS Workflow Artefact"


def _tickers(objective: str) -> list[str]:
    ignored = {"A", "I", "THE", "FOR", "WITH", "API"}
    values = re.findall(r"\b[A-Z]{1,5}\b", objective)
    return [item for item in values if item not in ignored]


def _looks_like_market_work(objective: str) -> bool:
    lower = objective.lower()
    return any(
        token in lower
        for token in ("stock", "market", "portfolio", "ticker", "price")
    )


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
