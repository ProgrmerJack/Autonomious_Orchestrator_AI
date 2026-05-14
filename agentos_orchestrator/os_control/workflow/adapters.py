from __future__ import annotations

import re
from dataclasses import dataclass
import datetime

from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowStep,
)


@dataclass(slots=True)
class WorkflowContext:
    objective: str
    lower: str
    mode: str
    app_target: str | None
    artifact_path: str | None
    research_artifact_path: str | None = None


def workflow_prefers_research_tool(lower: str, mode: str) -> bool:
    compact = re.sub(r"\s+", " ", lower).strip()
    if not compact:
        return False
    if "http://" in compact or "https://" in compact:
        return False
    if any(
        cue in compact
        for cue in (
            "checkout",
            "purchase",
            "buy ",
            "book ",
            "log in",
            "login",
            "sign in",
            "submit form",
            "fill out",
        )
    ):
        return False
    if any(
        cue in compact
        for cue in (
            "research",
            "analyze",
            "analyse",
            "compare",
            "investigate",
            "benchmark",
            "summarize",
            "summarise",
            "stock",
            "ticker",
            "market",
            "news",
            "trend",
            "competitive",
            "capability",
        )
    ):
        return True
    if mode in {"report", "presentation"} and re.search(
        r"\b(?:about|on|regarding)\b",
        compact,
    ):
        return True
    search_like = any(cue in compact for cue in ("search for", "look up", "find "))
    if not search_like:
        return False
    return any(
        cue in compact
        for cue in (
            "benchmark",
            "stock",
            "ticker",
            "market",
            "news",
            "agent",
            "workflow",
            "trend",
            "compare",
            "analyze",
            "analyse",
            "investigate",
        )
    )


class GenericAppWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.mode != "app-task":
            return []
        if context.app_target is None:
            return []
        intent = context.objective.strip().rstrip(".")
        return [
            DesktopWorkflowStep(
                action_type="type",
                selector="app-workspace",
                value=(
                    "Operator intent:\n"
                    f"{intent}\n\n"
                    "Execute this task in the active app workspace and "
                    "save progress when appropriate."
                ),
                description="Seed the active app workspace with task intent.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save work in the active application.",
            ),
        ]


class SpreadsheetWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.mode != "spreadsheet":
            return []
        cell = self._target_cell(context.lower)
        value = self._cell_value(context.objective)
        return [
            DesktopWorkflowStep(
                action_type="type",
                selector="spreadsheet-grid",
                value=f"{cell}: {value}",
                description="Write requested value into a spreadsheet cell.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save spreadsheet changes.",
            ),
        ]

    @staticmethod
    def _target_cell(lower: str) -> str:
        match = re.search(r"\b([A-Z]{1,3}[0-9]{1,5})\b", lower.upper())
        if match is not None:
            return match.group(1)
        return "A1"

    @staticmethod
    def _cell_value(objective: str) -> str:
        patterns = (
            r"(?:set|update|write|put)\s+.*?\s+to\s+(?P<value>.+)$",
            r"(?:set|update|write|put)\s+(?P<value>.+)$",
        )
        lower = objective.lower()
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match is not None:
                return match.group("value").strip(" .")
        return objective.strip().rstrip(".")


class ExplorerFileOpsWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.app_target != "explorer.exe":
            return []
        operation = self._operation(context.lower)
        if operation is None:
            return []
        value, metadata = self._payload(operation, context.objective)
        return [
            DesktopWorkflowStep(
                action_type="type",
                selector="explorer-file-list",
                value="Select source items for file operation.",
                description="Focus explorer file list before file operations.",
            ),
            DesktopWorkflowStep(
                action_type=f"{operation}_file",
                selector="explorer-file-list",
                value=value,
                metadata=metadata,
                description=f"Apply {operation} file operation semantics.",
            ),
        ]

    @staticmethod
    def _operation(lower: str) -> str | None:
        if "rename" in lower:
            return "rename"
        if "move" in lower:
            return "move"
        if "copy" in lower:
            return "copy"
        return None

    def _payload(
        self,
        operation: str,
        objective: str,
    ) -> tuple[str, dict[str, str]]:
        objective_clean = objective.strip().rstrip(".")
        if operation in {"copy", "move"}:
            source, destination = self._source_destination(
                operation,
                objective_clean,
            )
            return f"{source} -> {destination}", {
                "operation": operation,
                "source": source,
                "destination": destination,
            }
        source, new_name = self._rename_payload(objective_clean)
        return f"{source} -> {new_name}", {
            "operation": operation,
            "source": source,
            "new_name": new_name,
        }

    @staticmethod
    def _source_destination(operation: str, objective: str) -> tuple[str, str]:
        match = re.search(
            rf"{operation}\s+(?P<source>.+?)\s+to\s+(?P<dest>.+)$",
            objective,
            flags=re.I,
        )
        if match is not None:
            return match.group("source").strip(), match.group("dest").strip()
        return "source-item", "destination-folder"

    @staticmethod
    def _rename_payload(objective: str) -> tuple[str, str]:
        match = re.search(
            r"rename\s+(?P<source>.+?)\s+to\s+(?P<name>.+)$",
            objective,
            flags=re.I,
        )
        if match is not None:
            return match.group("source").strip(), match.group("name").strip()
        return "source-item", "renamed-item"


class FileManagerWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.artifact_path is None:
            return []
        if context.mode not in {"presentation", "report", "script"}:
            return []
        folder = self._parent_folder(context.artifact_path)
        return [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector="explorer.exe",
                value="explorer.exe",
                description="Open File Explorer for save/open operations.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="app-workspace",
                value=f"Navigate to {folder}",
                description="Prepare the target workspace folder.",
            ),
        ]

    @staticmethod
    def _parent_folder(path: str) -> str:
        parts = path.replace("\\", "/").split("/")
        if len(parts) <= 1:
            return path
        return "/".join(parts[:-1])


class BrowserWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if not self._needs_browser(context.lower):
            return []
        target = self._browser_target(context.lower)
        if target is None:
            return []
        return [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector="msedge.exe",
                value="msedge.exe",
                description="Launch a browser for research or navigation.",
            ),
            DesktopWorkflowStep(
                action_type="open_url",
                selector="browser-address-bar",
                value=target,
                description="Navigate to the requested URL or search query.",
            ),
        ]

    @staticmethod
    def _needs_browser(lower: str) -> bool:
        cues = (
            "search for",
            "look up",
            "research",
            "find ",
            "analyze",
            "analyse",
            "compare",
            "investigate",
            "browser",
            "website",
            "web ",
            "http://",
            "https://",
            "navigate to",
        )
        return any(cue in lower for cue in cues)

    def _browser_target(self, lower: str) -> str | None:
        match = re.search(r"https?://\S+", lower)
        if match is not None:
            return match.group(0).rstrip(".,)")
        query = self._search_query(lower)
        if not query:
            return None
        encoded = query.replace(" ", "+")
        return f"https://www.bing.com/search?q={encoded}"

    @staticmethod
    def _search_query(lower: str) -> str:
        patterns = (
            r"search for (?P<query>.+?)(?: and | then | using | in | on |$)",
            r"look up (?P<query>.+?)(?: and | then | using | in | on |$)",
            r"research (?P<query>.+?)(?: and | then | using | in | on |$)",
            (
                r"find (?P<query>.+?)"
                r"(?: and (?:analy(?:z|s)e|compare|summari(?:s|z)e)| then | using | in | on |$)"
            ),
            r"(?:analy(?:z|s)e|compare|investigate) (?P<query>.+?)(?: and | then | using | in | on |$)",
            r"navigate to (?P<query>.+?)(?: and | then | using | in | on |$)",
        )
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match is not None:
                return match.group("query").strip()
        fallback = re.sub(r"\s+", " ", lower).strip(" .")
        return fallback[:140]


class ResearchWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        artifact_path = (context.research_artifact_path or "").replace("\\", "/")
        if not artifact_path.endswith("research_brief.md"):
            return []
        if not workflow_prefers_research_tool(context.lower, context.mode):
            return []
        query = BrowserWorkflowAdapter._search_query(context.lower)
        request = {
            "kind": "workflow_research_brief",
            "objective": context.objective,
            "query": query,
            "output_path": artifact_path,
            "max_sources": 12 if context.mode == "report" else 8,
            "per_provider_limit": 4,
        }
        return [
            DesktopWorkflowStep(
                action_type="tool",
                selector="tool_executor:workflow_research",
                description=(
                    "Gather a bounded provider-backed research brief before "
                    "any browser-first UI handoff."
                ),
                metadata={
                    "tool_request": request,
                    "control_channel": "code",
                    "expected_observation": (
                        "A provider-backed research brief is generated in the "
                        "workspace before any UI handoff."
                    ),
                    "verification_contract": {
                        "kind": "receipt_success",
                        "expected": (
                            "The workflow research lane writes a research "
                            "brief with provider-backed sources."
                        ),
                        "target": "tool_executor:workflow_research",
                        "required": True,
                    },
                    "research_lane": {
                        "query": query,
                        "output_path": artifact_path,
                    },
                },
            )
        ]


class ApiIntentWorkflowAdapter:
    URL_RE = re.compile(r"https?://[^\s)>,]+", re.I)
    LOOPBACK_RE = re.compile(
        r"\b(?P<host>localhost|127\.0\.0\.1)(?::(?P<port>\d{1,5}))?(?P<path>/[^\s)>,]*)?",
        re.I,
    )

    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        endpoint = self._endpoint(context.objective)
        if endpoint is None or not self._looks_like_api_task(context.lower, endpoint):
            return []
        method = self._method(context.lower)
        metadata: dict[str, object] = {
            "control_channel": "api",
            "method": method,
            "expected_observation": (
                "The direct API action returns a structured response."
            ),
        }
        if method == "GET":
            metadata["methods"] = ["OPTIONS", "GET"]
        return [
            DesktopWorkflowStep(
                action_type="api_call",
                selector=endpoint,
                description=(
                    "Use the API lane directly instead of routing the task "
                    "through a browser-only handoff."
                ),
                metadata=metadata,
            )
        ]

    @classmethod
    def _endpoint(cls, objective: str) -> str | None:
        explicit = cls.URL_RE.search(objective)
        if explicit is not None:
            return explicit.group(0).rstrip(".,)")
        local = cls.LOOPBACK_RE.search(objective)
        if local is None:
            return None
        host = str(local.group("host") or "localhost")
        port = str(local.group("port") or "")
        path = str(local.group("path") or "")
        authority = f"{host}:{port}" if port else host
        return f"http://{authority}{path}"

    @staticmethod
    def _looks_like_api_task(lower: str, endpoint: str) -> bool:
        endpoint_lower = endpoint.lower()
        if any(
            token in endpoint_lower
            for token in (
                "localhost",
                "127.0.0.1",
                "/api",
                "graphql",
                "openapi",
                "swagger",
                "webhook",
                ".json",
            )
        ):
            return True
        return any(
            cue in lower
            for cue in (
                " api",
                "api ",
                "endpoint",
                "graphql",
                "openapi",
                "swagger",
                "webhook",
                "probe",
                "health",
            )
        )

    @staticmethod
    def _method(lower: str) -> str:
        if any(token in lower for token in (" delete ", " remove ")):
            return "DELETE"
        if any(token in lower for token in (" patch ", " update ")):
            return "PATCH"
        if " put " in lower:
            return "PUT"
        if any(
            token in lower for token in (" post ", " create ", " submit ", " send ")
        ):
            return "POST"
        return "GET"


class OfficeWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.mode == "presentation":
            return self._presentation_steps(context)
        if context.mode == "report":
            return self._document_steps(context)
        return []

    @staticmethod
    def _presentation_steps(
        context: WorkflowContext,
    ) -> list[DesktopWorkflowStep]:
        artifact_path = context.artifact_path or "artifacts/workflows/presentation"
        title = context.objective.strip().rstrip(".")
        return [
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^n",
                description="Create a new presentation window.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^o",
                description="Open the generated outline source for reference.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="presentation-canvas",
                value=(
                    f"Title: {title}\n"
                    "Slide 1: Objective\n"
                    "Slide 2: Research findings\n"
                    "Slide 3: Next actions\n"
                    f"Outline source: {artifact_path}"
                ),
                description="Fill the slides canvas with initial structure.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save the presentation draft.",
            ),
        ]

    @staticmethod
    def _document_steps(context: WorkflowContext) -> list[DesktopWorkflowStep]:
        artifact_path = context.artifact_path or "artifacts/workflows/report"
        title = context.objective.strip().rstrip(".")
        return [
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^n",
                description="Create a fresh document.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^o",
                description="Open the generated report reference if needed.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="document-canvas",
                value=(
                    f"{title}\n\n"
                    "Summary: This draft was prepared by AgentOS.\n"
                    "Action items:\n"
                    "1. Validate findings\n"
                    "2. Expand evidence\n"
                    "3. Finalize deliverable\n\n"
                    f"Report source: {artifact_path}"
                ),
                description=("Populate the report draft with a usable structure."),
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save the report draft.",
            ),
        ]


class EditorWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.mode != "script":
            return []
        artifact_path = (
            context.artifact_path or "artifacts/workflows/generated_script.py"
        )
        title = context.objective.strip().rstrip(".")
        return [
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^n",
                description="Open a new editor tab.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^o",
                description="Open the generated script file for editing.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="editor-canvas",
                value=(
                    '"""Starter automation script generated by AgentOS."""\n\n'
                    "from __future__ import annotations\n\n"
                    "def main() -> None:\n"
                    f'    """Goal: {title}."""\n'
                    '    print("Replace this starter logic with the real '
                    'workflow.")\n\n'
                    'if __name__ == "__main__":\n'
                    "    main()\n\n"
                    f"# Source: {artifact_path}"
                ),
                description="Write a starter script into the editor buffer.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save the edited script.",
            ),
        ]


class SpreadsheetCellEditIntentAdapter:
    """Intent-driven adapter for spreadsheet cell edits.

    Fires when the objective text contains a cell reference and an edit verb,
    regardless of whether ``mode == "spreadsheet"`` has been selected by the
    planner.  Emits a ``cell_edit`` action with structured
    *sandbox-receipt* metadata so the VirtualDesktopSandboxBackend can
    produce a typed receipt (``receipt["cell_edit"]``) for assertions.
    """

    _CELL_EDIT_VERBS = ("set", "update", "write", "enter", "put", "type")
    _CELL_PATTERN = re.compile(
        r"\b(?P<cell>[A-Z]{1,3}[0-9]{1,5}(?::[A-Z]{1,3}[0-9]{1,5})?)\b",
        re.I,
    )

    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        _EXPL = ("spreadsheet", "excel", "xlsx", "csv", "workbook")
        if context.mode == "spreadsheet" and any(kw in context.lower for kw in _EXPL):
            return []
        if not self._matches(context.lower):
            return []
        cell, value = self._extract_cell_edit(context.objective)
        is_formula = value.startswith("=")
        is_range = ":" in cell
        metadata: dict = {
            "cell": cell,
            "value": value,
            "formula": is_formula,
            "range_edit": is_range,
            "sandbox_receipt": {
                "operation": "cell_edit",
                "cell": cell,
                "value": value,
                "formula": is_formula,
                "range_edit": is_range,
            },
        }
        return [
            DesktopWorkflowStep(
                action_type="cell_edit",
                selector="spreadsheet-grid",
                value=f"{cell}: {value}",
                metadata=metadata,
                description=f"Edit cell {cell} with value: {value}.",
            ),
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save spreadsheet changes.",
            ),
        ]

    def _matches(self, lower: str) -> bool:
        has_verb = any(f" {v} " in f" {lower} " for v in self._CELL_EDIT_VERBS)
        has_cell = bool(self._CELL_PATTERN.search(lower.upper()))
        return has_verb and has_cell

    def _extract_cell_edit(self, objective: str) -> tuple[str, str]:
        match = self._CELL_PATTERN.search(objective.upper())
        cell = match.group("cell") if match else "A1"
        value = self._extract_value(objective)
        return cell, value

    @staticmethod
    def _extract_value(objective: str) -> str:
        patterns = (
            r"(?:set|update|write|put|enter|type)\s+\S+\s+to\s+(?P<v>.+)$",
            r"to\s+(?P<v>.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, objective, re.I)
            if match is not None:
                return match.group("v").strip(" .")
        return objective.strip().rstrip(".")


class FileOpsIntentAdapter:
    """Intent-driven adapter for copy / move / rename file operations.

    Fires when the objective text contains copy/move/rename verbs, even when
    the planner has not identified ``explorer.exe`` as the app target.  Emits
    typed *sandbox-receipt* metadata inside ``DesktopWorkflowStep.metadata``
    so the VirtualDesktopSandboxBackend and downstream tests can verify the
    specific file operation (``receipt["file_op"]``).
    """

    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.app_target == "explorer.exe":
            # ExplorerFileOpsWorkflowAdapter already handles explorer targets.
            return []
        operation = self._operation(context.lower)
        if operation is None:
            return []
        value, meta = self._payload(operation, context.objective)
        return [
            DesktopWorkflowStep(
                action_type=f"{operation}_file",
                selector="explorer-file-list",
                value=value,
                metadata=meta,
                description=(
                    f"Intent-driven {operation} operation with sandbox receipt."
                ),
            ),
        ]

    @staticmethod
    def _operation(lower: str) -> str | None:
        if "rename" in lower:
            return "rename"
        if "move" in lower:
            return "move"
        if "copy" in lower:
            return "copy"
        return None

    def _payload(
        self,
        operation: str,
        objective: str,
    ) -> tuple[str, dict]:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean = objective.strip().rstrip(".")
        if operation in {"copy", "move"}:
            source, destination = self._source_destination(operation, clean)
            meta: dict = {
                "operation": operation,
                "source": source,
                "destination": destination,
                "sandbox_receipt": {
                    "operation": operation,
                    "source": source,
                    "destination": destination,
                    "timestamp": ts,
                    "status": "pending",
                },
            }
            return f"{source} -> {destination}", meta
        source, new_name = self._rename_payload(clean)
        meta = {
            "operation": operation,
            "source": source,
            "new_name": new_name,
            "sandbox_receipt": {
                "operation": operation,
                "source": source,
                "new_name": new_name,
                "timestamp": ts,
                "status": "pending",
            },
        }
        return f"{source} -> {new_name}", meta

    @staticmethod
    def _source_destination(operation: str, objective: str) -> tuple[str, str]:
        match = re.search(
            rf"{operation}\s+(?P<src>.+?)\s+to\s+(?P<dst>.+)$",
            objective,
            flags=re.I,
        )
        if match is not None:
            return match.group("src").strip(), match.group("dst").strip()
        return "source-item", "destination-folder"

    @staticmethod
    def _rename_payload(objective: str) -> tuple[str, str]:
        match = re.search(
            r"rename\s+(?P<src>.+?)\s+to\s+(?P<name>.+)$",
            objective,
            flags=re.I,
        )
        if match is not None:
            return match.group("src").strip(), match.group("name").strip()
        return "source-item", "renamed-item"
