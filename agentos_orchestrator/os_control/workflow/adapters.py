from __future__ import annotations

import re
from dataclasses import dataclass, field
import datetime

from agentos_orchestrator.os_control.workflow.intent_parser import StructuredIntent
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
    intent: StructuredIntent = field(
        default_factory=lambda: StructuredIntent(raw_objective="")
    )


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
        if context.app_target in {"calc.exe"}:
            return []
        if context.intent.cross_app or context.intent.operations:
            return []
        if context.app_target in {"notepad.exe", "wordpad.exe"}:
            return self._editor_write_steps(context)
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

    @staticmethod
    def _editor_write_steps(
        context: WorkflowContext,
    ) -> list[DesktopWorkflowStep]:
        if context.app_target not in {"notepad.exe", "wordpad.exe"}:
            return []
        text = GenericAppWorkflowAdapter._editor_text(context.objective)
        if not text:
            return []
        return [
            DesktopWorkflowStep(
                action_type="type",
                selector="document-canvas",
                value=text,
                description="Write the requested text into the active editor.",
                metadata={
                    "verification_contract": {
                        "kind": "field_contains",
                        "expected": "The editor contains the requested text.",
                        "target": "document-canvas",
                        "value": text,
                        "required": True,
                    }
                },
            )
        ]

    @staticmethod
    def _editor_text(objective: str) -> str:
        match = re.search(
            r"\b(?:type|write|enter)\s+(?P<text>.+)$",
            objective.strip(),
            flags=re.I,
        )
        if match is None:
            return ""
        return match.group("text").strip().strip('"')


class CalculatorWorkflowAdapter:
    _DIGIT_NAMES = {
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
    }
    _OPERATOR_SELECTORS = {
        "+": "name=Plus",
        "-": "name=Minus",
        "*": "name=Multiply by",
        "/": "name=Divide by",
    }

    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.mode != "app-task" or context.app_target != "calc.exe":
            return []
        tokens = self._expression_tokens(context.lower)
        if not tokens:
            return []
        steps: list[DesktopWorkflowStep] = []
        for token in tokens:
            if token.isdigit():
                for digit in token:
                    selector = f"name={self._DIGIT_NAMES[digit]}"
                    steps.append(
                        DesktopWorkflowStep(
                            action_type="click",
                            selector=selector,
                            description=(
                                f"Press Calculator button {self._DIGIT_NAMES[digit]}."
                            ),
                        )
                    )
                continue
            selector = self._OPERATOR_SELECTORS.get(token)
            if selector is None:
                return []
            steps.append(
                DesktopWorkflowStep(
                    action_type="click",
                    selector=selector,
                    description=(
                        f"Press Calculator operator {selector.split('=', 1)[1]}."
                    ),
                )
            )
        steps.append(
            DesktopWorkflowStep(
                action_type="click",
                selector="name=Equals",
                description="Evaluate the requested Calculator expression.",
            )
        )
        return steps

    @classmethod
    def _expression_tokens(cls, lower: str) -> list[str]:
        normalized = str(lower or "")
        replacements = (
            (r"multiplied by", " * "),
            (r"multiply by", " * "),
            (r"times", " * "),
            (r"divided by", " / "),
            (r"divide by", " / "),
            (r"plus", " + "),
            (r"minus", " - "),
        )
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.I)
        tokens = re.findall(r"\d+|[+\-*/]", normalized)
        if len(tokens) < 3:
            return []
        if not tokens[0].isdigit() or not tokens[-1].isdigit():
            return []
        for index, token in enumerate(tokens):
            expects_number = index % 2 == 0
            if expects_number != token.isdigit():
                return []
        return tokens


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
        if not self._needs_browser(context):
            return []
        launch_target = context.intent.source_app_target or context.app_target or "msedge.exe"
        target = self._browser_target(context)
        if target is None:
            return []
        return [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=launch_target,
                value=launch_target,
                description="Launch a browser for research or navigation.",
            ),
            DesktopWorkflowStep(
                action_type="open_url",
                selector="browser-address-bar",
                value=target,
                description="Navigate to the requested URL or search query.",
                metadata={
                    "handoff_write": {
                        "active_url": target,
                        "search_query": context.intent.entities.get("query", ""),
                    }
                },
            ),
        ]

    @staticmethod
    def _needs_browser(context: WorkflowContext) -> bool:
        if context.intent.prefers_local_file_search() or context.intent.prefers_chat_search():
            return False
        if context.intent.prefers_web_search():
            return True
        lower = context.lower
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

    def _browser_target(self, context: WorkflowContext) -> str | None:
        lower = context.lower
        match = re.search(r"https?://\S+", lower)
        if match is not None:
            return match.group(0).rstrip(".,)")
        query = context.intent.entities.get("query") or self._search_query(lower)
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
        query = (
            context.intent.entities.get("query")
            or BrowserWorkflowAdapter._search_query(context.lower)
        )
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
        if context.intent.copy_semantics not in {"none", "file"}:
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


class CrossAppHandoffWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if not context.intent.cross_app:
            return []
        if not context.intent.is_clipboard_copy():
            return []
        if context.intent.destination_surface != "editor":
            return []
        query = context.intent.entities.get("query") or context.intent.object_hint
        clipboard_text = self._clipboard_text(context)
        steps: list[DesktopWorkflowStep] = []
        if context.intent.source_surface == "browser":
            browser_target = context.intent.source_app_target or "msedge.exe"
            steps.append(
                DesktopWorkflowStep(
                    action_type="launch_app",
                    selector=browser_target,
                    value=browser_target,
                    description="Launch the browser source surface.",
                )
            )
            if context.intent.prefers_web_search() and query:
                encoded = query.replace(" ", "+")
                target = f"https://www.bing.com/search?q={encoded}"
                steps.append(
                    DesktopWorkflowStep(
                        action_type="open_url",
                        selector="browser-address-bar",
                        value=target,
                        description="Open the browser search result before handoff.",
                        metadata={
                            "handoff_write": {
                                "active_url": target,
                                "search_query": query,
                            }
                        },
                    )
                )
        steps.append(
            DesktopWorkflowStep(
                action_type="set_clipboard",
                selector="workflow-clipboard",
                value=clipboard_text,
                description="Transfer the semantic result into the clipboard channel.",
                metadata={
                    "control_channel": "clipboard",
                    "handoff_write": {"copied_text": clipboard_text},
                    "verification_contract": {
                        "kind": "clipboard_contains",
                        "expected": "Clipboard contains the transferred semantic entity.",
                        "value": clipboard_text,
                        "required": True,
                    },
                },
            )
        )
        editor_target = context.intent.destination_app_target or "notepad.exe"
        steps.append(
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=editor_target,
                value=editor_target,
                description="Launch the destination editor surface.",
            )
        )
        steps.append(
            DesktopWorkflowStep(
                action_type="type",
                selector="document-canvas",
                value="{{copied_text}}",
                description="Write the transferred entity into the destination editor.",
                metadata={
                    "handoff_read": ["copied_text"],
                    "control_channel": "clipboard",
                    "verification_contract": {
                        "kind": "field_contains",
                        "expected": "Editor contains the transferred clipboard entity.",
                        "target": "document-canvas",
                        "value": clipboard_text,
                        "required": True,
                    },
                },
            )
        )
        return steps

    @staticmethod
    def _clipboard_text(context: WorkflowContext) -> str:
        objective_lower = context.objective.lower()
        if any(
            token in objective_lower
            for token in (
                "copy the url",
                "copy url",
                "copy the link",
                "copy link",
                "paste the url",
                "paste url",
                "web address",
                "address bar",
            )
        ):
            return "{{active_url}}"
        return (
            str(context.intent.entities.get("clipboard_text") or "").strip()
            or str(context.intent.entities.get("editor_text") or "").strip()
            or context.objective.strip().rstrip(".")
        )


class ChatWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.intent.source_surface != "chat_app":
            return []
        query = context.intent.entities.get("query") or context.intent.object_hint
        reply_text = context.intent.entities.get("clipboard_text") or (
            f"Draft summary for: {query}".strip()
        )
        target = context.intent.source_app_target or context.app_target or "teams.exe"
        steps = [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=target,
                value=target,
                description="Launch the chat surface.",
            )
        ]

        if context.intent.prefers_chat_search() and query:
            steps.append(
                DesktopWorkflowStep(
                    action_type="type",
                    selector="conversation-list",
                    value=f"Search messages: {query}",
                    description="Search the current chat history semantically.",
                    metadata={"handoff_write": {"message_query": query}},
                )
            )
        if "draft_reply" in context.intent.operations:
            steps.append(
                DesktopWorkflowStep(
                    action_type="type",
                    selector="chat-composer",
                    value=reply_text,
                    description="Draft the reply in the chat composer without forcing send.",
                    metadata={
                        "handoff_write": {"draft_reply": reply_text},
                        "verification_contract": {
                            "kind": "field_contains",
                            "expected": "The draft reply is present in the chat composer.",
                            "target": "chat-composer",
                            "value": reply_text,
                            "required": True,
                        },
                    },
                )
            )
        return steps


class EmailWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.intent.source_surface != "email" and context.intent.destination_surface != "email":
            return []
        steps: list[DesktopWorkflowStep] = []
        source_target = context.intent.source_app_target or "outlook.exe"
        destination_target = context.intent.destination_app_target or "outlook.exe"
        query = context.intent.entities.get("email_query") or context.intent.entities.get("query") or context.intent.object_hint
        recipient = context.intent.entities.get("recipient") or ""
        file_source = context.intent.entities.get("file_source") or context.intent.file_source_hint
        if context.intent.source_surface == "file_explorer" and file_source:
            steps.extend(
                [
                    DesktopWorkflowStep(
                        action_type="launch_app",
                        selector="explorer.exe",
                        value="explorer.exe",
                        description="Open File Explorer to locate the requested attachment.",
                    ),
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="explorer-file-list",
                        value=file_source,
                        description="Locate the attachment candidate in File Explorer.",
                        metadata={
                            "handoff_write": {"file_source": file_source},
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": "Explorer reflects the requested attachment search or selection.",
                                "target": "explorer-file-list",
                                "value": file_source,
                                "required": True,
                            },
                        },
                    ),
                ]
            )
        if context.intent.source_surface == "email":
            steps.append(
                DesktopWorkflowStep(
                    action_type="launch_app",
                    selector=source_target,
                    value=source_target,
                    description="Launch the email surface.",
                )
            )
            if "search_email" in context.intent.operations and query:
                steps.append(
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="email-search-box",
                        value=query,
                        description="Search the inbox for the requested message or invite.",
                        metadata={
                            "handoff_write": {"email_query": query},
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": "The email search box contains the requested query.",
                                "target": "email-search-box",
                                "value": query,
                                "required": True,
                            },
                        },
                    )
                )
        if context.intent.destination_surface == "email":
            steps.append(
                DesktopWorkflowStep(
                    action_type="launch_app",
                    selector=destination_target,
                    value=destination_target,
                    description="Launch the destination email draft surface.",
                )
            )
            if recipient:
                steps.append(
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="email-to-field",
                        value=recipient,
                        description="Address the draft to the requested recipient.",
                        metadata={
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": "The email recipient field contains the requested recipient.",
                                "target": "email-to-field",
                                "value": recipient,
                                "required": True,
                            },
                        },
                    )
                )
            if "attach_file" in context.intent.operations and file_source:
                steps.append(
                    DesktopWorkflowStep(
                        action_type="type",
                        selector="email-attachment-field",
                        value=file_source,
                        description="Attach the requested local file to the draft.",
                        metadata={
                            "verification_contract": {
                                "kind": "field_contains",
                                "expected": "The email attachment field reflects the requested file.",
                                "target": "email-attachment-field",
                                "value": file_source,
                                "required": True,
                            },
                        },
                    )
                )
        return steps


class CalendarWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.intent.source_surface != "calendar" and context.intent.destination_surface != "calendar":
            return []
        target = (
            context.intent.destination_app_target
            or context.intent.source_app_target
            or "outlook.exe /select outlook:calendar"
        )
        event_title = context.intent.entities.get("event_title") or context.intent.object_hint or context.objective.strip().rstrip(".")
        steps = [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=target,
                value=target,
                description="Launch the calendar surface.",
            )
        ]
        if "create_calendar_event" in context.intent.operations and event_title:
            steps.append(
                DesktopWorkflowStep(
                    action_type="type",
                    selector="calendar-event-editor",
                    value=event_title,
                    description="Draft the requested calendar event details.",
                    metadata={
                        "verification_contract": {
                            "kind": "field_contains",
                            "expected": "The calendar event editor contains the drafted event details.",
                            "target": "calendar-event-editor",
                            "value": event_title,
                            "required": True,
                        },
                    },
                )
            )
        return steps


class SettingsWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.intent.source_surface != "settings" and context.app_target != "settings.exe":
            return []
        setting_name = context.intent.entities.get("setting_name") or context.intent.object_hint or context.objective.strip().rstrip(".")
        setting_value = context.intent.entities.get("setting_value") or ""
        steps: list[DesktopWorkflowStep] = []
        if setting_name:
            steps.append(
                DesktopWorkflowStep(
                    action_type="type",
                    selector="settings-search-box",
                    value=setting_name,
                    description="Search Settings for the requested control.",
                    metadata={
                        "verification_contract": {
                            "kind": "field_contains",
                            "expected": "The Settings search box contains the requested control name.",
                            "target": "settings-search-box",
                            "value": setting_name,
                            "required": True,
                        },
                    },
                )
            )
        if "toggle_setting" in context.intent.operations and setting_value:
            steps.append(
                DesktopWorkflowStep(
                    action_type="click",
                    selector="settings-toggle",
                    value=f"{setting_name}:{setting_value}",
                    description="Toggle the requested system setting to the desired state.",
                    metadata={
                        "verification_contract": {
                            "kind": "state_changed",
                            "expected": "The requested Settings control changes state.",
                            "target": "settings-toggle",
                            "required": True,
                        },
                    },
                )
            )
        return steps


class PdfWorkflowAdapter:
    def steps_for(self, context: WorkflowContext) -> list[DesktopWorkflowStep]:
        if context.intent.source_surface != "pdf_viewer":
            return []
        if context.intent.destination_surface != "editor":
            return []
        query = context.intent.entities.get("query") or context.intent.object_hint
        extracted = f"PDF note: {query}".strip()
        return [
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=context.intent.source_app_target or "acrobat.exe",
                value=context.intent.source_app_target or "acrobat.exe",
                description="Launch the PDF surface.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="pdf-search-box",
                value=query,
                description="Search within the PDF for the requested content.",
            ),
            DesktopWorkflowStep(
                action_type="set_clipboard",
                selector="workflow-clipboard",
                value=extracted,
                description="Store the extracted PDF note on the clipboard.",
                metadata={
                    "control_channel": "clipboard",
                    "handoff_write": {"copied_text": extracted},
                },
            ),
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=context.intent.destination_app_target or "notepad.exe",
                value=context.intent.destination_app_target or "notepad.exe",
                description="Launch the destination editor.",
            ),
            DesktopWorkflowStep(
                action_type="type",
                selector="document-canvas",
                value="{{copied_text}}",
                description="Write the extracted PDF note into the editor.",
                metadata={"handoff_read": ["copied_text"]},
            ),
        ]
