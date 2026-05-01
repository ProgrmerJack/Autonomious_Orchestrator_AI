from __future__ import annotations

import re

from agentos_orchestrator.os_control.workflow.adapters import (
    BrowserWorkflowAdapter,
    EditorWorkflowAdapter,
    ExplorerFileOpsWorkflowAdapter,
    FileOpsIntentAdapter,
    FileManagerWorkflowAdapter,
    GenericAppWorkflowAdapter,
    OfficeWorkflowAdapter,
    SpreadsheetCellEditIntentAdapter,
    SpreadsheetWorkflowAdapter,
    WorkflowContext,
)
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
    WorkflowArtifact,
)


class DesktopWorkflowPlanner:
    MODE_KEYWORDS = {
        "presentation": ("presentation", "slides", "powerpoint"),
        "drawing": ("draw", "diagram", "sketch", "paint"),
        "report": ("report", "document", "doc", "writeup"),
        "script": ("script", ".py", "python file", "automation"),
        "spreadsheet": ("spreadsheet", "excel", "table", "sheet"),
    }

    APP_TARGET_KEYWORDS = (
        ("file explorer", "explorer.exe"),
        ("explorer", "explorer.exe"),
        ("powerpoint", "powerpnt.exe"),
        ("slides", "powerpnt.exe"),
        ("word", "winword.exe"),
        ("document", "winword.exe"),
        ("report", "winword.exe"),
        ("excel", "excel.exe"),
        ("spreadsheet", "excel.exe"),
        ("table", "excel.exe"),
        ("paint", "mspaint.exe"),
        ("drawing", "mspaint.exe"),
        ("sketch", "mspaint.exe"),
        ("notepad", "notepad.exe"),
        ("vscode", "code"),
        ("visual studio code", "code"),
        ("code editor", "code"),
        ("browser", "msedge.exe"),
        ("edge", "msedge.exe"),
        ("chrome", "chrome.exe"),
    )

    DEFAULT_MODE_TARGETS = {
        "presentation": "powerpnt.exe",
        "drawing": "mspaint.exe",
        "script": "code",
        "report": "winword.exe",
        "spreadsheet": "excel.exe",
    }

    def __init__(self) -> None:
        self.browser_adapter = BrowserWorkflowAdapter()
        self.file_manager_adapter = FileManagerWorkflowAdapter()
        self.explorer_file_ops_adapter = ExplorerFileOpsWorkflowAdapter()
        self.file_ops_intent_adapter = FileOpsIntentAdapter()
        self.generic_app_adapter = GenericAppWorkflowAdapter()
        self.office_adapter = OfficeWorkflowAdapter()
        self.editor_adapter = EditorWorkflowAdapter()
        self.spreadsheet_adapter = SpreadsheetWorkflowAdapter()
        self.spreadsheet_cell_edit_intent_adapter = SpreadsheetCellEditIntentAdapter()

    def plan(self, objective: str) -> DesktopWorkflowPlan:
        cleaned = re.sub(r"\s+", " ", objective).strip()
        sub_tasks = self._sub_tasks(cleaned)
        segments = self._task_segments(cleaned, sub_tasks)
        mode = self._plan_mode(segments)
        app_target = self._plan_app_target(segments)
        artifacts = [artifact for segment in segments for artifact in segment[3]]
        lower = cleaned.lower()
        requires_clarification = self._needs_clarification(cleaned, lower)
        clarification_questions = self._clarification_questions(
            cleaned,
            mode,
            requires_clarification,
        )
        steps = self._build_steps(segments)
        summary = self._summary(mode, app_target, artifacts, steps, sub_tasks)
        risks, notes = self._plan_guidance()
        return DesktopWorkflowPlan(
            objective=cleaned,
            mode=mode,
            app_target=app_target,
            summary=summary,
            steps=steps,
            artifacts=artifacts,
            risks=risks,
            notes=notes,
            sub_tasks=sub_tasks,
            requires_clarification=requires_clarification,
            clarification_questions=clarification_questions,
        )

    def _build_steps(
        self,
        segments: list[tuple[str, str, str | None, list[WorkflowArtifact]]],
    ) -> list[DesktopWorkflowStep]:
        steps: list[DesktopWorkflowStep] = []
        for item, segment_mode, target, segment_artifacts in segments:
            steps.extend(
                self._segment_steps(
                    item,
                    segment_mode,
                    target,
                    segment_artifacts,
                )
            )
        return steps

    @staticmethod
    def _plan_guidance() -> tuple[list[str], list[str]]:
        risks = [
            "Live desktop actions still require approval before execution.",
            "App-specific selectors may need refinement on first run.",
        ]
        notes = [
            "Workflow artifacts are created under artifacts/workflows/.",
            "Prefer the virtual desktop sandbox for safe dry-runs.",
        ]
        return risks, notes

    def _segment_steps(
        self,
        objective: str,
        mode: str,
        app_target: str | None,
        artifacts: list[WorkflowArtifact],
    ) -> list[DesktopWorkflowStep]:
        segment_steps: list[DesktopWorkflowStep] = []
        lower = objective.lower()
        artifact_path = None
        if artifacts:
            artifact_path = artifacts[0].path.replace("/", "\\")
        context = WorkflowContext(
            objective=objective,
            lower=lower,
            mode=mode,
            app_target=app_target,
            artifact_path=artifact_path,
        )
        segment_steps.extend(self.browser_adapter.steps_for(context))
        if app_target:
            self._append_launch_step(segment_steps, app_target)
        segment_steps.extend(self.file_manager_adapter.steps_for(context))
        segment_steps.extend(self.explorer_file_ops_adapter.steps_for(context))
        segment_steps.extend(self.file_ops_intent_adapter.steps_for(context))
        segment_steps.extend(self.spreadsheet_adapter.steps_for(context))
        segment_steps.extend(
            self.spreadsheet_cell_edit_intent_adapter.steps_for(context)
        )
        segment_steps.extend(self.generic_app_adapter.steps_for(context))
        before_mode_specific = len(segment_steps)
        segment_steps.extend(self.office_adapter.steps_for(context))
        segment_steps.extend(self.editor_adapter.steps_for(context))
        if mode == "drawing" and artifact_path:
            segment_steps.extend(self._drawing_steps(objective, artifact_path))
        mode_delta = len(segment_steps) - before_mode_specific
        if mode_delta == 0 and app_target and self._needs_focus_step(lower):
            segment_steps.append(self._focus_step())
        return segment_steps

    def _task_segments(
        self,
        objective: str,
        sub_tasks: list[str],
    ) -> list[tuple[str, str, str | None, list[WorkflowArtifact]]]:
        if not sub_tasks:
            sub_tasks = [objective]
        segments: list[tuple[str, str, str | None, list[WorkflowArtifact]]] = []
        for item in sub_tasks:
            lower = item.lower()
            mode = self._segment_mode(lower)
            app_target = self._segment_app_target(lower, mode)
            artifacts = self._artifacts(item, lower, mode)
            segments.append((item, mode, app_target, artifacts))
        return segments

    @staticmethod
    def _plan_mode(
        segments: list[tuple[str, str, str | None, list[WorkflowArtifact]]],
    ) -> str:
        modes = {segment[1] for segment in segments}
        if not modes:
            return "app-task"
        if "app-task" in modes and len(modes) > 1:
            modes.remove("app-task")
        if len(modes) == 1:
            return next(iter(modes))
        return "multi-app"

    @staticmethod
    def _plan_app_target(
        segments: list[tuple[str, str, str | None, list[WorkflowArtifact]]],
    ) -> str | None:
        targets = [segment[2] for segment in segments if segment[2]]
        if not targets:
            return None
        if len(set(targets)) == 1:
            return targets[0]
        return None

    @staticmethod
    def _append_launch_step(
        steps: list[DesktopWorkflowStep],
        app_target: str,
    ) -> None:
        if any(
            step.action_type == "launch_app" and step.value == app_target
            for step in steps
        ):
            return
        steps.append(
            DesktopWorkflowStep(
                action_type="launch_app",
                selector=app_target,
                value=app_target,
                description=f"Launch {app_target} for the requested workflow.",
            )
        )

    @staticmethod
    def _drawing_steps(
        objective: str,
        artifact_path: str,
    ) -> list[DesktopWorkflowStep]:
        title = objective.strip().rstrip(".")
        path_values = ["M 40 180 L 120 100 L 200 180 Z"]
        draw_steps = [
            DesktopWorkflowStep(
                action_type="draw_path",
                selector="name=Drawing Canvas",
                value=path,
                description="Draw a starter stroke sequence.",
            )
            for path in path_values
        ]
        return [
            DesktopWorkflowStep(
                action_type="type",
                selector="name=Drawing Canvas",
                value=(f"Concept: {title}\nReference sketch saved to {artifact_path}"),
                description=("Reference the generated concept in the drawing app."),
            ),
            *draw_steps,
            DesktopWorkflowStep(
                action_type="hotkey",
                selector="app-window",
                value="^s",
                description="Save the drawing draft.",
            ),
        ]

    def _artifacts(
        self,
        objective: str,
        lower: str,
        mode: str,
    ) -> list[WorkflowArtifact]:
        slug = self._slug(objective)
        base = f"artifacts/workflows/{slug}"
        artifact = self._artifact_for_mode(base, lower, mode)
        if artifact is None:
            return []
        return [artifact]

    def _artifact_for_mode(
        self,
        base: str,
        lower: str,
        mode: str,
    ) -> WorkflowArtifact | None:
        if mode == "presentation":
            return WorkflowArtifact(
                path=f"{base}/presentation_outline.md",
                kind="presentation-outline",
                description="Slide-by-slide outline generated from objective.",
            )
        if mode == "drawing":
            return WorkflowArtifact(
                path=f"{base}/drawing_concept.svg",
                kind="drawing-concept",
                description="Starter SVG concept for the drawing app.",
            )
        if mode == "report":
            return WorkflowArtifact(
                path=f"{base}/report.md",
                kind="report",
                description="Draft report with summary and next steps.",
            )
        if mode == "script":
            file_name = self._script_name(lower)
            return WorkflowArtifact(
                path=f"{base}/{file_name}",
                kind="script",
                description="Starter Python script generated from objective.",
            )
        return None

    @staticmethod
    def _segment_mode(lower: str) -> str:
        if DesktopWorkflowPlanner._is_explorer_file_op(lower):
            return "app-task"
        if DesktopWorkflowPlanner._is_spreadsheet_cell_intent(lower):
            return "spreadsheet"
        for mode, terms in DesktopWorkflowPlanner.MODE_KEYWORDS.items():
            if any(term in lower for term in terms):
                return mode
        return "app-task"

    @staticmethod
    def _is_explorer_file_op(lower: str) -> bool:
        if not any(token in lower for token in ("copy ", "move ", "rename ")):
            return False
        return "explorer" in lower or "file explorer" in lower

    @staticmethod
    def _is_spreadsheet_cell_intent(lower: str) -> bool:
        has_cell_ref = re.search(r"\b[a-z]{1,3}[0-9]{1,5}\b", lower) is not None
        if not has_cell_ref:
            return False
        return any(token in lower for token in ("set", "update", "cell", "sheet"))

    @staticmethod
    def _segment_app_target(lower: str, mode: str) -> str | None:
        explicit = DesktopWorkflowPlanner._explicit_app_target(lower)
        if explicit is not None:
            return explicit
        inferred = DesktopWorkflowPlanner._inferred_app_target(lower)
        if inferred is not None:
            return inferred
        if mode == "app-task" and DesktopWorkflowPlanner._is_knowledge_task(lower):
            return "msedge.exe"
        return DesktopWorkflowPlanner.DEFAULT_MODE_TARGETS.get(mode)

    @staticmethod
    def _is_knowledge_task(lower: str) -> bool:
        cues = (
            "search",
            "look up",
            "research",
            "find ",
            "analyze",
            "analyse",
            "compare",
            "investigate",
            "stock",
            "ticker",
            "market",
            "price",
            "news",
        )
        return any(cue in lower for cue in cues)

    @staticmethod
    def _inferred_app_target(lower: str) -> str | None:
        match = re.search(
            (
                r"\b(?:open|launch|start)\s+"
                r"(?P<name>[a-z0-9][a-z0-9 ._-]{1,40}?)"
                r"(?:\s+(?:app|application|program|tool))?"
                r"(?:\s+(?:and|then|to)\b|$)"
            ),
            lower,
            flags=re.I,
        )
        if match is None:
            return None
        name = match.group("name").strip().lower()
        if not name:
            return None
        reject = {
            "website",
            "web",
            "url",
            "task",
            "file",
            "folder",
            "report",
            "document",
            "script",
            "spreadsheet",
            "presentation",
            "slides",
            "drawing",
        }
        if name in reject:
            return None
        aliases = {
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "paint": "mspaint.exe",
            "notepad": "notepad.exe",
            "file explorer": "explorer.exe",
            "vscode": "code",
            "visual studio code": "code",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "edge": "msedge.exe",
            "chrome": "chrome.exe",
        }
        if name in aliases:
            return aliases[name]
        if name.endswith(".exe"):
            return name
        slug = re.sub(r"[^a-z0-9]+", "", name)
        if not slug:
            return None
        return f"{slug}.exe"

    @staticmethod
    def _explicit_app_target(lower: str) -> str | None:
        for key, value in DesktopWorkflowPlanner.APP_TARGET_KEYWORDS:
            if key in lower:
                return value
        return None

    @staticmethod
    def _sub_tasks(objective: str) -> list[str]:
        normalized = objective
        normalized = re.sub(r"\bthen\b", " and ", normalized, flags=re.I)
        normalized = re.sub(r"\bnext\b", " and ", normalized, flags=re.I)
        parts = [
            item.strip(" .")
            for item in re.split(r"\s+and\s+|\s*;\s*|\s*,\s*", normalized)
            if item.strip(" .")
        ]
        if len(parts) <= 1:
            return [objective.strip()] if objective.strip() else []
        return parts[:8]

    @staticmethod
    def _needs_clarification(cleaned: str, lower: str) -> bool:
        if not cleaned:
            return True
        if len(cleaned.split()) < 4:
            return True
        vague_markers = (
            "do it",
            "something",
            "anything",
            "whatever",
            "handle this",
            "make it better",
            "fix this",
        )
        return any(marker in lower for marker in vague_markers)

    @staticmethod
    def _clarification_questions(
        objective: str,
        mode: str,
        required: bool,
    ) -> list[str]:
        if not required:
            return []
        questions = [
            "What is the exact final deliverable you want?",
            "Which app should be used for this task?",
        ]
        if mode in {"report", "presentation", "script"}:
            questions.append(
                "Do you want me to create a new file or edit an existing one?"
            )
        if "research" in objective.lower():
            questions.append(
                "Should the browser step run first before drafting output?"
            )
        return questions

    @staticmethod
    def _needs_focus_step(lower: str) -> bool:
        return "search for" in lower or "open " in lower

    @staticmethod
    def _focus_step() -> DesktopWorkflowStep:
        return DesktopWorkflowStep(
            action_type="focus",
            selector="name=AgentOS",
            description="Focus the working window before manual refinement.",
        )

    @staticmethod
    def _summary(
        mode: str,
        app_target: str | None,
        artifacts: list[WorkflowArtifact],
        steps: list[DesktopWorkflowStep],
        sub_tasks: list[str],
    ) -> str:
        artifact_count = len(artifacts)
        app_text = app_target or "desktop workspace"
        app_count = sum(1 for step in steps if step.action_type == "launch_app")
        sub_count = len(sub_tasks)
        return (
            f"Planned a {mode} workflow targeting {app_text} with "
            f"{artifact_count} artifact(s), {app_count} app launch step(s), "
            f"and {sub_count} inferred sub-task(s)."
        )

    @staticmethod
    def _slug(value: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return (base or "workflow")[:48]

    @staticmethod
    def _script_name(lower: str) -> str:
        match = re.search(r"([a-zA-Z0-9_\-]+\.py)", lower)
        if match is not None:
            return match.group(1)
        return "generated_script.py"
