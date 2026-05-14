from __future__ import annotations

import re

from agentos_orchestrator.os_control.workflow.adapters import (
    ApiIntentWorkflowAdapter,
    BrowserWorkflowAdapter,
    EditorWorkflowAdapter,
    ExplorerFileOpsWorkflowAdapter,
    FileOpsIntentAdapter,
    FileManagerWorkflowAdapter,
    GenericAppWorkflowAdapter,
    OfficeWorkflowAdapter,
    ResearchWorkflowAdapter,
    SpreadsheetCellEditIntentAdapter,
    SpreadsheetWorkflowAdapter,
    WorkflowContext,
    workflow_prefers_research_tool,
)
from agentos_orchestrator.os_control.workflow.models import (
    DesktopWorkflowPlan,
    DesktopWorkflowStep,
    WorkflowArtifact,
)
from agentos_orchestrator.os_control.workflow.programmer import (
    build_programmer_tool_step,
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

    CLARITY_VAGUE_MARKERS = (
        "do it",
        "something",
        "anything",
        "whatever",
        "handle this",
        "make it better",
        "fix this",
    )
    ACTION_SIGNALS = (
        "open",
        "launch",
        "search",
        "find",
        "write",
        "create",
        "edit",
        "save",
        "move",
        "copy",
        "delete",
        "rename",
        "download",
        "upload",
        "run",
        "execute",
    )
    TARGET_SIGNALS = (
        "file",
        "folder",
        "document",
        "report",
        "script",
        "browser",
        "website",
        "url",
        "app",
        "excel",
        "word",
        "powerpoint",
        "vscode",
        "notepad",
        "paint",
        "explorer",
    )

    def __init__(self) -> None:
        self.api_intent_adapter = ApiIntentWorkflowAdapter()
        self.browser_adapter = BrowserWorkflowAdapter()
        self.research_adapter = ResearchWorkflowAdapter()
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
        risks, notes = self._plan_guidance()
        plan = DesktopWorkflowPlan(
            objective=cleaned,
            mode=mode,
            app_target=app_target,
            summary="",
            steps=steps,
            artifacts=artifacts,
            risks=risks,
            notes=notes,
            sub_tasks=sub_tasks,
            requires_clarification=requires_clarification,
            clarification_questions=clarification_questions,
        )
        programmer_step = build_programmer_tool_step(plan)
        if programmer_step is not None:
            insert_at = 0
            while (
                insert_at < len(plan.steps)
                and plan.steps[insert_at].action_type == "tool"
                and plan.steps[insert_at].selector == "tool_executor:workflow_research"
            ):
                insert_at += 1
            plan.steps = [
                *plan.steps[:insert_at],
                programmer_step,
                *plan.steps[insert_at:],
            ]
        plan.summary = self._summary(
            mode,
            app_target,
            artifacts,
            plan.steps,
            sub_tasks,
        )
        return plan

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
        research_artifact_path = None
        for artifact in artifacts:
            normalized_path = artifact.path.replace("/", "\\")
            if artifact.kind == "research-brief":
                if research_artifact_path is None:
                    research_artifact_path = normalized_path
                continue
            if artifact_path is None:
                artifact_path = normalized_path
        if artifact_path is None:
            artifact_path = research_artifact_path
        research_preferred = bool(
            research_artifact_path
            and research_artifact_path.lower().endswith("research_brief.md")
            and workflow_prefers_research_tool(lower, mode)
        )
        effective_app_target = app_target
        if research_preferred and app_target in {"msedge.exe", "chrome.exe"}:
            effective_app_target = None
        context = WorkflowContext(
            objective=objective,
            lower=lower,
            mode=mode,
            app_target=effective_app_target,
            artifact_path=artifact_path,
            research_artifact_path=research_artifact_path,
        )
        research_steps = self.research_adapter.steps_for(context)
        if research_steps:
            segment_steps.extend(research_steps)
        api_steps = self.api_intent_adapter.steps_for(context)
        if api_steps:
            segment_steps.extend(api_steps)
        elif not research_steps:
            segment_steps.extend(self.browser_adapter.steps_for(context))
        if effective_app_target and not api_steps:
            self._append_launch_step(segment_steps, effective_app_target)
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
        artifacts: list[WorkflowArtifact] = []
        if workflow_prefers_research_tool(lower, mode):
            artifacts.append(
                WorkflowArtifact(
                    path=f"{base}/research_brief.md",
                    kind="research-brief",
                    description=(
                        "Provider-backed research brief gathered before any "
                        "browser-first UI handoff."
                    ),
                )
            )
        artifact = self._artifact_for_mode(base, lower, mode)
        if artifact is not None:
            artifacts.append(artifact)
        return artifacts

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
        mode_scores: dict[str, int] = {}
        for mode, terms in DesktopWorkflowPlanner.MODE_KEYWORDS.items():
            mode_scores[mode] = sum(1 for term in terms if term in lower)
        best_mode = max(mode_scores, key=mode_scores.get) if mode_scores else "app-task"
        if mode_scores.get(best_mode, 0) > 0:
            return best_mode
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
        if DesktopWorkflowPlanner._is_api_task(lower):
            return None
        if mode == "app-task" and workflow_prefers_research_tool(lower, mode):
            return None
        if mode == "app-task" and DesktopWorkflowPlanner._is_knowledge_task(lower):
            return "msedge.exe"
        return DesktopWorkflowPlanner.DEFAULT_MODE_TARGETS.get(mode)

    @staticmethod
    def _is_api_task(lower: str) -> bool:
        if re.search(
            r"https?://\S*(?:/api|graphql|openapi|swagger|\.json)\S*",
            lower,
        ):
            return True
        if re.search(
            r"(?:localhost|127\.0\.0\.1)(?::\d{1,5})?(?:/\S*)?",
            lower,
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
            )
        )

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
        explicit_app_keys = {
            "file explorer",
            "explorer",
            "powerpoint",
            "slides",
            "word",
            "excel",
            "paint",
            "notepad",
            "vscode",
            "visual studio code",
            "code editor",
            "browser",
            "edge",
            "chrome",
        }
        explicit_scored: list[tuple[int, int, str]] = []
        fallback_scored: list[tuple[int, int, str]] = []
        for key, value in DesktopWorkflowPlanner.APP_TARGET_KEYWORDS:
            if key not in lower:
                continue
            item = (lower.count(key), len(key), value)
            if key in explicit_app_keys:
                explicit_scored.append(item)
            else:
                fallback_scored.append(item)
        if explicit_scored:
            explicit_scored.sort(reverse=True)
            return explicit_scored[0][2]
        if not fallback_scored:
            return None
        fallback_scored.sort(reverse=True)
        return fallback_scored[0][2]

    @staticmethod
    def _sub_tasks(objective: str) -> list[str]:
        normalized = objective
        normalized = re.sub(r"\bthen\b", " and ", normalized, flags=re.I)
        normalized = re.sub(r"\bnext\b", " and ", normalized, flags=re.I)
        raw_parts = [
            item.strip(" .")
            for item in re.split(r"\s+and\s+|\s*;\s*|\s*,\s*", normalized)
            if item.strip(" .")
        ]
        parts: list[str] = []
        for item in raw_parts:
            if parts and DesktopWorkflowPlanner._should_merge_continuation(
                parts[-1], item
            ):
                parts[-1] = f"{parts[-1]} and {item}"
                continue
            parts.append(item)
        if len(parts) <= 1:
            return [objective.strip()] if objective.strip() else []
        return parts[:8]

    @staticmethod
    def _should_merge_continuation(previous: str, current: str) -> bool:
        previous_lower = previous.lower().strip()
        current_lower = current.lower().strip()
        if not previous_lower or not current_lower:
            return False
        if not DesktopWorkflowPlanner._is_knowledge_task(previous_lower):
            return False
        continuation_patterns = (
            r"^(?:analy(?:z|s)e|compare|investigate|summari(?:z|s)e)\s+(?:it|them|that|those|this)\b",
            r"^(?:analy(?:z|s)e|compare|investigate|summari(?:z|s)e)\b$",
        )
        return any(
            re.search(pattern, current_lower) is not None
            for pattern in continuation_patterns
        )

    @staticmethod
    def _needs_clarification(cleaned: str, lower: str) -> bool:
        if not cleaned:
            return True
        if len(cleaned.split()) < 4:
            return True
        if any(
            marker in lower for marker in DesktopWorkflowPlanner.CLARITY_VAGUE_MARKERS
        ):
            return True

        action_hits = sum(
            1 for token in DesktopWorkflowPlanner.ACTION_SIGNALS if token in lower
        )
        target_hits = sum(
            1 for token in DesktopWorkflowPlanner.TARGET_SIGNALS if token in lower
        )
        unique_terms = {
            token
            for token in re.findall(r"[a-z0-9_\-]+", lower)
            if len(token) > 2
            and token not in {"the", "and", "for", "with", "then", "from", "into"}
        }
        clarity_score = (
            action_hits * 0.45 + target_hits * 0.4 + min(len(unique_terms), 12) * 0.03
        )
        return clarity_score < 0.55

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
