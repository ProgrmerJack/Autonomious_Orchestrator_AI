from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

from agentos_orchestrator.core.approvals import ApprovalRequired
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.os_control import (
    VirtualDesktopSandboxBackend,
    WindowsUiaBackend,
)
from agentos_orchestrator.os_control.workflow import (
    DesktopWorkflowService,
)
from agentos_orchestrator.product import CommandRegistry

from .channels import ChannelMessage


@dataclass(slots=True)
class ChannelResponse:
    status: str
    text: str
    payload: dict = field(default_factory=dict)


class GatewayCommandRouter:
    """OpenClaw-style channel command router for local AgentOS runs."""

    # Action verbs that suggest a desktop/browser task is needed.
    # Configurable: override on a subclass or instance to extend the set.
    DESKTOP_ACTION_MARKERS: tuple[str, ...] = (
        "open ",
        "launch ",
        "start ",
        "search for",
        "look up",
        "browse ",
        "navigate ",
        "go to",
        "click ",
        "type ",
        "fill ",
        "write ",
        "draft ",
        "edit ",
        "save ",
        "create ",
        "rename ",
        "move ",
        "copy ",
        "download ",
        "upload ",
        "submit ",
        "inspect ",
    )
    # Surface nouns/contexts that confirm a desktop intent.
    DESKTOP_SURFACE_MARKERS: tuple[str, ...] = (
        "browser",
        "desktop",
        "window",
        "website",
        "web page",
        "url",
        "app",
        "application",
        "file explorer",
        "explorer",
        "folder",
        "file",
        "document",
        "report",
        "spreadsheet",
        "sheet",
        "slides",
        "presentation",
        "script",
        "notepad",
        "word",
        "excel",
        "powerpoint",
        "paint",
        "vscode",
        "code editor",
        "chrome",
        "edge",
    )
    # Markers that indicate a pure research intent (not desktop).
    RESEARCH_ONLY_MARKERS: tuple[str, ...] = (
        "paper",
        "papers",
        "literature",
        "citation",
        "citations",
        "theorem",
        "proof",
        "survey",
        "sources",
    )
    # Markers that indicate a vague/ambiguous objective needing clarification.
    VAGUE_MARKERS: tuple[str, ...] = (
        "do it",
        "something",
        "anything",
        "whatever",
        "handle this",
        "make it better",
        "fix this",
    )

    def __init__(
        self,
        orchestrator: ResearchOrchestrator,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.command_registry = command_registry or CommandRegistry()
        self.workflow_service = DesktopWorkflowService(Path.cwd())

    def handle(self, message: ChannelMessage) -> ChannelResponse:
        text = message.text.strip()
        command, _, argument = text.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"/run", "run"}:
            return self._run(argument)
        if command in {"/resume", "resume"}:
            return self._resume(argument)
        if command in {"/approve", "approve"}:
            return self._approve(argument)
        if command in {"/deny", "deny"}:
            return self._deny(argument)
        if command in {"/pc", "pc", "/desktop", "desktop"}:
            return self._desktop_task(argument)
        workflow = self.command_registry.get(command)
        if workflow is not None:
            return self._run(workflow.objective_for(argument))
        if text:
            return self._run(text)
        return ChannelResponse("ignored", "No command text was provided.")

    def _run(self, objective: str) -> ChannelResponse:
        if not objective:
            return ChannelResponse(
                "error",
                "A research objective is required.",
            )
        if self._should_route_to_desktop(objective):
            return self._desktop_task(objective)
        if self._needs_clarification(objective):
            return ChannelResponse(
                "clarification_required",
                "Clarification required: what exact output do you want?",
                {
                    "questions": [
                        ("What exact output format do you want (brief/report/plan)?"),
                        "What sources should be prioritized?",
                    ]
                },
            )
        try:
            report = self.orchestrator.run(objective)
        except ApprovalRequired as exc:
            return ChannelResponse(
                "approval_required",
                f"Approval required: {exc.ticket.approval_id}",
                {"approval": asdict(exc.ticket)},
            )
        return ChannelResponse(
            "completed",
            f"Run completed: {report.run_id}",
            {"report": asdict(report)},
        )

    def _resume(self, run_id: str) -> ChannelResponse:
        if not run_id:
            return ChannelResponse("error", "A run id is required.")
        return ChannelResponse(
            "resumed",
            f"Loaded run: {run_id}",
            self.orchestrator.resume(run_id),
        )

    def _approve(self, token: str) -> ChannelResponse:
        if not token:
            return ChannelResponse("error", "An approval token is required.")
        ticket = self.orchestrator.approvals.approve(token)
        return ChannelResponse(
            "approved",
            f"Approved: {ticket.approval_id}",
            {"approval": asdict(ticket)},
        )

    def _deny(self, token: str) -> ChannelResponse:
        if not token:
            return ChannelResponse("error", "An approval token is required.")
        ticket = self.orchestrator.approvals.deny(token)
        return ChannelResponse(
            "denied",
            f"Denied: {ticket.approval_id}",
            {"approval": asdict(ticket)},
        )

    def _desktop_task(self, objective: str) -> ChannelResponse:
        if not objective:
            return ChannelResponse(
                "error",
                "A desktop workflow objective is required.",
            )
        backend = self.orchestrator.worker.pc_backend or WindowsUiaBackend()
        if not backend.available():
            backend = VirtualDesktopSandboxBackend(
                Path(self.orchestrator.state_path).with_name(
                    "virtual_desktop_sandbox.json"
                )
            )
        self.workflow_service.ensure_universal_mode(backend, max_steps=8)
        result = self.workflow_service.execute(objective, backend)
        if result.get("status") == "clarification_required":
            questions = result["plan"].get("clarification_questions") or []
            message = "Clarification required before desktop execution."
            if questions:
                message = f"{message} {questions[0]}"
            return ChannelResponse(
                "clarification_required",
                message,
                result,
            )
        return ChannelResponse(
            "completed",
            result["plan"]["summary"],
            result,
        )

    @staticmethod
    def _should_route_to_desktop(objective: str) -> bool:
        lower = re.sub(r"\s+", " ", objective).strip().lower()
        if not lower:
            return False
        has_action = any(
            marker in lower for marker in GatewayCommandRouter.DESKTOP_ACTION_MARKERS
        )
        has_surface = any(
            marker in lower for marker in GatewayCommandRouter.DESKTOP_SURFACE_MARKERS
        )
        has_path_or_url = (
            re.search(r"https?://|[a-z]:[/\\]|\.[a-z0-9]{1,6}\b", lower) is not None
        )
        if (
            any(
                marker in lower for marker in GatewayCommandRouter.RESEARCH_ONLY_MARKERS
            )
            and not has_surface
        ):
            return False
        return has_action and (has_surface or has_path_or_url)

    @staticmethod
    def _needs_clarification(objective: str) -> bool:
        cleaned = re.sub(r"\s+", " ", objective).strip().lower()
        if not cleaned:
            return True
        return any(marker in cleaned for marker in GatewayCommandRouter.VAGUE_MARKERS)
