from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agentos_orchestrator.core.approvals import ApprovalRequired
from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.product import CommandRegistry

from .channels import ChannelMessage


@dataclass(slots=True)
class ChannelResponse:
    status: str
    text: str
    payload: dict = field(default_factory=dict)


class GatewayCommandRouter:
    """OpenClaw-style channel command router for local AgentOS runs."""

    def __init__(
        self,
        orchestrator: ResearchOrchestrator,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.command_registry = command_registry or CommandRegistry()

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
        workflow = self.command_registry.get(command)
        if workflow is not None:
            return self._run(workflow.render(argument))
        if text:
            return self._run(text)
        return ChannelResponse("ignored", "No command text was provided.")

    def _run(self, objective: str) -> ChannelResponse:
        if not objective:
            return ChannelResponse(
                "error",
                "A research objective is required.",
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
