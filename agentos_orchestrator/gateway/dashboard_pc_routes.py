from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from agentos_orchestrator.core.types import ActionRequest, utc_now
from agentos_orchestrator.gateway.dashboard_support import (
    _blocked_status,
    _json_or_text,
    _pc_backend,
)
from agentos_orchestrator.os_control import UiAction
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.os_control.workflow import WorkflowVerificationError


def register_dashboard_pc_routes(
    app: Any,
    fastapi: Any,
    event_hub: Any,
    orchestrator: Any,
    workflow_service: Any,
    pc_receipts: deque[dict[str, Any]],
) -> None:
    @app.get("/pc/snapshot")
    async def pc_snapshot(
        backend: str = "windows-uia",
        limit: int = 120,
    ) -> dict:
        action = ActionRequest(
            agent_id="dashboard-pc-control",
            action_type="os.snapshot",
            target=f"{backend}://snapshot",
        )
        decision = orchestrator.authorization.authorize(
            "dashboard",
            action,
        )
        if not decision.allowed:
            return {"status": "blocked", "decision": asdict(decision)}
        nodes = _pc_backend(backend, orchestrator.state_path).snapshot()
        node_limit = max(1, min(limit, 500))
        return {
            "status": "ok",
            "backend": backend,
            "nodes": [asdict(node) for node in nodes[:node_limit]],
        }

    @app.post("/pc/debug-selector")
    async def pc_debug_selector(payload: dict) -> dict:
        backend = str(payload.get("backend") or "windows-uia")
        selector = str(payload.get("selector") or "").strip()
        limit = int(payload.get("limit") or 8)
        action = ActionRequest(
            agent_id="dashboard-pc-control",
            action_type="os.snapshot",
            target=f"{backend}://debug-selector",
        )
        decision = orchestrator.authorization.authorize(
            "dashboard",
            action,
        )
        if not decision.allowed:
            return {"status": "blocked", "decision": asdict(decision)}
        nodes = _pc_backend(backend, orchestrator.state_path).snapshot()
        report = debug_selector(selector, nodes, limit=limit)
        return {"status": "ok", "report": report.asdict()}

    @app.get("/pc/receipts")
    async def pc_receipt_history() -> list[dict]:
        return list(pc_receipts)

    @app.post("/pc/workflow/plan")
    async def pc_workflow_plan(payload: dict) -> dict:
        objective = str(payload.get("objective") or "").strip()
        if not objective:
            raise fastapi.HTTPException(
                status_code=400,
                detail="objective is required",
            )
        plan = workflow_service.plan(objective)
        return {"status": "ok", "plan": plan.asdict()}

    @app.post("/pc/workflow/execute")
    async def pc_workflow_execute(payload: dict) -> dict:
        objective = str(payload.get("objective") or "").strip()
        backend_name = str(payload.get("backend") or "virtual-desktop-sandbox")
        approval_token = payload.get("approval_token")
        if not objective:
            raise fastapi.HTTPException(
                status_code=400,
                detail="objective is required",
            )
        action = ActionRequest(
            agent_id="dashboard-pc-control",
            action_type="os.act",
            target=f"{backend_name}://workflow",
            payload={"workflow_objective": objective},
            approval_token=(str(approval_token) if approval_token else None),
        )
        decision = orchestrator.authorization.authorize(
            "dashboard",
            action,
        )
        if not decision.allowed:
            blocked = {
                "status": _blocked_status(decision.requires_approval),
                "decision": asdict(decision),
            }
            pc_receipts.appendleft(
                {
                    "created_at": utc_now(),
                    "backend": backend_name,
                    "selector": "workflow",
                    "action": "workflow.execute",
                    "result": blocked,
                }
            )
            return blocked
        backend = _pc_backend(backend_name, orchestrator.state_path)
        try:
            workflow_service.ensure_universal_mode(backend, max_steps=8)
            result = workflow_service.execute(objective, backend)
        except WorkflowVerificationError as exc:
            envelope = {
                "status": "verification_failed",
                "objective": objective,
                "failure": exc.asdict(),
            }
            pc_receipts.appendleft(
                {
                    "created_at": utc_now(),
                    "backend": backend_name,
                    "selector": "workflow",
                    "action": "workflow.execute",
                    "result": envelope,
                }
            )
            event_hub.publish({"pc_receipt": pc_receipts[0]})
            return envelope
        if result.get("status") == "clarification_required":
            envelope = result
        else:
            envelope = {"status": "executed", **result}
        pc_receipts.appendleft(
            {
                "created_at": utc_now(),
                "backend": backend_name,
                "selector": "workflow",
                "action": "workflow.execute",
                "result": envelope,
            }
        )
        event_hub.publish({"pc_receipt": pc_receipts[0]})
        return envelope

    @app.post("/pc/actions")
    async def pc_action(payload: dict) -> dict:
        backend = str(payload.get("backend") or "windows-uia")
        selector = str(payload.get("selector") or "").strip()
        action_type = str(payload.get("action") or "focus").strip()
        value = payload.get("value")
        approval_token = payload.get("approval_token")
        if not selector:
            raise fastapi.HTTPException(
                status_code=400,
                detail="selector is required",
            )
        action = ActionRequest(
            agent_id="dashboard-pc-control",
            action_type="os.act",
            target=f"{backend}://{selector}",
            payload={
                "action": action_type,
                "value_present": value is not None,
            },
            approval_token=(str(approval_token) if approval_token else None),
        )
        decision = orchestrator.authorization.authorize(
            "dashboard",
            action,
        )
        if not decision.allowed:
            blocked = {
                "status": _blocked_status(decision.requires_approval),
                "decision": asdict(decision),
            }
            pc_receipts.appendleft(
                {
                    "created_at": utc_now(),
                    "backend": backend,
                    "selector": selector,
                    "action": action_type,
                    "result": blocked,
                }
            )
            return blocked
        receipt = _pc_backend(backend, orchestrator.state_path).perform(
            UiAction(
                action_type=action_type,
                selector=selector,
                value=str(value) if value is not None else None,
            )
        )
        result = {"status": "executed", "receipt": _json_or_text(receipt)}
        pc_receipts.appendleft(
            {
                "created_at": utc_now(),
                "backend": backend,
                "selector": selector,
                "action": action_type,
                "result": result,
            }
        )
        event_hub.publish({"pc_receipt": pc_receipts[0]})
        return result
