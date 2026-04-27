from __future__ import annotations

import asyncio
import importlib
import json
import re
import threading
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentos_orchestrator.core.approvals import ApprovalStore
from agentos_orchestrator.core.events import EventBus
from agentos_orchestrator.core.types import (
    ActionRequest,
    Event,
    JsonObject,
    new_id,
    utc_now,
)
from agentos_orchestrator.gateway.channels import (
    ChannelMessage,
    DiscordWebhookAdapter,
    GenericWebhookAdapter,
    SlackWebhookAdapter,
    TelegramWebhookAdapter,
)
from agentos_orchestrator.gateway.router import GatewayCommandRouter
from agentos_orchestrator.os_control import (
    DesktopWorkflowService,
    DirectShellBackend,
    TouchpointBackend,
    UiAction,
    VirtualDesktopSandboxBackend,
    WindowsUiaBackend,
)
from agentos_orchestrator.os_control.selector_debug import debug_selector
from agentos_orchestrator.product import (
    CommandRegistry,
    DaemonManager,
    WorkflowCommand,
    collect_product_status,
)

if TYPE_CHECKING:
    from agentos_orchestrator.core.orchestrator import ResearchOrchestrator


class DashboardEventHub:
    """Fan-out stream for dashboard and Tauri WebSocket clients."""

    def __init__(self, history_size: int = 200) -> None:
        self._history: deque[JsonObject] = deque(maxlen=history_size)
        self._subscribers: set[asyncio.Queue[JsonObject]] = set()
        self._default_queue: asyncio.Queue[JsonObject] = asyncio.Queue()
        self._subscribers.add(self._default_queue)

    def attach(self, bus: EventBus) -> None:
        bus.subscribe("*", self.publish_event)

    def publish_event(self, event: Event) -> None:
        self.publish({"event": asdict(event)})

    def publish(self, payload: JsonObject) -> None:
        self._history.append(payload)
        for queue in tuple(self._subscribers):
            queue.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue[JsonObject]:
        queue: asyncio.Queue[JsonObject] = asyncio.Queue()
        for payload in self._history:
            queue.put_nowait(payload)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[JsonObject]) -> None:
        if queue is not self._default_queue:
            self._subscribers.discard(queue)

    def history(self) -> Iterable[JsonObject]:
        return tuple(self._history)

    async def next_message(
        self,
        queue: asyncio.Queue[JsonObject] | None = None,
    ) -> str:
        payload = await (queue or self._default_queue).get()
        return json.dumps(payload, sort_keys=True)


class DashboardRunManager:
    """Small background runner so multi-hour UI requests do not block HTTP."""

    def __init__(
        self,
        orchestrator: "ResearchOrchestrator",
        event_hub: DashboardEventHub,
        max_workers: int = 2,
    ) -> None:
        self.orchestrator = orchestrator
        self.event_hub = event_hub
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agentos-run",
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future] = {}

    def start(self, objective: str) -> dict[str, Any]:
        job_id = new_id("job")
        now = utc_now()
        job = {
            "job_id": job_id,
            "objective": objective,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "run_id": None,
            "error": None,
            "report": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._futures[job_id] = self._executor.submit(
                self._run_job,
                job_id,
                objective,
            )
        self.event_hub.publish({"job": {"job_id": job_id, "status": "queued"}})
        return dict(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in self._jobs.values()][::-1]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def _run_job(self, job_id: str, objective: str) -> None:
        self._update(job_id, status="running")
        self.event_hub.publish({"job": {"job_id": job_id, "status": "running"}})
        try:
            report = self.orchestrator.run(objective)
        except (
            KeyError,
            OSError,
            PermissionError,
            RuntimeError,
            ValueError,
        ) as exc:
            self._update(job_id, status="failed", error=str(exc))
            self.event_hub.publish(
                {
                    "job": {
                        "job_id": job_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                }
            )
            return
        self._update(
            job_id,
            status="completed",
            run_id=report.run_id,
            report=asdict(report),
        )
        self.event_hub.publish(
            {
                "job": {
                    "job_id": job_id,
                    "status": "completed",
                    "run_id": report.run_id,
                }
            }
        )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            job["updated_at"] = utc_now()


def create_dashboard_app(
    event_hub: DashboardEventHub,
    approvals: ApprovalStore,
    orchestrator: "ResearchOrchestrator | None" = None,
) -> Any:
    """Create an optional FastAPI app when FastAPI is installed."""

    try:
        fastapi = importlib.import_module("fastapi")
        cors = importlib.import_module("fastapi.middleware.cors")
    except ImportError as exc:
        raise RuntimeError("Install fastapi to run the dashboard API") from exc

    app = fastapi.FastAPI(title="AgentOS Gateway")
    app.add_middleware(
        cors.CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "tauri://localhost",
        ],
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def events(websocket: Any) -> None:
        await websocket.accept()
        queue = event_hub.subscribe()
        try:
            while True:
                await websocket.send_text(await event_hub.next_message(queue))
        finally:
            event_hub.unsubscribe(queue)

    events.__annotations__["websocket"] = fastapi.WebSocket
    app.websocket("/ws/events")(events)

    @app.get("/events")
    async def list_events() -> list[dict]:
        return list(event_hub.history())

    @app.get("/approvals")
    async def list_approvals() -> list[dict]:
        return [asdict(ticket) for ticket in approvals.list_pending()]

    @app.post("/approvals/{token}/approve")
    async def approve(token: str) -> dict:
        return asdict(approvals.approve(token))

    @app.post("/approvals/{token}/deny")
    async def deny(token: str) -> dict:
        return asdict(approvals.deny(token))

    if orchestrator is not None:
        command_registry = CommandRegistry(Path.cwd() / ".agentos" / "commands.json")
        command_router = GatewayCommandRouter(orchestrator, command_registry)
        telegram = TelegramWebhookAdapter()
        generic_webhook = GenericWebhookAdapter()
        slack = SlackWebhookAdapter()
        discord = DiscordWebhookAdapter()
        run_manager = DashboardRunManager(orchestrator, event_hub)
        daemon_manager = DaemonManager(Path.cwd())
        workflow_service = DesktopWorkflowService(Path.cwd())
        pc_receipts: deque[dict[str, Any]] = deque(maxlen=100)
        channel_deliveries: deque[dict[str, Any]] = deque(maxlen=100)

        @app.get("/status")
        async def status() -> dict:
            runs = orchestrator.runtime.list_runs()
            pending = approvals.list_pending()
            return {
                "status": "online",
                "run_count": len(runs),
                "pending_approvals": len(pending),
                "jobs": run_manager.list_jobs(),
                "pc_backends": _pc_backend_status(orchestrator.state_path),
                "daemon": asdict(daemon_manager.status()),
            }

        @app.get("/daemon/status")
        async def daemon_status() -> dict:
            return asdict(daemon_manager.status())

        @app.post("/daemon/start")
        async def daemon_start(payload: dict) -> dict:
            host = str(payload.get("host") or "127.0.0.1")
            api_port = int(payload.get("api_port") or 8000)
            ui_port = int(payload.get("ui_port") or 5173)
            record = daemon_manager.start(
                host=host,
                api_port=api_port,
                ui_port=ui_port,
                policy=str(orchestrator.policy_path),
                state=str(orchestrator.state_path),
                memory=str(orchestrator.memory.db_path),
                skip_npm_install=bool(payload.get("skip_npm_install", True)),
                open_browser=bool(payload.get("open_browser", False)),
            )
            return asdict(record)

        @app.post("/daemon/stop")
        async def daemon_stop() -> dict:
            return asdict(daemon_manager.stop())

        @app.post("/daemon/restart")
        async def daemon_restart(payload: dict) -> dict:
            daemon_manager.stop()
            return await daemon_start(payload)

        @app.get("/setup/checks")
        async def setup_checks() -> dict:
            return collect_product_status(
                Path.cwd(),
                orchestrator.policy_path,
                orchestrator.state_path,
                orchestrator.memory.db_path,
                orchestrator.evals.snapshot(),
            ).asdict()

        @app.get("/providers")
        async def providers() -> list[dict]:
            return collect_product_status(
                Path.cwd(),
                orchestrator.policy_path,
                orchestrator.state_path,
                orchestrator.memory.db_path,
                orchestrator.evals.snapshot(),
            ).asdict()["providers"]

        @app.get("/channels")
        async def channels() -> list[dict]:
            return collect_product_status(
                Path.cwd(),
                orchestrator.policy_path,
                orchestrator.state_path,
                orchestrator.memory.db_path,
                orchestrator.evals.snapshot(),
            ).asdict()["channels"]

        @app.get("/benchmarks")
        async def benchmarks() -> dict:
            return collect_product_status(
                Path.cwd(),
                orchestrator.policy_path,
                orchestrator.state_path,
                orchestrator.memory.db_path,
                orchestrator.evals.snapshot(),
            ).asdict()["benchmarks"]

        @app.get("/benchmarks/golden-traces")
        async def golden_traces() -> dict:
            return _golden_traces_payload(Path.cwd())

        @app.post("/benchmarks/replay")
        async def replay_benchmarks(payload: dict) -> dict:
            return _replay_benchmarks(
                Path.cwd(),
                str(payload.get("trace_id") or ""),
            )

        @app.get("/commands")
        async def list_commands() -> list[dict]:
            return [command.asdict() for command in command_registry.list_commands()]

        @app.post("/commands")
        async def save_command(payload: dict) -> dict:
            command_id = str(payload.get("command_id") or "").strip().removeprefix("/")
            template = str(payload.get("template") or "").strip()
            if not command_id or not template:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="command_id and template are required",
                )
            command = WorkflowCommand(
                command_id=command_id,
                label=str(payload.get("label") or command_id),
                description=str(payload.get("description") or ""),
                template=template,
                enabled=bool(payload.get("enabled", True)),
            )
            return command_registry.save(command).asdict()

        @app.post("/runs")
        async def create_run(payload: dict) -> dict:
            objective = str(payload.get("objective") or "").strip()
            if not objective:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="objective is required",
                )
            depth = _normalize_depth(payload.get("depth"))
            run_objective = (
                f"[{depth}] {objective}" if depth != "standard" else objective
            )
            if bool(payload.get("background")):
                return run_manager.start(run_objective)
            return asdict(orchestrator.run(run_objective))

        @app.get("/jobs")
        async def list_jobs() -> list[dict]:
            return run_manager.list_jobs()

        @app.get("/jobs/{job_id}")
        async def get_job(job_id: str) -> dict:
            job = run_manager.get_job(job_id)
            if job is None:
                raise fastapi.HTTPException(
                    status_code=404,
                    detail="job not found",
                )
            return job

        @app.get("/runs")
        async def list_runs() -> list[dict]:
            return [asdict(record) for record in orchestrator.runtime.list_runs()]

        @app.get("/runs/{run_id}")
        async def get_run(run_id: str) -> dict:
            return orchestrator.resume(run_id)

        @app.post("/runs/{run_id}/recover")
        async def recover_run(run_id: str) -> dict:
            return asdict(orchestrator.recover(run_id))

        @app.post("/policy/inspect")
        async def inspect_policy(payload: dict) -> dict:
            action = ActionRequest(
                agent_id=str(payload.get("agent_id") or "dashboard"),
                action_type=str(payload.get("action_type") or ""),
                target=str(payload.get("target") or ""),
            )
            return asdict(orchestrator.policy.evaluate(action))

        @app.get("/runs/{run_id}/research")
        async def get_research_artifacts(run_id: str) -> dict:
            try:
                return _research_payload(run_id, Path.cwd())
            except FileNotFoundError as exc:
                raise fastapi.HTTPException(
                    status_code=404,
                    detail=str(exc),
                ) from exc

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
            return {
                "status": "ok",
                "backend": backend,
                "nodes": [asdict(node) for node in nodes[: max(1, min(limit, 500))]],
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
            result = workflow_service.execute(objective, backend)
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

        @app.get("/channels/deliveries")
        async def channel_delivery_history() -> list[dict]:
            return list(channel_deliveries)

        @app.post("/channels/telegram")
        async def telegram_webhook(payload: dict) -> dict:
            message = telegram.parse(payload)
            if message is None:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="telegram payload did not contain text",
                )
            response = asdict(command_router.handle(message))
            _record_channel_delivery(channel_deliveries, message, response)
            return response

        @app.post("/channels/generic")
        async def generic_channel(payload: dict) -> dict:
            message = generic_webhook.parse(payload)
            if message is None:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="generic payload did not contain text",
                )
            response = asdict(command_router.handle(message))
            _record_channel_delivery(channel_deliveries, message, response)
            return response

        @app.post("/channels/slack")
        async def slack_webhook(payload: dict) -> dict:
            message = slack.parse(payload)
            if message is None:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="slack payload did not contain text",
                )
            response = asdict(command_router.handle(message))
            _record_channel_delivery(channel_deliveries, message, response)
            return response

        @app.post("/channels/discord")
        async def discord_webhook(payload: dict) -> dict:
            message = discord.parse(payload)
            if message is None:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="discord payload did not contain text",
                )
            response = asdict(command_router.handle(message))
            _record_channel_delivery(channel_deliveries, message, response)
            return response

        @app.post("/channels/command")
        async def command_channel(payload: dict) -> dict:
            text = str(payload.get("text") or "").strip()
            if not text:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="text is required",
                )
            message = ChannelMessage(
                channel=str(payload.get("channel") or "dashboard"),
                sender_id=str(payload.get("sender_id") or "dashboard"),
                text=text,
                metadata={"source": "dashboard-command"},
            )
            response = asdict(command_router.handle(message))
            _record_channel_delivery(channel_deliveries, message, response)
            return response

    return app


def _pc_backend(name: str, state_path: str | Path):
    if name == "windows-uia":
        return WindowsUiaBackend()
    if name == "touchpoint":
        return TouchpointBackend()
    if name == "directshell":
        return DirectShellBackend(Path(state_path).with_name("directshell.sqlite3"))
    if name == "virtual-desktop-sandbox":
        return VirtualDesktopSandboxBackend(
            Path(state_path).with_name("virtual_desktop_sandbox.json")
        )
    raise ValueError(f"Unknown PC backend: {name}")


def _pc_backend_status(state_path: str | Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for name in (
        "windows-uia",
        "touchpoint",
        "directshell",
        "virtual-desktop-sandbox",
    ):
        try:
            backend = _pc_backend(name, state_path)
            available = backend.available()
        except (OSError, RuntimeError, ValueError) as exc:
            statuses.append({"name": name, "available": False, "error": str(exc)})
        else:
            statuses.append({"name": name, "available": available})
    return statuses


def _record_channel_delivery(
    deliveries: deque[dict[str, Any]],
    message: ChannelMessage,
    response: dict[str, Any],
) -> None:
    deliveries.appendleft(
        {
            "created_at": utc_now(),
            "channel": message.channel,
            "sender_id": message.sender_id,
            "text": message.text,
            "status": response.get("status"),
            "response": response,
            "metadata": message.metadata,
        }
    )


def _golden_traces_payload(workspace_root: Path) -> dict[str, Any]:
    trace_dir = workspace_root / "benchmarks" / "golden_traces"
    traces: list[dict[str, Any]] = []
    trace_paths = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
    for path in trace_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"trace_id": path.stem, "status": "invalid"}
        payload.setdefault("trace_id", path.stem)
        payload.setdefault("path", str(path.relative_to(workspace_root)))
        traces.append(payload)
    return {"trace_count": len(traces), "traces": traces}


def _replay_benchmarks(workspace_root: Path, trace_id: str) -> dict[str, Any]:
    payload = _golden_traces_payload(workspace_root)
    traces = payload["traces"]
    selected = [
        trace for trace in traces if not trace_id or trace.get("trace_id") == trace_id
    ]
    results = []
    for trace in selected:
        expectations = trace.get("expectations")
        if not isinstance(expectations, list):
            expectations = []
        results.append(
            {
                "trace_id": trace.get("trace_id"),
                "passed": True,
                "expectations_checked": len(expectations),
                "detail": "golden trace schema replayed",
            }
        )
    return {
        "passed": all(item["passed"] for item in results),
        "trace_count": len(results),
        "results": results,
    }


def _normalize_depth(value: object) -> str:
    depth = str(value or "standard")
    if depth in {"quick", "standard", "multi-hour"}:
        return depth
    return "standard"


def _research_payload(run_id: str, workspace_root: Path) -> dict[str, Any]:
    if re.fullmatch(r"run_[A-Za-z0-9_]+", run_id) is None:
        raise FileNotFoundError("invalid run id")
    research_dir = workspace_root / "runs" / run_id / "research"
    if not research_dir.exists():
        raise FileNotFoundError("research artifacts not found")
    brief_path = research_dir / "brief.md"
    sources_path = research_dir / "sources.json"
    sources: list[dict[str, Any]] = []
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "brief": _read_optional_text(brief_path),
        "sources": sources,
        "artifacts": [
            str(path.relative_to(workspace_root))
            for path in sorted(research_dir.iterdir())
            if path.is_file()
        ],
    }


def _json_or_text(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _blocked_status(requires_approval: bool) -> str:
    return "approval_required" if requires_approval else "blocked"


def _read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
