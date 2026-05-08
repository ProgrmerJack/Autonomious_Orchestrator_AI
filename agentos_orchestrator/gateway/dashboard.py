from __future__ import annotations

import asyncio
import importlib
import json
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
from agentos_orchestrator.cognition.live_fire_eval import (
    LiveFireEvalRunner,
)
from agentos_orchestrator.cognition.live_fire_review import (
    load_live_fire_reviews,
    promote_live_fire_failure,
    write_shadow_training_heads,
)
from agentos_orchestrator.cognition.os_eval_packs import eval_pack_payload
from agentos_orchestrator.cognition.replay_debug import load_replay_debug
from agentos_orchestrator.gateway.auth import (
    GatewaySecurityError,
    GatewaySecurityManager,
)
from agentos_orchestrator.gateway.dashboard_channel_routes import (
    register_dashboard_channel_routes,
)
from agentos_orchestrator.gateway.dashboard_pc_routes import (
    register_dashboard_pc_routes,
)
from agentos_orchestrator.gateway.dashboard_support import (
    _blocked_status,
    _client_host,
    _extract_depth_from_objective,
    _extract_session_token,
    _golden_traces_payload,
    _json_or_text,
    _live_fire_config,
    _normalize_depth,
    _objective_with_depth,
    _pc_backend,
    _pc_backend_status,
    _record_channel_delivery,
    _replay_benchmarks,
    _request_origin,
    _requires_unsafe_ack,
    _research_payload,
    _string_list,
    _workspace_root,
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
)
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
    """Small background runner so long UI requests do not block HTTP."""

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

    def start(self, objective: str, depth: object = "adaptive") -> dict[str, Any]:
        normalized_depth = _normalize_depth(depth)
        explicit_depth = _extract_depth_from_objective(objective)
        # If caller did not provide an explicit depth (or left adaptive), keep
        # the objective's declared depth tag so deep runs are not downgraded.
        if explicit_depth and normalized_depth == "adaptive":
            normalized_depth = explicit_depth
        run_objective = _objective_with_depth(objective, normalized_depth)
        job_id = new_id("job")
        run_id = new_id("run")
        now = utc_now()
        job = {
            "job_id": job_id,
            "objective": objective,
            "depth": normalized_depth,
            "run_objective": run_objective,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "run_id": run_id,
            "error": None,
            "report": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._futures[job_id] = self._executor.submit(
                self._run_job,
                job_id,
                run_id,
                run_objective,
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

    def _run_job(self, job_id: str, run_id: str, run_objective: str) -> None:
        self._update(job_id, status="running")
        self.event_hub.publish({"job": {"job_id": job_id, "status": "running"}})
        try:
            report = self.orchestrator.run(run_objective, run_id=run_id)
        except Exception as exc:
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
        responses = importlib.import_module("fastapi.responses")
    except ImportError as exc:
        raise RuntimeError("Install fastapi to run the dashboard API") from exc

    auth = GatewaySecurityManager()
    app = fastapi.FastAPI(
        title="AgentOS Gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.gateway_auth = auth
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

    @app.middleware("http")
    async def enforce_gateway_auth(request: Any, call_next: Any):
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        client_host = _client_host(request)
        origin = _request_origin(request)
        try:
            if request.url.path == "/auth/session":
                auth.assert_loopback_client(client_host)
                auth.assert_local_origin(origin)
            else:
                auth.require_session(
                    session_token=_extract_session_token(request.headers),
                    client_host=client_host,
                    origin=origin,
                    csrf_token=request.headers.get("x-agentos-csrf"),
                    require_csrf=request.method.upper() not in {"GET", "HEAD"},
                    require_unsafe_ack=_requires_unsafe_ack(
                        request.url.path,
                        request.method,
                    ),
                    unsafe_ack=request.headers.get("x-agentos-unsafe"),
                )
        except GatewaySecurityError as exc:
            return responses.JSONResponse(
                {"detail": str(exc)},
                status_code=exc.status_code,
            )
        return await call_next(request)

    async def create_auth_session(payload: dict, request: Any) -> dict:
        session = auth.create_session(
            bootstrap_token=str(payload.get("bootstrap_token") or ""),
            client_host=_client_host(request),
            origin=_request_origin(request),
        )
        return session.asdict()

    create_auth_session.__annotations__["request"] = fastapi.Request
    app.post("/auth/session")(create_auth_session)

    async def events(websocket: Any) -> None:
        try:
            auth.require_session(
                session_token=str(websocket.query_params.get("session_token") or ""),
                client_host=_client_host(websocket),
                origin=_request_origin(websocket),
                csrf_token=str(websocket.query_params.get("csrf_token") or ""),
                require_csrf=True,
            )
        except GatewaySecurityError as exc:
            await websocket.close(code=4401, reason=str(exc))
            return
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
        app.state.workflow_service = workflow_service
        pc_receipts: deque[dict[str, Any]] = deque(maxlen=100)
        channel_deliveries: deque[dict[str, Any]] = deque(maxlen=100)

        register_dashboard_pc_routes(
            app,
            fastapi,
            event_hub,
            orchestrator,
            workflow_service,
            pc_receipts,
        )
        register_dashboard_channel_routes(
            app,
            fastapi,
            command_registry,
            command_router,
            telegram,
            generic_webhook,
            slack,
            discord,
            channel_deliveries,
        )

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

        @app.get("/benchmarks/eval-pack")
        async def universal_eval_pack() -> dict:
            return eval_pack_payload()

        @app.post("/benchmarks/live-fire-eval")
        async def live_fire_eval(payload: dict) -> dict:
            backend_name = str(payload.get("backend") or "virtual-desktop-sandbox")
            approval_token = payload.get("approval_token")
            if backend_name != "virtual-desktop-sandbox":
                action = ActionRequest(
                    agent_id="dashboard-pc-control",
                    action_type="os.act",
                    target=f"{backend_name}://live-fire/eval-pack",
                    approval_token=(str(approval_token) if approval_token else None),
                )
                decision = orchestrator.authorization.authorize(
                    "dashboard",
                    action,
                )
                if not decision.allowed:
                    return {
                        "status": _blocked_status(decision.requires_approval),
                        "decision": asdict(decision),
                    }
            backend = _pc_backend(backend_name, orchestrator.state_path)
            runner = LiveFireEvalRunner(
                backend,
                _workspace_root(orchestrator.state_path),
            )
            return runner.run(_live_fire_config(payload)).asdict()

        @app.get("/benchmarks/live-fire-review")
        async def live_fire_review(limit: int = 10) -> dict:
            root = _workspace_root(orchestrator.state_path)
            return load_live_fire_reviews(root, limit=max(1, min(limit, 100)))

        @app.post("/benchmarks/live-fire-review/promote")
        async def live_fire_review_promote(payload: dict) -> dict:
            run_id = str(payload.get("run_id") or "").strip()
            task_id = str(payload.get("task_id") or "").strip()
            if not run_id or not task_id:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="run_id and task_id are required",
                )
            return promote_live_fire_failure(
                _workspace_root(orchestrator.state_path),
                run_id,
                task_id,
            )

        @app.post("/benchmarks/live-fire-shadow-training")
        async def live_fire_shadow_training(payload: dict) -> dict:
            paths = _string_list(payload.get("trajectory_paths"))
            output_dir = str(payload.get("output_dir") or "").strip()
            return write_shadow_training_heads(
                _workspace_root(orchestrator.state_path),
                trajectory_paths=paths or None,
                output_dir=output_dir or None,
            )

        @app.post("/debug/replay")
        async def replay_debug(payload: dict) -> dict:
            return load_replay_debug(
                Path.cwd(),
                run_id=str(payload.get("run_id") or ""),
                limit=int(payload.get("limit") or 1),
            )

        @app.post("/runs")
        async def create_run(payload: dict) -> dict:
            objective = str(payload.get("objective") or "").strip()
            if not objective:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail="objective is required",
                )
            # Always run via background executor — never block the event loop.
            # The legacy synchronous path (no "background" key) was the root
            # cause of server deadlocks on long-running research jobs.
            return run_manager.start(objective, payload.get("depth"))

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

        @app.get("/runs/{run_id}/progress")
        async def get_run_progress(run_id: str) -> dict:
            """Return live progress for a running research job.

            Returns the most recently written progress.json if available,
            or a 404 if the research phase has not yet started.
            """
            progress_path = Path.cwd() / "runs" / run_id / "research" / "progress.json"
            if not progress_path.exists():
                raise fastapi.HTTPException(
                    status_code=404,
                    detail="Progress not yet available — research phase has not started.",
                )
            try:
                return json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise fastapi.HTTPException(
                    status_code=500,
                    detail=f"Could not read progress file: {exc}",
                ) from exc

    return app
