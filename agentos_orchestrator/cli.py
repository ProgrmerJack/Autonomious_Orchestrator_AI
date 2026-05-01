from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

from .cognition.adaptation_training import (
    AdaptationLongRunConfig,
    AdaptationTrainingConfig,
    UnknownAppAdaptationTrainer,
)
from .cognition.live_fire_eval import LiveFireEvalConfig, LiveFireEvalRunner
from .cognition.live_fire_review import (
    load_live_fire_reviews,
    promote_live_fire_failure,
    write_shadow_training_heads,
)
from .core.orchestrator import ResearchOrchestrator
from .core.policy import PermissionPolicy
from .core.types import ActionRequest
from .config import MarkdownAgentConfig
from .gateway import DashboardEventHub, create_dashboard_app
from .os_control import (
    DEFAULT_NOTEPAD_FILE_NAME,
    DEFAULT_NOTEPAD_PAYLOAD,
    DEFAULT_PAINT_FILE_NAME,
    DirectShellBackend,
    NotepadLiveFireConfig,
    NotepadLiveFireTrial,
    PaintLiveFireConfig,
    PaintLiveFireTrial,
    TouchpointBackend,
    UiAction,
    VirtualDesktopSandboxBackend,
    WindowsUiaBackend,
)
from .product import DaemonManager, collect_product_status


def add_runtime_options(
    parser: argparse.ArgumentParser,
    suppress_defaults: bool = False,
) -> None:
    policy_default = (
        argparse.SUPPRESS
        if suppress_defaults
        else ("examples/policies/deep_research.json")
    )
    state_default = (
        argparse.SUPPRESS if suppress_defaults else (".agentos/state.sqlite3")
    )
    memory_default = (
        argparse.SUPPRESS if suppress_defaults else (".agentos/memory.sqlite3")
    )
    parser.add_argument(
        "--policy",
        default=policy_default,
        help="Path to permission policy JSON.",
    )
    parser.add_argument(
        "--state",
        default=state_default,
        help="Path to durable run state SQLite database.",
    )
    parser.add_argument(
        "--memory",
        default=memory_default,
        help="Path to durable memory SQLite database.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentos",
        description="Secure autonomous deep research orchestrator.",
    )
    add_runtime_options(parser)

    runtime_parent = argparse.ArgumentParser(add_help=False)
    add_runtime_options(runtime_parent, suppress_defaults=True)

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="Start a new research run.",
        parents=[runtime_parent],
    )
    run_parser.add_argument("--objective", required=True)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Inspect a run checkpoint.",
        parents=[runtime_parent],
    )
    resume_parser.add_argument("--run-id", required=True)

    recover_parser = subparsers.add_parser(
        "recover",
        help="Recover and continue a durable run.",
        parents=[runtime_parent],
    )
    recover_parser.add_argument("--run-id", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-policy",
        help="Evaluate a single action against the policy.",
        parents=[runtime_parent],
    )
    inspect_parser.add_argument("--agent-id", default="manual")
    inspect_parser.add_argument("--action-type", required=True)
    inspect_parser.add_argument("--target", required=True)

    approve_parser = subparsers.add_parser(
        "approve",
        help="Approve a pending action by token.",
        parents=[runtime_parent],
    )
    approve_parser.add_argument("--token", required=True)

    deny_parser = subparsers.add_parser(
        "deny",
        help="Deny a pending action by token.",
        parents=[runtime_parent],
    )
    deny_parser.add_argument("--token", required=True)

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect markdown agent configuration.",
    )
    config_parser.add_argument("--root", default=".")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run product readiness checks for setup and daily use.",
        parents=[runtime_parent],
    )
    doctor_parser.add_argument("--compact", action="store_true")

    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Manage the detached local AgentOS gateway.",
        parents=[runtime_parent],
    )
    daemon_subparsers = daemon_parser.add_subparsers(
        dest="daemon_command",
        required=True,
    )
    daemon_subparsers.add_parser("status", help="Show daemon status.")
    daemon_subparsers.add_parser("stop", help="Stop the daemon.")
    daemon_subparsers.add_parser("restart", help="Restart the daemon.")
    daemon_start = daemon_subparsers.add_parser(
        "start",
        help="Start AgentOS in the background.",
    )
    daemon_start.add_argument("--host", default="127.0.0.1")
    daemon_start.add_argument("--api-port", type=int, default=8000)
    daemon_start.add_argument("--ui-port", type=int, default=5173)
    daemon_start.add_argument("--skip-npm-install", action="store_true")
    daemon_start.add_argument("--open-browser", action="store_true")

    dashboard_parser = subparsers.add_parser(
        "serve-dashboard",
        help="Run the optional FastAPI dashboard gateway.",
        parents=[runtime_parent],
    )
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8000)

    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch the local API and dashboard UI together.",
        parents=[runtime_parent],
    )
    launch_parser.add_argument("--host", default="127.0.0.1")
    launch_parser.add_argument("--api-port", type=int, default=8000)
    launch_parser.add_argument("--ui-port", type=int, default=5173)
    launch_parser.add_argument("--no-browser", action="store_true")
    launch_parser.add_argument("--skip-npm-install", action="store_true")

    snapshot_parser = subparsers.add_parser(
        "pc-snapshot",
        help="Read structured PC UI state through an OS-control backend.",
        parents=[runtime_parent],
    )
    snapshot_parser.add_argument("--backend", default="windows-uia")
    snapshot_parser.add_argument("--limit", type=int, default=120)

    act_parser = subparsers.add_parser(
        "pc-act",
        help="Perform a guarded PC UI action through an OS-control backend.",
        parents=[runtime_parent],
    )
    act_parser.add_argument("--backend", default="windows-uia")
    act_parser.add_argument("--action", required=True)
    act_parser.add_argument("--selector", required=True)
    act_parser.add_argument("--value")
    act_parser.add_argument("--approval-token")

    live_fire_parser = subparsers.add_parser(
        "pc-live-fire-notepad",
        help="Run the guarded Notepad sim-to-real smoke trial.",
        parents=[runtime_parent],
    )
    live_fire_parser.add_argument("--backend", default="windows-uia")
    live_fire_parser.add_argument(
        "--payload",
        default=DEFAULT_NOTEPAD_PAYLOAD,
    )
    live_fire_parser.add_argument(
        "--file-name",
        default=DEFAULT_NOTEPAD_FILE_NAME,
    )
    live_fire_parser.add_argument("--timeout", type=float, default=12.0)
    live_fire_parser.add_argument("--approval-token")

    paint_live_fire_parser = subparsers.add_parser(
        "pc-live-fire-paint",
        help="Run the guarded Paint sim-to-real drawing trial.",
        parents=[runtime_parent],
    )
    paint_live_fire_parser.add_argument("--backend", default="windows-uia")
    paint_live_fire_parser.add_argument(
        "--file-name",
        default=DEFAULT_PAINT_FILE_NAME,
    )
    paint_live_fire_parser.add_argument("--timeout", type=float, default=12.0)
    paint_live_fire_parser.add_argument("--approval-token")

    eval_live_fire_parser = subparsers.add_parser(
        "pc-live-fire-eval",
        help="Run the universal 100-task live-fire OS eval pack.",
        parents=[runtime_parent],
    )
    eval_live_fire_parser.add_argument(
        "--backend",
        default="virtual-desktop-sandbox",
    )
    eval_live_fire_parser.add_argument("--max-tasks", type=int)
    eval_live_fire_parser.add_argument("--surface", action="append")
    eval_live_fire_parser.add_argument("--intent", action="append")
    eval_live_fire_parser.add_argument("--run-id", default="")
    eval_live_fire_parser.add_argument(
        "--safe-windows-pack",
        action="store_true",
    )
    eval_live_fire_parser.add_argument("--repeat", type=int, default=1)
    eval_live_fire_parser.add_argument("--promote-after", type=int, default=1)
    eval_live_fire_parser.add_argument("--heldout-from", default="")
    eval_live_fire_parser.add_argument(
        "--no-promote-failures",
        action="store_true",
    )
    eval_live_fire_parser.add_argument("--replay-limit", type=int, default=10)
    eval_live_fire_parser.add_argument("--training-output", default="")
    eval_live_fire_parser.add_argument("--approval-token")

    eval_review_parser = subparsers.add_parser(
        "pc-live-fire-review",
        help="Review recent live-fire failures and optionally promote one.",
        parents=[runtime_parent],
    )
    eval_review_parser.add_argument("--limit", type=int, default=10)
    eval_review_parser.add_argument("--promote-run-id", default="")
    eval_review_parser.add_argument("--promote-task-id", default="")

    shadow_parser = subparsers.add_parser(
        "pc-live-fire-shadow-train",
        help="Write advisory shadow-training datasets for low-risk heads.",
        parents=[runtime_parent],
    )
    shadow_parser.add_argument("--trajectory", action="append")
    shadow_parser.add_argument("--output-dir", default="")

    train_parser = subparsers.add_parser(
        "pc-train-adaptation",
        help="Warm-start unknown-app adaptation from external GUI data and local trajectories.",
        parents=[runtime_parent],
    )
    train_parser.add_argument("--screenspot-limit", type=int, default=64)
    train_parser.add_argument("--click100k-limit", type=int, default=0)
    train_parser.add_argument("--gui-actor-limit", type=int, default=0)
    train_parser.add_argument("--screenspot-source", action="append")
    train_parser.add_argument("--osworld-archive-limit", type=int, default=0)
    train_parser.add_argument(
        "--osworld-archive-transition-limit",
        type=int,
        default=0,
    )
    train_parser.add_argument("--trajectory", action="append")
    train_parser.add_argument("--output-dir", default="")
    train_parser.add_argument("--cache-dir", default="")
    train_parser.add_argument("--cache-budget-gb", type=float, default=0.0)
    train_parser.add_argument("--stage-archives", action="store_true")
    train_parser.add_argument("--skip-osworld-manifest", action="store_true")

    longrun_parser = subparsers.add_parser(
        "pc-train-adaptation-longrun",
        help="Run shard-based adaptation training across many archive batches with optional local staging.",
        parents=[runtime_parent],
    )
    longrun_parser.add_argument("--shard-count", type=int, default=1)
    longrun_parser.add_argument("--screenspot-limit-per-shard", type=int, default=0)
    longrun_parser.add_argument("--click100k-limit-per-shard", type=int, default=0)
    longrun_parser.add_argument("--gui-actor-limit-per-shard", type=int, default=0)
    longrun_parser.add_argument("--screenspot-source", action="append")
    longrun_parser.add_argument("--osworld-archives-per-shard", type=int, default=0)
    longrun_parser.add_argument(
        "--osworld-archive-transition-limit-per-shard",
        type=int,
        default=0,
    )
    longrun_parser.add_argument(
        "--osworld-archive-candidate-multiplier",
        type=int,
        default=0,
    )
    longrun_parser.add_argument("--trajectory", action="append")
    longrun_parser.add_argument("--output-dir", default="")
    longrun_parser.add_argument("--state-path", default="")
    longrun_parser.add_argument("--cache-dir", default="")
    longrun_parser.add_argument("--cache-budget-gb", type=float, default=0.0)
    longrun_parser.add_argument("--stage-archives", action="store_true")
    longrun_parser.add_argument("--skip-osworld-manifest", action="store_true")
    longrun_parser.add_argument("--no-resume", action="store_true")
    longrun_parser.add_argument("--no-internal-trajectories", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "config":
        config = MarkdownAgentConfig.load(args.root)
        print(
            json.dumps(
                {
                    "heartbeat": config.heartbeat,
                    "heartbeat_enabled": config.heartbeat_enabled(),
                    "soul_chars": len(config.soul),
                    "agents_chars": len(config.agents),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "launch":
        return _launch_dashboard(args)

    if args.command == "doctor":
        product = collect_product_status(
            Path.cwd(),
            args.policy,
            args.state,
            args.memory,
        )
        payload = product.asdict()
        if args.compact:
            payload = {
                "readiness_score": payload["benchmarks"]["readiness_score"],
                "required_checks_passed": payload["benchmarks"][
                    "required_checks_passed"
                ],
                "required_checks_total": payload["benchmarks"]["required_checks_total"],
            }
        print(json.dumps(payload, indent=2))
        failed = [
            check
            for check in product.checks
            if check.required and check.status != "pass"
        ]
        return 0 if not failed else 2

    if args.command == "daemon":
        return _daemon(args)

    policy_path = Path(args.policy)

    if args.command == "inspect-policy":
        policy = PermissionPolicy.from_file(policy_path)
        decision = policy.evaluate(
            ActionRequest(
                agent_id=args.agent_id,
                action_type=args.action_type,
                target=args.target,
            )
        )
        print(json.dumps(asdict(decision), indent=2))
        return 0 if decision.allowed else 2

    orchestrator = ResearchOrchestrator.from_paths(
        policy_path=policy_path,
        state_path=args.state,
        memory_path=args.memory,
    )

    if args.command == "run":
        report = orchestrator.run(args.objective)
        print(json.dumps(asdict(report), indent=2))
        return 0

    if args.command == "resume":
        payload = orchestrator.resume(args.run_id)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "recover":
        report = orchestrator.recover(args.run_id)
        print(json.dumps(asdict(report), indent=2))
        return 0

    if args.command == "approve":
        ticket = orchestrator.approvals.approve(args.token)
        print(json.dumps(asdict(ticket), indent=2))
        return 0

    if args.command == "deny":
        ticket = orchestrator.approvals.deny(args.token)
        print(json.dumps(asdict(ticket), indent=2))
        return 0

    if args.command == "serve-dashboard":
        try:
            uvicorn = importlib.import_module("uvicorn")
        except ImportError as exc:
            raise RuntimeError(
                "Install uvicorn and fastapi to run the dashboard API"
            ) from exc

        event_hub = DashboardEventHub()
        event_hub.attach(orchestrator.event_bus)
        for event in orchestrator.event_log.list_events():
            event_hub.publish_event(event)
        app = create_dashboard_app(
            event_hub,
            orchestrator.approvals,
            orchestrator=orchestrator,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "pc-snapshot":
        action = ActionRequest(
            agent_id="manual-pc-control",
            action_type="os.snapshot",
            target=f"{args.backend}://snapshot",
        )
        auth_decision = orchestrator.authorization.authorize("manual", action)
        if not auth_decision.allowed:
            print(json.dumps(asdict(auth_decision), indent=2))
            return 2
        backend = _pc_backend(args.backend, args.state)
        nodes = backend.snapshot()[: args.limit]
        print(json.dumps([asdict(node) for node in nodes], indent=2))
        return 0

    if args.command == "pc-act":
        action = ActionRequest(
            agent_id="manual-pc-control",
            action_type="os.act",
            target=f"{args.backend}://{args.selector}",
            payload={
                "action": args.action,
                "value_present": args.value is not None,
            },
            approval_token=args.approval_token,
        )
        auth_decision = orchestrator.authorization.authorize("manual", action)
        if not auth_decision.allowed:
            print(json.dumps(asdict(auth_decision), indent=2))
            return 2
        backend = _pc_backend(args.backend, args.state)
        receipt = backend.perform(
            UiAction(args.action, args.selector, value=args.value)
        )
        print(json.dumps({"receipt": _json_or_text(receipt)}, indent=2))
        return 0

    if args.command == "pc-live-fire-notepad":
        action = ActionRequest(
            agent_id="manual-pc-control",
            action_type="os.act",
            target=f"{args.backend}://live-fire/notepad",
            payload={"trial": "notepad", "file_name": args.file_name},
            approval_token=args.approval_token,
        )
        auth_decision = orchestrator.authorization.authorize("manual", action)
        if not auth_decision.allowed:
            print(json.dumps(asdict(auth_decision), indent=2))
            return 2
        backend = _pc_backend(args.backend, args.state)
        notepad_trial = NotepadLiveFireTrial(
            backend=backend,
            workspace_root=Path.cwd(),
        )
        notepad_result = notepad_trial.run(
            NotepadLiveFireConfig(
                payload=args.payload,
                file_name=args.file_name,
                dialog_timeout_seconds=args.timeout,
            )
        )
        print(json.dumps(asdict(notepad_result), indent=2))
        return 0 if notepad_result.success else 1

    if args.command == "pc-live-fire-paint":
        action = ActionRequest(
            agent_id="manual-pc-control",
            action_type="os.act",
            target=f"{args.backend}://live-fire/paint",
            payload={"trial": "paint", "file_name": args.file_name},
            approval_token=args.approval_token,
        )
        auth_decision = orchestrator.authorization.authorize("manual", action)
        if not auth_decision.allowed:
            print(json.dumps(asdict(auth_decision), indent=2))
            return 2
        backend = _pc_backend(args.backend, args.state)
        paint_trial = PaintLiveFireTrial(
            backend=backend,
            workspace_root=Path.cwd(),
        )
        paint_result = paint_trial.run(
            PaintLiveFireConfig(
                file_name=args.file_name,
                dialog_timeout_seconds=args.timeout,
            )
        )
        print(json.dumps(asdict(paint_result), indent=2))
        return 0 if paint_result.success else 1

    if args.command == "pc-live-fire-eval":
        action = ActionRequest(
            agent_id="manual-pc-control",
            action_type="os.act",
            target=f"{args.backend}://live-fire/eval-pack",
            payload={
                "trial": "universal-eval-pack",
                "max_tasks": args.max_tasks,
                "surfaces": args.surface or [],
                "intents": args.intent or [],
                "safe_windows_pack": args.safe_windows_pack,
                "repeat": args.repeat,
            },
            approval_token=args.approval_token,
        )
        auth_decision = orchestrator.authorization.authorize("manual", action)
        if not auth_decision.allowed:
            print(json.dumps(asdict(auth_decision), indent=2))
            return 2
        backend = _pc_backend(args.backend, args.state)
        live_fire_config = LiveFireEvalConfig(
            run_id=args.run_id,
            max_tasks=args.max_tasks,
            surfaces=tuple(args.surface or ()),
            intents=tuple(args.intent or ()),
            windows_safe_pack=args.safe_windows_pack,
            repeat=args.repeat,
            promote_failures=not args.no_promote_failures,
            promote_after=args.promote_after,
            replay_limit=args.replay_limit,
            training_output=args.training_output,
        )
        live_fire_config.heldout_from = args.heldout_from
        eval_result = LiveFireEvalRunner(backend, Path.cwd()).run(live_fire_config)
        print(json.dumps(eval_result.asdict(), indent=2))
        return 0 if eval_result.success else 1

    if args.command == "pc-live-fire-review":
        if args.promote_run_id and args.promote_task_id:
            payload = promote_live_fire_failure(
                Path.cwd(),
                args.promote_run_id,
                args.promote_task_id,
            )
        else:
            payload = load_live_fire_reviews(Path.cwd(), limit=args.limit)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "pc-live-fire-shadow-train":
        payload = write_shadow_training_heads(
            Path.cwd(),
            trajectory_paths=args.trajectory or None,
            output_dir=args.output_dir or None,
        )
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("ready_for_shadow_training") else 1

    if args.command == "pc-train-adaptation":
        trainer = UnknownAppAdaptationTrainer(Path.cwd())
        adaptation_result = trainer.train(
            AdaptationTrainingConfig(
                screenspot_limit=args.screenspot_limit,
                screenspot_offset=0,
                click100k_limit=args.click100k_limit,
                click100k_offset=0,
                gui_actor_limit=args.gui_actor_limit,
                gui_actor_offset=0,
                screenspot_sources=tuple(
                    args.screenspot_source or ("windows", "web", "macos")
                ),
                include_internal_trajectories=True,
                trajectory_paths=tuple(args.trajectory or ()),
                download_osworld_manifest=not args.skip_osworld_manifest,
                osworld_archive_limit=args.osworld_archive_limit,
                osworld_archive_transition_limit=args.osworld_archive_transition_limit,
                cache_dir=args.cache_dir,
                cache_budget_bytes=max(0, int(args.cache_budget_gb * (1 << 30))),
                stage_remote_archives=args.stage_archives,
                output_dir=args.output_dir,
            )
        )
        print(json.dumps(adaptation_result.asdict(), indent=2))
        return 0 if adaptation_result.success else 1

    if args.command == "pc-train-adaptation-longrun":
        trainer = UnknownAppAdaptationTrainer(Path.cwd())
        longrun_result = trainer.train_long_run(
            AdaptationLongRunConfig(
                shard_count=args.shard_count,
                screenspot_limit_per_shard=args.screenspot_limit_per_shard,
                click100k_limit_per_shard=args.click100k_limit_per_shard,
                gui_actor_limit_per_shard=args.gui_actor_limit_per_shard,
                screenspot_sources=tuple(
                    args.screenspot_source or ("windows", "web", "macos")
                ),
                include_internal_trajectories_first_shard=(
                    not args.no_internal_trajectories
                ),
                trajectory_paths=tuple(args.trajectory or ()),
                download_osworld_manifest=not args.skip_osworld_manifest,
                osworld_archives_per_shard=args.osworld_archives_per_shard,
                osworld_archive_transition_limit_per_shard=(
                    args.osworld_archive_transition_limit_per_shard
                ),
                osworld_archive_candidate_multiplier=(
                    args.osworld_archive_candidate_multiplier
                ),
                cache_dir=args.cache_dir,
                cache_budget_bytes=max(0, int(args.cache_budget_gb * (1 << 30))),
                stage_remote_archives=args.stage_archives,
                output_dir=args.output_dir,
                state_path=args.state_path,
                resume=not args.no_resume,
            )
        )
        print(json.dumps(longrun_result.asdict(), indent=2))
        return 0 if longrun_result.success else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


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


def _launch_dashboard(args: argparse.Namespace) -> int:
    workspace_root = Path.cwd()
    dashboard_dir = workspace_root / "apps" / "dashboard"
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to launch the dashboard UI")
    if not dashboard_dir.exists():
        raise RuntimeError("apps/dashboard was not found")

    node_modules = dashboard_dir / "node_modules"
    if not args.skip_npm_install and not node_modules.exists():
        subprocess.run([npm, "install"], cwd=dashboard_dir, check=True)

    api_command = [
        sys.executable,
        "-m",
        "agentos_orchestrator",
        "--policy",
        args.policy,
        "--state",
        args.state,
        "--memory",
        args.memory,
        "serve-dashboard",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    ui_command = [
        npm,
        "run",
        "dev",
        "--",
        "--host",
        args.host,
        "--port",
        str(args.ui_port),
    ]
    api_process = subprocess.Popen(api_command, cwd=workspace_root)
    ui_env = os.environ.copy()
    ui_env["VITE_AGENTOS_API_BASE"] = f"http://{args.host}:{args.api_port}"
    ui_process = subprocess.Popen(
        ui_command,
        cwd=dashboard_dir,
        env=ui_env,
    )
    url = f"http://{args.host}:{args.ui_port}/"
    if not args.no_browser:
        time.sleep(2)
        webbrowser.open(url)
    print(
        json.dumps(
            {
                "status": "running",
                "api": f"http://{args.host}:{args.api_port}",
                "ui": url,
                "api_pid": api_process.pid,
                "ui_pid": ui_process.pid,
            },
            indent=2,
        )
    )
    try:
        while True:
            api_code = api_process.poll()
            ui_code = ui_process.poll()
            if api_code is not None or ui_code is not None:
                return int(api_code if api_code is not None else ui_code or 0)
            time.sleep(1)
    except KeyboardInterrupt:
        for process in (api_process, ui_process):
            if process.poll() is None:
                process.terminate()
        return 130


def _daemon(args: argparse.Namespace) -> int:
    manager = DaemonManager(Path.cwd(), sys.executable)
    if args.daemon_command == "status":
        print(json.dumps(asdict(manager.status()), indent=2))
        return 0
    if args.daemon_command == "stop":
        print(json.dumps(asdict(manager.stop()), indent=2))
        return 0
    if args.daemon_command == "restart":
        manager.stop()
        args.daemon_command = "start"
        if not hasattr(args, "host"):
            args.host = "127.0.0.1"
            args.api_port = 8000
            args.ui_port = 5173
            args.skip_npm_install = True
            args.open_browser = False
    if args.daemon_command == "start":
        record = manager.start(
            host=args.host,
            api_port=args.api_port,
            ui_port=args.ui_port,
            policy=args.policy,
            state=args.state,
            memory=args.memory,
            skip_npm_install=args.skip_npm_install,
            open_browser=args.open_browser,
        )
        print(json.dumps(asdict(record), indent=2))
        return 0
    raise ValueError(f"Unknown daemon command: {args.daemon_command}")


def _json_or_text(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
