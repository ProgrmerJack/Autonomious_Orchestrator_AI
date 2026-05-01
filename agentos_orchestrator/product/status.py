from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.cognition.adaptation_readiness import (
    collect_adaptation_readiness,
)


@dataclass(slots=True)
class SetupCheck:
    check_id: str
    label: str
    status: str
    detail: str
    required: bool = True
    repair_hint: str = ""


@dataclass(slots=True)
class ProviderStatus:
    provider_id: str
    label: str
    kind: str
    configured: bool
    detail: str


@dataclass(slots=True)
class ChannelStatus:
    channel_id: str
    label: str
    endpoint: str
    configured: bool
    detail: str


@dataclass(slots=True)
class ProductStatus:
    checks: list[SetupCheck]
    providers: list[ProviderStatus]
    channels: list[ChannelStatus]
    benchmarks: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def collect_product_status(
    workspace_root: str | Path,
    policy_path: str | Path,
    state_path: str | Path,
    memory_path: str | Path,
    eval_snapshot: dict[str, Any] | None = None,
) -> ProductStatus:
    root = Path(workspace_root)
    checks = setup_checks(root, policy_path, state_path, memory_path)
    providers = provider_statuses()
    channels = channel_statuses()
    return ProductStatus(
        checks=checks,
        providers=providers,
        channels=channels,
        benchmarks=benchmark_status(
            checks,
            providers,
            channels,
            root,
            eval_snapshot,
        ),
    )


def setup_checks(
    workspace_root: Path,
    policy_path: str | Path,
    state_path: str | Path,
    memory_path: str | Path,
) -> list[SetupCheck]:
    dashboard_dir = workspace_root / "apps" / "dashboard"
    policy = Path(policy_path)
    state = Path(state_path)
    memory = Path(memory_path)
    checks = [
        _check(
            "python",
            "Python runtime",
            sys.version_info >= (3, 11),
            (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            repair_hint=("Install Python 3.11+ and rerun scripts/install-agentos.ps1."),
        ),
        _check(
            "policy-file",
            "Permission policy",
            policy.exists(),
            str(policy),
            repair_hint=(
                "Create or select a policy JSON file before launching AgentOS."
            ),
        ),
        _policy_check(policy),
        _check(
            "state-path",
            "Durable state path",
            _path_parent_writable(state),
            str(state),
            repair_hint=(
                "Create the .agentos directory or choose a writable state path."
            ),
        ),
        _check(
            "memory-path",
            "Durable memory path",
            _path_parent_writable(memory),
            str(memory),
            repair_hint=(
                "Create the .agentos directory or choose a writable memory path."
            ),
        ),
        _check(
            "dashboard-package",
            "Dashboard package",
            (dashboard_dir / "package.json").exists(),
            str(dashboard_dir),
            repair_hint=(
                "Restore apps/dashboard/package.json from the project template."
            ),
        ),
        _check(
            "npm",
            "Node package manager",
            shutil.which("npm") is not None,
            shutil.which("npm") or "npm was not found on PATH",
            repair_hint="Install Node.js LTS so npm is available on PATH.",
        ),
        _check(
            "node-modules",
            "Dashboard dependencies",
            (dashboard_dir / "node_modules").exists(),
            str(dashboard_dir / "node_modules"),
            required=False,
            repair_hint="Run npm install from apps/dashboard.",
        ),
        _check(
            "launcher",
            "One-command launcher",
            (workspace_root / "AgentOS.cmd").exists(),
            str(workspace_root / "AgentOS.cmd"),
            repair_hint=("Run scripts/install-agentos.ps1 to create the launcher."),
        ),
    ]
    return checks


def provider_statuses() -> list[ProviderStatus]:
    providers = [
        ("openai", "OpenAI", "cloud", ("OPENAI_API_KEY",)),
        ("anthropic", "Anthropic", "cloud", ("ANTHROPIC_API_KEY",)),
        (
            "google",
            "Google Gemini",
            "cloud",
            ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        ),
        ("openrouter", "OpenRouter", "cloud", ("OPENROUTER_API_KEY",)),
        ("azure-openai", "Azure OpenAI", "cloud", ("AZURE_OPENAI_API_KEY",)),
        ("ollama", "Ollama", "local", ("OLLAMA_HOST",)),
        ("lm-studio", "LM Studio", "local", ("LM_STUDIO_BASE_URL",)),
    ]
    statuses: list[ProviderStatus] = []
    for provider_id, label, kind, env_names in providers:
        configured = any(os.environ.get(name) for name in env_names)
        detail = ", ".join(env_names)
        statuses.append(
            ProviderStatus(
                provider_id=provider_id,
                label=label,
                kind=kind,
                configured=configured,
                detail=detail,
            )
        )
    return statuses


def channel_statuses() -> list[ChannelStatus]:
    return [
        ChannelStatus(
            channel_id="dashboard-command",
            label="Dashboard command console",
            endpoint="/channels/command",
            configured=True,
            detail="Local UI command route",
        ),
        ChannelStatus(
            channel_id="generic-webhook",
            label="Generic webhook",
            endpoint="/channels/generic",
            configured=True,
            detail="Accepts sender_id, text, and metadata",
        ),
        ChannelStatus(
            channel_id="telegram",
            label="Telegram webhook",
            endpoint="/channels/telegram",
            configured=True,
            detail="Accepts Telegram-compatible update payloads",
        ),
        ChannelStatus(
            channel_id="slack",
            label="Slack",
            endpoint="/channels/slack",
            configured=bool(os.environ.get("SLACK_BOT_TOKEN")),
            detail=(
                "Accepts Slack event payloads; set SLACK_BOT_TOKEN for bot delivery"
            ),
        ),
        ChannelStatus(
            channel_id="discord",
            label="Discord",
            endpoint="/channels/discord",
            configured=bool(os.environ.get("DISCORD_BOT_TOKEN")),
            detail=(
                "Accepts Discord message payloads; set "
                "DISCORD_BOT_TOKEN for bot delivery"
            ),
        ),
    ]


def benchmark_status(
    checks: list[SetupCheck],
    providers: list[ProviderStatus],
    channels: list[ChannelStatus],
    workspace_root: Path | None = None,
    eval_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = [check for check in checks if check.required]
    passed_required = [check for check in required if check.status == "pass"]
    configured_providers = [item for item in providers if item.configured]
    configured_channels = [item for item in channels if item.configured]
    readiness = len(passed_required) / max(len(required), 1)
    eval_passed = bool((eval_snapshot or {}).get("passed", True))
    trace_count = _golden_trace_count(workspace_root)
    adaptation = (
        collect_adaptation_readiness(workspace_root).asdict()
        if workspace_root is not None
        else {}
    )
    return {
        "readiness_score": round(readiness, 3),
        "required_checks_passed": len(passed_required),
        "required_checks_total": len(required),
        "configured_providers": len(configured_providers),
        "configured_channels": len(configured_channels),
        "golden_traces": trace_count,
        "release_gate": readiness >= 1 and eval_passed and trace_count > 0,
        "evals_passed": eval_passed,
        "evals": eval_snapshot or {},
        "adaptation_readiness": adaptation,
    }


def _check(
    check_id: str,
    label: str,
    passed: bool,
    detail: str,
    required: bool = True,
    repair_hint: str = "",
) -> SetupCheck:
    return SetupCheck(
        check_id=check_id,
        label=label,
        status="pass" if passed else "fail",
        detail=detail,
        required=required,
        repair_hint="" if passed else repair_hint,
    )


def _policy_check(policy_path: Path) -> SetupCheck:
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SetupCheck(
            check_id="policy-safety",
            label="Policy safety posture",
            status="fail",
            detail=str(exc),
            repair_hint=(
                "Use a valid JSON policy with default deny and approval gates."
            ),
        )
    requires = set(payload.get("require_approval", {}).get("actions", []))
    default_deny = payload.get("default") == "deny"
    safe = default_deny and "os.act" in requires
    return SetupCheck(
        check_id="policy-safety",
        label="Policy safety posture",
        status="pass" if safe else "fail",
        detail=("default deny with os.act approval" if safe else "tighten policy"),
        repair_hint=(
            "Set default to deny and require approval for os.act." if not safe else ""
        ),
    )


def _path_parent_writable(path: Path) -> bool:
    parent = path.parent
    if parent.exists():
        return os.access(parent, os.W_OK)
    existing = parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return os.access(existing, os.W_OK)


def _golden_trace_count(workspace_root: Path | None) -> int:
    if workspace_root is None:
        return 0
    trace_dir = workspace_root / "benchmarks" / "golden_traces"
    if not trace_dir.exists():
        return 0
    return len(list(trace_dir.glob("*.json")))
