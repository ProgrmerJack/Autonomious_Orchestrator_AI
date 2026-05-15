"""Live-fire eval runner for practical universal OS control."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.app_family_registry import safe_windows_families
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.os_control.windows_family_adapter import (
    selector_matches_node,
)

from .active_inference import ActiveInferenceExplorer
from .app_adapters import AdapterRegistry
from .affordance_policy_memory import PersistentAffordancePolicyMemory
from .capability_profile import CapabilityProfiler
from .live_fire_eval_recipes import (
    abstract_state,
    actions_for_task,
    json_receipt,
    receipt_payload,
    snapshot_nodes,
)
from .mode_arbitration import ModeArbiter, ModeContext
from .os_eval_packs import (
    EvalTask,
    build_combined_live_fire_eval_pack,
    build_everyday_family_eval_pack,
    build_real_user_handoff_eval_pack,
    build_universal_app_eval_pack,
    normalize_eval_pack_name,
)
from .replay_debug import load_replay_debug
from .runtime_state import AgentRuntimeState, OutcomeEvaluation
from .safety_gates import FormalSafetyVerifier, SafetyPolicy
from .tool_executor import QuantAnalysisRequest, ToolExecutor
from .trajectory_recorder import TrajectoryRecorder
from .trajectory_training import TrajectoryTrainingBuilder
from .verification_contracts import (
    ensure_verification_contract,
    verify_action_contract,
)


WINDOWS_SAFE_SURFACES = safe_windows_families()
WINDOWS_SAFE_INTENTS = (
    "open_app",
    "find_target",
    "fill_form",
    "use_shortcut",
    "recover_modal",
    "stale_target_reground",
)
MILESTONE_TASK_TARGET = 50
MILESTONE_DURABLE_FAILURE_TARGET = 10


@dataclass(slots=True)
class LiveFireEvalConfig:
    run_id: str = ""
    max_tasks: int | None = None
    surfaces: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    pack: str = "universal"
    windows_safe_pack: bool = False
    repeat: int = 1
    promote_failures: bool = True
    promote_after: int = 1
    replay_limit: int = 10
    training_output: str = ""
    heldout_from: str = ""


@dataclass(slots=True)
class LiveFireTaskResult:
    task_id: str
    surface: str
    intent: str
    success: bool
    trajectory_path: str
    action_count: int
    failure_reason: str = ""
    promoted_trace_path: str = ""
    elapsed_ms: float = 0.0
    receipts: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _ActionContext:
    before_nodes: list[Any]
    before: Any
    profile: Any
    adapter_context: Any
    action: UiAction
    contract: Any
    mode_decision: Any


@dataclass(slots=True)
class LiveFireEvalRun:
    run_id: str
    backend: str
    success: bool
    task_count: int
    passed: int
    failed: int
    trajectory_paths: list[str]
    promoted_traces: list[str]
    replay_debug: dict[str, Any]
    training_summary: dict[str, Any]
    milestone: dict[str, Any]
    heldout_metrics: dict[str, Any]
    task_results: list[LiveFireTaskResult]
    summary_path: str = ""

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_results"] = [item.asdict() for item in self.task_results]
        return payload


class LiveFireEvalRunner:
    """Execute the universal app eval pack against a PC backend."""

    def __init__(
        self,
        backend: Any,
        workspace_root: str | Path,
        safety_verifier: FormalSafetyVerifier | None = None,
    ) -> None:
        self.backend = backend
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.artifact_root = self.workspace_root / "artifacts" / "live_fire_eval"
        self.profiler = CapabilityProfiler()
        self.adapters = AdapterRegistry()
        self.affordance_policies = PersistentAffordancePolicyMemory(self.workspace_root)
        self.explorer = ActiveInferenceExplorer(max_probes=2, random_seed=0)
        self.mode_arbiter = ModeArbiter()
        self.recorder = TrajectoryRecorder(self.workspace_root)
        self.tool_executor = ToolExecutor(self.workspace_root / ".agentos")
        self.safety_verifier = safety_verifier or FormalSafetyVerifier(
            SafetyPolicy(allowed_roots=[self.artifact_root])
        )

    def run(self, config: LiveFireEvalConfig | None = None) -> LiveFireEvalRun:
        config = config or LiveFireEvalConfig()
        run_id = config.run_id or _run_id()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        tasks = _repeat_tasks(_select_tasks(config), config.repeat)
        results = [self._run_task(run_id, task) for task in tasks]
        return self._finish_run(run_id, results, config)

    def _finish_run(
        self,
        run_id: str,
        results: list[LiveFireTaskResult],
        config: LiveFireEvalConfig,
    ) -> LiveFireEvalRun:
        promoted = self._promote_failures(run_id, results, config)
        trajectory_paths = [item.trajectory_path for item in results]
        replay = load_replay_debug(
            self.workspace_root,
            run_id=run_id,
            limit=max(config.replay_limit, len(results), 1),
        )
        training = self._write_training_dataset(
            run_id,
            trajectory_paths,
            config,
        )
        payload = LiveFireEvalRun(
            run_id=run_id,
            backend=_backend_name(self.backend),
            success=all(item.success for item in results),
            task_count=len(results),
            passed=sum(1 for item in results if item.success),
            failed=sum(1 for item in results if not item.success),
            trajectory_paths=trajectory_paths,
            promoted_traces=promoted,
            replay_debug=replay,
            training_summary=training,
            milestone=_milestone_payload(results, promoted),
            heldout_metrics=_heldout_metrics(results, config),
            task_results=results,
        )
        payload.summary_path = str(self._write_summary(payload))
        return payload

    def _run_task(self, run_id: str, task: EvalTask) -> LiveFireTaskResult:
        task_run_id = f"{run_id}_{task.task_id}"
        started = time.perf_counter()
        trajectory_path = self.recorder.start_run(task_run_id, task.objective)
        receipts: list[dict[str, Any]] = []
        verifications: list[dict[str, Any]] = []
        failure_reason = ""
        try:
            for action in actions_for_task(task, self.artifact_root, run_id):
                step = self._run_action_step(task, task_run_id, action)
                receipts.append(step["receipt"])
                verifications.append(step["verification"])
        finally:
            success, failure_reason = _task_outcome(task, verifications)
            self.recorder.finish_run(
                task_run_id,
                {
                    "success": success,
                    "task_id": task.task_id,
                    "failure_reason": failure_reason,
                },
            )
        return LiveFireTaskResult(
            task_id=task.task_id,
            surface=task.surface,
            intent=task.intent,
            success=success,
            trajectory_path=str(trajectory_path or ""),
            action_count=len(receipts),
            failure_reason=failure_reason,
            elapsed_ms=_elapsed_ms(started),
            receipts=receipts,
            verifications=verifications,
        )

    def _run_action_step(
        self,
        task: EvalTask,
        run_id: str,
        action: UiAction,
    ) -> dict[str, Any]:
        context = self._prepare_action_context(task, action)
        context.action = self._reground_action(
            context.action,
            context.before_nodes,
            task,
            context.profile,
        )
        context.contract = ensure_verification_contract(context.action)
        start = time.perf_counter()
        receipt_text, blocked = self._perform_or_block(
            context.action,
            task.objective,
        )
        if self._selector_not_found(receipt_text):
            regrounded = self._reground_action(
                context.action,
                context.before_nodes,
                task,
                context.profile,
                force=True,
            )
            if (
                regrounded.selector != context.action.selector
                or regrounded.action_type != context.action.action_type
            ):
                context.action = regrounded
                context.contract = ensure_verification_contract(context.action)
                receipt_text, blocked = self._perform_or_block(
                    context.action,
                    task.objective,
                )
        self._materialize_artifact(context.action, blocked=blocked)
        after_nodes = context.before_nodes if blocked else snapshot_nodes(self.backend)
        after = abstract_state(task.surface, after_nodes)
        verification = verify_action_contract(
            context.action,
            context.before,
            after,
            receipt_text,
        )
        self._record_action_context(
            context,
            run_id=run_id,
            after=after,
            receipt=receipt_text,
            task=task,
            verification=verification,
            latency_ms=_elapsed_ms(start),
        )
        self._record_policy_affordance(
            context,
            task=task,
            verification=verification,
            receipt=receipt_text,
        )
        return {
            "receipt": receipt_payload(receipt_text),
            "verification": verification.asdict(),
        }

    def _prepare_action_context(
        self,
        task: EvalTask,
        action: UiAction,
    ) -> _ActionContext:
        before_nodes = snapshot_nodes(self.backend)
        before = abstract_state(task.surface, before_nodes)
        profile = self.profiler.profile(before, nodes=before_nodes)
        adapter_context = self.adapters.context_for(profile, task.objective)
        enriched = self.adapters.enrich_action(action, profile, task.objective)
        contract = ensure_verification_contract(enriched)
        mode_context = ModeContext(
            perceived_element_count=len(before.elements),
            capability_profile=profile,
            adapter_context=adapter_context.to_prompt_dict(),
        )
        mode_decision = self.mode_arbiter.choose(
            task.objective,
            _Option(task.intent),
            before,
            AgentRuntimeState(objective=task.objective),
            mode_context,
        )
        return _ActionContext(
            before_nodes=before_nodes,
            before=before,
            profile=profile,
            adapter_context=adapter_context,
            action=enriched,
            contract=contract,
            mode_decision=mode_decision,
        )

    def _record_action_context(
        self,
        context: _ActionContext,
        *,
        run_id: str,
        after: Any,
        receipt: str,
        task: EvalTask,
        verification: Any,
        latency_ms: float,
    ) -> None:
        self.recorder.record_step(
            run_id=run_id,
            objective=task.objective,
            option_name=task.intent,
            before=context.before,
            after=after,
            action=context.action,
            expected_observation=context.contract.expected,
            receipt=receipt,
            outcome=_outcome_from_verification(verification),
            mode_decision=context.mode_decision,
            capability_profile=context.profile.to_prompt_dict(),
            adapter_context=context.adapter_context.to_prompt_dict(),
            verification_contract=context.contract.asdict(),
            verification_result=verification.asdict(),
            latency_ms=latency_ms,
        )

    def _perform_or_block(
        self,
        action: UiAction,
        objective: str,
    ) -> tuple[str, bool]:
        safety = self.safety_verifier.verify_action(
            action,
            objective=objective,
        )
        if not safety.allowed:
            return json_receipt(
                {
                    "status": "policy-blocked",
                    "reason": safety.reason,
                    "solver": safety.solver,
                }
            ), True
        if action.action_type == "tool":
            return self._run_tool(action, objective), False
        if action.action_type == "explore":
            return json_receipt({"status": "explored", "success": True}), False
        try:
            return str(self.backend.perform(action)), False
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return json_receipt({"status": "exception", "error": str(exc)}), False

    def _run_tool(self, action: UiAction, objective: str) -> str:
        del action
        result = self.tool_executor.run(
            QuantAnalysisRequest(
                objective=objective,
                code='print("RESULT: status=live_fire_tool_ok")',
                timeout_seconds=10,
            )
        )
        return json_receipt(
            {
                "status": "success" if result.success else "failed",
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
                "parsed_results": result.parsed_results,
            }
        )

    def _materialize_artifact(self, action: UiAction, blocked: bool = False) -> None:
        if blocked:
            return
        target = action.metadata.get("create_artifact_path")
        if not target:
            return
        path = Path(str(target)).resolve(strict=False)
        if self.artifact_root not in path.parents:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(action.metadata.get("artifact_content") or "AgentOS")
        path.write_text(content, encoding="utf-8")

    def _reground_action(
        self,
        action: UiAction,
        nodes: list[Any],
        task: EvalTask,
        profile: Any,
        force: bool = False,
    ) -> UiAction:
        if action.action_type in {"launch_app", "tool", "explore", "hotkey"}:
            return action
        if not force and self._selector_matches_nodes(action.selector, nodes):
            return action
        selector = self._surface_selector_for_nodes(
            task.surface,
            nodes,
            action_type=action.action_type,
        )
        grounded_probe = self.explorer.suggest_action(
            nodes,
            f"{task.surface} {task.intent} {task.objective}",
        )
        if not selector and grounded_probe is not None:
            selector = grounded_probe.selector
        if not selector:
            return action
        action_type = action.action_type
        if (
            action.action_type in {"click", "focus", "invoke"}
            and grounded_probe is not None
        ):
            if grounded_probe.action_type in {"click", "focus", "invoke"}:
                action_type = grounded_probe.action_type
            elif grounded_probe.action_type == "type" and not action.value:
                action_type = "focus"
        metadata = dict(action.metadata)
        metadata.setdefault("source", action.metadata.get("source", ""))
        metadata["regrounded"] = True
        metadata["regrounded_from"] = action.selector
        if grounded_probe is not None:
            metadata.setdefault(
                "rationale", grounded_probe.metadata.get("rationale", "")
            )
        grounded = UiAction(
            action_type=action_type,
            selector=selector,
            value=action.value,
            metadata=metadata,
        )
        contract = dict(grounded.metadata.get("verification_contract") or {})
        if contract:
            contract["target"] = selector
            grounded.metadata["verification_contract"] = contract
        return self.adapters.enrich_action(grounded, profile, task.objective)

    def _surface_selector_for_nodes(
        self,
        surface: str,
        nodes: list[Any],
        action_type: str = "",
    ) -> str:
        scored: list[tuple[float, str]] = []
        wants_text_entry = action_type in {"set_text", "type"}
        for node in nodes:
            role = str(getattr(node, "role", "")).lower()
            name = str(getattr(node, "name", "")).lower()
            selector = self._selector_for_node(node)
            score = 0.0
            if wants_text_entry:
                if role == "edit":
                    score += 1.1
                elif role == "document":
                    score += 0.7
                elif role in {"pane", "list", "tree"}:
                    score += 0.2
                elif role in {"button", "tab", "window"}:
                    score -= 0.3
            if surface == "browser":
                if role == "edit":
                    score += 0.5
                if any(token in name for token in {"address", "search", "url"}):
                    score += 0.6
            elif surface == "file_explorer":
                if wants_text_entry and role == "edit":
                    score += 0.65
                if role in {"tree", "list", "pane"}:
                    score += 0.45
                if any(
                    token in name
                    for token in (
                        {"address", "search", "path", "filename", "name"}
                        if wants_text_entry
                        else {"items", "files", "home", "folder"}
                    )
                ):
                    score += 0.5
            elif surface == "file_dialog":
                if role in {"edit", "document", "pane"}:
                    score += 0.4
                if any(
                    token in name
                    for token in {"file", "name", "filename", "address", "search"}
                ):
                    score += 0.65
            elif surface == "editor":
                if role in {"edit", "document", "pane"}:
                    score += 0.45
                if any(
                    token in name for token in {"editor", "text", "document", "canvas"}
                ):
                    score += 0.55
            elif surface == "terminal":
                if role in {"edit", "document", "pane"}:
                    score += 0.45
                if any(
                    token in name
                    for token in (
                        {
                            "prompt",
                            "command",
                            "input",
                            "terminal",
                            "powershell",
                            "console",
                        }
                        if wants_text_entry
                        else {"terminal", "powershell", "console", "command"}
                    )
                ):
                    score += 0.55
            if score > 0.0:
                scored.append((score, selector))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else ""

    @staticmethod
    def _selector_for_node(node: Any) -> str:
        name = str(getattr(node, "name", "") or "")
        if name:
            return f"name={name}"
        return str(getattr(node, "node_id", ""))

    @staticmethod
    def _selector_matches_nodes(selector: str, nodes: list[Any]) -> bool:
        for node in nodes:
            if selector_matches_node(selector, node):
                return True
        return False

    @staticmethod
    def _selector_not_found(receipt: str) -> bool:
        lower = receipt.lower()
        return "no ui element matched selector" in lower or "target not found" in lower

    def _record_policy_affordance(
        self,
        context: _ActionContext,
        *,
        task: EvalTask,
        verification: Any,
        receipt: str,
    ) -> None:
        if context.action.action_type in {"explore", "tool"}:
            return
        success = bool(verification.matched or not verification.required)
        if not success:
            return
        preferred_channel = ""
        if getattr(context.adapter_context, "preferred_channels", None):
            preferred_channel = str(context.adapter_context.preferred_channels[0])
        self.affordance_policies.record(
            context.profile.app_signature,
            task.objective,
            context.action,
            success=True,
            control_channel=str(
                context.action.metadata.get("control_channel")
                or preferred_channel
                or (
                    context.profile.control_channels[0]
                    if context.profile.control_channels
                    else ""
                )
            ),
            observed=str(getattr(verification, "observed", "") or receipt)[:500],
            evidence={
                "live_fire": True,
                "surface": task.surface,
                "intent": task.intent,
                "backend": _backend_name(self.backend),
                "source": context.action.metadata.get("source", "live_fire_eval"),
            },
        )

    def _promote_failures(
        self,
        run_id: str,
        results: list[LiveFireTaskResult],
        config: LiveFireEvalConfig,
    ) -> list[str]:
        if not config.promote_failures:
            return []
        promoted = []
        for result in results:
            if result.success:
                continue
            count = _record_failure(self.workspace_root, result)
            if count >= max(1, config.promote_after):
                promoted.append(
                    str(_promote_failure(run_id, result, self.workspace_root))
                )
        return promoted

    def _write_training_dataset(
        self,
        run_id: str,
        trajectory_paths: list[str],
        config: LiveFireEvalConfig,
    ) -> dict[str, Any]:
        output = config.training_output or str(
            self.workspace_root / ".agentos" / "training" / f"{run_id}.jsonl"
        )
        return TrajectoryTrainingBuilder(self.workspace_root).write_dataset(
            output_path=output,
            paths=list(trajectory_paths),
        )

    def _write_summary(self, result: LiveFireEvalRun) -> Path:
        target = self.workspace_root / ".agentos" / "live_fire_eval"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{result.run_id}.json"
        path.write_text(
            json.dumps(result.asdict(), indent=2),
            encoding="utf-8",
        )
        return path


@dataclass(slots=True)
class _Option:
    name: str


def _select_tasks(config: LiveFireEvalConfig) -> list[EvalTask]:
    pack = normalize_eval_pack_name(config.pack)
    if pack == "handoff":
        tasks = build_real_user_handoff_eval_pack().tasks
    elif pack == "everyday":
        tasks = build_everyday_family_eval_pack().tasks
    elif pack == "combined":
        tasks = build_combined_live_fire_eval_pack().tasks
    else:
        tasks = build_universal_app_eval_pack().tasks
    surfaces = config.surfaces
    intents = config.intents
    if config.windows_safe_pack:
        surfaces = surfaces or WINDOWS_SAFE_SURFACES
        intents = intents or WINDOWS_SAFE_INTENTS
    if surfaces:
        tasks = [task for task in tasks if task.surface in surfaces]
    if intents:
        tasks = [task for task in tasks if task.intent in intents]
    if config.max_tasks is not None:
        tasks = tasks[: max(0, config.max_tasks)]
    return tasks


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", type(backend).__name__))


def _repeat_tasks(tasks: list[EvalTask], repeat: int) -> list[EvalTask]:
    rounds = max(1, int(repeat or 1))
    if rounds == 1:
        return list(tasks)
    expanded: list[EvalTask] = []
    for round_index in range(1, rounds + 1):
        for task in tasks:
            expanded.append(_repeat_task(task, round_index))
    return expanded


def _repeat_task(task: EvalTask, round_index: int) -> EvalTask:
    return EvalTask(
        task_id=f"{task.task_id}_r{round_index}",
        surface=task.surface,
        intent=task.intent,
        objective=task.objective,
        expected_verifications=list(task.expected_verifications),
        failure_modes=list(task.failure_modes),
        metrics=list(task.metrics),
        live_fire=task.live_fire,
        workflow_surfaces=list(task.workflow_surfaces),
    )


def _milestone_payload(
    results: list[LiveFireTaskResult],
    promoted: list[str],
) -> dict[str, Any]:
    unsafe_blocks = sum(
        1
        for result in results
        for receipt in result.receipts
        if str(receipt.get("status") or "") == "policy-blocked"
    )
    return {
        "task_target": MILESTONE_TASK_TARGET,
        "durable_failure_target": MILESTONE_DURABLE_FAILURE_TARGET,
        "task_count": len(results),
        "durable_failures": len(promoted),
        "unsafe_action_blocks": unsafe_blocks,
        "safe_pack_ready": len(results) >= MILESTONE_TASK_TARGET,
        "durable_failures_ready": (len(promoted) >= MILESTONE_DURABLE_FAILURE_TARGET),
        "unsafe_blocks_ok": unsafe_blocks == 0,
    }


def _heldout_metrics(
    results: list[LiveFireTaskResult],
    config: LiveFireEvalConfig,
) -> dict[str, Any]:
    task_count = len(results)
    passed = sum(1 for item in results if item.success)
    failed = task_count - passed
    return {
        "metric": "heldout_live_fire_success_rate",
        "heldout": True,
        "heldout_from": config.heldout_from,
        "task_count": task_count,
        "passed": passed,
        "failed": failed,
        "success_rate": round(passed / task_count, 4) if task_count else 0.0,
        "surface_count": len({item.surface for item in results}),
        "intent_count": len({item.intent for item in results}),
        "by_surface": _group_success(results, "surface"),
        "by_intent": _group_success(results, "intent"),
        "failure_reasons": sorted(
            {item.failure_reason for item in results if item.failure_reason}
        ),
    }


def _group_success(
    results: list[LiveFireTaskResult],
    key: str,
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[LiveFireTaskResult]] = {}
    for item in results:
        grouped.setdefault(str(getattr(item, key)), []).append(item)
    payload: dict[str, dict[str, int | float]] = {}
    for group, items in sorted(grouped.items()):
        passed = sum(1 for item in items if item.success)
        total = len(items)
        payload[group] = {
            "task_count": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": round(passed / total, 4) if total else 0.0,
        }
    return payload


def _outcome_from_verification(result: Any) -> OutcomeEvaluation:
    return OutcomeEvaluation(
        expected=result.expected,
        observed=result.observed,
        matched=bool(result.matched or not result.required),
        failure_reason=result.reason or None,
        new_blocker=result.reason or None,
        suggested_repair=(
            "Re-ground and retry with a fresh snapshot." if result.reason else None
        ),
    )


def _task_outcome(
    task: EvalTask,
    verifications: list[dict[str, Any]],
) -> tuple[bool, str]:
    if task.intent == "approval_boundary":
        blocked = any(_is_policy_block(item) for item in verifications)
        return blocked, "" if blocked else "approval boundary was not blocked"
    failures = [item for item in verifications if _required_failed(item)]
    if not failures:
        return True, ""
    return False, str(failures[0].get("reason") or "verification failed")


def _required_failed(item: dict[str, Any]) -> bool:
    return bool(item.get("required", True)) and not bool(item.get("matched"))


def _is_policy_block(item: dict[str, Any]) -> bool:
    return "policy" in str(item.get("observed") or item.get("reason") or "")


def _record_failure(root: Path, result: LiveFireTaskResult) -> int:
    path = root / ".agentos" / "live_fire_eval_failures.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    key = _failure_key(result)
    record = dict(payload.get(key) or {})
    record["count"] = int(record.get("count") or 0) + 1
    record["task_id"] = result.task_id
    record["failure_reason"] = result.failure_reason
    record["last_seen"] = time.time()
    payload[key] = record
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return int(record["count"])


def _promote_failure(
    run_id: str,
    result: LiveFireTaskResult,
    root: Path,
) -> Path:
    trace_dir = root / "benchmarks" / "golden_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_id = f"live_fire_{result.task_id}_{_short_hash(result.failure_reason)}"
    replay = load_replay_debug(
        root,
        run_id=f"{run_id}_{result.task_id}",
        limit=1,
    )
    path = trace_dir / f"{trace_id}.json"
    path.write_text(
        json.dumps(
            _golden_trace_payload(trace_id, run_id, result, replay),
            indent=2,
        ),
        encoding="utf-8",
    )
    result.promoted_trace_path = str(path)
    return path


def _golden_trace_payload(
    trace_id: str,
    run_id: str,
    result: LiveFireTaskResult,
    replay_debug: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "name": f"Live-fire eval failure: {result.task_id}",
        "scope": "universal OS live-fire eval",
        "objective": f"Prevent regression for {result.task_id}.",
        "failure_modes": [result.failure_reason or "live_fire_eval_failure"],
        "steps": [
            {"kind": "live_fire.execute", "expect": "Task is run safely."},
            {"kind": "trajectory.record", "expect": "Failure is recorded."},
            {"kind": "debug.replay", "expect": "Replay explains failure."},
            {"kind": "shadow.train", "expect": "Trajectory trains heads."},
        ],
        "expectations": [
            "The failure remains reproducible from the live-fire trace.",
            "A future run either verifies the task or records a repair.",
        ],
        "metadata": {
            "run_id": run_id,
            "surface": result.surface,
            "intent": result.intent,
            "trajectory_path": result.trajectory_path,
        },
        "replay_debug": replay_debug,
    }


def _failure_key(result: LiveFireTaskResult) -> str:
    task_id = _canonical_task_id(result.task_id)
    return _short_hash(f"{task_id}:{result.failure_reason}")


def _canonical_task_id(task_id: str) -> str:
    return re.sub(r"_r\d+$", "", str(task_id or ""))


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _run_id() -> str:
    return f"live_eval_{int(time.time())}"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
