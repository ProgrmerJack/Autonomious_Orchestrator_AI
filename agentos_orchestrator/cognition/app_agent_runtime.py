from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.app_family_registry import spec_for_family
from agentos_orchestrator.os_control.base import UiAction, UiNode

from .affordance_policy_memory import PersistentAffordancePolicyMemory
from .app_adapters import AdapterRegistry
from .capability_profile import CapabilityProfile


@dataclass(slots=True)
class AppAgentSkillPack:
    skill_pack_id: str
    family: str
    app_context: str
    app_signature: str
    preferred_channels: list[str]
    affordance_hints: list[str]
    verification_contracts: list[str]
    repair_recipes: list[str]
    risk_notes: list[str] = field(default_factory=list)
    memory_channels: list[str] = field(default_factory=list)
    recommended_mode: str = "ui"
    safe_windows: bool = False
    live_fire: bool = False
    dom_like: bool = False
    api_like: bool = False
    visual_heavy: bool = False
    launch_target: str = ""
    primary_selector: str = ""
    policy_action: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            "skill_pack_id": self.skill_pack_id,
            "family": self.family,
            "app_context": self.app_context,
            "app_signature": self.app_signature,
            "preferred_channels": list(self.preferred_channels),
            "affordance_hints": list(self.affordance_hints),
            "verification_contracts": list(self.verification_contracts),
            "repair_recipes": list(self.repair_recipes),
            "risk_notes": list(self.risk_notes),
            "memory_channels": list(self.memory_channels),
            "recommended_mode": self.recommended_mode,
            "safe_windows": self.safe_windows,
            "live_fire": self.live_fire,
            "dom_like": self.dom_like,
            "api_like": self.api_like,
            "visual_heavy": self.visual_heavy,
            "launch_target": self.launch_target,
            "primary_selector": self.primary_selector,
            "policy_action": dict(self.policy_action),
        }


@dataclass(slots=True)
class AppAgentSession:
    skill_pack: AppAgentSkillPack
    adapter_context: dict[str, Any]
    objective: str = ""
    nodes_seen: int = 0
    handoff_notes: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "skill_pack_id": self.skill_pack.skill_pack_id,
            "family": self.skill_pack.family,
            "app_context": self.skill_pack.app_context,
            "app_signature": self.skill_pack.app_signature,
            "preferred_channels": list(self.skill_pack.preferred_channels),
            "recommended_mode": self.skill_pack.recommended_mode,
            "safe_windows": self.skill_pack.safe_windows,
            "live_fire": self.skill_pack.live_fire,
            "dom_like": self.skill_pack.dom_like,
            "api_like": self.skill_pack.api_like,
            "visual_heavy": self.skill_pack.visual_heavy,
            "launch_target": self.skill_pack.launch_target,
            "primary_selector": self.skill_pack.primary_selector,
            "adapter_context": dict(self.adapter_context),
            "policy_action": dict(self.skill_pack.policy_action),
            "nodes_seen": self.nodes_seen,
            "handoff_notes": list(self.handoff_notes),
        }


class AppAgentRuntime:
    """Resolve durable per-app skill packs from capability and policy memory.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        adapter_registry: AdapterRegistry | None = None,
        policy_memory: PersistentAffordancePolicyMemory | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.adapter_registry = adapter_registry or AdapterRegistry()
        self.policy_memory = policy_memory or PersistentAffordancePolicyMemory(
            self.workspace_root,
        )

    def resolve(
        self,
        profile: CapabilityProfile,
        objective: str = "",
        nodes: list[UiNode] | None = None,
    ) -> AppAgentSession:
        adapter = self.adapter_registry.select(profile)
        adapter_context = adapter.context(profile, objective).to_prompt_dict()
        spec = spec_for_family(adapter.family)
        memory_channels = self.policy_memory.preferred_channels(
            profile.app_signature,
        )
        policy_action = self.policy_memory.recommend_action(
            profile.app_signature,
            objective,
            nodes,
        )
        preferred_channels = _ordered_unique(
            memory_channels,
            list(adapter_context.get("preferred_channels", [])),
            list(profile.control_channels),
        )
        if not preferred_channels:
            preferred_channels = ["explore"]
        handoff_notes: list[str] = []
        if memory_channels:
            handoff_notes.append(
                f"Reuse memory-backed control lane '{memory_channels[0]}'."
            )
        if policy_action is not None:
            handoff_notes.append(
                "Recall successful affordance "
                f"{policy_action.action_type}:{policy_action.selector}."
            )
        skill_pack = AppAgentSkillPack(
            skill_pack_id=f"{spec.family}:{profile.app_signature}",
            family=spec.family,
            app_context=spec.app_context,
            app_signature=profile.app_signature,
            preferred_channels=preferred_channels,
            affordance_hints=list(adapter_context.get("affordance_hints", [])),
            verification_contracts=list(
                adapter_context.get("verification_contracts", []),
            ),
            repair_recipes=list(adapter_context.get("repair_recipes", [])),
            risk_notes=list(profile.risks),
            memory_channels=memory_channels,
            recommended_mode=spec.recommended_mode,
            safe_windows=spec.safe_windows,
            live_fire=spec.live_fire,
            dom_like=spec.dom_like,
            api_like=spec.api_like,
            visual_heavy=spec.visual_heavy,
            launch_target=spec.launch_target,
            primary_selector=spec.primary_selector,
            policy_action=_policy_action_dict(policy_action),
        )
        return AppAgentSession(
            skill_pack=skill_pack,
            adapter_context=adapter_context,
            objective=objective,
            nodes_seen=len(nodes or []),
            handoff_notes=handoff_notes,
        )

    def enrich_action(
        self,
        action: UiAction,
        session: AppAgentSession,
        objective: str = "",
    ) -> UiAction:
        metadata = dict(action.metadata or {})
        metadata["app_agent"] = session.asdict()
        metadata["workflow_objective"] = objective or session.objective
        if session.skill_pack.preferred_channels:
            metadata.setdefault(
                "control_channel",
                session.skill_pack.preferred_channels[0],
            )
        metadata.setdefault("app_signature", session.skill_pack.app_signature)
        metadata.setdefault("adapter_family", session.skill_pack.family)
        metadata.setdefault("adapter_context", dict(session.adapter_context))
        return UiAction(
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            metadata=metadata,
        )

    def record_outcome(
        self,
        action: UiAction,
        *,
        verification_result: dict[str, Any],
        receipt: Any,
    ) -> None:
        session = dict(action.metadata.get("app_agent") or {})
        app_signature = str(session.get("app_signature") or "")
        if not app_signature:
            return
        objective = str(action.metadata.get("workflow_objective") or "")
        control_channel = str(
            action.metadata.get("control_channel")
            or action.metadata.get("control_route")
            or ""
        )
        success = bool(verification_result.get("matched", True))
        observed = str(
            verification_result.get("observed")
            or verification_result.get("reason")
            or ""
        )
        evidence = {
            "verification": dict(verification_result),
            "receipt": receipt,
            "app_agent": session,
        }
        self.policy_memory.record(
            app_signature,
            objective,
            action,
            success,
            control_channel=control_channel,
            observed=observed,
            evidence=evidence,
        )


def _ordered_unique(*groups: list[str]) -> list[str]:
    ordered: list[str] = []
    for group in groups:
        for item in group:
            value = str(item or "").strip()
            if not value or value in ordered:
                continue
            ordered.append(value)
    return ordered


def _policy_action_dict(action: UiAction | None) -> dict[str, Any]:
    if action is None:
        return {}
    return {
        "action_type": action.action_type,
        "selector": action.selector,
        "value": action.value,
        "metadata": dict(action.metadata or {}),
    }
