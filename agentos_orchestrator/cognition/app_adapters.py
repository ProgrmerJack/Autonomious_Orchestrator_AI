"""App-family adapter registry for practical universal OS control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentos_orchestrator.app_family_registry import adapter_specs
from agentos_orchestrator.os_control.base import UiAction

from .capability_profile import CapabilityProfile
from .verification_contracts import (
    VerificationContract,
    ensure_verification_contract,
)


@dataclass(slots=True)
class AdapterContext:
    family: str
    preferred_channels: list[str]
    affordance_hints: list[str]
    verification_contracts: list[str]
    repair_recipes: list[str]
    risk_notes: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "preferred_channels": list(self.preferred_channels),
            "affordance_hints": list(self.affordance_hints),
            "verification_contracts": list(self.verification_contracts),
            "repair_recipes": list(self.repair_recipes),
            "risk_notes": list(self.risk_notes),
        }


@dataclass(slots=True)
class AppFamilyAdapter:
    family: str
    preferred_channels: list[str]
    affordance_hints: list[str]
    verification_contracts: list[str]
    repair_recipes: list[str]

    def matches(self, profile: CapabilityProfile) -> bool:
        return profile.app_family == self.family

    def context(
        self,
        profile: CapabilityProfile,
        objective: str = "",
    ) -> AdapterContext:
        del objective
        return AdapterContext(
            family=self.family,
            preferred_channels=_ordered_channels(
                self.preferred_channels,
                profile.control_channels,
            ),
            affordance_hints=list(self.affordance_hints),
            verification_contracts=list(self.verification_contracts),
            repair_recipes=list(self.repair_recipes),
            risk_notes=list(profile.risks),
        )

    def enrich_action(
        self,
        action: UiAction,
        profile: CapabilityProfile,
        objective: str = "",
    ) -> UiAction:
        context = self.context(profile, objective)
        action.metadata.setdefault("adapter_family", self.family)
        action.metadata.setdefault("adapter_context", context.to_prompt_dict())
        action.metadata.setdefault(
            "control_channel",
            context.preferred_channels[0],
        )
        if "verification_contract" not in action.metadata:
            contract = self._contract_for_action(action)
            if contract is not None:
                action.metadata["verification_contract"] = contract.asdict()
        ensure_verification_contract(action)
        return action

    def _contract_for_action(
        self,
        action: UiAction,
    ) -> VerificationContract | None:
        selector = str(action.selector or "").strip().lower()
        metadata = dict(action.metadata or {})
        if self.family == "email" and action.action_type in {"click", "invoke"}:
            if "send" in selector:
                return VerificationContract(
                    kind="send_outcome",
                    expected=("The final email state proves the message was sent."),
                    target=action.selector,
                    metadata={
                        "recipient": str(metadata.get("recipient") or ""),
                        "attachment": str(metadata.get("attachment") or ""),
                        "expected_state": "sent",
                    },
                )
        if self.family == "calendar" and action.action_type in {"click", "invoke"}:
            if any(token in selector for token in ("invite", "meeting", "send")):
                return VerificationContract(
                    kind="invite_outcome",
                    expected=(
                        "The final calendar state proves the invite was created."
                    ),
                    target=action.selector,
                    metadata={
                        "event_title": str(metadata.get("event_title") or ""),
                        "expected_state": "invited",
                    },
                )
        if self.family == "settings" and action.action_type in {"click", "invoke"}:
            if any(
                token in selector for token in ("toggle", "night light", "nightlight")
            ):
                setting_name, setting_state = _settings_toggle_target(action)
                return VerificationContract(
                    kind="toggle_state",
                    expected=(
                        "The final Settings state proves the requested toggle state."
                    ),
                    target=action.selector,
                    metadata={
                        "setting_name": setting_name,
                        "expected_state": setting_state or "on",
                    },
                )
        if self.family == "file_dialog" and action.action_type in {
            "type",
            "hotkey",
        }:
            return VerificationContract(
                kind="field_contains",
                expected=("The file dialog field contains the requested path or name."),
                target=action.selector,
                value=str(action.value or ""),
            )
        if self.family == "terminal" and action.action_type in {
            "type",
            "hotkey",
        }:
            return VerificationContract(
                kind="receipt_success",
                expected=("The terminal or backend receipt reports command progress."),
                target=action.selector,
                required=False,
            )
        if self.family == "browser" and action.action_type in {
            "click",
            "select",
        }:
            return VerificationContract(
                kind="state_changed",
                expected=("The page state, focused element, or active tab changes."),
                target=action.selector,
            )
        return None


def _settings_toggle_target(action: UiAction) -> tuple[str, str]:
    metadata = dict(action.metadata or {})
    setting_name = str(metadata.get("setting_name") or "").strip()
    setting_state = str(metadata.get("setting_state") or "").strip().lower()
    if setting_name and setting_state:
        return setting_name, setting_state
    left, _, right = str(action.value or "").partition(":")
    return left.strip(), right.strip().lower()


class AdapterRegistry:
    """Select app-family adapters without hard-coding one-off handlers."""

    def __init__(self, adapters: list[AppFamilyAdapter] | None = None) -> None:
        self.adapters = adapters or default_adapters()

    def select(self, profile: CapabilityProfile) -> AppFamilyAdapter:
        for adapter in self.adapters:
            if adapter.matches(profile):
                return adapter
        return self.adapters_by_family()["unknown"]

    def context_for(
        self,
        profile: CapabilityProfile,
        objective: str = "",
    ) -> AdapterContext:
        return self.select(profile).context(profile, objective)

    def enrich_action(
        self,
        action: UiAction,
        profile: CapabilityProfile,
        objective: str = "",
    ) -> UiAction:
        return self.select(profile).enrich_action(action, profile, objective)

    def adapters_by_family(self) -> dict[str, AppFamilyAdapter]:
        return {adapter.family: adapter for adapter in self.adapters}

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "family": adapter.family,
                "preferred_channels": list(adapter.preferred_channels),
                "affordance_hints": list(adapter.affordance_hints),
                "verification_contracts": list(adapter.verification_contracts),
                "repair_recipes": list(adapter.repair_recipes),
            }
            for adapter in self.adapters
        ]


def default_adapters() -> list[AppFamilyAdapter]:
    return [_adapter(*spec) for spec in DEFAULT_ADAPTER_SPECS]


DEFAULT_ADAPTER_SPECS = adapter_specs()


def _adapter(
    family: str,
    channels: list[str],
    hints: list[str],
    contracts: list[str],
    repairs: list[str],
) -> AppFamilyAdapter:
    return AppFamilyAdapter(
        family=family,
        preferred_channels=channels,
        affordance_hints=hints,
        verification_contracts=contracts,
        repair_recipes=repairs,
    )


def _ordered_channels(preferred: list[str], available: list[str]) -> list[str]:
    ordered = [channel for channel in preferred if channel in available]
    ordered.extend(channel for channel in available if channel not in ordered)
    return ordered or ["explore"]


def adapter_families() -> list[str]:
    return [adapter.family for adapter in default_adapters()]
