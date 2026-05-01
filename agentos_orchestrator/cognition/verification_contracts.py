"""Typed verification contracts for universal OS-agent actions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos_orchestrator.os_control.base import UiAction

from .abstract_world_model import AbstractUIState
from .runtime_state import diff_ui_states


CONTRACT_KINDS = {
    "receipt_success",
    "field_contains",
    "modal_closed",
    "file_exists",
    "export_hash_changed",
    "clipboard_contains",
    "window_title_changed",
    "tab_focused",
    "process_launched",
    "state_changed",
}


@dataclass(slots=True)
class VerificationContract:
    kind: str
    expected: str
    target: str = ""
    value: str = ""
    path: str = ""
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationResult:
    kind: str
    matched: bool
    expected: str
    observed: str
    required: bool = True
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_verification_contract(action: UiAction) -> VerificationContract:
    raw = action.metadata.get("verification_contract")
    if isinstance(raw, VerificationContract):
        contract = raw
    elif isinstance(raw, dict):
        contract = _contract_from_dict(raw)
    else:
        contract = default_contract_for_action(action)
    contract.metadata["action_type"] = action.action_type
    contract.metadata["selector"] = action.selector
    if action.metadata.get("regrounded"):
        contract.metadata["regrounded"] = True
    action.metadata["verification_contract"] = contract.asdict()
    action.metadata.setdefault("expected_observation", contract.expected)
    return contract


def default_contract_for_action(action: UiAction) -> VerificationContract:
    for builder in (
        _path_contract,
        _type_contract,
        _escape_contract,
        _tool_contract,
        _explore_contract,
    ):
        contract = builder(action)
        if contract is not None:
            return contract
    return VerificationContract(
        kind="state_changed",
        expected=(f"The UI responds to {action.action_type} on {action.selector}."),
        target=action.selector,
    )


def _path_contract(action: UiAction) -> VerificationContract | None:
    metadata = action.metadata
    if not metadata.get("path") and not metadata.get("file_path"):
        return None
    path = str(metadata.get("path") or metadata.get("file_path"))
    return VerificationContract(
        kind="file_exists",
        expected=f"The file exists at {path}.",
        path=path,
    )


def _type_contract(action: UiAction) -> VerificationContract | None:
    if action.action_type != "type":
        return None
    value = str(action.value or action.metadata.get("text") or "")
    return VerificationContract(
        kind="field_contains",
        expected="The focused field contains the typed value.",
        target=action.selector,
        value=value,
    )


def _escape_contract(action: UiAction) -> VerificationContract | None:
    if action.action_type != "hotkey":
        return None
    if str(action.value).lower() not in {"{esc}", "escape"}:
        return None
    return VerificationContract(
        kind="modal_closed",
        expected=("The active modal closes or focus returns to the parent UI."),
        target=action.selector,
    )


def _tool_contract(action: UiAction) -> VerificationContract | None:
    if action.action_type != "tool":
        return None
    return VerificationContract(
        kind="receipt_success",
        expected="The tool receipt reports success.",
        target=action.selector,
    )


def _explore_contract(action: UiAction) -> VerificationContract | None:
    if action.action_type != "explore":
        return None
    return VerificationContract(
        kind="state_changed",
        expected="Exploration produces observable state or probe evidence.",
        target=action.selector,
        required=False,
    )


def verify_action_contract(
    action: UiAction,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
) -> VerificationResult:
    contract = ensure_verification_contract(action)
    receipt_info = _parse_receipt(receipt)
    handlers = {
        "receipt_success": _verify_receipt_success,
        "field_contains": _verify_field_contains,
        "modal_closed": _verify_modal_closed,
        "file_exists": _verify_file_exists,
        "export_hash_changed": _verify_export_hash_changed,
        "clipboard_contains": _verify_clipboard_contains,
        "window_title_changed": _verify_window_title_changed,
        "tab_focused": _verify_tab_focused,
        "process_launched": _verify_process_launched,
        "state_changed": _verify_state_changed,
    }
    handler = handlers.get(contract.kind, _verify_state_changed)
    return handler(contract, before, after, receipt, receipt_info)


def _contract_from_dict(raw: dict[str, Any]) -> VerificationContract:
    kind = str(raw.get("kind") or "state_changed")
    if kind not in CONTRACT_KINDS:
        kind = "state_changed"
    return VerificationContract(
        kind=kind,
        expected=str(raw.get("expected") or "The action outcome is verified."),
        target=str(raw.get("target") or ""),
        value=str(raw.get("value") or ""),
        path=str(raw.get("path") or ""),
        required=bool(raw.get("required", True)),
        metadata=dict(raw.get("metadata") or {}),
    )


def _verify_receipt_success(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, after
    matched = _receipt_success(receipt, receipt_info)
    return _result(
        contract,
        matched,
        receipt[:500],
        "receipt did not report success",
    )


def _verify_field_contains(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before
    observed = " ".join(item.semantic_label for item in after.elements).lower()
    value = contract.value.lower().strip()
    receipt_text = f"{receipt} {json.dumps(receipt_info, sort_keys=True)}".lower()
    matched = bool(value) and (value in observed or value in receipt_text)
    if not value and _receipt_success(receipt, receipt_info):
        matched = True
    if not matched and _receipt_reports_text_entry_success(contract, receipt_info):
        matched = True
    return _result(
        contract,
        matched,
        observed[:500],
        "typed value was not observed",
    )


def _verify_modal_closed(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del receipt, receipt_info
    was_open = bool(before.active_modal)
    matched = was_open and not after.active_modal
    if not was_open:
        matched = after.layout_mode != "modal_open"
    observed = f"before_modal={before.active_modal}; after_modal={after.active_modal}"
    return _result(contract, matched, observed, "modal is still active")


def _verify_file_exists(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, after, receipt_info
    matched = bool(contract.path) and Path(contract.path).exists()
    observed = f"path={contract.path}; exists={matched}; receipt={receipt[:250]}"
    return _result(contract, matched, observed, "expected file does not exist")


def _verify_export_hash_changed(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, after
    old_hash = str(contract.metadata.get("before_hash") or "")
    new_hash = str(
        receipt_info.get("sha256") or contract.metadata.get("after_hash") or ""
    )
    matched = bool(old_hash and new_hash and old_hash != new_hash)
    return _result(
        contract,
        matched,
        receipt[:500],
        "export hash did not change",
    )


def _verify_clipboard_contains(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, after
    clipboard = str(
        receipt_info.get("clipboard") or receipt_info.get("text") or receipt
    )
    matched = contract.value.lower() in clipboard.lower() if contract.value else False
    return _result(
        contract,
        matched,
        clipboard[:500],
        "clipboard did not contain value",
    )


def _verify_window_title_changed(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del receipt, receipt_info
    matched = before.app_context != after.app_context or bool(after.task_progress)
    observed = f"before={before.app_context}; after={after.app_context}"
    return _result(
        contract,
        matched,
        observed,
        "window context did not change",
    )


def _verify_tab_focused(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, receipt, receipt_info
    labels = " ".join(item.semantic_label for item in after.elements).lower()
    matched = (
        contract.value.lower() in labels
        if contract.value
        else after.focus_region == "header"
    )
    return _result(
        contract,
        matched,
        labels[:500],
        "target tab was not focused",
    )


def _verify_process_launched(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    del before, after
    status = str(receipt_info.get("status") or receipt).lower()
    matched = any(token in status for token in ("launched", "started", "executed"))
    return _result(
        contract,
        matched,
        receipt[:500],
        "process launch was not observed",
    )


def _verify_state_changed(
    contract: VerificationContract,
    before: AbstractUIState,
    after: AbstractUIState,
    receipt: str,
    receipt_info: dict[str, Any],
) -> VerificationResult:
    diff = diff_ui_states(before, after)
    matched = bool(diff) or _receipt_success(receipt, receipt_info)
    if not matched and _focus_receipt_found_target(contract, receipt_info):
        matched = True
    observed = json.dumps(diff, sort_keys=True) if diff else receipt[:500]
    return _result(contract, matched, observed, "no state change was detected")


def _focus_receipt_found_target(
    contract: VerificationContract,
    receipt_info: dict[str, Any],
) -> bool:
    action_type = str(contract.metadata.get("action_type") or "").lower()
    if action_type != "focus":
        return False
    status = str(receipt_info.get("status") or "").lower()
    if status not in {"focused", "matched"}:
        return False
    if status == "focused":
        return True
    if not contract.metadata.get("regrounded"):
        return False
    return any(
        str(receipt_info.get(key) or "").strip()
        for key in ("selector", "matched_name", "matched_role")
    )


def _receipt_reports_text_entry_success(
    contract: VerificationContract,
    receipt_info: dict[str, Any],
) -> bool:
    action_type = str(contract.metadata.get("action_type") or "").lower()
    if action_type not in {"set_text", "type"}:
        return False
    status = str(receipt_info.get("status") or "").lower()
    if status not in {"typed", "value-set"}:
        return False
    matched_role = str(receipt_info.get("matched_role") or "").lower()
    if matched_role:
        return any(token in matched_role for token in ("edit", "document"))
    return bool(str(receipt_info.get("selector") or "").strip())


def _result(
    contract: VerificationContract,
    matched: bool,
    observed: str,
    reason: str,
) -> VerificationResult:
    return VerificationResult(
        kind=contract.kind,
        matched=matched,
        expected=contract.expected,
        observed=observed,
        required=contract.required,
        reason="" if matched else reason,
        evidence={"target": contract.target, "path": contract.path},
    )


def _parse_receipt(receipt: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(receipt or ""))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _receipt_success(receipt: str, receipt_info: dict[str, Any]) -> bool:
    if receipt_info.get("success") is True:
        return True
    status = str(receipt_info.get("status") or "").lower()
    if status in {
        "ok",
        "success",
        "executed",
        "launched",
        "typed",
        "invoked",
        "hotkey-sent",
    }:
        return True
    return any(token in str(receipt).lower() for token in ("ok", "success", "executed"))


def known_contract_kinds() -> set[str]:
    return set(CONTRACT_KINDS)
