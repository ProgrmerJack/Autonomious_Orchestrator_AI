from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import UiAction, UiNode


_CANONICAL_NAMES = {
    "app-workspace": "Application Workspace",
    "browser-address-bar": "Browser Address Bar",
    "document-canvas": "Document Canvas",
    "editor-canvas": "Editor Canvas",
    "email-search-box": "Email Search Box",
    "email-to-field": "Email Recipient Field",
    "email-attachment-field": "Email Attachment Field",
    "email-send-button": "Send Email",
    "email-status-text": "Email Status",
    "calendar-grid": "Calendar Grid",
    "calendar-event-editor": "Calendar Event Editor",
    "calendar-invite-button": "Create Invite",
    "calendar-details": "Calendar Details",
    "settings-search-box": "Settings Search Box",
    "settings-toggle": "Settings Toggle",
    "settings-status-text": "Settings Status",
}


def normalize_windows_nodes(nodes: list[UiNode]) -> list[UiNode]:
    return [_normalize_node(node) for node in nodes]


def selector_matches_node(selector: str, node: UiNode) -> bool:
    normalized = _normalize(selector)
    if not normalized:
        return False
    aliases = [_normalize(item) for item in _selector_aliases(node)]
    metadata = dict(node.metadata or {})
    values = {
        "name": node.name,
        "role": node.role,
        "automation_id": str(metadata.get("automation_id") or ""),
        "class_name": str(metadata.get("class_name") or ""),
        "node_id": node.node_id,
        "semantic_name": str(metadata.get("semantic_name") or ""),
        "semantic_selector": str(metadata.get("semantic_selector") or ""),
    }
    if "=" in normalized:
        field, expected = normalized.split("=", 1)
        if field == "alias":
            return any(expected in alias for alias in aliases)
        if field in values:
            return expected in _normalize(values[field])
    if normalized in aliases:
        return True
    return any(normalized in _normalize(value) for value in values.values())


def adapt_windows_action(
    action: UiAction,
    nodes: list[UiNode],
    backend_name: str,
) -> tuple[UiAction, UiNode | None]:
    matched = _match_node(action, nodes)
    if matched is None:
        return action, None
    metadata = dict(action.metadata or {})
    aliases = _selector_aliases(matched)
    if aliases:
        metadata.setdefault("selector_aliases", list(aliases))
        metadata.setdefault("semantic_selector", aliases[0])
        metadata.setdefault(
            "semantic_name",
            _CANONICAL_NAMES.get(aliases[0], aliases[0]),
        )
    metadata.setdefault("matched_name", matched.name)
    metadata.setdefault("matched_role", matched.role)
    metadata.setdefault("matched_node_id", matched.node_id)
    if matched.bounds is not None:
        left, top, width, height = matched.bounds
        metadata.setdefault("bounds", [left, top, width, height])
        metadata.setdefault("x", left + (width // 2))
        metadata.setdefault("y", top + (height // 2))
    selector = action.selector
    if _uses_semantic_selector(action.selector) or (
        backend_name == "rust-native-windows"
        and _uses_uia_only_selector(action.selector)
    ):
        selector = _preferred_selector(matched, backend_name)
        metadata["resolved_from"] = action.selector
        metadata["resolved_selector"] = selector
    return (
        UiAction(
            action_type=action.action_type,
            selector=selector,
            value=action.value,
            metadata=metadata,
        ),
        matched,
    )


def enrich_windows_receipt(
    action: UiAction,
    receipt: dict[str, Any],
    before_nodes: list[UiNode],
    after_nodes: list[UiNode],
    matched_node: UiNode | None = None,
) -> dict[str, Any]:
    payload = dict(receipt)
    metadata = dict(action.metadata or {})
    if matched_node is not None:
        payload.setdefault("matched_name", matched_node.name)
        payload.setdefault("matched_role", matched_node.role)
        payload.setdefault("matched_node_id", matched_node.node_id)
    payload.setdefault(
        "semantic_selector",
        str(metadata.get("semantic_selector") or ""),
    )
    family = _family_hint(action)
    before = normalize_windows_nodes(before_nodes)
    after = normalize_windows_nodes(after_nodes)
    if family == "email":
        email = _email_outcome(action, before, after)
        if email is not None:
            payload["email"] = email
    if family == "calendar":
        calendar = _calendar_outcome(action, before, after)
        if calendar is not None:
            payload["calendar"] = calendar
    if family == "settings":
        setting = _settings_outcome(action, after)
        if setting is not None:
            payload["setting"] = setting
    return payload


def _normalize_node(node: UiNode) -> UiNode:
    metadata = dict(node.metadata or {})
    aliases = _selector_aliases(node)
    if aliases:
        metadata["selector_aliases"] = list(aliases)
        metadata.setdefault("semantic_selector", aliases[0])
        metadata.setdefault(
            "semantic_name",
            _CANONICAL_NAMES.get(aliases[0], aliases[0]),
        )
    return UiNode(
        node_id=node.node_id,
        role=node.role,
        name=node.name,
        bounds=node.bounds,
        enabled=node.enabled,
        focused=node.focused,
        metadata=metadata,
    )


def _match_node(action: UiAction, nodes: list[UiNode]) -> UiNode | None:
    selector = str(action.selector or "").strip()
    family = _family_hint(action)
    normalized_nodes = normalize_windows_nodes(nodes)
    if selector:
        if _uses_semantic_selector(selector):
            matched = _best_semantic_match(normalized_nodes, selector, family)
            if matched is not None:
                return matched
        for node in normalized_nodes:
            if selector_matches_node(selector, node):
                return node
    implicit = _implicit_semantic_target(action, family)
    if implicit:
        return _best_semantic_match(normalized_nodes, implicit, family)
    return None


def _best_semantic_match(
    nodes: list[UiNode],
    selector: str,
    family: str,
) -> UiNode | None:
    best_score = 0.0
    best_node: UiNode | None = None
    for node in nodes:
        score = _semantic_score(node, selector, family)
        if score > best_score:
            best_score = score
            best_node = node
    return best_node


def _semantic_score(node: UiNode, selector: str, family: str) -> float:
    normalized_selector = _normalize(selector)
    aliases = [_normalize(item) for item in _selector_aliases(node, family)]
    if normalized_selector in aliases:
        score = 100.0
    else:
        score = 0.0
    haystack = _node_haystack(node)
    role = _normalize(node.role)
    if normalized_selector == "email-search-box":
        if "edit" in role:
            score += 24.0
        if any(
            token in haystack
            for token in (
                "search mailbox",
                "search current mailbox",
                "mailbox",
                "search",
            )
        ):
            score += 30.0
    elif normalized_selector == "email-to-field":
        if "edit" in role:
            score += 24.0
        if any(
            token in haystack for token in ("to", "recipient", "recipients")
        ):
            score += 32.0
    elif normalized_selector == "email-attachment-field":
        if any(
            token in haystack for token in ("attach", "attachment", "file")
        ):
            score += 34.0
        if any(token in role for token in ("edit", "button")):
            score += 14.0
    elif normalized_selector == "email-send-button":
        if "button" in role:
            score += 22.0
        if "send" in haystack:
            score += 34.0
    elif normalized_selector == "calendar-grid":
        if any(token in role for token in ("table", "grid")):
            score += 28.0
        if any(
            token in haystack
            for token in ("calendar", "schedule", "appointment", "meeting")
        ):
            score += 26.0
    elif normalized_selector == "calendar-event-editor":
        if any(token in role for token in ("edit", "document")):
            score += 20.0
        if any(
            token in haystack
            for token in (
                "event",
                "appointment",
                "meeting",
                "subject",
                "title",
            )
        ):
            score += 32.0
    elif normalized_selector == "calendar-invite-button":
        if "button" in role:
            score += 20.0
        if any(
            token in haystack
            for token in ("invite", "send update", "send", "meeting")
        ):
            score += 34.0
    elif normalized_selector == "browser-address-bar":
        if "edit" in role:
            score += 24.0
        if any(
            token in haystack
            for token in (
                "address and search bar",
                "address bar",
                "search or enter web address",
                "search the web",
                "show page address",
            )
        ):
            score += 36.0
    elif normalized_selector in {"document-canvas", "editor-canvas", "app-workspace"}:
        if any(token in role for token in ("document", "edit", "pane")):
            score += 24.0
        if any(
            token in haystack
            for token in (
                "text editor",
                "notepad",
                "wordpad",
                "plain text",
                "document",
                "editor",
            )
        ):
            score += 34.0
    elif normalized_selector == "settings-search-box":
        if "edit" in role:
            score += 24.0
        if any(
            token in haystack
            for token in (
                "find a setting",
                "search settings",
                "settings search",
                "search",
            )
        ):
            score += 34.0
    elif normalized_selector == "settings-toggle":
        if any(
            token in role for token in ("button", "check", "switch", "toggle")
        ):
            score += 24.0
        if any(
            token in haystack
            for token in (
                "night light",
                "nightlight",
                "bluetooth",
                "wi-fi",
                "wifi",
            )
        ):
            score += 32.0
    if node.enabled:
        score += 2.0
    if node.focused:
        score += 1.0
    return score


def _selector_aliases(node: UiNode, family_hint: str = "") -> list[str]:
    metadata = dict(node.metadata or {})
    existing = [
        str(item) for item in list(metadata.get("selector_aliases") or [])
    ]
    aliases = [item for item in existing if item]
    haystack = _node_haystack(node)
    role = _normalize(node.role)
    family = family_hint or _infer_family_from_haystack(haystack)
    if family == "email":
        if "edit" in role and any(
            token in haystack
            for token in ("search mailbox", "search current mailbox", "mailbox")
        ):
            aliases.append("email-search-box")
        if "edit" in role and any(
            token in haystack for token in ("to", "recipient", "recipients")
        ):
            aliases.append("email-to-field")
        if any(token in haystack for token in ("attach", "attachment")):
            aliases.append("email-attachment-field")
        if "button" in role and "send" in haystack:
            aliases.append("email-send-button")
        if any(
            token in haystack
            for token in ("message sent", "mail sent", "sent items")
        ):
            aliases.append("email-status-text")
    if family == "calendar":
        if any(token in role for token in ("table", "grid")) and any(
            token in haystack
            for token in ("calendar", "schedule", "appointment", "meeting")
        ):
            aliases.append("calendar-grid")
        if any(token in role for token in ("edit", "document")) and any(
            token in haystack
            for token in (
                "event",
                "appointment",
                "meeting",
                "subject",
                "title",
            )
        ):
            aliases.append("calendar-event-editor")
        if "button" in role and any(
            token in haystack for token in ("invite", "send update", "meeting")
        ):
            aliases.append("calendar-invite-button")
        if any(
            token in haystack
            for token in (
                "invite sent",
                "meeting sent",
                "event details",
                "appointment",
            )
        ):
            aliases.append("calendar-details")
    if family == "settings":
        if "edit" in role and any(
            token in haystack
            for token in (
                "find a setting",
                "search settings",
                "settings search",
            )
        ):
            aliases.append("settings-search-box")
        if any(
            token in role for token in ("button", "check", "switch", "toggle")
        ) and any(
            token in haystack
            for token in (
                "night light",
                "nightlight",
                "bluetooth",
                "wi-fi",
                "wifi",
            )
        ):
            aliases.append("settings-toggle")
        if any(
            token in haystack
            for token in (
                "night light on",
                "night light off",
                "nightlight on",
                "nightlight off",
            )
        ):
            aliases.append("settings-status-text")
    if family == "editor":
        if any(token in role for token in ("document", "edit")) and any(
            token in haystack
            for token in (
                "text editor",
                "notepad",
                "wordpad",
                "plain text",
                "document",
                "editor",
            )
        ):
            aliases.extend(
                ["document-canvas", "editor-canvas", "app-workspace"]
            )
    if family == "browser":
        if "edit" in role and any(
            token in haystack
            for token in (
                "address and search bar",
                "address bar",
                "search or enter web address",
                "search the web",
                "show page address",
            )
        ):
            aliases.append("browser-address-bar")
    if family == "email" and "search" in haystack and "edit" in role:
        aliases.append("email-search-box")
    if family == "settings" and "search" in haystack and "edit" in role:
        aliases.append("settings-search-box")
    deduped: list[str] = []
    for alias in aliases:
        if alias not in deduped:
            deduped.append(alias)
    return deduped


def _family_hint(action: UiAction) -> str:
    metadata = dict(action.metadata or {})
    for candidate in (
        metadata.get("adapter_family"),
        metadata.get("semantic_selector"),
        action.selector,
    ):
        text = _normalize(str(candidate or ""))
        if text.startswith("email") or text.startswith("outlook"):
            return "email"
        if text.startswith("browser"):
            return "browser"
        if text.startswith("calendar"):
            return "calendar"
        if text.startswith(("document", "editor", "app-workspace")):
            return "editor"
        if text.startswith("settings"):
            return "settings"
    return ""


def _implicit_semantic_target(action: UiAction, family: str) -> str:
    selector = _normalize(action.selector)
    if family == "email" and action.action_type in {"click", "invoke"}:
        if "send" in selector:
            return "email-send-button"
    if family == "calendar" and action.action_type in {"click", "invoke"}:
        if any(token in selector for token in ("invite", "send", "meeting")):
            return "calendar-invite-button"
    if family == "settings" and action.action_type in {"click", "invoke"}:
        if any(
            token in selector for token in ("toggle", "night light", "nightlight")
        ):
            return "settings-toggle"
    return ""


def _preferred_selector(node: UiNode, backend_name: str) -> str:
    metadata = dict(node.metadata or {})
    if backend_name != "rust-native-windows":
        automation_id = str(metadata.get("automation_id") or "").strip()
        if automation_id:
            return f"automation_id={automation_id}"
    if node.name:
        return f"name={node.name}"
    if backend_name != "rust-native-windows":
        class_name = str(metadata.get("class_name") or "").strip()
        if class_name:
            return f"class_name={class_name}"
    return node.node_id


def _uses_semantic_selector(selector: str) -> bool:
    cleaned = str(selector or "").strip().lower()
    if not cleaned:
        return False
    if "=" in cleaned or "," in cleaned:
        return False
    return cleaned.startswith(
        ("browser-", "email-", "calendar-", "settings-")
    )


def _uses_uia_only_selector(selector: str) -> bool:
    cleaned = _normalize(selector)
    return cleaned.startswith(("automation_id=", "class_name="))


def _email_outcome(
    action: UiAction,
    before_nodes: list[UiNode],
    after_nodes: list[UiNode],
) -> dict[str, Any] | None:
    metadata = dict(action.metadata or {})
    before_aliases = _alias_set(before_nodes)
    after_aliases = _alias_set(after_nodes)
    labels = _labels(after_nodes)
    compose_cleared = (
        "email-send-button" in before_aliases
        and "email-send-button" not in after_aliases
    )
    saw_status = any(
        token in labels
        for token in ("message sent", "mail sent", "sent items", "delivered")
    )
    if not compose_cleared and not saw_status:
        return None
    recipient = str(metadata.get("recipient") or "").strip()
    attachment = _attachment_name(metadata.get("attachment") or "")
    return {
        "status": "sent",
        "recipient": recipient,
        "attachment": attachment,
        "proof": {
            "compose_cleared": compose_cleared,
            "sent_keyword": saw_status,
        },
    }


def _calendar_outcome(
    action: UiAction,
    before_nodes: list[UiNode],
    after_nodes: list[UiNode],
) -> dict[str, Any] | None:
    metadata = dict(action.metadata or {})
    before_aliases = _alias_set(before_nodes)
    after_aliases = _alias_set(after_nodes)
    labels = _labels(after_nodes)
    editor_cleared = (
        "calendar-invite-button" in before_aliases
        and "calendar-invite-button" not in after_aliases
    )
    saw_status = any(
        token in labels
        for token in (
            "invite sent",
            "meeting sent",
            "meeting updated",
            "invitation sent",
        )
    )
    if not editor_cleared and not saw_status:
        return None
    return {
        "status": "invited",
        "event_title": str(metadata.get("event_title") or "").strip(),
        "proof": {
            "editor_cleared": editor_cleared,
            "invite_keyword": saw_status,
        },
    }


def _settings_outcome(
    action: UiAction,
    after_nodes: list[UiNode],
) -> dict[str, Any] | None:
    metadata = dict(action.metadata or {})
    setting_name, state = _setting_target(action)
    labels = _labels(after_nodes)
    if not setting_name:
        setting_name = str(metadata.get("setting_name") or "").strip()
    if not state:
        state = str(metadata.get("setting_state") or "").strip().lower()
    if not setting_name or not state:
        return None
    state_tokens = {state}
    if state == "on":
        state_tokens.update({"enabled", "active"})
    if state == "off":
        state_tokens.update({"disabled", "inactive"})
    observed = setting_name.lower() in labels and any(
        token in labels for token in state_tokens
    )
    if not observed:
        return None
    return {"name": setting_name, "state": state}


def _setting_target(action: UiAction) -> tuple[str, str]:
    metadata = dict(action.metadata or {})
    name = str(metadata.get("setting_name") or "").strip()
    state = str(metadata.get("setting_state") or "").strip().lower()
    if name and state:
        return name, state
    raw = str(action.value or "")
    left, _, right = raw.partition(":")
    return left.strip(), right.strip().lower()


def _attachment_name(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return Path(text).name or text


def _alias_set(nodes: list[UiNode]) -> set[str]:
    aliases: set[str] = set()
    for node in nodes:
        aliases.update(_selector_aliases(node))
    return aliases


def _labels(nodes: list[UiNode]) -> str:
    chunks: list[str] = []
    for node in nodes:
        metadata = dict(node.metadata or {})
        chunks.append(node.name)
        chunks.append(str(metadata.get("text") or ""))
        chunks.append(str(metadata.get("value") or ""))
        chunks.append(str(metadata.get("semantic_name") or ""))
        chunks.extend(
            str(item) for item in list(metadata.get("selector_aliases") or [])
        )
    return " ".join(chunk for chunk in chunks if chunk).lower()


def _node_haystack(node: UiNode) -> str:
    metadata = dict(node.metadata or {})
    chunks = [
        node.node_id,
        node.name,
        node.role,
        str(metadata.get("automation_id") or ""),
        str(metadata.get("class_name") or ""),
        str(metadata.get("text") or ""),
        str(metadata.get("value") or ""),
        str(metadata.get("semantic_name") or ""),
    ]
    chunks.extend(
        str(item) for item in list(metadata.get("selector_aliases") or [])
    )
    return _normalize(" ".join(chunk for chunk in chunks if chunk))


def _infer_family_from_haystack(haystack: str) -> str:
    if any(
        token in haystack
        for token in ("mailbox", "outlook", "inbox", "compose", "message sent")
    ):
        return "email"
    if any(
        token in haystack
        for token in ("calendar", "appointment", "meeting", "invite sent")
    ):
        return "calendar"
    if any(
        token in haystack
        for token in ("find a setting", "settings", "night light", "nightlight")
    ):
        return "settings"
    if any(
        token in haystack
        for token in (
            "address and search bar",
            "search or enter web address",
            "browser",
            "tab groups",
        )
    ):
        return "browser"
    if any(
        token in haystack
        for token in ("notepad", "wordpad", "text editor", "plain text")
    ):
        return "editor"
    return ""


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())
