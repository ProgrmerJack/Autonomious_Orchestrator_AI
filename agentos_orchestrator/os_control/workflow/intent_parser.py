from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


_FILE_EXTENSION_RE = re.compile(r"\.[a-z0-9]{2,5}\b", re.I)
_LOCAL_PATH_TOKENS = (
    "downloads",
    "desktop",
    "documents",
    "pictures",
    "videos",
    "music",
    "artifacts/",
    "artifacts\\",
)
_IRREVERSIBLE_VERBS = (
    "delete",
    "remove",
    "purchase",
    "buy",
    "pay",
    "submit",
    "send",
    "post",
    "share",
    "invite",
    "trade",
    "wire",
    "withdraw",
)


@dataclass(slots=True)
class StructuredIntent:
    raw_objective: str
    primary_domain: str = "general"
    object_hint: str = ""
    source_surface: str | None = None
    destination_surface: str | None = None
    search_scope: str = "none"
    copy_semantics: str = "none"
    file_operation: str = ""
    file_source_hint: str = ""
    file_destination_hint: str = ""
    file_pattern: str = ""
    app_target: str | None = None
    source_app_target: str | None = None
    destination_app_target: str | None = None
    app_mentions: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    irreversible_verbs: list[str] = field(default_factory=list)
    expected_verification: list[str] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    path_hints: list[str] = field(default_factory=list)
    safety_predicates: list[str] = field(default_factory=list)
    cross_app: bool = False

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> StructuredIntent:
        if not isinstance(payload, dict):
            return cls(raw_objective="")
        return cls(
            raw_objective=str(payload.get("raw_objective") or ""),
            primary_domain=str(payload.get("primary_domain") or "general"),
            object_hint=str(payload.get("object_hint") or ""),
            source_surface=_optional_text(payload.get("source_surface")),
            destination_surface=_optional_text(payload.get("destination_surface")),
            search_scope=str(payload.get("search_scope") or "none"),
            copy_semantics=str(payload.get("copy_semantics") or "none"),
            file_operation=str(payload.get("file_operation") or ""),
            file_source_hint=str(payload.get("file_source_hint") or ""),
            file_destination_hint=str(payload.get("file_destination_hint") or ""),
            file_pattern=str(payload.get("file_pattern") or ""),
            app_target=_optional_text(payload.get("app_target")),
            source_app_target=_optional_text(payload.get("source_app_target")),
            destination_app_target=_optional_text(payload.get("destination_app_target")),
            app_mentions=[str(item) for item in payload.get("app_mentions") or []],
            operations=[str(item) for item in payload.get("operations") or []],
            irreversible_verbs=[
                str(item) for item in payload.get("irreversible_verbs") or []
            ],
            expected_verification=[
                str(item) for item in payload.get("expected_verification") or []
            ],
            entities={
                str(key): str(value)
                for key, value in dict(payload.get("entities") or {}).items()
            },
            path_hints=[str(item) for item in payload.get("path_hints") or []],
            safety_predicates=[
                str(item) for item in payload.get("safety_predicates") or []
            ],
            cross_app=bool(payload.get("cross_app")),
        )

    def prefers_local_file_search(self) -> bool:
        return self.search_scope == "local_files"

    def prefers_web_search(self) -> bool:
        return self.search_scope == "web"

    def prefers_chat_search(self) -> bool:
        return self.search_scope == "app_messages"

    def is_file_copy(self) -> bool:
        return self.copy_semantics == "file"

    def is_clipboard_copy(self) -> bool:
        return self.copy_semantics in {"text", "link", "clipboard"}

    def is_file_workflow(self) -> bool:
        return self.prefers_local_file_search() or bool(self.file_operation)


def parse_structured_intent(objective: str) -> StructuredIntent:
    cleaned = re.sub(r"\s+", " ", str(objective or "")).strip()
    lower = cleaned.lower()
    mentioned_surfaces = _mentioned_surfaces(lower)
    path_hints = [token for token in _LOCAL_PATH_TOKENS if token in lower]
    file_pattern = _file_pattern(cleaned)
    file_operation = _file_operation(lower)
    copy_semantics = _copy_semantics(lower, file_operation, file_pattern)
    search_scope = _search_scope(lower, path_hints, file_pattern, file_operation)
    source_surface, destination_surface = _source_and_destination_surfaces(
        lower,
        mentioned_surfaces,
        search_scope,
        copy_semantics,
    )
    source_app_target = _surface_target(source_surface, lower)
    destination_app_target = _surface_target(destination_surface, lower)
    app_target = _single_app_target(source_app_target, destination_app_target)
    operations = _operations(
        lower,
        search_scope=search_scope,
        file_operation=file_operation,
        copy_semantics=copy_semantics,
        source_surface=source_surface,
        destination_surface=destination_surface,
    )
    object_hint = _object_hint(cleaned, lower, search_scope)
    file_source_hint, file_destination_hint = _file_hints(cleaned, file_operation)
    entities = _entities(
        cleaned,
        lower,
        object_hint=object_hint,
        file_source_hint=file_source_hint,
        file_destination_hint=file_destination_hint,
        source_surface=source_surface,
        destination_surface=destination_surface,
        copy_semantics=copy_semantics,
    )
    expected_verification = _expected_verification(
        search_scope=search_scope,
        file_operation=file_operation,
        copy_semantics=copy_semantics,
        destination_surface=destination_surface,
        source_surface=source_surface,
    )
    irreversible_verbs = [
        verb for verb in _IRREVERSIBLE_VERBS if re.search(rf"\b{re.escape(verb)}\b", lower)
    ]
    safety_predicates = _safety_predicates(
        search_scope=search_scope,
        file_operation=file_operation,
        copy_semantics=copy_semantics,
        source_surface=source_surface,
        destination_surface=destination_surface,
        irreversible_verbs=irreversible_verbs,
    )
    primary_domain = _primary_domain(
        source_surface=source_surface,
        destination_surface=destination_surface,
        search_scope=search_scope,
        file_operation=file_operation,
    )
    return StructuredIntent(
        raw_objective=cleaned,
        primary_domain=primary_domain,
        object_hint=object_hint,
        source_surface=source_surface,
        destination_surface=destination_surface,
        search_scope=search_scope,
        copy_semantics=copy_semantics,
        file_operation=file_operation,
        file_source_hint=file_source_hint,
        file_destination_hint=file_destination_hint,
        file_pattern=file_pattern,
        app_target=app_target,
        source_app_target=source_app_target,
        destination_app_target=destination_app_target,
        app_mentions=mentioned_surfaces,
        operations=operations,
        irreversible_verbs=irreversible_verbs,
        expected_verification=expected_verification,
        entities=entities,
        path_hints=path_hints,
        safety_predicates=safety_predicates,
        cross_app=bool(source_surface and destination_surface and source_surface != destination_surface),
    )


def _mentioned_surfaces(lower: str) -> list[str]:
    mentions: list[str] = []
    if any(token in lower for token in ("chrome", "edge", "browser", "website", "url", "web ")):
        mentions.append("browser")
    if any(token in lower for token in ("file explorer", "explorer", "downloads", "desktop", "documents", "folder")):
        mentions.append("file_explorer")
    if any(token in lower for token in ("notepad", "editor", "document canvas", "text editor")):
        mentions.append("editor")
    if any(
        token in lower
        for token in (
            "outlook",
            "email draft",
            "compose email",
            "in my email",
            "email ",
            " inbox",
            "mail app",
        )
    ):
        mentions.append("email")
    if any(
        token in lower
        for token in (
            "my calendar",
            "calendar app",
            "outlook calendar",
            "appointment",
            "meeting",
            "schedule",
            "calendar event",
        )
    ):
        mentions.append("calendar")
    if any(token in lower for token in ("slack", "teams", "chat", "conversation", "messages")):
        mentions.append("chat_app")
    if any(token in lower for token in ("pdf", "acrobat")):
        mentions.append("pdf_viewer")
    if any(
        token in lower
        for token in (
            "open settings",
            "windows settings",
            "settings app",
            "night light",
            "bluetooth",
            "wi-fi",
            "wifi",
            "dark mode",
            "focus assist",
        )
    ):
        mentions.append("settings")
    ordered: list[str] = []
    for item in mentions:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _search_scope(
    lower: str,
    path_hints: list[str],
    file_pattern: str,
    file_operation: str,
) -> str:
    if any(token in lower for token in ("messages about", "search for messages", "search messages")):
        return "app_messages"
    if any(token in lower for token in ("search", "find", "locate", "latest")):
        if file_operation or file_pattern or path_hints or " pdf" in lower or "downloads" in lower:
            return "local_files"
        if any(token in lower for token in ("chrome", "edge", "browser", "website", "nearest", "address", "store", "url", "web ")):
            return "web"
    if any(token in lower for token in ("research", "look up", "compare", "investigate", "analy", "analyse")):
        return "web"
    return "none"


def _copy_semantics(lower: str, file_operation: str, file_pattern: str) -> str:
    if "copy" not in lower and "paste" not in lower:
        return "none"
    if file_operation == "copy":
        return "file"
    if any(token in lower for token in ("address", "summary", "text", "message", "reply", "note", "paragraph")):
        return "text"
    if any(token in lower for token in ("link", "url")):
        return "link"
    if file_pattern and re.search(r"\bcopy\s+.+\s+to\s+.+", lower):
        return "file"
    if "into notepad" in lower or "into the editor" in lower or "paste" in lower:
        return "clipboard"
    return "clipboard"


def _file_operation(lower: str) -> str:
    if "rename" in lower:
        return "rename"
    if "move" in lower:
        return "move"
    if "copy" in lower and (
        any(token in lower for token in ("file", "folder", "downloads", "desktop"))
        or _FILE_EXTENSION_RE.search(lower) is not None
        or re.search(r"\bcopy\s+.+\s+to\s+.+", lower) is not None
    ):
        return "copy"
    return ""


def _source_and_destination_surfaces(
    lower: str,
    mentioned_surfaces: list[str],
    search_scope: str,
    copy_semantics: str,
) -> tuple[str | None, str | None]:
    source_surface = mentioned_surfaces[0] if mentioned_surfaces else None
    destination_surface = None
    if search_scope == "local_files":
        source_surface = "file_explorer"
    elif search_scope == "web" and source_surface is None:
        source_surface = "browser"
    elif search_scope == "app_messages":
        source_surface = "chat_app"
    if any(
        token in lower
        for token in (
            "in my email",
            "from my email",
            "search my email",
            "search email",
            "email inbox",
        )
    ):
        source_surface = "email"
    elif source_surface is None and any(
        token in lower
        for token in (
            "open settings",
            "windows settings",
            "settings app",
        )
    ):
        source_surface = "settings"
    if any(token in lower for token in ("into notepad", "into the editor", "in notepad", "paste into notepad")):
        destination_surface = "editor"
    elif any(
        token in lower
        for token in (
            "to an email draft",
            "to my email",
            "into email",
            "email draft",
            "draft email",
            "compose email",
        )
    ):
        destination_surface = "email"
    elif any(
        token in lower
        for token in (
            "on my calendar",
            "to my calendar",
            "into my calendar",
            "add to calendar",
            "calendar event",
            "create event",
            "schedule it",
        )
    ):
        destination_surface = "calendar"
    elif any(token in lower for token in ("reply with", "draft reply", "reply ")):
        destination_surface = "chat_app"
    elif any(token in lower for token in ("into word", "into document")):
        destination_surface = "editor"
    if destination_surface is None and "email" in mentioned_surfaces and any(
        token in lower for token in ("attach", "draft", "compose", "send email")
    ):
        destination_surface = "email"
    if destination_surface is None and copy_semantics in {"text", "link", "clipboard"} and "editor" in mentioned_surfaces:
        destination_surface = "editor"
    return source_surface, destination_surface


def _surface_target(surface: str | None, lower: str) -> str | None:
    if surface == "browser":
        if "chrome" in lower:
            return "chrome.exe"
        return "msedge.exe"
    if surface == "file_explorer":
        return "explorer.exe"
    if surface == "editor":
        if "vscode" in lower or "visual studio code" in lower:
            return "code"
        return "notepad.exe"
    if surface == "email":
        return "outlook.exe"
    if surface == "calendar":
        return "outlook.exe /select outlook:calendar"
    if surface == "chat_app":
        if "slack" in lower:
            return "slack.exe"
        return "teams.exe"
    if surface == "pdf_viewer":
        return "acrobat.exe"
    if surface == "settings":
        return "settings.exe"
    return None


def _single_app_target(
    source_app_target: str | None,
    destination_app_target: str | None,
) -> str | None:
    if source_app_target and destination_app_target and source_app_target != destination_app_target:
        return None
    return source_app_target or destination_app_target


def _operations(
    lower: str,
    *,
    search_scope: str,
    file_operation: str,
    copy_semantics: str,
    source_surface: str | None,
    destination_surface: str | None,
) -> list[str]:
    ops: list[str] = []
    if search_scope == "local_files":
        ops.append("search_local_files")
    elif search_scope == "web":
        ops.append("search_web")
    elif search_scope == "app_messages":
        ops.append("search_messages")
    if file_operation:
        ops.append(file_operation)
    if source_surface == "email" and any(
        token in lower for token in ("search", "find", "locate", "look up")
    ):
        ops.append("search_email")
    if copy_semantics in {"text", "link", "clipboard"}:
        ops.append("copy_to_clipboard")
    if destination_surface == "editor":
        ops.append("write_editor")
    if destination_surface == "email" and any(
        token in lower for token in ("draft", "compose", "email", "attach")
    ):
        ops.append("draft_email")
    if destination_surface == "email" and "attach" in lower:
        ops.append("attach_file")
    if destination_surface == "calendar" and any(
        token in lower for token in ("calendar", "event", "meeting", "invite", "schedule")
    ):
        ops.append("create_calendar_event")
    if source_surface == "settings":
        ops.append("search_settings")
        if any(token in lower for token in ("turn on", "turn off", "enable", "disable")):
            ops.append("toggle_setting")
    if source_surface == "chat_app" and destination_surface == "chat_app":
        ops.append("draft_reply")
    return ops


def _object_hint(cleaned: str, lower: str, search_scope: str) -> str:
    if search_scope == "local_files":
        match = re.search(
            r"(?:find|locate|rename)\s+(?:the\s+)?(?P<object>.+?)(?:\s+in\s+downloads|\s+to\s+|$)",
            lower,
        )
        if match is not None:
            return match.group("object").strip(" .")
    if search_scope == "app_messages":
        match = re.search(r"messages about (?P<object>.+?)(?:,| and |$)", lower)
        if match is not None:
            return match.group("object").strip(" .")
    patterns = (
        r"search(?: for)? (?P<object>.+?)(?: and | then | into | in | on |$)",
        r"find (?P<object>.+?)(?: and | then | into | in | on |$)",
        r"look up (?P<object>.+?)(?: and | then | into | in | on |$)",
    )
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match is not None:
            return match.group("object").strip(" .")
    return cleaned[:160]


def _file_hints(cleaned: str, file_operation: str) -> tuple[str, str]:
    if not file_operation:
        return "", ""
    if file_operation in {"copy", "move"}:
        match = re.search(
            rf"{file_operation}\s+(?P<src>.+?)\s+to\s+(?P<dst>.+)$",
            cleaned,
            re.I,
        )
        if match is not None:
            return match.group("src").strip(), match.group("dst").strip()
    match = re.search(r"rename\s+(?P<src>.+?)\s+to\s+(?P<dst>.+)$", cleaned, re.I)
    if match is not None:
        return match.group("src").strip(), match.group("dst").strip()
    return "", ""


def _file_pattern(cleaned: str) -> str:
    match = _FILE_EXTENSION_RE.search(cleaned)
    if match is None:
        return ""
    return match.group(0)


def _entities(
    cleaned: str,
    lower: str,
    *,
    object_hint: str,
    file_source_hint: str,
    file_destination_hint: str,
    source_surface: str | None,
    destination_surface: str | None,
    copy_semantics: str,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    if object_hint:
        payload["query"] = object_hint
    if file_source_hint:
        payload["file_source"] = file_source_hint
    if file_destination_hint:
        payload["file_destination"] = file_destination_hint
    if copy_semantics in {"text", "link", "clipboard"}:
        payload["clipboard_text"] = _clipboard_payload(cleaned, lower, object_hint)
    if destination_surface == "editor":
        payload["editor_text"] = payload.get("clipboard_text") or object_hint
    if source_surface == "email" and payload.get("query"):
        payload["email_query"] = payload["query"]
    if destination_surface == "email":
        recipient = _recipient(cleaned, lower)
        if recipient:
            payload["recipient"] = recipient
        attachment_source = payload.get("file_source") or _attachment_source(cleaned, lower)
        if attachment_source:
            payload["file_source"] = attachment_source
    if destination_surface == "calendar":
        event_title = _calendar_event_title(cleaned, lower, object_hint)
        if event_title:
            payload["event_title"] = event_title
    if source_surface == "settings":
        setting_name = _setting_name(lower, object_hint)
        if setting_name:
            payload["setting_name"] = setting_name
        setting_value = _setting_value(lower)
        if setting_value:
            payload["setting_value"] = setting_value
    return payload


def _attachment_source(cleaned: str, lower: str) -> str:
    if "attach" not in lower:
        return ""
    match = re.search(
        r"attach\s+(?P<src>.+?)\s+to\s+(?:an\s+)?email",
        cleaned,
        re.I,
    )
    if match is not None:
        return match.group("src").strip(" .")
    return ""


def _recipient(cleaned: str, lower: str) -> str:
    if "email" not in lower:
        return ""
    match = re.search(
        r"email\s+(?:draft\s+)?for\s+(?P<recipient>.+?)(?:$|\s+about\s+|\s+with\s+)",
        cleaned,
        re.I,
    )
    if match is not None:
        return match.group("recipient").strip(" .")
    return ""


def _calendar_event_title(cleaned: str, lower: str, object_hint: str) -> str:
    if "calendar" not in lower and "event" not in lower:
        return ""
    if object_hint:
        return object_hint.strip()
    return cleaned.strip().rstrip(".")


def _setting_name(lower: str, object_hint: str) -> str:
    known = {
        "night light": "Night Light",
        "bluetooth": "Bluetooth",
        "wi-fi": "Wi-Fi",
        "wifi": "Wi-Fi",
        "dark mode": "Dark Mode",
        "focus assist": "Focus Assist",
    }
    for token, label in known.items():
        if token in lower:
            return label
    if object_hint:
        return object_hint.strip()
    return ""


def _setting_value(lower: str) -> str:
    if any(token in lower for token in ("turn on", "enable")):
        return "on"
    if any(token in lower for token in ("turn off", "disable")):
        return "off"
    return ""


def _clipboard_payload(cleaned: str, lower: str, object_hint: str) -> str:
    if "address" in lower:
        return f"Address result for: {object_hint}".strip()
    if "summary" in lower:
        return f"Summary for: {object_hint}".strip()
    if "link" in lower or "url" in lower:
        return f"Link result for: {object_hint}".strip()
    return cleaned.strip().rstrip(".")


def _expected_verification(
    *,
    search_scope: str,
    file_operation: str,
    copy_semantics: str,
    destination_surface: str | None,
    source_surface: str | None,
) -> list[str]:
    expected: list[str] = []
    if search_scope == "local_files":
        expected.append("file_exists")
    if file_operation == "rename":
        expected.extend(["file_exists", "state_changed"])
    elif file_operation in {"copy", "move"}:
        expected.extend(["file_exists", "receipt_success"])
    if copy_semantics in {"text", "link", "clipboard"}:
        expected.append("clipboard_contains")
    if destination_surface == "editor":
        expected.append("field_contains")
    if source_surface == "email":
        expected.append("state_changed")
    if destination_surface == "email":
        expected.extend(["field_contains", "state_changed"])
    if destination_surface == "calendar":
        expected.extend(["field_contains", "state_changed"])
    if source_surface == "settings":
        expected.extend(["field_contains", "state_changed"])
    if source_surface == "chat_app":
        expected.append("state_changed")
    return _ordered(expected)


def _safety_predicates(
    *,
    search_scope: str,
    file_operation: str,
    copy_semantics: str,
    source_surface: str | None,
    destination_surface: str | None,
    irreversible_verbs: list[str],
) -> list[str]:
    predicates: list[str] = []
    if search_scope == "local_files":
        predicates.append("local_file_route")
    if search_scope == "web":
        predicates.append("external_navigation_route")
    if file_operation:
        predicates.append("file_operation_matches_intent")
    if copy_semantics in {"text", "link", "clipboard"}:
        predicates.append("clipboard_entity_matches_intent")
    if source_surface:
        predicates.append(f"source_surface:{source_surface}")
    if destination_surface:
        predicates.append(f"destination_surface:{destination_surface}")
    if irreversible_verbs:
        predicates.append("approval_required_for_irreversible_action")
    return predicates


def _primary_domain(
    *,
    source_surface: str | None,
    destination_surface: str | None,
    search_scope: str,
    file_operation: str,
) -> str:
    if source_surface and destination_surface and source_surface != destination_surface:
        return "cross_app"
    if file_operation or search_scope == "local_files":
        return "file_ops"
    if source_surface == "settings":
        return "settings"
    if source_surface == "email" or destination_surface == "email":
        return "email"
    if source_surface == "calendar" or destination_surface == "calendar":
        return "calendar"
    if search_scope == "app_messages" or source_surface == "chat_app":
        return "chat"
    if search_scope == "web" or source_surface == "browser":
        return "browser"
    if source_surface == "pdf_viewer":
        return "pdf"
    if destination_surface == "editor" or source_surface == "editor":
        return "editor"
    return "general"


def _ordered(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        compact = str(value or "").strip()
        if compact and compact not in ordered:
            ordered.append(compact)
    return ordered


def _optional_text(value: Any) -> str | None:
    compact = str(value or "").strip()
    return compact or None