from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AppFamilySpec:
    family: str
    app_context: str
    aliases: tuple[str, ...]
    profile_cues: tuple[str, ...]
    profile_confidence: float
    launch_target: str
    launch_cues: tuple[str, ...]
    primary_selector: str
    surface_role: str
    surface_name: str
    preferred_channels: tuple[str, ...]
    affordance_hints: tuple[str, ...]
    verification_contracts: tuple[str, ...]
    repair_recipes: tuple[str, ...]
    eval_surface: bool = True
    live_fire: bool = False
    safe_windows: bool = False
    dom_like: bool = False
    api_like: bool = False
    visual_heavy: bool = False
    recommended_mode: str = "ui"
    clipboard_base: float = 0.55
    latency_base: float = 0.45
    action_policy: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, str] = field(default_factory=dict)


APP_FAMILY_SPECS: tuple[AppFamilySpec, ...] = (
    AppFamilySpec(
        family="browser",
        app_context="browser",
        aliases=("browser", "web"),
        profile_cues=("address", "tab", "url", "back", "forward"),
        profile_confidence=0.86,
        launch_target="browser",
        launch_cues=("browser", "edge", "chrome"),
        primary_selector="browser-address-bar",
        surface_role="Edit",
        surface_name="Address and search bar",
        preferred_channels=("accessibility", "dom", "ocr", "explore"),
        affordance_hints=("tabs", "address bar", "links", "forms", "page actions"),
        verification_contracts=("tab_focused", "field_contains", "state_changed"),
        repair_recipes=(
            "refresh DOM/accessibility",
            "escape modal",
            "bounded link probe",
        ),
        safe_windows=True,
        dom_like=True,
        recommended_mode="hybrid",
        clipboard_base=0.7,
        latency_base=0.7,
        action_policy={
            "require_approval_terms": (
                "checkout",
                "purchase",
                "buy now",
                "confirm payment",
                "authorize payment",
                "upload file",
                "download file",
            ),
            "require_approval_selectors": (
                "checkout",
                "purchase",
                "buy now",
                "confirm payment",
                "upload",
                "download",
            ),
            "forbidden_terms": (
                "install extension",
                "add extension",
                "grant persistent permission",
                "allow notifications",
            ),
            "forbidden_selectors": (
                "install extension",
                "add extension",
                "allow notifications",
            ),
        },
        live_fire=True,
    ),
    AppFamilySpec(
        family="file_explorer",
        app_context="file_explorer",
        aliases=("file_explorer",),
        profile_cues=("file explorer", "navigation pane", "folder"),
        profile_confidence=0.85,
        launch_target="explorer.exe",
        launch_cues=("explorer",),
        primary_selector="explorer-file-list",
        surface_role="List",
        surface_name="Explorer File List",
        preferred_channels=("accessibility", "api", "explore"),
        affordance_hints=("tree", "file list", "breadcrumb", "context menu"),
        verification_contracts=("file_exists", "window_title_changed", "state_changed"),
        repair_recipes=(
            "refresh folder",
            "validate path",
            "use structured file action",
        ),
        safe_windows=True,
        live_fire=True,
        api_like=True,
    ),
    AppFamilySpec(
        family="file_dialog",
        app_context="file_dialog",
        aliases=("file_dialog",),
        profile_cues=("save as", "open file", "file name", "filename"),
        profile_confidence=0.94,
        launch_target="notepad.exe",
        launch_cues=(),
        primary_selector="document-canvas",
        surface_role="Document",
        surface_name="Document Canvas",
        preferred_channels=("accessibility", "api", "ocr", "explore"),
        affordance_hints=("filename field", "save/open button", "folder picker"),
        verification_contracts=("field_contains", "file_exists", "modal_closed"),
        repair_recipes=(
            "select filename",
            "validate allowed root",
            "escape unrelated modal",
        ),
        safe_windows=True,
        live_fire=True,
        api_like=True,
        action_policy={
            "require_approval_terms": (
                "overwrite existing file",
                "replace existing file",
                "upload file",
                "attach file",
                "save outside workspace",
                "open outside workspace",
            ),
            "require_approval_selectors": (
                "overwrite",
                "replace",
                "upload",
                "attach",
            ),
            "forbidden_terms": (
                "c:\\windows\\system32",
                "program files",
                "appdata\\roaming",
                "\\.ssh\\",
                "id_rsa",
                "secret",
            ),
            "forbidden_selectors": (
                "system32",
                ".ssh",
                "id_rsa",
            ),
        },
    ),
    AppFamilySpec(
        family="terminal",
        app_context="terminal",
        aliases=("terminal",),
        profile_cues=("terminal", "powershell", "command prompt", "console"),
        profile_confidence=0.86,
        launch_target="powershell.exe",
        launch_cues=("powershell", "terminal", "cmd"),
        primary_selector="app-workspace",
        surface_role="Pane",
        surface_name="Application Workspace",
        preferred_channels=("api", "accessibility", "ocr"),
        affordance_hints=("prompt", "stdout", "current directory", "exit code"),
        verification_contracts=(
            "receipt_success",
            "process_launched",
            "clipboard_contains",
        ),
        repair_recipes=(
            "prefer tool executor",
            "read prompt before typing",
            "capture exit code",
        ),
        safe_windows=True,
        live_fire=True,
        api_like=True,
        recommended_mode="tool",
        clipboard_base=0.7,
        latency_base=0.7,
    ),
    AppFamilySpec(
        family="editor",
        app_context="text_editor",
        aliases=("editor", "text_editor"),
        profile_cues=("editor", "document", "text area", "notepad"),
        profile_confidence=0.78,
        launch_target="notepad.exe",
        launch_cues=("notepad", "winword"),
        primary_selector="document-canvas",
        surface_role="Document",
        surface_name="Document Canvas",
        preferred_channels=("accessibility", "api", "clipboard", "explore"),
        affordance_hints=("text area", "tabs", "command palette", "status bar"),
        verification_contracts=("field_contains", "file_exists", "state_changed"),
        repair_recipes=(
            "focus editor",
            "use clipboard for large text",
            "verify save path",
        ),
        safe_windows=True,
        live_fire=True,
        clipboard_base=0.7,
    ),
    AppFamilySpec(
        family="office_form",
        app_context="office_form",
        aliases=("office_form", "spreadsheet"),
        profile_cues=("ribbon", "spreadsheet", "worksheet", "cell"),
        profile_confidence=0.75,
        launch_target="excel.exe",
        launch_cues=("excel", "calc", "spreadsheet"),
        primary_selector="spreadsheet-grid",
        surface_role="Table",
        surface_name="Spreadsheet Grid",
        preferred_channels=("accessibility", "ocr", "explore"),
        affordance_hints=("ribbon", "cells", "form fields", "save/export controls"),
        verification_contracts=("field_contains", "export_hash_changed", "file_exists"),
        repair_recipes=(
            "select active cell",
            "verify export artifact",
            "recover focus",
        ),
        live_fire=True,
    ),
    AppFamilySpec(
        family="pdf_viewer",
        app_context="pdf_viewer",
        aliases=("pdf_viewer",),
        profile_cues=("pdf", "page", "zoom", "acrobat"),
        profile_confidence=0.76,
        launch_target="acrobat.exe",
        launch_cues=("acrobat", "pdf"),
        primary_selector="pdf-search-box",
        surface_role="Edit",
        surface_name="PDF Search Box",
        preferred_channels=("accessibility", "ocr", "explore"),
        affordance_hints=("page text", "search box", "zoom", "download button"),
        verification_contracts=("field_contains", "file_exists", "state_changed"),
        repair_recipes=(
            "use search",
            "verify page text",
            "download via browser if possible",
        ),
        live_fire=True,
    ),
    AppFamilySpec(
        family="chat_app",
        app_context="chat_app",
        aliases=("chat_app",),
        profile_cues=("message", "send", "chat", "conversation"),
        profile_confidence=0.72,
        launch_target="teams.exe",
        launch_cues=("teams", "chat"),
        primary_selector="chat-composer",
        surface_role="Edit",
        surface_name="Chat Composer",
        preferred_channels=("accessibility", "ocr", "clipboard", "explore"),
        affordance_hints=("message composer", "send button", "thread list"),
        verification_contracts=("field_contains", "receipt_success", "state_changed"),
        repair_recipes=("focus composer", "paste large text", "verify message echo"),
        action_policy={
            "require_approval_terms": (
                "send",
                "post",
                "publish",
                "share",
                "invite",
            ),
            "require_approval_selectors": (
                "send",
                "post",
                "publish",
                "share",
            ),
            "forbidden_terms": (
                "channel broadcast",
                "mass notify",
                "notify everyone",
            ),
            "forbidden_selectors": (
                "channel broadcast",
                "mass notify",
            ),
        },
        clipboard_base=0.7,
        latency_base=0.7,
        live_fire=True,
    ),
    AppFamilySpec(
        family="email",
        app_context="email",
        aliases=("email", "mail", "inbox"),
        profile_cues=("inbox", "subject", "compose", "draft", "attachment"),
        profile_confidence=0.78,
        launch_target="outlook.exe",
        launch_cues=("outlook", "email", "mail", "inbox"),
        primary_selector="email-search-box",
        surface_role="Edit",
        surface_name="Email Search Box",
        preferred_channels=("accessibility", "api", "ocr", "clipboard", "explore"),
        affordance_hints=(
            "message search",
            "inbox list",
            "recipient field",
            "attachment controls",
        ),
        verification_contracts=(
            "field_contains",
            "send_outcome",
            "receipt_success",
        ),
        repair_recipes=(
            "search inbox first",
            "keep draft unsent",
            "verify attachment field",
        ),
        eval_surface=False,
        api_like=True,
        clipboard_base=0.68,
        latency_base=0.6,
        action_policy={
            "require_approval_terms": (
                "send",
                "forward",
                "reply all",
                "share externally",
            ),
            "require_approval_selectors": (
                "send",
                "forward",
                "reply all",
            ),
            "forbidden_terms": (
                "purge inbox",
                "delete all email",
                "empty trash",
            ),
            "forbidden_selectors": (
                "purge inbox",
                "empty trash",
            ),
        },
    ),
    AppFamilySpec(
        family="calendar",
        app_context="calendar",
        aliases=("calendar", "events"),
        profile_cues=("calendar", "meeting", "appointment", "schedule", "event"),
        profile_confidence=0.77,
        launch_target="outlook.exe /select outlook:calendar",
        launch_cues=("calendar", "meeting", "appointment", "schedule"),
        primary_selector="calendar-grid",
        surface_role="Table",
        surface_name="Calendar Grid",
        preferred_channels=("accessibility", "api", "ocr", "clipboard", "explore"),
        affordance_hints=(
            "calendar grid",
            "date picker",
            "event editor",
            "invite list",
        ),
        verification_contracts=(
            "field_contains",
            "invite_outcome",
            "receipt_success",
        ),
        repair_recipes=(
            "search by event title",
            "open draft event",
            "avoid sending invites automatically",
        ),
        eval_surface=False,
        api_like=True,
        clipboard_base=0.62,
        latency_base=0.58,
        action_policy={
            "require_approval_terms": (
                "send invite",
                "share calendar",
                "cancel meeting",
                "delete event",
            ),
            "require_approval_selectors": (
                "send invite",
                "share calendar",
                "cancel",
                "delete",
            ),
            "forbidden_terms": (
                "delete all events",
                "share whole calendar publicly",
            ),
            "forbidden_selectors": (
                "delete all events",
                "share publicly",
            ),
        },
    ),
    AppFamilySpec(
        family="settings",
        app_context="settings",
        aliases=("settings", "system_settings"),
        profile_cues=("settings", "night light", "bluetooth", "wi-fi", "system"),
        profile_confidence=0.76,
        launch_target="settings.exe",
        launch_cues=("settings", "night light", "bluetooth", "wi-fi", "wifi"),
        primary_selector="settings-search-box",
        surface_role="Edit",
        surface_name="Settings Search Box",
        preferred_channels=("accessibility", "api", "ocr", "explore"),
        affordance_hints=(
            "settings search",
            "toggle",
            "category list",
            "status text",
        ),
        verification_contracts=(
            "field_contains",
            "toggle_state",
            "receipt_success",
        ),
        repair_recipes=(
            "search the setting first",
            "verify toggle state",
            "avoid security-sensitive pages",
        ),
        eval_surface=False,
        api_like=True,
        latency_base=0.55,
        action_policy={
            "require_approval_terms": (
                "remote desktop",
                "network adapter",
                "firewall rule",
                "developer mode",
            ),
            "require_approval_selectors": (
                "remote desktop",
                "firewall",
                "developer mode",
            ),
            "forbidden_terms": (
                "disable defender",
                "disable firewall",
                "turn off smartscreen",
            ),
            "forbidden_selectors": (
                "disable defender",
                "disable firewall",
                "turn off smartscreen",
            ),
        },
    ),
    AppFamilySpec(
        family="electron_app",
        app_context="electron_app",
        aliases=("electron_app",),
        profile_cues=("chrome_widgetwin", "electron"),
        profile_confidence=0.66,
        launch_target="electron-demo.exe",
        launch_cues=("electron",),
        primary_selector="electron-command-palette",
        surface_role="Edit",
        surface_name="Electron Command Palette",
        preferred_channels=("accessibility", "dom", "ocr", "explore"),
        affordance_hints=("webview", "command palette", "sidebars", "modal overlays"),
        verification_contracts=("state_changed", "field_contains", "modal_closed"),
        repair_recipes=(
            "refresh accessibility",
            "try DOM bridge",
            "fall back to marks",
        ),
        dom_like=True,
        recommended_mode="hybrid",
    ),
    AppFamilySpec(
        family="design_canvas",
        app_context="design_canvas",
        aliases=("design_canvas",),
        profile_cues=("canvas", "layers", "brush", "photoshop", "design"),
        profile_confidence=0.75,
        launch_target="photoshop.exe",
        launch_cues=("photoshop", "designer", "adobe", "paint", "gimp"),
        primary_selector="drawing-canvas",
        surface_role="Canvas",
        surface_name="Drawing Canvas",
        preferred_channels=("accessibility", "vision", "ocr", "explore"),
        affordance_hints=("canvas", "layers", "tool palette", "export controls"),
        verification_contracts=(
            "state_changed",
            "export_hash_changed",
            "receipt_success",
        ),
        repair_recipes=(
            "use safe non-destructive tool",
            "verify canvas changed",
            "recover focus",
        ),
        live_fire=True,
        visual_heavy=True,
        recommended_mode="hybrid",
    ),
    AppFamilySpec(
        family="trading_terminal",
        app_context="trading_terminal",
        aliases=("trading_terminal",),
        profile_cues=("order ticket", "watchlist", "bid", "ask", "position"),
        profile_confidence=0.78,
        launch_target="trading-terminal.exe",
        launch_cues=("trading",),
        primary_selector="order-ticket",
        surface_role="Edit",
        surface_name="Trading Order Ticket",
        preferred_channels=("accessibility", "api", "ocr", "explore"),
        affordance_hints=("watchlist", "chart", "order ticket", "position grid"),
        verification_contracts=("field_contains", "receipt_success", "state_changed"),
        repair_recipes=(
            "avoid submit/send actions",
            "verify order ticket fields",
            "prefer read-only paths",
        ),
        live_fire=True,
        api_like=True,
        recommended_mode="hybrid",
        latency_base=0.7,
        action_policy={
            "require_approval_terms": (
                "submit order",
                "place order",
                "buy",
                "sell",
                "trade",
                "confirm order",
            ),
            "require_approval_selectors": (
                "submit order",
                "place order",
                "buy",
                "sell",
                "confirm",
            ),
            "forbidden_terms": (
                "disable risk limits",
                "wire transfer",
                "withdraw all",
            ),
            "forbidden_selectors": (
                "disable risk limits",
                "wire transfer",
                "withdraw all",
            ),
        },
    ),
    AppFamilySpec(
        family="enterprise_grid",
        app_context="enterprise_grid",
        aliases=("enterprise_grid",),
        profile_cues=("record grid", "case queue", "customer id", "enterprise"),
        profile_confidence=0.72,
        launch_target="enterprise-grid.exe",
        launch_cues=("enterprise",),
        primary_selector="enterprise-record-grid",
        surface_role="Table",
        surface_name="Enterprise Record Grid",
        preferred_channels=("accessibility", "api", "ocr", "explore"),
        affordance_hints=(
            "record grid",
            "filter field",
            "details pane",
            "workflow buttons",
        ),
        verification_contracts=("field_contains", "state_changed", "receipt_success"),
        repair_recipes=(
            "filter before editing",
            "verify selected record",
            "respect approval boundaries",
        ),
        live_fire=True,
        api_like=True,
        action_policy={
            "require_approval_terms": (
                "bulk delete",
                "mass update",
                "approve all",
                "export all",
                "close case",
                "delete record",
            ),
            "require_approval_selectors": (
                "bulk delete",
                "mass update",
                "approve all",
                "export all",
                "close case",
                "delete",
            ),
            "forbidden_terms": (
                "purge tenant",
                "disable audit",
                "drop table",
            ),
            "forbidden_selectors": (
                "purge tenant",
                "disable audit",
            ),
        },
    ),
    AppFamilySpec(
        family="unknown",
        app_context="unknown",
        aliases=("unknown",),
        profile_cues=(),
        profile_confidence=0.45,
        launch_target="agentos-eval-app",
        launch_cues=(),
        primary_selector="app-workspace",
        surface_role="Pane",
        surface_name="Application Workspace",
        preferred_channels=("accessibility", "ocr", "vision", "explore"),
        affordance_hints=("visible controls", "focused region", "modal state"),
        verification_contracts=("state_changed", "modal_closed", "receipt_success"),
        repair_recipes=(
            "bounded exploration",
            "ask frontier with marks",
            "escalate low confidence",
        ),
        eval_surface=False,
        recommended_mode="explore",
        visual_heavy=True,
    ),
)


def app_family_specs(*, include_unknown: bool = True) -> tuple[AppFamilySpec, ...]:
    if include_unknown:
        return APP_FAMILY_SPECS
    return tuple(spec for spec in APP_FAMILY_SPECS if spec.family != "unknown")


def app_family_names(*, include_unknown: bool = True) -> tuple[str, ...]:
    return tuple(
        spec.family for spec in app_family_specs(include_unknown=include_unknown)
    )


def eval_surface_families() -> tuple[str, ...]:
    return tuple(spec.family for spec in APP_FAMILY_SPECS if spec.eval_surface)


def live_fire_families() -> frozenset[str]:
    return frozenset(spec.family for spec in APP_FAMILY_SPECS if spec.live_fire)


def safe_windows_families() -> tuple[str, ...]:
    return tuple(spec.family for spec in APP_FAMILY_SPECS if spec.safe_windows)


def spec_for_family(family: str) -> AppFamilySpec:
    lookup = {spec.family: spec for spec in APP_FAMILY_SPECS}
    return lookup.get(family, lookup["unknown"])


def family_for_context(app_context: str) -> str | None:
    normalized = app_context.strip().lower()
    if normalized in {"", "unknown"}:
        return None
    for spec in APP_FAMILY_SPECS:
        if normalized == spec.family or normalized in spec.aliases:
            return spec.family
    return None


def profile_rules() -> list[tuple[str, float, tuple[str, ...]]]:
    return [
        (spec.family, spec.profile_confidence, spec.profile_cues)
        for spec in APP_FAMILY_SPECS
        if spec.profile_cues
    ]


def adapter_specs() -> list[tuple[str, list[str], list[str], list[str], list[str]]]:
    return [
        (
            spec.family,
            list(spec.preferred_channels),
            list(spec.affordance_hints),
            list(spec.verification_contracts),
            list(spec.repair_recipes),
        )
        for spec in APP_FAMILY_SPECS
    ]


def launch_target_for_family(family: str) -> str:
    return spec_for_family(family).launch_target


def primary_selector_for_family(family: str) -> str:
    return spec_for_family(family).primary_selector


def app_context_for_family(family: str) -> str:
    return spec_for_family(family).app_context


def sandbox_surface_for_app(app_name: str) -> tuple[str, str, str] | None:
    lower = app_name.lower()
    for spec in APP_FAMILY_SPECS:
        if lower == spec.launch_target.lower():
            return spec.primary_selector, spec.surface_role, spec.surface_name
    for spec in APP_FAMILY_SPECS:
        if any(cue in lower for cue in spec.launch_cues):
            return spec.primary_selector, spec.surface_role, spec.surface_name
    return None


def spec_metadata(spec: AppFamilySpec) -> dict[str, str | bool]:
    payload: dict[str, str | bool] = {
        "app_family": spec.family,
        "app_context": spec.app_context,
        "sandbox": True,
        "adaptive_registry": True,
    }
    payload.update(spec.extra_metadata)
    return payload
