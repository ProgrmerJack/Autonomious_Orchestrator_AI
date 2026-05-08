use serde_json::{json, Map, Value};

use crate::models::{default_bounds, NodeRecord};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SurfaceSpec {
    pub(crate) family: &'static str,
    pub(crate) node_id: &'static str,
    pub(crate) role: &'static str,
    pub(crate) name: &'static str,
}

pub(crate) fn surface_for_app_name(app_name: &str) -> SurfaceSpec {
    let lower = app_name.to_lowercase();
    if contains_any(&lower, &["browser", "edge", "chrome"]) {
        return SurfaceSpec {
            family: "browser",
            node_id: "browser-address-bar",
            role: "Edit",
            name: "Address and search bar",
        };
    }
    if lower.contains("explorer") {
        return SurfaceSpec {
            family: "file_explorer",
            node_id: "explorer-file-list",
            role: "List",
            name: "Explorer File List",
        };
    }
    if lower.contains("trading") {
        return SurfaceSpec {
            family: "trading_terminal",
            node_id: "order-ticket",
            role: "Edit",
            name: "Trading Order Ticket",
        };
    }
    if lower.contains("enterprise") {
        return SurfaceSpec {
            family: "enterprise_grid",
            node_id: "enterprise-record-grid",
            role: "Table",
            name: "Enterprise Record Grid",
        };
    }
    if contains_any(&lower, &["powershell", "terminal", "cmd"]) {
        return SurfaceSpec {
            family: "terminal",
            node_id: "app-workspace",
            role: "Pane",
            name: "Application Workspace",
        };
    }
    if contains_any(&lower, &["notepad", "winword"]) {
        return SurfaceSpec {
            family: "editor",
            node_id: "document-canvas",
            role: "Document",
            name: "Document Canvas",
        };
    }
    if contains_any(&lower, &["excel", "calc", "spreadsheet"]) {
        return SurfaceSpec {
            family: "office_form",
            node_id: "spreadsheet-grid",
            role: "Table",
            name: "Spreadsheet Grid",
        };
    }
    if contains_any(&lower, &["acrobat", "pdf"]) {
        return SurfaceSpec {
            family: "pdf_viewer",
            node_id: "pdf-search-box",
            role: "Edit",
            name: "PDF Search Box",
        };
    }
    if contains_any(&lower, &["teams", "chat"]) {
        return SurfaceSpec {
            family: "chat_app",
            node_id: "chat-composer",
            role: "Edit",
            name: "Chat Composer",
        };
    }
    if lower.contains("electron") {
        return SurfaceSpec {
            family: "electron_app",
            node_id: "electron-command-palette",
            role: "Edit",
            name: "Electron Command Palette",
        };
    }
    if contains_any(&lower, &["photoshop", "designer", "adobe", "paint", "gimp"]) {
        return SurfaceSpec {
            family: "design_canvas",
            node_id: "drawing-canvas",
            role: "Canvas",
            name: "Drawing Canvas",
        };
    }
    if lower.contains("powerpnt") {
        return SurfaceSpec {
            family: "presentation",
            node_id: "presentation-canvas",
            role: "Document",
            name: "Presentation Canvas",
        };
    }
    if lower.contains("code") || lower.contains("vscode") {
        return SurfaceSpec {
            family: "editor",
            node_id: "editor-canvas",
            role: "Document",
            name: "Editor Canvas",
        };
    }
    SurfaceSpec {
        family: "unknown",
        node_id: "app-workspace",
        role: "Pane",
        name: "Application Workspace",
    }
}

pub(crate) fn surface_node(spec: &SurfaceSpec) -> NodeRecord {
    let mut node = node_record(spec.node_id, spec.role, spec.name);
    node.metadata.insert(
        "app_family".to_string(),
        Value::String(spec.family.to_string()),
    );
    node.metadata.insert(
        "app_context".to_string(),
        Value::String(spec.family.to_string()),
    );
    node.metadata.insert(
        "panel_type".to_string(),
        Value::String("primary".to_string()),
    );
    node
}

pub(crate) fn support_panel_nodes(spec: &SurfaceSpec, window_id: &str) -> Vec<NodeRecord> {
    let panels: Vec<(&str, &str, &str, Vec<i32>, &str)> = match spec.family {
        "browser" => vec![
            (
                "browser-tabs",
                "TabList",
                "Browser Tabs",
                vec![180, 118, 900, 28],
                "tab_strip",
            ),
            (
                "browser-main-doc",
                "Document",
                "Browser Document",
                vec![180, 170, 860, 560],
                "document",
            ),
            (
                "browser-research-panel",
                "Pane",
                "Research Side Panel",
                vec![1048, 170, 280, 560],
                "side_panel",
            ),
        ],
        "file_explorer" => vec![
            (
                "explorer-navigation-tree",
                "Tree",
                "Explorer Navigation Tree",
                vec![180, 160, 240, 620],
                "navigation",
            ),
            (
                "explorer-preview-pane",
                "Pane",
                "Explorer Preview Pane",
                vec![1060, 160, 260, 620],
                "preview",
            ),
        ],
        "terminal" => vec![
            (
                "terminal-toolbar",
                "ToolBar",
                "Terminal Toolbar",
                vec![180, 160, 920, 44],
                "toolbar",
            ),
            (
                "terminal-input",
                "Edit",
                "Sandbox Terminal",
                vec![180, 210, 920, 570],
                "primary",
            ),
        ],
        "editor" => vec![
            (
                "editor-explorer",
                "Tree",
                "Editor Explorer",
                vec![180, 160, 230, 620],
                "navigation",
            ),
            (
                "editor-outline",
                "Tree",
                "Editor Outline",
                vec![1120, 160, 200, 620],
                "side_panel",
            ),
        ],
        "office_form" => vec![
            (
                "office-ribbon",
                "ToolBar",
                "Office Ribbon",
                vec![180, 150, 1140, 76],
                "toolbar",
            ),
            (
                "formula-bar",
                "Edit",
                "Formula Bar",
                vec![180, 232, 1140, 34],
                "formula",
            ),
        ],
        "pdf_viewer" => vec![
            (
                "pdf-thumbnail-pane",
                "List",
                "PDF Thumbnail Pane",
                vec![180, 160, 220, 620],
                "navigation",
            ),
            (
                "pdf-document",
                "Document",
                "PDF Document",
                vec![410, 160, 690, 620],
                "primary",
            ),
        ],
        "chat_app" => vec![
            (
                "chat-thread-list",
                "List",
                "Chat Thread List",
                vec![180, 160, 260, 620],
                "navigation",
            ),
            (
                "chat-history",
                "Document",
                "Chat History",
                vec![450, 160, 640, 520],
                "primary",
            ),
        ],
        "design_canvas" => vec![
            (
                "design-toolbox",
                "ToolBar",
                "Design Toolbox",
                vec![180, 160, 90, 620],
                "toolbar",
            ),
            (
                "layers-panel",
                "Pane",
                "Layers Panel",
                vec![1050, 160, 270, 620],
                "side_panel",
            ),
        ],
        "trading_terminal" => vec![
            (
                "market-watchlist",
                "Table",
                "Market Watchlist",
                vec![180, 160, 280, 620],
                "watchlist",
            ),
            (
                "price-chart",
                "Chart",
                "Price Chart",
                vec![470, 160, 560, 390],
                "chart",
            ),
            (
                "positions-grid",
                "Table",
                "Positions Grid",
                vec![470, 560, 850, 220],
                "positions",
            ),
        ],
        "enterprise_grid" => vec![
            (
                "enterprise-filter-panel",
                "Pane",
                "Enterprise Filter Panel",
                vec![180, 160, 260, 620],
                "filters",
            ),
            (
                "enterprise-detail-panel",
                "Pane",
                "Enterprise Detail Panel",
                vec![1080, 160, 240, 620],
                "detail",
            ),
        ],
        _ => Vec::new(),
    };
    panels
        .into_iter()
        .filter(|(node_id, _, _, _, _)| *node_id != spec.node_id)
        .map(|(node_id, role, name, bounds, panel_type)| {
            let mut node = panel_node(node_id, role, name, spec.family, panel_type, bounds);
            node.metadata
                .insert("parent".to_string(), Value::String(window_id.to_string()));
            node
        })
        .collect()
}

pub(crate) fn node_record(node_id: &str, role: &str, name: &str) -> NodeRecord {
    node_record_with_bounds(node_id, role, name, vec![180, 160, 920, 620])
}

pub(crate) fn node_record_with_bounds(
    node_id: &str,
    role: &str,
    name: &str,
    bounds: Vec<i32>,
) -> NodeRecord {
    let mut metadata = Map::new();
    metadata.insert("sandbox".to_string(), Value::Bool(true));
    metadata.insert("adaptive_registry".to_string(), Value::Bool(true));
    NodeRecord {
        node_id: node_id.to_string(),
        role: role.to_string(),
        name: name.to_string(),
        focused: false,
        enabled: true,
        bounds,
        text: String::new(),
        value: String::new(),
        metadata,
    }
}

pub(crate) fn window_node(node_id: &str, name: &str) -> NodeRecord {
    node_record_with_bounds(node_id, "Window", name, vec![140, 110, 1040, 760])
}

pub(crate) fn edit_node(node_id: &str, name: &str, value: &str) -> NodeRecord {
    let mut node = node_record(node_id, "Edit", name);
    node.value = value.to_string();
    node.text = value.to_string();
    node
}

pub(crate) fn panel_node(
    node_id: &str,
    role: &str,
    name: &str,
    family: &str,
    panel_type: &str,
    bounds: Vec<i32>,
) -> NodeRecord {
    let mut node = node_record_with_bounds(node_id, role, name, bounds);
    node.metadata
        .insert("app_family".to_string(), Value::String(family.to_string()));
    node.metadata.insert(
        "panel_type".to_string(),
        Value::String(panel_type.to_string()),
    );
    node
}

pub(crate) fn document_node(node_id: &str, name: &str) -> NodeRecord {
    node_record(node_id, "Document", name)
}

pub(crate) fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

pub(crate) fn capabilities_payload() -> Value {
    json!({
        "type": "sandbox.capabilities",
        "status": "ok",
        "sandbox": true,
        "is_simulated": true,
        "rights": "simulated-virtual-rights",
        "capabilities": [
            "snapshot",
            "act",
            "exec",
            "reset",
            "native-snapshot",
            "native-act",
            "native-input",
            "adaptive-app-surfaces",
            "multi-panel-app-surfaces",
            "virtual-file-mutation",
            "virtual-filesystem",
            "virtual-processes",
            "clipboard",
            "modal-mutation",
            "sandbox-confined-simulated-privileges",
            "stateful-control-receipts",
            "history-preserving-reset",
            "window-panel-mutation"
        ],
        "families": [
            "browser",
            "file_explorer",
            "terminal",
            "editor",
            "office_form",
            "pdf_viewer",
            "chat_app",
            "electron_app",
            "design_canvas",
            "presentation",
            "trading_terminal",
            "enterprise_grid",
            "unknown"
        ]
    })
}

pub(crate) fn empty_bounds() -> Vec<i32> {
    default_bounds()
}
