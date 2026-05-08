use serde_json::{Map, Value};
use std::fs;
use std::io;
use std::path::Path;

use crate::models::SandboxState;
use crate::surfaces::{document_node, edit_node, node_record, panel_node};

pub(crate) fn load_state(state_path: Option<&Path>) -> io::Result<SandboxState> {
    if let Some(path) = state_path {
        if path.exists() {
            let data = fs::read_to_string(path)?;
            return serde_json::from_str(&data).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid sandbox state: {error}"),
                )
            });
        }
    }
    Ok(default_state())
}

pub(crate) fn save_state(state_path: Option<&Path>, state: &SandboxState) -> io::Result<()> {
    if let Some(path) = state_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(state).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to serialize sandbox state: {error}"),
            )
        })?;
        fs::write(path, body)?;
    }
    Ok(())
}

pub(crate) fn default_state() -> SandboxState {
    SandboxState {
        focused: "window-browser".to_string(),
        last_action: None,
        agent_history: Vec::new(),
        clipboard: String::new(),
        virtual_processes: Vec::new(),
        modals: Vec::new(),
        virtual_file_contents: default_virtual_file_contents(),
        virtual_files: vec![
            "artifacts/workflows/report.md".to_string(),
            "artifacts/workflows/slides.pptx".to_string(),
            "artifacts/workflows/notes.txt".to_string(),
        ],
        terminal_log: Vec::new(),
        nodes: vec![
            node_record("window-browser", "Window", "Sandbox Browser"),
            panel_node(
                "browser-tab-strip",
                "TabList",
                "Browser Tabs",
                "browser",
                "tab_strip",
                vec![120, 92, 900, 26],
            ),
            panel_node(
                "browser-toolbar",
                "ToolBar",
                "Browser Toolbar",
                "browser",
                "toolbar",
                vec![120, 118, 1200, 40],
            ),
            edit_node(
                "browser-address-bar",
                "Address and search bar",
                "about:blank",
            ),
            panel_node(
                "browser-side-panel",
                "Pane",
                "Browser Side Panel",
                "browser",
                "side_panel",
                vec![1030, 170, 290, 760],
            ),
            document_node("browser-main-doc", "Blank Page"),
            panel_node(
                "browser-status-bar",
                "StatusBar",
                "Browser Status Bar",
                "browser",
                "status",
                vec![120, 932, 1200, 24],
            ),
            document_node("terminal-input", "Sandbox Terminal"),
        ],
    }
}

pub(crate) fn default_virtual_file_contents() -> Map<String, Value> {
    let mut contents = Map::new();
    contents.insert(
        "artifacts/workflows/report.md".to_string(),
        Value::String(String::new()),
    );
    contents.insert(
        "artifacts/workflows/slides.pptx".to_string(),
        Value::String(String::new()),
    );
    contents.insert(
        "artifacts/workflows/notes.txt".to_string(),
        Value::String(String::new()),
    );
    contents
}

pub(crate) fn remember_receipt(state: &mut SandboxState, receipt: &Value) {
    state.last_action = Some(receipt.clone());
    state.agent_history.push(receipt.clone());
    if state.agent_history.len() > 5000 {
        let excess = state.agent_history.len() - 5000;
        state.agent_history.drain(0..excess);
    }
}

pub(crate) fn recent_history(history: &[Value]) -> Vec<Value> {
    let start = history.len().saturating_sub(20);
    history[start..].to_vec()
}
