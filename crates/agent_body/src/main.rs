use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "--health") {
        println!("{{\"status\":\"ok\",\"name\":\"agent_body\",\"version\":\"0.1.0\"}}");
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--describe") {
        println!(
            "{{\"capabilities\":[\"event-bridge\",\"stdin-actions\",\"snapshot\",\"act\",\"exec\",\"capabilities\",\"sandbox-state\",\"adaptive-app-surfaces\",\"full-rights-virtual-sandbox\"]}}"
        );
        return Ok(());
    }
    let state_path = state_path_from_args(&args);
    if let Some(command_json) = arg_value(&args, "--command-json") {
        return run_single_command(state_path.as_deref(), command_json);
    }
    serve(state_path.as_deref())
}

fn serve(state_path: Option<&Path>) -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    writeln!(stdout, "{{\"type\":\"body.started\",\"status\":\"ready\"}}")?;
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = handle_command_str(state_path, trimmed)?;
        writeln!(stdout, "{}", response)?;
        stdout.flush()?;
    }
    Ok(())
}

fn run_single_command(state_path: Option<&Path>, command_json: &str) -> io::Result<()> {
    println!("{}", handle_command_str(state_path, command_json)?);
    Ok(())
}

fn handle_command_str(state_path: Option<&Path>, command_json: &str) -> io::Result<String> {
    let command = parse_command(command_json)?;
    let mut state = load_state(state_path)?;
    let response = handle_command(&mut state, command);
    save_state(state_path, &state)?;
    Ok(response.to_string())
}

fn parse_command(command_json: &str) -> io::Result<CommandEnvelope> {
    serde_json::from_str(command_json).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid command json: {error}"),
        )
    })
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn state_path_from_args(args: &[String]) -> Option<PathBuf> {
    arg_value(args, "--state-file")
        .or_else(|| arg_value(args, "--state-path"))
        .map(PathBuf::from)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeRecord {
    node_id: String,
    role: String,
    name: String,
    focused: bool,
    enabled: bool,
    #[serde(default = "default_bounds")]
    bounds: Vec<i32>,
    text: String,
    value: String,
    metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SandboxState {
    focused: String,
    last_action: Option<Value>,
    virtual_files: Vec<String>,
    terminal_log: Vec<String>,
    nodes: Vec<NodeRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommandEnvelope {
    Snapshot,
    Capabilities,
    Reset,
    Exec {
        argv: Vec<String>,
    },
    Act {
        action_type: String,
        selector: String,
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        metadata: Option<Map<String, Value>>,
    },
}

fn load_state(state_path: Option<&Path>) -> io::Result<SandboxState> {
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

fn save_state(state_path: Option<&Path>, state: &SandboxState) -> io::Result<()> {
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

fn default_state() -> SandboxState {
    SandboxState {
        focused: "window-browser".to_string(),
        last_action: None,
        virtual_files: vec![
            "artifacts/workflows/report.md".to_string(),
            "artifacts/workflows/slides.pptx".to_string(),
            "artifacts/workflows/notes.txt".to_string(),
        ],
        terminal_log: Vec::new(),
        nodes: vec![
            node_record("window-browser", "Window", "Sandbox Browser"),
            edit_node(
                "browser-address-bar",
                "Address and search bar",
                "about:blank",
            ),
            document_node("browser-main-doc", "Blank Page"),
            document_node("terminal-input", "Sandbox Terminal"),
        ],
    }
}

fn handle_command(state: &mut SandboxState, command: CommandEnvelope) -> Value {
    match command {
        CommandEnvelope::Snapshot => json!({
            "type": "sandbox.snapshot",
            "status": "ok",
            "focused": state.focused,
            "virtual_files": state.virtual_files,
            "terminal_log": state.terminal_log,
            "nodes": state.nodes,
        }),
        CommandEnvelope::Capabilities => capabilities_payload(),
        CommandEnvelope::Reset => {
            *state = default_state();
            json!({
                "type": "sandbox.reset",
                "status": "reset",
                "sandbox": true,
                "rights": "full-virtual-rights",
            })
        }
        CommandEnvelope::Exec { argv } => execute_command(state, argv),
        CommandEnvelope::Act {
            action_type,
            selector,
            value,
            metadata,
        } => apply_action(state, &action_type, &selector, value.as_deref(), metadata),
    }
}

fn execute_command(state: &mut SandboxState, argv: Vec<String>) -> Value {
    let joined = argv.join(" ");
    ensure_terminal_node(state);
    state.focused = "terminal-input".to_string();
    state.terminal_log.push(joined.clone());
    if let Some(node) = find_node_mut(state, "terminal-input") {
        node.focused = true;
        node.text = joined.clone();
        node.value = joined.clone();
    }
    let receipt = json!({
        "type": "sandbox.exec",
        "status": "executed",
        "sandbox": true,
        "rights": "full-virtual-rights",
        "argv": argv,
        "selector": "terminal-input",
        "exit_code": 0,
        "stdout": format!("Executed inside sandbox: {joined}"),
    });
    state.last_action = Some(receipt.clone());
    receipt
}

fn apply_action(
    state: &mut SandboxState,
    action_type: &str,
    selector: &str,
    value: Option<&str>,
    metadata: Option<Map<String, Value>>,
) -> Value {
    let metadata = metadata.unwrap_or_default();
    if action_type == "launch_app" {
        let app_name = value.unwrap_or(selector);
        let surface = launch_app(state, app_name);
        let receipt = json!({
            "type": "sandbox.act",
            "status": "launched",
            "sandbox": true,
            "rights": "full-virtual-rights",
            "action_type": action_type,
            "selector": selector,
            "launched": app_name,
            "surface": surface,
        });
        state.last_action = Some(receipt.clone());
        return receipt;
    }

    let matched = find_node_index(state, selector);
    if requires_node(action_type) && matched.is_none() {
        let receipt = json!({
            "type": "sandbox.act",
            "status": "selector-not-found",
            "sandbox": true,
            "rights": "full-virtual-rights",
            "action_type": action_type,
            "selector": selector,
        });
        state.last_action = Some(receipt.clone());
        return receipt;
    }

    if let Some(index) = matched {
        let mut extra = Map::new();
        match action_type {
            "focus" | "click" | "invoke" => focus_node(state, index),
            "type" | "set_text" | "set_value" => {
                set_node_text(state, index, value.unwrap_or(""));
                if state.nodes[index].node_id == "spreadsheet-grid" {
                    let cell_edit = cell_edit_payload(&metadata, value.unwrap_or(""));
                    apply_cell_edit(state, index, &cell_edit);
                    extra.insert("cell_edit".to_string(), cell_edit);
                }
            }
            "cell_edit" => {
                let cell_edit = cell_edit_payload(&metadata, value.unwrap_or(""));
                apply_cell_edit(state, index, &cell_edit);
                extra.insert("cell_edit".to_string(), cell_edit);
            }
            "draw_path" => {
                if let Some(node) = state.nodes.get_mut(index) {
                    node.metadata.insert(
                        "last_path".to_string(),
                        Value::String(value.unwrap_or("").to_string()),
                    );
                }
            }
            "copy_file" | "move_file" | "rename_file" => {
                let file_op = mutate_virtual_files(state, action_type, &metadata);
                extra.insert("file_op".to_string(), file_op);
            }
            _ => {}
        }

        let node = &state.nodes[index];
        let status = match action_type {
            "focus" | "click" | "invoke" => "focused",
            "type" | "set_text" | "set_value" => "value-set",
            "cell_edit" => "value-set",
            "draw_path" => "drawn",
            "copy_file" | "move_file" | "rename_file" => "file-op-executed",
            _ => "executed",
        };
        let mut receipt = json!({
            "type": "sandbox.act",
            "status": status,
            "sandbox": true,
            "rights": "full-virtual-rights",
            "action_type": action_type,
            "selector": selector,
            "matched_name": node.name,
            "matched_role": node.role,
            "matched_node_id": node.node_id,
            "value": value,
        });
        if let Some(receipt_map) = receipt.as_object_mut() {
            for (key, value) in extra {
                receipt_map.insert(key, value);
            }
        }
        state.last_action = Some(receipt.clone());
        return receipt;
    }

    let receipt = json!({
        "type": "sandbox.act",
        "status": "executed",
        "sandbox": true,
        "rights": "full-virtual-rights",
        "action_type": action_type,
        "selector": selector,
        "value": value,
    });
    state.last_action = Some(receipt.clone());
    receipt
}

fn requires_node(action_type: &str) -> bool {
    matches!(
        action_type,
        "focus"
            | "click"
            | "invoke"
            | "type"
            | "set_text"
            | "set_value"
            | "cell_edit"
            | "draw_path"
            | "copy_file"
            | "move_file"
            | "rename_file"
    )
}

fn find_node_index(state: &SandboxState, selector: &str) -> Option<usize> {
    let normalized = selector.to_lowercase();
    state
        .nodes
        .iter()
        .position(|node| node_matches(node, &normalized))
}

fn node_matches(node: &NodeRecord, selector: &str) -> bool {
    if let Some(name) = selector.strip_prefix("name=") {
        return node.name.to_lowercase().contains(name);
    }
    if let Some(role) = selector.strip_prefix("role=") {
        return node.role.to_lowercase().contains(role);
    }
    if selector == "drawing-canvas" || selector == "design-canvas" {
        let node_id = node.node_id.to_lowercase();
        let name = node.name.to_lowercase();
        return node_id == "drawing-canvas"
            || node_id == "design-canvas"
            || name.contains("canvas");
    }
    node.node_id.to_lowercase().contains(selector) || node.name.to_lowercase().contains(selector)
}

fn focus_node(state: &mut SandboxState, index: usize) {
    for node in &mut state.nodes {
        node.focused = false;
    }
    if let Some(node) = state.nodes.get_mut(index) {
        node.focused = true;
        state.focused = node.node_id.clone();
    }
}

fn set_node_text(state: &mut SandboxState, index: usize, value: &str) {
    focus_node(state, index);
    if let Some(node) = state.nodes.get_mut(index) {
        node.text = value.to_string();
        node.value = value.to_string();
    }
}

fn mutate_virtual_files(
    state: &mut SandboxState,
    action_type: &str,
    metadata: &Map<String, Value>,
) -> Value {
    let source = metadata
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("source-item")
        .to_string();
    let destination = metadata
        .get("destination")
        .and_then(Value::as_str)
        .unwrap_or("destination-item")
        .to_string();
    let new_name = metadata
        .get("new_name")
        .and_then(Value::as_str)
        .unwrap_or("renamed-item")
        .to_string();
    match action_type {
        "copy_file" => state.virtual_files.push(destination.clone()),
        "move_file" => {
            state.virtual_files.retain(|item| item != &source);
            state.virtual_files.push(destination.clone());
        }
        "rename_file" => {
            let mut renamed = false;
            for item in &mut state.virtual_files {
                if item == &source {
                    *item = new_name.clone();
                    renamed = true;
                }
            }
            if !renamed {
                state.virtual_files.push(new_name.clone());
            }
        }
        _ => {}
    }
    let operation = action_type.replace("_file", "");
    json!({
        "operation": operation,
        "source": source,
        "destination": if action_type == "rename_file" { Value::Null } else { Value::String(destination) },
        "new_name": if action_type == "rename_file" { Value::String(new_name) } else { Value::Null },
        "resulting_file_count": state.virtual_files.len(),
    })
}

fn cell_edit_payload(metadata: &Map<String, Value>, raw_value: &str) -> Value {
    let (parsed_cell, parsed_value) = parse_cell_edit(raw_value);
    let cell = metadata
        .get("cell")
        .and_then(Value::as_str)
        .unwrap_or(&parsed_cell)
        .to_uppercase();
    let value = metadata
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or(&parsed_value)
        .to_string();
    let formula = metadata
        .get("formula")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| value.starts_with('='));
    let range_edit = metadata
        .get("range_edit")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| cell.contains(':'));
    json!({
        "cell": cell,
        "value": value,
        "formula": formula,
        "range_edit": range_edit,
    })
}

fn parse_cell_edit(raw_value: &str) -> (String, String) {
    if let Some((cell, value)) = raw_value.split_once(':') {
        return (cell.trim().to_uppercase(), value.trim().to_string());
    }
    ("A1".to_string(), raw_value.trim().to_string())
}

fn apply_cell_edit(state: &mut SandboxState, index: usize, payload: &Value) {
    let Some(node) = state.nodes.get_mut(index) else {
        return;
    };
    let cell = payload
        .get("cell")
        .and_then(Value::as_str)
        .unwrap_or("A1")
        .to_string();
    let value = payload
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let range_edit = payload
        .get("range_edit")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let cells = node
        .metadata
        .entry("cells".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(cells_map) = cells {
        if range_edit && cell.contains(':') {
            for part in cell.split(':') {
                cells_map.insert(part.to_string(), Value::String(value.clone()));
            }
        } else {
            cells_map.insert(cell, Value::String(value));
        }
    }
}

fn ensure_terminal_node(state: &mut SandboxState) {
    if find_node_index(state, "terminal-input").is_none() {
        state
            .nodes
            .push(document_node("terminal-input", "Sandbox Terminal"));
    }
}

fn launch_app(state: &mut SandboxState, app_name: &str) -> Value {
    let window_id = format!("window-{}", sanitize_id(app_name));
    if find_node_index(state, &window_id).is_some() {
        if let Some(index) = find_node_index(state, &window_id) {
            focus_node(state, index);
        }
        return json!({"window_id": window_id, "existing": true});
    }
    state.nodes.push(window_node(&window_id, app_name));
    let surface_spec = surface_for_app_name(app_name);
    let surface = surface_node(&surface_spec);
    let payload = json!({
        "family": surface_spec.family,
        "selector": surface_spec.node_id,
        "role": surface_spec.role,
        "name": surface_spec.name,
    });
    state.nodes.push(surface);
    payload
}

fn find_node_mut<'a>(state: &'a mut SandboxState, selector: &str) -> Option<&'a mut NodeRecord> {
    let index = find_node_index(state, selector)?;
    state.nodes.get_mut(index)
}

fn sanitize_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn node_record(node_id: &str, role: &str, name: &str) -> NodeRecord {
    node_record_with_bounds(node_id, role, name, vec![180, 160, 920, 620])
}

fn node_record_with_bounds(node_id: &str, role: &str, name: &str, bounds: Vec<i32>) -> NodeRecord {
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

fn window_node(node_id: &str, name: &str) -> NodeRecord {
    node_record_with_bounds(node_id, "Window", name, vec![140, 110, 1040, 760])
}

fn edit_node(node_id: &str, name: &str, value: &str) -> NodeRecord {
    let mut node = node_record(node_id, "Edit", name);
    node.value = value.to_string();
    node.text = value.to_string();
    node
}

fn document_node(node_id: &str, name: &str) -> NodeRecord {
    node_record(node_id, "Document", name)
}

fn default_bounds() -> Vec<i32> {
    vec![0, 0, 100, 30]
}

#[derive(Debug, Clone, Copy)]
struct SurfaceSpec {
    family: &'static str,
    node_id: &'static str,
    role: &'static str,
    name: &'static str,
}

fn surface_for_app_name(app_name: &str) -> SurfaceSpec {
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

fn surface_node(spec: &SurfaceSpec) -> NodeRecord {
    let mut node = node_record(spec.node_id, spec.role, spec.name);
    node.metadata.insert(
        "app_family".to_string(),
        Value::String(spec.family.to_string()),
    );
    node.metadata.insert(
        "app_context".to_string(),
        Value::String(spec.family.to_string()),
    );
    node
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn capabilities_payload() -> Value {
    json!({
        "type": "sandbox.capabilities",
        "status": "ok",
        "sandbox": true,
        "rights": "full-virtual-rights",
        "capabilities": [
            "snapshot",
            "act",
            "exec",
            "reset",
            "adaptive-app-surfaces",
            "virtual-file-mutation",
            "stateful-control-receipts"
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
            "trading_terminal",
            "enterprise_grid",
            "unknown"
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_default_nodes() {
        let mut state = default_state();
        let payload = handle_command(&mut state, CommandEnvelope::Snapshot);
        let nodes = payload.get("nodes").and_then(Value::as_array).unwrap();
        assert!(nodes.len() >= 3);
    }

    #[test]
    fn act_updates_text_field() {
        let mut state = default_state();
        let payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "set_text".to_string(),
                selector: "name=Address and search bar".to_string(),
                value: Some("https://example.com".to_string()),
                metadata: None,
            },
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("value-set")
        );
        let node = find_node_mut(&mut state, "browser-address-bar").unwrap();
        assert_eq!(node.value, "https://example.com");
    }

    #[test]
    fn exec_records_terminal_command() {
        let mut state = default_state();
        let payload = handle_command(
            &mut state,
            CommandEnvelope::Exec {
                argv: vec!["python".to_string(), "-V".to_string()],
            },
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("executed")
        );
        assert!(state
            .terminal_log
            .iter()
            .any(|entry| entry.contains("python -V")));
    }

    #[test]
    fn capabilities_report_full_rights_virtual_sandbox() {
        let mut state = default_state();
        let payload = handle_command(&mut state, CommandEnvelope::Capabilities);
        assert_eq!(payload.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(
            payload.get("rights").and_then(Value::as_str),
            Some("full-virtual-rights")
        );
        let families = payload.get("families").and_then(Value::as_array).unwrap();
        assert!(families.iter().any(|item| item == "trading_terminal"));
    }

    #[test]
    fn launch_app_adds_adaptive_surface_family() {
        let mut state = default_state();
        let payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "launch_app".to_string(),
                selector: "trading-terminal.exe".to_string(),
                value: Some("trading-terminal.exe".to_string()),
                metadata: None,
            },
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("launched")
        );
        assert!(find_node_index(&state, "order-ticket").is_some());
    }

    #[test]
    fn drawing_canvas_alias_matches_design_surface() {
        let mut state = default_state();
        handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "launch_app".to_string(),
                selector: "mspaint.exe".to_string(),
                value: Some("mspaint.exe".to_string()),
                metadata: None,
            },
        );
        let payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "draw_path".to_string(),
                selector: "drawing-canvas".to_string(),
                value: Some("M 0 0 L 1 1".to_string()),
                metadata: None,
            },
        );
        assert_eq!(payload.get("status").and_then(Value::as_str), Some("drawn"));
    }
}
