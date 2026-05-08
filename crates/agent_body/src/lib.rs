mod models;
mod native;
mod state;
mod surfaces;

use serde_json::{json, Map, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use crate::models::{CommandEnvelope, NodeRecord, SandboxState};
use crate::state::{default_state, load_state, recent_history, remember_receipt, save_state};
use crate::surfaces::{
    capabilities_payload, document_node, empty_bounds, node_record_with_bounds,
    support_panel_nodes, surface_for_app_name, surface_node, window_node,
};

pub fn run(args: Vec<String>) -> io::Result<()> {
    if args.iter().any(|arg| arg == "--health") {
        println!("{{\"status\":\"ok\",\"name\":\"agent_body\",\"version\":\"0.1.0\"}}");
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--describe") {
        println!(
            "{{\"capabilities\":[\"event-bridge\",\"stdin-actions\",\"snapshot\",\"act\",\"exec\",\"capabilities\",\"native-snapshot\",\"native-act\",\"native-input\",\"sandbox-state\",\"adaptive-app-surfaces\",\"multi-panel-app-surfaces\",\"virtual-filesystem\",\"virtual-processes\",\"clipboard\",\"modal-mutation\",\"history-preserving-reset\",\"window-panel-mutation\",\"simulated-virtual-sandbox\"]}}"
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

fn handle_command(state: &mut SandboxState, command: CommandEnvelope) -> Value {
    match command {
        CommandEnvelope::Snapshot => json!({
            "type": "sandbox.snapshot",
            "status": "ok",
            "focused": state.focused,
            "virtual_files": state.virtual_files,
            "terminal_log": state.terminal_log,
            "clipboard": state.clipboard,
            "virtual_processes": state.virtual_processes,
            "modals": state.modals,
            "virtual_file_contents": state.virtual_file_contents,
            "agent_history_count": state.agent_history.len(),
            "recent_agent_history": recent_history(&state.agent_history),
            "nodes": state.nodes,
        }),
        CommandEnvelope::Capabilities => capabilities_payload(),
        CommandEnvelope::NativeSnapshot => native::snapshot(),
        CommandEnvelope::NativeAct {
            action_type,
            selector,
            value,
            metadata,
        } => native::apply_action(&action_type, &selector, value.as_deref(), metadata),
        CommandEnvelope::Reset => {
            let history = state.agent_history.clone();
            *state = default_state();
            state.agent_history = history;
            json!({
                "type": "sandbox.reset",
                "status": "reset",
                "sandbox": true,
                "rights": "simulated-virtual-rights",
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

    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut exit_code = 0;

    if !argv.is_empty() {
        let cmd = argv[0].as_str();
        match cmd {
            "ls" | "dir" => {
                stdout = state.virtual_files.join("\n");
                if stdout.is_empty() {
                    stdout = ".".to_string();
                }
            }
            "cat" | "type" => {
                if argv.len() > 1 {
                    let path = &argv[1];
                    if let Some(content) = state
                        .virtual_file_contents
                        .get(path)
                        .and_then(|value| value.as_str())
                    {
                        stdout = content.to_string();
                    } else {
                        stderr = format!("{cmd}: {path}: No such file or directory");
                        exit_code = 1;
                    }
                } else {
                    stderr = format!("{cmd}: missing operand");
                    exit_code = 1;
                }
            }
            "echo" => {
                stdout = argv[1..].join(" ");
            }
            "pwd" => {
                stdout = "/sandbox/workspace".to_string();
            }
            "python" | "python3" => {
                if argv.len() > 1 {
                    let path = &argv[1];
                    if state.virtual_file_contents.contains_key(path) {
                        stdout = format!("Simulated execution of {path} completed successfully.");
                    } else if path == "-c" {
                        stdout = "Simulated execution completed.".to_string();
                    } else {
                        stderr = format!(
                            "{cmd}: can't open file '{path}': [Errno 2] No such file or directory"
                        );
                        exit_code = 2;
                    }
                } else {
                    stdout = "Python 3.10.12 (sandbox virtual python)\nType \"help\", \"copyright\", \"credits\" or \"license\" for more information.".to_string();
                }
            }
            "git" => {
                if argv.len() > 1 {
                    match argv[1].as_str() {
                        "status" => stdout = "On branch main\nYour branch is up to date with 'origin/main'.\n\nnothing to commit, working tree clean".to_string(),
                        "log" => stdout = "commit a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0\nAuthor: Sandbox User <sandbox@example.com>\nDate:   Today\n\n    Initial virtual commit".to_string(),
                        "clone" => stdout = "Cloning into virtual repository... done.".to_string(),
                        _ => stdout = format!("Simulated git {} executed successfully.", argv[1]),
                    }
                } else {
                    stdout = "usage: git [--version] [--help] [-C <path>] [-c <name>=<value>]\n           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]\n           [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]\n           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]\n           [--super-prefix=<path>] [--config-env=<name>=<envvar>]\n           <command> [<args>]".to_string();
                }
            }
            "npm" | "yarn" | "pnpm" => {
                stdout = format!("Simulated {} completed in 1.42s.", cmd);
            }
            "curl" | "wget" => {
                stdout = "HTTP/1.1 200 OK\nContent-Type: text/html\n\n<html><head><title>Virtual Page</title></head><body><h1>Simulated Fetch</h1></body></html>".to_string();
            }
            "mkdir" | "touch" | "rm" | "cp" | "mv" => {}
            _ => {
                stderr = format!("bash: {cmd}: command not found");
                exit_code = 127;
            }
        }
    }

    let process = json!({
        "pid": state.virtual_processes.len() + 1,
        "command": joined.clone(),
        "status": "exited",
        "exit_code": exit_code,
        "sandbox": true,
    });
    state.virtual_processes.push(process.clone());

    let combined_output = if !stderr.is_empty() {
        if !stdout.is_empty() {
            format!("{}\n{}", stdout, stderr)
        } else {
            stderr.clone()
        }
    } else {
        stdout.clone()
    };

    if let Some(node) = find_node_mut(state, "terminal-input") {
        node.focused = true;
        node.text = joined.clone();
        node.value = joined.clone();
    }
    let receipt = json!({
        "type": "sandbox.exec",
        "status": "executed",
        "sandbox": true,
        "rights": "simulated-virtual-rights",
        "argv": argv,
        "selector": "terminal-input",
        "exit_code": exit_code,
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "output": combined_output,
    });
    remember_receipt(state, &receipt);
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
            "rights": "simulated-virtual-rights",
            "action_type": action_type,
            "selector": selector,
            "launched": app_name,
            "surface": surface,
        });
        remember_receipt(state, &receipt);
        return receipt;
    }

    if let Some(receipt) =
        apply_virtual_system_action(state, action_type, selector, value, &metadata)
    {
        remember_receipt(state, &receipt);
        return receipt;
    }

    let matched = find_node_index(state, selector);
    if requires_node(action_type) && matched.is_none() {
        let receipt = json!({
            "type": "sandbox.act",
            "status": "selector-not-found",
            "sandbox": true,
            "rights": "simulated-virtual-rights",
            "action_type": action_type,
            "selector": selector,
        });
        remember_receipt(state, &receipt);
        return receipt;
    }

    if let Some(index) = matched {
        let mut extra = Map::new();
        match action_type {
            "focus" | "click" | "invoke" => focus_node(state, index),
            "type" | "set_text" | "set_value" => {
                set_node_text(state, index, value.unwrap_or(""));
                sync_browser_address_bar_navigation(state, index, value.unwrap_or(""));
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
            "move_window" | "resize_window" | "open_panel" | "close_panel" | "select_tab" => {
                let panel_receipt =
                    mutate_window_panel(state, index, action_type, value, &metadata);
                if let Some(panel_map) = panel_receipt.as_object() {
                    for (key, value) in panel_map {
                        extra.insert(key.clone(), value.clone());
                    }
                }
            }
            _ => {}
        }

        let node = &state.nodes[index];
        let status = match action_type {
            "focus" | "click" | "invoke" => "focused",
            "type" | "set_text" | "set_value" | "cell_edit" => "value-set",
            "draw_path" => "drawn",
            "copy_file" | "move_file" | "rename_file" => "file-op-executed",
            "move_window" | "resize_window" => "window-updated",
            "open_panel" => "panel-opened",
            "close_panel" => "panel-closed",
            "select_tab" => "tab-selected",
            _ => "executed",
        };
        let mut receipt = json!({
            "type": "sandbox.act",
            "status": status,
            "sandbox": true,
            "rights": "simulated-virtual-rights",
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
        remember_receipt(state, &receipt);
        return receipt;
    }

    if matches!(action_type, "navigate" | "goto" | "open_url") {
        let url = value.unwrap_or(selector);
        update_browser_url(state, url);
        let receipt = json!({
            "type": "sandbox.act",
            "status": "navigated",
            "sandbox": true,
            "rights": "simulated-virtual-rights",
            "action_type": action_type,
            "selector": selector,
            "value": url,
        });
        remember_receipt(state, &receipt);
        return receipt;
    }

    let receipt = json!({
        "type": "sandbox.act",
        "status": "executed",
        "sandbox": true,
        "rights": "simulated-virtual-rights",
        "action_type": action_type,
        "selector": selector,
        "value": value,
    });
    remember_receipt(state, &receipt);
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
            | "move_window"
            | "resize_window"
            | "open_panel"
            | "close_panel"
            | "select_tab"
    )
}

fn sync_browser_address_bar_navigation(state: &mut SandboxState, index: usize, value: &str) {
    if state
        .nodes
        .get(index)
        .map(|node| node.name.to_lowercase().contains("address"))
        .unwrap_or(false)
    {
        update_browser_url(state, value);
    }
}

fn update_browser_url(state: &mut SandboxState, url: &str) {
    let target_url = if url.trim().is_empty() {
        "about:blank"
    } else {
        url.trim()
    };
    for node in &mut state.nodes {
        match node.node_id.as_str() {
            "browser-address-bar" => {
                node.value = target_url.to_string();
                node.text = target_url.to_string();
                node.metadata
                    .insert("value".to_string(), Value::String(target_url.to_string()));
            }
            "browser-main-doc" => {
                node.name = format!("Sandbox Page - {}", target_url);
                node.text = format!("Sandbox content loaded for {}", target_url);
                node.metadata
                    .insert("url".to_string(), Value::String(target_url.to_string()));
                node.metadata.insert(
                    "text".to_string(),
                    Value::String(format!("Sandbox content loaded for {}", target_url)),
                );
            }
            _ => {}
        }
    }
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

fn apply_virtual_system_action(
    state: &mut SandboxState,
    action_type: &str,
    selector: &str,
    value: Option<&str>,
    metadata: &Map<String, Value>,
) -> Option<Value> {
    if matches!(
        action_type,
        "create_file"
            | "write_file"
            | "read_file"
            | "delete_file"
            | "copy_file"
            | "move_file"
            | "rename_file"
            | "download_file"
            | "upload_file"
    ) {
        return Some(virtual_file_action(
            state,
            action_type,
            selector,
            value,
            metadata,
        ));
    }
    if matches!(
        action_type,
        "set_clipboard" | "get_clipboard" | "clipboard_copy"
    ) {
        return Some(clipboard_action(state, action_type, selector, value));
    }
    if action_type == "execute_command" {
        return Some(execute_command(
            state,
            vec![value.unwrap_or(selector).to_string()],
        ));
    }
    if matches!(action_type, "open_modal" | "close_modal") {
        return Some(modal_action(state, action_type, selector, value));
    }
    None
}

fn virtual_file_action(
    state: &mut SandboxState,
    action_type: &str,
    selector: &str,
    value: Option<&str>,
    metadata: &Map<String, Value>,
) -> Value {
    let source = metadata_text(metadata, "path")
        .or_else(|| metadata_text(metadata, "source"))
        .or(value.map(str::to_string))
        .or_else(|| (!selector.is_empty()).then(|| selector.to_string()))
        .unwrap_or_else(|| "artifacts/workflows/item.txt".to_string());
    let destination = metadata_text(metadata, "destination")
        .unwrap_or_else(|| "artifacts/workflows/copied-item.txt".to_string());
    let new_name = metadata_text(metadata, "new_name")
        .or_else(|| metadata_text(metadata, "name"))
        .unwrap_or_else(|| source.clone());
    let content = metadata_text(metadata, "content")
        .or_else(|| metadata_text(metadata, "text"))
        .or_else(|| value.filter(|item| *item != source).map(str::to_string))
        .unwrap_or_default();

    let mut operation = action_type.replace("_file", "");
    let mut read_content = Value::Null;
    match action_type {
        "create_file" | "write_file" | "upload_file" => {
            if !state.virtual_files.contains(&source) {
                state.virtual_files.push(source.clone());
            }
            state
                .virtual_file_contents
                .insert(source.clone(), Value::String(content.clone()));
        }
        "read_file" => {
            operation = "read".to_string();
            read_content = state
                .virtual_file_contents
                .get(&source)
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
        }
        "delete_file" => {
            operation = "delete".to_string();
            state.virtual_files.retain(|item| item != &source);
            state.virtual_file_contents.remove(&source);
        }
        "download_file" => {
            let download_path = if destination == source {
                format!("downloads/{source}")
            } else {
                destination.clone()
            };
            if !state.virtual_files.contains(&download_path) {
                state.virtual_files.push(download_path.clone());
            }
            let stored = state
                .virtual_file_contents
                .get(&source)
                .cloned()
                .unwrap_or_else(|| Value::String(content.clone()));
            state.virtual_file_contents.insert(download_path, stored);
        }
        "copy_file" | "move_file" | "rename_file" => {
            let file_op = mutate_virtual_files(state, action_type, metadata);
            let stored = state
                .virtual_file_contents
                .get(&source)
                .cloned()
                .unwrap_or_else(|| Value::String(content.clone()));
            if action_type == "copy_file" {
                state
                    .virtual_file_contents
                    .insert(destination.clone(), stored);
            } else if action_type == "move_file" {
                state.virtual_file_contents.remove(&source);
                state
                    .virtual_file_contents
                    .insert(destination.clone(), stored);
            } else {
                state.virtual_file_contents.remove(&source);
                state.virtual_file_contents.insert(new_name.clone(), stored);
            }
            return json!({
                "type": "sandbox.act",
                "status": "file-op-executed",
                "sandbox": true,
                "rights": "simulated-virtual-rights",
                "action_type": action_type,
                "selector": selector,
                "value": value,
                "file_op": file_op,
            });
        }
        _ => {}
    }

    json!({
        "type": "sandbox.act",
        "status": "file-op-executed",
        "sandbox": true,
        "rights": "simulated-virtual-rights",
        "action_type": action_type,
        "selector": selector,
        "value": value,
        "file_op": {
            "operation": operation.clone(),
            "source": source,
            "destination": if operation == "rename" || operation == "read" { Value::Null } else { Value::String(destination) },
            "new_name": if operation == "rename" { Value::String(new_name) } else { Value::Null },
            "content": read_content,
            "resulting_file_count": state.virtual_files.len(),
        }
    })
}

fn clipboard_action(
    state: &mut SandboxState,
    action_type: &str,
    selector: &str,
    value: Option<&str>,
) -> Value {
    if matches!(action_type, "set_clipboard" | "clipboard_copy") {
        state.clipboard = value.unwrap_or(selector).to_string();
    }
    json!({
        "type": "sandbox.act",
        "status": "clipboard-updated",
        "sandbox": true,
        "rights": "simulated-virtual-rights",
        "action_type": action_type,
        "selector": selector,
        "value": value,
        "clipboard": state.clipboard,
    })
}

fn modal_action(
    state: &mut SandboxState,
    action_type: &str,
    selector: &str,
    value: Option<&str>,
) -> Value {
    if action_type == "close_modal" {
        let closed = state.modals.pop().unwrap_or(Value::Null);
        for node in &mut state.nodes {
            if node.role.eq_ignore_ascii_case("Dialog") {
                node.enabled = false;
            }
        }
        return json!({
            "type": "sandbox.act",
            "status": "modal-closed",
            "sandbox": true,
            "rights": "simulated-virtual-rights",
            "action_type": action_type,
            "selector": selector,
            "modal": closed,
        });
    }
    let modal_id = format!("modal-{}", state.modals.len() + 1);
    let modal_name = value.unwrap_or(selector).to_string();
    let modal = json!({"modal_id": modal_id, "name": modal_name});
    state.modals.push(modal.clone());
    let mut node =
        node_record_with_bounds(&modal_id, "Dialog", &modal_name, vec![360, 220, 560, 320]);
    node.metadata
        .insert("panel_type".to_string(), Value::String("modal".to_string()));
    state.nodes.push(node);
    json!({
        "type": "sandbox.act",
        "status": "modal-opened",
        "sandbox": true,
        "rights": "simulated-virtual-rights",
        "action_type": action_type,
        "selector": selector,
        "value": value,
        "modal": modal,
    })
}

fn metadata_text(metadata: &Map<String, Value>, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
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

fn mutate_window_panel(
    state: &mut SandboxState,
    index: usize,
    action_type: &str,
    value: Option<&str>,
    metadata: &Map<String, Value>,
) -> Value {
    match action_type {
        "move_window" | "resize_window" => {
            if let Some(bounds) = metadata.get("bounds").and_then(Value::as_array) {
                if bounds.len() == 4 {
                    if let Some(node) = state.nodes.get_mut(index) {
                        node.bounds = bounds
                            .iter()
                            .filter_map(Value::as_i64)
                            .map(|item| item as i32)
                            .collect();
                        if node.bounds.len() != 4 {
                            node.bounds = empty_bounds();
                        }
                    }
                }
            }
            json!({"status": "window-updated", "bounds": state.nodes[index].bounds.clone()})
        }
        "select_tab" => {
            let tab = metadata
                .get("tab")
                .and_then(Value::as_str)
                .or(value)
                .unwrap_or("Tab 1")
                .to_string();
            if let Some(node) = state.nodes.get_mut(index) {
                node.metadata
                    .insert("selected_tab".to_string(), Value::String(tab.clone()));
            }
            json!({"status": "tab-selected", "tab": tab})
        }
        "close_panel" => {
            if let Some(node) = state.nodes.get_mut(index) {
                node.enabled = false;
                node.metadata
                    .insert("visible".to_string(), Value::Bool(false));
            }
            json!({"status": "panel-closed"})
        }
        "open_panel" => {
            let panel_name = metadata
                .get("panel_name")
                .and_then(Value::as_str)
                .or(value)
                .unwrap_or("Agent Panel")
                .to_string();
            let panel_id = metadata
                .get("panel_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("panel-{}", state.nodes.len() + 1));
            let role = metadata
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("Pane");
            let bounds = metadata
                .get("bounds")
                .and_then(Value::as_array)
                .filter(|items| items.len() == 4)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_i64)
                        .map(|item| item as i32)
                        .collect::<Vec<i32>>()
                })
                .filter(|items| items.len() == 4)
                .unwrap_or_else(|| vec![220, 180, 420, 560]);
            let mut panel = node_record_with_bounds(&panel_id, role, &panel_name, bounds);
            panel.metadata.insert(
                "parent".to_string(),
                Value::String(state.nodes[index].node_id.clone()),
            );
            panel.metadata.insert(
                "panel_type".to_string(),
                Value::String(
                    metadata
                        .get("panel_type")
                        .and_then(Value::as_str)
                        .unwrap_or("dynamic")
                        .to_string(),
                ),
            );
            state.nodes.push(panel);
            json!({"status": "panel-opened", "panel_id": panel_id})
        }
        _ => json!({}),
    }
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
    let support_panels = support_panel_nodes(&surface_spec, &window_id);
    let payload = json!({
        "family": surface_spec.family,
        "selector": surface_spec.node_id,
        "role": surface_spec.role,
        "name": surface_spec.name,
        "panel_count": support_panels.len() + 1,
    });
    state.nodes.push(surface);
    state.nodes.extend(support_panels);
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
        let doc = find_node_mut(&mut state, "browser-main-doc").unwrap();
        assert_eq!(doc.name, "Sandbox Page - https://example.com");
        assert_eq!(
            doc.metadata.get("url").and_then(Value::as_str),
            Some("https://example.com")
        );
    }

    #[test]
    fn navigate_action_updates_browser_document() {
        let mut state = default_state();
        let payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "navigate".to_string(),
                selector: "https://example.org/report".to_string(),
                value: None,
                metadata: None,
            },
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("navigated")
        );
        let doc = find_node_mut(&mut state, "browser-main-doc").unwrap();
        assert_eq!(doc.name, "Sandbox Page - https://example.org/report");
        assert_eq!(
            doc.metadata.get("url").and_then(Value::as_str),
            Some("https://example.org/report")
        );
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
    fn capabilities_report_simulated_virtual_sandbox() {
        let mut state = default_state();
        let payload = handle_command(&mut state, CommandEnvelope::Capabilities);
        assert_eq!(payload.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(
            payload.get("rights").and_then(Value::as_str),
            Some("simulated-virtual-rights")
        );
        assert_eq!(
            payload.get("is_simulated").and_then(Value::as_bool),
            Some(true)
        );
        let families = payload.get("families").and_then(Value::as_array).unwrap();
        assert!(families.iter().any(|item| item == "trading_terminal"));
    }

    #[test]
    fn native_unsupported_action_reports_without_side_effects() {
        let mut state = default_state();
        let payload = handle_command(
            &mut state,
            CommandEnvelope::NativeAct {
                action_type: "teleport".to_string(),
                selector: "native-desktop".to_string(),
                value: None,
                metadata: None,
            },
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("unsupported-action")
        );
        assert_eq!(payload.get("native").and_then(Value::as_bool), Some(true));
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
        assert!(find_node_index(&state, "market-watchlist").is_some());
        assert!(find_node_index(&state, "positions-grid").is_some());
    }

    #[test]
    fn reset_preserves_agent_history() {
        let mut state = default_state();
        handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "focus".to_string(),
                selector: "browser-address-bar".to_string(),
                value: None,
                metadata: None,
            },
        );
        assert_eq!(state.agent_history.len(), 1);
        handle_command(&mut state, CommandEnvelope::Reset);
        assert_eq!(state.agent_history.len(), 1);
        let snapshot = handle_command(&mut state, CommandEnvelope::Snapshot);
        assert_eq!(
            snapshot.get("agent_history_count").and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn virtual_system_actions_are_confined_and_stateful() {
        let mut state = default_state();
        let mut metadata = Map::new();
        metadata.insert(
            "path".to_string(),
            Value::String("artifacts/workflows/new.txt".to_string()),
        );
        metadata.insert(
            "content".to_string(),
            Value::String("hello sandbox".to_string()),
        );
        let write_payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "write_file".to_string(),
                selector: "".to_string(),
                value: None,
                metadata: Some(metadata),
            },
        );
        assert_eq!(
            write_payload.get("status").and_then(Value::as_str),
            Some("file-op-executed")
        );
        assert!(state
            .virtual_file_contents
            .contains_key("artifacts/workflows/new.txt"));

        let clip_payload = handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "set_clipboard".to_string(),
                selector: "".to_string(),
                value: Some("copied text".to_string()),
                metadata: None,
            },
        );
        assert_eq!(
            clip_payload.get("clipboard").and_then(Value::as_str),
            Some("copied text")
        );

        handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "execute_command".to_string(),
                selector: "python -V".to_string(),
                value: None,
                metadata: None,
            },
        );
        assert_eq!(state.virtual_processes.len(), 1);

        handle_command(
            &mut state,
            CommandEnvelope::Act {
                action_type: "open_modal".to_string(),
                selector: "Sandbox Dialog".to_string(),
                value: None,
                metadata: None,
            },
        );
        assert_eq!(state.modals.len(), 1);
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
