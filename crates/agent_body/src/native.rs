use serde_json::{json, Map, Value};

#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    pub(crate) fn snapshot() -> Value {
        match Enigo::new(&Settings::default()) {
            Ok(enigo) => {
                let display = enigo.main_display().unwrap_or((0, 0));
                let cursor = enigo.location().unwrap_or((0, 0));
                let mut nodes = vec![
                    json!({
                        "node_id": "native-desktop",
                        "role": "Desktop",
                        "name": "Windows Desktop",
                        "focused": true,
                        "enabled": true,
                        "bounds": [0, 0, display.0, display.1],
                        "metadata": {
                            "native": true,
                            "backend": "rust-native-windows",
                            "control_channel": "native-input"
                        }
                    }),
                    json!({
                        "node_id": "native-cursor",
                        "role": "Pointer",
                        "name": "Mouse Cursor",
                        "focused": false,
                        "enabled": true,
                        "bounds": [cursor.0, cursor.1, 1, 1],
                        "metadata": {
                            "native": true,
                            "x": cursor.0,
                            "y": cursor.1
                        }
                    }),
                ];
                let mut uia_status = "disabled";
                #[cfg(feature = "uia-windows")]
                {
                    match collect_uia_nodes(256) {
                        Ok(extra) => {
                            uia_status = "ok";
                            nodes.extend(extra);
                        }
                        Err(error) => {
                            uia_status = "error";
                            nodes.push(json!({
                                "node_id": "native-uia-error",
                                "role": "Error",
                                "name": "UIA enumeration failed",
                                "focused": false,
                                "enabled": false,
                                "bounds": [0, 0, 0, 0],
                                "metadata": {
                                    "native": true,
                                    "error": error
                                }
                            }));
                        }
                    }
                }
                json!({
                    "type": "native.snapshot",
                    "status": "ok",
                    "backend": "rust-native-windows",
                    "native": true,
                    "platform": "windows",
                    "uia": uia_status,
                    "screen": {
                        "x": 0,
                        "y": 0,
                        "width": display.0,
                        "height": display.1,
                    },
                    "cursor": {"x": cursor.0, "y": cursor.1},
                    "nodes": nodes
                })
            }
            Err(error) => unavailable(format!("native input unavailable: {error}")),
        }
    }

    #[cfg(feature = "uia-windows")]
    fn collect_uia_nodes(limit: usize) -> Result<Vec<Value>, String> {
        use uiautomation::types::TreeScope;
        use uiautomation::UIAutomation;

        let automation = UIAutomation::new().map_err(|e| e.to_string())?;
        let root = automation.get_root_element().map_err(|e| e.to_string())?;
        let condition = automation
            .create_true_condition()
            .map_err(|e| e.to_string())?;
        let walker = automation
            .create_tree_walker_with_condition(&condition)
            .map_err(|e| e.to_string())?;
        let mut nodes = Vec::new();
        // Enumerate top-level windows
        let windows = root
            .find_all(TreeScope::Children, &condition)
            .map_err(|e| e.to_string())?;
        for window in windows {
            if nodes.len() >= limit {
                break;
            }
            walk(&walker, &window, 0, &mut nodes, limit);
        }
        Ok(nodes)
    }

    #[cfg(feature = "uia-windows")]
    fn walk(
        walker: &uiautomation::UITreeWalker,
        element: &uiautomation::UIElement,
        depth: usize,
        nodes: &mut Vec<Value>,
        limit: usize,
    ) {
        if nodes.len() >= limit || depth > 5 {
            return;
        }
        let name = element.get_name().unwrap_or_default();
        let role = element
            .get_control_type()
            .map(|c| format!("{:?}", c))
            .unwrap_or_default();
        let automation_id = element.get_automation_id().unwrap_or_default();
        let rect = element.get_bounding_rectangle().ok();
        let bounds = rect
            .map(|r| json!([r.get_left(), r.get_top(), r.get_width(), r.get_height()]))
            .unwrap_or(json!([0, 0, 0, 0]));
        let enabled = element.is_enabled().unwrap_or(false);
        let focused = element.has_keyboard_focus().unwrap_or(false);
        let node_id = if automation_id.is_empty() {
            format!("uia-{}-{}", depth, nodes.len())
        } else {
            format!("uia-{}", automation_id)
        };
        nodes.push(json!({
            "node_id": node_id,
            "role": role,
            "name": name,
            "focused": focused,
            "enabled": enabled,
            "bounds": bounds,
            "metadata": {
                "native": true,
                "backend": "rust-native-windows-uia",
                "automation_id": automation_id,
                "depth": depth
            }
        }));
        if let Ok(first_child) = walker.get_first_child(element) {
            let mut current = first_child;
            loop {
                if nodes.len() >= limit {
                    break;
                }
                walk(walker, &current, depth + 1, nodes, limit);
                match walker.get_next_sibling(&current) {
                    Ok(sibling) => current = sibling,
                    Err(_) => break,
                }
            }
        }
    }

    pub(crate) fn apply_action(
        action_type: &str,
        selector: &str,
        value: Option<&str>,
        metadata: Option<Map<String, Value>>,
    ) -> Value {
        let metadata = metadata.unwrap_or_default();
        match action_type {
            "launch_app" => launch_app(selector, value),
            "open_url" => open_url(selector, value),
            "wait" => wait(value),
            "hotkey" => with_enigo(action_type, selector, value, |enigo| {
                send_hotkey(enigo, value.unwrap_or(selector))?;
                Ok(json!({"status": "hotkey-sent"}))
            }),
            "move_cursor" => with_enigo(action_type, selector, value, |enigo| {
                let (x, y) = coordinate(&metadata, selector, value)
                    .ok_or_else(|| "move_cursor requires x/y coordinates".to_string())?;
                enigo
                    .move_mouse(x, y, Coordinate::Abs)
                    .map_err(|error| error.to_string())?;
                Ok(json!({"status": "cursor-moved", "x": x, "y": y}))
            }),
            "click" | "invoke" => with_enigo(action_type, selector, value, |enigo| {
                let (x, y) = coordinate(&metadata, selector, value)
                    .ok_or_else(|| "click requires x/y coordinates".to_string())?;
                enigo
                    .move_mouse(x, y, Coordinate::Abs)
                    .map_err(|error| error.to_string())?;
                enigo
                    .button(Button::Left, Direction::Click)
                    .map_err(|error| error.to_string())?;
                Ok(json!({"status": "clicked", "x": x, "y": y}))
            }),
            "type" | "set_text" | "set_value" => {
                with_enigo(action_type, selector, value, |enigo| {
                    if let Some((x, y)) = coordinate(&metadata, selector, value) {
                        enigo
                            .move_mouse(x, y, Coordinate::Abs)
                            .map_err(|error| error.to_string())?;
                        enigo
                            .button(Button::Left, Direction::Click)
                            .map_err(|error| error.to_string())?;
                    }
                    enigo
                        .text(value.unwrap_or(""))
                        .map_err(|error| error.to_string())?;
                    Ok(json!({"status": "typed"}))
                })
            }
            "scroll" => with_enigo(action_type, selector, value, |enigo| {
                let amount = numeric_value(value)
                    .or_else(|| metadata_i64(&metadata, "amount"))
                    .unwrap_or(0) as i32;
                enigo
                    .scroll(amount, Axis::Vertical)
                    .map_err(|error| error.to_string())?;
                Ok(json!({"status": "scrolled", "amount": amount}))
            }),
            "draw_path" => with_enigo(action_type, selector, value, |enigo| {
                let points = draw_points(value.unwrap_or(""), &metadata)?;
                let Some((first_x, first_y)) = points.first().copied() else {
                    return Err("draw_path requires at least two points".to_string());
                };
                enigo
                    .move_mouse(first_x, first_y, Coordinate::Abs)
                    .map_err(|error| error.to_string())?;
                enigo
                    .button(Button::Left, Direction::Press)
                    .map_err(|error| error.to_string())?;
                for (x, y) in points.iter().copied().skip(1) {
                    enigo
                        .move_mouse(x, y, Coordinate::Abs)
                        .map_err(|error| error.to_string())?;
                }
                enigo
                    .button(Button::Left, Direction::Release)
                    .map_err(|error| error.to_string())?;
                Ok(json!({"status": "drawn", "points": points.len()}))
            }),
            _ => json!({
                "type": "native.act",
                "status": "unsupported-action",
                "native": true,
                "platform": "windows",
                "action_type": action_type,
                "selector": selector,
                "value": value,
            }),
        }
    }

    fn with_enigo<F>(action_type: &str, selector: &str, value: Option<&str>, action: F) -> Value
    where
        F: FnOnce(&mut Enigo) -> Result<Value, String>,
    {
        let mut base = match Enigo::new(&Settings::default()) {
            Ok(mut enigo) => match action(&mut enigo) {
                Ok(payload) => payload,
                Err(error) => json!({"status": "error", "error": error}),
            },
            Err(error) => json!({
                "status": "unavailable",
                "error": format!("native input unavailable: {error}"),
            }),
        };
        decorate(&mut base, action_type, selector, value);
        base
    }

    fn launch_app(selector: &str, value: Option<&str>) -> Value {
        let target = value.unwrap_or(selector).trim();
        if target.is_empty() {
            return decorate_new(
                "error",
                "launch_app",
                selector,
                value,
                json!({"error": "launch_app requires a target"}),
            );
        }
        match Command::new(target).spawn() {
            Ok(process) => decorate_new(
                "launched",
                "launch_app",
                selector,
                value,
                json!({"launched": target, "process_id": process.id()}),
            ),
            Err(first_error) => match Command::new("cmd")
                .args(["/C", "start", "", target])
                .spawn()
            {
                Ok(process) => decorate_new(
                    "launched",
                    "launch_app",
                    selector,
                    value,
                    json!({
                        "launched": target,
                        "process_id": process.id(),
                        "shell_start": true
                    }),
                ),
                Err(second_error) => decorate_new(
                    "error",
                    "launch_app",
                    selector,
                    value,
                    json!({
                        "error": format!(
                            "launch failed: {first_error}; shell start failed: {second_error}"
                        )
                    }),
                ),
            },
        }
    }

    fn open_url(selector: &str, value: Option<&str>) -> Value {
        let target = value.unwrap_or(selector).trim();
        if target.is_empty() {
            return decorate_new(
                "error",
                "open_url",
                selector,
                value,
                json!({"error": "open_url requires a URL"}),
            );
        }
        match Command::new("cmd")
            .args(["/C", "start", "", target])
            .spawn()
        {
            Ok(process) => decorate_new(
                "launched",
                "open_url",
                selector,
                value,
                json!({"url": target, "process_id": process.id()}),
            ),
            Err(error) => decorate_new(
                "error",
                "open_url",
                selector,
                value,
                json!({"error": format!("open_url failed: {error}")}),
            ),
        }
    }

    fn wait(value: Option<&str>) -> Value {
        let millis = numeric_value(value).unwrap_or(250).max(0) as u64;
        thread::sleep(Duration::from_millis(millis));
        decorate_new("waited", "wait", "", value, json!({"milliseconds": millis}))
    }

    fn decorate_new(
        status: &str,
        action_type: &str,
        selector: &str,
        value: Option<&str>,
        extra: Value,
    ) -> Value {
        let mut payload = extra;
        if !payload.is_object() {
            payload = json!({"detail": payload});
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("status".to_string(), Value::String(status.to_string()));
        }
        decorate(&mut payload, action_type, selector, value);
        payload
    }

    fn decorate(payload: &mut Value, action_type: &str, selector: &str, value: Option<&str>) {
        if let Some(map) = payload.as_object_mut() {
            map.insert("type".to_string(), Value::String("native.act".to_string()));
            map.insert("native".to_string(), Value::Bool(true));
            map.insert("platform".to_string(), Value::String("windows".to_string()));
            map.insert(
                "backend".to_string(),
                Value::String("rust-native-windows".to_string()),
            );
            map.insert(
                "action_type".to_string(),
                Value::String(action_type.to_string()),
            );
            map.insert("selector".to_string(), Value::String(selector.to_string()));
            map.insert(
                "value".to_string(),
                value.map(Value::from).unwrap_or(Value::Null),
            );
        }
    }

    fn coordinate(
        metadata: &Map<String, Value>,
        selector: &str,
        value: Option<&str>,
    ) -> Option<(i32, i32)> {
        let direct = metadata_i64(metadata, "x")
            .zip(metadata_i64(metadata, "y"))
            .map(|(x, y)| (x as i32, y as i32));
        if direct.is_some() {
            return direct;
        }
        if let Some(bounds) = metadata.get("bbox").or_else(|| metadata.get("bounds")) {
            if let Some(items) = bounds.as_array() {
                if items.len() >= 4 {
                    let x = items[0].as_i64()? + (items[2].as_i64()? / 2);
                    let y = items[1].as_i64()? + (items[3].as_i64()? / 2);
                    return Some((x as i32, y as i32));
                }
            }
        }
        parse_point(selector).or_else(|| value.and_then(parse_point))
    }

    fn parse_point(raw: &str) -> Option<(i32, i32)> {
        let cleaned = raw
            .trim()
            .trim_start_matches("point=")
            .trim_start_matches("coords=");
        let mut parts = cleaned.split(',').map(str::trim);
        let x = parts.next()?.parse::<i32>().ok()?;
        let y = parts.next()?.parse::<i32>().ok()?;
        Some((x, y))
    }

    fn metadata_i64(metadata: &Map<String, Value>, key: &str) -> Option<i64> {
        metadata.get(key).and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_f64().map(|item| item.round() as i64))
                .or_else(|| value.as_str()?.parse::<i64>().ok())
        })
    }

    fn numeric_value(value: Option<&str>) -> Option<i64> {
        value?.trim().parse::<i64>().ok()
    }

    fn draw_points(raw: &str, metadata: &Map<String, Value>) -> Result<Vec<(i32, i32)>, String> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| format!("draw_path must be JSON points: {error}"))?;
        let points = if let Some(items) = value.get("points").and_then(Value::as_array) {
            items
        } else if let Some(items) = value.as_array() {
            items
        } else {
            return Err("draw_path requires an array or {points: [...]}".to_string());
        };
        let bounds = metadata_bounds(metadata);
        let mut resolved = Vec::new();
        for point in points {
            if let Some(items) = point.as_array() {
                if items.len() >= 2 {
                    let raw_x = items[0].as_f64().unwrap_or(0.0);
                    let raw_y = items[1].as_f64().unwrap_or(0.0);
                    let (x, y) = resolve_draw_point(raw_x, raw_y, bounds);
                    resolved.push((x, y));
                    continue;
                }
            }
            if let Some(map) = point.as_object() {
                let raw_x = map.get("x").and_then(Value::as_f64).unwrap_or(0.0);
                let raw_y = map.get("y").and_then(Value::as_f64).unwrap_or(0.0);
                let (x, y) = resolve_draw_point(raw_x, raw_y, bounds);
                resolved.push((x, y));
                continue;
            }
            return Err("draw_path contains an invalid point".to_string());
        }
        if resolved.len() < 2 {
            return Err("draw_path requires at least two points".to_string());
        }
        Ok(resolved)
    }

    fn metadata_bounds(metadata: &Map<String, Value>) -> Option<(i32, i32, i32, i32)> {
        let bounds = metadata.get("bbox").or_else(|| metadata.get("bounds"))?;
        let items = bounds.as_array()?;
        if items.len() < 4 {
            return None;
        }
        Some((
            items[0].as_i64()? as i32,
            items[1].as_i64()? as i32,
            items[2].as_i64()? as i32,
            items[3].as_i64()? as i32,
        ))
    }

    fn resolve_draw_point(
        raw_x: f64,
        raw_y: f64,
        bounds: Option<(i32, i32, i32, i32)>,
    ) -> (i32, i32) {
        if let Some((left, top, width, height)) = bounds {
            let x = if raw_x.abs() <= 1.0 {
                left + (raw_x * f64::from(width)).round() as i32
            } else {
                raw_x.round() as i32
            };
            let y = if raw_y.abs() <= 1.0 {
                top + (raw_y * f64::from(height)).round() as i32
            } else {
                raw_y.round() as i32
            };
            return (x, y);
        }
        (raw_x.round() as i32, raw_y.round() as i32)
    }

    fn send_hotkey(enigo: &mut Enigo, raw: &str) -> Result<(), String> {
        let keys: Vec<Key> = raw
            .split('+')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(key_from_token)
            .collect::<Result<Vec<_>, _>>()?;
        let Some((last, modifiers)) = keys.split_last() else {
            return Err("hotkey requires at least one key".to_string());
        };
        for key in modifiers {
            enigo
                .key(*key, Direction::Press)
                .map_err(|error| error.to_string())?;
        }
        enigo
            .key(*last, Direction::Click)
            .map_err(|error| error.to_string())?;
        for key in modifiers.iter().rev() {
            enigo
                .key(*key, Direction::Release)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn key_from_token(token: &str) -> Result<Key, String> {
        let normalized = token
            .trim()
            .trim_matches('{')
            .trim_matches('}')
            .to_ascii_lowercase();
        let key = match normalized.as_str() {
            "ctrl" | "control" => Key::Control,
            "alt" | "option" => Key::Alt,
            "shift" => Key::Shift,
            "meta" | "win" | "windows" | "super" => Key::Meta,
            "enter" | "return" => Key::Return,
            "tab" => Key::Tab,
            "esc" | "escape" => Key::Escape,
            "space" => Key::Space,
            "backspace" => Key::Backspace,
            "delete" | "del" => Key::Delete,
            "up" | "uparrow" => Key::UpArrow,
            "down" | "downarrow" => Key::DownArrow,
            "left" | "leftarrow" => Key::LeftArrow,
            "right" | "rightarrow" => Key::RightArrow,
            "home" => Key::Home,
            "end" => Key::End,
            "pageup" => Key::PageUp,
            "pagedown" => Key::PageDown,
            "insert" | "ins" => Key::Insert,
            value if value.starts_with('f') => {
                return function_key(value)
                    .ok_or_else(|| format!("unsupported function key token: {token}"))
            }
            value if value.chars().count() == 1 => Key::Unicode(value.chars().next().unwrap()),
            _ => return Err(format!("unsupported hotkey token: {token}")),
        };
        Ok(key)
    }

    fn function_key(value: &str) -> Option<Key> {
        match value.get(1..)?.parse::<u8>().ok()? {
            1 => Some(Key::F1),
            2 => Some(Key::F2),
            3 => Some(Key::F3),
            4 => Some(Key::F4),
            5 => Some(Key::F5),
            6 => Some(Key::F6),
            7 => Some(Key::F7),
            8 => Some(Key::F8),
            9 => Some(Key::F9),
            10 => Some(Key::F10),
            11 => Some(Key::F11),
            12 => Some(Key::F12),
            13 => Some(Key::F13),
            14 => Some(Key::F14),
            15 => Some(Key::F15),
            16 => Some(Key::F16),
            17 => Some(Key::F17),
            18 => Some(Key::F18),
            19 => Some(Key::F19),
            20 => Some(Key::F20),
            21 => Some(Key::F21),
            22 => Some(Key::F22),
            23 => Some(Key::F23),
            24 => Some(Key::F24),
            _ => None,
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;

    pub(crate) fn snapshot() -> Value {
        unavailable("native Windows control is only available on Windows")
    }

    pub(crate) fn apply_action(
        action_type: &str,
        selector: &str,
        value: Option<&str>,
        _metadata: Option<Map<String, Value>>,
    ) -> Value {
        json!({
            "type": "native.act",
            "status": "unavailable",
            "native": true,
            "platform": std::env::consts::OS,
            "action_type": action_type,
            "selector": selector,
            "value": value,
            "error": "native Windows control is only available on Windows",
        })
    }
}

pub(crate) fn snapshot() -> Value {
    platform::snapshot()
}

pub(crate) fn apply_action(
    action_type: &str,
    selector: &str,
    value: Option<&str>,
    metadata: Option<Map<String, Value>>,
) -> Value {
    platform::apply_action(action_type, selector, value, metadata)
}

fn unavailable(reason: impl Into<String>) -> Value {
    json!({
        "type": "native.snapshot",
        "status": "unavailable",
        "native": true,
        "platform": std::env::consts::OS,
        "error": reason.into(),
        "nodes": [],
    })
}
