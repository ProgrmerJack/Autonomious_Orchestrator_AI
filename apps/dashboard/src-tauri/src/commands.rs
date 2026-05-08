use tauri::{async_runtime, State, WebviewWindow};

use crate::bridge::{DaemonRecord, DesktopBootstrap, DesktopBridge, DesktopBridgeError};

#[tauri::command]
pub async fn desktop_bootstrap(
    state: State<'_, DesktopBridge>,
    webview_window: WebviewWindow,
) -> Result<DesktopBootstrap, DesktopBridgeError> {
    let bridge = state.inner().clone();
    let label = webview_window.label().to_string();
    async_runtime::spawn_blocking(move || bridge.bootstrap(&label))
        .await
        .map_err(|error| {
            DesktopBridgeError::internal(format!(
                "desktop bootstrap task failed to join: {error}"
            ))
        })?
}

#[tauri::command]
pub async fn desktop_daemon_status(
    ipc_token: String,
    state: State<'_, DesktopBridge>,
    webview_window: WebviewWindow,
) -> Result<DaemonRecord, DesktopBridgeError> {
    let bridge = state.inner().clone();
    let label = webview_window.label().to_string();
    async_runtime::spawn_blocking(move || bridge.daemon_status(&label, &ipc_token))
        .await
        .map_err(|error| {
            DesktopBridgeError::internal(format!(
                "desktop daemon status task failed to join: {error}"
            ))
        })?
}

#[tauri::command]
pub async fn desktop_daemon_control(
    action: String,
    ipc_token: String,
    state: State<'_, DesktopBridge>,
    webview_window: WebviewWindow,
) -> Result<DaemonRecord, DesktopBridgeError> {
    let bridge = state.inner().clone();
    let label = webview_window.label().to_string();
    async_runtime::spawn_blocking(move || bridge.daemon_control(&label, &ipc_token, &action))
        .await
        .map_err(|error| {
            DesktopBridgeError::internal(format!(
                "desktop daemon control task failed to join: {error}"
            ))
        })?
}