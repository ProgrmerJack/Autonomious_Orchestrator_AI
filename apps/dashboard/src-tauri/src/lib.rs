mod bridge;
mod commands;

use bridge::DesktopBridge;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(DesktopBridge::new())
        .invoke_handler(tauri::generate_handler![
            commands::desktop_bootstrap,
            commands::desktop_daemon_status,
            commands::desktop_daemon_control,
        ])
        .run(tauri::generate_context!())
        .expect("failed to run AgentOS dashboard");
}
