use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Clone)]
pub struct DesktopBridge {
    inner: Arc<DesktopBridgeInner>,
}

struct DesktopBridgeInner {
    workspace_root: PathBuf,
    log_path: PathBuf,
    runtime: Mutex<BridgeRuntime>,
}

struct BridgeRuntime {
    ipc_token: String,
    bootstrap_token: String,
    backend: Option<ManagedBackend>,
    last_api_url: String,
}

struct ManagedBackend {
    child: Child,
    api_url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DesktopBridgeError {
    pub kind: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonRecord {
    pub status: String,
    pub launcher_pid: Option<u32>,
    pub api_url: String,
    pub ui_url: String,
    pub log_path: String,
    pub started_at: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DashboardSession {
    pub session_token: String,
    pub csrf_token: String,
    pub issued_at: String,
    pub expires_at: String,
    pub unsafe_ack_value: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DesktopBootstrap {
    pub api_base: String,
    pub session: DashboardSession,
    pub daemon: DaemonRecord,
    pub ipc_token: String,
}

struct PythonLauncher {
    program: String,
    prefix_args: Vec<String>,
}

impl DesktopBridge {
    pub fn new() -> Self {
        let workspace_root = workspace_root();
        let log_path = workspace_root
            .join(".agentos")
            .join("logs")
            .join("desktop-dashboard.log");
        Self {
            inner: Arc::new(DesktopBridgeInner {
                workspace_root,
                log_path,
                runtime: Mutex::new(BridgeRuntime {
                    ipc_token: new_token(),
                    bootstrap_token: new_token(),
                    backend: None,
                    last_api_url: String::new(),
                }),
            }),
        }
    }

    pub fn bootstrap(&self, window_label: &str) -> Result<DesktopBootstrap, DesktopBridgeError> {
        self.ensure_main_window(window_label)?;
        let (api_base, daemon, bootstrap_token, ipc_token) = {
            let mut runtime = self.runtime_lock()?;
            self.refresh_backend_state(&mut runtime)?;
            if runtime.backend.is_none() {
                let backend = spawn_backend(
                    &self.inner.workspace_root,
                    &self.inner.log_path,
                    &runtime.bootstrap_token,
                )?;
                runtime.last_api_url = backend.api_url.clone();
                runtime.backend = Some(backend);
            }
            (
                runtime
                    .backend
                    .as_ref()
                    .map(|backend| backend.api_url.clone())
                    .unwrap_or_else(|| runtime.last_api_url.clone()),
                daemon_record(&runtime, &self.inner.log_path, "managed by desktop bridge"),
                runtime.bootstrap_token.clone(),
                runtime.ipc_token.clone(),
            )
        };
        let session = create_dashboard_session(&api_base, &bootstrap_token)?;
        Ok(DesktopBootstrap {
            api_base,
            session,
            daemon,
            ipc_token,
        })
    }

    pub fn daemon_status(
        &self,
        window_label: &str,
        ipc_token: &str,
    ) -> Result<DaemonRecord, DesktopBridgeError> {
        self.authorize_ipc(window_label, ipc_token)?;
        let mut runtime = self.runtime_lock()?;
        self.refresh_backend_state(&mut runtime)?;
        Ok(daemon_record(
            &runtime,
            &self.inner.log_path,
            "managed by desktop bridge",
        ))
    }

    pub fn daemon_control(
        &self,
        window_label: &str,
        ipc_token: &str,
        action: &str,
    ) -> Result<DaemonRecord, DesktopBridgeError> {
        self.authorize_ipc(window_label, ipc_token)?;
        let mut runtime = self.runtime_lock()?;
        self.refresh_backend_state(&mut runtime)?;
        match action {
            "start" => {
                if runtime.backend.is_none() {
                    let backend = spawn_backend(
                        &self.inner.workspace_root,
                        &self.inner.log_path,
                        &runtime.bootstrap_token,
                    )?;
                    runtime.last_api_url = backend.api_url.clone();
                    runtime.backend = Some(backend);
                }
            }
            "stop" => stop_backend(&mut runtime)?,
            "restart" => {
                stop_backend(&mut runtime)?;
                let backend = spawn_backend(
                    &self.inner.workspace_root,
                    &self.inner.log_path,
                    &runtime.bootstrap_token,
                )?;
                runtime.last_api_url = backend.api_url.clone();
                runtime.backend = Some(backend);
            }
            other => {
                return Err(DesktopBridgeError::invalid_input(format!(
                    "unsupported daemon action: {other}"
                )))
            }
        }
        Ok(daemon_record(
            &runtime,
            &self.inner.log_path,
            "managed by desktop bridge",
        ))
    }

    fn ensure_main_window(&self, window_label: &str) -> Result<(), DesktopBridgeError> {
        if window_label == "main" {
            return Ok(());
        }
        Err(DesktopBridgeError::permission(
            "desktop bridge commands are limited to the main window",
        ))
    }

    fn authorize_ipc(
        &self,
        window_label: &str,
        ipc_token: &str,
    ) -> Result<(), DesktopBridgeError> {
        self.ensure_main_window(window_label)?;
        let runtime = self.runtime_lock()?;
        if runtime.ipc_token == ipc_token {
            return Ok(());
        }
        Err(DesktopBridgeError::permission(
            "desktop bridge IPC token is invalid",
        ))
    }

    fn refresh_backend_state(
        &self,
        runtime: &mut BridgeRuntime,
    ) -> Result<(), DesktopBridgeError> {
        let Some(backend) = runtime.backend.as_mut() else {
            return Ok(());
        };
        match backend.child.try_wait() {
            Ok(Some(_status)) => {
                runtime.backend = None;
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) => Err(DesktopBridgeError::internal(format!(
                "failed to inspect dashboard backend process: {error}"
            ))),
        }
    }

    fn runtime_lock(&self) -> Result<MutexGuard<'_, BridgeRuntime>, DesktopBridgeError> {
        self.inner
            .runtime
            .lock()
            .map_err(|_| DesktopBridgeError::internal("desktop bridge state is poisoned"))
    }
}

impl Drop for DesktopBridgeInner {
    fn drop(&mut self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            let _ = stop_backend(&mut runtime);
        }
    }
}

impl DesktopBridgeError {
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: "internal",
            message: message.into(),
        }
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            kind: "invalid_input",
            message: message.into(),
        }
    }

    pub(crate) fn permission(message: impl Into<String>) -> Self {
        Self {
            kind: "permission_denied",
            message: message.into(),
        }
    }

    pub(crate) fn startup(message: impl Into<String>) -> Self {
        Self {
            kind: "startup_failed",
            message: message.into(),
        }
    }
}

fn daemon_record(runtime: &BridgeRuntime, log_path: &Path, detail: &str) -> DaemonRecord {
    if let Some(backend) = runtime.backend.as_ref() {
        return DaemonRecord {
            status: "running".to_string(),
            launcher_pid: Some(backend.child.id()),
            api_url: backend.api_url.clone(),
            ui_url: "tauri://localhost/".to_string(),
            log_path: log_path.display().to_string(),
            started_at: None,
            detail: detail.to_string(),
        };
    }
    DaemonRecord {
        status: "stopped".to_string(),
        launcher_pid: None,
        api_url: runtime.last_api_url.clone(),
        ui_url: "tauri://localhost/".to_string(),
        log_path: log_path.display().to_string(),
        started_at: None,
        detail: "desktop bridge backend is stopped".to_string(),
    }
}

fn create_dashboard_session(
    api_base: &str,
    bootstrap_token: &str,
) -> Result<DashboardSession, DesktopBridgeError> {
    let endpoint = format!("{api_base}/auth/session");
    let deadline = Instant::now() + Duration::from_secs(30);
    let payload = json!({ "bootstrap_token": bootstrap_token });
    loop {
        match ureq::post(&endpoint)
            .set("Content-Type", "application/json")
            .send_json(payload.clone())
        {
            Ok(response) => {
                return response.into_json::<DashboardSession>().map_err(|error| {
                    DesktopBridgeError::startup(format!(
                        "dashboard session response was invalid: {error}"
                    ))
                })
            }
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().unwrap_or_default();
                if Instant::now() >= deadline || matches!(status, 400 | 401 | 403) {
                    return Err(DesktopBridgeError::startup(format!(
                        "dashboard session bootstrap failed with status {status}: {body}"
                    )));
                }
            }
            Err(ureq::Error::Transport(error)) => {
                if Instant::now() >= deadline {
                    return Err(DesktopBridgeError::startup(format!(
                        "dashboard backend did not become ready: {error}"
                    )));
                }
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn spawn_backend(
    workspace_root: &Path,
    log_path: &Path,
    bootstrap_token: &str,
) -> Result<ManagedBackend, DesktopBridgeError> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            DesktopBridgeError::startup(format!("failed to create desktop log directory: {error}"))
        })?;
    }

    let api_port = reserve_loopback_port()?;
    let api_url = format!("http://127.0.0.1:{api_port}");
    let launchers = python_launchers(workspace_root);
    let mut last_error = String::new();

    for launcher in launchers {
        let stdout_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .map_err(|error| {
                DesktopBridgeError::startup(format!("failed to open desktop log file: {error}"))
            })?;
        let stderr_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .map_err(|error| {
                DesktopBridgeError::startup(format!("failed to open desktop log file: {error}"))
            })?;
        let mut command = Command::new(&launcher.program);
        command
            .args(&launcher.prefix_args)
            .arg("-m")
            .arg("agentos_orchestrator")
            .arg("serve-dashboard")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(api_port.to_string())
            .current_dir(workspace_root)
            .env("AGENTOS_DASHBOARD_BOOTSTRAP_TOKEN", bootstrap_token)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout_log))
            .stderr(Stdio::from(stderr_log));

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;

            command.creation_flags(0x08000000);
        }

        match command.spawn() {
            Ok(child) => {
                return Ok(ManagedBackend { child, api_url });
            }
            Err(error) => {
                last_error = format!("{}: {error}", launcher.program);
            }
        }
    }

    Err(DesktopBridgeError::startup(format!(
        "failed to launch dashboard backend with any Python runtime: {last_error}"
    )))
}

fn stop_backend(runtime: &mut BridgeRuntime) -> Result<(), DesktopBridgeError> {
    let Some(mut backend) = runtime.backend.take() else {
        return Ok(());
    };
    terminate_process_tree(&mut backend.child).map_err(|error| {
        DesktopBridgeError::internal(format!("failed to stop dashboard backend: {error}"))
    })
}

fn terminate_process_tree(child: &mut Child) -> Result<(), std::io::Error> {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = child.wait();
        return Ok(());
    }

    #[cfg(not(target_os = "windows"))]
    {
        child.kill()?;
        let _ = child.wait();
        Ok(())
    }
}

fn reserve_loopback_port() -> Result<u16, DesktopBridgeError> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|error| {
        DesktopBridgeError::startup(format!("failed to reserve loopback port: {error}"))
    })?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|error| {
            DesktopBridgeError::startup(format!("failed to resolve reserved loopback port: {error}"))
        })
}

fn new_token() -> String {
    Uuid::new_v4().as_simple().to_string()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn python_launchers(workspace_root: &Path) -> Vec<PythonLauncher> {
    vec![
        PythonLauncher {
            program: workspace_root
                .join(".venv")
                .join(if cfg!(target_os = "windows") {
                    "Scripts/python.exe"
                } else {
                    "bin/python"
                })
                .display()
                .to_string(),
            prefix_args: Vec::new(),
        },
        PythonLauncher {
            program: "python".to_string(),
            prefix_args: Vec::new(),
        },
        PythonLauncher {
            program: "py".to_string(),
            prefix_args: vec!["-3".to_string()],
        },
    ]
}