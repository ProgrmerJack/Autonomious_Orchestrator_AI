use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub(crate) fn default_bounds() -> Vec<i32> {
    vec![0, 0, 100, 30]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NodeRecord {
    pub(crate) node_id: String,
    pub(crate) role: String,
    pub(crate) name: String,
    pub(crate) focused: bool,
    pub(crate) enabled: bool,
    #[serde(default = "default_bounds")]
    pub(crate) bounds: Vec<i32>,
    pub(crate) text: String,
    pub(crate) value: String,
    pub(crate) metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SandboxState {
    pub(crate) focused: String,
    pub(crate) last_action: Option<Value>,
    #[serde(default)]
    pub(crate) agent_history: Vec<Value>,
    #[serde(default)]
    pub(crate) clipboard: String,
    #[serde(default)]
    pub(crate) virtual_processes: Vec<Value>,
    #[serde(default)]
    pub(crate) modals: Vec<Value>,
    #[serde(default)]
    pub(crate) virtual_file_contents: Map<String, Value>,
    #[serde(default)]
    pub(crate) virtual_files: Vec<String>,
    #[serde(default)]
    pub(crate) terminal_log: Vec<String>,
    pub(crate) nodes: Vec<NodeRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CommandEnvelope {
    Snapshot,
    Capabilities,
    NativeSnapshot,
    NativeAct {
        action_type: String,
        selector: String,
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        metadata: Option<Map<String, Value>>,
    },
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
