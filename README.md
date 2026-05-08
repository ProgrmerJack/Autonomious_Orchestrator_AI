# Autonomous Orchestrator AI

This repository turns the architectural plan in [Plan.md](Plan.md) into a working foundation for a secure, OS-level deep research agent.

The implementation follows a hybrid model:

- Python is the brain: orchestration, event routing, policy checks, checkpointing, MCP wiring, and worker coordination.
- Rust is the body: a low-latency process boundary for native input,
  accessibility, file I/O, and event streaming integrations.
- Security is default-deny: worker agents must declare intended actions before execution, and the policy engine blocks forbidden actions before any tool is invoked.

## What Works Now

- Event-driven orchestration with a durable SQLite event log.
- Checkpoint save/load for long-running research runs.
- Supervisor, worker, verification, and memory compression roles.
- Permission boundary mapping with allow, deny, and approval-required decisions.
- MCP stdio JSON-RPC client seam for tools/resources/prompts.
- Live deep research retrieval through public scholarly APIs, with MCP seams for richer sources.
- OS control abstraction with Windows UI Automation, Rust-native Windows
  input fallback, direct `rust-native-windows`, and the honest simulated
  virtual desktop sandbox.
- Sandbox provider seams for local dry-run, c/ua, Firecracker, and Kata-style execution.
- A Rust body process that can be built independently and now exposes both
  the simulated virtual desktop protocol and first-party native Windows
  commands through `native_snapshot` and `native_act`.
- Markdown runtime configuration via [SOUL.md](SOUL.md), [AGENTS.md](AGENTS.md), and [HEARTBEAT.md](HEARTBEAT.md).
- Human approval CLI commands, Telegram-style channel routing, and an optional FastAPI/Tauri dashboard gateway.
- Operator dashboard controls for background research jobs, run history,
  artifact review, run recovery, policy inspection, approval execution, and
  Windows UI Automation snapshots/actions.
- One-install Windows launcher that bootstraps Python and dashboard
  dependencies, creates a desktop shortcut, and starts the API plus UI from a
  single command.
- Unit tests covering policy gates, checkpoints, and end-to-end orchestration.

## Competitive Product Capabilities

AgentOS now targets the practical strengths of OpenClaw, OpenCode, and
OpenHands as concrete local product surfaces:

- Always-on gateway operation through `agentos daemon start/status/stop`.
- One-install Windows launcher through [scripts/install-agentos.ps1](scripts/install-agentos.ps1)
  and [AgentOS.cmd](AgentOS.cmd).
- UI-first operation for research, PC control, approvals, run recovery,
  readiness checks, provider inventory, channel commands, and benchmarks.
- Provider inventory for OpenAI, Anthropic, Google Gemini, OpenRouter, Azure
  OpenAI, Ollama, and LM Studio without exposing secret values.
- Channel command routes for dashboard commands, generic webhooks, and
  Telegram-compatible payloads.
- SDK-style automation through `agentos_orchestrator.sdk.AgentOSClient`.
- Benchmark-oriented runtime status combining readiness checks, provider and
  channel counts, and live event evaluator results.
- Secure PC control with default-deny policy, exact approval tokens, trust
  degradation, Windows accessibility-tree snapshots/actions, and a
  `rust-native-windows` backend for coordinate input, hotkeys, launch/open,
  scroll, and draw-path actions.

## One-Install Windows Start

From the workspace root, run the installer once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\install-agentos.ps1
```

The installer creates `.venv`, installs the Python dashboard extra, installs
dashboard npm dependencies, builds the UI, creates [AgentOS.cmd](AgentOS.cmd),
adds an `AgentOS` desktop shortcut, and launches the local app.

After that, start AgentOS with one command:

```powershell
.\AgentOS.cmd
```

You can also launch both services directly through the CLI:

```powershell
python -m agentos_orchestrator launch
```

For daily use, run AgentOS as a detached local gateway:

```powershell
python -m agentos_orchestrator daemon start --open-browser
python -m agentos_orchestrator daemon status
python -m agentos_orchestrator daemon stop
```

Run readiness checks at any time:

```powershell
python -m agentos_orchestrator doctor
```

Open `http://127.0.0.1:5173/`. The dashboard is the intended control surface
for research, approvals, run inspection/recovery, policy checks, PC snapshots,
and approval-gated PC actions.

## Quick Start CLI

From the workspace root:

```powershell
python -m agentos_orchestrator run --objective "map the literature on accessibility-tree desktop agents" --policy examples/policies/deep_research.json
```

The run writes evidence artifacts under `runs/<run_id>/research/` and records
events, workflow steps, checkpoints, approvals, trust state, and memory in
`.agentos/`.

Run verification:

```powershell
python -m unittest discover -s tests
```

Build the Rust body when Rust is available:

```powershell
cargo check --manifest-path crates/agent_body/Cargo.toml
```

Select a PC backend explicitly when needed:

```powershell
python -m agentos_orchestrator pc-snapshot --backend windows-uia
python -m agentos_orchestrator pc-snapshot --backend rust-native-windows
python -m agentos_orchestrator pc-snapshot --backend virtual-desktop-sandbox
```

`windows-uia` is the structured accessibility backend. When it cannot satisfy
a coordinate-native action, it can delegate guarded input to the Rust native
body instead of failing immediately. `rust-native-windows` exposes the Rust
input path directly. `virtual-desktop-sandbox` remains the safe simulated
backend for tests and dry-runs.

Inspect markdown configuration:

```powershell
python -m agentos_orchestrator config --root .
```

Approve or deny a pending authorization token:

```powershell
python -m agentos_orchestrator approve --token <approval-token>
python -m agentos_orchestrator deny --token <approval-token>
```

Run the optional dashboard API by itself after installing the dashboard extra:

```powershell
python -m pip install -e ".[dashboard]"
python -m agentos_orchestrator serve-dashboard --host 127.0.0.1 --port 8000
```

Run the Vite dashboard frontend from [apps/dashboard](apps/dashboard):

```powershell
npm install
npm run dev
```

The dashboard is the intended non-coding control surface:

- Deep Research starts background jobs so long runs do not block the UI.
- Research depth is real backend behavior: `Quick` uses fewer query variants,
  `Standard` broadens coverage, and `Multi-hour` expands source retrieval,
  query plans, compressed digests, and research planning artifacts while keeping
  evidence on disk instead of in chat context.
- PC Control reads the Windows accessibility tree and prepares selectors.
- Guarded Action requests or executes approval-gated UIA actions.
- Approvals can approve, deny, and execute exact approved host actions.
- Runs shows completed run history, checkpoints, recovery, and generated
  research artifacts.
- System exposes runtime status, PC backend status, and policy inspection.
- Events streams durable run, policy, verification, memory, and job activity.

Send Telegram-compatible webhook payloads to the gateway:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/channels/telegram -ContentType application/json -Body '{"message":{"text":"/run accessibility tree GUI agents","chat":{"id":42}}}'
```

Send generic channel commands to the gateway:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/channels/command -ContentType application/json -Body '{"text":"/run accessibility tree GUI agents"}'
```

Script the local gateway with the Python SDK:

```python
from agentos_orchestrator.sdk import AgentOSClient

client = AgentOSClient("http://127.0.0.1:8000")
client.start_run("map desktop agent safety literature", depth="multi-hour")
```

Inspect read-only PC UI state through Windows UI Automation:

```powershell
python -m agentos_orchestrator pc-snapshot --limit 20 --policy examples/policies/deep_research.json
```

Request an approval-gated PC action:

```powershell
python -m agentos_orchestrator pc-act --action invoke --selector "name=Calculator" --policy examples/policies/deep_research.json
```

The command prints a pending approval token and also surfaces it in the
dashboard. Approve it, then rerun with `--approval-token <token>`.

## Architecture

```text
User request
  -> SupervisorAgent plans constrained worker tasks
  -> PermissionPolicy verifies declared action boundaries
  -> WorkerAgents execute through MCP, OS-control, or sandbox adapters
  -> DurableEventLog records every meaningful state transition
  -> CheckpointStore saves resumable state at decision points
  -> VerificationAgent reviews outputs before synthesis
  -> CognitiveCompressor commits only supported facts to memory
  -> Dashboard gateway streams events and pending approvals when enabled
```

## Safety Posture

This project intentionally does not grant blanket administrative access. The implementation preserves the plan's autonomy goals while enforcing sandboxing and least privilege. Real host execution, OS actions, and broad network access should remain behind explicit policy and human approval.

For complex scientific workflows that need end-to-end multi-agent research infrastructure, [K-Dense Web](https://www.k-dense.ai) is also worth considering.
