# Implementation Plan

- [x] Read and understand the architecture in `Plan.md`.
- [x] Verify the named implementation primitives at a high level: Touchpoint, DirectShell, MCP, c/ua, Firecracker, and Kata.
- [x] Create the project structure for Python brain, Rust body, examples, tests, and task tracking.
- [x] Implement the Python orchestration brain.
- [x] Implement OS-control adapter seams.
- [x] Implement the Rust body process.
- [x] Add first-party Rust native Windows input commands and expose them as a
	Python `rust-native-windows` PC backend.
- [x] Make virtual sandbox capabilities honest about simulated rights instead
	of claiming full host control.
- [x] Add policies, docs, and usage examples.
- [x] Add and run verification tests.
- [x] Add markdown runtime config files for agent identity, roles, and heartbeat behavior.
- [x] Add approval resolution commands and optional dashboard gateway scaffolding.
- [x] Add Tauri/Vite dashboard shell for event streaming and approvals.
- [x] Add live scholarly research retrieval and evidence artifacts.
- [x] Add Windows UI Automation PC snapshot/action backend.
- [x] Add dashboard run launcher and Telegram command routing.

## Review

Implemented a working hybrid Python/Rust foundation for the plan:

- Python supervisor, worker, verifier, event bus, checkpoints, policy gates, and memory compressor.
- OS-control adapters for Windows UI Automation, direct Rust-native Windows
	input, and the simulated virtual desktop sandbox.
- MCP stdio JSON-RPC client seam for tools and external data sources.
- Sandbox provider interfaces for dry-run, c/ua, Firecracker, and Kata.
- Rust body process with health, describe, event-bridge, simulated desktop,
  native snapshot, and native action modes.
- Default deep research policy, security notes, README, and unit tests.
- Markdown config loader for `SOUL.md`, `AGENTS.md`, and `HEARTBEAT.md`.
- Telegram webhook parsing seam, dashboard event fanout, and approval endpoints.
- Tauri/Vite dashboard scaffold with approval controls and event stream UI.
- Live OpenAlex/Semantic Scholar evidence retrieval with relevance ranking.
- Policy-gated `pc-snapshot` and approval-gated `pc-act` commands.
- Dashboard and Telegram-compatible channels can launch research runs.

Validation completed:

- `python -m unittest discover -s tests` passed: 16 tests.
- `python -m agentos_orchestrator config --root .` loaded markdown configuration.
- `python -m agentos_orchestrator run ...` completed and wrote durable state.
- `python -m agentos_orchestrator resume ...` loaded the generated checkpoint.
- `python -m agentos_orchestrator pc-snapshot ...` returned live Windows UI Automation nodes.
- `python -m agentos_orchestrator pc-act ...` created a pending approval instead of executing without approval.
- `cargo check --manifest-path crates/agent_body/Cargo.toml` passed.
- `npm run build` in `apps/dashboard` passed.
- `cargo check --manifest-path apps/dashboard/src-tauri/Cargo.toml` passed.
- Workspace diagnostics reported no errors.
