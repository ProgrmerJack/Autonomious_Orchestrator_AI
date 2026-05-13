# AgentOS Current Architecture Audit

Date: 2026-05-08

## Executive Verdict

AgentOS is no longer accurately described by the earlier audit that called the Rust crate a JSON-only stub and the Windows UIA backend PowerShell-shaped. The current runtime is a hybrid system with Python orchestration, policy, research, and workflow state, plus a Rust `agent_body` process that exposes both the simulated desktop protocol and first-party native Windows input commands.

The project is improving, but it is still not a promise of flawless OS control in every possible situation. The realistic target is practical universality: profile a surface, choose a control channel, verify the action, execute through policy gates, and either prove success or create a repairable blocker.

## Remediated Since The Original Audit

- Rust body split: `crates/agent_body/src/main.rs` is now a thin wrapper around `lib.rs`, with separate `models.rs`, `native.rs`, `state.rs`, and `surfaces.rs` modules.
- Native Rust commands: `agent_body` now supports `native_snapshot` and `native_act`, with Windows native input for launch/open, hotkey, cursor, click, type, scroll, wait, and draw-path actions.
- Python Rust bridge: `RustNativeWindowsBackend` exposes the native body as the `rust-native-windows` PC backend.
- Windows UIA modernization: `WindowsUiaBackend` uses in-process UI automation and can fall back to the Rust native input lane for coordinate/native actions.
- PC workflow wiring: dashboard workflow execution enables the universal V2 desktop agent before executing PC workflows.
- Simulated-rights honesty: the Python virtual desktop sandbox and Rust sandbox capability payload now report simulated virtual rights, not full host control.
- Repo hygiene: generated root Rust-native state and stale egg-info packaging metadata are ignored/removed.

## Remaining High-Priority Gaps

1. `agents.py`, `dashboard.py`, `deep_research.py`, and the dashboard `main.tsx` remain large modules and should keep being decomposed by route, role, and view boundary.
2. MCP remains strongest in the research path; worker tool execution still needs first-party MCP-backed filesystem, shell-sandbox, UI automation, and research-aggregator tools.
3. Research HTTP migration is partial. `httpx` is available, but many provider and retrieval paths still use synchronous `urllib` calls.
4. Policy target validation should become an explicit action-to-target matrix for high-risk actions such as OS actions and subprocess execution.
5. Training artifacts are loaded by the V2 runtime, but learned priors should remain advisory until live Windows safe-pack metrics prove improvement without increasing unsafe-action blocks.
6. The Rust native body performs native input, but it is not yet a complete cross-platform accessibility engine. Windows UIA still provides structured element discovery.

## Validation Snapshot

- Rust crate tests pass with native command and simulated capability coverage.
- Full Python test suite passes after the focused regression fixes and cleanup.
- VS Code still reports existing line-length diagnostics in large legacy Python modules; the touched backend, bridge, workflow, and test files are clean.

## Operational Position

Safe to use for guarded research, simulated desktop workflows, and approval-gated Windows UIA/Rust-native PC experiments. Not safe to describe as unconditional, failure-free full OS control. The next meaningful proof point is the Plan.md milestone: at least 50 real Windows safe-pack tasks, at least 10 durable promoted failures, and no unsafe-action increase before widening scope.

