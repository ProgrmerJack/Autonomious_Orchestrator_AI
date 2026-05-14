# Universal OS Control Agent Research Plan

Date: 2026-05-08

## Executive Thesis

AgentOS should not try to become universal by making a larger click loop. The research consensus and this repo's current architecture point to a stronger target: a self-healing OS control substrate that selects the safest control lane for each step, verifies intent before execution, validates outcomes after execution, converts every failure into training data, and can shift between app APIs, code, accessibility, native input, and visual grounding without losing the user's goal.

The practical definition of "anything on PC" should be:

1. Understand the current surface through fused observations.
2. Choose the best available control lane, preferring APIs and code over GUI input when they are safer.
3. Execute only after policy, intent, and risk checks pass.
4. Prove the result with execution-based verification.
5. If blocked, diagnose, repair, or safely ask for approval.
6. Promote the failure into durable replay and training data.

This is ambitious, but it is also compatible with what already exists in AgentOS.

## Current AgentOS Foundation

AgentOS already has several pieces that should be kept and expanded instead of replaced:

- [agentos_orchestrator/cognition/universal_agent_v2.py](../agentos_orchestrator/cognition/universal_agent_v2.py): the current cognitive loop with world model, perception, mode arbitration, safety gates, repair planning, trajectory recording, and semantic memory.
- [agentos_orchestrator/cognition/capability_profile.py](../agentos_orchestrator/cognition/capability_profile.py): surface profiling by app family, accessibility quality, DOM presence, canvas likelihood, and control channels.
- [agentos_orchestrator/cognition/app_adapters.py](../agentos_orchestrator/cognition/app_adapters.py): app-family adapters with preferred channels, repair recipes, and verification contracts.
- [agentos_orchestrator/cognition/control_surface_discovery.py](../agentos_orchestrator/cognition/control_surface_discovery.py): discovery of loopback APIs, OpenAPI, GraphQL, local manifests, workspace artifacts, and visible API surfaces.
- [agentos_orchestrator/cognition/tool_executor.py](../agentos_orchestrator/cognition/tool_executor.py): sandboxed code execution for quantitative analysis, HTTP probes, and artifact generation.
- [agentos_orchestrator/cognition/safety_gates.py](../agentos_orchestrator/cognition/safety_gates.py): deterministic safety gates for risky actions.
- [agentos_orchestrator/cognition/verification_contracts.py](../agentos_orchestrator/cognition/verification_contracts.py): typed post-action verification contracts.
- [agentos_orchestrator/cognition/runtime_state.py](../agentos_orchestrator/cognition/runtime_state.py): temporal trace, blockers, action records, hypothesis state, and outcome evaluation.
- [agentos_orchestrator/cognition/blocker_repair.py](../agentos_orchestrator/cognition/blocker_repair.py): deterministic repair plans for modals, stale selectors, invalid input, missing resources, and approvals.
- [agentos_orchestrator/cognition/affordance_policy_memory.py](../agentos_orchestrator/cognition/affordance_policy_memory.py): durable affordance memory by app signature.
- [agentos_orchestrator/cognition/trajectory_training.py](../agentos_orchestrator/cognition/trajectory_training.py): conversion of traces into perception, affordance, policy, world-model, and outcome-critic training heads.
- [agentos_orchestrator/cognition/live_fire_eval.py](../agentos_orchestrator/cognition/live_fire_eval.py): safe live-fire Windows task evaluation and durable failure capture.
- [agentos_orchestrator/os_control/base.py](../agentos_orchestrator/os_control/base.py): common `snapshot()` and `perform()` OS-control protocol.
- [agentos_orchestrator/os_control/windows_uia_backend.py](../agentos_orchestrator/os_control/windows_uia_backend.py): Windows UI Automation backend with Rust-native fallback.
- [agentos_orchestrator/os_control/rust_native_windows_backend.py](../agentos_orchestrator/os_control/rust_native_windows_backend.py): Rust-backed native Windows launch, hotkey, coordinate input, scroll, type, and draw path.
- [agentos_orchestrator/os_control/visual_fallback.py](../agentos_orchestrator/os_control/visual_fallback.py): accessibility-first plus visual refinement fallback.
- [crates/agent_body/src/native.rs](../crates/agent_body/src/native.rs): first-party Rust native Windows input primitives.
- [agentos_orchestrator/os_control/workflow/service.py](../agentos_orchestrator/os_control/workflow/service.py): dashboard workflow entry point that can enable Universal Desktop Agent V2.

The gap is not that AgentOS lacks a universal agent. The gap is that the universal agent needs a stronger substrate: richer action contracts, pre-action verification, control-lane arbitration, isolated execution, app-specific skill packs, and benchmark-driven promotion.

## External Research Evidence

The plan below is grounded in these findings:

- OSWorld and OSWorld-Verified define the right evaluation style: real desktop tasks, reproducible setup, cross-app workflows, and execution-based evaluation instead of prompt-only judgment. The OSWorld-Verified update is especially relevant to AgentOS because it documents 300-plus benchmark repairs, centralized verification, AWS-backed reproducibility, and the exact failure pressures that matter in deployment: environment drift, timing dependencies, anti-bot friction, ambiguous tasks, and brittle evaluators. Source: [OSWorld-Verified](https://xlang.ai/blog/osworld-verified).
- Microsoft UFO2 shows a scalable desktop architecture pattern: a HostAgent delegates to AppAgents, uses Windows UIA, Win32, WinCOM, screenshots, control filters, app APIs, speculative multi-action execution, a continuous knowledge substrate, and Picture-in-Picture execution for isolation. Sources: [UFO overview](https://microsoft.github.io/UFO/overview/) and [UFO2 paper](https://arxiv.org/html/2504.14603).
- OpenCUA reinforces the data and model side of the roadmap: large-scale cross-OS computer-use traces, reflective action supervision, and open foundation models are now strong enough to set meaningful open-source baselines. Its reported 45.0 percent OSWorld-Verified result is a useful target for what open substrate improvements can unlock before proprietary frontier models are involved. Source: [OpenCUA](https://arxiv.org/html/2508.09123v3).
- Agent-S shows that strong GUI agents now combine a main planner, specialized grounding, reflection, local code execution, and behavior best-of-N. Its README reports Agent S3 reaching 66 percent OSWorld in a 100-step setting and 72.6 percent with behavior best-of-N. Source: [Agent-S](https://github.com/simular-ai/Agent-S).
- CoAct-1's key design lesson is that coding must be treated as an action. Data processing, reports, file transforms, stock analysis, and presentations should be done through code when possible, with GUI used for tasks only the GUI can do. It also supports the HostAgent plus delegated executor pattern already emerging in this repo. Source: [CoAct-1](https://arxiv.org/abs/2508.03923).
- CUA emphasizes infrastructure: desktop sandboxes, screenshot, shell, mouse, keyboard, mobile gestures, OSWorld and Windows Arena benchmarking, trajectory export, and background computer use that does not steal the user's active cursor. Source: [CUA](https://github.com/trycua/cua).
- VeriSafe Agent shows why pre-action verification matters. Reflection after execution cannot prevent irreversible actions. Logic-based pre-action verification can catch intent/action mismatches before the action takes effect. Source: [VeriSafe Agent](https://arxiv.org/html/2503.18492).
- CORA shows the right safety control shape: post-policy, pre-action selective execution, action-conditional risk scoring, conformal risk calibration, Goal-Lock, and a diagnostician that chooses confirm, reflect, or abort. Source: [CORA](https://arxiv.org/html/2604.09155).
- The 2026 deployment-grounded reliability framework for computer-use agents maps directly onto what AgentOS is missing: explicit separation of perception, decision, and execution layers, plus a lifecycle split between creation, deployment, operation, and maintenance. Its central lesson is that failure manifestation and failure origin are often different, so controls must be attached where authority is bound, not only where incidents become visible. Source: [Securing Computer-Use Agents](https://arxiv.org/abs/2605.07110).
- Anthropic's computer-use guidance reinforces the deployment posture: use a VM/container or isolated environment, constrain access, avoid sensitive accounts when possible, and treat computer-use agents as high-risk executors. Source: [Anthropic computer use](https://www.anthropic.com/news/3-5-models-and-computer-use).

## Target Architecture

The target architecture is a five-layer system.

### 1. Observation Fusion Layer

Create a canonical `ObservationFrame` that merges:

- Accessibility tree nodes from Windows UIA.
- Native screen and cursor state from Rust.
- Screenshot and OCR evidence.
- App capability profile.
- Discovered APIs and local manifests.
- Focus, modal, process, file, and clipboard state.
- Stable UI fingerprint and semantic diff.

Current hooks: [runtime_state.py](../agentos_orchestrator/cognition/runtime_state.py), [capability_profile.py](../agentos_orchestrator/cognition/capability_profile.py), [windows_uia_backend.py](../agentos_orchestrator/os_control/windows_uia_backend.py), [rust_native_windows_backend.py](../agentos_orchestrator/os_control/rust_native_windows_backend.py), [visual_fallback.py](../agentos_orchestrator/os_control/visual_fallback.py).

Required upgrades:

- Add stable UI canonicalization that ignores clocks, counters, cursor noise, and transient animations while preserving semantic changes.
- Attach every observed element to a control-channel confidence vector: `api`, `code`, `accessibility`, `native`, `vision`, `clipboard`, `manual`.
- Persist observation fingerprints so repeated failures can be clustered across sessions.

### 2. HostAgent and AppAgent Runtime

Keep one global HostAgent that owns the user goal, safety budget, app routing, and cross-app task graph. Add AppAgents as specialized workers for app families: browser, Figma/design canvas, Office/presentation, spreadsheet, terminal, file explorer, PDF, chat, trading terminal, enterprise grid, and unknown app.

Current hooks: [app_adapters.py](../agentos_orchestrator/cognition/app_adapters.py), [app_family_registry.py](../agentos_orchestrator/app_family_registry.py), [capability_profile.py](../agentos_orchestrator/cognition/capability_profile.py), [affordance_policy_memory.py](../agentos_orchestrator/cognition/affordance_policy_memory.py).

Required upgrades:

- Promote app-family adapters into AppAgent skill packs with channel preference, safe actions, forbidden actions, expected modals, repair recipes, and verification templates.
- Give each AppAgent a small durable memory keyed by app signature and version.
- Let HostAgent decompose tasks across apps, hand state to the right AppAgent, and recover if the active app changes unexpectedly.

### 3. Four-Lane Action Router

Every step should route through the highest-reliability lane available:

1. API/MCP lane: local app APIs, browser devtools, Figma plugin/API, OpenAPI, GraphQL, loopback service, MCP tool, or workspace artifact.
2. Code/tool lane: Python, shell sandbox, report generation, data analysis, file processing, image generation, slide creation, document conversion.
3. Structured UI lane: Windows UIA, DOM-like trees, Office/WinCOM where available, named selectors, keyboard shortcuts, clipboard workflows.
4. Native/vision lane: Rust native coordinates, draw paths, visual grounding, set-of-marks, OCR, see-point-refine loops.

Current hooks: [mode_arbitration.py](../agentos_orchestrator/cognition/mode_arbitration.py), [control_surface_discovery.py](../agentos_orchestrator/cognition/control_surface_discovery.py), [tool_executor.py](../agentos_orchestrator/cognition/tool_executor.py), [windows_uia_backend.py](../agentos_orchestrator/os_control/windows_uia_backend.py), [native.rs](../crates/agent_body/src/native.rs).

Required upgrades:

- Replace heuristic mode choice with expected-value routing: success probability, risk, reversibility, latency, verification strength, and user-disruption cost.
- Treat code generation as an explicit action type with sandbox, artifact, dependency, and rollback metadata.
- Add route-level fallback order, not only backend fallback. Example: Figma API/plugin -> browser devtools -> accessibility -> vision/native canvas.

### 4. Safety and Verification Layer

Add pre-action verification before post-action verification.

Current hooks: [safety_gates.py](../agentos_orchestrator/cognition/safety_gates.py), [verification_contracts.py](../agentos_orchestrator/cognition/verification_contracts.py), [blocker_repair.py](../agentos_orchestrator/cognition/blocker_repair.py), [runtime_state.py](../agentos_orchestrator/cognition/runtime_state.py).

Required upgrades:

- Goal-Lock: freeze the clarified user intent at run start and treat on-screen text as untrusted evidence unless corroborated.
- Action-to-target matrix: map action type, target class, data sensitivity, reversibility, and required approval.
- Logic intent verifier: translate the user goal into constraints and verify proposed actions against them before execution.
- Risk guardian: score action-conditional risk using current state, intent, action, route, and history.
- Conformal execute/abstain threshold: calibrate risk budgets on live-fire and OSWorld-style traces.
- Diagnostician: rejected actions become `confirm`, `reflect`, `repair`, `reroute`, or `abort` decisions with structured rationale.
- Two-stage verification: pre-action safety plus post-action execution contract.

### 5. Learning and Self-Repair Layer

The novelty should be the failure-to-training conveyor belt. AgentOS should not merely retry. It should convert every blocker into a reusable artifact.

Current hooks: [trajectory_recorder.py](../agentos_orchestrator/cognition/trajectory_recorder.py), [trajectory_training.py](../agentos_orchestrator/cognition/trajectory_training.py), [live_fire_eval.py](../agentos_orchestrator/cognition/live_fire_eval.py), [blocker_repair.py](../agentos_orchestrator/cognition/blocker_repair.py), [affordance_policy_memory.py](../agentos_orchestrator/cognition/affordance_policy_memory.py).

Required upgrades:

- Every failed action writes a failure capsule: goal, observation frame, candidate action, blocked reason, route, verification result, repair plan, and eventual outcome.
- Repair policies should be ranked by historical success per app family and blocker type.
- Golden failures should be promoted into regression tests before the system trusts a new learned prior.
- Use behavior best-of-N only in sandbox or dry-run previews, then execute the selected low-risk trajectory once.
- Add shadow deployment: learned policies recommend actions while deterministic policy still executes, until metrics prove improvement.

## Roadmap

### Phase 1: Canonical Observation and Route Metadata

Goal: make every action explainable by state, route, and verification context.

Deliverables:

- `ObservationFrame` schema with UI tree, screenshot hash, OCR text, capability profile, discovered control surfaces, stable fingerprint, and semantic diff.
- `ActionProposal` schema with route, reversibility, expected state change, verification contract, required approval, and rollback notes.
- Route metadata added to every `UiAction.metadata` before execution.
- Observation snapshots stored with trajectories and live-fire results.

Acceptance gates:

- Existing safe-pack workflows still pass.
- At least 20 real or sandbox workflows produce complete observation/action/proof records.
- No action receipt lacks route and verification metadata.

Implementation status:

- Started in [control_substrate.py](../agentos_orchestrator/cognition/control_substrate.py) with typed `ObservationFrame`, `ActionProposal`, `PreActionDecision`, four-lane routing, pre-action checks, and the append-only Adaptive Control Ledger.
- Wired into [service.py](../agentos_orchestrator/os_control/workflow/service.py) so desktop workflow actions are routed, verified, stamped with control metadata, and recorded before execution.
- Covered in [test_workflow_service.py](../tests/test_workflow_service.py) with route metadata, durable ledger, and pre-action blocking tests.

### Phase 2: HostAgent/AppAgent Skill Packs

Goal: stop treating all apps as unknown canvases.

Deliverables:

- AppAgent skill packs for browser, file explorer, terminal, editor, spreadsheet, presentation, design canvas/Figma, PDF, chat, trading terminal, enterprise grid, unknown.
- AppAgent registry backed by existing app family registry.
- Per-AppAgent channel ordering, verification templates, known safe shortcuts, modal recipes, and risk notes.
- App signature memory using existing affordance policy memory.

Acceptance gates:

- Each skill pack has at least 5 safe tasks in virtual sandbox or live safe-pack.
- Unknown-app flow can profile a new window and select a channel without hand-coded one-off logic.
- Re-running a successful app-family task reuses memory and reduces step count.

Implementation status:

- [app_agent_runtime.py](../agentos_orchestrator/cognition/app_agent_runtime.py) now carries `action_policy` through AppAgent skill-pack resolution and records policy anchors in session memory.
- [app_family_registry.py](../agentos_orchestrator/app_family_registry.py) now defines explicit policy coverage for `browser`, `file_dialog`, and `enterprise_grid` in addition to the earlier `chat_app` and `trading_terminal` families.
- [service.py](../agentos_orchestrator/os_control/workflow/service.py) and [control_substrate.py](../agentos_orchestrator/cognition/control_substrate.py) already feed app-family context into routing and pre-action verification, so skill-pack metadata now affects both execution choice and approval/forbid behavior.
- [test_workflow_service.py](../tests/test_workflow_service.py) covers browser checkout approval, file-dialog path escape blocking, enterprise-grid bulk-delete approval, and the family-policy regression surface.

### Phase 3: Code-as-Action Programmer Lane

Goal: use code for artifacts, analysis, and data tasks instead of wasting GUI steps.

Deliverables:

- Programmer action type for Python/shell snippets with sandbox, dependency, artifact, and verification metadata.
- Artifact pipelines for markdown, PDF, DOCX, PPTX, CSV, PNG charts, image assets, and web pages.
- Domain recipes for stock analysis, financial reports, presentations, file transformations, and structured research briefs.
- GUI handoff after code execution when the final task requires interaction with PowerPoint, Figma, browser, or another app.

Acceptance gates:

- Stock analysis workflow creates a verified report artifact without opening a browser unless needed.
- Presentation workflow creates a PPTX artifact and verifies file existence/hash.
- Report workflow creates markdown/PDF and verifies content criteria.
- Code lane refuses high-risk local execution unless sandbox and policy gates allow it.

Implementation status:

- [planner.py](../agentos_orchestrator/os_control/workflow/planner.py) and [reasoner.py](../agentos_orchestrator/os_control/workflow/reasoner.py) now emit explicit `tool` and `api_call` steps directly instead of relying on late service-time inference for the first non-UI promotion.
- [service.py](../agentos_orchestrator/os_control/workflow/service.py) already materializes `code_tool` and `api_mcp` routes, preserves explicit `api_call` endpoints, and now executes a bounded `workflow_research_brief` tool request through the existing research provider stack.
- [adapters.py](../agentos_orchestrator/os_control/workflow/adapters.py) now adds a first-class research adapter that emits `tool_executor:workflow_research` for research/search/report objectives and suppresses browser-first fallback when a provider-backed brief can satisfy the first step.
- The workflow research lane writes `research_brief.md` into the workspace before any browser handoff and is now the default first move for objectives like market analysis, benchmark search, external-topic reports, and external-topic presentations.
- [retrieval.py](../agentos_orchestrator/research/retrieval.py) now applies platform-aware headless browser prefetch budgets before JS-heavy enrichment and crawl-worker rendering, capping URL count, per-page render size, and aggregate prefetched text so browser-backed research stays enabled without recreating the transient Windows OOM spike.
- [planner.py](../agentos_orchestrator/os_control/workflow/planner.py) now merges pronoun-style analytical continuations such as `find X and analyze it` back into a single segment so the research lane does not duplicate evidence gathering across artificial fragments.
- [reasoner.py](../agentos_orchestrator/os_control/workflow/reasoner.py) now gives deterministic direct-lane research/programmer/API decisions precedence over model-backed UI suggestions so the explicit non-UI path remains authoritative.
- [programmer.py](../agentos_orchestrator/os_control/workflow/programmer.py) now treats `research_brief.md` as an input artifact and synthesizes both `report.md` and `presentation_outline.md` directly from the brief before any document or slide-editing UI handoff.
- [artifacts.py](../agentos_orchestrator/research/artifacts.py) now emits `frontier_schedule.json`, `frontier_shards.json`, and `frontier_shards.md` for every research run, turning detached crawl and broker shard state into explicit run-level scheduling and shard summaries instead of hiding that state behind the broker only.
- [planning_synthesis.py](../agentos_orchestrator/research/planning_synthesis.py) now surfaces the detached frontier schedule directly in `analysis_report.md`, so the report records shard count, backlog pressure, assigned query slices, and representative queued or processed URLs.
- Remaining gap: the first shard-aware reporting layer now exists, but true 10k-scale fanout still needs detached execution beyond crawl-only workers, durable shard-level synthesis packets, and a final merge stage that can reconcile shard claims without forcing one live synthesis window to hold the full crawl in memory.

### Phase 4: Pre-Action Verification and Risk Control

Goal: prevent unsafe actions before they happen.

Deliverables:

- Goal-Lock object created at task start and included in every verifier call.
- Intent constraint compiler for common task families.
- Pre-action verifier that checks action proposals against intent constraints and action-to-target policy.
- Guardian risk model interface, starting with deterministic heuristics and later calibrated learned scores.
- Diagnostician output schema: `confirm`, `reflect`, `repair`, `reroute`, `abort`.

Acceptance gates:

- Unsafe actions in payments, credential entry, external messaging, file deletion, trading/order placement, and permission grants are blocked or require approval.
- Benign tasks do not suffer more than a defined interruption budget.
- Every blocked action produces an actionable repair or confirmation path.

Implementation status:

- [control_substrate.py](../agentos_orchestrator/cognition/control_substrate.py) already evaluates family-specific pre-action policy before execution and now limits policy matching to semantic action fields instead of accidentally matching policy definitions themselves.
- [service.py](../agentos_orchestrator/os_control/workflow/service.py) now skips selector preflight for explicit API actions and preserves route-specific verification contracts when a control route is materialized into a tool lane.
- [test_workflow_service.py](../tests/test_workflow_service.py) verifies the current guardrail surface end to end with approval, forbid, file-exists, receipt-success, and direct-lane regression tests.

### Phase 5: Speculative Planning with Safe Commit

Goal: get the speed of multi-action planning without reckless execution.

Deliverables:

- Generate multiple candidate action sequences in sandbox or dry-run mode.
- Score each sequence by route reliability, risk, expected verification strength, and historical success.
- Commit only the first verified low-risk prefix on the host.
- Re-observe after each committed prefix and invalidate stale plans.

Acceptance gates:

- Step count drops on repetitive UI tasks without increasing failed or blocked actions.
- Stale selector failures decrease.
- The system never commits speculative irreversible actions without explicit approval.

### Phase 6: Isolated Desktop Execution

Goal: let the agent work without disrupting the user's active mouse, keyboard, or windows.

Deliverables:

- Adapter seam for a background/isolated desktop backend: Windows VM, RDP session, CUA-style sandbox, or local virtual desktop.
- Shared clipboard and artifact sync policy.
- Replayable video/screenshot/trajectory export.
- Dashboard toggle for host control vs isolated control.

Acceptance gates:

- At least one browser/report workflow completes in isolated mode without stealing active focus.
- Artifacts sync back to the workspace with verified hashes.
- High-risk tasks default to isolated execution when possible.

### Phase 7: Benchmark and Promotion System

Goal: make capability growth measurable and prevent regressions.

Deliverables:

- OSWorld-style task format for AgentOS safe-pack: setup, instruction, allowed tools, evaluation script, cleanup.
- Live-fire tiers: sandbox, isolated desktop, host safe-pack, approval-gated host tasks.
- Safety benchmark based on OS-Harm/Phone-Harm style categories: misuse, injection, model misbehavior, privacy, financial, destructive file actions.
- Promotion gates for learned policies and new AppAgents.

Acceptance gates:

- 50 Windows safe-pack tasks pass with no unsafe-action increase.
- 10 durable failures are promoted into regression tests.
- Every release reports task success, step count, blocked-action rate, false-block rate, recovery rate, and artifact verification rate.

## Example Task Coverage

### Draw or Design

Preferred route:

1. Use code to generate vector geometry or bitmap assets.
2. If app supports API/plugin, import assets directly.
3. If app is a canvas, use AppAgent design-canvas skill pack.
4. Use Rust `draw_path` only when route and risk checks pass.
5. Verify through screenshot/image diff or exported artifact hash.

Repo hooks: [native.rs](../crates/agent_body/src/native.rs), [visual_fallback.py](../agentos_orchestrator/os_control/visual_fallback.py), [app_adapters.py](../agentos_orchestrator/cognition/app_adapters.py).

### Analyze Stocks

Preferred route:

1. Code/tool lane fetches data and computes indicators.
2. Research lane gathers current context through a provider-backed `research_brief.md` when external evidence is needed.
3. Code lane creates charts and report artifacts.
4. GUI lane only opens a browser, spreadsheet, or presentation app if final delivery requires it.

Repo hooks: [tool_executor.py](../agentos_orchestrator/cognition/tool_executor.py), [adapters.py](../agentos_orchestrator/os_control/workflow/adapters.py), [service.py](../agentos_orchestrator/os_control/workflow/service.py).

### Create Presentations and Reports

Preferred route:

1. If the objective needs external context, the research lane generates a bounded provider-backed brief before any browser-first UI handoff.
2. Code lane synthesizes `report.md` and `presentation_outline.md` directly from that brief before any document or slide UI handoff.
3. Verify files by existence, hash, slide/page count, and required text.
4. UI lane opens PowerPoint, browser, or design app only for manual-style finishing.
5. If GUI export is required, post-action verification checks exported hash.

Repo hooks: [tool_executor.py](../agentos_orchestrator/cognition/tool_executor.py), [verification_contracts.py](../agentos_orchestrator/cognition/verification_contracts.py), [os_control/workflow/service.py](../agentos_orchestrator/os_control/workflow/service.py).

### Use Figma

Preferred route:

1. Check for local API/plugin/devtools route.
2. If running in browser, inspect DOM/devtools and network surfaces.
3. If canvas-only, use screenshot, OCR, set-of-marks, visual grounding, and native input.
4. Generate design assets through code first, then import or place them.
5. Verify by exported image, selected layers, screenshot diff, or file artifact.

Repo hooks: [control_surface_discovery.py](../agentos_orchestrator/cognition/control_surface_discovery.py), [visual_fallback.py](../agentos_orchestrator/os_control/visual_fallback.py), [capability_profile.py](../agentos_orchestrator/cognition/capability_profile.py).

### Arbitrary Unknown App

Preferred route:

1. Profile app family and capabilities.
2. Discover APIs, local files, docs, command surfaces, and visible controls.
3. Try reversible probes first.
4. Record successful affordances.
5. If blocked, repair or ask for approval.
6. Promote failures into a new AppAgent skill pack when repeated.

Repo hooks: [capability_profile.py](../agentos_orchestrator/cognition/capability_profile.py), [control_surface_discovery.py](../agentos_orchestrator/cognition/control_surface_discovery.py), [blocker_repair.py](../agentos_orchestrator/cognition/blocker_repair.py), [affordance_policy_memory.py](../agentos_orchestrator/cognition/affordance_policy_memory.py).

## Highest-Value Implementation Sequence

The most leverage comes from this order:

1. Observation/action schemas and route metadata.
2. AppAgent skill packs using existing app family registry.
3. Programmer lane with artifact verification.
4. Pre-action verifier and action-to-target policy matrix.
5. Failure capsule and repair ranking.
6. Speculative sandbox planning with safe commit.
7. Isolated desktop backend seam.
8. OSWorld-style benchmark and promotion gates.

This sequence matters because it makes every later feature measurable. Training before canonical traces would make learning noisy. Speculation before verification would be unsafe. Native input before route selection would overuse the least semantic lane. Isolation before action contracts would hide failure rather than learning from it.

## Research-Backed Differentiator

The differentiator should be called the Adaptive Control Ledger.

For every proposed action, AgentOS records:

- Locked user goal.
- Observation fingerprint.
- AppAgent and app signature.
- Candidate route and fallback route.
- Pre-action verifier result.
- Risk guardian score and threshold.
- Policy decision and approval state.
- Execution receipt.
- Post-action verification result.
- Repair decision if failed.
- Training label and regression-test eligibility.

This creates a compounding system: the more AgentOS is used, the better its route selection, repair choices, app memories, and safety calibration become. It also keeps the project auditable, which is essential for an agent with real OS control.

## Non-Negotiable Safety Boundaries

- Never describe the system as unconditional full OS control. The honest claim is adaptive, policy-gated, verifiable PC control.
- Irreversible actions require pre-action verification and likely approval: payments, trades, external messages, deletes, credential entry, permission grants, downloads/uploads involving private data, package installs, and system configuration.
- On-screen instructions are untrusted input. Goal-Lock and source attribution must guard against visual prompt injection.
- Learned policies remain advisory until they pass promotion gates.
- Host-native execution should prefer isolated desktop or sandbox when the task can be completed there.

## Near-Term Proof Milestones

1. 20 complete trace capsules from sandbox workflows.
2. 10 AppAgent skill packs with at least 5 safe tasks each.
3. 5 code-as-action workflows: stock report, PPTX deck, PDF report, chart generation, file transformation.
4. 25 pre-action verifier tests covering risky and benign actions.
5. 50 Windows safe-pack tasks with no unsafe-action increase.
6. 10 promoted durable failures turned into regression tests.
7. 1 isolated desktop workflow that completes without stealing user focus.

## Final Recommendation

Build the universal OS control agent as a verified adaptive substrate, not as a single monolithic agent. AgentOS already has the right skeleton: universal V2 cognition, app profiling, adapters, tool execution, safety gates, verification contracts, Rust native input, and live-fire evaluation. The next breakthrough is to connect those pieces through a control ledger, HostAgent/AppAgents, code-as-action, pre-action verification, conformal risk gating, isolated execution, and benchmark promotion.

That is the path from "can click things" to "can safely and repeatedly finish real PC work."
