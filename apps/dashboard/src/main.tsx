import React, { FormEvent, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
    Activity,
    AlertTriangle,
    Check,
    ClipboardList,
    Database,
    Eye,
    FileText,
    Gauge,
    History,
    MessageSquare,
    MonitorCog,
    Pause,
    Plug,
    Play,
    RefreshCw,
    Search,
    Send,
    Settings,
    ShieldAlert,
    TerminalSquare
} from 'lucide-react';
import './styles.css';

const API_BASE =
    import.meta.env.VITE_AGENTOS_API_BASE || 'http://127.0.0.1:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

type JsonMap = Record<string, unknown>;
type View = 'research' | 'pc' | 'approvals' | 'runs' | 'events' | 'channels' | 'live-fire' | 'system';

type Approval = {
    approval_id: string;
    token: string;
    reasons: string[];
    status: string;
    action?: {
        action_type: string;
        target: string;
        payload: {
            action?: string;
            value_present?: boolean;
        };
    };
};

type BackendStatus = {
    name: string;
    available: boolean;
    error?: string;
};

type SystemStatus = {
    status: string;
    run_count: number;
    pending_approvals: number;
    jobs: RunJob[];
    pc_backends: BackendStatus[];
    daemon?: DaemonRecord;
};

type DaemonRecord = {
    status: string;
    launcher_pid: number | null;
    api_url: string;
    ui_url: string;
    log_path: string;
    started_at?: string | null;
    detail: string;
};

type SetupCheck = {
    check_id: string;
    label: string;
    status: string;
    detail: string;
    required: boolean;
    repair_hint?: string;
};

type ProviderStatus = {
    provider_id: string;
    label: string;
    kind: string;
    configured: boolean;
    detail: string;
};

type ChannelStatus = {
    channel_id: string;
    label: string;
    endpoint: string;
    configured: boolean;
    detail: string;
};

type ProductStatus = {
    checks: SetupCheck[];
    providers: ProviderStatus[];
    channels: ChannelStatus[];
    benchmarks: JsonMap;
};

type RunRecord = {
    run_id: string;
    objective: string;
    status: string;
    created_at: string;
    updated_at: string;
};

type RunJob = {
    job_id: string;
    objective: string;
    status: string;
    created_at: string;
    updated_at: string;
    run_id: string | null;
    error: string | null;
};

type EventPayload = {
    event?: {
        type: string;
        source: string;
        created_at: string;
        payload: JsonMap;
    };
    job?: JsonMap;
};

type UiNode = {
    node_id: string;
    role: string;
    name: string;
    bounds?: [number, number, number, number] | null;
    enabled: boolean;
    focused: boolean;
    metadata: {
        automation_id?: string;
        class_name?: string;
        process_id?: number;
        parent?: string;
    };
};

type ResearchArtifacts = {
    run_id: string;
    brief: string;
    sources: Array<{
        title: string;
        provider: string;
        url: string;
        year?: number;
        citation_count?: number;
        score?: number;
    }>;
    artifacts: string[];
};

type PcActionResponse = {
    status: 'approval_required' | 'blocked' | 'executed';
    decision?: {
        approval?: Approval;
        reasons?: string[];
    };
    receipt?: unknown;
};

type SelectorDebugReport = {
    selector: string;
    exact_matches: number;
    ready: boolean;
    guidance: string;
    candidates: Array<{
        selector: string;
        role: string;
        name: string;
        score: number;
        reasons: string[];
    }>;
};

type WorkflowCommand = {
    command_id: string;
    label: string;
    description: string;
    template: string;
    enabled: boolean;
};

type ChannelDelivery = {
    created_at: string;
    channel: string;
    sender_id: string;
    text: string;
    status: string;
};

type LiveFireFailure = {
    run_id: string;
    task_id: string;
    surface: string;
    intent: string;
    classification: string;
    durable: boolean;
    promotable: boolean;
    failure_reason: string;
    replay_payload: JsonMap;
    existing_golden_trace?: string;
};

type LiveFireRunReview = {
    run_id: string;
    backend: string;
    success: boolean;
    passed: number;
    failed: number;
    task_count: number;
    created_at: number;
    failures: LiveFireFailure[];
};

type LiveFireReviewPayload = {
    runs: LiveFireRunReview[];
    failed_tasks: LiveFireFailure[];
    milestone: {
        real_windows_task_target: number;
        durable_failure_target: number;
        real_windows_tasks: number;
        durable_promoted_failures: number;
        unsafe_action_blocks: number;
        ready_to_widen_scope: boolean;
    };
    triage_classes: string[];
};

type ShadowTrainingSummary = {
    advisory_only: boolean;
    ready_for_shadow_training: boolean;
    head_order: string[];
    total_examples: number;
    heads: Record<string, {
        path: string;
        examples: number;
        ready: boolean;
        advisory_only: boolean;
    }>;
    source_paths?: string[];
};

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, options);
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
    }
    return (await response.json()) as T;
}

function selectorFor(node: UiNode): string {
    if (node.metadata.automation_id) {
        return `automation_id=${node.metadata.automation_id}`;
    }
    if (node.name) {
        return `name=${node.name}`;
    }
    return `role=${node.role}`;
}

function approvalAction(approval: Approval) {
    const target = approval.action?.target || '';
    const separator = target.indexOf('://');
    return {
        backend: separator >= 0 ? target.slice(0, separator) : 'windows-uia',
        selector: separator >= 0 ? target.slice(separator + 3) : target,
        action: approval.action?.payload?.action || 'focus',
        hasHiddenValue: Boolean(approval.action?.payload?.value_present)
    };
}

function App() {
    const [view, setView] = useState<View>('research');
    const [events, setEvents] = useState<EventPayload[]>([]);
    const [approvals, setApprovals] = useState<Approval[]>([]);
    const [runs, setRuns] = useState<RunRecord[]>([]);
    const [jobs, setJobs] = useState<RunJob[]>([]);
    const [system, setSystem] = useState<SystemStatus | null>(null);
    const [product, setProduct] = useState<ProductStatus | null>(null);
    const [daemon, setDaemon] = useState<DaemonRecord | null>(null);
    const [objective, setObjective] = useState('');
    const [depth, setDepth] = useState('standard');
    const [runStatus, setRunStatus] = useState('idle');
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const [streamStatus, setStreamStatus] = useState('disconnected');
    const [error, setError] = useState('');
    const [pcBackend, setPcBackend] = useState('windows-uia');
    const [pcNodes, setPcNodes] = useState<UiNode[]>([]);
    const [pcSelector, setPcSelector] = useState('');
    const [pcAction, setPcAction] = useState('focus');
    const [pcValue, setPcValue] = useState('');
    const [pcReceipt, setPcReceipt] = useState<unknown>(null);
    const [pcDebug, setPcDebug] = useState<SelectorDebugReport | null>(null);
    const [pcReceipts, setPcReceipts] = useState<JsonMap[]>([]);
    const [selectedRun, setSelectedRun] = useState<ResearchArtifacts | null>(null);
    const [runCheckpoint, setRunCheckpoint] = useState<JsonMap | null>(null);
    const [policyActionType, setPolicyActionType] = useState('os.snapshot');
    const [policyTarget, setPolicyTarget] = useState('windows-uia://snapshot');
    const [policyDecision, setPolicyDecision] = useState<JsonMap | null>(null);
    const [channelName, setChannelName] = useState('dashboard');
    const [channelText, setChannelText] = useState('/run desktop agent safety');
    const [channelResponse, setChannelResponse] = useState<JsonMap | null>(null);
    const [commands, setCommands] = useState<WorkflowCommand[]>([]);
    const [deliveries, setDeliveries] = useState<ChannelDelivery[]>([]);
    const [goldenTraces, setGoldenTraces] = useState<JsonMap | null>(null);
    const [benchmarkReplay, setBenchmarkReplay] = useState<JsonMap | null>(null);
    const [liveFireReview, setLiveFireReview] = useState<LiveFireReviewPayload | null>(null);
    const [selectedLiveFireFailure, setSelectedLiveFireFailure] = useState<LiveFireFailure | null>(null);
    const [liveFireStatus, setLiveFireStatus] = useState('idle');
    const [shadowTraining, setShadowTraining] = useState<ShadowTrainingSummary | null>(null);

    const visibleEvents = useMemo(() => events.slice(0, 80), [events]);
    const backendAvailable = system?.pc_backends.find(
        (backend) => backend.name === pcBackend
    )?.available;

    useEffect(() => {
        const socket = new WebSocket(`${WS_BASE}/ws/events`);
        socket.onopen = () => setStreamStatus('streaming');
        socket.onclose = () => setStreamStatus('disconnected');
        socket.onerror = () => setStreamStatus('error');
        socket.onmessage = (message) => {
            const payload = JSON.parse(message.data) as EventPayload;
            setEvents((current) => [payload, ...current].slice(0, 120));
            if (
                payload.event?.type === 'run.completed' ||
                payload.event?.type === 'approval.requested' ||
                payload.job
            ) {
                void refreshAll();
            }
        };
        return () => socket.close();
    }, []);

    useEffect(() => {
        void refreshAll();
        const timer = window.setInterval(() => void refreshAll(), 5000);
        return () => window.clearInterval(timer);
    }, []);

    useEffect(() => {
        if (!activeJobId) {
            return;
        }
        const job = jobs.find((item) => item.job_id === activeJobId);
        if (!job) {
            return;
        }
        setRunStatus(job.status);
        if (job.status === 'completed' || job.status === 'failed') {
            setActiveJobId(null);
        }
    }, [activeJobId, jobs]);

    useEffect(() => {
        if (view === 'live-fire') {
            void loadLiveFireReview();
        }
    }, [view]);

    async function refreshAll() {
        try {
            const [
                statusPayload,
                approvalsPayload,
                runsPayload,
                jobsPayload,
                productPayload,
                daemonPayload,
                commandsPayload,
                deliveriesPayload,
                receiptsPayload,
                tracesPayload
            ] =
                await Promise.all([
                    fetchJson<SystemStatus>('/status'),
                    fetchJson<Approval[]>('/approvals'),
                    fetchJson<RunRecord[]>('/runs'),
                    fetchJson<RunJob[]>('/jobs'),
                    fetchJson<ProductStatus>('/setup/checks'),
                    fetchJson<DaemonRecord>('/daemon/status'),
                    fetchJson<WorkflowCommand[]>('/commands'),
                    fetchJson<ChannelDelivery[]>('/channels/deliveries'),
                    fetchJson<JsonMap[]>('/pc/receipts'),
                    fetchJson<JsonMap>('/benchmarks/golden-traces')
                ]);
            setSystem(statusPayload);
            setProduct(productPayload);
            setDaemon(daemonPayload);
            setApprovals(approvalsPayload);
            setRuns([...runsPayload].reverse());
            setJobs(jobsPayload);
            setCommands(commandsPayload);
            setDeliveries(deliveriesPayload);
            setPcReceipts(receiptsPayload);
            setGoldenTraces(tracesPayload);
            setError('');
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        }
    }

    async function resolveApproval(
        approval: Approval,
        action: 'approve' | 'deny',
        executeAfter = false
    ) {
        await fetchJson<Approval>(`/approvals/${approval.token}/${action}`, {
            method: 'POST'
        });
        if (action === 'approve' && executeAfter) {
            const request = approvalAction(approval);
            if (!request.hasHiddenValue) {
                const response = await runPcAction(approval.token, request);
                setPcReceipt(response);
            }
        }
        await refreshAll();
    }

    async function startRun(event: FormEvent) {
        event.preventDefault();
        const trimmed = objective.trim();
        if (!trimmed) {
            return;
        }
        setRunStatus('queued');
        const job = await fetchJson<RunJob>('/runs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ objective: trimmed, depth, background: true })
        });
        setActiveJobId(job.job_id);
        setObjective('');
        setRunStatus('running in background');
        await refreshAll();
    }

    async function loadSnapshot() {
        setPcReceipt(null);
        const payload = await fetchJson<{
            nodes?: UiNode[];
            status?: string;
            decision?: unknown;
        }>(
            `/pc/snapshot?backend=${encodeURIComponent(pcBackend)}&limit=160`
        );
        setPcNodes(Array.isArray(payload.nodes) ? payload.nodes : []);
        if (!Array.isArray(payload.nodes)) {
            setPcReceipt(payload);
            await refreshAll();
        }
    }

    async function requestPcAction(event: FormEvent) {
        event.preventDefault();
        const response = await runPcAction();
        setPcReceipt(response);
        await refreshAll();
    }

    async function debugPcSelector() {
        const payload = await fetchJson<{
            status: string;
            report?: SelectorDebugReport;
            decision?: JsonMap;
        }>('/pc/debug-selector', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend: pcBackend, selector: pcSelector })
        });
        setPcDebug(payload.report || null);
        if (!payload.report) {
            setPcReceipt(payload);
        }
    }

    async function controlDaemon(action: 'start' | 'stop' | 'restart') {
        const payload = await fetchJson<DaemonRecord>(`/daemon/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        setDaemon(payload);
        await refreshAll();
    }

    async function replayBenchmarks() {
        const payload = await fetchJson<JsonMap>('/benchmarks/replay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        setBenchmarkReplay(payload);
        await refreshAll();
    }

    async function loadLiveFireReview() {
        const payload = await fetchJson<LiveFireReviewPayload>(
            '/benchmarks/live-fire-review?limit=12'
        );
        setLiveFireReview(payload);
        if (!selectedLiveFireFailure && payload.failed_tasks.length > 0) {
            setSelectedLiveFireFailure(payload.failed_tasks[0]);
        }
    }

    async function runSafeLiveFire() {
        setLiveFireStatus('running safe Windows matrix');
        const payload = await fetchJson<JsonMap>('/benchmarks/live-fire-eval', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                backend: 'windows-uia',
                windows_safe_pack: true,
                max_tasks: 10,
                repeat: 1,
                promote_after: 2,
                replay_limit: 25
            })
        });
        setLiveFireStatus(String(payload.status || payload.run_id || 'complete'));
        await loadLiveFireReview();
        await refreshAll();
    }

    async function promoteSelectedFailure() {
        if (!selectedLiveFireFailure) {
            return;
        }
        setLiveFireStatus('promoting failure');
        const payload = await fetchJson<JsonMap>('/benchmarks/live-fire-review/promote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: selectedLiveFireFailure.run_id,
                task_id: selectedLiveFireFailure.task_id
            })
        });
        setLiveFireStatus(String(payload.status || 'review updated'));
        await loadLiveFireReview();
        await refreshAll();
    }

    async function runShadowTraining() {
        setLiveFireStatus('writing advisory heads');
        const payload = await fetchJson<ShadowTrainingSummary>(
            '/benchmarks/live-fire-shadow-training',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            }
        );
        setShadowTraining(payload);
        setLiveFireStatus(
            payload.ready_for_shadow_training ? 'shadow heads ready' : 'shadow heads incomplete'
        );
    }

    async function runPcAction(
        approvalToken?: string,
        override?: { backend: string; selector: string; action: string }
    ) {
        const selector = override?.selector || pcSelector.trim();
        const action = override?.action || pcAction;
        const backend = override?.backend || pcBackend;
        const response = await fetchJson<PcActionResponse>('/pc/actions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                backend,
                selector,
                action,
                value: action === 'type' || action === 'set_text' ? pcValue : null,
                approval_token: approvalToken
            })
        });
        return response;
    }

    async function openRun(runId: string) {
        const payload = await fetchJson<ResearchArtifacts>(
            `/runs/${runId}/research`
        );
        setSelectedRun(payload);
        setRunCheckpoint(null);
        setView('runs');
    }

    async function inspectRun(runId: string) {
        const payload = await fetchJson<JsonMap>(`/runs/${runId}`);
        setRunCheckpoint(payload);
        setSelectedRun(null);
        setView('runs');
    }

    async function recoverRun(runId: string) {
        const payload = await fetchJson<JsonMap>(`/runs/${runId}/recover`, {
            method: 'POST'
        });
        setRunCheckpoint(payload);
        await refreshAll();
    }

    async function inspectPolicy(event: FormEvent) {
        event.preventDefault();
        const payload = await fetchJson<JsonMap>('/policy/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action_type: policyActionType,
                target: policyTarget
            })
        });
        setPolicyDecision(payload);
    }

    async function sendChannelCommand(event: FormEvent) {
        event.preventDefault();
        const payload = await fetchJson<JsonMap>('/channels/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                channel: channelName,
                sender_id: 'dashboard',
                text: channelText
            })
        });
        setChannelResponse(payload);
        await refreshAll();
    }

    return (
        <main className="app-shell">
            <header>
                <div>
                    <h1>AgentOS Control</h1>
                    <p>{streamStatus}</p>
                </div>
                <button onClick={refreshAll} title="Refresh console">
                    <RefreshCw size={18} />
                </button>
            </header>

            <section className="status-strip" aria-label="System status">
                <div><strong>{system?.status || 'unknown'}</strong><span>API</span></div>
                <div><strong>{system?.run_count ?? runs.length}</strong><span>Runs</span></div>
                <div><strong>{approvals.length}</strong><span>Approvals</span></div>
                <div><strong>{backendAvailable ? 'ready' : 'limited'}</strong><span>PC control</span></div>
            </section>

            {error && <p className="notice">{error}</p>}

            <nav className="tabs" aria-label="Console views">
                <button className={view === 'research' ? 'active' : ''} onClick={() => setView('research')}>
                    <Search size={16} /> Research
                </button>
                <button className={view === 'pc' ? 'active' : ''} onClick={() => setView('pc')}>
                    <MonitorCog size={16} /> PC Control
                </button>
                <button className={view === 'approvals' ? 'active' : ''} onClick={() => setView('approvals')}>
                    <ShieldAlert size={16} /> Approvals
                </button>
                <button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}>
                    <History size={16} /> Runs
                </button>
                <button className={view === 'events' ? 'active' : ''} onClick={() => setView('events')}>
                    <Activity size={16} /> Events
                </button>
                <button className={view === 'channels' ? 'active' : ''} onClick={() => setView('channels')}>
                    <MessageSquare size={16} /> Channels
                </button>
                <button className={view === 'live-fire' ? 'active' : ''} onClick={() => setView('live-fire')}>
                    <AlertTriangle size={16} /> Live Fire
                </button>
                <button className={view === 'system' ? 'active' : ''} onClick={() => setView('system')}>
                    <Settings size={16} /> System
                </button>
            </nav>

            {view === 'research' && (
                <section className="layout two-col">
                    <div className="panel">
                        <h2><Search size={18} /> Deep Research</h2>
                        <form className="stack" onSubmit={startRun}>
                            <textarea
                                aria-label="Research objective"
                                value={objective}
                                onChange={(event) => setObjective(event.target.value)}
                                placeholder="Ask for a multi-hour research objective, audit, market map, literature review, or implementation plan"
                            />
                            <div className="toolbar">
                                <select value={depth} onChange={(event) => setDepth(event.target.value)} aria-label="Research depth">
                                    <option value="quick">Quick</option>
                                    <option value="standard">Standard</option>
                                    <option value="multi-hour">Multi-hour</option>
                                </select>
                                <button disabled={runStatus === 'queued'} title="Start background run">
                                    <Play size={16} /> Start
                                </button>
                            </div>
                        </form>
                        <p className="muted">{runStatus}</p>
                    </div>
                    <div className="panel">
                        <h2><ClipboardList size={18} /> Background Jobs</h2>
                        {jobs.length === 0 && <p className="muted">No background jobs yet</p>}
                        {jobs.map((job) => (
                            <article className="row" key={job.job_id}>
                                <div>
                                    <strong>{job.status}</strong>
                                    <p>{job.objective}</p>
                                    {job.error && <p className="danger">{job.error}</p>}
                                </div>
                                {job.run_id && (
                                    <button onClick={() => openRun(job.run_id!)} title="Open research brief">
                                        <FileText size={16} />
                                    </button>
                                )}
                            </article>
                        ))}
                    </div>
                </section>
            )}

            {view === 'pc' && (
                <section className="layout pc-layout">
                    <div className="panel">
                        <h2><Eye size={18} /> Accessibility Snapshot</h2>
                        <div className="toolbar">
                            <select value={pcBackend} onChange={(event) => setPcBackend(event.target.value)} aria-label="PC backend">
                                {(system?.pc_backends || [{ name: 'windows-uia', available: true }]).map((backend) => (
                                    <option key={backend.name} value={backend.name}>{backend.name}</option>
                                ))}
                            </select>
                            <button onClick={loadSnapshot} title="Read PC snapshot">
                                <RefreshCw size={16} /> Snapshot
                            </button>
                        </div>
                        <div className="node-list">
                            {pcNodes.length === 0 && <p className="muted">No snapshot loaded</p>}
                            {pcNodes.map((node) => (
                                <button
                                    className="node-row"
                                    key={node.node_id}
                                    onClick={() => setPcSelector(selectorFor(node))}
                                    title="Use this selector"
                                >
                                    <span>{node.role}</span>
                                    <strong>{node.name || node.metadata.class_name || node.node_id}</strong>
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="panel">
                        <h2><TerminalSquare size={18} /> Guarded Action</h2>
                        <form className="stack" onSubmit={requestPcAction}>
                            <input
                                aria-label="UI selector"
                                value={pcSelector}
                                onChange={(event) => setPcSelector(event.target.value)}
                                placeholder="name=Refresh approvals"
                            />
                            <div className="toolbar">
                                <select value={pcAction} onChange={(event) => setPcAction(event.target.value)} aria-label="PC action">
                                    <option value="focus">Focus</option>
                                    <option value="invoke">Invoke</option>
                                    <option value="click">Click</option>
                                    <option value="type">Type</option>
                                    <option value="set_text">Set text</option>
                                </select>
                                <button title="Request or execute action"><Send size={16} /> Run</button>
                            </div>
                            <button type="button" className="secondary" onClick={debugPcSelector} title="Debug selector">
                                <Eye size={16} /> Debug selector
                            </button>
                            {(pcAction === 'type' || pcAction === 'set_text') && (
                                <input
                                    aria-label="Action value"
                                    value={pcValue}
                                    onChange={(event) => setPcValue(event.target.value)}
                                    placeholder="Text to enter"
                                />
                            )}
                        </form>
                        {pcDebug && (
                            <div className="debug-box">
                                <h3>Selector Debug</h3>
                                <p>{pcDebug.guidance}</p>
                                {pcDebug.candidates.map((candidate) => (
                                    <article className="row" key={`${candidate.selector}-${candidate.score}`}>
                                        <div>
                                            <strong>{candidate.selector}</strong>
                                            <p>{candidate.name || candidate.role}</p>
                                            <span>{candidate.reasons.join(', ')}</span>
                                        </div>
                                        <span className="pill">{candidate.score}</span>
                                    </article>
                                ))}
                            </div>
                        )}
                        {pcReceipt && <pre>{JSON.stringify(pcReceipt, null, 2)}</pre>}
                        {pcReceipts.length > 0 && <h3>Recent Receipts</h3>}
                        {pcReceipts.slice(0, 4).map((receipt, index) => (
                            <pre key={index}>{JSON.stringify(receipt, null, 2)}</pre>
                        ))}
                    </div>
                </section>
            )}

            {view === 'approvals' && (
                <section className="panel full">
                    <h2><ShieldAlert size={18} /> Approvals</h2>
                    {approvals.length === 0 && <p className="muted">No pending approvals</p>}
                    {approvals.map((approval) => {
                        const request = approvalAction(approval);
                        return (
                            <article className="approval" key={approval.approval_id}>
                                <div>
                                    <strong>{approval.action?.action_type || approval.approval_id}</strong>
                                    <p>{request.selector || approval.approval_id}</p>
                                    <p className="muted">{approval.reasons.join(' ')}</p>
                                </div>
                                <div className="actions">
                                    <button onClick={() => resolveApproval(approval, 'approve', true)} title="Approve and execute">
                                        <Check size={16} /> Execute
                                    </button>
                                    <button className="secondary" onClick={() => resolveApproval(approval, 'deny')} title="Deny">
                                        <Pause size={16} /> Deny
                                    </button>
                                </div>
                            </article>
                        );
                    })}
                </section>
            )}

            {view === 'runs' && (
                <section className="layout two-col runs-view">
                    <div className="panel">
                        <h2><History size={18} /> Run History</h2>
                        {runs.length === 0 && <p className="muted">No runs yet</p>}
                        {runs.map((run) => (
                            <article className="row" key={run.run_id}>
                                <div>
                                    <strong>{run.status}</strong>
                                    <p>{run.objective}</p>
                                    <span>{run.run_id}</span>
                                </div>
                                <div className="actions">
                                    <button type="button" onClick={() => openRun(run.run_id)} title="Open artifacts">
                                        <FileText size={16} />
                                    </button>
                                    <button type="button" className="secondary" onClick={() => inspectRun(run.run_id)} title="Inspect checkpoint">
                                        <Eye size={16} />
                                    </button>
                                    {run.status !== 'completed' && (
                                        <button type="button" className="secondary" onClick={() => recoverRun(run.run_id)} title="Recover run">
                                            <RefreshCw size={16} />
                                        </button>
                                    )}
                                </div>
                            </article>
                        ))}
                    </div>
                    <div className="panel research-detail">
                        <h2><FileText size={18} /> Research Artifacts</h2>
                        {!selectedRun && !runCheckpoint && <p className="muted">Select a run</p>}
                        {selectedRun && (
                            <>
                                <div className="artifact-list">
                                    {selectedRun.artifacts.map((artifact) => <span key={artifact}>{artifact}</span>)}
                                </div>
                                <pre>{selectedRun.brief}</pre>
                                <h3>Sources</h3>
                                {selectedRun.sources.map((source) => (
                                    <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                                        {source.title} <span>{source.provider}</span>
                                    </a>
                                ))}
                            </>
                        )}
                        {runCheckpoint && <pre>{JSON.stringify(runCheckpoint, null, 2)}</pre>}
                    </div>
                </section>
            )}

            {view === 'events' && (
                <section className="panel full events">
                    <h2><Activity size={18} /> Event Stream</h2>
                    {visibleEvents.map((item, index) => (
                        <article key={index}>
                            <div>
                                <strong>{item.event?.type || 'job'}</strong>
                                <span>{item.event?.source || String(item.job?.status || '')}</span>
                            </div>
                            <pre>{JSON.stringify(item.event?.payload || item.job || {}, null, 2)}</pre>
                        </article>
                    ))}
                </section>
            )}

            {view === 'channels' && (
                <section className="layout two-col">
                    <div className="panel">
                        <h2><MessageSquare size={18} /> Channels</h2>
                        {(product?.channels || []).map((channel) => (
                            <article className="row" key={channel.channel_id}>
                                <div>
                                    <strong>{channel.label}</strong>
                                    <p>{channel.endpoint}</p>
                                    <span>{channel.detail}</span>
                                </div>
                                <span className={channel.configured ? 'pill pass' : 'pill'}>
                                    {channel.configured ? 'ready' : 'setup'}
                                </span>
                            </article>
                        ))}
                    </div>
                    <div className="panel">
                        <h2><Send size={18} /> Command</h2>
                        <form className="stack" onSubmit={sendChannelCommand}>
                            <select value={channelName} onChange={(event) => setChannelName(event.target.value)} aria-label="Channel">
                                <option value="dashboard">Dashboard</option>
                                <option value="generic-webhook">Generic webhook</option>
                                <option value="telegram">Telegram</option>
                                <option value="slack">Slack</option>
                                <option value="discord">Discord</option>
                            </select>
                            <textarea
                                aria-label="Channel command"
                                value={channelText}
                                onChange={(event) => setChannelText(event.target.value)}
                                placeholder="/run accessibility tree GUI agents"
                            />
                            <button title="Send channel command"><Send size={16} /> Send</button>
                        </form>
                        <h3>Workflow Commands</h3>
                        {commands.map((command) => (
                            <article className="row" key={command.command_id}>
                                <div>
                                    <strong>/{command.command_id}</strong>
                                    <p>{command.description}</p>
                                </div>
                                <button
                                    type="button"
                                    className="secondary"
                                    onClick={() => setChannelText(`/${command.command_id} desktop agent evaluation`)}
                                    title="Use command"
                                >
                                    <Send size={16} /> Use
                                </button>
                            </article>
                        ))}
                        <h3>Delivery Log</h3>
                        {deliveries.slice(0, 6).map((delivery) => (
                            <article className="row" key={`${delivery.created_at}-${delivery.channel}`}>
                                <div>
                                    <strong>{delivery.channel}</strong>
                                    <p>{delivery.text}</p>
                                    <span>{delivery.sender_id}</span>
                                </div>
                                <span className="pill">{delivery.status}</span>
                            </article>
                        ))}
                        {channelResponse && <pre>{JSON.stringify(channelResponse, null, 2)}</pre>}
                    </div>
                </section>
            )}

            {view === 'live-fire' && (
                <section className="layout two-col live-fire-view">
                    <div className="panel">
                        <h2><AlertTriangle size={18} /> Live-Fire Review</h2>
                        <div className="actions live-fire-actions">
                            <button type="button" onClick={runSafeLiveFire} title="Run safe Windows UIA matrix">
                                <Play size={16} /> Safe Windows run
                            </button>
                            <button type="button" className="secondary" onClick={loadLiveFireReview} title="Refresh live-fire review">
                                <RefreshCw size={16} /> Refresh
                            </button>
                            <button type="button" className="secondary" onClick={runShadowTraining} title="Write advisory shadow-training heads">
                                <Database size={16} /> Shadow train
                            </button>
                        </div>
                        <p className="muted">{liveFireStatus}</p>
                        {liveFireReview && (
                            <div className="milestone-grid">
                                <div>
                                    <strong>{liveFireReview.milestone.real_windows_tasks}/{liveFireReview.milestone.real_windows_task_target}</strong>
                                    <span>Windows tasks</span>
                                </div>
                                <div>
                                    <strong>{liveFireReview.milestone.durable_promoted_failures}/{liveFireReview.milestone.durable_failure_target}</strong>
                                    <span>Golden failures</span>
                                </div>
                                <div>
                                    <strong>{liveFireReview.milestone.unsafe_action_blocks}</strong>
                                    <span>Unsafe blocks</span>
                                </div>
                            </div>
                        )}
                        <h3>Recent Runs</h3>
                        {!liveFireReview && <p className="muted">No live-fire review loaded</p>}
                        {liveFireReview?.runs.map((run) => (
                            <article className="row" key={run.run_id}>
                                <div>
                                    <strong>{run.run_id}</strong>
                                    <p>{run.backend} · {run.passed}/{run.task_count} passed</p>
                                    <span>{run.failed} failed</span>
                                </div>
                                <span className={run.success ? 'pill pass' : 'pill fail'}>
                                    {run.success ? 'pass' : 'review'}
                                </span>
                            </article>
                        ))}
                        <h3>Failed Tasks</h3>
                        <div className="failure-list">
                            {liveFireReview?.failed_tasks.length === 0 && <p className="muted">No failed tasks in recent runs</p>}
                            {liveFireReview?.failed_tasks.map((failure) => (
                                <button
                                    type="button"
                                    className={selectedLiveFireFailure?.task_id === failure.task_id ? 'failure-item active' : 'failure-item'}
                                    key={`${failure.run_id}-${failure.task_id}`}
                                    onClick={() => setSelectedLiveFireFailure(failure)}
                                    title="Inspect replay payload"
                                >
                                    <span>{failure.classification}</span>
                                    <strong>{failure.task_id}</strong>
                                    <small>{failure.surface} / {failure.intent}</small>
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="panel">
                        <h2><Eye size={18} /> Replay Payload</h2>
                        {!selectedLiveFireFailure && <p className="muted">Select a failed task</p>}
                        {selectedLiveFireFailure && (
                            <>
                                <article className="row">
                                    <div>
                                        <strong>{selectedLiveFireFailure.classification}</strong>
                                        <p>{selectedLiveFireFailure.failure_reason}</p>
                                        <span>{selectedLiveFireFailure.run_id}</span>
                                    </div>
                                    <button
                                        type="button"
                                        disabled={!selectedLiveFireFailure.promotable}
                                        onClick={promoteSelectedFailure}
                                        title="Promote durable failure to golden trace"
                                    >
                                        <Check size={16} /> Promote
                                    </button>
                                </article>
                                <pre>{JSON.stringify(selectedLiveFireFailure.replay_payload, null, 2)}</pre>
                            </>
                        )}
                        {shadowTraining && (
                            <>
                                <h3>Shadow Heads</h3>
                                <div className="shadow-heads">
                                    {shadowTraining.head_order.map((head) => (
                                        <article className="row" key={head}>
                                            <div>
                                                <strong>{head}</strong>
                                                <p>{shadowTraining.heads[head]?.path}</p>
                                            </div>
                                            <span className={shadowTraining.heads[head]?.ready ? 'pill pass' : 'pill'}>
                                                {shadowTraining.heads[head]?.examples || 0}
                                            </span>
                                        </article>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                </section>
            )}

            {view === 'system' && (
                <section className="layout two-col">
                    <div className="panel">
                        <h2><Settings size={18} /> Policy</h2>
                        <form className="stack" onSubmit={inspectPolicy}>
                            <input
                                aria-label="Action type"
                                value={policyActionType}
                                onChange={(event) => setPolicyActionType(event.target.value)}
                                placeholder="os.snapshot"
                            />
                            <input
                                aria-label="Action target"
                                value={policyTarget}
                                onChange={(event) => setPolicyTarget(event.target.value)}
                                placeholder="windows-uia://snapshot"
                            />
                            <button title="Inspect policy decision"><Eye size={16} /> Inspect</button>
                        </form>
                        {policyDecision && <pre>{JSON.stringify(policyDecision, null, 2)}</pre>}
                        <h3>Daemon</h3>
                        <article className="row">
                            <div>
                                <strong>{daemon?.status || system?.daemon?.status || 'unknown'}</strong>
                                <p>{daemon?.detail || system?.daemon?.detail || 'local gateway lifecycle'}</p>
                                <span>{daemon?.api_url || system?.daemon?.api_url}</span>
                            </div>
                            <div className="actions">
                                <button type="button" className="secondary" onClick={() => controlDaemon('start')} title="Start daemon">
                                    <Play size={16} /> Start
                                </button>
                                <button type="button" className="secondary" onClick={() => controlDaemon('restart')} title="Restart daemon">
                                    <RefreshCw size={16} /> Restart
                                </button>
                                <button type="button" className="secondary" onClick={() => controlDaemon('stop')} title="Stop daemon">
                                    <Pause size={16} /> Stop
                                </button>
                            </div>
                        </article>
                        <h3>Readiness</h3>
                        <div className="check-list">
                            {(product?.checks || []).map((check) => (
                                <article className="row" key={check.check_id}>
                                    <div>
                                        <strong>{check.label}</strong>
                                        <p>{check.detail}</p>
                                        {check.repair_hint && <span>{check.repair_hint}</span>}
                                    </div>
                                    <span className={check.status === 'pass' ? 'pill pass' : 'pill fail'}>
                                        {check.status}
                                    </span>
                                </article>
                            ))}
                        </div>
                    </div>
                    <div className="panel">
                        <h2><Activity size={18} /> Runtime</h2>
                        <div className="system-grid">
                            <div><strong>{system?.status || 'unknown'}</strong><span>API</span></div>
                            <div><strong>{jobs.length}</strong><span>Jobs</span></div>
                            <div><strong>{runs.length}</strong><span>Runs</span></div>
                            <div><strong>{approvals.length}</strong><span>Approvals</span></div>
                        </div>
                        <h3><Plug size={14} /> Providers</h3>
                        {(product?.providers || []).map((provider) => (
                            <article className="row" key={provider.provider_id}>
                                <div>
                                    <strong>{provider.label}</strong>
                                    <p>{provider.kind}</p>
                                </div>
                                <span className={provider.configured ? 'pill pass' : 'pill'}>
                                    {provider.configured ? 'configured' : 'available'}
                                </span>
                            </article>
                        ))}
                        <h3><Gauge size={14} /> Benchmarks</h3>
                        {product?.benchmarks && <pre>{JSON.stringify(product.benchmarks, null, 2)}</pre>}
                        <div className="actions">
                            <button type="button" className="secondary" onClick={replayBenchmarks} title="Replay benchmark traces">
                                <Gauge size={16} /> Replay
                            </button>
                        </div>
                        {goldenTraces && <pre>{JSON.stringify(goldenTraces, null, 2)}</pre>}
                        {benchmarkReplay && <pre>{JSON.stringify(benchmarkReplay, null, 2)}</pre>}
                        <h3>Backends</h3>
                        {(system?.pc_backends || []).map((backend) => (
                            <article className="row" key={backend.name}>
                                <div>
                                    <strong>{backend.name}</strong>
                                    <p>{backend.available ? 'available' : backend.error || 'unavailable'}</p>
                                </div>
                            </article>
                        ))}
                    </div>
                </section>
            )}
        </main>
    );
}

createRoot(document.getElementById('root')!).render(<App />);