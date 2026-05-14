import type { DaemonRecord } from './desktop';

export type JsonMap = Record<string, unknown>;

export type View =
    | 'research'
    | 'pc'
    | 'approvals'
    | 'runs'
    | 'events'
    | 'channels'
    | 'live-fire'
    | 'system';

export type Approval = {
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

export type BackendStatus = {
    name: string;
    available: boolean;
    error?: string;
};

export type RunRecord = {
    run_id: string;
    objective: string;
    status: string;
    created_at: string;
    updated_at: string;
};

export type RunJob = {
    job_id: string;
    objective: string;
    status: string;
    created_at: string;
    updated_at: string;
    run_id: string | null;
    error: string | null;
};

export type SystemStatus = {
    status: string;
    run_count: number;
    pending_approvals: number;
    jobs: RunJob[];
    pc_backends: BackendStatus[];
    daemon?: DaemonRecord;
};

export type SetupCheck = {
    check_id: string;
    label: string;
    status: string;
    detail: string;
    required: boolean;
    repair_hint?: string;
};

export type ProviderStatus = {
    provider_id: string;
    label: string;
    kind: string;
    configured: boolean;
    detail: string;
};

export type ChannelStatus = {
    channel_id: string;
    label: string;
    endpoint: string;
    configured: boolean;
    detail: string;
};

export type ProductStatus = {
    checks: SetupCheck[];
    providers: ProviderStatus[];
    channels: ChannelStatus[];
    benchmarks: JsonMap;
};

export type EventPayload = {
    event?: {
        type: string;
        source: string;
        created_at: string;
        payload: JsonMap;
    };
    job?: JsonMap;
};

export type UiNode = {
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

export type ResearchArtifacts = {
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

export type RunProgress = {
    run_id?: string;
    depth?: string;
    stage?: string;
    pass_index?: number;
    max_passes?: number;
    query_index?: number;
    query_total?: number;
    active_query?: string;
    recent_queries?: string[];
    elapsed_seconds?: number;
    sources_found?: number;
    stop_reason?: string | null;
    last_updated?: string;
    cycle?: number;
    max_cycles?: number;
    worker_count?: number;
    frontier_batch_size?: number;
    frontier_url_count?: number;
    direct_urls?: number;
    judged_results?: number;
    discovered_domains?: number;
    novelty_rate?: number;
    domain_count?: number;
    detached_frontier?: JsonMap;
    frontier_shards?: JsonMap[];
    detached_merge?: JsonMap;
    passes?: JsonMap[];
};

export type PcActionResponse = {
    status: 'approval_required' | 'blocked' | 'executed';
    decision?: {
        approval?: Approval;
        reasons?: string[];
    };
    receipt?: unknown;
};

export type SelectorDebugReport = {
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

export type WorkflowCommand = {
    command_id: string;
    label: string;
    description: string;
    template: string;
    enabled: boolean;
};

export type ChannelDelivery = {
    created_at: string;
    channel: string;
    sender_id: string;
    text: string;
    status: string;
};

export type LiveFireFailure = {
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

export type LiveFireRunReview = {
    run_id: string;
    backend: string;
    success: boolean;
    passed: number;
    failed: number;
    task_count: number;
    created_at: number;
    failures: LiveFireFailure[];
};

export type LiveFireReviewPayload = {
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

export type ShadowTrainingSummary = {
    advisory_only: boolean;
    ready_for_shadow_training: boolean;
    head_order: string[];
    total_examples: number;
    heads: Record<
        string,
        {
            path: string;
            examples: number;
            ready: boolean;
            advisory_only: boolean;
        }
    >;
    source_paths?: string[];
};