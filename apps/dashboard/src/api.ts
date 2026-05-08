export type DashboardSession = {
    session_token: string;
    csrf_token: string;
    issued_at: string;
    expires_at: string;
    unsafe_ack_value: string;
};

const queryBootstrap = new URLSearchParams(window.location.search).get('bootstrap');

export const DEFAULT_API_BASE =
    import.meta.env.VITE_AGENTOS_API_BASE || 'http://127.0.0.1:8000';

let apiBase = DEFAULT_API_BASE.replace(/\/$/, '');

export function getApiBase(): string {
    return apiBase;
}

export function setApiBase(nextApiBase: string): void {
    apiBase = (nextApiBase || DEFAULT_API_BASE).replace(/\/$/, '');
}

function rawFetch(path: string, options?: RequestInit) {
    return fetch(`${getApiBase()}${path}`, options);
}

export function resolveBootstrapToken(): string {
    const fromStorage = window.localStorage.getItem('agentosBootstrapToken') || '';
    const fromEnv = import.meta.env.VITE_AGENTOS_BOOTSTRAP_TOKEN || '';
    return queryBootstrap || fromEnv || fromStorage;
}

function rememberBootstrapToken(token: string): void {
    if (!token) {
        return;
    }
    window.localStorage.setItem('agentosBootstrapToken', token);
}

export async function establishDashboardSession(
    bootstrapToken: string
): Promise<DashboardSession> {
    const response = await rawFetch('/auth/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bootstrap_token: bootstrapToken })
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
    }
    rememberBootstrapToken(bootstrapToken);
    return (await response.json()) as DashboardSession;
}

export async function fetchDashboardJson<T>(
    path: string,
    session: DashboardSession,
    options?: RequestInit,
    unsafe = false
): Promise<T> {
    const headers = new Headers(options?.headers || {});
    headers.set('Authorization', `Bearer ${session.session_token}`);
    headers.set('X-AgentOS-Csrf', session.csrf_token);
    if (unsafe) {
        headers.set('X-AgentOS-Unsafe', session.unsafe_ack_value);
    }
    const response = await rawFetch(path, { ...options, headers });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
    }
    return (await response.json()) as T;
}

export function dashboardEventSocketUrl(session: DashboardSession): string {
    const wsBase = getApiBase().replace(/^http/, 'ws');
    const query = new URLSearchParams({
        session_token: session.session_token,
        csrf_token: session.csrf_token
    });
    return `${wsBase}/ws/events?${query.toString()}`;
}