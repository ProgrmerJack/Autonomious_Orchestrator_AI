import { invoke } from '@tauri-apps/api/core';

import type { DashboardSession } from './api';

export type DaemonRecord = {
    status: string;
    launcher_pid: number | null;
    api_url: string;
    ui_url: string;
    log_path: string;
    started_at?: string | null;
    detail: string;
};

export type DesktopBootstrap = {
    api_base: string;
    session: DashboardSession;
    daemon: DaemonRecord;
    ipc_token: string;
};

type DesktopDaemonAction = 'start' | 'stop' | 'restart';

type TauriWindow = Window & {
    __TAURI_INTERNALS__?: unknown;
};

export function isDesktopShell(): boolean {
    return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in (window as TauriWindow);
}

function bridgeErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    if (error && typeof error === 'object' && 'message' in error) {
        const message = (error as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) {
            return message;
        }
    }
    return String(error);
}

async function invokeDesktop<T>(command: string, args?: Record<string, unknown>): Promise<T> {
    try {
        return await invoke<T>(command, args);
    } catch (error) {
        throw new Error(bridgeErrorMessage(error));
    }
}

export async function desktopBootstrap(): Promise<DesktopBootstrap> {
    return invokeDesktop<DesktopBootstrap>('desktop_bootstrap');
}

export async function desktopDaemonStatus(ipcToken: string): Promise<DaemonRecord> {
    return invokeDesktop<DaemonRecord>('desktop_daemon_status', { ipcToken });
}

export async function desktopDaemonControl(
    action: DesktopDaemonAction,
    ipcToken: string
): Promise<DaemonRecord> {
    return invokeDesktop<DaemonRecord>('desktop_daemon_control', { action, ipcToken });
}