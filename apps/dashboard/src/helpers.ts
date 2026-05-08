import type { Approval, RunProgress, UiNode } from './types';

export function selectorFor(node: UiNode): string {
    if (node.metadata.automation_id) {
        return `automation_id=${node.metadata.automation_id}`;
    }
    if (node.name) {
        return `name=${node.name}`;
    }
    return `role=${node.role}`;
}

export function approvalAction(approval: Approval) {
    const target = approval.action?.target || '';
    const separator = target.indexOf('://');
    return {
        backend: separator >= 0 ? target.slice(0, separator) : 'windows-uia',
        selector: separator >= 0 ? target.slice(separator + 3) : target,
        action: approval.action?.payload?.action || 'focus',
        hasHiddenValue: Boolean(approval.action?.payload?.value_present)
    };
}

export function runProgressSummary(progress: RunProgress | null): string {
    if (!progress) {
        return '';
    }
    const stage = String(progress.stage || '');
    if (stage.startsWith('pc-research')) {
        const cycle = progress.cycle ?? progress.pass_index ?? 0;
        const maxCycles = progress.max_cycles ?? progress.max_passes;
        const cycleText = maxCycles ? `${cycle}/${maxCycles}` : String(cycle);
        return `Cycle ${cycleText} · ${progress.discovered_domains ?? 0} domains · ${progress.direct_urls ?? 0} reads`;
    }
    if (stage.startsWith('retrieval')) {
        const pass = progress.pass_index ?? 0;
        const maxPasses = progress.max_passes;
        const passText = maxPasses ? `${pass}/${maxPasses}` : String(pass);
        const queryText = progress.query_total
            ? ` · query ${progress.query_index ?? 0}/${progress.query_total}`
            : '';
        const sourceText =
            typeof progress.sources_found === 'number'
                ? ` · ${progress.sources_found} sources`
                : '';
        return `Pass ${passText}${queryText}${sourceText}`;
    }
    if (typeof progress.elapsed_seconds === 'number') {
        return `${Math.round(progress.elapsed_seconds)}s elapsed`;
    }
    return stage || 'Progress available';
}

export function runProgressQuery(progress: RunProgress | null): string {
    if (!progress) {
        return '';
    }
    if (progress.active_query) {
        return progress.active_query;
    }
    const recent = Array.isArray(progress.recent_queries)
        ? progress.recent_queries.find((query) => Boolean(query))
        : '';
    return recent || '';
}