from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import ResearchSettings


def settings_for_depth(depth: str) -> ResearchSettings:
    if depth == "quick":
        return ResearchSettings(
            depth="quick",
            max_sources=6,
            per_provider=4,
            max_query_variants=2,
        )
    if depth == "multi-hour":
        return ResearchSettings(
            depth="multi-hour",
            max_sources=1800,
            per_provider=140,
            max_query_variants=120,
        )
    return ResearchSettings(
        depth="standard",
        max_sources=18,
        per_provider=10,
        max_query_variants=8,
    )


def settings_for_current_web(settings: ResearchSettings) -> ResearchSettings:
    if settings.depth == "multi-hour":
        return ResearchSettings(
            depth=settings.depth,
            max_sources=min(max(settings.max_sources, 1200), 1600),
            per_provider=min(max(settings.per_provider, 48), 72),
            max_query_variants=min(max(settings.max_query_variants, 48), 72),
        )
    if settings.depth == "standard":
        return ResearchSettings(
            depth=settings.depth,
            max_sources=max(settings.max_sources, 24),
            per_provider=max(settings.per_provider, 12),
            max_query_variants=max(settings.max_query_variants, 10),
        )
    return ResearchSettings(
        depth=settings.depth,
        max_sources=max(settings.max_sources, 12),
        per_provider=max(settings.per_provider, 8),
        max_query_variants=max(settings.max_query_variants, 4),
    )


def settings_for_general_complex_objective(
    settings: ResearchSettings,
    objective: str,
    *,
    looks_like_academic_query: Callable[[str], bool],
    looks_like_software_agent_query: Callable[[str], bool],
    looks_like_comprehensive_research: Callable[[str], bool],
) -> ResearchSettings:
    if settings.depth == "multi-hour":
        return settings
    if looks_like_academic_query(objective):
        return settings
    lower = objective.lower()
    if not (
        looks_like_software_agent_query(objective)
        or looks_like_comprehensive_research(lower)
    ):
        return settings
    if settings.depth == "standard":
        return ResearchSettings(
            depth=settings.depth,
            max_sources=max(settings.max_sources, 72),
            per_provider=max(settings.per_provider, 24),
            max_query_variants=max(settings.max_query_variants, 24),
        )
    return ResearchSettings(
        depth=settings.depth,
        max_sources=max(settings.max_sources, 18),
        per_provider=max(settings.per_provider, 8),
        max_query_variants=max(settings.max_query_variants, 6),
    )


def query_parallel_worker_count(
    depth: str,
    query_count: int,
    provider_count: int,
    *,
    current_web_mode: bool,
) -> int:
    if query_count <= 0:
        return 0
    if depth == "multi-hour":
        return max(
            1,
            min(16, query_count, max(8, provider_count * 2)),
        )
    if current_web_mode:
        return max(1, min(12, query_count, max(6, provider_count * 2)))
    return max(1, min(4, query_count))


def provider_parallel_worker_count(
    provider_count: int,
    *,
    depth: str,
    current_web_mode: bool,
) -> int:
    if provider_count <= 0:
        return 0
    if depth == "multi-hour":
        return max(1, min(4, provider_count))
    if current_web_mode:
        return max(1, min(6, provider_count))
    return max(1, min(3, provider_count))


def current_web_targets(depth: str) -> dict[str, int | float]:
    if depth == "multi-hour":
        return {
            "max_retrieval_passes": 48,
            "depth_pass_floor": 8,
            "max_low_novelty_streak": 6,
            "min_unique_urls": 400,
            "min_perspective_count": 6,
            "min_perspective_ratio": 0.75,
        }
    if depth == "standard":
        return {
            "max_retrieval_passes": 8,
            "depth_pass_floor": 4,
            "min_perspective_count": 4,
            "min_perspective_ratio": 0.7,
        }
    return {
        "max_retrieval_passes": 3,
        "depth_pass_floor": 2,
    }


def current_web_target_overrides(
    targets: dict[str, Any],
    depth: str,
) -> dict[str, Any]:
    merged = dict(targets)
    for key, value in current_web_targets(depth).items():
        if key == "max_retrieval_passes":
            merged[key] = max(int(merged.get(key) or 0), int(value))
        else:
            merged[key] = value

    provider_floor = 6 if depth == "multi-hour" else 3 if depth == "standard" else 2
    merged["min_provider_count"] = max(
        int(merged.get("min_provider_count") or 0),
        provider_floor,
    )
    source_floor = 120 if depth == "multi-hour" else 36 if depth == "standard" else 12
    merged["min_source_count"] = max(
        int(merged.get("min_source_count") or 0),
        source_floor,
    )
    strong_floor = 8 if depth == "multi-hour" else 3 if depth == "standard" else 1
    merged["min_strong_or_moderate"] = max(
        int(merged.get("min_strong_or_moderate") or 0),
        strong_floor,
    )
    merged["min_scholarly_sources"] = 0
    merged["min_novelty_rate"] = 0.0
    return merged
