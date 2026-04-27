from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentos_orchestrator.core.orchestrator import ResearchOrchestrator
from agentos_orchestrator.research import (
    DeepResearchEngine,
    ResearchBrief,
    ResearchSource,
)


class FakeResearchEngine(DeepResearchEngine):
    def run(
        self,
        objective: str,
        run_id: str,
        pc_context: dict | None = None,
        planning_context: dict | None = None,
        evidence_targets: dict | None = None,
    ) -> ResearchBrief:
        del pc_context, planning_context, evidence_targets
        return ResearchBrief(
            objective=objective,
            query=objective,
            summary="Router test research completed.",
            sources=[
                ResearchSource(
                    provider="test",
                    title="Router Source",
                    url="https://example.com/router",
                    abstract="Router evidence.",
                )
            ],
            artifacts=[],
            confidence=0.88,
        )


def write_policy(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def base_network_hosts() -> list[str]:
    return [
        "api.openalex.org",
        "api.semanticscholar.org",
        "api.crossref.org",
        "api.github.com",
        "generativelanguage.googleapis.com",
    ]


def new_orchestrator(root: Path, policy_payload: dict) -> ResearchOrchestrator:
    policy_path = root / "policy.json"
    write_policy(policy_path, policy_payload)
    return ResearchOrchestrator.from_paths(
        policy_path=policy_path,
        state_path=root / "state.sqlite3",
        memory_path=root / "memory.sqlite3",
    )


def temp_root() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()
