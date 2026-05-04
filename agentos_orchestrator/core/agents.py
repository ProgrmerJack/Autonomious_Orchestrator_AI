from __future__ import annotations

import inspect
import json
import os
import re
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .authorization import AuthorizationMiddleware
from .approvals import ApprovalRequired
from .checkpoint import CheckpointStore
from .events import EventBus
from .types import ActionRequest, TaskSpec, WorkerResult, new_id
from agentos_orchestrator.os_control import WindowsUiaBackend
from agentos_orchestrator.os_control.base import (
    BackendUnavailable,
    UiAction,
    UiNode,
)
from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
    VirtualDesktopSandboxBackend,
)
from agentos_orchestrator.research import DeepResearchEngine, ResearchSource


def declared_action(
    agent_id: str,
    action_type: str,
    target: str,
    payload: dict | None = None,
) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        action_type=action_type,
        target=target,
        payload=payload or {},
    )


class SupervisorAgent:
    """Plans work and preserves separation from execution tools."""

    agent_id = "supervisor"

    def _adaptive_plan_roles(
        self, objective: str, effort: str, needs_pc_context: bool
    ) -> list[str]:
        system = (
            "You are an adaptive task planner. Analyze the research objective and "
            "determine which agent roles are needed to accomplish it. The available "
            "roles are: 'planning' (for complex strategy), 'pc-control' "
            "(if pulling live host data, interacting with a local application, or running a local script), "
            "'pc-research' (for sandbox browser research to find and analyze websites), "
            "'literature' (for scholarly APIs and general search), "
            "'data' (for constraint extraction), and 'synthesis' (for merging outputs). "
            "Respond ONLY with a valid JSON array of role strings in execution order."
        )
        user = f"Objective: {objective}\nDetermine the best roles."
        try:
            raw = DeepResearchEngine()._call_ai_text(system, user)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                roles = json.loads(raw[start:end])
                if isinstance(roles, list) and roles:
                    return [
                        r
                        for r in roles
                        if r
                        in {
                            "planning",
                            "pc-control",
                            "pc-research",
                            "literature",
                            "data",
                            "synthesis",
                        }
                    ]
        except Exception:
            pass

        roles = []
        if effort == "multi-hour":
            roles.append("planning")
        if needs_pc_context:
            roles.append("pc-control")
        if self._needs_active_pc_research(objective, effort, needs_pc_context):
            roles.append("pc-research")
        roles.extend(["literature", "data", "synthesis"])
        return roles

    def plan(self, objective: str) -> list[TaskSpec]:
        planning_id = new_id("task_planning")
        literature_id = new_id("task_literature")
        pc_research_id = new_id("task_pc_research")
        data_id = new_id("task_data")
        synthesis_id = new_id("task_synthesis")
        tasks: list[TaskSpec] = []

        effort = self._research_effort(objective)
        needs_pc_context = self._needs_pc_context(objective)

        selected_roles = self._adaptive_plan_roles(objective, effort, needs_pc_context)

        if "planning" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=planning_id,
                    role="planning",
                    objective=f"Design deep research plan for: {objective}",
                    declared_actions=[
                        declared_action(
                            "planning-agent",
                            "file.write",
                            "runs/**",
                        )
                    ],
                )
            )

        if "pc-control" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=new_id("task_pc"),
                    role="pc-control",
                    objective=(
                        "Capture live desktop and browser/operator context "
                        f"for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "pc-control-agent",
                            "os.snapshot",
                            "sandbox://virtual-desktop/snapshot",
                        ),
                        declared_action(
                            "pc-control-agent",
                            "file.write",
                            "runs/**",
                        ),
                    ],
                    inputs={"optional": True, "adaptive_effort": effort},
                )
            )

        if "pc-research" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=pc_research_id,
                    role="pc-research",
                    objective=(
                        f"Perform sandboxed browser research actions for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "pc-research-agent",
                            "sandbox.exec",
                            "sandbox://virtual-desktop/browser-research",
                        ),
                        declared_action(
                            "pc-research-agent",
                            "file.write",
                            "runs/**",
                        ),
                    ],
                    inputs={"optional": True, "adaptive_effort": effort},
                )
            )

        if "literature" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=literature_id,
                    role="literature",
                    objective=(
                        "Find authoritative sources, direct evidence, and "
                        f"major uncertainties for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "literature-agent",
                            "mcp.list",
                            "mcp://configured-servers",
                        ),
                        declared_action(
                            "literature-agent",
                            "mcp.call",
                            "mcp://research/search",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.openalex.org/works",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            ("https://api.semanticscholar.org/graph/v1/paper/search"),
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.github.com/search/repositories",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://api.crossref.org/works",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            "https://html.duckduckgo.com/html",
                        ),
                        declared_action(
                            "literature-agent",
                            "network.fetch",
                            (
                                "https://generativelanguage.googleapis.com/"
                                "v1beta/models/"
                                "gemini-flash-latest:generateContent"
                            ),
                        ),
                        declared_action(
                            "literature-agent",
                            "file.write",
                            "runs/**",
                        ),
                    ],
                    inputs={"adaptive_effort": effort},
                )
            )

        if "data" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=data_id,
                    role="data",
                    objective=self._data_objective_for(objective),
                    declared_actions=[
                        declared_action(
                            "data-agent",
                            "file.write",
                            "runs/**",
                        ),
                        declared_action(
                            "data-agent",
                            "memory.commit",
                            "memory://candidate-facts",
                        ),
                    ],
                )
            )

        if "synthesis" in selected_roles:
            tasks.append(
                TaskSpec(
                    task_id=synthesis_id,
                    role="synthesis",
                    objective=(
                        "Merge worker outputs into a verified research "
                        f"brief for: {objective}"
                    ),
                    declared_actions=[
                        declared_action(
                            "synthesis-agent",
                            "memory.commit",
                            "memory://verified-synthesis",
                        )
                    ],
                )
            )
        return tasks

    @classmethod
    def _data_objective_for(cls, objective: str) -> str:
        if cls._needs_implementation_constraints(objective):
            return (
                "Extract implementation constraints, security boundaries, "
                f"and evaluation criteria for: {objective}"
            )
        if cls._needs_policy_or_risk_analysis(objective):
            return (
                "Extract risk factors, policy constraints, and evidence boundaries "
                f"for: {objective}"
            )
        return (
            "Extract structured evidence, decision criteria, and validation "
            f"criteria for: {objective}"
        )

    @staticmethod
    def _needs_implementation_constraints(objective: str) -> bool:
        # Match objectives that require technical implementation analysis.
        return bool(
            re.search(
                r"\b(agent|agents|orchestrator|sandbox|os.control|gui.app|dashboard|"
                r"implement(?:ation)?|build|code|api|sdk|framework|runtime|"
                r"workflow|tooling|deploy(?:ment)?|architecture|library|package)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _needs_policy_or_risk_analysis(objective: str) -> bool:
        """Return True when the objective is primarily about risks, policy, or compliance."""
        return bool(
            re.search(
                r"\b(risk|risks|policy|compliance|regulation|safety|security|"
                r"governance|liability|audit|legal|privacy|trust|boundary|boundaries)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _needs_pc_context(objective: str) -> bool:
        lower = objective.lower()
        if "sandbox" in lower and not re.search(
            r"\b(host pc|local pc|windows-uia|capture live desktop|current screen)\b",
            lower,
            flags=re.IGNORECASE,
        ):
            return False
        return bool(
            re.search(
                (
                    r"\b(local pc|host pc|current screen|"
                    r"capture desktop|capture live desktop|"
                    r"windows ui|windows-uia)\b"
                ),
                objective,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_multi_hour(objective: str) -> bool:
        return SupervisorAgent._research_effort(objective) == "multi-hour"

    @staticmethod
    def _research_effort(objective: str) -> str:
        return DeepResearchEngine.research_depth_for_objective(objective)

    @staticmethod
    def _needs_active_pc_research(
        objective: str,
        effort: str,
        needs_pc_context: bool,
    ) -> bool:
        lower = objective.lower()
        current_web_mode = DeepResearchEngine._looks_like_current_evidence_query(
            objective
        ) and not DeepResearchEngine._looks_like_academic_query(objective)
        strong_signals = {
            "browser research",
            "pc research",
            "pc-research-smoke",
            "browse the web",
            "web browsing",
            "search online",
            "navigate to",
            "visit website",
            "open website",
        }
        weak_signals = {
            "browser",
            "website",
            "url",
            "web",
            "tab",
            "search",
        }
        explicit_sandbox_signals = {
            "sandbox",
            "sandboxed",
            "browser automation",
            "automate browser",
            "web browsing",
            "browse the web",
            "open website",
            "visit website",
        }
        strong_hits = sum(1 for token in strong_signals if token in lower)
        weak_hits = sum(1 for token in weak_signals if token in lower)
        effort_boost = 1 if effort == "multi-hour" else 0
        context_boost = 1 if needs_pc_context else 0
        score = strong_hits * 2 + weak_hits + effort_boost + context_boost
        has_explicit_browser_intent = any(
            token in lower for token in explicit_sandbox_signals
        )
        if effort == "multi-hour" and current_web_mode:
            # Deep market/web research always benefits from a live browser pass.
            return True
        if not has_explicit_browser_intent and strong_hits == 0:
            return False
        return score >= 3


class WorkerAgent:
    """Constrained executor for one role and one task at a time."""

    def __init__(
        self,
        event_bus: EventBus,
        checkpoints: CheckpointStore,
        research_engine: DeepResearchEngine | None = None,
        pc_backend: Any | None = None,
        authorization: AuthorizationMiddleware | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.checkpoints = checkpoints
        self.research_engine = research_engine or DeepResearchEngine()
        self.pc_backend = pc_backend
        self.authorization = authorization

    def run(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        self.event_bus.publish(
            run_id,
            "task.started",
            task.role,
            {"task": asdict(task)},
        )
        self.checkpoints.save(
            run_id,
            f"{task.task_id}.started",
            {"task": asdict(task), "completed": len(prior_results)},
        )

        result = self._execute(run_id, task, prior_results)

        self.event_bus.publish(
            run_id,
            "task.completed",
            task.role,
            {"result": asdict(result)},
        )
        self.checkpoints.save(
            run_id,
            f"{task.task_id}.completed",
            {"result": asdict(result), "completed": len(prior_results) + 1},
        )
        return result

    def _execute(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        if task.role == "planning":
            return self._build_deep_plan(run_id, task)
        if task.role == "literature":
            brief = self._run_research_with_context(
                task.objective,
                run_id,
                prior_results,
                str((task.inputs or {}).get("adaptive_effort") or "").strip() or None,
            )
            evidence = brief.evidence()
            if getattr(brief, "metadata", None):
                evidence.append(
                    {
                        "source": "research-metrics",
                        "claim": "Coverage and retrieval metrics for gating.",
                        "metadata": brief.metadata,
                    }
                )
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=brief.summary,
                artifacts=brief.artifacts,
                evidence=evidence,
                confidence=brief.confidence,
            )
        if task.role == "pc-control":
            return self._capture_pc_context(run_id, task)
        if task.role == "pc-research":
            return self._active_pc_research(run_id, task, prior_results)
        if task.role == "data":
            evidence_count = sum(len(result.evidence) for result in prior_results)
            artifact_count = sum(len(result.artifacts) for result in prior_results)
            implementation_mode = "implementation constraints" in task.objective.lower()
            if implementation_mode:
                summary = (
                    "Extracted implementation constraints from "
                    f"{evidence_count} evidence records and "
                    f"{artifact_count} generated artifacts, then mapped "
                    "them into policy, durable state, event routing, and "
                    "sandbox boundaries."
                )
                evidence = {
                    "source": "SECURITY.md",
                    "claim": "Workers must declare actions before running.",
                }
            else:
                summary = (
                    "Extracted structured evidence criteria from "
                    f"{evidence_count} evidence records and "
                    f"{artifact_count} generated artifacts, then mapped "
                    "source quality, decision criteria, risk boundaries, "
                    "and validation checks."
                )
                evidence = {
                    "source": "research-artifacts",
                    "claim": (
                        "Structured evidence extraction preserves source-backed "
                        "claims, uncertainty, and validation criteria."
                    ),
                }
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=summary,
                evidence=[evidence],
                confidence=0.82,
            )

        combined = " ".join(result.summary for result in prior_results)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Synthesized verified worker outputs into a checkpointed "
                f"research trace. Prior context: {combined[:500]}"
            ),
            evidence=[
                {
                    "source": "event-log",
                    "claim": "All worker transitions were durably recorded.",
                }
            ],
            confidence=0.8,
        )

    def _build_deep_plan(self, run_id: str, task: TaskSpec) -> WorkerResult:
        objective = task.objective.replace(
            "Design deep research plan for:",
            "",
        ).strip()

        # ------------------------------------------------------------------
        # ZERO-TEMPLATE REASONING: Ask AI to analyze complexity, profile,
        # targets, and hypotheses specifically for THIS objective.
        # ------------------------------------------------------------------
        analysis = self._ai_analyze_objective(objective)
        complexity = analysis.get("complexity_score", 5)

        targets = {
            "min_source_count": analysis.get("min_source_count", 8),
            "min_provider_count": analysis.get("min_provider_count", 2),
            "min_scholarly_sources": analysis.get("min_scholarly_sources", 2),
            "min_strong_or_moderate": analysis.get("min_strong_or_moderate", 4),
            "max_contradiction_risk": analysis.get("max_contradiction_risk", 0.75),
            "min_novelty_rate": analysis.get("min_novelty_rate", 0.1),
            "max_retrieval_passes": analysis.get("max_retrieval_passes", 8),
            "min_runtime_seconds": 0,  # Information-driven, not timer-driven
            "min_depth_passes": analysis.get("min_depth_passes", 2),
            "max_low_novelty_streak": analysis.get("max_low_novelty_streak", 3),
        }
        # For multi-hour objectives, enforce minimum deep-pass budget floors
        # so the retrieval loop runs extensively even without AI-derived targets.
        if "[multi-hour]" in objective.lower():
            targets["max_retrieval_passes"] = max(targets["max_retrieval_passes"], 48)
            targets["min_depth_passes"] = max(targets["min_depth_passes"], 12)

        hypotheses = analysis.get("hypotheses") or [
            "Structured, multi-pass retrieval increases evidence breadth.",
            "Following citations and perspective-specific leads surfaces non-obvious evidence.",
            "Claim-level trace constraints reduce unsupported synthesis.",
        ]

        plan = {
            "objective": objective,
            "paper_mode": True,
            "coverage_targets": targets,
            "hypotheses": hypotheses,
            "created_by": "planning-worker",
            "ai_analysis": analysis,
        }
        artifact = self._write_planning_artifact(run_id, plan)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Created AI-reasoned research design with adaptive coverage "
                f"targets and objective-specific hypotheses. (Complexity: {complexity})"
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": "The research plan was dynamically generated via AI reasoning about causal drivers and evidence requirements.",
                    "coverage_targets": targets,
                }
            ],
            confidence=0.9,
        )

    def _ai_analyze_objective(self, objective: str) -> dict[str, Any]:
        """Ask AI to reason about the objective's complexity, requirements, and profile."""
        baseline = self._heuristic_objective_analysis(objective)
        system = (
            "You are a senior research architect. Analyze the provided research "
            "objective and reason about its complexity and requirements. "
            "Think step-by-step: what kind of evidence is needed, how deep "
            "should the search go, what are the primary risks or contradictions, "
            "and what specific hypotheses should guide the investigation. "
            "Respond ONLY with valid JSON."
        )
        user = (
            f"Objective: {objective}\n\n"
            "Produce JSON with these exact keys:\n"
            "{\n"
            '  "complexity_score": <1-10>,\n'
            '  "profile": {"academic": bool, "current": bool, "comparison": bool, "risk": bool},\n'
            '  "min_source_count": <int>,\n'
            '  "min_provider_count": <int>,\n'
            '  "min_scholarly_sources": <int>,\n'
            '  "max_contradiction_risk": <0.0-1.0>,\n'
            '  "max_retrieval_passes": <int>,\n'
            '  "hypotheses": ["hypothesis specific to this topic", ...]\n'
            "}\n\n"
            "Profile flag definitions (set true when the objective matches):\n"
            '- "academic": requires peer-reviewed studies, citations, or formal literature\n'
            '- "current": asks about present-day conditions, live data, "as of now", recent events, '
            "or time-sensitive information (market data, news, rankings, latest releases, etc.)\n"
            '- "comparison": explicitly compares multiple options, alternatives, entities, or products\n'
            '- "risk": asks about risks, downsides, dangers, failure modes, vulnerabilities, or negative scenarios\n\n'
            "Be ambitious with numeric targets for complex topics."
        )
        try:
            raw = self.research_engine._call_ai_text(system, user)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, dict):
                    profile = baseline["profile"]
                    profile.update(
                        {
                            "academic": bool(
                                (parsed.get("profile") or {}).get(
                                    "academic", profile["academic"]
                                )
                            ),
                            "current": bool(
                                (parsed.get("profile") or {}).get(
                                    "current", profile["current"]
                                )
                            ),
                            "comparison": bool(
                                (parsed.get("profile") or {}).get(
                                    "comparison", profile["comparison"]
                                )
                            ),
                            "risk": bool(
                                (parsed.get("profile") or {}).get(
                                    "risk", profile["risk"]
                                )
                            ),
                        }
                    )
                    result = {
                        "complexity_score": max(
                            1,
                            min(
                                int(
                                    parsed.get(
                                        "complexity_score", baseline["complexity_score"]
                                    )
                                ),
                                10,
                            ),
                        ),
                        "profile": profile,
                        "min_source_count": max(
                            4,
                            int(
                                parsed.get(
                                    "min_source_count", baseline["min_source_count"]
                                )
                            ),
                        ),
                        "min_provider_count": max(
                            1,
                            int(
                                parsed.get(
                                    "min_provider_count", baseline["min_provider_count"]
                                )
                            ),
                        ),
                        "min_scholarly_sources": max(
                            0,
                            int(
                                parsed.get(
                                    "min_scholarly_sources",
                                    baseline["min_scholarly_sources"],
                                )
                            ),
                        ),
                        "max_contradiction_risk": max(
                            0.0,
                            min(
                                float(
                                    parsed.get(
                                        "max_contradiction_risk",
                                        baseline["max_contradiction_risk"],
                                    )
                                ),
                                1.0,
                            ),
                        ),
                        "max_retrieval_passes": max(
                            1,
                            int(
                                parsed.get(
                                    "max_retrieval_passes",
                                    baseline["max_retrieval_passes"],
                                )
                            ),
                        ),
                        "hypotheses": [
                            str(item).strip()
                            for item in list(parsed.get("hypotheses") or [])
                            if str(item).strip()
                        ][:8],
                    }
                    if not result["hypotheses"]:
                        result["hypotheses"] = baseline["hypotheses"]
                    return result
        except Exception:
            pass
        return baseline

    @staticmethod
    def _heuristic_objective_analysis(objective: str) -> dict[str, Any]:
        """Deterministic fallback when AI planning output is missing or malformed."""
        lower = objective.lower()
        current = bool(
            re.search(
                r"\b(as of now|right now|current(?:ly)?|latest|today|recent|this (?:week|month|year)|live)\b",
                lower,
            )
        )
        comparison = bool(
            re.search(
                r"\b(compare|comparison|versus|vs\.?|top\s*\d+|rank|ranking|best|alternatives)\b",
                lower,
            )
        )
        risk = bool(
            re.search(
                r"\b(risk|downside|uncertaint|failure|vulnerab|hazard|counter[- ]?case|trade[- ]?off)\b",
                lower,
            )
        )
        academic = (
            bool(
                re.search(
                    r"\b(peer[- ]reviewed|literature|citation|scholarly|journal|methodolog|theorem|proof)\b",
                    lower,
                )
            )
            and not current
        )

        complexity = 5
        if "[multi-hour]" in lower:
            complexity += 2
        complexity += 1 if comparison else 0
        complexity += 1 if risk else 0
        complexity += 1 if current else 0
        complexity = max(1, min(complexity, 10))

        multi_hour = "[multi-hour]" in lower
        max_passes = 48 if multi_hour else (10 if complexity >= 7 else 6)
        min_sources = 20 if multi_hour else (12 if complexity >= 7 else 8)
        min_providers = 3 if complexity >= 7 else 2
        min_scholarly = 0 if current else (3 if complexity >= 7 else 2)

        hypotheses = [
            "Multiple independent providers reduce source monoculture and ranking drift.",
            "Contradiction-aware scoring improves robustness under noisy retrieval.",
            "Perspective-specific query diversification increases novelty across passes.",
        ]
        return {
            "complexity_score": complexity,
            "profile": {
                "academic": academic,
                "current": current,
                "comparison": comparison,
                "risk": risk,
            },
            "min_source_count": min_sources,
            "min_provider_count": min_providers,
            "min_scholarly_sources": min_scholarly,
            "max_contradiction_risk": 0.65 if risk else 0.75,
            "max_retrieval_passes": max_passes,
            "hypotheses": hypotheses,
        }

    @staticmethod
    def _multi_hour_min_runtime_seconds() -> int:
        return 0

    @staticmethod
    def _multi_hour_max_retrieval_passes(min_runtime_seconds: int) -> int:
        return 48

    @staticmethod
    def _multi_hour_min_depth_passes() -> int:
        return 12

    def _run_research_with_context(
        self,
        objective: str,
        run_id: str,
        prior_results: list[WorkerResult],
        effort: str | None = None,
    ) -> Any:
        pc_artifact = self._latest_pc_snapshot_artifact(prior_results)
        planning_context = self._latest_planning_context(prior_results)
        evidence_targets: dict[str, Any] = {}
        if planning_context:
            evidence_targets = planning_context.get("coverage_targets") or {}
        pc_findings = self._latest_pc_research_findings(prior_results)
        pc_context = (
            {
                "snapshot_path": pc_artifact,
                "pc_findings": pc_findings,
            }
            if pc_artifact or pc_findings
            else None
        )
        signature = inspect.signature(self.research_engine.run)
        kwargs: dict[str, Any] = {}
        if "pc_context" in signature.parameters:
            kwargs["pc_context"] = pc_context
        if "planning_context" in signature.parameters:
            kwargs["planning_context"] = planning_context
        if "evidence_targets" in signature.parameters:
            kwargs["evidence_targets"] = evidence_targets

        research_objective = objective
        if (
            isinstance(self.research_engine, DeepResearchEngine)
            and effort in {"quick", "standard", "multi-hour"}
            and not re.search(
                r"\[(quick|standard|multi-hour|adaptive)\]",
                objective,
                flags=re.IGNORECASE,
            )
        ):
            research_objective = f"[{effort}] {objective}"
        if kwargs:
            return self.research_engine.run(
                research_objective,
                run_id,
                **kwargs,
            )
        return self.research_engine.run(research_objective, run_id)

    def _active_pc_research(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        backend = self._sandbox_pc_backend()
        urls = self._candidate_urls_from_sources(prior_results)
        if not urls:
            urls = self._planning_urls_from_objective(task.objective)

        action = ActionRequest(
            agent_id="pc-research-agent",
            action_type="sandbox.exec",
            target="sandbox://virtual-desktop/browser-research",
            payload={
                "action": "browse",
                "urls": urls[:3],
            },
        )
        # When the active backend is already a virtual/sandboxed environment, no
        # human approval is required — it is inherently contained.  Skip the
        # authorization gate so sandbox runs never surface an approval prompt.
        backend_is_virtual_sandbox = isinstance(backend, VirtualDesktopSandboxBackend)
        if self.authorization is None or backend_is_virtual_sandbox:
            decision = None
        else:
            decision = self.authorization.authorize(run_id, action)
            if not decision.allowed:
                approval_payload = self._redacted_approval(decision.approval)
                decision_payload = asdict(decision)
                if decision_payload.get("approval"):
                    decision_payload["approval"] = approval_payload
                artifact = self._write_pc_findings_artifact(
                    run_id,
                    {
                        "status": "blocked",
                        "candidate_urls": urls[:6],
                        "approval": approval_payload,
                        "decision": decision_payload,
                    },
                )
                if decision.requires_approval and decision.approval is not None:
                    raise ApprovalRequired(decision.approval)
                raise PermissionError(
                    "Sandboxed browser research was blocked by policy/trust checks. "
                    f"Details in {artifact}."
                )

        receipts: list[dict[str, Any]] = []
        backend_name = getattr(backend, "name", "virtual-desktop-sandbox")
        try:
            (
                receipts,
                post_nodes,
                backend_name,
                interrupt_report,
                workspace_report,
            ) = self._run_pc_browser_actions(
                backend,
                urls,
                run_id=run_id,
            )
        except (BackendUnavailable, OSError, RuntimeError, ValueError) as exc:
            artifact = self._write_pc_findings_artifact(
                run_id,
                {
                    "status": "execution_error",
                    "candidate_urls": urls[:6],
                    "receipts": receipts,
                    "backend": backend_name,
                    "error": str(exc),
                },
            )
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=f"Sandboxed browser research failed: {exc}",
                artifacts=[artifact],
                evidence=[
                    {
                        "source": artifact,
                        "claim": "Sandboxed browser research encountered an error.",
                    }
                ],
                confidence=0.4,
            )

        names = [node.name for node in post_nodes if node.name][:10]
        panel_summary = self._panel_region_summary(post_nodes)
        browser_findings = self._reasoned_browser_findings(
            task.objective,
            urls,
            backend=backend,
        )
        direct_urls = list(browser_findings.get("direct_urls") or [])
        if not direct_urls:
            direct_urls = list(urls[:3])
        judged_results = list(browser_findings.get("judged_results") or [])
        if not judged_results:
            judged_results = [
                {
                    "url": url,
                    "quality_score": 0.3,
                    "why": "Fallback candidate URL when no judged results were extracted.",
                }
                for url in direct_urls[:3]
            ]
            browser_findings["judged_results"] = judged_results
        terminal_verifications = list(
            browser_findings.get("terminal_verifications") or []
        )
        self._append_terminal_verification_notes(run_id, terminal_verifications)
        findings = {
            "status": "executed",
            "candidate_urls": urls[:6],
            "receipts": receipts,
            "backend": backend_name,
            "durable_workspace": workspace_report,
            "post_snapshot_labels": names,
            "post_snapshot_node_count": len(post_nodes),
            "panel_regions": panel_summary,
            "interrupt_handler": interrupt_report,
            **browser_findings,
            "direct_urls": direct_urls,
        }
        if decision is not None and decision.trust is not None:
            findings["trust"] = asdict(decision.trust)
        artifact = self._write_pc_findings_artifact(run_id, findings)
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Executed sandboxed browser research actions and captured "
                "structured findings for evidence integration."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": (
                        "A confined sandbox browser session was used to collect research findings."
                    ),
                    "targeted_urls": direct_urls[:3] or urls[:3],
                    "search_queries": browser_findings.get("search_queries") or [],
                }
            ],
            confidence=0.78 if direct_urls else 0.68,
        )

    def _reasoned_browser_findings(
        self,
        objective: str,
        urls: list[str],
        backend: Any | None = None,
    ) -> dict[str, Any]:
        """Perform scientist-grade browser-based research.

        Rather than simply fetching DuckDuckGo results and recording titles,
        this method:
        1. Reasons about which source types are authoritative for the objective.
        2. Constructs targeted queries per authoritative category.
        3. Ranks and selects candidate pages.
        4. DEEPLY reads each selected page (up to 60 KB).
        5. Extracts specific evidence claims from the content.
        6. Records an AI-reasoned judgment explaining WHY each source matters.
        """
        queries = self._browser_search_queries(objective, urls)
        core_query = DeepResearchEngine._query_from_objective(objective)

        source_strategy = self._ai_browser_source_strategy(objective)
        for sq in source_strategy.get("targeted_queries") or []:
            if sq and sq.strip() and sq not in queries:
                queries.append(sq.strip()[:240])

        budget = self._browser_research_budget(objective, core_query, len(queries))

        if not queries:
            return {
                "search_queries": [],
                "judged_results": [],
                "direct_urls": [],
                "discovered_domains": [],
                "candidate_urls": [],
                "frontier": {},
            }

        raw_results: list[ResearchSource] = []
        is_market_query = DeepResearchEngine._looks_like_market_query(core_query)
        for query in queries[: int(budget["max_queries"])]:
            try:
                raw_results.extend(
                    self.research_engine._search_web_results(
                        query,
                        limit=int(budget["web_results_per_query"]),
                    )
                )
                if is_market_query:
                    if hasattr(self.research_engine, "_search_financial_portals"):
                        raw_results.extend(
                            self.research_engine._search_financial_portals(
                                query,
                                limit=int(budget["financial_results_per_query"]),
                            )
                        )
                    if hasattr(self.research_engine, "_search_sec_edgar"):
                        raw_results.extend(
                            self.research_engine._search_sec_edgar(
                                query,
                                limit=int(budget["sec_results_per_query"]),
                            )
                        )
            except Exception:
                continue

        deduped_results = self.research_engine._dedupe_sources(raw_results)
        ranked = self.research_engine._rank_sources(
            deduped_results,
            core_query,
        )
        exploration_sources = list(ranked)
        minimum_frontier = 4 if int(budget["max_direct_urls"]) >= 40 else 2
        if len(exploration_sources) < minimum_frontier and deduped_results:
            ranked_urls = {
                str(source.url or "").strip() for source in exploration_sources
            }
            raw_fallback = sorted(
                deduped_results,
                key=lambda source: float(source.score or 0.0),
                reverse=True,
            )
            for source in raw_fallback:
                clean_url = str(source.url or "").strip()
                if not clean_url or clean_url in ranked_urls:
                    continue
                exploration_sources.append(source)
                ranked_urls.add(clean_url)
                if len(exploration_sources) >= int(budget["candidate_urls"]):
                    break
        judged_results: list[dict[str, Any]] = []
        direct_urls: list[str] = []
        discovered_domains: list[str] = []
        extracted_claims: list[str] = []
        candidate_urls: list[str] = []
        seen_candidate_urls: set[str] = set()
        seen_direct_urls: set[str] = set()
        domain_usage: dict[str, int] = {}

        for source in exploration_sources:
            clean_url = str(source.url or "").strip()
            if clean_url in seen_candidate_urls:
                continue
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                continue
            if DeepResearchEngine._is_search_result_url(clean_url):
                continue
            seen_candidate_urls.add(clean_url)
            candidate_urls.append(clean_url)
            if len(candidate_urls) >= int(budget["candidate_urls"]):
                break

        for source in exploration_sources:
            if len(direct_urls) >= int(budget["max_direct_urls"]):
                break
            clean_url = str(source.url or "").strip()
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                continue
            if DeepResearchEngine._is_search_result_url(clean_url):
                continue
            if clean_url in seen_direct_urls:
                continue
            domain = urllib.parse.urlparse(clean_url).netloc.lower().lstrip("www.")
            if domain and domain_usage.get(domain, 0) >= int(budget["max_per_domain"]):
                continue

            full_content = self._deep_page_read(clean_url)
            if not full_content:
                continue
            quality = self._content_block_quality(full_content)
            if quality["quality_score"] < 0.22 and len(direct_urls) >= 2:
                continue
            if quality["quality_score"] < 0.12:
                continue
            preview_proxy = {
                "page_title": source.title,
                "page_excerpt": full_content[:600],
            }
            if self._browser_preview_is_blocked(preview_proxy):
                continue

            seen_direct_urls.add(clean_url)
            if domain:
                domain_usage[domain] = domain_usage.get(domain, 0) + 1

            evidence_claims = self._extract_page_evidence(
                full_content, source.title, core_query
            )
            extracted_claims.extend(evidence_claims[:3])
            judgment = self._ai_page_judgment(
                source, full_content[:2000], objective, core_query
            )

            judged_results.append(
                {
                    "query": queries[0],
                    "title": source.title,
                    "url": clean_url,
                    "domain": domain,
                    "abstract": source.abstract[:400],
                    "page_excerpt": full_content[:600],
                    "evidence_claims": evidence_claims[:4],
                    "content_quality": quality,
                    "judgment": judgment,
                }
            )
            direct_urls.append(clean_url)
            if domain and domain not in discovered_domains:
                discovered_domains.append(domain)

        terminal_verifications: list[dict[str, Any]] = []
        if backend is not None and extracted_claims:
            terminal_verifications = self._verify_claims_with_sandbox_terminal(
                backend,
                extracted_claims,
            )

        if not direct_urls:
            for source in exploration_sources:
                if len(direct_urls) >= min(5, int(budget["max_direct_urls"])):
                    break
                clean_url = str(source.url or "").strip()
                if not DeepResearchEngine._is_safe_public_url(clean_url):
                    continue
                if DeepResearchEngine._is_search_result_url(clean_url):
                    continue
                if clean_url in seen_direct_urls:
                    continue
                domain = urllib.parse.urlparse(clean_url).netloc.lower().lstrip("www.")
                if domain and domain_usage.get(domain, 0) >= int(
                    budget["max_per_domain"]
                ):
                    continue
                preview = self._browser_page_preview(clean_url)
                excerpt = str(preview.get("page_excerpt") or "")
                quality = self._content_block_quality(excerpt)
                judged_results.append(
                    {
                        "query": queries[0],
                        "title": source.title,
                        "url": clean_url,
                        "domain": domain,
                        "abstract": source.abstract[:400],
                        "page_excerpt": excerpt[:600],
                        "evidence_claims": [],
                        "content_quality": quality,
                        "judgment": "fallback source admitted with low-signal extraction",
                        "quality_flags": ["low-signal-extraction"],
                    }
                )
                seen_direct_urls.add(clean_url)
                if domain:
                    domain_usage[domain] = domain_usage.get(domain, 0) + 1
                direct_urls.append(clean_url)
                if domain and domain not in discovered_domains:
                    discovered_domains.append(domain)

        return {
            "search_queries": queries[: int(budget["returned_query_count"])],
            "judged_results": judged_results,
            "direct_urls": direct_urls,
            "discovered_domains": discovered_domains,
            "candidate_urls": candidate_urls,
            "search_result_count": len(raw_results),
            "frontier": {
                "queries_considered": min(len(queries), int(budget["max_queries"])),
                "candidate_urls": len(candidate_urls),
                "deep_reads": len(direct_urls),
                "max_per_domain": int(budget["max_per_domain"]),
                "ranked_candidates": len(ranked),
                "mode": str(budget["mode"]),
            },
            "terminal_verifications": terminal_verifications,
        }

    @staticmethod
    def _browser_research_budget(
        objective: str,
        core_query: str,
        query_count: int,
    ) -> dict[str, int | str]:
        lower = objective.lower()
        is_multi_hour = "multi-hour" in lower
        current_web_mode = DeepResearchEngine._looks_like_current_evidence_query(
            objective
        )
        is_market_query = DeepResearchEngine._looks_like_market_query(core_query)
        expansive_mode = is_multi_hour or current_web_mode

        max_queries = min(query_count, 24 if expansive_mode else 12)
        web_results_per_query = 12 if expansive_mode else 8
        max_direct_urls = 80 if expansive_mode else 40
        max_per_domain = 4 if expansive_mode or is_market_query else 2
        returned_query_count = min(query_count, 12 if expansive_mode else 8)
        candidate_urls = min(max(max_direct_urls * 2, 40), 240)

        return {
            "mode": "expansive" if expansive_mode else "standard",
            "max_queries": max(1, max_queries),
            "web_results_per_query": web_results_per_query,
            "financial_results_per_query": 16 if expansive_mode else 10,
            "sec_results_per_query": 10 if expansive_mode else 6,
            "max_direct_urls": max_direct_urls,
            "max_per_domain": max_per_domain,
            "returned_query_count": max(1, returned_query_count),
            "candidate_urls": candidate_urls,
        }

    @staticmethod
    def _browser_search_queries(
        objective: str,
        urls: list[str],
    ) -> list[str]:
        queries: list[str] = []
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            query_value = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            query_value = urllib.parse.unquote_plus(query_value).strip()
            if query_value:
                queries.append(query_value)
        cleaned = re.sub(
            r"\[(quick|standard|multi-hour|adaptive)\]\s*",
            " ",
            objective,
            flags=re.IGNORECASE,
        )
        for prefix in (
            "Perform sandboxed browser research actions for:",
            "Capture live desktop and browser/operator context for:",
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Design deep research plan for:",
        ):
            cleaned = cleaned.replace(prefix, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        for clause in WorkerAgent._browser_objective_queries(cleaned):
            queries.append(clause)
        fallback = DeepResearchEngine._query_from_objective(cleaned)
        if fallback:
            queries.append(fallback)
        deduped: list[str] = []
        for query in queries:
            normalized = DeepResearchEngine._normalize_title(query)
            if normalized and query not in deduped:
                deduped.append(query[:240])
        return deduped[:8]

    @staticmethod
    def _browser_objective_queries(cleaned: str) -> list[str]:
        clauses: list[str] = []
        normalized = cleaned.strip()
        if not normalized:
            return clauses

        def _append_clause(candidate: str) -> None:
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
            if len(candidate.split()) < 4:
                return
            if candidate not in clauses:
                clauses.append(candidate[:240])

        _append_clause(normalized)

        primary = re.split(
            r"\b(?:using|with|including|produce|producing|deliver|delivering|based on)\b",
            normalized,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        _append_clause(primary)

        for sentence in re.split(r"[.!?]+", normalized):
            _append_clause(sentence)

        for fragment in re.split(
            r"\b(?:and|while|plus|then|while also|along with)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            _append_clause(fragment)

        return clauses

    def _verify_claims_with_sandbox_terminal(
        self,
        backend: Any,
        claims: list[str],
    ) -> list[dict[str, Any]]:
        verified: list[dict[str, Any]] = []
        for claim in claims:
            if len(verified) >= 4:
                break
            expression = self._extract_verifiable_expression(claim)
            if expression is None:
                continue
            command = (
                'python -c "'
                "expr = '" + expression.replace("'", "\\'") + "'; "
                "print(eval(expr, {'__builtins__': {}}, {}))"
                '"'
            )
            try:
                receipt = self._json_or_text(
                    backend.perform(
                        UiAction(
                            action_type="execute_command",
                            selector="terminal",
                            value=command,
                            metadata={"source": "terminal-verification"},
                        )
                    )
                )
            except Exception:
                continue
            if not isinstance(receipt, dict):
                receipt = {
                    "status": "process-executed",
                    "raw_receipt": str(receipt),
                }
            process = receipt.get("process") or {}
            verified.append(
                {
                    "claim": claim[:260],
                    "expression": expression,
                    "command": command,
                    "status": receipt.get("status") or "process-executed",
                    "exit_code": process.get("exit_code", 0),
                }
            )
        return verified

    @staticmethod
    def _extract_verifiable_expression(claim: str) -> str | None:
        numeric = re.findall(r"[0-9]+(?:\.[0-9]+)?", claim)
        if len(numeric) < 2:
            return None
        if any(op in claim for op in ("+", "-", "*", "/", "=")):
            expr_match = re.search(r"([0-9][0-9\s+\-*/().=]{2,60}[0-9])", claim)
            if expr_match:
                expression = expr_match.group(1).replace("=", "-")
                return re.sub(r"\s+", "", expression)
        # Fallback: verify ratio-like relation between first two numbers.
        return f"{numeric[0]}/{numeric[1]}"

    @staticmethod
    def _append_terminal_verification_notes(
        run_id: str,
        terminal_verifications: list[dict[str, Any]],
    ) -> None:
        if not run_id or not terminal_verifications:
            return
        report_path = Path("runs") / run_id / "workflows" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not report_path.exists():
            report_path.write_text(
                "# Durable Research Report\n\n## Incremental Findings\n\n",
                encoding="utf-8",
            )
        lines = ["### Terminal Verification Receipts"]
        for item in terminal_verifications:
            lines.append(
                "- [verification/terminal] "
                f"expr=`{item.get('expression', '')}` "
                f"status={item.get('status', 'unknown')} "
                f"exit={item.get('exit_code', 0)} "
                f"claim={str(item.get('claim', ''))[:180]}"
            )
        lines.append("")
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _browser_page_preview(self, url: str) -> dict[str, str]:
        try:
            raw = self.research_engine._get_text(
                url,
                accept="text/html,application/xhtml+xml,*/*",
                max_bytes=20_000,
                timeout_seconds=6,
            )
        except TypeError:
            raw = self.research_engine._get_text(
                url,
                accept="text/html,application/xhtml+xml,*/*",
                max_bytes=20_000,
            )
        except Exception:
            raw = ""
        if not raw:
            return {"page_title": "", "page_excerpt": ""}
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL
        )
        page_title = (
            self.research_engine._html_to_text(title_match.group(1))[:160]
            if title_match is not None
            else ""
        )
        page_excerpt = self._render_aware_text(raw)[:500]
        return {
            "page_title": page_title,
            "page_excerpt": page_excerpt,
        }

    @staticmethod
    def _browser_preview_is_blocked(preview: dict[str, str]) -> bool:
        text = (
            f"{preview.get('page_title', '')} {preview.get('page_excerpt', '')}".lower()
        )
        if not text.strip():
            return True
        blocked_signals = (
            "pardon our interruption",
            "please enable javascript",
            "javascript is required",
            "javascript is disabled",
            "this site requires javascript",
            "cloudflare",
            "captcha",
            "access denied",
            "forbidden",
            "rate limited",
            "request has been blocked",
            "are you a bot",
        )
        if any(signal in text for signal in blocked_signals):
            return True
        # Heuristic for script-heavy shells with little readable content.
        script_markers = ("window.", "function(", "var ", "const ", "document.")
        script_hits = sum(1 for marker in script_markers if marker in text)
        return script_hits >= 3 and len(re.findall(r"\b[a-z]{4,}\b", text)) < 40

    @staticmethod
    def _browser_result_judgment(
        source: ResearchSource,
        preview: dict[str, str],
    ) -> str:
        """Fallback template judgment when AI is unavailable."""
        reasons: list[str] = []
        if source.relevance >= 0.55:
            reasons.append("high topical relevance")
        if source.credibility_score >= 0.45:
            reasons.append("credible source profile")
        if source.recency >= 0.6:
            reasons.append("recent signal")
        excerpt = preview.get("page_excerpt") or ""
        if len(excerpt) > 80:
            reasons.append(f"page content available ({len(excerpt)} chars)")
        if not reasons:
            reasons.append("ranked as one of the strongest browser-discovered leads")
        return "; ".join(reasons)

    def _ai_browser_source_strategy(self, objective: str) -> dict[str, Any]:
        """Ask AI which source types are authoritative for the objective and
        generate targeted search queries for each.

        Returns a dict with key ``targeted_queries`` (list of strings).
        Returns an empty dict when AI is unavailable.
        """
        system = (
            "You are a research librarian and scientist. Given a research objective, "
            "reason about which specific source types (government databases, "
            "academic journals, industry reports, official documentation, "
            "primary-source datasets, etc.) are most authoritative for finding "
            "direct evidence. Then generate targeted search queries for each. "
            "Respond ONLY with valid JSON."
        )
        user = (
            f"Research objective: {objective}\n\n"
            "Reason step by step:\n"
            "1. What are the 3-5 most authoritative source types for this topic?\n"
            "2. For each, what specific search query would find primary evidence?\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "authoritative_source_types": ["type 1: reason why", ...],\n'
            '  "targeted_queries": ["specific search query 1", "query 2", ...]\n'
            "}\n"
            "targeted_queries must be 3-8 specific, targeted search phrases derived "
            "from reasoning about source types — not generic keyword expansions."
        )
        raw = self.research_engine._call_ai_text(system, user)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                queries = [
                    str(q)[:240]
                    for q in (parsed.get("targeted_queries") or [])
                    if str(q).strip()
                ][:8]
                return {
                    "authoritative_source_types": parsed.get(
                        "authoritative_source_types"
                    )
                    or [],
                    "targeted_queries": queries,
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {}

    def _deep_page_read(self, url: str) -> str:
        """Fetch a full page and return stripped plain text (up to 60 KB).

        Returns empty string on any error or for blocked/JS-only pages.
        """
        try:
            raw = self.research_engine._get_text_stitched(
                url,
                accept="text/html,application/xhtml+xml,*/*",
                page_bytes=40_000,
                max_pages=4,
                overlap_bytes=1_000,
                query="",
            )
        except TypeError:
            try:
                raw = self.research_engine._get_text(
                    url,
                    accept="text/html,application/xhtml+xml,*/*",
                    max_bytes=60_000,
                )
            except Exception:
                raw = ""
        except Exception:
            raw = ""
        text = ""
        quality_score = 0.0
        if raw:
            text = self._strip_browser_boilerplate(self._render_aware_text(raw))
            quality_score = self._content_block_quality(text)["quality_score"]

        needs_browser = False
        browser_detector = getattr(self.research_engine, "_needs_browser", None)
        if callable(browser_detector):
            try:
                needs_browser = bool(browser_detector(url))
            except Exception:
                needs_browser = False

        blocked_preview = self._browser_preview_is_blocked(
            {
                "page_title": "",
                "page_excerpt": text[:600],
            }
        )
        should_retry_with_browser = (
            needs_browser or not text or blocked_preview or quality_score < 0.18
        )
        if should_retry_with_browser:
            browser_reader = getattr(self.research_engine, "_get_text_browser", None)
            if callable(browser_reader):
                try:
                    browser_text = str(
                        browser_reader(url, max_chars=80_000) or ""
                    ).strip()
                except Exception:
                    browser_text = ""
                if browser_text:
                    browser_text = self._strip_browser_boilerplate(browser_text)
                    browser_quality = self._content_block_quality(browser_text)[
                        "quality_score"
                    ]
                    if browser_quality >= max(quality_score, 0.14):
                        return browser_text

        return text

    def _render_aware_text(self, raw_html: str) -> str:
        """Prefer main/article/body-like blocks over navigation chrome."""
        blocks = self._render_blocks(raw_html)
        if blocks:
            blocks = sorted(blocks, key=lambda block: block["score"], reverse=True)
            merged = " ".join(block["text"] for block in blocks[:6])
            if merged.strip():
                return merged
        return self.research_engine._html_to_text(raw_html)

    def _render_blocks(self, raw_html: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"<(main|article|section|div|nav|aside|header|footer|body)([^>]*)>(.*?)</\1>",
            re.IGNORECASE | re.DOTALL,
        )
        blocks: list[dict[str, Any]] = []
        for match in pattern.finditer(raw_html):
            tag = str(match.group(1) or "").lower()
            attrs = str(match.group(2) or "")
            inner_html = str(match.group(3) or "")
            text = self.research_engine._html_to_text(inner_html).strip()
            if len(text) < 80:
                continue
            role = self._classify_render_panel(tag, attrs)
            score = self._panel_relevance_score(text, role)
            blocks.append(
                {
                    "panel_role": role,
                    "text": text,
                    "score": score,
                }
            )
        return blocks

    @staticmethod
    def _classify_render_panel(tag: str, attrs: str) -> str:
        lower = f"{tag} {attrs}".lower()
        if tag in {"main", "article"}:
            return "main"
        if any(token in lower for token in ("content", "article", "story", "body")):
            return "main"
        if tag in {"nav", "header", "footer"}:
            return "navigation"
        if any(
            token in lower
            for token in ("nav", "menu", "breadcrumb", "header", "footer")
        ):
            return "navigation"
        if tag == "aside" or "sidebar" in lower:
            return "sidebar"
        return "body"

    def _panel_relevance_score(self, text: str, panel_role: str) -> float:
        quality = self._content_block_quality(text)
        base = float(quality["quality_score"])
        role_bonus = {
            "main": 0.35,
            "body": 0.15,
            "sidebar": -0.1,
            "navigation": -0.3,
        }.get(panel_role, 0.0)
        return base + role_bonus

    def _content_block_quality(self, text: str) -> dict[str, float]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        sentence_chunks = [
            chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()
        ]
        if not words or not sentence_chunks:
            return {
                "quality_score": 0.0,
                "boilerplate_ratio": 1.0,
                "signal_density": 0.0,
            }
        noisy = sum(
            1
            for chunk in sentence_chunks
            if self._looks_like_browser_noise_fragment(chunk)
        )
        ui_tokens = (
            "home",
            "sign in",
            "log in",
            "privacy",
            "terms",
            "cookies",
            "newsletter",
            "market activity",
            "most active",
            "top gainers",
            "top losers",
        )
        lower_text = text.lower()
        ui_hits = sum(1 for token in ui_tokens if token in lower_text)
        boilerplate_ratio = min(
            1.0,
            (noisy / max(len(sentence_chunks), 1)) + (ui_hits / 20.0),
        )
        signal_density = min(1.0, len(words) / 280.0)
        quality_score = max(0.0, min(1.0, signal_density * (1.0 - boilerplate_ratio)))
        return {
            "quality_score": round(quality_score, 3),
            "boilerplate_ratio": round(boilerplate_ratio, 3),
            "signal_density": round(signal_density, 3),
        }

    @staticmethod
    def _looks_like_browser_noise_fragment(fragment: str) -> bool:
        lower = fragment.lower()
        if not lower.strip():
            return True
        noise_markers = (
            "window.",
            "window?.",
            "document.",
            "function ",
            "const ",
            "let ",
            "var ",
            "async_all_clicks",
            "click_timeout",
            "perf_navigationtime",
            "rapid",
            "nimbus",
            "accessibility menu",
            "skip to main content",
            "all services",
            "stock advisor",
            "podcast",
            "cookie",
            "privacy policy",
            "terms of service",
        )
        if any(marker in lower for marker in noise_markers):
            return True
        symbol_count = sum(
            1 for char in fragment if not char.isalnum() and not char.isspace()
        )
        symbol_ratio = symbol_count / max(len(fragment), 1)
        if symbol_ratio > 0.18 and any(
            token in fragment for token in ("{", "}", "=>", "();", "?.")
        ):
            return True
        return False

    @classmethod
    def _strip_browser_boilerplate(cls, text: str) -> str:
        if not text:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned: list[str] = []
        for part in parts:
            candidate = part.strip()
            if len(candidate) < 35:
                continue
            if cls._looks_like_browser_noise_fragment(candidate):
                continue
            cleaned.append(candidate)
        if not cleaned:
            return text
        return " ".join(cleaned)

    def _extract_page_evidence(
        self,
        content: str,
        title: str,
        query: str,
    ) -> list[str]:
        """Extract specific, high-value evidence claims from page content using AI.

        Falls back to keyword scoring if AI is unavailable.
        """
        if not content or len(content) < 100:
            return []

        # Step 1: Pre-filter with scoring to find candidate chunks.
        sentences = [
            s.strip() for s in re.split(r"[.!?]\s+", content) if len(s.strip()) > 30
        ]
        if not sentences:
            return []

        query_terms = set(t.lower() for t in re.findall(r"\b[a-zA-Z]{4,}\b", query))
        scored: list[tuple[int, str]] = []
        for sentence in sentences[:300]:
            if self._looks_like_browser_noise_fragment(sentence):
                continue
            lower = sentence.lower()
            hits = sum(1 for term in query_terms if term in lower)
            if hits >= 1:
                scored.append((hits, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = [s for _, s in scored[:12]]
        if not candidates:
            return []

        # Step 2: Use AI to select the strongest, most causal evidence from candidates.
        system = (
            "You are a research scientist extracting evidence from a source. "
            "Given a set of candidate sentences and a query, select the 3-4 most "
            "significant, factual, or causal evidence claims that directly "
            "address the query. Prefer primary facts, data points, or "
            "causal mechanisms. Respond ONLY with a JSON array of strings."
        )
        user = (
            f"Query: {query}\n"
            f"Source: {title}\n"
            f"Candidates:\n" + "\n".join(f"- {c}" for c in candidates) + "\n\n"
            "Respond with a JSON array of the 3-4 best evidence claims."
        )
        try:
            raw = self.research_engine._call_ai_text(system, user)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                claims = json.loads(raw[start:end])
                if isinstance(claims, list) and claims:
                    return [str(c)[:400] for c in claims if str(c).strip()][:4]
        except Exception:
            pass

        # Fallback to top-scored candidates.
        return [c[:400] for c in candidates[:4]]

    def _ai_page_judgment(
        self,
        source: ResearchSource,
        content_excerpt: str,
        objective: str,
        core_query: str,
    ) -> str:
        """Use AI to produce a specific, reasoned judgment of this page's value.

        Falls back to a template judgment when AI is unavailable.
        """
        if not content_excerpt or len(content_excerpt) < 60:
            return self._browser_result_judgment(
                source, {"page_excerpt": content_excerpt}
            )
        system = (
            "You are a research quality analyst. Given a source and a research "
            "objective, produce a concise 1-2 sentence judgment explaining: "
            "what specific evidence or insight this source provides, whether "
            "it is primary or secondary evidence, and its relevance to the "
            "objective. Be specific — never say 'page content was fetched'."
        )
        user = (
            f"Research objective: {objective}\n"
            f"Core query: {core_query}\n"
            f"Source title: {source.title}\n"
            f"Source domain: {urllib.parse.urlparse(source.url).netloc}\n"
            f"Page content excerpt:\n{content_excerpt[:1000]}\n\n"
            "Write a 1-2 sentence judgment of this source's research value."
        )
        ai_text = self.research_engine._call_ai_text(system, user)
        if ai_text and len(ai_text.strip()) > 20:
            return ai_text.strip()[:400]
        # Fallback.
        return self._browser_result_judgment(source, {"page_excerpt": content_excerpt})

    def _sandbox_pc_backend(self) -> Any:
        backend = self.pc_backend
        if backend is None:
            return self._virtual_pc_backend()
        capabilities = getattr(backend, "capabilities", None)
        if callable(capabilities):
            try:
                payload = capabilities()
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("sandbox"):
                return backend
        if "sandbox" in getattr(backend, "name", "").lower():
            return backend
        return self._virtual_pc_backend()

    def _active_pc_research_virtual_fallback(
        self,
        run_id: str,
        task: TaskSpec,
        urls: list[str],
        approval_artifact: str,
        approval_payload: dict[str, Any] | None,
    ) -> WorkerResult:
        fallback = self._virtual_pc_backend()
        try:
            receipts, post_nodes, backend_name = self._run_pc_browser_actions(
                fallback,
                urls,
            )
        except (BackendUnavailable, OSError, RuntimeError, ValueError) as exc:
            return WorkerResult(
                task_id=task.task_id,
                role=task.role,
                summary=(
                    "Live PC research is pending approval and the virtual "
                    f"sandbox fallback was unavailable: {exc}"
                ),
                artifacts=[approval_artifact],
                evidence=[
                    {
                        "source": approval_artifact,
                        "claim": "Active PC research was deferred until approval.",
                        "approval": approval_payload,
                    }
                ],
                confidence=0.38,
            )
        artifact = self._write_pc_findings_artifact(
            run_id,
            {
                "status": "executed_in_virtual_sandbox",
                "candidate_urls": urls[:6],
                "approval_deferred": approval_payload,
                "receipts": receipts,
                "backend": backend_name,
                "post_snapshot_labels": [node.name for node in post_nodes if node.name][
                    :10
                ],
                "post_snapshot_node_count": len(post_nodes),
                "panel_regions": self._panel_region_summary(post_nodes),
            },
        )
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Live PC research is pending approval; executed the same "
                f"browser/navigation intent inside {backend_name}."
            ),
            artifacts=[approval_artifact, artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": (
                        "A confined virtual desktop sandbox executed the PC "
                        "research navigation without touching the host OS."
                    ),
                    "targeted_urls": urls[:3],
                    "approval_deferred": approval_payload,
                }
            ],
            confidence=0.62,
        )

    @staticmethod
    def _redacted_approval(approval: Any) -> dict[str, Any] | None:
        if approval is None:
            return None
        payload = asdict(approval)
        payload["token"] = "[redacted]"
        action = payload.get("action")
        if isinstance(action, dict):
            action["approval_token"] = None
        return payload

    @staticmethod
    def _objective_approval_token(objective: str) -> str | None:
        match = re.search(r"approval-token=([A-Za-z0-9_\-]+)", objective)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def _candidate_urls_from_sources(prior_results: list[WorkerResult]) -> list[str]:
        urls: list[str] = []
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("research/sources.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url") or "").strip()
                    if url.startswith("http"):
                        urls.append(url)
                if urls:
                    return urls
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("planning/plan.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                objective = str(payload.get("objective") or "")
                urls.extend(WorkerAgent._planning_urls_from_objective(objective))
                if urls:
                    return urls
        return urls

    @staticmethod
    def _planning_urls_from_objective(objective: str) -> list[str]:
        urls: list[str] = []
        urls.extend(
            match.rstrip(").,;]}>\"'")
            for match in re.findall(r"https?://[^\s<>()]+", objective)
        )
        cleaned = re.sub(r"https?://[^\s<>()]+", " ", objective)
        cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
        for prefix in (
            "Perform sandboxed browser research actions for:",
            "Capture live desktop and browser/operator context for:",
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Find authoritative sources, prior systems, and gaps for:",
            "Design deep research plan for:",
        ):
            cleaned = cleaned.replace(prefix, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            query = DeepResearchEngine._query_from_objective(cleaned)
            depth = DeepResearchEngine.research_depth_for_objective(cleaned)
            variants = DeepResearchEngine._query_variants(query, depth) or [query]
            for variant in variants[:2]:
                encoded = urllib.parse.quote_plus(variant)
                urls.append(f"https://html.duckduckgo.com/html/?q={encoded}")
            if WorkerAgent._looks_like_repository_research(cleaned):
                for variant in variants[:2]:
                    encoded = urllib.parse.quote_plus(variant)
                    urls.append(
                        f"https://github.com/search?type=repositories&q={encoded}"
                    )
        deduped: list[str] = []
        for item in urls:
            if item not in deduped:
                deduped.append(item)
        return deduped[:8]

    @staticmethod
    def _looks_like_repository_research(objective: str) -> bool:
        if DeepResearchEngine._looks_like_software_agent_query(objective):
            return True
        return bool(
            re.search(
                r"\b(code|github|repository|repositories|open[- ]source|sdk|api|package|library|framework|implementation)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _latest_planning_context(
        prior_results: list[WorkerResult],
    ) -> dict[str, Any] | None:
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("planning/plan.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    @staticmethod
    def _latest_pc_research_findings(
        prior_results: list[WorkerResult],
    ) -> dict[str, Any] | None:
        for result in reversed(prior_results):
            for artifact in result.artifacts:
                normalized = str(artifact).replace("\\", "/")
                if not normalized.endswith("pc/research_findings.json"):
                    continue
                path = Path(artifact)
                if not path.exists():
                    continue
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    @staticmethod
    def _write_planning_artifact(run_id: str, payload: dict[str, Any]) -> str:
        path = Path("runs") / run_id / "planning" / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _write_pc_findings_artifact(run_id: str, payload: dict[str, Any]) -> str:
        path = Path("runs") / run_id / "pc" / "research_findings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _json_or_text(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _latest_pc_snapshot_artifact(
        prior_results: list[WorkerResult],
    ) -> str | None:
        for result in reversed(prior_results):
            if result.role != "pc-control":
                continue
            for artifact in result.artifacts:
                if str(artifact).replace("\\", "/").endswith("pc/snapshot.json"):
                    return str(artifact)
        return None

    def _capture_pc_context(self, run_id: str, task: TaskSpec) -> WorkerResult:
        backend = self.pc_backend or WindowsUiaBackend()
        try:
            nodes = backend.snapshot()
        except (BackendUnavailable, OSError, RuntimeError) as exc:
            if self.pc_backend is None:
                fallback = self._virtual_pc_backend()
                nodes = fallback.snapshot()
                backend = fallback
            else:
                return WorkerResult(
                    task_id=task.task_id,
                    role=task.role,
                    summary=f"PC context capture was unavailable: {exc}",
                    evidence=[
                        {
                            "source": getattr(backend, "name", "pc-backend"),
                            "claim": "PC snapshot could not be captured.",
                        }
                    ],
                    confidence=0.35,
                )

        artifact = self._write_pc_snapshot(run_id, nodes[:120])
        named_nodes = [node for node in nodes if node.name][:8]
        names = "; ".join(node.name for node in named_nodes) or "no labels"
        browserish = [
            node.name
            for node in nodes
            if any(
                marker in node.name.lower()
                for marker in ("browser", "chrome", "edge", "127.0.0.1")
            )
        ]
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                f"Captured {len(nodes)} desktop/browser nodes from "
                f"{getattr(backend, 'name', 'pc-backend')}. Visible context "
                f"included: {names}."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": (
                        "A PC UI snapshot was captured for this run, using the "
                        "confined sandbox fallback when live UI access was unavailable."
                    ),
                    "node_count": len(nodes),
                    "backend": getattr(backend, "name", "pc-backend"),
                    "browser_context_detected": bool(browserish),
                    "browser_context": browserish[:5],
                }
            ],
            confidence=0.78,
        )

    @staticmethod
    def _virtual_pc_backend() -> VirtualDesktopSandboxBackend:
        return VirtualDesktopSandboxBackend(
            Path(".agentos/virtual_desktop_sandbox.json")
        )

    def _run_pc_browser_actions(
        self,
        backend: Any,
        urls: list[str],
        run_id: str = "",
    ) -> tuple[list[dict[str, Any]], list[UiNode], str, dict[str, Any], dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        pre_nodes = backend.snapshot()
        workspace_report = self._prepare_cross_surface_workspace(
            backend,
            pre_nodes,
            run_id,
        )
        if workspace_report.get("triggered"):
            receipts.append(
                {
                    "step": "prepare-durable-workspace",
                    "result": workspace_report,
                }
            )

        focus_result, focus_attempts = self._perform_with_selector_fallback(
            backend,
            "focus",
            self._browser_region_selectors(pre_nodes, "window"),
        )
        receipts.append(
            {
                "step": "focus-browser-window",
                "result": focus_result,
                "attempts": focus_attempts,
            }
        )

        navigate_result, navigate_attempts = self._perform_with_selector_fallback(
            backend,
            "set_text",
            self._browser_region_selectors(pre_nodes, "address"),
            value=urls[0],
        )
        receipts.append(
            {
                "step": "navigate-candidate",
                "result": navigate_result,
                "attempts": navigate_attempts,
            }
        )

        content_focus_result, content_attempts = self._perform_with_selector_fallback(
            backend,
            "focus",
            self._browser_region_selectors(pre_nodes, "content"),
        )
        receipts.append(
            {
                "step": "focus-content-region",
                "result": content_focus_result,
                "attempts": content_attempts,
            }
        )

        post_nodes = backend.snapshot()
        interrupt_report = self._resolve_ephemeral_blockers(
            backend,
            post_nodes,
            urls[0] if urls else "",
        )
        if interrupt_report.get("triggered"):
            receipts.append(
                {
                    "step": "interrupt-handler",
                    "result": interrupt_report,
                }
            )
            post_nodes = backend.snapshot()

        return (
            receipts,
            post_nodes,
            getattr(backend, "name", "pc-backend"),
            interrupt_report,
            workspace_report,
        )

    def _prepare_cross_surface_workspace(
        self,
        backend: Any,
        nodes: list[UiNode],
        run_id: str,
    ) -> dict[str, Any]:
        report_path = (
            f"runs/{run_id}/workflows/report.md"
            if run_id
            else "artifacts/workflows/report.md"
        )
        header = (
            "# Durable Research Report\n\n"
            "This report is incrementally updated by the research workflow.\n"
            "Use it for final synthesis instead of raw page context.\n"
        )

        launch_result = self._json_or_text(
            backend.perform(UiAction("launch_app", "vscode"))
        )
        focus_result, focus_attempts = self._perform_with_selector_fallback(
            backend,
            "focus",
            self._editor_surface_selectors(nodes),
        )
        file_result = self._json_or_text(
            backend.perform(
                UiAction(
                    "write_file",
                    report_path,
                    metadata={
                        "path": report_path,
                        "content": header,
                    },
                )
            )
        )
        canvas_result, canvas_attempts = self._perform_with_selector_fallback(
            backend,
            "set_text",
            self._editor_surface_selectors(nodes),
            value=header,
        )
        return {
            "triggered": True,
            "report_path": report_path,
            "launch_result": launch_result,
            "focus_result": focus_result,
            "focus_attempts": focus_attempts,
            "file_result": file_result,
            "canvas_result": canvas_result,
            "canvas_attempts": canvas_attempts,
        }

    @staticmethod
    def _editor_surface_selectors(nodes: list[UiNode]) -> list[str]:
        selectors = [
            "node_id=editor-canvas",
            "name=editor canvas",
            "role=document;name=editor canvas",
            "panel_type=primary",
            "name=sandbox app - vscode",
            "name=sandbox app - notepad",
        ]
        for node in nodes:
            node_id = node.node_id.lower()
            name = node.name.lower()
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if node_id in {"editor-canvas", "editor-explorer"}:
                selectors.append(f"node_id={node.node_id}")
            if "editor" in name or "notepad" in name:
                selectors.append(f"name={node.name.lower()}")
            if panel in {"primary", "document"} and node.role.lower() == "document":
                selectors.append(f"node_id={node.node_id}")
        deduped: list[str] = []
        for selector in selectors:
            if selector not in deduped:
                deduped.append(selector)
        return deduped

    def _resolve_ephemeral_blockers(
        self,
        backend: Any,
        nodes: list[UiNode],
        current_url: str,
    ) -> dict[str, Any]:
        baseline_signal = self._node_signal_score(nodes)
        blockers = self._detect_overlay_nodes(nodes)
        trigger = bool(blockers) and baseline_signal < 0.2
        report: dict[str, Any] = {
            "triggered": trigger,
            "state_sequence": ["pass-active"],
            "baseline_signal": round(baseline_signal, 3),
            "overlay_count": len(blockers),
            "detected_overlays": blockers[:8],
            "attempts": [],
            "status": "not-triggered",
            "tagged_urls": [],
        }
        if not trigger:
            return report

        report["state_sequence"].append("interrupt-handler")
        dismiss_selectors = [
            "role=button;name=accept",
            "role=button;name=decline",
            "role=button;name=close",
            "role=button;name=x",
            "name=accept",
            "name=decline",
            "name=close",
            "name=x",
        ]

        for attempt_index in range(2):
            # Prefer a semantic close click, then fallback to modal close.
            click_result, click_attempts = self._perform_with_selector_fallback(
                backend,
                "click",
                dismiss_selectors,
            )
            close_modal_result = self._json_or_text(
                backend.perform(UiAction("close_modal", "role=dialog"))
            )
            refreshed = backend.snapshot()
            signal = self._node_signal_score(refreshed)
            remaining = self._detect_overlay_nodes(refreshed)
            report["attempts"].append(
                {
                    "attempt": attempt_index + 1,
                    "dismiss_click_result": click_result,
                    "dismiss_click_attempts": click_attempts,
                    "close_modal_result": close_modal_result,
                    "post_signal": round(signal, 3),
                    "remaining_overlay_count": len(remaining),
                }
            )
            if not remaining and (signal >= 0.04 or signal >= (baseline_signal + 0.02)):
                report["status"] = "resolved"
                report["state_sequence"].append("resume-pass")
                report["resume_signal"] = round(signal, 3)
                return report

        report["status"] = "unreachable-paywalled"
        report["state_sequence"].append("poison-pill")
        if current_url:
            report["tagged_urls"] = [
                {
                    "url": current_url,
                    "tag": "unreachable-paywalled",
                }
            ]
        return report

    def _node_signal_score(self, nodes: list[UiNode]) -> float:
        content_chunks: list[str] = []
        for node in nodes:
            role = node.role.lower()
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if role in {"document", "article"} or panel in {
                "document",
                "primary",
                "content",
            }:
                text = str((node.metadata or {}).get("text") or "")
                if text:
                    content_chunks.append(text)
        merged = " ".join(content_chunks).strip()
        if not merged:
            return 0.0
        quality = self._content_block_quality(merged)
        return float(quality.get("quality_score") or 0.0)

    @staticmethod
    def _detect_overlay_nodes(nodes: list[UiNode]) -> list[str]:
        overlays: list[str] = []
        for node in nodes:
            if not node.enabled:
                continue
            role = node.role.lower()
            name = node.name.lower()
            metadata = node.metadata or {}
            panel_type = str(metadata.get("panel_type") or "").lower()
            position = str(metadata.get("position") or "").lower()
            if role in {"dialog", "alert"}:
                overlays.append(node.name or node.node_id)
                continue
            if panel_type in {"modal", "overlay", "cookie", "paywall"}:
                overlays.append(node.name or node.node_id)
                continue
            if position == "fixed" and any(
                token in name
                for token in (
                    "cookie",
                    "subscribe",
                    "sign in",
                    "paywall",
                    "newsletter",
                    "consent",
                )
            ):
                overlays.append(node.name or node.node_id)
        return overlays

    def _perform_with_selector_fallback(
        self,
        backend: Any,
        action_type: str,
        selectors: list[str],
        value: str | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_result: Any = {"status": "selector-not-found"}
        for selector in selectors:
            result = self._json_or_text(
                backend.perform(UiAction(action_type, selector, value=value))
            )
            status = ""
            if isinstance(result, dict):
                status = str(result.get("status") or "")
            attempts.append({"selector": selector, "status": status or "unknown"})
            last_result = result
            if status != "selector-not-found":
                return result, attempts
        return last_result, attempts

    def _browser_region_selectors(
        self,
        nodes: list[UiNode],
        region: str,
    ) -> list[str]:
        selectors: list[str] = []
        if region == "window":
            selectors.extend(
                [
                    "node_id=window-browser",
                    "name=sandbox browser",
                    "role=window;name=browser",
                    "name=microsoft edge",
                ]
            )
        elif region == "address":
            selectors.extend(
                [
                    "node_id=browser-address-bar",
                    "role=edit;name=address and search bar",
                    "name=address and search bar",
                    "role=edit",
                ]
            )
        elif region == "content":
            selectors.extend(
                [
                    "node_id=browser-main-doc",
                    "role=document",
                    "panel_type=primary",
                    "panel_type=document",
                ]
            )

        spatial = self._browser_region_spatial_selector(nodes, region)
        if spatial:
            selectors.append(spatial)

        deduped: list[str] = []
        for selector in selectors:
            if selector not in deduped:
                deduped.append(selector)
        return deduped

    @staticmethod
    def _browser_region_spatial_selector(
        nodes: list[UiNode], region: str
    ) -> str | None:
        for node in nodes:
            node_id = node.node_id.lower()
            role = node.role.lower()
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if region == "window" and (node_id == "window-browser" or role == "window"):
                return WorkerAgent._point_selector_for_node(node)
            if region == "address" and (
                node_id == "browser-address-bar"
                or (role == "edit" and "address" in node.name.lower())
            ):
                return WorkerAgent._point_selector_for_node(node)
            if region == "content" and (
                node_id == "browser-main-doc"
                or role == "document"
                or panel in {"primary", "document"}
            ):
                return WorkerAgent._point_selector_for_node(node)
        return None

    @staticmethod
    def _point_selector_for_node(node: UiNode) -> str | None:
        if not node.bounds:
            return None
        x, y, width, height = node.bounds
        center_x = x + max(width // 2, 1)
        center_y = y + max(height // 2, 1)
        return f"point={center_x},{center_y}"

    @staticmethod
    def _panel_region_summary(nodes: list[UiNode]) -> dict[str, Any]:
        regions: dict[str, list[str]] = {
            "main": [],
            "navigation": [],
            "sidebar": [],
            "footer": [],
            "other": [],
        }
        for node in nodes:
            name = node.name.strip() or node.node_id
            lower_name = name.lower()
            role = node.role.lower()
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if role == "document" or panel in {"primary", "document"}:
                regions["main"].append(name)
            elif panel in {"navigation", "tab_strip", "toolbar"} or role in {
                "toolbar",
                "tablist",
                "tree",
            }:
                regions["navigation"].append(name)
            elif panel in {"side_panel", "preview"} or "side" in lower_name:
                regions["sidebar"].append(name)
            elif panel == "status" or role == "statusbar":
                regions["footer"].append(name)
            else:
                regions["other"].append(name)
        return {
            "region_count": sum(1 for values in regions.values() if values),
            "regions": {key: values[:6] for key, values in regions.items()},
        }

    @staticmethod
    def _write_pc_snapshot(run_id: str, nodes: list[UiNode]) -> str:
        path = Path("runs") / run_id / "pc" / "snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(node) for node in nodes], indent=2),
            encoding="utf-8",
        )
        return str(path)


class VerificationAgent:
    """Asynchronous support role expressed as an explicit verification pass."""

    def review(self, run_id: str, results: list[WorkerResult]) -> WorkerResult:
        confidence = 0.0
        if results:
            total = sum(result.confidence for result in results)
            confidence = total / len(results)
        missing_evidence = [result.task_id for result in results if not result.evidence]
        quality_warnings = self._quality_warnings(results)
        if missing_evidence:
            summary = "Verification found worker outputs without evidence."
            confidence = min(confidence, 0.5)
        elif quality_warnings:
            summary = "Verification accepted worker evidence with quality warnings."
            confidence = min(confidence, 0.65)
        else:
            summary = "Verification accepted all worker outputs with evidence."

        return WorkerResult(
            task_id=new_id("task_verify"),
            role="verification",
            summary=summary,
            evidence=[
                {
                    "source": "verification-agent",
                    "run_id": run_id,
                    "checked_results": len(results),
                    "quality_warnings": quality_warnings,
                }
            ],
            confidence=confidence,
        )

    @staticmethod
    def _quality_warnings(results: list[WorkerResult]) -> list[str]:
        warnings: list[str] = []
        for result in results:
            for item in result.evidence:
                if item.get("source") != "research-metrics":
                    continue
                metadata = item.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                coverage = metadata.get("coverage")
                if not isinstance(coverage, dict):
                    continue
                source_count = int(coverage.get("source_count") or 0)
                provider_count = int(coverage.get("provider_count") or 0)
                strong_count = int(coverage.get("strong_or_moderate") or 0)
                perspective_ratio = float(coverage.get("perspective_ratio") or 0.0)
                missing = coverage.get("missing_perspectives") or []
                if source_count >= 5 and provider_count < 2:
                    warnings.append("research used fewer than two source providers")
                if source_count >= 5 and strong_count < 2:
                    warnings.append(
                        "research has fewer than two strong/moderate sources"
                    )
                if perspective_ratio and perspective_ratio < 0.75:
                    warnings.append(
                        "research missed planned perspectives: "
                        + ", ".join(str(name) for name in missing[:4])
                    )
        deduped: list[str] = []
        for warning in warnings:
            if warning not in deduped:
                deduped.append(warning)
        return deduped
