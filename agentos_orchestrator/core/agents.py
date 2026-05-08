from __future__ import annotations

import io
import inspect
import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .authorization import AuthorizationMiddleware
from .approvals import ApprovalRequired
from .checkpoint import CheckpointStore
from .events import EventBus
from .objective_analysis import ObjectiveAnalysisMixin
from .types import ActionRequest, TaskSpec, WorkerResult, new_id
from agentos_orchestrator.cognition.frontier_api import (
    FrontierClient,
    FrontierPrompt,
    default_provider_from_env,
)
from agentos_orchestrator.os_control import WindowsUiaBackend
from agentos_orchestrator.os_control.base import (
    BackendUnavailable,
    UiAction,
    UiNode,
)
from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
    VirtualDesktopSandboxBackend,
)
from agentos_orchestrator.os_control.workflow.service import DesktopWorkflowService
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

    def _normalize_selected_roles(
        self,
        proposed_roles: list[str],
        objective: str,
        effort: str,
        needs_pc_context: bool,
    ) -> list[str]:
        allowed = {
            "planning",
            "pc-control",
            "pc-research",
            "literature",
            "data",
            "synthesis",
        }
        roles: list[str] = []
        for role in proposed_roles:
            if role in allowed and role not in roles:
                roles.append(role)

        def _insert_before(anchor: str, role: str) -> None:
            if role in roles:
                return
            try:
                index = roles.index(anchor)
            except ValueError:
                roles.append(role)
            else:
                roles.insert(index, role)

        if self._needs_planning_role(objective, effort, needs_pc_context):
            if "planning" not in roles:
                roles.insert(0, "planning")
        if needs_pc_context and "pc-control" not in roles:
            _insert_before("literature", "pc-control")
        if self._needs_active_pc_research(objective, effort, needs_pc_context):
            _insert_before("literature", "pc-research")
        for role in ("literature", "data", "synthesis"):
            if role not in roles:
                roles.append(role)
        return roles

    def _adaptive_plan_roles(
        self, objective: str, effort: str, needs_pc_context: bool
    ) -> list[str]:
        system = (
            "You are an adaptive task planner. Analyze the research objective and "
            "determine which agent roles are needed to accomplish it. The available "
            "roles are: 'planning' (for complex strategy), 'pc-control' "
            "(if pulling live host data, interacting with a local application, or running a local script), "
            "'pc-research' (for sandbox browser and terminal research whenever live websites, official docs, "
            "vendor pages, product pages, rankings, current web evidence, or direct browser exploration would "
            "materially improve coverage or reduce source monoculture), "
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
                    return self._normalize_selected_roles(
                        list(roles),
                        objective,
                        effort,
                        needs_pc_context,
                    )
        except Exception as _role_exc:
            import warnings
            warnings.warn(
                f"AI role selection failed ({_role_exc!r}); falling back to "
                "heuristic role assignment. Check AI API connectivity.",
                RuntimeWarning,
                stacklevel=3,
            )

        roles = []
        if self._needs_planning_role(objective, effort, needs_pc_context):
            roles.append("planning")
        if needs_pc_context:
            roles.append("pc-control")
        if self._needs_active_pc_research(objective, effort, needs_pc_context):
            roles.append("pc-research")
        roles.extend(["literature", "data", "synthesis"])
        return self._normalize_selected_roles(
            roles,
            objective,
            effort,
            needs_pc_context,
        )

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

    def _needs_planning_role(
        self,
        objective: str,
        effort: str,
        needs_pc_context: bool,
    ) -> bool:
        if effort == "multi-hour" or needs_pc_context:
            return True
        if self._needs_active_pc_research(objective, effort, needs_pc_context):
            return True
        analysis = WorkerAgent._heuristic_objective_analysis(objective)
        profile = analysis.get("profile") or {}
        try:
            complexity = int(analysis.get("complexity_score") or 0)
        except (TypeError, ValueError):
            complexity = 0
        return complexity >= 7 and any(
            bool(profile.get(key)) for key in ("comparison", "risk", "current")
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
    def _has_explicit_current_web_cues(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(as of now|right now|current(?:ly)?|latest|today|recent|near-term|newest|this week|this month|this year|live|breaking|ongoing)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _needs_active_pc_research(
        objective: str,
        effort: str,
        needs_pc_context: bool,
    ) -> bool:
        score = SupervisorAgent._pc_research_signal_score(
            objective,
            effort,
            needs_pc_context,
        )
        if score >= 3:
            return True
        lower = objective.lower()
        current_web_mode = SupervisorAgent._has_explicit_current_web_cues(
            objective
        ) and not DeepResearchEngine._looks_like_academic_query(objective)
        comparison_mode = bool(
            re.search(
                r"\b(compare|comparison|versus|vs\.?|rank|ranking|best|alternatives)\b",
                lower,
            )
        )
        risk_mode = bool(
            re.search(
                r"\b(risk|risks|downside|uncertaint|failure|vulnerab|hazard|trade[- ]?off)\b",
                lower,
            )
        )
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
        has_explicit_browser_intent = any(
            token in lower for token in explicit_sandbox_signals
        )
        if current_web_mode:
            # Time-sensitive web research should always get an active browser pass.
            return True
        if comparison_mode and not DeepResearchEngine._looks_like_academic_query(
            objective
        ):
            return True
        if (
            effort == "multi-hour"
            and risk_mode
            and not DeepResearchEngine._looks_like_academic_query(objective)
        ):
            return True
        if has_explicit_browser_intent and (strong_hits > 0 or weak_hits > 0):
            return True
        return False

    @staticmethod
    def _pc_research_signal_score(
        objective: str,
        effort: str,
        needs_pc_context: bool,
    ) -> int:
        lower = objective.lower()
        analysis = WorkerAgent._heuristic_objective_analysis(objective)
        profile = analysis.get("profile") or {}
        try:
            complexity = int(analysis.get("complexity_score") or 0)
        except (TypeError, ValueError):
            complexity = 0

        score = 0
        if effort == "multi-hour":
            score += 3
        if needs_pc_context:
            score += 2
        if complexity >= 7:
            score += 2
        elif complexity >= 5:
            score += 1
        if bool(profile.get("comparison")):
            score += 2
        if bool(profile.get("risk")):
            score += 2
        if not bool(profile.get("academic")):
            score += 1
        if SupervisorAgent._needs_implementation_constraints(objective):
            score += 1
        if DeepResearchEngine._looks_like_software_agent_query(objective):
            score += 3
        if re.search(
            r"\b(evidence|source-backed|sources|official|documentation|docs|vendor|release notes|benchmark|report|filing|specification|website|web data|primary source)\b",
            lower,
            flags=re.IGNORECASE,
        ):
            score += 1
        if re.search(
            r"\b(sandbox|sandboxed|browser automation|browse the web|search online|open website|visit website|navigate to)\b",
            lower,
            flags=re.IGNORECASE,
        ):
            score += 2
        return score


class WorkerAgent(ObjectiveAnalysisMixin):
    """Constrained executor for one role and one task at a time."""

    def __init__(
        self,
        event_bus: EventBus,
        checkpoints: CheckpointStore,
        research_engine: DeepResearchEngine | None = None,
        pc_backend: Any | None = None,
        authorization: AuthorizationMiddleware | None = None,
        frontier_client: FrontierClient | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.checkpoints = checkpoints
        self.research_engine = research_engine or DeepResearchEngine()
        self.pc_backend = pc_backend
        self.authorization = authorization
        self.frontier_client = (
            frontier_client
            if frontier_client is not None
            else default_provider_from_env()
        )

    def available_capabilities(self) -> set[str]:
        capabilities = {
            "planning",
            "literature",
            "data-extraction",
            "synthesis",
            "pc-control",
            "pc-research",
            "sandbox.exec",
        }
        backend = self.pc_backend
        if backend is None:
            return capabilities
        backend_capabilities = getattr(backend, "capabilities", None)
        if callable(backend_capabilities):
            try:
                payload = backend_capabilities()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                if payload.get("sandbox"):
                    capabilities.add("sandbox.exec")
                for item in payload.get("capabilities") or []:
                    if str(item).strip():
                        capabilities.add(str(item).strip())
        backend_name = str(getattr(backend, "name", "") or "").lower()
        if "sandbox" in backend_name:
            capabilities.add("sandbox.exec")
        return capabilities

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
            mcp_metadata = dict((getattr(brief, "metadata", {}) or {}).get("mcp") or {})
            for action in list(mcp_metadata.get("actions") or []):
                evidence.append(
                    {
                        "source": str(action.get("target") or "mcp://research"),
                        "claim": "MCP research action executed during literature retrieval.",
                        "metadata": action,
                    }
                )
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

        hypotheses = list(analysis.get("hypotheses") or [])
        if not hypotheses:
            if not self._heuristic_planning_allowed():
                raise RuntimeError(
                    "AI planning returned no hypotheses and heuristic fallback "
                    "is disabled. Set AGENTOS_ALLOW_HEURISTIC_PLANNING=1 to "
                    "permit deterministic planning fallbacks."
                )
            hypotheses = self._fallback_plan_hypotheses(objective)
        browser_research = self._planning_browser_research(objective, analysis)

        plan = {
            "objective": objective,
            "paper_mode": True,
            "coverage_targets": targets,
            "hypotheses": hypotheses,
            "created_by": "planning-worker",
            "ai_analysis": analysis,
            "browser_research": browser_research,
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

    def _fallback_plan_hypotheses(self, objective: str) -> list[str]:
        depth = DeepResearchEngine.research_depth_for_objective(objective)
        core = DeepResearchEngine._query_core_terms(objective) or objective
        hypotheses: list[str] = []
        seen: set[str] = set()

        for perspective in DeepResearchEngine._generic_perspectives(objective, depth):
            name = str(perspective.get("name") or "evidence").replace("-", " ")
            goal = str(perspective.get("goal") or "").strip().rstrip(".")
            if goal:
                hypothesis = f"{name.capitalize()} evidence will matter because it can {goal.lower()}."
            else:
                hypothesis = f"Evidence from the {name} perspective may materially change the conclusion for {core}."
            normalized = DeepResearchEngine._normalize_title(hypothesis)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            hypotheses.append(hypothesis)
            if len(hypotheses) >= 4:
                break

        if not hypotheses:
            hypotheses.append(
                f"Independent evidence aligned to {core} will reduce unsupported synthesis."
            )
        return hypotheses

    def _planning_browser_research(
        self,
        objective: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        depth = DeepResearchEngine.research_depth_for_objective(objective)
        core_query = DeepResearchEngine._query_from_objective(objective)
        reference_query = core_query or objective
        ai_strategy: dict[str, Any] = {}
        strategy_builder = getattr(self.research_engine, "_ai_research_strategy", None)
        if callable(strategy_builder):
            try:
                ai_strategy = strategy_builder(objective, core_query, depth) or {}
            except Exception:
                ai_strategy = {}

        diagnostic_queries = self._software_agent_diagnostic_queries(objective)
        objective_queries = (
            []
            if diagnostic_queries
            else list(self._browser_objective_queries(objective))
        )
        search_queries: list[str] = []
        seen_queries: set[str] = set()
        for candidate in (
            list(ai_strategy.get("reasoning_queries") or [])
            + diagnostic_queries
            + objective_queries
            + ([] if diagnostic_queries else [core_query])
        ):
            query_text = self._distill_browser_query(candidate)
            if not query_text:
                continue
            if DeepResearchEngine._is_low_signal_query_variant(
                query_text,
                reference_query,
            ):
                continue
            if DeepResearchEngine._is_noisy_query_variant(
                query_text,
                reference_query,
            ):
                continue
            normalized = DeepResearchEngine._normalize_title(query_text)
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            search_queries.append(query_text)

        seed_urls: list[str] = []
        seen_seed_urls: set[str] = set()
        for candidate in ai_strategy.get("authoritative_domains") or []:
            normalized_url = self._planning_seed_url(candidate)
            if not normalized_url or normalized_url in seen_seed_urls:
                continue
            seen_seed_urls.add(normalized_url)
            seed_urls.append(normalized_url)
        for candidate in self._browser_authoritative_seed_urls(objective, core_query):
            normalized_url = self._planning_seed_url(candidate)
            if not normalized_url or normalized_url in seen_seed_urls:
                continue
            seen_seed_urls.add(normalized_url)
            seed_urls.append(normalized_url)
        for candidate in self._software_agent_diagnostic_seed_urls(objective):
            normalized_url = self._planning_seed_url(candidate)
            if not normalized_url or normalized_url in seen_seed_urls:
                continue
            seen_seed_urls.add(normalized_url)
            seed_urls.append(normalized_url)

        authoritative_domains: list[str] = []
        seen_authoritative_domains: set[str] = set()
        for candidate in ai_strategy.get("authoritative_domains") or []:
            text = str(candidate or "").strip().lower().rstrip("/")
            if not text or text in seen_authoritative_domains:
                continue
            seen_authoritative_domains.add(text)
            authoritative_domains.append(text)
        for seed_url in seed_urls:
            domain = urllib.parse.urlparse(seed_url).netloc.lower().lstrip("www.")
            if not domain or domain in seen_authoritative_domains:
                continue
            seen_authoritative_domains.add(domain)
            authoritative_domains.append(domain)

        current_web_mode = SupervisorAgent._has_explicit_current_web_cues(
            objective
        ) and not DeepResearchEngine._looks_like_academic_query(objective)
        query_limit = 12 if depth == "multi-hour" or current_web_mode else 6
        if DeepResearchEngine._looks_like_software_agent_query(objective):
            query_limit = max(query_limit, 14)
        return {
            "enabled": SupervisorAgent._needs_active_pc_research(
                objective,
                depth,
                False,
            ),
            "search_queries": search_queries[:query_limit],
            "seed_urls": seed_urls[:8],
            "authoritative_domains": authoritative_domains[:12],
            "profile": analysis.get("profile") or {},
        }

    @staticmethod
    def _planning_seed_url(candidate: Any) -> str | None:
        text = str(candidate or "").strip().rstrip("/")
        if not text:
            return None
        if not re.match(r"^[a-z]+://", text, flags=re.IGNORECASE):
            text = f"https://{text.lstrip('/')}"
        if not DeepResearchEngine._is_safe_public_url(text):
            return None
        if DeepResearchEngine._is_search_result_url(text):
            return None
        return text

    @staticmethod
    def _browser_authoritative_seed_urls(
        objective: str,
        core_query: str,
    ) -> list[str]:
        combined = f"{objective} {core_query}".strip()
        if not DeepResearchEngine._looks_like_market_query(combined):
            return []
        seeds = [
            "https://sec.gov",
            "https://reuters.com/markets",
            "https://fred.stlouisfed.org",
        ]
        if DeepResearchEngine._looks_like_current_evidence_query(combined):
            seeds.append("https://www.bls.gov")
        return seeds[:4]

    @staticmethod
    def _multi_hour_min_runtime_seconds() -> int:
        return 0

    @staticmethod
    def _multi_hour_max_retrieval_passes(min_runtime_seconds: int) -> int:
        del min_runtime_seconds
        return 120

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
            brief = self.research_engine.run(
                research_objective,
                run_id,
                **kwargs,
            )
        else:
            brief = self.research_engine.run(research_objective, run_id)
        return self._merge_mcp_research_sources(brief)

    def _merge_mcp_research_sources(self, brief: Any) -> Any:
        try:
            from agentos_orchestrator.mcp import run_mcp_research_execution
        except Exception:
            return brief

        query = str(getattr(brief, "query", "") or getattr(brief, "objective", ""))
        if not query.strip():
            return brief
        execution = run_mcp_research_execution(query, limit=5)
        hits = list(getattr(execution, "hits", []) or [])
        diagnostics = list(getattr(execution, "diagnostics", []) or [])
        actions = [
            {
                "action_type": action.action_type,
                "target": action.target,
                "status": action.status,
                "server": action.server,
                "tool": action.tool,
                "result_count": action.result_count,
                "error": action.error,
                "stderr": list(action.stderr),
            }
            for action in list(getattr(execution, "actions", []) or [])
        ]
        if not hits and not diagnostics and not actions:
            return brief

        existing_keys = {
            (str(source.url).strip(), str(source.title).strip().lower())
            for source in getattr(brief, "sources", [])
        }
        mcp_sources: list[ResearchSource] = []
        for hit in hits:
            key = (str(hit.url).strip(), str(hit.title).strip().lower())
            if key in existing_keys:
                continue
            existing_keys.add(key)
            mcp_sources.append(
                ResearchSource(
                    provider=hit.provider,
                    title=hit.title,
                    url=hit.url,
                    abstract=hit.abstract,
                    score=45.0,
                    evidence_grade="tool-observation",
                    quality_flags=["mcp-tool"],
                )
            )
        if mcp_sources:
            brief.sources = list(getattr(brief, "sources", [])) + mcp_sources
        metadata = dict(getattr(brief, "metadata", {}) or {})
        metadata["mcp"] = {
            "source_count": len(mcp_sources),
            "diagnostics": diagnostics,
            "actions": actions,
        }
        brief.metadata = metadata
        return brief

    def _active_pc_research(
        self,
        run_id: str,
        task: TaskSpec,
        prior_results: list[WorkerResult],
    ) -> WorkerResult:
        backend = self._sandbox_pc_backend()
        action_type, target = self._pc_research_action_surface(backend)
        research_mode = self._pc_research_mode_label(backend)
        planning_context = self._latest_planning_context(prior_results) or {}
        browser_plan = planning_context.get("browser_research") or {}
        urls = self._candidate_urls_from_sources(prior_results)
        if not urls:
            urls = self._planning_urls_from_objective(task.objective)
        navigation_limit = self._pc_browser_navigation_limit(
            task.objective,
            urls,
            browser_plan,
        )

        action = ActionRequest(
            agent_id="pc-research-agent",
            action_type=action_type,
            target=target,
            payload={
                "action": "browse",
                "urls": urls[:navigation_limit],
                "max_navigation_urls": navigation_limit,
                "search_queries": list(browser_plan.get("search_queries") or [])[:12],
                "seed_urls": list(browser_plan.get("seed_urls") or [])[:8],
            },
        )
        self._write_pc_research_progress(
            run_id,
            {
                "stage": "pc-research-started",
                "objective": task.objective,
                "backend": getattr(backend, "name", "virtual-desktop-sandbox"),
                "candidate_urls": len(urls),
                "navigation_limit": navigation_limit,
                "search_queries": len(list(browser_plan.get("search_queries") or [])),
            },
        )
        # When the active backend is already a virtual/sandboxed environment, no
        # human approval is required — it is inherently contained.  Skip the
        # authorization gate so sandbox runs never surface an approval prompt.
        backend_is_virtual_sandbox = self._backend_is_sandbox(backend)
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
                if not backend_is_virtual_sandbox:
                    return self._active_pc_research_virtual_fallback(
                        run_id,
                        task,
                        urls[:navigation_limit],
                        artifact,
                        approval_payload,
                        list(browser_plan.get("search_queries") or [])[:12],
                    )
                if decision.requires_approval and decision.approval is not None:
                    raise ApprovalRequired(decision.approval)
                raise PermissionError(
                    f"{research_mode.capitalize()} was blocked by policy/trust checks. "
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
                browser_findings,
            ) = self._run_pc_browser_frontier_session(
                run_id,
                task.objective,
                backend,
                urls,
                browser_plan,
                navigation_limit,
            )
        except (BackendUnavailable, OSError, RuntimeError, ValueError) as exc:
            self._write_pc_research_progress(
                run_id,
                {
                    "stage": "pc-research-error",
                    "backend": backend_name,
                    "error": str(exc),
                },
            )
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
                summary=f"{research_mode.capitalize()} failed: {exc}",
                artifacts=[artifact],
                evidence=[
                    {
                        "source": artifact,
                        "claim": (
                            f"{research_mode.capitalize()} encountered an error."
                        ),
                    }
                ],
                confidence=0.4,
            )

        names = [node.name for node in post_nodes if node.name][:10]
        panel_summary = self._panel_region_summary(post_nodes)
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
        self._write_pc_research_progress(
            run_id,
            {
                "stage": "pc-research-completed",
                "backend": backend_name,
                "direct_urls": len(direct_urls),
                "judged_results": len(browser_findings.get("judged_results") or []),
                "discovered_domains": len(
                    browser_findings.get("discovered_domains") or []
                ),
                "worker_sessions": len(browser_findings.get("worker_sessions") or []),
            },
        )
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                f"Executed {research_mode} actions and captured structured "
                "findings for evidence integration."
            ),
            artifacts=[artifact],
            evidence=[
                {
                    "source": artifact,
                    "claim": (
                        f"A {research_mode} session was used to collect research findings."
                    ),
                    "targeted_urls": direct_urls[:3] or urls[:3],
                    "search_queries": browser_findings.get("search_queries") or [],
                }
            ],
            confidence=0.78 if direct_urls else 0.68,
        )

    @staticmethod
    def _pc_browser_navigation_limit(
        objective: str,
        urls: list[str],
        browser_plan: dict[str, Any] | None = None,
    ) -> int:
        safe_urls = [
            str(url).strip()
            for url in urls
            if isinstance(url, str) and str(url).strip()
        ]
        candidate_count = len(safe_urls)
        if candidate_count == 0:
            return 5
        query_count = 0
        if isinstance(browser_plan, dict):
            query_count = len(
                [
                    str(item).strip()
                    for item in (browser_plan.get("search_queries") or [])
                    if str(item).strip()
                ]
            )
        if WorkerAgent._pc_browser_expansive_mode(objective, browser_plan):
            adaptive_target = max(
                12,
                min(candidate_count, 24),
                min(query_count * 2, 24),
            )
            return max(12, min(adaptive_target, 24))
        if candidate_count >= 8 or query_count >= 5:
            return min(candidate_count, 8)
        return min(candidate_count, 5)

    @staticmethod
    def _pc_browser_cycle_count(
        objective: str,
        browser_plan: dict[str, Any] | None = None,
    ) -> int:
        query_count = 0
        if isinstance(browser_plan, dict):
            query_count = len(browser_plan.get("search_queries") or [])
        if WorkerAgent._pc_browser_expansive_mode(objective, browser_plan):
            return max(3, min(8, 2 + max(1, query_count // 4)))
        if query_count >= 6:
            return min(4, 1 + (query_count // 4))
        return 1

    @staticmethod
    def _pc_parallel_browser_worker_count(
        objective: str,
        backend: Any,
        frontier_urls: list[str],
    ) -> int:
        if not frontier_urls:
            return 1
        if not isinstance(backend, VirtualDesktopSandboxBackend):
            return 1
        if WorkerAgent._pc_browser_expansive_mode(objective):
            return max(1, min(4, len(frontier_urls) // 8 or 1))
        if len(frontier_urls) >= 8:
            return min(3, max(2, len(frontier_urls) // 5))
        return 1

    @staticmethod
    def _pc_browser_batch_limit(
        objective: str,
        navigation_limit: int,
    ) -> int:
        if WorkerAgent._pc_browser_expansive_mode(objective):
            return max(4, min(int(navigation_limit or 1), 12))
        return max(2, min(int(navigation_limit or 1), 8))

    @staticmethod
    def _pc_browser_expansive_mode(
        objective: str,
        browser_plan: dict[str, Any] | None = None,
    ) -> bool:
        current_web_mode = SupervisorAgent._has_explicit_current_web_cues(
            objective
        ) and not DeepResearchEngine._looks_like_academic_query(objective)
        depth = DeepResearchEngine.research_depth_for_objective(objective)
        if depth == "multi-hour" or current_web_mode:
            return True
        if DeepResearchEngine._looks_like_software_agent_query(objective):
            return True
        browser_enabled = bool((browser_plan or {}).get("enabled"))
        query_count = len((browser_plan or {}).get("search_queries") or [])
        signal_score = SupervisorAgent._pc_research_signal_score(
            objective,
            depth,
            False,
        )
        return browser_enabled or query_count >= 6 or signal_score >= 5

    @staticmethod
    def _empty_browser_findings() -> dict[str, Any]:
        return {
            "search_queries": [],
            "judged_results": [],
            "direct_urls": [],
            "discovered_domains": [],
            "candidate_urls": [],
            "search_result_count": 0,
            "frontier": {},
            "terminal_verifications": [],
        }

    def _run_pc_browser_frontier_session(
        self,
        run_id: str,
        objective: str,
        backend: Any,
        urls: list[str],
        browser_plan: dict[str, Any],
        navigation_limit: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[UiNode],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        frontier_graph = self._load_frontier_graph(run_id)
        market_query = DeepResearchEngine._looks_like_market_query(
            DeepResearchEngine._query_from_objective(objective)
        )
        frontier_graph = self._sanitize_frontier_graph(
            frontier_graph,
            market_query=market_query,
        )
        frontier_queue = self._frontier_session_seed_urls(
            urls,
            browser_plan,
            frontier_graph,
            market_query=market_query,
        )
        receipts: list[dict[str, Any]] = []
        post_nodes = backend.snapshot()
        backend_name = getattr(backend, "name", "virtual-desktop-sandbox")
        interrupt_report: dict[str, Any] = {
            "triggered": False,
            "navigated_urls": [],
            "reports": [],
        }
        workspace_report: dict[str, Any] = {"triggered": False}
        combined_findings = self._empty_browser_findings()
        checkpoints: list[dict[str, Any]] = []
        worker_sessions: list[dict[str, Any]] = []
        visited_urls: set[str] = set()
        cycle_count = self._pc_browser_cycle_count(objective, browser_plan)
        worker_count = self._pc_parallel_browser_worker_count(
            objective,
            backend,
            frontier_queue,
        )

        for cycle_index in range(cycle_count):
            cycle_batch_limit = self._pc_browser_batch_limit(
                objective,
                navigation_limit,
            )
            frontier_batch = [
                url
                for url in frontier_queue
                if str(url).strip() and str(url).strip() not in visited_urls
            ][: max(1, worker_count) * cycle_batch_limit]
            if not frontier_batch:
                break

            batches = self._partition_frontier_urls(frontier_batch, worker_count)
            worker_backends = self._pc_browser_worker_backends(
                backend,
                run_id,
                len(batches),
            )
            batch_results: list[dict[str, Any]] = []
            if len(batches) > 1:
                with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                    futures = [
                        pool.submit(
                            self._execute_browser_frontier_batch,
                            worker_backends[index],
                            objective,
                            batch,
                            browser_plan,
                            run_id,
                            cycle_index,
                            f"worker-{index + 1}",
                        )
                        for index, batch in enumerate(batches)
                    ]
                    for future in futures:
                        batch_results.append(future.result())
            else:
                batch_results.append(
                    self._execute_browser_frontier_batch(
                        worker_backends[0],
                        objective,
                        batches[0],
                        browser_plan,
                        run_id,
                        cycle_index,
                        "worker-1",
                    )
                )

            cycle_findings = self._empty_browser_findings()
            session_summary = {
                "cycle": cycle_index + 1,
                "worker_count": len(batch_results),
                "batch_sizes": [
                    len(result.get("batch_urls") or []) for result in batch_results
                ],
                "direct_urls": 0,
                "judged_results": 0,
            }
            for result in batch_results:
                receipts.extend(result.get("receipts") or [])
                if result.get("post_nodes"):
                    post_nodes = list(result["post_nodes"])
                backend_name = str(result.get("backend_name") or backend_name)
                if (result.get("workspace_report") or {}).get("triggered"):
                    workspace_report = dict(result["workspace_report"])
                batch_interrupt = result.get("interrupt_report") or {}
                interrupt_report["triggered"] = bool(
                    interrupt_report.get("triggered")
                    or batch_interrupt.get("triggered")
                )
                interrupt_report.setdefault("reports", []).extend(
                    list(batch_interrupt.get("reports") or [])
                )
                interrupt_report.setdefault("navigated_urls", []).extend(
                    list(batch_interrupt.get("navigated_urls") or [])
                )
                filtered_findings = self._filter_browser_findings_portals(
                    result.get("findings") or {},
                    market_query=market_query,
                )
                cycle_findings = self._merge_browser_findings(
                    cycle_findings,
                    filtered_findings,
                )
                combined_findings = self._merge_browser_findings(
                    combined_findings,
                    filtered_findings,
                )
                visited_urls.update(
                    str(url).strip() for url in (result.get("batch_urls") or [])
                )
            session_summary["direct_urls"] = len(
                cycle_findings.get("direct_urls") or []
            )
            session_summary["judged_results"] = len(
                cycle_findings.get("judged_results") or []
            )
            worker_sessions.append(session_summary)

            cycle_findings = self._filter_browser_findings_portals(
                cycle_findings,
                market_query=market_query,
            )
            combined_findings = self._filter_browser_findings_portals(
                combined_findings,
                market_query=market_query,
            )

            frontier_graph = self._merge_browser_findings_into_frontier_graph(
                frontier_graph,
                cycle_findings,
                run_id,
                cycle_index,
            )
            checkpoint = self._browser_reasoning_checkpoint(
                objective,
                frontier_graph,
                cycle_findings,
                cycle_index,
            )
            checkpoints.append(checkpoint)
            frontier_graph.setdefault("reasoning_checkpoints", []).append(checkpoint)
            frontier_graph = self._merge_browser_checkpoint_into_frontier_graph(
                frontier_graph,
                checkpoint,
                run_id,
                cycle_index,
            )
            frontier_graph = self._sanitize_frontier_graph(
                frontier_graph,
                market_query=market_query,
            )
            self._write_pc_research_progress(
                run_id,
                {
                    "stage": "pc-research-active",
                    "cycle": cycle_index + 1,
                    "max_cycles": cycle_count,
                    "worker_count": len(batch_results),
                    "frontier_batch_size": len(frontier_batch),
                    "frontier_url_count": len(frontier_queue),
                    "direct_urls": len(combined_findings.get("direct_urls") or []),
                    "judged_results": len(
                        combined_findings.get("judged_results") or []
                    ),
                    "discovered_domains": len(
                        combined_findings.get("discovered_domains") or []
                    ),
                },
            )
            frontier_queue.extend(
                self._filter_browser_seed_urls(
                    self._urls_from_browser_checkpoint(checkpoint),
                    market_query=market_query,
                )
            )
            frontier_queue.extend(
                self._frontier_graph_seed_urls(
                    frontier_graph,
                    limit=max(16, cycle_batch_limit * 2),
                    market_query=market_query,
                )
            )
            frontier_queue = self._interleave_public_urls_by_domain(frontier_queue)
            if not checkpoint.get("continue_research", True):
                break

        interrupt_report["navigated_urls"] = self._dedupe_public_urls(
            list(interrupt_report.get("navigated_urls") or [])
        )
        combined_findings = self._filter_browser_findings_portals(
            combined_findings,
            market_query=market_query,
        )
        graph_summary = self._frontier_graph_summary(frontier_graph)
        combined_findings["frontier_graph"] = {
            "path": str(self._run_frontier_graph_path(run_id)).replace("\\", "/"),
            "summary": graph_summary,
        }
        combined_findings["frontier_checkpoints"] = checkpoints
        combined_findings["worker_sessions"] = worker_sessions
        combined_findings.setdefault("frontier", {}).update(
            {
                "persistent_graph": True,
                "worker_count": worker_count,
                "cycles": len(worker_sessions),
                "graph_url_count": graph_summary.get("url_count", 0),
                "graph_domain_count": graph_summary.get("domain_count", 0),
            }
        )
        self._save_frontier_graph(run_id, frontier_graph)
        return (
            receipts,
            post_nodes,
            backend_name,
            interrupt_report,
            workspace_report,
            combined_findings,
        )

    def _execute_browser_frontier_batch(
        self,
        backend: Any,
        objective: str,
        batch_urls: list[str],
        browser_plan: dict[str, Any],
        run_id: str,
        cycle_index: int,
        worker_label: str,
    ) -> dict[str, Any]:
        batch_run_id = (
            f"{run_id}/pc-browser-{worker_label}-cycle-{cycle_index + 1}"
            if run_id
            else ""
        )
        batch_limit = max(
            1,
            min(
                len(batch_urls),
                self._pc_browser_batch_limit(objective, len(batch_urls)),
            ),
        )
        frontier_state = {
            "cycle": cycle_index + 1,
            "worker": worker_label,
            "batch_urls": batch_urls[:12],
        }
        (
            receipts,
            post_nodes,
            backend_name,
            interrupt_report,
            workspace_report,
        ) = self._run_pc_browser_actions(
            backend,
            batch_urls,
            run_id=batch_run_id,
            navigation_limit=batch_limit,
            objective=objective,
            frontier_state=frontier_state,
        )
        findings = self._reasoned_browser_findings(
            objective,
            batch_urls,
            browser_plan=browser_plan,
            backend=backend,
        )
        return {
            "batch_urls": list(batch_urls),
            "receipts": receipts,
            "post_nodes": post_nodes,
            "backend_name": backend_name,
            "interrupt_report": interrupt_report,
            "workspace_report": workspace_report,
            "findings": findings,
        }

    def _persistent_frontier_graph_path(self) -> Path:
        path = Path(".agentos") / "browser_frontier_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _run_frontier_graph_path(run_id: str) -> Path:
        path = Path("runs") / run_id / "pc" / "frontier_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _empty_frontier_graph() -> dict[str, Any]:
        return {
            "version": 1,
            "urls": {},
            "domains": {},
            "claims": {},
            "contradictions": [],
            "reasoning_checkpoints": [],
            "runs": {},
            "last_updated": "",
        }

    def _load_frontier_graph(self, run_id: str) -> dict[str, Any]:
        path = self._persistent_frontier_graph_path()
        if not path.exists():
            graph = self._empty_frontier_graph()
        else:
            try:
                graph = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                graph = self._empty_frontier_graph()
        graph.setdefault("runs", {}).setdefault(run_id, {})
        return graph

    def _save_frontier_graph(self, run_id: str, graph: dict[str, Any]) -> None:
        graph["last_updated"] = datetime.now(UTC).isoformat()
        persistent_path = self._persistent_frontier_graph_path()
        persistent_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        self._run_frontier_graph_path(run_id).write_text(
            json.dumps(graph, indent=2),
            encoding="utf-8",
        )

    def _frontier_session_seed_urls(
        self,
        urls: list[str],
        browser_plan: dict[str, Any],
        frontier_graph: dict[str, Any],
        *,
        market_query: bool = False,
    ) -> list[str]:
        seeds: list[str] = []
        seeds.extend(str(url) for url in (browser_plan.get("seed_urls") or []))
        seeds.extend(urls)
        seeds.extend(self._planning_browser_urls({"browser_research": browser_plan}))
        seed_limit = max(
            16,
            min(
                48,
                len(seeds) + (len(browser_plan.get("search_queries") or []) * 2),
            ),
        )
        seeds.extend(
            self._frontier_graph_seed_urls(
                frontier_graph,
                limit=seed_limit,
                market_query=market_query,
            )
        )
        return self._interleave_public_urls_by_domain(
            self._filter_browser_seed_urls(seeds, market_query=market_query)
        )

    def _frontier_graph_seed_urls(
        self,
        frontier_graph: dict[str, Any],
        limit: int = 8,
        *,
        market_query: bool = False,
    ) -> list[str]:
        ranked = sorted(
            list((frontier_graph.get("urls") or {}).values()),
            key=lambda item: (
                float(item.get("priority") or 0.0),
                -int(item.get("visits") or 0),
            ),
            reverse=True,
        )
        urls: list[str] = []
        for item in ranked:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            if (
                int(item.get("visits") or 0) > 0
                and str(item.get("status") or "") == "judged"
            ):
                continue
            urls.append(url)
            if len(urls) >= limit:
                break
        return self._filter_browser_seed_urls(urls, market_query=market_query)

    def _filter_browser_seed_urls(
        self,
        urls: list[str],
        *,
        market_query: bool = False,
    ) -> list[str]:
        filtered: list[str] = []
        del market_query
        for url in urls:
            clean_url = str(url or "").strip()
            if not clean_url:
                continue
            if self._is_low_primary_signal_portal_url(clean_url):
                continue
            filtered.append(clean_url)
        return self._dedupe_public_urls(filtered)

    @classmethod
    def _is_low_primary_signal_portal_url(cls, url: str) -> bool:
        domain = cls._browser_source_domain(url)
        if not domain:
            return False
        for blocked in cls._market_portal_browser_domains():
            if domain == blocked or domain.endswith(f".{blocked}"):
                return True
        return False

    def _sanitize_browser_reasoning_checkpoint_payload(
        self,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        if not checkpoint:
            return checkpoint

        sanitized = dict(checkpoint)
        sanitized["follow_up_queries"] = self._dedupe_list(
            [
                query_text[:240]
                for item in (checkpoint.get("follow_up_queries") or [])
                for query_text in [self._distill_browser_query(str(item or ""))]
                if query_text
            ]
        )[:18]
        sanitized["domain_leads"] = self._dedupe_list(
            [
                clean_domain
                for item in (checkpoint.get("domain_leads") or [])
                for clean_domain in [str(item or "").strip().lower().lstrip("www.")]
                if clean_domain
                and not self._is_low_primary_signal_portal_url(
                    f"https://{clean_domain}"
                )
            ]
        )[:16]
        sanitized["url_leads"] = self._dedupe_public_urls(
            [
                clean_url
                for item in (checkpoint.get("url_leads") or [])
                for clean_url in [str(item or "").strip()]
                if clean_url
                and not self._is_low_primary_signal_portal_url(clean_url)
            ]
        )[:16]
        sanitized["missing_evidence"] = self._dedupe_list(
            [
                str(item).strip()[:240]
                for item in (checkpoint.get("missing_evidence") or [])
                if str(item).strip()
            ]
        )[:16]
        sanitized["contradictions"] = self._dedupe_list(
            [
                str(item).strip()[:280]
                for item in (checkpoint.get("contradictions") or [])
                if str(item).strip()
            ]
        )[:16]
        sanitized["continue_research"] = bool(
            checkpoint.get("continue_research", True)
        )
        return sanitized

    def _filter_browser_findings_portals(
        self,
        findings: dict[str, Any],
        *,
        market_query: bool = False,
    ) -> dict[str, Any]:
        del market_query
        if not findings:
            return findings

        sanitized = dict(findings)
        candidate_urls = [
            str(url).strip()
            for url in (findings.get("candidate_urls") or [])
            if str(url).strip()
            and not self._is_low_primary_signal_portal_url(str(url).strip())
        ]
        direct_urls = [
            str(url).strip()
            for url in (findings.get("direct_urls") or [])
            if str(url).strip()
            and not self._is_low_primary_signal_portal_url(str(url).strip())
        ]
        allowed_urls = set(candidate_urls) | set(direct_urls)
        sanitized["candidate_urls"] = self._dedupe_public_urls(candidate_urls)
        sanitized["direct_urls"] = self._dedupe_public_urls(direct_urls)
        sanitized["judged_results"] = [
            item
            for item in (findings.get("judged_results") or [])
            if str(item.get("url") or "").strip() in allowed_urls
        ]
        sanitized["discovered_domains"] = [
            domain
            for domain in (findings.get("discovered_domains") or [])
            if domain
            and not self._is_low_primary_signal_portal_url(f"https://{domain}")
        ]
        return sanitized

    def _sanitize_frontier_graph(
        self,
        frontier_graph: dict[str, Any],
        *,
        market_query: bool = False,
    ) -> dict[str, Any]:
        del market_query
        if not frontier_graph:
            return frontier_graph

        sanitized = dict(frontier_graph)
        url_entries = dict(frontier_graph.get("urls") or {})
        allowed_urls = {
            url
            for url in url_entries
            if not self._is_low_primary_signal_portal_url(str(url))
        }
        sanitized["urls"] = {
            url: payload for url, payload in url_entries.items() if url in allowed_urls
        }

        domain_entries = dict(frontier_graph.get("domains") or {})
        sanitized_domains: dict[str, Any] = {}
        for domain, payload in domain_entries.items():
            clean_domain = str(domain or "").strip().lower().lstrip("www.")
            if not clean_domain:
                continue
            if self._is_low_primary_signal_portal_url(f"https://{clean_domain}"):
                continue
            cleaned_payload = dict(payload or {})
            cleaned_payload["urls"] = [
                url for url in (cleaned_payload.get("urls") or []) if url in allowed_urls
            ]
            sanitized_domains[clean_domain] = cleaned_payload
        sanitized["domains"] = sanitized_domains

        claim_entries = dict(frontier_graph.get("claims") or {})
        sanitized_claims: dict[str, Any] = {}
        for key, payload in claim_entries.items():
            cleaned_payload = dict(payload or {})
            cleaned_sources = [
                url
                for url in (cleaned_payload.get("sources") or [])
                if url in allowed_urls
            ]
            if cleaned_sources:
                cleaned_payload["sources"] = cleaned_sources
                sanitized_claims[key] = cleaned_payload
        sanitized["claims"] = sanitized_claims
        sanitized["reasoning_checkpoints"] = [
            self._sanitize_browser_reasoning_checkpoint_payload(checkpoint)
            for checkpoint in (frontier_graph.get("reasoning_checkpoints") or [])
            if isinstance(checkpoint, dict)
        ]
        return sanitized

    def _merge_browser_checkpoint_into_frontier_graph(
        self,
        frontier_graph: dict[str, Any],
        checkpoint: dict[str, Any],
        run_id: str,
        cycle_index: int,
    ) -> dict[str, Any]:
        contradictions = frontier_graph.setdefault("contradictions", [])
        runs = frontier_graph.setdefault("runs", {})
        run_entry = runs.setdefault(run_id, {})
        run_entry.setdefault("checkpoint_cycles", []).append(cycle_index + 1)

        for contradiction in checkpoint.get("contradictions") or []:
            text = str(contradiction or "").strip()
            if not text:
                continue
            normalized = DeepResearchEngine._normalize_title(text)
            if any(
                DeepResearchEngine._normalize_title(str(item.get("text") or ""))
                == normalized
                for item in contradictions
                if isinstance(item, dict)
            ):
                continue
            contradictions.append(
                {
                    "text": text[:320],
                    "run_id": run_id,
                    "cycle": cycle_index + 1,
                }
            )

        frontier_urls = frontier_graph.setdefault("urls", {})
        frontier_domains = frontier_graph.setdefault("domains", {})
        for url in checkpoint.get("url_leads") or []:
            clean_url = str(url or "").strip()
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                continue
            if self._is_low_primary_signal_portal_url(clean_url):
                continue
            domain = urllib.parse.urlparse(clean_url).netloc.lower().lstrip("www.")
            entry = frontier_urls.setdefault(
                clean_url,
                {"url": clean_url, "domain": domain, "visits": 0},
            )
            entry["last_run"] = run_id
            entry["last_cycle"] = cycle_index + 1
            entry["status"] = "checkpoint-lead"
            entry["priority"] = max(float(entry.get("priority") or 0.0), 0.72)
            if domain:
                domain_entry = frontier_domains.setdefault(
                    domain,
                    {"urls": [], "observations": 0},
                )
                if clean_url not in domain_entry["urls"]:
                    domain_entry["urls"].append(clean_url)

        for domain in checkpoint.get("domain_leads") or []:
            clean_domain = str(domain or "").strip().lower().lstrip("www.")
            if not clean_domain:
                continue
            if self._is_low_primary_signal_portal_url(f"https://{clean_domain}"):
                continue
            domain_entry = frontier_domains.setdefault(
                clean_domain,
                {"urls": [], "observations": 0},
            )
            domain_entry["last_run"] = run_id
            domain_entry["priority"] = max(
                float(domain_entry.get("priority") or 0.0),
                0.7,
            )
            if checkpoint.get("missing_evidence"):
                domain_entry["missing_evidence"] = self._dedupe_list(
                    list(domain_entry.get("missing_evidence") or [])
                    + [
                        str(item).strip()[:240]
                        for item in (checkpoint.get("missing_evidence") or [])
                        if str(item).strip()
                    ]
                )
        return frontier_graph

    @staticmethod
    def _dedupe_public_urls(urls: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            clean = str(url or "").strip().rstrip(").,;]}>'\"")
            if not clean:
                continue
            if not DeepResearchEngine._is_safe_public_url(clean):
                continue
            if clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped

    @staticmethod
    def _interleave_public_urls_by_domain(urls: list[str]) -> list[str]:
        deduped = WorkerAgent._dedupe_public_urls(urls)
        if len(deduped) <= 2:
            return deduped

        grouped: dict[str, list[str]] = {}
        domain_order: list[str] = []
        for url in deduped:
            domain = WorkerAgent._browser_source_domain(url) or url
            if domain not in grouped:
                grouped[domain] = []
                domain_order.append(domain)
            grouped[domain].append(url)

        interleaved: list[str] = []
        while True:
            progressed = False
            for domain in domain_order:
                bucket = grouped.get(domain) or []
                if not bucket:
                    continue
                interleaved.append(bucket.pop(0))
                progressed = True
            if not progressed:
                break
        return interleaved

    @staticmethod
    def _partition_frontier_urls(
        urls: list[str],
        worker_count: int,
    ) -> list[list[str]]:
        buckets: list[list[str]] = [[] for _ in range(max(1, worker_count))]
        for index, url in enumerate(urls):
            buckets[index % len(buckets)].append(url)
        return [bucket for bucket in buckets if bucket]

    def _pc_browser_worker_backends(
        self,
        backend: Any,
        run_id: str,
        worker_count: int,
    ) -> list[Any]:
        if worker_count <= 1 or not isinstance(backend, VirtualDesktopSandboxBackend):
            return [backend]
        workers: list[Any] = [backend]
        for index in range(1, worker_count):
            workers.append(
                VirtualDesktopSandboxBackend(
                    Path(".agentos")
                    / f"virtual_browser_worker_{run_id}_{index + 1}.json"
                )
            )
        return workers

    def _merge_browser_findings(
        self,
        current: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(current)
        for key in (
            "search_queries",
            "direct_urls",
            "candidate_urls",
            "discovered_domains",
            "terminal_verifications",
        ):
            merged[key] = self._dedupe_list(
                list(current.get(key) or []) + list(update.get(key) or [])
            )
        merged["judged_results"] = self._dedupe_judged_results(
            list(current.get("judged_results") or [])
            + list(update.get("judged_results") or [])
        )
        merged["search_result_count"] = int(
            current.get("search_result_count") or 0
        ) + int(update.get("search_result_count") or 0)
        merged_frontier = dict(current.get("frontier") or {})
        merged_frontier.update(update.get("frontier") or {})
        merged["frontier"] = merged_frontier
        return merged

    @staticmethod
    def _dedupe_list(items: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = (
                json.dumps(item, sort_keys=True)
                if isinstance(item, dict)
                else str(item)
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _dedupe_judged_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            result.append(item)
        return result

    def _merge_browser_findings_into_frontier_graph(
        self,
        frontier_graph: dict[str, Any],
        findings: dict[str, Any],
        run_id: str,
        cycle_index: int,
    ) -> dict[str, Any]:
        urls = frontier_graph.setdefault("urls", {})
        domains = frontier_graph.setdefault("domains", {})
        claims = frontier_graph.setdefault("claims", {})
        run_entry = frontier_graph.setdefault("runs", {}).setdefault(run_id, {})
        run_entry["last_cycle"] = cycle_index + 1
        run_entry["last_updated"] = datetime.now(UTC).isoformat()

        direct_urls = {
            str(url).strip()
            for url in (findings.get("direct_urls") or [])
            if str(url).strip()
        }
        candidate_urls = [
            str(url).strip()
            for url in (findings.get("candidate_urls") or [])
            if str(url).strip()
        ]
        judged_index = {
            str(item.get("url") or "").strip(): item
            for item in (findings.get("judged_results") or [])
            if str(item.get("url") or "").strip()
        }
        verified_claims = {
            DeepResearchEngine._normalize_title(str(item.get("claim") or ""))
            for item in (findings.get("terminal_verifications") or [])
            if str(item.get("claim") or "").strip()
            and int(item.get("exit_code") or 1) == 0
        }

        for url in candidate_urls:
            domain = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
            entry = urls.setdefault(url, {"url": url, "domain": domain, "visits": 0})
            entry["last_run"] = run_id
            entry["last_cycle"] = cycle_index + 1
            entry["status"] = "direct" if url in direct_urls else "candidate"
            entry["priority"] = max(
                float(entry.get("priority") or 0.0),
                0.9 if url in direct_urls else 0.55,
            )
            if url in direct_urls:
                entry["visits"] = int(entry.get("visits") or 0) + 1
            judged = judged_index.get(url) or {}
            if judged:
                entry["title"] = str(judged.get("title") or entry.get("title") or "")[
                    :200
                ]
                entry["judgment"] = str(judged.get("judgment") or "")[:400]
                entry["quality_flags"] = self._dedupe_list(
                    list(entry.get("quality_flags") or [])
                    + list(judged.get("quality_flags") or [])
                )
                evidence_claims = [
                    str(claim).strip()[:400]
                    for claim in (judged.get("evidence_claims") or [])
                    if str(claim).strip()
                ]
                entry["evidence_claims"] = self._dedupe_list(
                    list(entry.get("evidence_claims") or []) + evidence_claims
                )
                contradiction_risk = self.research_engine._contradiction_risk(
                    f"{judged.get('judgment') or ''} {' '.join(evidence_claims[:3])}"
                )
                entry["contradiction_risk"] = max(
                    float(entry.get("contradiction_risk") or 0.0),
                    float(contradiction_risk),
                )
                for claim in evidence_claims:
                    claim_key = DeepResearchEngine._normalize_title(claim)
                    if not claim_key:
                        continue
                    claim_entry = claims.setdefault(
                        claim_key,
                        {"claim": claim, "sources": [], "verified_count": 0},
                    )
                    if url not in claim_entry["sources"]:
                        claim_entry["sources"].append(url)
                    if claim_key in verified_claims:
                        claim_entry["verified_count"] = (
                            int(claim_entry.get("verified_count") or 0) + 1
                        )
            if domain:
                domain_entry = domains.setdefault(
                    domain, {"urls": [], "observations": 0}
                )
                if url not in domain_entry["urls"]:
                    domain_entry["urls"].append(url)
                domain_entry["observations"] = (
                    int(domain_entry.get("observations") or 0) + 1
                )
                domain_entry["last_run"] = run_id
        return frontier_graph

    def _browser_reasoning_checkpoint(
        self,
        objective: str,
        frontier_graph: dict[str, Any],
        findings: dict[str, Any],
        cycle_index: int,
    ) -> dict[str, Any]:
        summary = {
            "cycle": cycle_index + 1,
            "frontier": self._frontier_graph_summary(frontier_graph),
            "judged_results": [
                {
                    "title": str(item.get("title") or "")[:160],
                    "url": str(item.get("url") or "")[:240],
                    "judgment": str(item.get("judgment") or "")[:280],
                    "evidence_claims": list(item.get("evidence_claims") or [])[:2],
                }
                for item in list(findings.get("judged_results") or [])[:16]
            ],
            "terminal_verifications": list(
                findings.get("terminal_verifications") or []
            )[:12],
            "prior_contradictions": list(frontier_graph.get("contradictions") or [])[
                :12
            ],
        }
        system = (
            "You are a browser research coordinator. Based on the browser findings, "
            "identify missing evidence, contradiction leads, and the best next "
            "queries or URLs to pursue. Respond only with JSON."
        )
        user = (
            f"Objective: {objective}\n"
            f"Browser findings JSON: {json.dumps(summary, ensure_ascii=True)}\n\n"
            "Return JSON with keys follow_up_queries, domain_leads, url_leads, "
            "missing_evidence, contradictions, continue_research."
        )
        raw = self.research_engine._call_ai_text(system, user)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                return self._sanitize_browser_reasoning_checkpoint_payload({
                    "cycle": cycle_index + 1,
                    "follow_up_queries": self._dedupe_list(
                        [
                            str(item)[:240]
                            for item in (parsed.get("follow_up_queries") or [])
                            if str(item).strip()
                        ]
                    )[:18],
                    "domain_leads": self._dedupe_list(
                        [
                            str(item).strip()
                            for item in (parsed.get("domain_leads") or [])
                            if str(item).strip()
                        ]
                    )[:16],
                    "url_leads": self._dedupe_public_urls(
                        [
                            str(item).strip()
                            for item in (parsed.get("url_leads") or [])
                            if str(item).strip()
                        ]
                    )[:16],
                    "missing_evidence": self._dedupe_list(
                        [
                            str(item)[:240]
                            for item in (parsed.get("missing_evidence") or [])
                            if str(item).strip()
                        ]
                    )[:16],
                    "contradictions": self._dedupe_list(
                        [
                            str(item)[:280]
                            for item in (parsed.get("contradictions") or [])
                            if str(item).strip()
                        ]
                    )[:16],
                    "continue_research": bool(parsed.get("continue_research", True)),
                })
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return self._browser_reasoning_checkpoint_fallback(
            objective,
            frontier_graph,
            findings,
            cycle_index,
        )

    def _browser_reasoning_checkpoint_fallback(
        self,
        objective: str,
        frontier_graph: dict[str, Any],
        findings: dict[str, Any],
        cycle_index: int,
    ) -> dict[str, Any]:
        judged_results = list(findings.get("judged_results") or [])
        direct_domains = {
            urllib.parse.urlparse(str(url)).netloc.lower().lstrip("www.")
            for url in (findings.get("direct_urls") or [])
            if str(url).strip()
        }
        candidate_urls = [
            str(url).strip()
            for url in (findings.get("candidate_urls") or [])
            if str(url).strip()
            and not self._is_low_primary_signal_portal_url(str(url).strip())
        ]
        url_leads = [
            url
            for url in candidate_urls
            if url not in (findings.get("direct_urls") or [])
        ][:8]
        domain_leads = self._dedupe_list(
            [
                urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
                for url in url_leads
                if urllib.parse.urlparse(url).netloc
                and urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
                not in direct_domains
            ]
        )[:6]

        missing_evidence: list[str] = []
        if len(direct_domains) < 2:
            missing_evidence.append(
                "Need broader domain coverage beyond the current browser pages."
            )
        if not findings.get("terminal_verifications"):
            missing_evidence.append(
                "Need independent verification of browser-derived claims."
            )
        if len(judged_results) < 3:
            missing_evidence.append(
                "Need more direct pages with substantive evidence extraction."
            )

        contradictions: list[str] = []
        follow_up_queries: list[str] = []
        for result in judged_results[:6]:
            title = str(result.get("title") or "").strip()
            judgment = str(result.get("judgment") or "").strip()
            evidence_claims = [
                str(claim).strip()
                for claim in (result.get("evidence_claims") or [])
                if str(claim).strip()
            ]
            contradiction_risk = self.research_engine._contradiction_risk(
                f"{judgment} {' '.join(evidence_claims[:3])}"
            )
            if contradiction_risk >= 0.3:
                contradictions.append(
                    f"Potential contradiction or unresolved uncertainty from {title[:80] or result.get('url')}."
                )
                follow_up_queries.append(
                    (
                        self.research_engine._query_core_terms(
                            f"{title} counterevidence {objective}"
                        )
                        or objective
                    )[:240]
                )
            for claim in evidence_claims[:1]:
                query_text = self.research_engine._query_core_terms(f"{title} {claim}")
                if query_text and len(query_text.split()) >= 3:
                    follow_up_queries.append(query_text[:240])
        if not follow_up_queries and missing_evidence:
            follow_up_queries.append(
                (
                    self.research_engine._query_core_terms(
                        f"{objective} independent verification"
                    )
                    or objective
                )[:240]
            )
        graph_contradictions = [
            str(item.get("text") or "").strip()
            for item in (frontier_graph.get("contradictions") or [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        for contradiction in graph_contradictions[:6]:
            follow_up_queries.append(
                (
                    self.research_engine._query_core_terms(
                        f"{objective} {contradiction} independent evidence"
                    )
                    or f"{objective} {contradiction}"
                )[:240]
            )
        return self._sanitize_browser_reasoning_checkpoint_payload({
            "cycle": cycle_index + 1,
            "follow_up_queries": self._dedupe_list(follow_up_queries)[:18],
            "domain_leads": domain_leads[:16],
            "url_leads": self._dedupe_public_urls(url_leads)[:16],
            "missing_evidence": missing_evidence[:16],
            "contradictions": self._dedupe_list(contradictions)[:16],
            "continue_research": bool(
                url_leads
                or follow_up_queries
                or missing_evidence
                or graph_contradictions
            ),
        })

    def _urls_from_browser_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> list[str]:
        urls: list[str] = []
        urls.extend(
            self._dedupe_public_urls(
                [str(url).strip() for url in (checkpoint.get("url_leads") or [])]
            )
        )
        for domain in checkpoint.get("domain_leads") or []:
            candidate = self._planning_seed_url(domain)
            if candidate:
                urls.append(candidate)
        for query in checkpoint.get("follow_up_queries") or []:
            query_text = str(query).strip()
            if not query_text:
                continue
            urls.append(
                "https://html.duckduckgo.com/html/?q="
                + urllib.parse.quote_plus(query_text[:240])
            )
        return self._dedupe_public_urls(urls)

    @staticmethod
    def _frontier_graph_summary(frontier_graph: dict[str, Any]) -> dict[str, Any]:
        urls = list((frontier_graph.get("urls") or {}).values())
        ranked_urls = sorted(
            urls,
            key=lambda item: float(item.get("priority") or 0.0),
            reverse=True,
        )
        domains = list((frontier_graph.get("domains") or {}).items())
        ranked_domains = sorted(
            domains,
            key=lambda item: (
                float((item[1] or {}).get("priority") or 0.0),
                int((item[1] or {}).get("observations") or 0),
            ),
            reverse=True,
        )
        return {
            "url_count": len(urls),
            "domain_count": len(frontier_graph.get("domains") or {}),
            "claim_count": len(frontier_graph.get("claims") or {}),
            "contradiction_count": len(frontier_graph.get("contradictions") or []),
            "top_urls": [str(item.get("url") or "") for item in ranked_urls[:6]],
            "top_domains": [str(domain) for domain, _payload in ranked_domains[:10]],
        }

    def _reasoned_browser_findings(
        self,
        objective: str,
        urls: list[str],
        browser_plan: dict[str, Any] | None = None,
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
        queries = self._browser_search_queries(
            objective,
            urls,
            list((browser_plan or {}).get("search_queries") or []),
        )
        core_query = DeepResearchEngine._query_from_objective(objective)

        source_strategy = self._ai_browser_source_strategy(objective)
        for sq in source_strategy.get("targeted_queries") or []:
            if sq and sq.strip() and sq not in queries:
                queries.append(sq.strip()[:240])
        queries = self._sanitize_browser_search_queries(
            queries,
            core_query or objective,
        )

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

        navigation_seed_sources: list[ResearchSource] = []
        seen_navigation_urls: set[str] = set()
        navigation_seed_limit = max(4, int(budget["max_direct_urls"]) // 8)
        for url in urls[:navigation_seed_limit]:
            clean_url = str(url or "").strip()
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                continue
            if DeepResearchEngine._is_search_result_url(clean_url):
                continue
            if clean_url in seen_navigation_urls:
                continue
            seen_navigation_urls.add(clean_url)
            domain = urllib.parse.urlparse(clean_url).netloc.lower().lstrip("www.")
            navigation_seed_sources.append(
                ResearchSource(
                    provider="pc-browser-research",
                    title=self.research_engine._label_from_url(clean_url)[:160],
                    url=clean_url,
                    authors=[domain] if domain else [],
                    abstract=(
                        "Sandbox browser navigation seed captured from the active "
                        "browser research context."
                    ),
                    score=96.0,
                    evidence_grade="tool-observation",
                    quality_flags=["browser-navigation-seed"],
                )
            )

        # GENERALITY: portal denylist applies to *every* topic, not only
        # market queries.  Aggregator hosts (yahoo/marketwatch/msn/medium/
        # substack/fool/zacks/...) re-publish third-party content instead
        # of originating it, regardless of subject domain.  Apply the
        # filter to both candidate results and the navigation seeds so
        # blocked portals cannot leak into exploration_sources via either
        # path and pollute direct_urls / judged_results downstream.
        if raw_results:
            raw_results = self._filter_market_browser_sources(
                raw_results,
                navigation_seed_sources,
            )
        if navigation_seed_sources:
            navigation_seed_sources = self._filter_market_browser_sources(
                navigation_seed_sources,
                raw_results,
            )

        deduped_results = self.research_engine._dedupe_sources(
            navigation_seed_sources + raw_results
        )
        ranked = self.research_engine._rank_sources(
            deduped_results,
            core_query,
        )
        prioritized_navigation_urls = {source.url for source in navigation_seed_sources}
        minimum_domain_target = (
            4
            if is_market_query and str(budget["mode"]) == "expansive"
            else 3 if str(budget["mode"]) == "expansive" else 2
        )
        ranked = self._augment_browser_domain_coverage(
            ranked,
            deduped_results,
            minimum_domains=minimum_domain_target,
            prioritized_urls=prioritized_navigation_urls,
        )
        exploration_sources = list(navigation_seed_sources)
        exploration_sources.extend(
            source for source in ranked if source.url not in prioritized_navigation_urls
        )
        exploration_sources = self._interleave_browser_sources_by_domain(
            exploration_sources,
            prioritized_urls=prioritized_navigation_urls,
        )
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
        preview_direct_limit = min(5, int(budget["max_direct_urls"]))
        # Cache the portal denylist once per call so admit-time lookups are
        # cheap.  Using the same set the filter uses guarantees admit and
        # filter agree on what counts as a low-primary-signal portal.
        _portal_blocklist = self._market_portal_browser_domains()

        def _is_portal_blocked(clean_url: str) -> bool:
            domain = self._browser_source_domain(clean_url)
            if not domain:
                return False
            if domain in _portal_blocklist:
                return True
            for blocked in _portal_blocklist:
                if domain == blocked or domain.endswith(f".{blocked}"):
                    return True
            return False

        def admit_preview_source(
            source: ResearchSource,
            require_new_domain: bool = False,
        ) -> bool:
            if len(direct_urls) >= preview_direct_limit:
                return False
            clean_url = str(source.url or "").strip()
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                return False
            if DeepResearchEngine._is_search_result_url(clean_url):
                return False
            if clean_url in seen_direct_urls:
                return False
            domain = urllib.parse.urlparse(clean_url).netloc.lower().lstrip("www.")
            if require_new_domain and domain and domain in discovered_domains:
                return False
            if domain and domain_usage.get(domain, 0) >= int(budget["max_per_domain"]):
                return False
            # Portal guard: never admit a low-primary-signal aggregator host
            # when at least one authoritative direct_url has already been
            # admitted.  Prevents Yahoo/Marketwatch/Medium-style drift into
            # judged_results when better sources are available.
            if _is_portal_blocked(clean_url) and direct_urls:
                return False
            preview = self._browser_page_preview(clean_url)
            excerpt = str(preview.get("page_excerpt") or "")
            if self._browser_preview_is_blocked(preview):
                return False
            quality = self._content_block_quality(excerpt)
            if quality["quality_score"] < 0.22:
                return False
            judged_results.append(
                {
                    "query": queries[0],
                    "title": str(preview.get("page_title") or source.title),
                    "url": clean_url,
                    "domain": domain,
                    "abstract": source.abstract[:400],
                    "page_excerpt": excerpt[:600],
                    "evidence_claims": [],
                    "content_quality": quality,
                    "judgment": self._browser_result_judgment(source, preview),
                    "quality_flags": list(source.quality_flags or [])
                    + ["preview-only"],
                }
            )
            seen_direct_urls.add(clean_url)
            if domain:
                domain_usage[domain] = domain_usage.get(domain, 0) + 1
            direct_urls.append(clean_url)
            if domain and domain not in discovered_domains:
                discovered_domains.append(domain)
            return True

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
            # Portal guard at deep-read boundary: same contract as admit
            # \u2014 never read a low-primary-signal aggregator when an
            # authoritative direct_url has already been admitted.
            if _is_portal_blocked(clean_url) and direct_urls:
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
                    "quality_flags": list(source.quality_flags or []),
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

        if 0 < len(discovered_domains) < minimum_domain_target:
            for source in exploration_sources:
                if len(discovered_domains) >= minimum_domain_target:
                    break
                admit_preview_source(source, require_new_domain=True)

        if not direct_urls:
            for source in exploration_sources:
                if not admit_preview_source(source):
                    continue
                if len(direct_urls) >= preview_direct_limit:
                    break

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

        max_queries = min(query_count, 48 if expansive_mode else 16)
        web_results_per_query = 12 if expansive_mode else 8
        max_direct_urls = 160 if expansive_mode else 40
        max_per_domain = 4 if expansive_mode or is_market_query else 2
        returned_query_count = min(query_count, 24 if expansive_mode else 8)
        candidate_urls = min(max(max_direct_urls * 4, 240), 1200)

        return {
            "mode": "expansive" if expansive_mode else "standard",
            "max_queries": max(1, max_queries),
            "web_results_per_query": web_results_per_query,
            "financial_results_per_query": 16 if expansive_mode else 10,
            "sec_results_per_query": 12 if expansive_mode else 6,
            "max_direct_urls": max_direct_urls,
            "max_per_domain": max_per_domain,
            "returned_query_count": max(1, returned_query_count),
            "candidate_urls": candidate_urls,
        }

    @staticmethod
    def _browser_search_queries(
        objective: str,
        urls: list[str],
        plan_queries: list[str] | None = None,
    ) -> list[str]:
        queries: list[str] = []
        prioritized_queries: list[str] = []
        for query in plan_queries or []:
            query_text = WorkerAgent._distill_browser_query(query)
            if query_text:
                query_text = query_text[:240]
                prioritized_queries.append(query_text)
                queries.append(query_text)
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            query_value = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            query_value = WorkerAgent._distill_browser_query(
                urllib.parse.unquote_plus(query_value)
            )
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
        diagnostic_queries = WorkerAgent._software_agent_diagnostic_queries(cleaned)
        queries.extend(diagnostic_queries)
        objective_queries = (
            []
            if diagnostic_queries
            else WorkerAgent._browser_objective_queries(cleaned)
        )
        for clause in objective_queries:
            query_text = WorkerAgent._distill_browser_query(clause)
            if query_text:
                queries.append(query_text)
        fallback = (
            ""
            if diagnostic_queries
            else DeepResearchEngine._query_from_objective(cleaned)
        )
        if fallback:
            queries.append(fallback)
        query_context = fallback or DeepResearchEngine._query_from_objective(
            cleaned
        ) or cleaned
        queries = WorkerAgent._sanitize_browser_search_queries(
            queries,
            query_context,
        )
        if not queries:
            minimal_fallback = WorkerAgent._distill_browser_query(cleaned)
            if not minimal_fallback:
                minimal_fallback = DeepResearchEngine._trim_query_variant_text(cleaned)
            if minimal_fallback:
                queries = [minimal_fallback[:240]]
        current_web_mode = SupervisorAgent._has_explicit_current_web_cues(
            cleaned
        ) and not DeepResearchEngine._looks_like_academic_query(cleaned)
        query_limit = (
            20 if "multi-hour" in objective.lower() or current_web_mode else 12
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for query in [*prioritized_queries, *queries]:
            normalized = DeepResearchEngine._normalize_title(query)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query[:240])
        return deduped[:query_limit]

    @staticmethod
    def _sanitize_browser_search_queries(
        queries: list[str],
        query_context: str,
    ) -> list[str]:
        distilled: list[str] = []
        seen: set[str] = set()
        for candidate in queries:
            query_text = WorkerAgent._distill_browser_query(candidate)
            if not query_text:
                continue
            normalized = DeepResearchEngine._normalize_title(query_text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            distilled.append(query_text[:240])
        if not distilled:
            fallback = DeepResearchEngine._trim_query_variant_text(query_context)
            return [fallback] if fallback else []
        sanitized = DeepResearchEngine._sanitize_query_variants(
            distilled,
            query_context,
        )
        return sanitized or distilled

    @staticmethod
    def _distill_browser_query(candidate: str) -> str:
        text = re.sub(r"\s+", " ", str(candidate or "")).strip(" ,.;:-")
        if not text:
            return ""

        for prefix in (
            "Perform sandboxed browser research actions for:",
            "Capture live desktop and browser/operator context for:",
            "Find authoritative sources, direct evidence, and major uncertainties for:",
            "Design deep research plan for:",
        ):
            text = text.replace(prefix, " ")

        boilerplate_patterns = (
            r"\b(?:use|using) all available [^.;]+",
            r"\ball available general-purpose research means[^.;]*",
            r"\bbrowser-grounded web research[^.;]*",
            r"\bsandboxed exploration[^.;]*",
            r"\bcurrent-web evidence[^.;]*",
            r"\bcross-checking\b",
            r"\bdo not use [^.;]+",
            r"\bproduce (?:an?|the) [^.;]+",
            r"\bthe evidence for and against each thesis[^.;]*",
            r"\buncertainty bounds\b",
            r"\bcatalyst quality\b",
            r"\bexecution risk\b",
            r"\bvaluation-sensitive considerations\b",
            r"\bclear reasons? for the ranking\b",
        )
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
        if not text:
            return ""

        lowered = text.lower()
        meta_markers = (
            "all available",
            "analyst-grade",
            "scientist-grade",
            "ranked candidates",
            "uncertainty bounds",
            "current-web",
            "do not use",
        )
        tokens = re.findall(r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b", lowered)
        if len(tokens) >= 12 or any(marker in lowered for marker in meta_markers):
            distilled = DeepResearchEngine._query_core_terms(text)
            if distilled:
                text = distilled
                lowered = text.lower()
                tokens = re.findall(r"\b[0-9a-z]+(?:-[0-9a-z]+)*\b", lowered)

        meta_tokens = {
            "a",
            "about",
            "against",
            "all",
            "an",
            "and",
            "available",
            "analysis",
            "analyst",
            "analyst-grade",
            "browser",
            "by",
            "cross-checking",
            "current-web",
            "each",
            "evidence",
            "for",
            "from",
            "general-purpose",
            "grade",
            "in",
            "into",
            "means",
            "of",
            "on",
            "over",
            "produce",
            "ranking",
            "reasons",
            "report",
            "sandbox",
            "sandboxed",
            "scientist",
            "scientist-grade",
            "that",
            "the",
            "thesis",
            "through",
            "to",
            "tools",
            "under",
            "uncertainty",
            "use",
            "using",
            "web",
            "with",
        }
        signal_tokens = [token for token in tokens if token not in meta_tokens]
        if len(signal_tokens) < 3:
            return ""
        return text[:240]

    @staticmethod
    def _browser_source_domain(url: str) -> str:
        return urllib.parse.urlparse(str(url or "")).netloc.lower().lstrip("www.")

    @staticmethod
    def _market_portal_browser_domains() -> set[str]:
        # Low-primary-signal aggregator/portal domains.  These hosts
        # *re-publish* third-party content, generate algorithmic listicles,
        # or surface ad-driven feeds rather than originating primary
        # evidence.  The set is intentionally cross-topic so the same
        # denylist applies to market, science, policy, and software
        # research.  Authoritative primary-source domains (sec.gov,
        # fred.stlouisfed.org, arxiv.org, nature.com, reuters.com,
        # apnews.com, etc.) must NEVER appear here.
        return {
            # Finance/news aggregators that re-publish wires.
            "finance.yahoo.com",
            "yahoo.com",
            "news.yahoo.com",
            "marketwatch.com",
            "cnbc.com",
            "news.google.com",
            "news.bing.com",
            "msn.com",
            "businessinsider.com",
            "www.businessinsider.com",
            # Algorithmic-listicle / opinion-aggregator hosts.
            "fool.com",
            "www.fool.com",
            "zacks.com",
            "www.zacks.com",
            "investorplace.com",
            "www.investorplace.com",
            "benzinga.com",
            "www.benzinga.com",
            "thestreet.com",
            "www.thestreet.com",
            "seekingalpha.com",
            "www.seekingalpha.com",
            "247wallst.com",
            # Cross-topic content farms.
            "medium.com",
            "substack.com",
            "quora.com",
            "answers.com",
        }

    def _filter_market_browser_sources(
        self,
        sources: list[ResearchSource],
        navigation_seed_sources: list[ResearchSource],
    ) -> list[ResearchSource]:
        blocked_domains = self._market_portal_browser_domains()
        if not blocked_domains:
            return list(sources)

        def _is_blocked(url: str) -> bool:
            domain = self._browser_source_domain(url)
            if not domain:
                return False
            if domain in blocked_domains:
                return True
            # Catch subdomains of blocked hosts (e.g. ``finance.yahoo.com``
            # vs ``yahoo.com``) so the filter cannot be bypassed by a
            # different subdomain on the same operator.
            for blocked in blocked_domains:
                if domain == blocked or domain.endswith(f".{blocked}"):
                    return True
            return False

        preferred: list[ResearchSource] = []
        blocked: list[ResearchSource] = []
        for source in sources:
            if _is_blocked(str(source.url or "")):
                blocked.append(source)
            else:
                preferred.append(source)

        # Authoritative coverage = any non-blocked navigation seed OR any
        # non-blocked candidate source.  As soon as a single primary-source
        # alternative exists we drop blocked aggregators completely — they
        # never strengthen the evidence base, only dilute it.
        has_authoritative_seed = any(
            not _is_blocked(str(seed.url or ""))
            for seed in navigation_seed_sources
        )
        if preferred or has_authoritative_seed:
            return preferred
        # Last-resort fallback: nothing authoritative was retrieved at all.
        # Keep blocked entries only so the run does not stall, but flag them.
        for source in blocked:
            flags = list(source.quality_flags or [])
            if "low-primary-signal-portal" not in flags:
                flags.append("low-primary-signal-portal")
                source.quality_flags = flags
        return blocked

    def _interleave_browser_sources_by_domain(
        self,
        sources: list[ResearchSource],
        *,
        prioritized_urls: set[str] | None = None,
    ) -> list[ResearchSource]:
        prioritized_urls = prioritized_urls or set()
        deduped: list[ResearchSource] = []
        seen_urls: set[str] = set()
        for source in sources:
            clean_url = str(source.url or "").strip()
            if not clean_url or clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            deduped.append(source)

        prioritized: list[ResearchSource] = []
        grouped: dict[str, list[ResearchSource]] = {}
        domain_order: list[str] = []
        for source in deduped:
            clean_url = str(source.url or "").strip()
            if clean_url in prioritized_urls:
                prioritized.append(source)
                continue
            domain = self._browser_source_domain(clean_url) or clean_url
            if domain not in grouped:
                grouped[domain] = []
                domain_order.append(domain)
            grouped[domain].append(source)

        interleaved = list(prioritized)
        while True:
            progressed = False
            for domain in domain_order:
                bucket = grouped.get(domain) or []
                if not bucket:
                    continue
                interleaved.append(bucket.pop(0))
                progressed = True
            if not progressed:
                break
        return interleaved

    def _augment_browser_domain_coverage(
        self,
        ranked: list[ResearchSource],
        deduped_results: list[ResearchSource],
        *,
        minimum_domains: int,
        prioritized_urls: set[str] | None = None,
    ) -> list[ResearchSource]:
        if minimum_domains <= 1:
            return list(ranked)

        prioritized_urls = prioritized_urls or set()
        augmented = list(ranked)
        seen_urls = {
            str(source.url or "").strip()
            for source in augmented
            if str(source.url or "").strip()
        }
        domains = {
            self._browser_source_domain(source.url)
            for source in augmented
            if self._browser_source_domain(source.url)
        }
        if len(domains) >= minimum_domains:
            return augmented

        fallback_pool = sorted(
            deduped_results,
            key=lambda source: float(source.score or 0.0),
            reverse=True,
        )
        for source in fallback_pool:
            clean_url = str(source.url or "").strip()
            if not clean_url or clean_url in seen_urls or clean_url in prioritized_urls:
                continue
            if not DeepResearchEngine._is_safe_public_url(clean_url):
                continue
            if DeepResearchEngine._is_search_result_url(clean_url):
                continue
            domain = self._browser_source_domain(clean_url)
            if not domain or domain in domains:
                continue
            augmented.append(source)
            seen_urls.add(clean_url)
            domains.add(domain)
            if len(domains) >= minimum_domains:
                break
        return augmented

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
                ][:12]
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
        except Exception as _claim_exc:
            import warnings
            warnings.warn(
                f"AI evidence-claim ranking failed ({_claim_exc!r}); "
                "using top-scored candidates as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        available = getattr(backend, "available", None)
        if callable(available):
            try:
                if not available():
                    return self._virtual_pc_backend()
            except Exception:
                return self._virtual_pc_backend()
        capabilities = getattr(backend, "capabilities", None)
        if callable(capabilities):
            try:
                payload = capabilities()
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("sandbox"):
                return backend
        if (
            callable(getattr(backend, "snapshot", None))
            and callable(getattr(backend, "perform", None))
        ):
            return backend
        if "sandbox" in getattr(backend, "name", "").lower():
            return backend
        return self._virtual_pc_backend()

    @staticmethod
    def _backend_capability_payload(backend: Any) -> dict[str, Any] | None:
        capabilities = getattr(backend, "capabilities", None)
        if not callable(capabilities):
            return None
        try:
            payload = capabilities()
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    @classmethod
    def _backend_is_sandbox(cls, backend: Any) -> bool:
        if isinstance(backend, VirtualDesktopSandboxBackend):
            return True
        payload = cls._backend_capability_payload(backend)
        if isinstance(payload, dict) and payload.get("sandbox"):
            return True
        return "sandbox" in str(getattr(backend, "name", "") or "").lower()

    @classmethod
    def _backend_supports_durable_workspace(cls, backend: Any) -> bool:
        if cls._backend_is_sandbox(backend):
            return True
        backend_name = str(getattr(backend, "name", "") or "").lower()
        if backend_name in {"windows-uia", "touchpoint"}:
            return False
        payload = cls._backend_capability_payload(backend)
        if not isinstance(payload, dict):
            return False
        supported = {
            str(item).strip().lower()
            for item in (payload.get("capabilities") or [])
            if str(item).strip()
        }
        return bool({"write_file", "launch_app"} & supported)

    @classmethod
    def _pc_research_action_surface(cls, backend: Any) -> tuple[str, str]:
        if cls._backend_is_sandbox(backend):
            return ("sandbox.exec", "sandbox://virtual-desktop/browser-research")
        backend_name = str(getattr(backend, "name", "pc-backend") or "pc-backend")
        scheme = re.sub(r"[^a-z0-9.+-]", "-", backend_name.lower()).strip("-")
        return ("os.act", f"{scheme or 'pc-backend'}://browser-research")

    @classmethod
    def _pc_research_mode_label(cls, backend: Any) -> str:
        return (
            "sandboxed browser research"
            if cls._backend_is_sandbox(backend)
            else "live PC browser research"
        )

    def _active_pc_research_virtual_fallback(
        self,
        run_id: str,
        task: TaskSpec,
        urls: list[str],
        approval_artifact: str,
        approval_payload: dict[str, Any] | None,
        search_queries: list[str] | None = None,
    ) -> WorkerResult:
        fallback = self._virtual_pc_backend()
        try:
            (
                receipts,
                post_nodes,
                backend_name,
                interrupt_report,
                workspace_report,
                browser_findings,
            ) = self._run_pc_browser_frontier_session(
                run_id,
                task.objective,
                fallback,
                urls,
                {
                    "seed_urls": urls[:8],
                    "search_queries": list(search_queries or [])[:12],
                },
                max(1, len(urls) or 1),
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
                "status": "executed",
                "execution_mode": "virtual_sandbox_fallback",
                "candidate_urls": urls[:6],
                "approval_deferred": approval_payload,
                "receipts": receipts,
                "backend": backend_name,
                "durable_workspace": workspace_report,
                "post_snapshot_labels": [node.name for node in post_nodes if node.name][
                    :10
                ],
                "post_snapshot_node_count": len(post_nodes),
                "panel_regions": self._panel_region_summary(post_nodes),
                "interrupt_handler": interrupt_report,
                **browser_findings,
                "direct_urls": list(browser_findings.get("direct_urls") or urls[:3]),
                "judged_results": list(
                    browser_findings.get("judged_results")
                    or [
                        {
                            "url": url,
                            "quality_score": 0.3,
                            "why": "Fallback sandbox URL when no judged results were extracted.",
                        }
                        for url in urls[:3]
                    ]
                ),
                "search_queries": list(
                    browser_findings.get("search_queries") or search_queries or []
                ),
            },
        )
        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                "Live PC research is pending approval; executed the same "
                f"sandboxed browser/navigation intent inside {backend_name}."
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
                urls.extend(WorkerAgent._planning_browser_urls(payload))
                if urls:
                    return urls
                objective = str(payload.get("objective") or "")
                urls.extend(WorkerAgent._planning_urls_from_objective(objective))
                if urls:
                    return urls
        return urls

    @staticmethod
    def _planning_browser_urls(payload: dict[str, Any]) -> list[str]:
        browser_plan = payload.get("browser_research") or {}
        urls: list[str] = []
        for item in browser_plan.get("seed_urls") or []:
            normalized_url = WorkerAgent._planning_seed_url(item)
            if normalized_url:
                urls.append(normalized_url)
        for item in browser_plan.get("search_queries") or []:
            query_text = str(item or "").strip()
            if not query_text:
                continue
            urls.append(
                "https://html.duckduckgo.com/html/?q="
                + urllib.parse.quote_plus(query_text[:240])
            )
        deduped: list[str] = []
        for item in urls:
            if item not in deduped:
                deduped.append(item)
        return deduped[:40]

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
            current_web_mode = SupervisorAgent._has_explicit_current_web_cues(
                cleaned
            ) and not DeepResearchEngine._looks_like_academic_query(cleaned)
            diagnostic_queries = WorkerAgent._software_agent_diagnostic_queries(cleaned)
            query_candidates = list(diagnostic_queries)
            if not diagnostic_queries:
                query_candidates.extend(WorkerAgent._browser_objective_queries(cleaned))
            variant_query = query
            if diagnostic_queries:
                variant_query = diagnostic_queries[0]
            if query and not diagnostic_queries:
                query_candidates.append(query)
            query_candidates.extend(
                DeepResearchEngine._query_variants(variant_query, depth)
                or [variant_query]
            )
            variants: list[str] = []
            seen_variants: set[str] = set()
            for candidate in query_candidates:
                query_text = str(candidate or "").strip()[:240]
                if not query_text:
                    continue
                normalized = DeepResearchEngine._normalize_title(query_text)
                if not normalized or normalized in seen_variants:
                    continue
                seen_variants.add(normalized)
                variants.append(query_text)
            search_variant_limit = (
                12 if depth == "multi-hour" or current_web_mode else 4
            )
            for variant in variants[:search_variant_limit]:
                encoded = urllib.parse.quote_plus(variant)
                urls.append(f"https://html.duckduckgo.com/html/?q={encoded}")
            if WorkerAgent._looks_like_repository_research(cleaned):
                repo_limit = 6 if search_variant_limit > 6 else 2
                for variant in variants[:repo_limit]:
                    encoded = urllib.parse.quote_plus(variant)
                    urls.append(
                        f"https://github.com/search?type=repositories&q={encoded}"
                    )
        deduped: list[str] = []
        for item in urls:
            if item not in deduped:
                deduped.append(item)
        url_limit = 40 if depth == "multi-hour" or current_web_mode else 12
        return deduped[:url_limit]

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
    def _software_agent_diagnostic_queries(objective: str) -> list[str]:
        return DeepResearchEngine._software_agent_diagnostic_queries(objective)

    @staticmethod
    def _software_agent_diagnostic_seed_urls(objective: str) -> list[str]:
        return DeepResearchEngine._software_agent_diagnostic_seed_urls(objective)

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
    def _write_pc_research_progress(run_id: str, payload: dict[str, Any]) -> None:
        if not run_id:
            return
        path = Path("runs") / run_id / "research" / "progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        progress = {
            "run_id": run_id,
            **existing,
            **payload,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

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

        # --- V2 cognitive loop ---------------------------------------------------
        # When the task carries a non-trivial objective, invoke the V2 agent to
        # actually *act* on the desktop rather than just capture a passive snapshot.
        # This upgrades the pc-control worker from snapshot-only to full cognitive
        # loop execution.  Failures here are non-fatal: the baseline snapshot result
        # is returned with reduced confidence.
        objective = (task.objective or "").strip()
        v2_evidence: dict[str, Any] | None = None
        v2_artifacts: list[str] = []
        v2_confidence_boost: float = 0.0

        _is_snapshot_only_objective = (
            not objective
            or re.search(
                r"\b(snapshot|capture|context|scan|observe|status)\b",
                objective,
                flags=re.IGNORECASE,
            )
            is not None
        ) and not re.search(
            r"\b(click|type|navigate|open|close|launch|move|drag|fill|search|write|execute)\b",
            objective,
            flags=re.IGNORECASE,
        )

        if objective and not _is_snapshot_only_objective:
            try:
                service = DesktopWorkflowService(workspace_root=Path("runs") / run_id)
                service.ensure_universal_mode_v2(backend, max_steps=20)
                v2_result = service.execute(objective, backend)
                v2_receipts = v2_result.get("receipts") or []
                cognitive_receipt = next(
                    (
                        r
                        for r in reversed(v2_receipts)
                        if r.get("action_type") == "universal_agent_run"
                    ),
                    None,
                )
                if cognitive_receipt:
                    inner = cognitive_receipt.get("receipt") or {}
                    v2_evidence = {
                        "source": artifact,
                        "claim": (
                            "V2 cognitive loop executed against the live desktop "
                            "for the stated objective."
                        ),
                        "run_id": str(inner.get("run_id") or ""),
                        "success": bool(inner.get("success")),
                        "steps": int(inner.get("steps") or 0),
                        "mcts_simulations": int(inner.get("mcts_simulations") or 0),
                        "avg_latency_ms": float(inner.get("avg_latency_ms") or 0.0),
                        "model_used_ratio": float(inner.get("model_used_ratio") or 0.0),
                        "node_count": len(nodes),
                        "backend": getattr(backend, "name", "pc-backend"),
                        "browser_context_detected": bool(browserish),
                        "browser_context": browserish[:5],
                    }
                    v2_artifacts = [
                        str(a) for a in (v2_result.get("artifacts") or []) if a
                    ]
                    v2_confidence_boost = 0.12 if bool(inner.get("success")) else 0.04
            except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
                # V2 wiring failed; degrade gracefully to snapshot-only mode.
                pass
        # -------------------------------------------------------------------------

        base_evidence = {
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
        evidence_list = [v2_evidence or base_evidence]
        if v2_evidence:
            evidence_list.append(base_evidence)

        return WorkerResult(
            task_id=task.task_id,
            role=task.role,
            summary=(
                f"Captured {len(nodes)} desktop/browser nodes from "
                f"{getattr(backend, 'name', 'pc-backend')}. Visible context "
                f"included: {names}."
            ),
            artifacts=[artifact] + v2_artifacts,
            evidence=evidence_list,
            confidence=min(0.90, 0.78 + v2_confidence_boost),
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
        navigation_limit: int = 5,
        objective: str = "",
        frontier_state: dict[str, Any] | None = None,
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

        frontier_focus = self._perform_frontier_browser_action(
            backend,
            pre_nodes,
            objective,
            "window",
            "focus",
            frontier_state=frontier_state,
        )
        if frontier_focus is not None:
            focus_result, focus_attempts = frontier_focus
        else:
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

        navigation_urls = [
            url
            for url in urls
            if isinstance(url, str)
            and DeepResearchEngine._is_safe_public_url(url)
            and not DeepResearchEngine._is_search_result_url(url)
        ]
        if not navigation_urls:
            navigation_urls = [url for url in urls if isinstance(url, str) and url]
        navigation_urls = navigation_urls[: max(1, int(navigation_limit or 5))]

        pre_nodes, browser_surface_report = self._ensure_browser_surface(
            backend,
            pre_nodes,
            navigation_urls[0] if navigation_urls else "",
        )
        if browser_surface_report is not None:
            receipts.append(
                {
                    "step": "ensure-browser-surface",
                    "result": browser_surface_report,
                }
            )

        post_nodes = pre_nodes
        interrupt_reports: list[dict[str, Any]] = []
        if not navigation_urls:
            receipts.append(
                {
                    "step": "navigate-candidate",
                    "result": {"status": "no-url-candidates"},
                    "attempts": [],
                }
            )
        for index, url in enumerate(navigation_urls, start=1):
            frontier_navigation = self._perform_frontier_browser_action(
                backend,
                post_nodes,
                objective,
                "address",
                "set_text",
                value=url,
                frontier_state={
                    **(frontier_state or {}),
                    "step": "navigate-candidate",
                    "url": url,
                    "index": index,
                },
            )
            if frontier_navigation is not None:
                navigate_result, navigate_attempts = frontier_navigation
            else:
                navigate_result, navigate_attempts = (
                    self._perform_with_selector_fallback(
                        backend,
                        "set_text",
                        self._browser_region_selectors(post_nodes, "address"),
                        value=url,
                    )
                )
            receipts.append(
                {
                    "step": "navigate-candidate",
                    "index": index,
                    "url": url,
                    "result": navigate_result,
                    "attempts": navigate_attempts,
                }
            )

            hydration_result = self._hydrate_browser_navigation_context(
                backend,
                url,
            )
            if hydration_result is not None:
                receipts.append(
                    {
                        "step": "hydrate-browser-content",
                        "index": index,
                        "url": url,
                        "result": hydration_result,
                    }
                )

            post_nodes = backend.snapshot()
            frontier_content_focus = self._perform_frontier_browser_action(
                backend,
                post_nodes,
                objective,
                "content",
                "focus",
                frontier_state={
                    **(frontier_state or {}),
                    "step": "focus-content-region",
                    "url": url,
                    "index": index,
                },
            )
            if frontier_content_focus is not None:
                content_focus_result, content_attempts = frontier_content_focus
            else:
                content_focus_result, content_attempts = (
                    self._perform_with_selector_fallback(
                        backend,
                        "focus",
                        self._browser_region_selectors(post_nodes, "content"),
                    )
                )
            receipts.append(
                {
                    "step": "focus-content-region",
                    "index": index,
                    "url": url,
                    "result": content_focus_result,
                    "attempts": content_attempts,
                }
            )

            post_nodes = backend.snapshot()
            interrupt_report = self._resolve_ephemeral_blockers(
                backend,
                post_nodes,
                url,
            )
            interrupt_reports.append(interrupt_report)
            if interrupt_report.get("triggered"):
                receipts.append(
                    {
                        "step": "interrupt-handler",
                        "index": index,
                        "url": url,
                        "result": interrupt_report,
                    }
                )
                post_nodes = backend.snapshot()

        interrupt_report = {
            "triggered": any(item.get("triggered") for item in interrupt_reports),
            "navigated_urls": navigation_urls,
            "reports": interrupt_reports,
        }

        return (
            receipts,
            post_nodes,
            getattr(backend, "name", "pc-backend"),
            interrupt_report,
            workspace_report,
        )

    def _hydrate_browser_navigation_context(
        self,
        backend: Any,
        url: str,
    ) -> dict[str, Any] | None:
        hydrator = getattr(backend, "hydrate_browser_page", None)
        if not callable(hydrator):
            return None

        preview = self._browser_page_preview(url)
        title = str(
            preview.get("page_title")
            or self.research_engine._label_from_url(str(url or ""))[:160]
        ).strip()
        text = str(self._deep_page_read(url) or preview.get("page_excerpt") or "").strip()
        if not title and not text:
            return {
                "status": "empty",
                "url": url,
                "content_chars": 0,
            }

        try:
            backend_receipt = hydrator(
                url,
                title=title,
                text=text,
                source="research-engine-fetch",
            )
        except Exception as exc:
            return {
                "status": "error",
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "status": "hydrated",
            "url": url,
            "title": title,
            "content_chars": len(text),
            "backend": getattr(backend, "name", "pc-backend"),
            "receipt": backend_receipt,
        }

    def _prepare_cross_surface_workspace(
        self,
        backend: Any,
        nodes: list[UiNode],
        run_id: str,
    ) -> dict[str, Any]:
        if not self._backend_supports_durable_workspace(backend):
            return {
                "triggered": False,
                "status": "skipped",
                "reason": "backend-lacks-durable-workspace-ops",
                "backend": getattr(backend, "name", "pc-backend"),
            }

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
            "status": "prepared",
            "report_path": report_path,
            "launch_result": launch_result,
            "focus_result": focus_result,
            "focus_attempts": focus_attempts,
            "file_result": file_result,
            "canvas_result": canvas_result,
            "canvas_attempts": canvas_attempts,
        }

    def _ensure_browser_surface(
        self,
        backend: Any,
        nodes: list[UiNode],
        seed_url: str = "",
    ) -> tuple[list[UiNode], dict[str, Any] | None]:
        if self._has_browser_surface(nodes):
            return (
                nodes,
                {
                    "status": "already-open",
                    "backend": getattr(backend, "name", "pc-backend"),
                },
            )
        if self._backend_is_sandbox(backend):
            return (
                nodes,
                {
                    "status": "no-browser-detected",
                    "backend": getattr(backend, "name", "pc-backend"),
                },
            )

        launch_targets: list[str] = []
        if DeepResearchEngine._is_safe_public_url(seed_url):
            launch_targets.append(seed_url)
        explicit_browser = str(os.environ.get("AGENTOS_BROWSER_APP", "") or "").strip()
        if explicit_browser:
            launch_targets.append(explicit_browser)
        launch_targets.extend(["msedge", "chrome", "firefox"])

        attempts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for launch_target in launch_targets:
            if launch_target in seen:
                continue
            seen.add(launch_target)
            try:
                launch_result = self._json_or_text(
                    backend.perform(
                        UiAction("launch_app", launch_target, value=launch_target)
                    )
                )
            except Exception as exc:
                attempts.append(
                    {
                        "target": launch_target,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            attempts.append(
                {
                    "target": launch_target,
                    "status": "launched",
                    "result": launch_result,
                }
            )
            try:
                refreshed = backend.snapshot()
            except Exception as exc:
                attempts[-1]["snapshot_error"] = f"{type(exc).__name__}: {exc}"
                continue
            if self._has_browser_surface(refreshed):
                return (
                    refreshed,
                    {
                        "status": "launched",
                        "backend": getattr(backend, "name", "pc-backend"),
                        "launch_target": launch_target,
                        "attempts": attempts,
                    },
                )

        return (
            nodes,
            {
                "status": "unavailable",
                "backend": getattr(backend, "name", "pc-backend"),
                "attempts": attempts,
            },
        )

    @staticmethod
    def _has_browser_surface(nodes: list[UiNode]) -> bool:
        browser_markers = (
            "browser",
            "edge",
            "chrome",
            "firefox",
            "brave",
            "safari",
            "duckduckgo",
            "address",
            "tab",
            "127.0.0.1",
            "http://",
            "https://",
        )
        for node in nodes:
            node_name = str(node.name or "").lower()
            node_role = str(node.role or "").lower()
            if any(marker in node_name for marker in browser_markers):
                return True
            node_url = str((node.metadata or {}).get("url") or "").lower()
            if node_role in {"document", "window"} and any(
                marker in node_url for marker in ("http://", "https://")
            ):
                return True
        return False

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

    def _perform_frontier_browser_action(
        self,
        backend: Any,
        nodes: list[UiNode],
        objective: str,
        region: str,
        action_type: str,
        value: str | None = None,
        frontier_state: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]] | None:
        selectors, attempt = self._resolve_frontier_browser_selector(
            nodes,
            objective,
            region,
            frontier_state=frontier_state,
        )
        if not selectors or attempt is None:
            return None
        attempts: list[dict[str, Any]] = []
        for selector in selectors:
            result = self._json_or_text(
                backend.perform(UiAction(action_type, selector, value=value))
            )
            status = ""
            if isinstance(result, dict):
                status = str(result.get("status") or "")
            frontier_attempt = dict(attempt)
            frontier_attempt["selector"] = selector
            frontier_attempt["status"] = status or "unknown"
            attempts.append(frontier_attempt)
            if status != "selector-not-found":
                return result, attempts
        return None

    def _resolve_frontier_browser_selector(
        self,
        nodes: list[UiNode],
        objective: str,
        region: str,
        frontier_state: dict[str, Any] | None = None,
    ) -> tuple[list[str] | None, dict[str, Any] | None]:
        if self.frontier_client is None:
            return None, None
        prompt_payload = self._build_browser_frontier_prompt(
            nodes,
            objective,
            region,
            frontier_state=frontier_state,
        )
        if prompt_payload is None:
            return None, None
        prompt, selector_map = prompt_payload
        try:
            decision = self.frontier_client.choose_action(prompt)
        except Exception as exc:
            return None, {"strategy": "frontier", "error": str(exc)[:240]}
        selectors = selector_map.get(int(decision.target_id or 0))
        if not selectors:
            return None, {
                "strategy": "frontier",
                "action": decision.action,
                "target_id": decision.target_id,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "status": "no-selector",
            }
        return selectors, {
            "strategy": "frontier",
            "action": decision.action,
            "target_id": decision.target_id,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
        }

    def _build_browser_frontier_prompt(
        self,
        nodes: list[UiNode],
        objective: str,
        region: str,
        frontier_state: dict[str, Any] | None = None,
    ) -> tuple[FrontierPrompt, dict[int, list[str]]] | None:
        candidates = self._browser_frontier_candidate_nodes(nodes, region)
        if not candidates:
            return None
        annotated_png, mark_payload, selector_map = self._render_browser_frontier_frame(
            candidates,
            region,
        )
        state_context = {
            "target_region": region,
            "frontier_state": frontier_state or {},
            "candidate_nodes": [
                {
                    "node_id": node.node_id,
                    "role": node.role,
                    "name": node.name,
                    "bounds": list(node.bounds) if node.bounds else None,
                }
                for node in candidates[:10]
            ],
        }
        prompt = FrontierPrompt(
            objective=(
                f"{objective}\n"
                f"Select the browser {region} control that best advances the current research step."
            ),
            annotated_png=annotated_png,
            mark_payload=mark_payload,
            tool_context=(
                "Choose the correct browser node for the requested region. "
                "The local executor will map the chosen tag to a real selector."
            ),
            state_context=state_context,
            confidence_floor=0.35,
        )
        return prompt, selector_map

    @staticmethod
    def _browser_frontier_candidate_nodes(
        nodes: list[UiNode],
        region: str,
    ) -> list[UiNode]:
        matches: list[UiNode] = []
        for node in nodes:
            node_id = node.node_id.lower()
            role = node.role.lower()
            name = node.name.lower()
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if region == "window" and (
                node_id == "window-browser" or role == "window" or "browser" in name
            ):
                matches.append(node)
            elif region == "address" and (
                node_id == "browser-address-bar"
                or (role == "edit" and "address" in name)
                or panel == "toolbar"
            ):
                matches.append(node)
            elif region == "content" and (
                node_id == "browser-main-doc"
                or role == "document"
                or panel in {"primary", "document"}
            ):
                matches.append(node)
        if matches:
            return matches[:12]
        enabled_nodes = [node for node in nodes if node.enabled]
        return enabled_nodes[:12]

    def _render_browser_frontier_frame(
        self,
        nodes: list[UiNode],
        region: str,
    ) -> tuple[bytes, dict[str, Any], dict[int, list[str]]]:
        normalized_bounds = [
            self._frontier_node_bounds(node, index) for index, node in enumerate(nodes)
        ]
        canvas_width = max(
            1280, max((x + width + 40) for x, _, width, _ in normalized_bounds)
        )
        canvas_height = max(
            900, max((y + height + 40) for _, y, _, height in normalized_bounds)
        )
        image = Image.new("RGB", (canvas_width, canvas_height), (244, 240, 232))
        draw = ImageDraw.Draw(image)
        selector_map: dict[int, list[str]] = {}
        marks: list[dict[str, Any]] = []
        palette = {
            "window": ((32, 80, 129), (225, 236, 248)),
            "address": ((138, 82, 24), (250, 239, 220)),
            "content": ((34, 105, 74), (226, 244, 236)),
        }
        outline_color, fill_color = palette.get(region, ((70, 70, 70), (245, 245, 245)))
        for index, node in enumerate(nodes, start=1):
            x, y, width, height = normalized_bounds[index - 1]
            draw.rectangle(
                [x, y, x + width, y + height],
                fill=fill_color,
                outline=outline_color,
                width=3,
            )
            tag_top = max(4, y - 26)
            draw.rectangle(
                [x, tag_top, x + 34, tag_top + 22],
                fill=outline_color,
                outline=outline_color,
            )
            draw.text((x + 8, tag_top + 4), str(index), fill=(255, 255, 255))
            label = (node.name or node.node_id or node.role)[:56]
            subtitle = f"{node.role} | {node.node_id}"[:64]
            draw.text((x + 10, y + 10), label, fill=(24, 24, 24))
            draw.text((x + 10, y + 30), subtitle, fill=(72, 72, 72))
            selectors = self._frontier_selectors_for_node(node, region)
            if selectors:
                selector_map[index] = selectors
            marks.append(
                {
                    "id": index,
                    "label": label,
                    "role": node.role,
                    "node_id": node.node_id,
                    "selectors": selectors,
                    "bounds": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                    "metadata": dict(node.metadata or {}),
                }
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), {"region": region, "marks": marks}, selector_map

    @staticmethod
    def _frontier_node_bounds(node: UiNode, index: int) -> tuple[int, int, int, int]:
        if node.bounds:
            x, y, width, height = node.bounds
            return max(x, 12), max(y, 12), max(width, 140), max(height, 56)
        return 40, 40 + (index * 88), 760, 64

    @staticmethod
    def _frontier_selector_for_node(node: UiNode) -> str | None:
        if node.node_id:
            return f"node_id={node.node_id}"
        if node.role and node.name:
            return f"role={node.role};name={node.name}"
        if node.name:
            return f"name={node.name}"
        return WorkerAgent._point_selector_for_node(node)

    @classmethod
    def _frontier_selectors_for_node(
        cls,
        node: UiNode,
        region: str,
    ) -> list[str]:
        selectors: list[str] = []
        name = node.name.lower().strip()
        role = node.role.lower().strip()
        if region == "window":
            if name:
                selectors.append(f"name={name}")
            if role == "window":
                selectors.append("role=window;name=browser")
        elif region == "address":
            if role == "edit" and name:
                selectors.append(f"role=edit;name={name}")
            if name:
                selectors.append(f"name={name}")
        elif region == "content":
            if role == "document":
                selectors.append("role=document")
            panel = str((node.metadata or {}).get("panel_type") or "").lower()
            if panel in {"primary", "document"}:
                selectors.append(f"panel_type={panel}")
            if name:
                selectors.append(f"name={name}")
        if node.role and node.name:
            selectors.append(f"role={role};name={name}")
        point_selector = cls._point_selector_for_node(node)
        if point_selector:
            selectors.append(point_selector)
        node_selector = cls._frontier_selector_for_node(node)
        if node_selector:
            selectors.append(node_selector)
        deduped: list[str] = []
        for selector in selectors:
            if selector and selector not in deduped:
                deduped.append(selector)
        return deduped

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
