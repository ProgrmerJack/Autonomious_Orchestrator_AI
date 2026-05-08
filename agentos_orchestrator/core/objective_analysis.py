from __future__ import annotations

import json
import os
import re
from typing import Any

from agentos_orchestrator.research import DeepResearchEngine


class ObjectiveAnalysisMixin:
    @staticmethod
    def _heuristic_planning_allowed() -> bool:
        return os.environ.get("AGENTOS_ALLOW_HEURISTIC_PLANNING", "").strip() == "1"

    def _planning_ai_provider_configured(self) -> bool:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            return True
        load_env = getattr(self.research_engine, "_load_env_from_dotenv", None)
        if callable(load_env):
            load_env()
        return bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    @staticmethod
    def _extract_first_json_object(raw: str) -> dict[str, Any] | None:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        parsed = json.loads(raw[start:end])
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _validate_objective_analysis_payload(parsed: Any) -> str | None:
        if not isinstance(parsed, dict):
            return "response must be a JSON object"
        required_int_fields = (
            "complexity_score",
            "min_source_count",
            "min_provider_count",
            "min_scholarly_sources",
            "max_retrieval_passes",
        )
        for field_name in required_int_fields:
            try:
                int(parsed.get(field_name))
            except (TypeError, ValueError):
                return f"{field_name} must be an integer"

        profile = parsed.get("profile")
        if not isinstance(profile, dict):
            return "profile must be a JSON object"
        for field_name in ("academic", "current", "comparison", "risk"):
            if not isinstance(profile.get(field_name), bool):
                return f"profile.{field_name} must be a boolean"

        try:
            contradiction_risk = float(parsed.get("max_contradiction_risk"))
        except (TypeError, ValueError):
            return "max_contradiction_risk must be a float between 0.0 and 1.0"
        if contradiction_risk < 0.0 or contradiction_risk > 1.0:
            return "max_contradiction_risk must be between 0.0 and 1.0"

        hypotheses = parsed.get("hypotheses")
        if not isinstance(hypotheses, list):
            return "hypotheses must be a JSON array"
        normalized_hypotheses = [
            str(item).strip() for item in hypotheses if str(item).strip()
        ]
        if not normalized_hypotheses:
            return "hypotheses must contain at least one non-empty string"
        return None

    def _request_valid_objective_analysis(
        self,
        system: str,
        user: str,
    ) -> dict[str, Any]:
        if not self._planning_ai_provider_configured():
            if self._heuristic_planning_allowed():
                return {}
            raise RuntimeError(
                "No LLM provider configured for AI planning. Set GEMINI_API_KEY "
                "or GOOGLE_API_KEY, or opt into heuristic planning with "
                "AGENTOS_ALLOW_HEURISTIC_PLANNING=1."
            )

        retry_prompt = user
        for attempt in range(1, 4):
            try:
                raw = self.research_engine._call_ai_text(system, retry_prompt)
                if not str(raw).strip():
                    return {}
                parsed = self._extract_first_json_object(raw)
                if parsed is None:
                    raise ValueError("response did not contain a JSON object")
                validation_error = self._validate_objective_analysis_payload(parsed)
                if validation_error is None:
                    return parsed
                raise ValueError(validation_error)
            except Exception as exc:
                retry_prompt = (
                    f"{user}\n\n"
                    f"Your previous response failed validation on attempt {attempt}: {exc}. "
                    "Return ONLY one valid JSON object containing every required key, "
                    "with correct scalar types and a non-empty hypotheses array."
                )
        return {}

    def _ai_analyze_objective(self, objective: str) -> dict[str, Any]:
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
        parsed = self._request_valid_objective_analysis(system, user)
        if not parsed:
            return baseline

        profile = baseline["profile"]
        profile.update(
            {
                "academic": bool((parsed.get("profile") or {}).get("academic")),
                "current": bool((parsed.get("profile") or {}).get("current")),
                "comparison": bool(
                    (parsed.get("profile") or {}).get("comparison")
                ),
                "risk": bool((parsed.get("profile") or {}).get("risk")),
            }
        )
        return {
            "complexity_score": max(1, min(int(parsed["complexity_score"]), 10)),
            "profile": profile,
            "min_source_count": max(4, int(parsed["min_source_count"])),
            "min_provider_count": max(1, int(parsed["min_provider_count"])),
            "min_scholarly_sources": max(0, int(parsed["min_scholarly_sources"])),
            "max_contradiction_risk": max(
                0.0,
                min(float(parsed["max_contradiction_risk"]), 1.0),
            ),
            "max_retrieval_passes": max(1, int(parsed["max_retrieval_passes"])),
            "hypotheses": [
                str(item).strip()
                for item in list(parsed.get("hypotheses") or [])
                if str(item).strip()
            ][:8],
        }

    @staticmethod
    def _heuristic_objective_analysis(objective: str) -> dict[str, Any]:
        lower = objective.lower()
        software_agent_query = DeepResearchEngine._looks_like_software_agent_query(
            objective
        )
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
        if software_agent_query and re.search(
            r"\b(compare|comparison|comparable|benchmark|claude|gpt|gemini|openhands|openclaw)\b",
            lower,
        ):
            comparison = True
        if software_agent_query and re.search(
            r"\b(fix|issue|issues|bug|bugs|failure|failures|not using|underperform|gap|gaps|shallow|template)\b",
            lower,
        ):
            risk = True
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
        if software_agent_query:
            complexity += 2
        complexity = max(1, min(complexity, 10))

        multi_hour = "[multi-hour]" in lower
        max_passes = 120 if multi_hour else (18 if complexity >= 7 else 8)
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