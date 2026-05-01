"""Deterministic safety gates for frontier-controlled OS execution.

Frontier models may plan creatively, but the local executor must remain rigid.
This verifier is the layer between macro-planning and physical execution. It
translates UI actions into simple constraints and rejects plans that violate
immutable OS rules such as path containment and destructive-action approval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from agentos_orchestrator.os_control.base import UiAction


@dataclass(slots=True)
class SafetyViolation:
    code: str
    message: str
    action_index: int | None = None
    action_selector: str = ""


@dataclass(slots=True)
class SafetyDecision:
    allowed: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    solver: str = "deterministic"

    @property
    def reason(self) -> str:
        if self.allowed:
            return "allowed"
        return "; ".join(v.message for v in self.violations)


@dataclass(slots=True)
class SafetyPolicy:
    """Immutable rules for the local physical executor."""

    allowed_roots: list[Path] = field(default_factory=list)
    require_approval_for_destructive: bool = True
    allow_network_tools: bool = False
    max_actions_per_plan: int = 50
    destructive_keywords: frozenset[str] = frozenset(
        {
            "delete",
            "remove",
            "trash",
            "erase",
            "format",
            "wipe",
            "reset",
            "shutdown",
            "reboot",
            "rm ",
            "rmdir",
            "del ",
            "remove-item",
        }
    )
    high_stakes_actions: frozenset[str] = frozenset(
        {"delete", "remove", "move", "format", "trade", "submit", "purchase"}
    )


class FormalSafetyVerifier:
    """Verify proposed OS actions before backend execution."""

    _PATH_KEYS = {
        "path",
        "file",
        "folder",
        "source",
        "destination",
        "source_path",
        "dest_path",
        "target_path",
        "output_path",
        "cwd",
    }

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()

    def verify_action(
        self,
        action: UiAction,
        objective: str = "",
        approval_token: str | None = None,
    ) -> SafetyDecision:
        return self.verify_plan(
            [action], objective=objective, approval_token=approval_token
        )

    def verify_plan(
        self,
        actions: list[UiAction],
        objective: str = "",
        approval_token: str | None = None,
    ) -> SafetyDecision:
        del objective
        violations: list[SafetyViolation] = []
        if len(actions) > self.policy.max_actions_per_plan:
            violations.append(
                SafetyViolation(
                    code="plan_too_long",
                    message=f"Plan has {len(actions)} actions, limit is {self.policy.max_actions_per_plan}",
                )
            )

        for index, action in enumerate(actions):
            violations.extend(self._check_destructive(action, index, approval_token))
            violations.extend(self._check_paths(action, index))
            violations.extend(self._check_shell_like_payloads(action, index))
            violations.extend(self._check_network_tool(action, index))

        solver = "z3" if self._z3_can_prove(len(violations) == 0) else "deterministic"
        return SafetyDecision(
            allowed=not violations, violations=violations, solver=solver
        )

    def _check_destructive(
        self,
        action: UiAction,
        index: int,
        approval_token: str | None,
    ) -> list[SafetyViolation]:
        haystack = " ".join(
            str(part).lower()
            for part in [
                action.action_type,
                action.selector,
                action.value,
                action.metadata,
            ]
            if part is not None
        )
        destructive = any(
            _keyword_matches_haystack(keyword, haystack)
            for keyword in self.policy.destructive_keywords
        )
        high_stakes_action = (
            action.action_type.lower() in self.policy.high_stakes_actions
        )
        if (
            destructive or high_stakes_action
        ) and self.policy.require_approval_for_destructive:
            if not approval_token:
                return [
                    SafetyViolation(
                        code="destructive_requires_approval",
                        message=f"Action requires approval: {action.action_type} {action.selector}",
                        action_index=index,
                        action_selector=action.selector,
                    )
                ]
        return []

    def _check_paths(self, action: UiAction, index: int) -> list[SafetyViolation]:
        roots = [self._norm_path(root) for root in self.policy.allowed_roots]
        if not roots:
            return []
        violations: list[SafetyViolation] = []
        paths = self._extract_paths(action)
        for key, raw_path in paths:
            candidate = self._norm_path(Path(raw_path))
            if not self._path_in_roots(candidate, roots):
                violations.append(
                    SafetyViolation(
                        code="path_outside_allowed_roots",
                        message=f"Path for {key} is outside allowed roots: {raw_path}",
                        action_index=index,
                        action_selector=action.selector,
                    )
                )
        return violations

    def _check_shell_like_payloads(
        self, action: UiAction, index: int
    ) -> list[SafetyViolation]:
        payload = " ".join(
            str(action.metadata.get(key, ""))
            for key in {"command", "script", "powershell", "shell"}
        ).lower()
        if not payload:
            return []
        dangerous_patterns = [
            r"\brm\s+-rf\b",
            r"\bdel\s+/[sq]\b",
            r"\bremove-item\b.*\b-recurse\b",
            r"\bformat\b",
            r"\bshutdown\b",
            r"\btaskkill\b.*\b/f\b",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, payload):
                return [
                    SafetyViolation(
                        code="dangerous_command",
                        message=f"Dangerous shell payload blocked: {pattern}",
                        action_index=index,
                        action_selector=action.selector,
                    )
                ]
        return []

    def _check_network_tool(
        self, action: UiAction, index: int
    ) -> list[SafetyViolation]:
        if self.policy.allow_network_tools:
            return []
        if action.metadata.get("allow_network") is True:
            return [
                SafetyViolation(
                    code="network_tool_blocked",
                    message="Network tool access is disabled by safety policy",
                    action_index=index,
                    action_selector=action.selector,
                )
            ]
        return []

    def _extract_paths(self, action: UiAction) -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        for key, value in action.metadata.items():
            if key.lower() not in self._PATH_KEYS:
                continue
            if isinstance(value, (str, os.PathLike)) and str(value).strip():
                paths.append((key, str(value)))
        return paths

    @staticmethod
    def _norm_path(path: Path) -> Path:
        try:
            return path.expanduser().resolve(strict=False)
        except OSError:
            return Path(os.path.abspath(os.fspath(path))).resolve(strict=False)

    @staticmethod
    def _path_in_roots(candidate: Path, roots: list[Path]) -> bool:
        candidate_norm = os.path.normcase(os.fspath(candidate))
        for root in roots:
            root_norm = os.path.normcase(os.fspath(root))
            try:
                common = os.path.commonpath([candidate_norm, root_norm])
            except ValueError:
                continue
            if common == root_norm:
                return True
        return False

    @staticmethod
    def _z3_can_prove(no_violations: bool) -> bool:
        """Use Z3 when installed; otherwise deterministic checks are authoritative."""
        try:
            import z3  # type: ignore[import-not-found]
        except ImportError:
            return False
        allowed = z3.Bool("plan_allowed")
        solver = z3.Solver()
        solver.add(allowed == bool(no_violations))
        return solver.check() == z3.sat and bool(z3.is_true(solver.model()[allowed]))


def default_safety_verifier(workspace_root: str | Path) -> FormalSafetyVerifier:
    return FormalSafetyVerifier(
        SafetyPolicy(allowed_roots=[Path(workspace_root).resolve(strict=False)])
    )


def _keyword_matches_haystack(keyword: str, haystack: str) -> bool:
    normalized = keyword.strip().lower()
    if not normalized:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None
