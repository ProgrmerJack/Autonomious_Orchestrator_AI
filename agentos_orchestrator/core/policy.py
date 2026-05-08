from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .types import ActionRequest


class PermissionViolation(RuntimeError):
    pass


@dataclass(slots=True)
class PermissionDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    requires_approval: bool = False


class PermissionPolicy:
    """Default-deny boundary mapper for worker-declared actions."""

    def __init__(self, document: dict) -> None:
        self.document = document
        self.name = str(document.get("name", "unnamed"))
        self.default = str(document.get("default", "deny"))
        self.allow = document.get("allow", {})
        self.forbid = document.get("forbid", {})
        self.require_approval = document.get("require_approval", {})

    @classmethod
    def from_file(cls, path: str | Path) -> "PermissionPolicy":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def evaluate(self, request: ActionRequest) -> PermissionDecision:
        reasons: list[str] = []
        approval_needed = self._requires_approval(request)
        forbidden_actions = self.forbid.get("actions", [])
        if self._matches_any(request.action_type, forbidden_actions):
            return PermissionDecision(
                False,
                [f"action '{request.action_type}' is forbidden"],
            )

        forbidden_paths = self.forbid.get("paths", [])
        if self._target_matches_path(request.target, forbidden_paths):
            return PermissionDecision(
                False,
                [f"target '{request.target}' matches a forbidden path"],
            )

        for keyword in self.forbid.get("keywords", []):
            candidate = f"{request.target} {request.payload}".lower()
            if keyword.lower() in candidate:
                return PermissionDecision(
                    False,
                    [f"request contains forbidden keyword '{keyword}'"],
                )

        if approval_needed and not request.approval_token:
            return PermissionDecision(
                False,
                [f"action '{request.action_type}' requires explicit approval"],
                requires_approval=True,
            )

        allowed_by_action = self._matches_any(
            request.action_type,
            self.allow.get("actions", []),
        )
        allowed_by_path = self._target_matches_path(
            request.target,
            self.allow.get("paths", []),
        )
        allowed_by_host = self._host_allowed(request.target)

        if allowed_by_action:
            reasons.append(f"action '{request.action_type}' is allowed")
        if allowed_by_path:
            reasons.append(f"target '{request.target}' is in allowed paths")
        if allowed_by_host:
            reasons.append(f"host for '{request.target}' is allowed")

        needs_path_check = self._target_needs_path_check(request.action_type)
        if allowed_by_action and needs_path_check:
            if allowed_by_path or allowed_by_host:
                return PermissionDecision(
                    True,
                    reasons,
                    requires_approval=approval_needed,
                )
            return PermissionDecision(
                False,
                [f"target '{request.target}' is not allowed"],
            )

        if allowed_by_action:
            return PermissionDecision(
                True,
                reasons,
                requires_approval=approval_needed,
            )

        if self.default == "allow":
            return PermissionDecision(
                True,
                ["policy default is allow"],
                requires_approval=approval_needed,
            )
        return PermissionDecision(
            False,
            [f"action '{request.action_type}' is not allowed by policy"],
        )

    def assert_allowed(self, request: ActionRequest) -> PermissionDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PermissionViolation("; ".join(decision.reasons))
        return decision

    def verify_task_declarations(
        self,
        requests: list[ActionRequest],
    ) -> list[PermissionDecision]:
        return [self.assert_allowed(request) for request in requests]

    def _requires_approval(self, request: ActionRequest) -> bool:
        return self._matches_any(
            request.action_type,
            self.require_approval.get("actions", []),
        )

    def _host_allowed(self, target: str) -> bool:
        parsed = urlparse(target)
        if not parsed.hostname:
            return False
        return self._matches_any(
            parsed.hostname,
            self.allow.get("network_hosts", []),
        )

    @staticmethod
    def _target_needs_path_check(action_type: str) -> bool:
        """Return True for any action whose *target* must pass a path/host
        allow-list check before the action is permitted.

        High-blast-radius OS actions (os.act, subprocess.exec, shell.run,
        os.shell) are included alongside the original file/network trio so
        that a policy that allows ``os.act`` without specifying a target
        scope cannot be silently bypassed.
        """
        return action_type in {
            # File I/O
            "file.read",
            "file.write",
            "file.delete",
            "file.move",
            "file.exec",
            # Network
            "network.fetch",
            "network.request",
            # OS / process execution
            "os.act",
            "os.shell",
            "subprocess.exec",
            "shell.run",
            "process.spawn",
        }

    @staticmethod
    def _matches_any(value: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    @staticmethod
    def _target_matches_path(target: str, patterns: list[str]) -> bool:
        normalized = target.replace("\\", "/")
        return any(
            fnmatch.fnmatchcase(normalized, pattern.replace("\\", "/"))
            for pattern in patterns
        )
