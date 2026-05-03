from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .types import ActionRequest, utc_now


@dataclass(slots=True)
class TrustDecision:
    level: str
    score_delta: int
    cumulative_score: int
    requires_approval: bool
    reasons: list[str] = field(default_factory=list)


class TrustMonitor:
    """Behavioral authorization outside the LLM prompt surface."""

    def __init__(
        self,
        db_path: str | Path,
        approval_threshold: int = 6,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.approval_threshold = approval_threshold
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_schema(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trust_state (
                        run_id TEXT PRIMARY KEY,
                        cumulative_score INTEGER NOT NULL,
                        level TEXT NOT NULL,
                        last_action TEXT,
                        last_target TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trust_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        action_json TEXT NOT NULL,
                        score_delta INTEGER NOT NULL,
                        cumulative_score INTEGER NOT NULL,
                        level TEXT NOT NULL,
                        reasons_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

    def assess(self, run_id: str, action: ActionRequest) -> TrustDecision:
        previous = self._load_state(run_id)
        delta, reasons = self._score(action, previous)
        cumulative = int(previous.get("cumulative_score", 0)) + delta
        level = self._level(cumulative)
        requires_approval = cumulative >= self.approval_threshold
        decision = TrustDecision(
            level=level,
            score_delta=delta,
            cumulative_score=cumulative,
            requires_approval=requires_approval,
            reasons=reasons,
        )
        self._save_state(run_id, action, decision)
        return decision

    def get_level(self, run_id: str) -> str:
        return str(self._load_state(run_id).get("level", "high"))

    def _score(
        self,
        action: ActionRequest,
        previous: dict,
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        candidate = f"{action.target} {action.payload}".lower()
        action_type = action.action_type.lower()

        if action_type in {"host.admin", "system.registry.write"}:
            score += 10
            reasons.append("admin-grade action requested")
        if action_type == "os.act":
            sandbox_target = str(action.target or "").lower()
            if sandbox_target.startswith("sandbox://"):
                reasons.append("os.act confined to sandbox isolation — no trust impact")
            else:
                score += 3
                reasons.append("high-impact execution action requested")
        elif action_type == "sandbox.exec":
            sandbox_target = str(action.target or "").lower()
            if sandbox_target.startswith("sandbox://"):
                reasons.append("sandbox-confined execution action")
            else:
                score += 2
                reasons.append("non-confined sandbox execution target")
        if self._looks_sensitive(candidate):
            score += 6
            reasons.append("target resembles a sensitive local resource")
        if self._looks_like_prompt_injection(candidate):
            score += 8
            reasons.append("payload resembles indirect prompt injection")
        if self._is_web_to_local_shift(action, previous):
            score += 4
            reasons.append("behavior shifted from web research to local probing")

        parsed = urlparse(action.target)
        last_target = str(previous.get("last_target", ""))
        last_action = str(previous.get("last_action", ""))
        last_host = urlparse(last_target).hostname
        if parsed.hostname and last_host and parsed.hostname != last_host:
            reasons.append("network host changed during active run")
            if not (
                action_type == "network.fetch"
                and last_action in {"mcp.call", "mcp.list", "network.fetch"}
            ):
                score += 1

        return score, reasons

    def _load_state(self, run_id: str) -> dict:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT cumulative_score, level, last_action, last_target
                FROM trust_state
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return {"cumulative_score": 0, "level": "high"}
        return {
            "cumulative_score": row[0],
            "level": row[1],
            "last_action": row[2],
            "last_target": row[3],
        }

    def _save_state(
        self,
        run_id: str,
        action: ActionRequest,
        decision: TrustDecision,
    ) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO trust_state(
                        run_id,
                        cumulative_score,
                        level,
                        last_action,
                        last_target,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        cumulative_score = excluded.cumulative_score,
                        level = excluded.level,
                        last_action = excluded.last_action,
                        last_target = excluded.last_target,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        decision.cumulative_score,
                        decision.level,
                        action.action_type,
                        action.target,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO trust_events(
                        run_id,
                        action_json,
                        score_delta,
                        cumulative_score,
                        level,
                        reasons_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        json.dumps(asdict(action), sort_keys=True),
                        decision.score_delta,
                        decision.cumulative_score,
                        decision.level,
                        json.dumps(decision.reasons, sort_keys=True),
                        now,
                    ),
                )

    @staticmethod
    def _level(score: int) -> str:
        if score >= 9:
            return "quarantined"
        if score >= 6:
            return "degraded"
        if score >= 3:
            return "guarded"
        return "high"

    @staticmethod
    def _looks_sensitive(candidate: str) -> bool:
        markers = (
            "/.ssh/",
            "\\.ssh\\",
            "credentials",
            "private_key",
            "system32",
            "registry",
            "appdata/roaming/microsoft/credentials",
        )
        return any(marker in candidate for marker in markers)

    @staticmethod
    def _looks_like_prompt_injection(candidate: str) -> bool:
        markers = (
            "ignore previous instructions",
            "developer message",
            "system prompt",
            "exfiltrate",
            "disable security",
        )
        return any(marker in candidate for marker in markers)

    @staticmethod
    def _is_web_to_local_shift(
        action: ActionRequest,
        previous: dict,
    ) -> bool:
        last_action = str(previous.get("last_action", ""))
        if last_action not in {"network.fetch", "mcp.call", "mcp.list"}:
            return False
        if action.action_type not in {"file.read", "file.write", "os.act"}:
            return False
        target = action.target.replace("\\", "/").lower()
        return target.startswith("c:/") or target.startswith("/")
