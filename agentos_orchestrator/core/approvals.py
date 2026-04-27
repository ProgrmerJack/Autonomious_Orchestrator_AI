from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import ActionRequest, JsonObject, new_id, utc_now


@dataclass(slots=True)
class ApprovalTicket:
    approval_id: str
    token: str
    run_id: str
    action: JsonObject
    reasons: list[str]
    status: str
    created_at: str
    resolved_at: str | None = None


class ApprovalRequired(RuntimeError):
    def __init__(self, ticket: ApprovalTicket) -> None:
        self.ticket = ticket
        super().__init__(f"Approval required: {ticket.approval_id} ({ticket.status})")


class ApprovalStore:
    """Asynchronous human approval ledger."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                    CREATE TABLE IF NOT EXISTS approvals (
                        approval_id TEXT PRIMARY KEY,
                        token TEXT NOT NULL UNIQUE,
                        run_id TEXT NOT NULL,
                        action_json TEXT NOT NULL,
                        reasons_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        resolved_at TEXT
                    )
                    """
                )

    def request(
        self,
        run_id: str,
        action: ActionRequest,
        reasons: list[str],
    ) -> ApprovalTicket:
        ticket = ApprovalTicket(
            approval_id=new_id("approval"),
            token=secrets.token_urlsafe(24),
            run_id=run_id,
            action=asdict(action),
            reasons=reasons,
            status="pending",
            created_at=utc_now(),
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO approvals(
                        approval_id,
                        token,
                        run_id,
                        action_json,
                        reasons_json,
                        status,
                        created_at,
                        resolved_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket.approval_id,
                        ticket.token,
                        ticket.run_id,
                        json.dumps(ticket.action, sort_keys=True),
                        json.dumps(ticket.reasons, sort_keys=True),
                        ticket.status,
                        ticket.created_at,
                        ticket.resolved_at,
                    ),
                )
        return ticket

    def approve(self, token: str) -> ApprovalTicket:
        return self._resolve(token, "approved")

    def deny(self, token: str) -> ApprovalTicket:
        return self._resolve(token, "denied")

    def get(self, token: str) -> ApprovalTicket | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    approval_id,
                    token,
                    run_id,
                    action_json,
                    reasons_json,
                    status,
                    created_at,
                    resolved_at
                FROM approvals
                WHERE token = ?
                """,
                (token,),
            ).fetchone()
        if row is None:
            return None
        return self._from_row(row)

    def is_approved(self, token: str | None) -> bool:
        if token is None:
            return False
        ticket = self.get(token)
        return ticket is not None and ticket.status == "approved"

    def is_approved_for(
        self,
        token: str | None,
        action: ActionRequest,
    ) -> bool:
        if token is None:
            return False
        ticket = self.get(token)
        if ticket is None or ticket.status != "approved":
            return False
        expected = asdict(action)
        expected["approval_token"] = None
        recorded = dict(ticket.action)
        recorded["approval_token"] = None
        if recorded == expected:
            return True
        return (
            str(recorded.get("action_type")) == action.action_type
            and str(recorded.get("target")) == action.target
        )

    def find_approved_for(
        self,
        run_id: str,
        action: ActionRequest,
    ) -> ApprovalTicket | None:
        expected = asdict(action)
        expected["approval_token"] = None
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    approval_id,
                    token,
                    run_id,
                    action_json,
                    reasons_json,
                    status,
                    created_at,
                    resolved_at
                FROM approvals
                WHERE run_id = ? AND status = 'approved'
                ORDER BY resolved_at DESC
                """,
                (run_id,),
            ).fetchall()
        for row in rows:
            ticket = self._from_row(row)
            recorded = dict(ticket.action)
            recorded["approval_token"] = None
            if recorded == expected:
                return ticket
        for row in rows:
            ticket = self._from_row(row)
            recorded = dict(ticket.action)
            if (
                str(recorded.get("action_type")) == action.action_type
                and str(recorded.get("target")) == action.target
            ):
                return ticket
        return None

    def list_pending(self, run_id: str | None = None) -> list[ApprovalTicket]:
        query = (
            "SELECT approval_id, token, run_id, action_json, reasons_json, "
            "status, created_at, resolved_at FROM approvals "
            "WHERE status = 'pending'"
        )
        parameters: list[object] = []
        if run_id is not None:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY created_at ASC"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._from_row(row) for row in rows]

    def _resolve(self, token: str, status: str) -> ApprovalTicket:
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE approvals
                    SET status = ?, resolved_at = ?
                    WHERE token = ?
                    """,
                    (status, now, token),
                )
        ticket = self.get(token)
        if ticket is None:
            raise KeyError("approval token was not found")
        return ticket

    @staticmethod
    def _from_row(row: tuple) -> ApprovalTicket:
        return ApprovalTicket(
            approval_id=row[0],
            token=row[1],
            run_id=row[2],
            action=json.loads(row[3]),
            reasons=json.loads(row[4]),
            status=row[5],
            created_at=row[6],
            resolved_at=row[7],
        )
