from __future__ import annotations

import ipaddress
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hmac import compare_digest
from urllib.parse import urlparse


_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


class GatewaySecurityError(PermissionError):
    def __init__(self, message: str, status_code: int = 403) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayAuthenticationError(GatewaySecurityError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=401)


@dataclass(slots=True)
class GatewaySession:
    session_token: str
    csrf_token: str
    issued_at: str
    expires_at: str
    unsafe_ack_value: str

    def asdict(self) -> dict[str, str]:
        return asdict(self)


class GatewaySecurityManager:
    UNSAFE_ACK_VALUE = "allow-local-side-effects"

    def __init__(
        self,
        bootstrap_token: str | None = None,
        session_ttl_seconds: int = 12 * 60 * 60,
    ) -> None:
        self.bootstrap_token = (
            bootstrap_token
            or os.environ.get("AGENTOS_DASHBOARD_BOOTSTRAP_TOKEN")
            or secrets.token_urlsafe(32)
        )
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, GatewaySession] = {}

    def create_session(
        self,
        bootstrap_token: str,
        client_host: str,
        origin: str | None = None,
    ) -> GatewaySession:
        self.assert_loopback_client(client_host)
        self.assert_local_origin(origin)
        if not compare_digest(str(bootstrap_token or ""), self.bootstrap_token):
            raise GatewayAuthenticationError("invalid bootstrap token")
        self._prune_expired_sessions()
        issued_at = datetime.now(tz=UTC)
        session = GatewaySession(
            session_token=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(24),
            issued_at=issued_at.isoformat(),
            expires_at=(
                issued_at + timedelta(seconds=self.session_ttl_seconds)
            ).isoformat(),
            unsafe_ack_value=self.UNSAFE_ACK_VALUE,
        )
        self._sessions[session.session_token] = session
        return session

    def require_session(
        self,
        session_token: str,
        client_host: str,
        origin: str | None = None,
        csrf_token: str | None = None,
        require_csrf: bool = False,
        require_unsafe_ack: bool = False,
        unsafe_ack: str | None = None,
    ) -> GatewaySession:
        self.assert_loopback_client(client_host)
        self.assert_local_origin(origin)
        self._prune_expired_sessions()
        if not session_token:
            raise GatewayAuthenticationError("missing dashboard session token")
        session = self._sessions.get(session_token)
        if session is None:
            raise GatewayAuthenticationError("dashboard session is invalid or expired")
        if require_csrf and not compare_digest(
            str(csrf_token or ""),
            session.csrf_token,
        ):
            raise GatewaySecurityError("missing or invalid CSRF token")
        if require_unsafe_ack and unsafe_ack != self.UNSAFE_ACK_VALUE:
            raise GatewaySecurityError(
                "unsafe action acknowledgement header is required"
            )
        return session

    def assert_loopback_client(self, client_host: str) -> None:
        normalized = str(client_host or "").strip().lower()
        if normalized in _LOCAL_HOSTS:
            return
        try:
            address = ipaddress.ip_address(normalized)
        except ValueError as exc:
            raise GatewaySecurityError(
                "dashboard API only accepts loopback clients"
            ) from exc
        if not address.is_loopback:
            raise GatewaySecurityError("dashboard API only accepts loopback clients")

    def assert_local_origin(self, origin: str | None) -> None:
        if not origin:
            return
        parsed = urlparse(origin)
        if parsed.scheme == "tauri":
            return
        hostname = str(parsed.hostname or "").strip().lower()
        if not hostname:
            raise GatewaySecurityError("dashboard origin must be local")
        if hostname in _LOCAL_HOSTS:
            return
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise GatewaySecurityError("dashboard origin must be local") from exc
        if not address.is_loopback:
            raise GatewaySecurityError("dashboard origin must be local")

    def _prune_expired_sessions(self) -> None:
        now = datetime.now(tz=UTC)
        expired = [
            token
            for token, session in self._sessions.items()
            if datetime.fromisoformat(session.expires_at) <= now
        ]
        for token in expired:
            self._sessions.pop(token, None)
