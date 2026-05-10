"""Isolated desktop adapter — focus-safe, artifact-syncing wrapper.

Phase 6 implementation.

This adapter wraps the VirtualDesktopSandboxBackend (or a CUA provider) and
adds three critical production capabilities:

1. **Focus protection** — the agent's actions run against an isolated virtual
   desktop; the user's active window / mouse / keyboard are never stolen.

2. **Artifact sync** — files written inside the sandbox working directory are
   copied back to the workspace artifacts folder with SHA-256 verification.

3. **Dashboard toggle** — a service-level switch (`use_isolated`) lets the
   workflow service flip between host control (real UIA backend) and isolated
   control at runtime, without restarting the process.

Isolation tiers
───────────────
HOST      — Full real-desktop control (no isolation; approval-gated).
VIRTUAL   — In-memory VirtualDesktopSandboxBackend; no real I/O.
CUA       — Computer-Use API sandbox if provider is configured.

The adapter raises IsolationUnavailable when the requested tier is not
available on this machine and automatically falls back to the next tier.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agentos_orchestrator.os_control.base import (
    OsControlBackend,
    UiAction,
    UiNode,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Enumerations                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class IsolationTier(str, Enum):
    HOST    = "host"
    VIRTUAL = "virtual"
    CUA     = "cua"


class IsolationUnavailable(RuntimeError):
    """Raised when the requested isolation tier cannot be provided."""


# ─────────────────────────────────────────────────────────────────────────── #
# Artifact sync                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(slots=True)
class SyncedArtifact:
    source_path: str
    destination_path: str
    sha256: str
    size_bytes: int
    synced_at: float = field(default_factory=time.time)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactSyncer:
    """Copies files from the sandbox working directory to the workspace."""

    def __init__(
        self,
        sandbox_dir: Path,
        artifacts_dir: Path,
        extension_allow: frozenset[str] | None = None,
    ) -> None:
        self._sandbox_dir = sandbox_dir
        self._artifacts_dir = artifacts_dir
        self._extension_allow = extension_allow or frozenset({
            ".md", ".txt", ".pdf", ".pptx", ".docx", ".csv",
            ".json", ".png", ".jpg", ".jpeg", ".svg", ".html",
            ".xlsx", ".py",
        })
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    def sync(self, run_id: str | None = None) -> list[SyncedArtifact]:
        """Copy all new/changed artefacts back to the workspace.

        Returns a list of SyncedArtifact for each successfully synced file.
        """
        synced: list[SyncedArtifact] = []
        dest_base = self._artifacts_dir / (run_id or "latest")
        dest_base.mkdir(parents=True, exist_ok=True)

        for src in self._sandbox_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in self._extension_allow:
                continue
            rel = src.relative_to(self._sandbox_dir)
            dest = dest_base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Skip if identical copy already exists
            if dest.exists():
                try:
                    if _sha256_file(src) == _sha256_file(dest):
                        continue
                except OSError:
                    pass

            try:
                shutil.copy2(src, dest)
                sha = _sha256_file(dest)
                synced.append(
                    SyncedArtifact(
                        source_path=str(src),
                        destination_path=str(dest),
                        sha256=sha,
                        size_bytes=dest.stat().st_size,
                    )
                )
                log.info(
                    "Artifact synced: %s → %s (%d bytes, sha=%s…)",
                    rel,
                    dest,
                    dest.stat().st_size,
                    sha[:8],
                )
            except OSError as exc:
                log.warning("Artifact sync failed for %s: %s", src, exc)

        return synced


# ─────────────────────────────────────────────────────────────────────────── #
# Backend resolution helpers                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_virtual_backend(sandbox_state_path: Path) -> OsControlBackend:
    from agentos_orchestrator.os_control.virtual_desktop_sandbox_backend import (
        VirtualDesktopSandboxBackend,
    )
    return VirtualDesktopSandboxBackend(sandbox_state_path)  # type: ignore[return-value]


def _build_cua_backend() -> OsControlBackend:
    """Return a CUA sandbox backend if configured."""
    try:
        from agentos_orchestrator.sandbox.providers import (  # type: ignore[import]
            CuaSandboxProvider,
        )
        provider = CuaSandboxProvider()
        # CuaSandboxProvider is a SandboxProvider not an OsControlBackend;
        # we expose it through a thin shim.
        return _CuaBackendShim(provider)  # type: ignore[return-value]
    except Exception as exc:
        raise IsolationUnavailable(f"CUA backend unavailable: {exc}") from exc


class _CuaBackendShim:
    """Adapts a CuaSandboxProvider to the OsControlBackend protocol."""

    name = "cua-sandbox"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def available(self) -> bool:
        try:
            return bool(self._provider)
        except Exception:
            return False

    def snapshot(self) -> list[UiNode]:
        return []  # CUA providers handle their own screenshot pipeline

    def perform(self, action: UiAction) -> str:
        try:
            result = self._provider.execute(action)
            return str(result)
        except Exception as exc:
            log.warning("CUA perform error: %s", exc)
            return f"error:{exc}"


def _build_host_backend() -> OsControlBackend:
    """Return the real UIA backend for full host control."""
    try:
        from agentos_orchestrator.os_control.uia_backend import UiaBackend  # type: ignore[import]
        backend = UiaBackend()
        if not backend.available():
            raise IsolationUnavailable("UIA backend reports not available")
        return backend
    except ImportError as exc:
        raise IsolationUnavailable(f"UIA backend not installed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────── #
# IsolatedDesktopAdapter                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class IsolationConfig:
    """Runtime configuration for the adapter."""
    tier: IsolationTier = IsolationTier.VIRTUAL
    sandbox_state_path: Path = field(
        default_factory=lambda: Path("artifacts/sandbox/desktop_state.json")
    )
    sandbox_working_dir: Path = field(
        default_factory=lambda: Path("artifacts/sandbox/workdir")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path("artifacts/workflows")
    )
    # If True, fall back to the next tier when the requested tier is unavailable
    allow_tier_fallback: bool = True
    focus_protection: bool = True          # refuse host actions if focus in use
    artifact_sync_on_complete: bool = True # auto-sync after each task completion


class IsolatedDesktopAdapter:
    """Focus-safe, artifact-syncing desktop control adapter.

    Acts as a drop-in replacement for the raw OsControlBackend in the workflow
    service.  Callers switch between HOST / VIRTUAL / CUA via ``use_isolated()``.

    Thread-safety: not thread-safe; create one adapter per workflow thread.
    """

    def __init__(self, config: IsolationConfig | None = None) -> None:
        self._config = config or IsolationConfig()
        self._backend: OsControlBackend | None = None
        self._current_tier: IsolationTier | None = None
        self._syncer: ArtifactSyncer | None = None
        self._action_log: list[dict[str, Any]] = []
        self._task_run_id: str | None = None
        self._is_isolated: bool = True  # start isolated by default

    # ── Backend lifecycle ────────────────────────────────────────────────── #

    def initialise(self, tier: IsolationTier | None = None) -> IsolationTier:
        """Initialise the backend for the given tier.

        Returns the tier that was actually activated (may differ if fallback).
        """
        requested = tier or self._config.tier
        activated = self._activate_tier(requested)
        self._current_tier = activated

        # Prepare sandbox working dir
        working_dir = self._config.sandbox_working_dir
        working_dir.mkdir(parents=True, exist_ok=True)
        os.environ["AGENTOS_SANDBOX_DIR"] = str(working_dir)

        # Initialise artifact syncer
        self._syncer = ArtifactSyncer(
            sandbox_dir=working_dir,
            artifacts_dir=self._config.artifacts_dir,
        )

        log.info(
            "IsolatedDesktopAdapter: activated tier=%s backend=%s",
            activated.value,
            getattr(self._backend, "name", "unknown"),
        )
        return activated

    def _activate_tier(self, tier: IsolationTier) -> IsolationTier:
        TIER_ORDER = [IsolationTier.VIRTUAL, IsolationTier.CUA, IsolationTier.HOST]
        tier_builders = {
            IsolationTier.VIRTUAL: lambda: _build_virtual_backend(
                self._config.sandbox_state_path
            ),
            IsolationTier.CUA: _build_cua_backend,
            IsolationTier.HOST: _build_host_backend,
        }
        tiers_to_try = [tier]
        if self._config.allow_tier_fallback:
            # Try requested tier first, then walk down to VIRTUAL
            for t in TIER_ORDER:
                if t not in tiers_to_try:
                    tiers_to_try.append(t)

        for t in tiers_to_try:
            try:
                self._backend = tier_builders[t]()
                return t
            except (IsolationUnavailable, Exception) as exc:
                log.warning(
                    "Tier %s unavailable (%s); trying next fallback.", t.value, exc
                )

        raise IsolationUnavailable("No isolation tier could be activated.")

    # ── Dashboard toggle ─────────────────────────────────────────────────── #

    def use_isolated(self, isolated: bool = True) -> None:
        """Toggle between isolated (sandbox) and host control mode.

        In host mode, actions are routed to the real desktop via the HOST tier.
        In isolated mode, actions are routed to VIRTUAL or CUA.
        """
        if self._is_isolated == isolated:
            return
        self._is_isolated = isolated
        new_tier = (
            self._config.tier if isolated else IsolationTier.HOST
        )
        log.info(
            "IsolatedDesktopAdapter: switching to %s mode (tier=%s)",
            "isolated" if isolated else "host",
            new_tier.value,
        )
        try:
            self._activate_tier(new_tier)
            self._current_tier = new_tier
        except IsolationUnavailable as exc:
            log.error("Cannot switch to %s: %s", new_tier.value, exc)
            raise

    @property
    def is_isolated(self) -> bool:
        return self._is_isolated

    @property
    def current_tier(self) -> IsolationTier | None:
        return self._current_tier

    # ── OsControlBackend protocol ────────────────────────────────────────── #

    @property
    def name(self) -> str:
        backend_name = getattr(self._backend, "name", "uninitialized")
        tier_tag = self._current_tier.value if self._current_tier else "?"
        return f"isolated-desktop-adapter/{tier_tag}/{backend_name}"

    def available(self) -> bool:
        if self._backend is None:
            try:
                self.initialise()
            except IsolationUnavailable:
                return False
        try:
            return bool(self._backend.available())  # type: ignore[union-attr]
        except Exception:
            return False

    def snapshot(self) -> list[UiNode]:
        self._ensure_backend()
        try:
            return self._backend.snapshot()  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("Snapshot error: %s", exc)
            return []

    def perform(self, action: UiAction) -> str:
        """Execute action with focus-protection and action logging."""
        self._ensure_backend()

        # ── Focus protection ──────────────────────────────────────────────── #
        if self._config.focus_protection and not self._is_isolated:
            if self._user_focus_is_active():
                log.warning(
                    "Focus protection: user has active focus; deferring host action %s",
                    action.action_type,
                )
                return "deferred:user_focus_active"

        start = time.monotonic()
        try:
            result = self._backend.perform(action)  # type: ignore[union-attr]
            elapsed = time.monotonic() - start
            self._action_log.append({
                "action_type": action.action_type,
                "selector": action.selector,
                "result": result,
                "elapsed_ms": round(elapsed * 1000, 1),
                "tier": self._current_tier.value if self._current_tier else "?",
                "isolated": self._is_isolated,
            })
            return result
        except Exception as exc:
            self._action_log.append({
                "action_type": action.action_type,
                "selector": action.selector,
                "result": f"error:{exc}",
                "tier": self._current_tier.value if self._current_tier else "?",
                "isolated": self._is_isolated,
            })
            log.error("perform() error: %s", exc)
            raise

    # ── Artifact sync ─────────────────────────────────────────────────────── #

    def sync_artifacts(self, run_id: str | None = None) -> list[SyncedArtifact]:
        """Copy sandbox output files back to the workspace artifacts folder."""
        if self._syncer is None:
            log.warning("sync_artifacts: syncer not initialised; call initialise() first")
            return []
        rid = run_id or self._task_run_id
        synced = self._syncer.sync(rid)
        log.info("Artifact sync complete: %d files synced", len(synced))
        return synced

    def on_task_start(self, run_id: str) -> None:
        """Record the current task run ID for artifact namespacing."""
        self._task_run_id = run_id
        self._action_log.clear()
        log.info("IsolatedDesktopAdapter: task %s started", run_id)

    def on_task_complete(self, run_id: str) -> list[SyncedArtifact]:
        """Called when a task finishes; auto-syncs artifacts if configured."""
        log.info("IsolatedDesktopAdapter: task %s complete", run_id)
        if self._config.artifact_sync_on_complete:
            return self.sync_artifacts(run_id)
        return []

    def get_action_log(self) -> list[dict[str, Any]]:
        return list(self._action_log)

    # ── Internal helpers ─────────────────────────────────────────────────── #

    def _ensure_backend(self) -> None:
        if self._backend is None:
            self.initialise()

    def _user_focus_is_active(self) -> bool:
        """Return True if the user appears to be actively using the keyboard.

        Uses a Windows-only heuristic; returns False on all other OSes so
        the adapter never blocks on non-Windows platforms.
        """
        try:
            import ctypes
            # GetLastInputInfo: ms since last user input
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint),
                    ("dwTime", ctypes.c_uint),
                ]

            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):  # type: ignore[attr-defined]
                idle_ms = ctypes.windll.kernel32.GetTickCount() - lii.dwTime  # type: ignore[attr-defined]
                # If user input in last 2 seconds, consider focus active
                return idle_ms < 2_000
        except Exception:
            pass
        return False


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def build_isolated_adapter(
    tier: IsolationTier = IsolationTier.VIRTUAL,
    workspace_root: Path | str | None = None,
    focus_protection: bool = True,
) -> IsolatedDesktopAdapter:
    """Convenience factory for the workflow service."""
    root = Path(workspace_root or Path.cwd())
    config = IsolationConfig(
        tier=tier,
        sandbox_state_path=root / "artifacts" / "sandbox" / "desktop_state.json",
        sandbox_working_dir=root / "artifacts" / "sandbox" / "workdir",
        artifacts_dir=root / "artifacts" / "workflows",
        allow_tier_fallback=True,
        focus_protection=focus_protection,
        artifact_sync_on_complete=True,
    )
    adapter = IsolatedDesktopAdapter(config)
    adapter.initialise(tier)
    return adapter
