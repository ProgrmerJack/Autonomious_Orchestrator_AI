"""Tests for IsolatedDesktopAdapter (Phase 6)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentos_orchestrator.os_control.isolated_desktop_adapter import (
    ArtifactSyncer,
    IsolatedDesktopAdapter,
    IsolationConfig,
    IsolationTier,
    SyncedArtifact,
    build_isolated_adapter,
)
from agentos_orchestrator.os_control.base import UiAction


# ─────────────────────────────────────────────────────────────────────────── #
# ArtifactSyncer                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestArtifactSyncer:
    def test_sync_copies_allowed_extensions(self, tmp_path):
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        artifacts_dir = tmp_path / "artifacts"

        (sandbox_dir / "report.md").write_text("# Report\nContent.", encoding="utf-8")
        (sandbox_dir / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        (sandbox_dir / "temp.tmp").write_text("ignored", encoding="utf-8")

        syncer = ArtifactSyncer(sandbox_dir, artifacts_dir)
        synced = syncer.sync("run-001")

        synced_names = {Path(s.source_path).name for s in synced}
        assert "report.md" in synced_names
        assert "chart.png" in synced_names
        assert "temp.tmp" not in synced_names

    def test_sync_returns_synced_artifact_with_sha256(self, tmp_path):
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        artifacts_dir = tmp_path / "artifacts"
        (sandbox_dir / "output.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        syncer = ArtifactSyncer(sandbox_dir, artifacts_dir)
        synced = syncer.sync()
        assert len(synced) == 1
        s = synced[0]
        assert isinstance(s, SyncedArtifact)
        assert len(s.sha256) == 64
        assert s.size_bytes > 0

    def test_sync_skips_identical_files(self, tmp_path):
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        artifacts_dir = tmp_path / "artifacts"
        content = "# Same content\n"
        (sandbox_dir / "notes.md").write_text(content, encoding="utf-8")

        syncer = ArtifactSyncer(sandbox_dir, artifacts_dir)
        synced1 = syncer.sync("run-A")
        synced2 = syncer.sync("run-A")  # same run_id → identical dest path
        # Second sync: file is identical, should be skipped
        assert len(synced2) == 0

    def test_sync_copies_nested_files(self, tmp_path):
        sandbox_dir = tmp_path / "sandbox"
        subdir = sandbox_dir / "subdir"
        subdir.mkdir(parents=True)
        artifacts_dir = tmp_path / "artifacts"
        (subdir / "nested.pdf").write_bytes(b"%PDF-1.4")

        syncer = ArtifactSyncer(sandbox_dir, artifacts_dir)
        synced = syncer.sync()
        assert any("nested.pdf" in s.destination_path for s in synced)


# ─────────────────────────────────────────────────────────────────────────── #
# IsolatedDesktopAdapter — VIRTUAL tier (no real desktop required)             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestIsolatedDesktopAdapterVirtual:
    @pytest.fixture
    def adapter(self, tmp_path) -> IsolatedDesktopAdapter:
        config = IsolationConfig(
            tier=IsolationTier.VIRTUAL,
            sandbox_state_path=tmp_path / "state.json",
            sandbox_working_dir=tmp_path / "workdir",
            artifacts_dir=tmp_path / "artifacts",
            allow_tier_fallback=False,
            focus_protection=False,
        )
        a = IsolatedDesktopAdapter(config)
        return a

    def test_available_returns_bool(self, adapter):
        result = adapter.available()
        assert isinstance(result, bool)

    def test_name_contains_tier(self, adapter):
        adapter.available()  # triggers initialise
        assert "virtual" in adapter.name.lower()

    def test_is_isolated_true_by_default(self, adapter):
        assert adapter.is_isolated is True

    def test_use_isolated_false_changes_flag(self, adapter, tmp_path):
        adapter.available()  # initialise backend
        with patch(
            "agentos_orchestrator.os_control.isolated_desktop_adapter._build_host_backend"
        ) as mock_host:
            mock_host.return_value = MagicMock(available=lambda: True, name="mock-uia")
            adapter.use_isolated(False)
        assert adapter.is_isolated is False

    def test_on_task_start_clears_action_log(self, adapter):
        adapter.on_task_start("run-xyz")
        assert adapter.get_action_log() == []

    def test_sync_artifacts_returns_list(self, adapter, tmp_path):
        adapter.available()  # ensure initialised
        synced = adapter.sync_artifacts("run-001")
        assert isinstance(synced, list)

    def test_on_task_complete_returns_synced_artifacts(self, adapter, tmp_path):
        adapter.available()
        # Write a file into the workdir so there's something to sync
        workdir = tmp_path / "workdir"
        workdir.mkdir(exist_ok=True)
        (workdir / "result.md").write_text("# Done", encoding="utf-8")
        synced = adapter.on_task_complete("run-001")
        assert isinstance(synced, list)


# ─────────────────────────────────────────────────────────────────────────── #
# Focus protection                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFocusProtection:
    def test_focus_protection_defers_host_action_when_user_active(self, tmp_path):
        config = IsolationConfig(
            tier=IsolationTier.VIRTUAL,
            sandbox_state_path=tmp_path / "state.json",
            sandbox_working_dir=tmp_path / "workdir",
            artifacts_dir=tmp_path / "artifacts",
            focus_protection=True,
        )
        adapter = IsolatedDesktopAdapter(config)
        adapter._is_isolated = False  # simulate host mode

        mock_backend = MagicMock()
        mock_backend.perform.return_value = "ok"
        adapter._backend = mock_backend

        with patch.object(adapter, "_user_focus_is_active", return_value=True):
            result = adapter.perform(UiAction(action_type="click", selector="btn"))

        assert result == "deferred:user_focus_active"
        mock_backend.perform.assert_not_called()

    def test_focus_protection_allows_action_when_user_idle(self, tmp_path):
        config = IsolationConfig(
            tier=IsolationTier.VIRTUAL,
            sandbox_state_path=tmp_path / "state.json",
            sandbox_working_dir=tmp_path / "workdir",
            artifacts_dir=tmp_path / "artifacts",
            focus_protection=True,
        )
        adapter = IsolatedDesktopAdapter(config)
        adapter._is_isolated = False

        mock_backend = MagicMock()
        mock_backend.perform.return_value = "action-id-001"
        adapter._backend = mock_backend

        with patch.object(adapter, "_user_focus_is_active", return_value=False):
            result = adapter.perform(UiAction(action_type="click", selector="btn"))

        assert result == "action-id-001"


# ─────────────────────────────────────────────────────────────────────────── #
# build_isolated_adapter factory                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def test_build_isolated_adapter_returns_adapter(tmp_path):
    adapter = build_isolated_adapter(
        tier=IsolationTier.VIRTUAL,
        workspace_root=tmp_path,
        focus_protection=False,
    )
    assert isinstance(adapter, IsolatedDesktopAdapter)
    assert adapter.current_tier == IsolationTier.VIRTUAL
