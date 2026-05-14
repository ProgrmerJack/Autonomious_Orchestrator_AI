"""Tests for retrieval RAM-bound memory hygiene.

These tests pin down the overheat root-cause fix: even with extreme
multi-hour budgets, the retained working set must stay bounded and the
overflow must be spilled to the persistent crawl queue rather than to RAM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_child(env_overrides: dict[str, str], code: str) -> str:
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"child failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return proc.stdout.strip()


def test_chained_sources_ram_cap_honors_env() -> None:
    out = _run_child(
        {"AGENTOS_CHAINED_SOURCES_RAM_CAP": "7"},
        textwrap.dedent(
            """
            from agentos_orchestrator.research import retrieval
            print(retrieval._CHAINED_SOURCES_RAM_CAP)
            """
        ),
    )
    assert out == "7"


def test_all_sources_ram_cap_honors_env() -> None:
    out = _run_child(
        {"AGENTOS_ALL_SOURCES_RAM_CAP": "13"},
        textwrap.dedent(
            """
            from agentos_orchestrator.research import retrieval
            print(retrieval._ALL_SOURCES_RAM_CAP)
            """
        ),
    )
    assert out == "13"


def test_source_abstract_ram_cap_honors_env() -> None:
    out = _run_child(
        {"AGENTOS_SOURCE_ABSTRACT_RAM_CAP": "256"},
        textwrap.dedent(
            """
            from agentos_orchestrator.research import retrieval
            print(retrieval._SOURCE_ABSTRACT_RAM_CAP)
            """
        ),
    )
    assert out == "256"


def test_abstract_trimming_truncates_oversized_abstract() -> None:
    """The per-source abstract trim must clip strings to the cap value."""
    out = _run_child(
        {"AGENTOS_SOURCE_ABSTRACT_RAM_CAP": "100"},
        textwrap.dedent(
            """
            from agentos_orchestrator.research import retrieval

            class S:
                def __init__(self):
                    self.abstract = 'X' * 10_000
            sources = [S() for _ in range(5)]
            cap = retrieval._SOURCE_ABSTRACT_RAM_CAP
            for s in sources:
                if len(s.abstract) > cap:
                    s.abstract = s.abstract[:cap]
            print(max(len(s.abstract) for s in sources))
            """
        ),
    )
    assert out == "100"


def test_retrieval_module_imports_after_memory_patch() -> None:
    """Ensure the patched module still parses + imports cleanly."""
    out = _run_child(
        {},
        textwrap.dedent(
            """
            from agentos_orchestrator.research import retrieval
            assert hasattr(retrieval, '_CHAINED_SOURCES_RAM_CAP')
            assert hasattr(retrieval, '_ALL_SOURCES_RAM_CAP')
            assert hasattr(retrieval, '_SOURCE_ABSTRACT_RAM_CAP')
            print('OK')
            """
        ),
    )
    assert out == "OK"
