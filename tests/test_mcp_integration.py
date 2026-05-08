from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from agentos_orchestrator.core.agents import WorkerAgent
from agentos_orchestrator.core.checkpoint import CheckpointStore
from agentos_orchestrator.core.events import DurableEventLog, EventBus
from agentos_orchestrator.mcp import run_mcp_research_query
from agentos_orchestrator.research import (
    DeepResearchEngine,
    ResearchBrief,
    ResearchSource,
)


_INTEGRATION_SERVER = textwrap.dedent(
    """
    import json
    import sys

    initialized = False

    def send(payload):
        print(json.dumps(payload), flush=True)

    for raw in iter(input, ""):
        message = json.loads(raw)
        method = message.get("method")
        if method == "initialize":
            print("integration server ready", file=sys.stderr, flush=True)
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"protocolVersion": message["params"]["protocolVersion"]},
            })
        elif method == "notifications/initialized":
            initialized = True
        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"tools": [{"name": "research.search"}], "initialized": initialized},
            })
        elif method == "tools/call":
            query = message.get("params", {}).get("arguments", {}).get("query", "")
            send({
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "results": [
                        {
                            "title": f"MCP source for {query}",
                            "url": "https://example.com/mcp-source",
                            "abstract": "Tool-backed MCP evidence",
                        }
                    ]
                },
            })
    """
)


class _FakeResearchEngine(DeepResearchEngine):
    def run(
        self,
        objective: str,
        run_id: str,
        pc_context: dict[str, object] | None = None,
        planning_context: dict[str, object] | None = None,
        evidence_targets: dict[str, object] | None = None,
    ) -> ResearchBrief:
        del run_id, pc_context, planning_context, evidence_targets
        return ResearchBrief(
            objective=objective,
            query="test query",
            summary="base summary",
            sources=[
                ResearchSource(
                    provider="base",
                    title="Base Source",
                    url="https://example.com/base",
                    abstract="Base evidence",
                    score=50.0,
                )
            ],
            artifacts=[],
            confidence=0.8,
        )


class McpIntegrationTests(unittest.TestCase):
    def test_run_mcp_research_query_reads_env_configured_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "mcp_integration_server.py"
            script_path.write_text(_INTEGRATION_SERVER, encoding="utf-8")
            payload = json.dumps(
                [
                    {
                        "name": "integration",
                        "command": [sys.executable, "-u", str(script_path)],
                        "research_tool": "research.search",
                    }
                ]
            )
            with patch.dict("os.environ", {"AGENTOS_MCP_SERVERS_JSON": payload}):
                hits, diagnostics = run_mcp_research_query("orchestrator", limit=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].provider, "mcp:integration")
        self.assertIn("orchestrator", hits[0].title)
        self.assertEqual(diagnostics[0]["status"], "ok")
        self.assertIn("integration server ready", diagnostics[0]["stderr"])

    def test_worker_research_context_merges_mcp_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "mcp_integration_server.py"
            script_path.write_text(_INTEGRATION_SERVER, encoding="utf-8")
            payload = json.dumps(
                [
                    {
                        "name": "integration",
                        "command": [sys.executable, "-u", str(script_path)],
                        "research_tool": "research.search",
                    }
                ]
            )
            event_log = DurableEventLog(Path(temp_dir) / "events.sqlite3")
            worker = WorkerAgent(
                EventBus(event_log),
                CheckpointStore(Path(temp_dir) / "state.sqlite3"),
                research_engine=_FakeResearchEngine(),
            )
            with patch.dict("os.environ", {"AGENTOS_MCP_SERVERS_JSON": payload}):
                brief = worker._run_research_with_context(
                    "test objective",
                    "run_mcp",
                    [],
                    effort="standard",
                )

        self.assertEqual(len(brief.sources), 2)
        self.assertTrue(
            any(source.provider == "mcp:integration" for source in brief.sources)
        )
        self.assertEqual(brief.metadata["mcp"]["source_count"], 1)
