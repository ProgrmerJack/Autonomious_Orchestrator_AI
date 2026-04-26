from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.core.checkpoint import CheckpointStore
from agentos_orchestrator.core.durable import DurableExecutionStore
from agentos_orchestrator.core.events import DurableEventLog, EventBus


class DurabilityTests(unittest.TestCase):
    def test_events_and_checkpoints_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite3"
            log = DurableEventLog(db_path)
            bus = EventBus(log)
            event = bus.publish(
                "run_1",
                "task.completed",
                "worker",
                {"ok": True},
            )
            self.assertIsNotNone(event.sequence)
            events = DurableEventLog(db_path).list_events(run_id="run_1")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].payload["ok"], True)

            store = CheckpointStore(db_path)
            store.save("run_1", "stage", {"cursor": 3})
            loaded = CheckpointStore(db_path).load("run_1")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.state["cursor"], 3)

    def test_durable_execution_store_lists_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite3"
            store = DurableExecutionStore(db_path)
            store.save_manifest(
                "run_1",
                "objective",
                [{"task_id": "task_1"}],
            )
            store.complete_run("run_1")

            runs = store.list_runs(status="completed")

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].run_id, "run_1")
            self.assertEqual(runs[0].tasks[0]["task_id"], "task_1")


if __name__ == "__main__":
    unittest.main()
