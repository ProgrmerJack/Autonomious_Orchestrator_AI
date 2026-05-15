from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from agentos_orchestrator import cli
from agentos_orchestrator.research import ResearchBrief, ResearchSource


class DashboardEventLoopPolicyTests(unittest.TestCase):
    def test_windows_dashboard_defaults_to_proactor_policy(self) -> None:
        selector_policy = type("SelectorPolicy", (), {})
        proactor_policy = type("ProactorPolicy", (), {})
        configure_policy = getattr(
            cli,
            "_configure_dashboard_event_loop_policy",
        )

        with (
            mock.patch.object(cli.os, "name", "nt"),
            mock.patch.object(cli.os, "getenv", return_value=""),
            mock.patch.object(
                cli.asyncio,
                "WindowsSelectorEventLoopPolicy",
                selector_policy,
                create=True,
            ),
            mock.patch.object(
                cli.asyncio,
                "WindowsProactorEventLoopPolicy",
                proactor_policy,
                create=True,
            ),
            mock.patch.object(
                cli.asyncio,
                "get_event_loop_policy",
                return_value=selector_policy(),
            ),
            mock.patch.object(
                cli.asyncio,
                "set_event_loop_policy",
            ) as set_policy,
        ):
            configure_policy()

        set_policy.assert_called_once()
        self.assertIsInstance(set_policy.call_args.args[0], proactor_policy)

    def test_windows_dashboard_selector_override(self) -> None:
        selector_policy = type("SelectorPolicy", (), {})
        proactor_policy = type("ProactorPolicy", (), {})
        configure_policy = getattr(
            cli,
            "_configure_dashboard_event_loop_policy",
        )

        with (
            mock.patch.object(cli.os, "name", "nt"),
            mock.patch.object(cli.os, "getenv", return_value="selector"),
            mock.patch.object(
                cli.asyncio,
                "WindowsSelectorEventLoopPolicy",
                selector_policy,
                create=True,
            ),
            mock.patch.object(
                cli.asyncio,
                "WindowsProactorEventLoopPolicy",
                proactor_policy,
                create=True,
            ),
            mock.patch.object(
                cli.asyncio,
                "get_event_loop_policy",
                return_value=proactor_policy(),
            ),
            mock.patch.object(
                cli.asyncio,
                "set_event_loop_policy",
            ) as set_policy,
        ):
            configure_policy()

        set_policy.assert_called_once()
        self.assertIsInstance(set_policy.call_args.args[0], selector_policy)


class ReplayMergeCliTests(unittest.TestCase):
    def test_replay_merge_dispatches_to_deep_research_engine(self) -> None:
        brief = ResearchBrief(
            objective="checkpoint objective",
            query="checkpoint objective",
            summary="replayed summary",
            sources=[
                ResearchSource(
                    provider="web-search",
                    title="Checkpoint Source",
                    url="https://example.org/checkpoint-source",
                )
            ],
            artifacts=["runs/run_merge/research/synthesis_packet.json"],
            confidence=0.75,
            metadata={"replay_mode": "detached-merge-only"},
        )

        with (
            mock.patch.object(cli, "DeepResearchEngine") as engine_cls,
            mock.patch("builtins.print") as print_mock,
        ):
            engine_cls.return_value.replay_detached_merge.return_value = brief

            exit_code = cli.main(
                [
                    "replay-merge",
                    "--run-id",
                    "run_merge",
                    "--workspace-root",
                    "workspace",
                ]
            )

        self.assertEqual(exit_code, 0)
        engine_cls.assert_called_once()
        engine_cls.return_value.replay_detached_merge.assert_called_once_with(
            "run_merge"
        )
        print_mock.assert_called_once()


class LiveFireCliTests(unittest.TestCase):
    def test_pc_live_fire_eval_forwards_pack_selection(self) -> None:
        fake_orchestrator = mock.Mock()
        fake_orchestrator.authorization.authorize.return_value = SimpleNamespace(
            allowed=True
        )
        fake_result = mock.Mock()
        fake_result.success = True
        fake_result.asdict.return_value = {"success": True, "task_count": 1}

        with (
            mock.patch.object(
                cli.ResearchOrchestrator,
                "from_paths",
                return_value=fake_orchestrator,
            ),
            mock.patch.object(cli, "_pc_backend", return_value=object()),
            mock.patch.object(cli, "LiveFireEvalRunner") as runner_cls,
            mock.patch("builtins.print"),
        ):
            runner_cls.return_value.run.return_value = fake_result

            exit_code = cli.main(
                [
                    "pc-live-fire-eval",
                    "--backend",
                    "virtual-desktop-sandbox",
                    "--pack",
                    "everyday",
                    "--max-tasks",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = runner_cls.return_value.run.call_args.args[0]
        self.assertEqual(config.pack, "everyday")
        self.assertEqual(config.max_tasks, 2)


if __name__ == "__main__":
    unittest.main()
