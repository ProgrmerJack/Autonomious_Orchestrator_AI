from __future__ import annotations

import unittest
from unittest import mock

from agentos_orchestrator import cli


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


if __name__ == "__main__":
    unittest.main()
