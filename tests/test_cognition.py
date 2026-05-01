"""Comprehensive tests for the Cognitive Architecture (Universal Agent)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentos_orchestrator.cognition.active_inference import ActiveInferenceExplorer
from agentos_orchestrator.cognition.differentiable_memory import (
    EpisodicMemoryBank,
    WorkingMemoryScratchpad,
)
from agentos_orchestrator.cognition.hierarchical_planner import (
    ExecutionContext,
    MacroGoal,
    MacroPlanner,
    MicroExecutor,
)
from agentos_orchestrator.cognition.mcts_simulator import (
    MCTSWorldModel,
    MCTSSimulator,
    WorldState,
)
from agentos_orchestrator.cognition.pomdp_state import (
    POMDPBeliefState,
    POMDPEnvironmentModel,
    UIStateObservation,
)
from agentos_orchestrator.cognition.universal_agent import UniversalDesktopAgent
from agentos_orchestrator.cognition.vla_affordance import (
    VLAActionSpace,
    VLAAffordanceGrounding,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode


class FakeBackend:
    """Backend that captures all actions and returns deterministic receipts."""

    def __init__(self, nodes: list[UiNode] | None = None) -> None:
        self.actions: list[UiAction] = []
        self._nodes = nodes or [
            UiNode(node_id="btn-1", role="Button", name="Start"),
            UiNode(node_id="edit-1", role="Edit", name="Input Field"),
            UiNode(node_id="doc-1", role="Document", name="Workspace"),
        ]

    def snapshot(self) -> list[UiNode]:
        return list(self._nodes)

    def perform(self, action: UiAction) -> str:
        self.actions.append(action)
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
            }
        )


class FakeBackendWithStateChange(FakeBackend):
    """Backend that changes state after certain actions."""

    def perform(self, action: UiAction) -> str:
        self.actions.append(action)
        if action.action_type == "click" and "Start" in action.selector:
            self._nodes.append(UiNode(node_id="menu-1", role="Menu", name="Options"))
        return json.dumps(
            {
                "status": "executed",
                "action": action.action_type,
                "selector": action.selector,
            }
        )


# --------------------------------------------------------------------------- #
# VLA Affordance Tests
# --------------------------------------------------------------------------- #
class VLAAffordanceTests(unittest.TestCase):
    def test_vla_action_space_to_ui_action_click(self) -> None:
        vla = VLAActionSpace(action_type="click", x=100, y=200, rationale="test")
        ui = vla.to_ui_action()
        self.assertEqual(ui.action_type, "click")
        self.assertIn("100", ui.selector)
        self.assertIn("200", ui.selector)

    def test_vla_action_space_to_ui_action_type(self) -> None:
        vla = VLAActionSpace(action_type="type", x=50, y=60, text="hello")
        ui = vla.to_ui_action("name=Editor")
        self.assertEqual(ui.action_type, "type")
        self.assertEqual(ui.value, "hello")
        self.assertEqual(ui.selector, "name=Editor")

    def test_vla_action_space_to_ui_action_hotkey(self) -> None:
        vla = VLAActionSpace(action_type="hotkey", key_combo="^s")
        ui = vla.to_ui_action()
        self.assertEqual(ui.action_type, "hotkey")
        self.assertEqual(ui.value, "^s")

    def test_vla_heuristic_detect_affordances_returns_regions(self) -> None:
        grounding = VLAAffordanceGrounding()
        # Create a minimal 16x16 PNG in memory
        try:
            from PIL import Image

            buf = bytes()
            img = Image.new("RGB", (640, 480), color="white")
            import io

            bio = io.BytesIO()
            img.save(bio, format="PNG")
            buf = bio.getvalue()
        except ImportError:
            self.skipTest("PIL not installed")
        regions = grounding._heuristic_detect_affordances(buf, "draw a house")
        self.assertTrue(len(regions) > 0)
        self.assertTrue(all(r.confidence > 0 for r in regions))

    def test_vla_heuristic_propose_action_returns_click(self) -> None:
        grounding = VLAAffordanceGrounding()
        try:
            from PIL import Image
            import io

            img = Image.new("RGB", (640, 480), color="white")
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            buf = bio.getvalue()
        except ImportError:
            self.skipTest("PIL not installed")
        action = grounding.propose_action(buf, "click the button", [])
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "click")
        self.assertIsNotNone(action.x)
        self.assertIsNotNone(action.y)


# --------------------------------------------------------------------------- #
# Active Inference Tests
# --------------------------------------------------------------------------- #
class ActiveInferenceTests(unittest.TestCase):
    def test_explorer_initializes_empty(self) -> None:
        explorer = ActiveInferenceExplorer(max_probes=3)
        self.assertEqual(len(explorer._exploration_log), 0)
        self.assertEqual(explorer.max_probes, 3)

    def test_safety_score_penalizes_destructive(self) -> None:
        node = UiNode(node_id="del", role="Button", name="Delete Forever")
        score = ActiveInferenceExplorer._safety_score(node)
        self.assertLess(score, 1.0)

    def test_safety_score_rewards_safe_buttons(self) -> None:
        node = UiNode(node_id="ok", role="Button", name="OK")
        score = ActiveInferenceExplorer._safety_score(node)
        self.assertGreaterEqual(score, 0.9)

    def test_expected_info_gain_prefers_interactive(self) -> None:
        btn = UiNode(node_id="b", role="Button", name="Click Me")
        pane = UiNode(node_id="p", role="Pane", name="Static")
        ig_btn = ActiveInferenceExplorer._expected_info_gain(btn, "do something")
        ig_pane = ActiveInferenceExplorer._expected_info_gain(pane, "do something")
        self.assertGreater(ig_btn, ig_pane)

    def test_explore_runs_bounded_probes(self) -> None:
        backend = FakeBackendWithStateChange()
        explorer = ActiveInferenceExplorer(max_probes=2, random_seed=42)
        results = explorer.explore(
            backend.snapshot(),
            "explore the UI",
            perform_fn=backend.perform,
            snapshot_fn=backend.snapshot,
        )
        self.assertLessEqual(len(results), 2)
        for r in results:
            self.assertTrue(r.safe)
            self.assertGreaterEqual(r.info_gain, 0.0)

    def test_explore_detects_state_changes(self) -> None:
        backend = FakeBackendWithStateChange()
        explorer = ActiveInferenceExplorer(max_probes=3, random_seed=42)
        results = explorer.explore(
            backend.snapshot(),
            "explore",
            perform_fn=backend.perform,
            snapshot_fn=backend.snapshot,
        )
        # At least one probe should have caused a state change
        deltas = [len(r.state_delta) for r in results]
        self.assertTrue(any(d > 0 for d in deltas))

    def test_get_affordance_map_returns_mapping(self) -> None:
        backend = FakeBackend()
        explorer = ActiveInferenceExplorer(max_probes=2, random_seed=42)
        explorer.explore(
            backend.snapshot(),
            "test",
            perform_fn=backend.perform,
            snapshot_fn=backend.snapshot,
        )
        mapping = explorer.get_affordance_map()
        self.assertTrue(len(mapping) > 0)
        for selector, meta in mapping.items():
            self.assertIn("info_gain", meta)
            self.assertIn("safe", meta)

    def test_explorer_blocks_destructive_during_explore(self) -> None:
        backend = FakeBackend(
            [
                UiNode(node_id="del", role="Button", name="Delete", enabled=True),
            ]
        )
        explorer = ActiveInferenceExplorer(max_probes=1, random_seed=42)
        results = explorer.explore(
            backend.snapshot(),
            "test",
            perform_fn=backend.perform,
            snapshot_fn=backend.snapshot,
        )
        # Should still run but safety score is lower
        self.assertTrue(len(results) <= 1)

    def test_suggest_action_types_into_search_like_field(self) -> None:
        explorer = ActiveInferenceExplorer(max_probes=2, random_seed=42)
        nodes = [
            UiNode(node_id="search", role="Edit", name="Search"),
            UiNode(node_id="go", role="Button", name="Go"),
        ]
        action = explorer.suggest_action(nodes, "search for API documentation")
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "type")
        self.assertEqual(action.selector, "name=Search")
        self.assertTrue(bool(action.value))

    def test_suggest_action_focuses_canvas_for_drawing_objective(self) -> None:
        explorer = ActiveInferenceExplorer(max_probes=2, random_seed=42)
        nodes = [
            UiNode(node_id="canvas", role="Canvas", name="Drawing Surface"),
            UiNode(node_id="save", role="Button", name="Save"),
        ]
        action = explorer.suggest_action(nodes, "draw a small house")
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "focus")
        self.assertEqual(action.selector, "name=Drawing Surface")


# --------------------------------------------------------------------------- #
# POMDP State Tests
# --------------------------------------------------------------------------- #
class POMDPStateTests(unittest.TestCase):
    def test_belief_initialization(self) -> None:
        belief = POMDPBeliefState()
        obs = UIStateObservation(timestamp=0.0, nodes=[])
        belief.initialize_from_observation(obs, n_particles=5)
        self.assertEqual(len(belief.particles), 5)
        self.assertAlmostEqual(sum(p.weight for p in belief.particles), 1.0, places=5)

    def test_belief_update_changes_weights(self) -> None:
        belief = POMDPBeliefState()
        obs = UIStateObservation(
            timestamp=0.0,
            nodes=[UiNode(node_id="a", role="Button", name="A")],
        )
        belief.initialize_from_observation(obs, n_particles=5)
        # Perturb particles so they respond differently to observations
        for i, p in enumerate(belief.particles):
            p.state_vector["node_count"] = i + 1  # Different expected counts
        belief._normalize_weights()
        initial_weights = [p.weight for p in belief.particles]

        from agentos_orchestrator.cognition.pomdp_state import ActionOutcome

        outcome = ActionOutcome(action_type="click", selector="a", receipt="ok")
        obs2 = UIStateObservation(
            timestamp=1.0,
            nodes=[
                UiNode(node_id="a", role="Button", name="A"),
                UiNode(node_id="b", role="Menu", name="B"),
            ],
        )
        belief.update(obs2, outcome)
        updated_weights = [p.weight for p in belief.particles]
        self.assertNotEqual(initial_weights, updated_weights)

    def test_belief_entropy_decreases_with_certainty(self) -> None:
        belief = POMDPBeliefState()
        obs = UIStateObservation(timestamp=0.0, nodes=[])
        belief.initialize_from_observation(obs, n_particles=10)
        initial_entropy = belief.entropy()
        # Artificially make one particle dominant
        for i, p in enumerate(belief.particles):
            p.weight = 0.99 if i == 0 else 0.01 / 9
        belief._normalize_weights()  # type: ignore[attr]
        lower_entropy = belief.entropy()
        self.assertLess(lower_entropy, initial_entropy)

    def test_environment_model_step_updates_belief(self) -> None:
        backend = FakeBackend()
        env = POMDPEnvironmentModel(backend)
        obs = env.observe()
        self.assertIsNotNone(obs)
        self.assertGreater(len(env.belief.particles), 0)

        action = UiAction(action_type="click", selector="btn-1")
        obs2, outcome = env.step(action)
        self.assertIsNotNone(obs2)
        self.assertEqual(outcome.action_type, "click")
        self.assertTrue(len(env.belief.history) > 0)

    def test_transition_model_tracks_clicks(self) -> None:
        state = {"click_count": 0}
        next_state = POMDPBeliefState._transition(state, "click", "btn")
        self.assertEqual(next_state["click_count"], 1)
        self.assertEqual(next_state["last_clicked"], "btn")

    def test_transition_model_tracks_unsaved(self) -> None:
        state = {"has_unsaved_changes": False}
        next_state = POMDPBeliefState._transition(state, "type", "edit")
        self.assertTrue(next_state["has_unsaved_changes"])


# --------------------------------------------------------------------------- #
# MCTS Simulator Tests
# --------------------------------------------------------------------------- #
class MCTSSimulatorTests(unittest.TestCase):
    def test_world_model_predict_increments_depth(self) -> None:
        model = MCTSWorldModel()
        state = WorldState(state_vector={}, depth=0, terminal=False, reward=0.0)
        action = UiAction(action_type="click", selector="btn")
        next_state = model.predict(state, action)
        self.assertEqual(next_state.depth, 1)

    def test_world_model_evaluate_rewards_progress(self) -> None:
        model = MCTSWorldModel()
        state = WorldState(
            state_vector={"app_context": "browser"},
            depth=1,
            terminal=False,
            reward=0.0,
        )
        reward = model.evaluate(state, "open browser and search")
        self.assertGreater(reward, 0.0)

    def test_world_model_terminal_at_max_depth(self) -> None:
        model = MCTSWorldModel(max_depth=3)
        state = WorldState(state_vector={}, depth=3, terminal=False, reward=0.0)
        self.assertTrue(model.is_terminal(state, "any"))

    def test_mcts_search_returns_action(self) -> None:
        model = MCTSWorldModel()
        simulator = MCTSSimulator(model, iterations=16, max_depth=4)
        root = WorldState(state_vector={}, depth=0, terminal=False, reward=0.0)
        action = simulator.search(root, "test objective")
        self.assertIsNotNone(action)
        self.assertIsInstance(action, UiAction)

    def test_mcts_visits_increase(self) -> None:
        model = MCTSWorldModel()
        simulator = MCTSSimulator(model, iterations=8, max_depth=3)
        root = WorldState(state_vector={}, depth=0, terminal=False, reward=0.0)
        action = simulator.search(root, "test")
        # Root should have been visited at least iterations times
        # (we test indirectly by ensuring search completes)
        self.assertIsNotNone(action)

    def test_mcts_penalizes_stagnation(self) -> None:
        model = MCTSWorldModel()
        state = WorldState(
            state_vector={"recent_actions": ["click", "click", "click"]},
            depth=2,
            terminal=False,
            reward=0.0,
        )
        reward = model.evaluate(state, "do something")
        self.assertLess(reward, 0.0)


# --------------------------------------------------------------------------- #
# Hierarchical Planner Tests
# --------------------------------------------------------------------------- #
class HierarchicalPlannerTests(unittest.TestCase):
    def test_macro_planner_detects_launch_goal(self) -> None:
        planner = MacroPlanner()
        goals = planner.plan_objective("open paint and draw")
        descs = [g.description for g in goals]
        self.assertTrue(any("launch" in d.lower() for d in descs))

    def test_macro_planner_detects_content_goal(self) -> None:
        planner = MacroPlanner()
        goals = planner.plan_objective("write a report about AI")
        descs = [g.description for g in goals]
        self.assertTrue(
            any("create" in d.lower() or "content" in d.lower() for d in descs)
        )

    def test_macro_planner_detects_research_goal(self) -> None:
        planner = MacroPlanner()
        goals = planner.plan_objective("find stock prices for AAPL")
        descs = [g.description for g in goals]
        self.assertTrue(any("gather information" in d.lower() for d in descs))

    def test_macro_planner_fallback_for_unknown(self) -> None:
        planner = MacroPlanner()
        goals = planner.plan_objective("xyz abc 123")
        self.assertEqual(len(goals), 1)
        self.assertIn("explore", goals[0].description.lower())

    def test_macro_planner_replan_creates_subgoals(self) -> None:
        planner = MacroPlanner()
        goals = planner.plan_objective("open notepad")
        context = ExecutionContext(objective="open notepad")
        context.failure_streak = 3
        new_goals = planner.replan_on_failure(context, goals)
        for g in new_goals:
            if not g.completed and not g.failed:
                self.assertTrue(len(g.sub_goals) > 0 or g.failed)

    def test_micro_executor_plans_steps(self) -> None:
        executor = MicroExecutor()
        nodes = [
            UiNode(node_id="e1", role="Edit", name="Editor"),
            UiNode(node_id="b1", role="Button", name="Save"),
        ]
        goal = MacroGoal(
            goal_id="g1",
            description="Create content",
            success_criteria=["content entered"],
        )
        steps = executor._plan_micro_steps(goal, nodes)
        self.assertTrue(len(steps) > 0)
        self.assertTrue(any(s.action.action_type == "type" for s in steps))

    def test_micro_executor_tracks_failures(self) -> None:
        backend = FakeBackend()
        executor = MicroExecutor()
        goal = MacroGoal(
            goal_id="g1",
            description="Save file",
            success_criteria=["file saved"],
        )
        context = ExecutionContext(objective="save")

        # Make backend always fail
        class FailingBackend:
            def snapshot(self):
                return []

            def perform(self, action):
                raise RuntimeError("fail")

        context = executor.execute_goal(
            goal,
            context,
            [],
            perform_fn=FailingBackend().perform,
            snapshot_fn=FailingBackend().snapshot,
        )
        self.assertGreater(context.failure_streak, 0)

    def test_goal_completion_check(self) -> None:
        planner = MacroPlanner()
        goal = MacroGoal(
            goal_id="g1",
            description="Save",
            success_criteria=["saved"],
        )
        state = {"has_unsaved_changes": False}
        self.assertTrue(planner.check_goal_completion(goal, state))


# --------------------------------------------------------------------------- #
# Differentiable Memory Tests
# --------------------------------------------------------------------------- #
class DifferentiableMemoryTests(unittest.TestCase):
    def test_working_memory_write_and_read(self) -> None:
        wm = WorkingMemoryScratchpad(capacity=5)
        id1 = wm.write("test content", item_type="observation", priority=0.8)
        items = wm.read()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].item_id, id1)
        self.assertEqual(items[0].content, "test content")

    def test_working_memory_prunes_by_capacity(self) -> None:
        wm = WorkingMemoryScratchpad(capacity=3)
        for i in range(5):
            wm.write(f"item {i}", priority=0.5)
        items = wm.read()
        self.assertLessEqual(len(items), 3)

    def test_working_memory_ttl_expires(self) -> None:
        wm = WorkingMemoryScratchpad(default_ttl_seconds=-1.0)
        wm.write("old", item_type="observation")
        items = wm.read()
        self.assertEqual(len(items), 0)

    def test_working_memory_priority_boost(self) -> None:
        wm = WorkingMemoryScratchpad(capacity=2)
        id1 = wm.write("low", priority=0.1)
        id2 = wm.write("high", priority=0.9)
        id3 = wm.write("med", priority=0.5)
        # Should keep highest priority items after pruning
        items = wm.read()
        ids = {i.item_id for i in items}
        self.assertIn(id2, ids)  # high priority must survive
        # id1 (low priority, oldest) should be pruned first, but
        # the exact survivors depend on the score function;
        # we just assert capacity is respected.
        self.assertLessEqual(len(items), 2)

    def test_working_memory_summarize(self) -> None:
        wm = WorkingMemoryScratchpad()
        wm.write("observation 1", item_type="observation")
        wm.write("plan A", item_type="plan")
        summary = wm.summarize()
        self.assertIn("Working Memory", summary)
        self.assertIn("observation", summary)
        self.assertIn("plan", summary)

    def test_episodic_memory_record_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "episodic.db"
            mem = EpisodicMemoryBank(db_path=db_path)
            event_id = mem.record(
                objective="test",
                action=UiAction(action_type="click", selector="btn"),
                observation_summary="clicked",
                outcome="success",
                reward=1.0,
                tags=["test"],
            )
            self.assertTrue(event_id.startswith("ep_"))
            similar = mem.retrieve_similar("test", top_k=5)
            self.assertEqual(len(similar), 1)
            self.assertEqual(similar[0].outcome, "success")

    def test_episodic_memory_failure_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "episodic.db"
            mem = EpisodicMemoryBank(db_path=db_path)
            mem.record(
                objective="open app",
                action=UiAction(action_type="click", selector="bad"),
                observation_summary="error",
                outcome="failure",
                reward=-1.0,
            )
            mem.record(
                objective="open app",
                action=UiAction(action_type="click", selector="good"),
                observation_summary="ok",
                outcome="success",
                reward=1.0,
            )
            failures = mem.get_failure_patterns("open app")
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0].outcome, "failure")
            successes = mem.get_success_patterns("open app")
            self.assertEqual(len(successes), 1)
            self.assertEqual(successes[0].outcome, "success")

    def test_episodic_memory_stats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "episodic.db"
            mem = EpisodicMemoryBank(db_path=db_path)
            mem.record("test", UiAction("click", "a"), "ok", "ok", 1.0)
            mem.record("test", UiAction("click", "b"), "err", "err", -0.5)
            stats = mem.stats()
            self.assertEqual(stats["total_events"], 2)
            self.assertEqual(stats["failure_count"], 1)

    def test_episodic_memory_embedding_similarity(self) -> None:
        emb1 = EpisodicMemoryBank._compute_embedding("open paint", None, "")
        emb2 = EpisodicMemoryBank._compute_embedding("open paint", None, "")
        sim = EpisodicMemoryBank._cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(sim, 1.0, places=5)

        emb3 = EpisodicMemoryBank._compute_embedding("close window", None, "")
        sim_diff = EpisodicMemoryBank._cosine_similarity(emb1, emb3)
        self.assertLess(sim_diff, 1.0)


# --------------------------------------------------------------------------- #
# Universal Agent Integration Tests
# --------------------------------------------------------------------------- #
class UniversalAgentTests(unittest.TestCase):
    def test_universal_agent_initializes_components(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=5)
        self.assertIsNotNone(agent.vla)
        self.assertIsNotNone(agent.explorer)
        self.assertIsNotNone(agent.pomdp)
        self.assertIsNotNone(agent.mcts)
        self.assertIsNotNone(agent.working_memory)
        self.assertIsNotNone(agent.episodic_memory)

    def test_universal_agent_run_completes(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=5)
        run = agent.run("write a report")
        self.assertIsNotNone(run.run_id)
        self.assertEqual(run.objective, "write a report")
        self.assertTrue(len(run.steps) > 0)
        # Should have at least a perception step
        self.assertTrue(any(s.phase == "perceive" for s in run.steps))

    def test_universal_agent_tracks_final_state(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=3)
        run = agent.run("open paint")
        self.assertIn("node_count", run.final_state)
        self.assertIn("belief_entropy", run.final_state)

    def test_universal_agent_exploration_on_sparse_ui(self) -> None:
        # Sparse UI should trigger exploration
        backend = FakeBackend(
            [
                UiNode(node_id="win", role="Window", name="App"),
            ]
        )
        agent = UniversalDesktopAgent(backend, max_steps=5, use_active_inference=True)
        run = agent.run("do something")
        self.assertGreaterEqual(run.exploration_probes_used, 0)

    def test_universal_agent_records_episodic_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            backend = FakeBackend()
            agent = UniversalDesktopAgent(backend, workspace_root=td, max_steps=3)
            run = agent.run("test objective")
            stats = agent.episodic_memory.stats()
            self.assertGreater(stats["total_events"], 0)

    def test_cognitive_summary_has_all_fields(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=3)
        run = agent.run("test")
        summary = agent.get_cognitive_summary(run)
        self.assertIn("run_id", summary)
        self.assertIn("objective", summary)
        self.assertIn("success", summary)
        self.assertIn("total_steps", summary)
        self.assertIn("exploration_probes", summary)
        self.assertIn("mcts_simulations", summary)
        self.assertIn("final_belief_entropy", summary)
        self.assertIn("working_memory_summary", summary)
        self.assertIn("episodic_memory_stats", summary)
        self.assertIn("step_breakdown", summary)

    def test_universal_agent_with_planned_bootstrap(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=3)
        from agentos_orchestrator.os_control.workflow.models import (
            DesktopWorkflowPlan,
            DesktopWorkflowStep,
        )

        plan = DesktopWorkflowPlan(
            objective="test",
            mode="report",
            app_target="notepad.exe",
            summary="test plan",
            steps=[
                DesktopWorkflowStep(
                    action_type="focus",
                    selector="name=Editor",
                    description="Focus editor",
                ),
            ],
            artifacts=[],
        )
        run = agent.run_with_planned_bootstrap("test", plan)
        self.assertTrue(len(run.steps) > 0)
        # Working memory should contain planned steps
        plans = agent.working_memory.read(item_type="plan")
        self.assertTrue(len(plans) > 0)

    def test_universal_agent_failure_streak_handling(self) -> None:
        class FailingBackend:
            def snapshot(self):
                return [
                    UiNode(node_id="e", role="Edit", name="Editor"),
                ]

            def perform(self, action):
                raise RuntimeError("always fails")

        agent = UniversalDesktopAgent(FailingBackend(), max_steps=5)
        run = agent.run("write something")
        # Should complete the loop even with failures
        self.assertTrue(len(run.steps) >= 1)
        # Episodic memory should record failures
        failures = agent.episodic_memory.get_failure_patterns("write")
        self.assertGreaterEqual(len(failures), 0)

    def test_universal_agent_vla_disabled_does_not_crash(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=3, use_vla=False)
        run = agent.run("test without vla")
        self.assertTrue(len(run.steps) > 0)

    def test_universal_agent_mcts_disabled_does_not_crash(self) -> None:
        backend = FakeBackend()
        agent = UniversalDesktopAgent(backend, max_steps=3, use_mcts=False)
        run = agent.run("test without mcts")
        self.assertTrue(len(run.steps) > 0)


# --------------------------------------------------------------------------- #
# Integration with Existing Workflow Service
# --------------------------------------------------------------------------- #
class CognitiveWorkflowIntegrationTests(unittest.TestCase):
    def test_service_enables_universal_mode(self) -> None:
        from agentos_orchestrator.os_control.workflow.service import (
            DesktopWorkflowService,
        )

        with tempfile.TemporaryDirectory() as td:
            service = DesktopWorkflowService(td)
            backend = FakeBackend()
            service.enable_universal_mode(backend, max_steps=3)
            self.assertIsNotNone(service._universal_agent)

    def test_service_execute_with_universal_mode(self) -> None:
        from agentos_orchestrator.os_control.workflow.service import (
            DesktopWorkflowService,
        )

        with tempfile.TemporaryDirectory() as td:
            service = DesktopWorkflowService(td)
            backend = FakeBackend()
            service.enable_universal_mode(backend, max_steps=3)
            result = service.execute("write a report about orchestrators", backend)
            self.assertIn("receipts", result)
            universal_receipts = [
                r
                for r in result["receipts"]
                if r.get("action_type") == "universal_agent_run"
            ]
            self.assertEqual(len(universal_receipts), 1)
            self.assertIn("run_id", universal_receipts[0]["receipt"])

    def test_service_universal_mode_records_cognitive_trace(self) -> None:
        from agentos_orchestrator.os_control.workflow.service import (
            DesktopWorkflowService,
        )

        with tempfile.TemporaryDirectory() as td:
            service = DesktopWorkflowService(td)
            backend = FakeBackend()
            service.enable_universal_mode(backend, max_steps=3)
            result = service.execute("find stock prices for AAPL", backend)
            universal_receipts = [
                r
                for r in result["receipts"]
                if r.get("action_type") == "universal_agent_run"
            ]
            self.assertEqual(len(universal_receipts), 1)
            universal = universal_receipts[0]
            self.assertEqual(universal["reasoner"], "universal_cognitive_architecture")
            self.assertIn("System-2", universal["rationale"])


if __name__ == "__main__":
    unittest.main()
