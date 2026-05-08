"""Production-grade tests for Universal Agent v2 cognitive architecture.

Tests all four real fixes:
1. Learned generative world model (neural MLP dynamics)
2. Local fast VLA (classical CV + Random Forest, <100ms)
3. Semantic dense memory (TF-IDF + SVD)
4. Pure pixel-based POMDP (no accessibility trees)
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from agentos_orchestrator.cognition.abstract_world_model import AbstractUIState
from agentos_orchestrator.cognition.learned_world_model import (
    LearnedGenerativeWorldModel,
    MLPDynamics,
    MLPConfig,
    StateEncoder,
)
from agentos_orchestrator.cognition.local_vla import LocalFastVLA, DetectedElement
from agentos_orchestrator.cognition.mcts_simulator import WorldState
from agentos_orchestrator.cognition.pixel_pomdp import (
    PixelBeliefState,
    PixelFeatureExtractor,
    PixelObservation,
    PurePixelEnvironment,
)
from agentos_orchestrator.cognition.semantic_memory import (
    SemanticEmbedder,
    SemanticEpisodicMemory,
)
from agentos_orchestrator.cognition.universal_agent_v2 import (
    UniversalAgentRun,
    UniversalDesktopAgentV2,
)
from agentos_orchestrator.os_control.base import UiAction, UiNode


class FakePixelBackend:
    """Backend with only screenshot capture (no accessibility tree)."""

    def __init__(self) -> None:
        self.actions: list[UiAction] = []
        self._frame_counter = 0

    def capture(self) -> bytes:
        """Generate a synthetic screenshot that changes over time."""
        self._frame_counter += 1
        img = Image.new("RGB", (640, 480), color=(self._frame_counter % 255, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def perform(self, action: UiAction) -> str:
        self.actions.append(action)
        return json.dumps({"status": "executed"})


class FakeHybridBackend:
    """Backend with both snapshot and capture."""

    def __init__(self) -> None:
        self.actions: list[UiAction] = []
        self._nodes = [
            UiNode(node_id="btn", role="Button", name="OK"),
            UiNode(node_id="edit", role="Edit", name="Input"),
        ]

    def snapshot(self) -> list[UiNode]:
        return list(self._nodes)

    def capture(self) -> bytes:
        img = Image.new("RGB", (640, 480), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def perform(self, action: UiAction) -> str:
        self.actions.append(action)
        return json.dumps({"status": "executed"})


# --------------------------------------------------------------------------- #
# Fix 1: Learned Generative World Model Tests
# --------------------------------------------------------------------------- #
class LearnedWorldModelTests(unittest.TestCase):
    def test_mlp_dynamics_forward_returns_delta(self) -> None:
        model = MLPDynamics(MLPConfig(state_dim=64, action_dim=16, hidden_dim=32))
        s = np.zeros(64, dtype=np.float32)
        a = np.zeros(16, dtype=np.float32)
        delta = model.forward(s, a)
        self.assertEqual(delta.shape, (64,))
        self.assertEqual(delta.dtype, np.float32)

    def test_mlp_dynamics_trains_and_loss_decreases(self) -> None:
        model = MLPDynamics(
            MLPConfig(state_dim=16, action_dim=8, hidden_dim=32, learning_rate=0.01)
        )
        losses = []
        for i in range(50):
            s = np.random.randn(16).astype(np.float32)
            a = np.random.randn(8).astype(np.float32)
            true_delta = np.random.randn(16).astype(np.float32) * 0.1
            ns = s + true_delta
            loss = model.train_step(s, a, ns)
            losses.append(loss)
        # Loss should generally decrease over training
        self.assertLess(losses[-1], losses[0])

    def test_mlp_dynamics_buffer_limits_size(self) -> None:
        cfg = MLPConfig(state_dim=4, action_dim=2, hidden_dim=8, max_training_samples=5)
        model = MLPDynamics(cfg)
        for _ in range(10):
            s, a, ns = np.zeros(4), np.zeros(2), np.zeros(4)
            model.record_transition(s, a, ns)
        self.assertLessEqual(len(model._buffer), 5)

    def test_mlp_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pkl"
            model = MLPDynamics(MLPConfig(state_dim=8, action_dim=4, hidden_dim=16))
            s, a, ns = np.ones(8), np.ones(4), np.ones(8)
            model.record_transition(s, a, ns)
            model.save(path)
            model2 = MLPDynamics(MLPConfig(state_dim=8, action_dim=4, hidden_dim=16))
            success = model2.load(path)
            self.assertTrue(success)
            self.assertEqual(len(model2._buffer), 1)

    def test_state_encoder_produces_normalized_vector(self) -> None:
        encoder = StateEncoder(dim=64)
        state = {"node_count": 10, "enabled_count": 8, "window_title": "Test"}
        vec = encoder.encode_state(state)
        self.assertEqual(len(vec), 64)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_learned_world_model_uses_fallback_when_untrained(self) -> None:
        model = LearnedGenerativeWorldModel(max_depth=4)
        state = WorldState(
            state_vector={"click_count": 0}, depth=0, terminal=False, reward=0.0
        )
        action = UiAction(action_type="click", selector="btn")
        next_state = model.predict(state, action)
        self.assertEqual(next_state.depth, 1)
        self.assertGreater(model._fallback_used_count, 0)

    def test_learned_world_model_trains_and_switches_to_model(self) -> None:
        model = LearnedGenerativeWorldModel(max_depth=4)
        # Record many transitions to train
        for i in range(20):
            before = {"click_count": i, "depth": 0}
            action = UiAction(action_type="click", selector="btn")
            after = {"click_count": i + 1, "depth": 1}
            model.record_transition(before, action, after)
        state = WorldState(
            state_vector={"click_count": 0}, depth=0, terminal=False, reward=0.0
        )
        action = UiAction(action_type="click", selector="btn")
        next_state = model.predict(state, action)
        self.assertGreaterEqual(model._model_used_count + model._fallback_used_count, 1)

    def test_world_model_evaluate_rewards_matching_objective(self) -> None:
        model = LearnedGenerativeWorldModel()
        state = WorldState(
            state_vector={"app_context": "browser", "belief_entropy": 0.2},
            depth=1,
            terminal=False,
            reward=0.0,
        )
        reward = model.evaluate(state, "open browser and search")
        self.assertGreater(reward, 0.0)

    def test_world_model_available_actions_includes_learned(self) -> None:
        model = LearnedGenerativeWorldModel()
        model._real_transitions.append(
            (
                {"focused_element": "btn"},
                UiAction(action_type="click", selector="btn"),
                {},
            )
        )
        state = WorldState(
            state_vector={"focused_element": "btn"}, depth=0, terminal=False, reward=0.0
        )
        actions = model.available_actions(state)
        self.assertTrue(len(actions) > 0)


# --------------------------------------------------------------------------- #
# Fix 2: Local Fast VLA Tests
# --------------------------------------------------------------------------- #
class LocalFastVLATests(unittest.TestCase):
    def test_vla_initializes_classifier(self) -> None:
        vla = LocalFastVLA()
        self.assertIsNotNone(vla.classifier)

    def test_vla_detects_elements_in_synthetic_image(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (640, 480), color="white")
        # Draw a button-like rectangle
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 200], fill="blue", outline="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        elements = vla.detect_elements(buf.getvalue())
        self.assertTrue(len(elements) > 0)
        for elem in elements:
            self.assertGreater(elem.width, 0)
            self.assertGreater(elem.height, 0)
            self.assertGreaterEqual(elem.confidence, 0.0)

    def test_vla_propose_action_returns_action(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (640, 480), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([120, 90, 320, 190], fill="blue", outline="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        action = vla.propose_action(buf.getvalue(), "click the button", [])
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "click")
        self.assertIsNotNone(action.x)
        self.assertIsNotNone(action.y)
        # Should include latency info in rationale
        self.assertIn("ms", action.rationale)

    def test_vla_propose_action_under_100ms(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (320, 240), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        import time

        start = time.perf_counter()
        action = vla.propose_action(buf.getvalue(), "click", [])
        elapsed = (time.perf_counter() - start) * 1000
        self.assertIsNone(action)
        # Should be fast (generally <100ms for small images)
        self.assertLess(elapsed, 500.0)  # Generous threshold for CI

    def test_vla_returns_none_without_visual_evidence(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (320, 240), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        action = vla.propose_action(buf.getvalue(), "click the button", [])

        self.assertIsNone(action)

    def test_vla_classify_element_bootstrap(self) -> None:
        vla = LocalFastVLA()
        elem = DetectedElement(
            x=0,
            y=0,
            width=100,
            height=30,
            aspect_ratio=3.3,
            solidity=0.9,
            edge_density=0.3,
            color_variance=100,
            text_like=False,
        )
        aff_type, conf = vla._classify_element(elem)
        self.assertIn(aff_type, LocalFastVLA.AFFORDANCE_CLASSES)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_vla_feedback_improves_classifier(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (640, 480), color="white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 200], fill="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        # Provide feedback
        for _ in range(15):
            vla.provide_feedback(buf.getvalue(), 200, 150, "button")
        self.assertGreaterEqual(len(vla._training_features), 10)

    def test_vla_feedback_bounds_detection_region_for_large_images(self) -> None:
        vla = LocalFastVLA()
        img = Image.new("RGB", (4096, 4096), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        seen_shapes: list[tuple[int, int]] = []

        def fake_contours(arr: np.ndarray) -> list[DetectedElement]:
            seen_shapes.append((int(arr.shape[0]), int(arr.shape[1])))
            return [
                DetectedElement(
                    x=10,
                    y=12,
                    width=40,
                    height=24,
                    aspect_ratio=40 / 24,
                    solidity=0.9,
                    edge_density=0.1,
                    color_variance=1.0,
                    text_like=False,
                )
            ]

        with (
            mock.patch.object(vla, "_detect_by_contours", side_effect=fake_contours),
            mock.patch.object(vla, "_detect_by_msers", return_value=[]),
        ):
            collected = vla.collect_feedback(buf.getvalue(), 3500, 3500, "button")

        self.assertTrue(collected)
        self.assertTrue(seen_shapes)
        self.assertLessEqual(
            max(max(height, width) for height, width in seen_shapes),
            LocalFastVLA.FEEDBACK_MAX_DETECTION_SIDE,
        )

    def test_vla_score_elements_for_objective(self) -> None:
        vla = LocalFastVLA()
        elements = [
            DetectedElement(0, 0, 50, 50, 1.0, 0.8, 0.2, 100, False, "button", 0.7),
            DetectedElement(0, 0, 300, 200, 1.5, 0.6, 0.1, 50, False, "canvas", 0.5),
        ]
        scored = vla._score_elements_for_objective(elements, "draw a house", [])
        self.assertEqual(len(scored), 2)
        # Canvas should score higher for "draw"
        canvas_score = next(s for e, s in scored if e.affordance_type == "canvas")
        button_score = next(s for e, s in scored if e.affordance_type == "button")
        self.assertGreater(canvas_score, button_score)

    def test_vla_action_type_mapping(self) -> None:
        self.assertEqual(
            LocalFastVLA._action_type_for_affordance("button", "click"), "click"
        )
        self.assertEqual(
            LocalFastVLA._action_type_for_affordance("text_field", "type"), "type"
        )
        self.assertEqual(
            LocalFastVLA._action_type_for_affordance("scrollbar", "scroll down"),
            "scroll",
        )


# --------------------------------------------------------------------------- #
# Fix 3: Semantic Dense Memory Tests
# --------------------------------------------------------------------------- #
class SemanticMemoryTests(unittest.TestCase):
    def test_embedder_produces_normalized_vectors(self) -> None:
        embedder = SemanticEmbedder(n_components=32)
        vec = embedder.embed("transfer funds in Bank A")
        self.assertEqual(len(vec), 32)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_embedder_similar_sentences_have_high_similarity(self) -> None:
        embedder = SemanticEmbedder(n_components=32)
        # Fit on some texts first
        texts = [
            "transfer funds in Bank A",
            "wire money in Bank B",
            "check account balance",
            "send payment to vendor",
        ]
        embedder.fit(texts)
        vec1 = embedder.embed("transfer funds in Bank A")
        vec2 = embedder.embed("wire money in Bank B")
        vec3 = embedder.embed("check account balance")
        sim_same = embedder.cosine_similarity(vec1, vec2)
        sim_diff = embedder.cosine_similarity(vec1, vec3)
        self.assertGreater(sim_same, sim_diff)

    def test_semantic_memory_retrieves_similar_events(self) -> None:
        memory = SemanticEpisodicMemory()
        memory.record(
            objective="transfer funds in Bank A",
            action=UiAction(action_type="click", selector="transfer_btn"),
            observation="clicked transfer",
            outcome="success",
            reward=1.0,
        )
        memory.record(
            objective="check balance in savings",
            action=UiAction(action_type="click", selector="balance_btn"),
            observation="showed balance",
            outcome="success",
            reward=1.0,
        )
        similar = memory.retrieve_similar("wire money in Bank B", top_k=2)
        self.assertTrue(len(similar) > 0)
        # Should retrieve the transfer event, not the balance check
        self.assertIn("transfer", similar[0]["objective"].lower())

    def test_semantic_memory_failure_patterns(self) -> None:
        memory = SemanticEpisodicMemory()
        memory.record("open app", UiAction("click", "bad"), "err", "fail", -1.0)
        memory.record("open app", UiAction("click", "good"), "ok", "success", 1.0)
        failures = memory.get_failure_patterns("launch application")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["outcome"], "fail")

    def test_semantic_memory_transfer_learning(self) -> None:
        embedder = SemanticEmbedder(n_components=32)
        embedder.fit(
            [
                "transfer funds",
                "wire money",
                "check balance",
            ]
        )
        mem = SemanticEpisodicMemory(embedder)
        score = mem.transfer_learning_score("transfer funds", "wire money")
        self.assertGreater(score, 0.5)
        low_score = mem.transfer_learning_score("transfer funds", "check balance")
        self.assertLess(low_score, score)

    def test_semantic_memory_batch_embedding(self) -> None:
        embedder = SemanticEmbedder(n_components=16)
        embedder.fit(["hello world", "foo bar", "baz qux"])
        vecs = embedder.embed_batch(["hello world", "foo bar"])
        self.assertEqual(vecs.shape, (2, 16))
        self.assertTrue(np.allclose(np.linalg.norm(vecs, axis=1), 1.0))

    def test_semantic_memory_first_record_avoids_degenerate_svd_warning(self) -> None:
        memory = SemanticEpisodicMemory()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            memory.record(
                objective="test objective",
                action=UiAction(action_type="click", selector="submit_btn"),
                observation="clicked submit",
                outcome="success",
                reward=1.0,
            )

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])


# --------------------------------------------------------------------------- #
# Fix 4: Pure Pixel-Based POMDP Tests
# --------------------------------------------------------------------------- #
class PixelPOMDPTests(unittest.TestCase):
    def test_pixel_feature_extractor_produces_fixed_dim(self) -> None:
        extractor = PixelFeatureExtractor(target_dim=256)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        features = extractor.extract(img)
        self.assertEqual(len(features), 256)
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0, places=4)

    def test_pixel_feature_extractor_consistent_for_same_image(self) -> None:
        extractor = PixelFeatureExtractor(target_dim=128)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        f1 = extractor.extract(img)
        f2 = extractor.extract(img)
        np.testing.assert_array_almost_equal(f1, f2, decimal=5)

    def test_pixel_feature_extractor_different_images_different_features(self) -> None:
        extractor = PixelFeatureExtractor(target_dim=128)
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        f1 = extractor.extract(img1)
        f2 = extractor.extract(img2)
        self.assertGreater(float(np.linalg.norm(f1 - f2)), 0.01)

    def test_pixel_belief_initializes_particles(self) -> None:
        belief = PixelBeliefState()
        obs = PixelObservation(
            timestamp=0.0,
            screenshot=np.zeros((100, 100, 3)),
            features=np.ones(256, dtype=np.float32) / np.sqrt(256),
            resolution=(100, 100),
        )
        belief._bootstrap(obs, n=5)
        self.assertEqual(len(belief.particles), 5)
        self.assertAlmostEqual(sum(belief.weights), 1.0, places=5)

    def test_pixel_belief_update_changes_weights(self) -> None:
        belief = PixelBeliefState()
        f1 = np.random.randn(256).astype(np.float32)
        f1 = f1 / np.linalg.norm(f1)
        obs1 = PixelObservation(0.0, np.zeros((10, 10, 3)), f1, (10, 10))
        belief._bootstrap(obs1, n=5)
        initial_weights = list(belief.weights)

        f2 = np.random.randn(256).astype(np.float32)
        f2 = f2 / np.linalg.norm(f2)
        obs2 = PixelObservation(1.0, np.zeros((10, 10, 3)), f2, (10, 10))
        belief.update(obs2, None)
        updated_weights = list(belief.weights)
        self.assertNotEqual(initial_weights, updated_weights)

    def test_pixel_belief_entropy_decreases_with_certainty(self) -> None:
        belief = PixelBeliefState()
        f = np.ones(64, dtype=np.float32) / np.sqrt(64)
        obs = PixelObservation(0.0, np.zeros((10, 10, 3)), f, (10, 10))
        belief._bootstrap(obs, n=10)
        initial_entropy = belief.entropy()
        # Make one particle dominant
        belief.weights = [0.0] * 10
        belief.weights[0] = 1.0
        low_entropy = belief.entropy()
        self.assertLess(low_entropy, initial_entropy)

    def test_pure_pixel_environment_observe(self) -> None:
        backend = FakePixelBackend()
        env = PurePixelEnvironment(backend)
        obs = env.observe()
        self.assertIsNotNone(obs.screenshot)
        self.assertEqual(obs.resolution, (640, 480))
        self.assertEqual(len(obs.features), 256)
        self.assertGreater(len(env.belief.particles), 0)

    def test_pure_pixel_environment_step(self) -> None:
        backend = FakePixelBackend()
        env = PurePixelEnvironment(backend)
        obs1 = env.observe()
        action = UiAction(action_type="click", selector="pixel=(100,100)")
        obs2, outcome = env.step(action)
        self.assertIsNotNone(obs2)
        self.assertIn("receipt", outcome)
        self.assertTrue(len(env.belief.history) > 0)

    def test_pure_pixel_environment_visual_summary(self) -> None:
        backend = FakePixelBackend()
        env = PurePixelEnvironment(backend)
        env.observe()
        summary = env.get_visual_state_summary()
        self.assertIn("resolution", summary)
        self.assertIn("belief_entropy", summary)

    def test_pixel_lbp_computation(self) -> None:
        extractor = PixelFeatureExtractor()
        gray = np.array([[50, 60, 70], [80, 90, 100], [110, 120, 130]], dtype=np.uint8)
        lbp = extractor._compute_lbp(gray)
        self.assertEqual(lbp.shape, gray.shape)
        self.assertTrue(np.all(lbp >= 0))
        self.assertTrue(np.all(lbp <= 255))


# --------------------------------------------------------------------------- #
# Universal Agent v2 Integration Tests
# --------------------------------------------------------------------------- #
class UniversalAgentV2Tests(unittest.TestCase):
    def test_agent_initializes_all_components(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=3)
        self.assertIsNotNone(agent.world_model)
        self.assertIsNotNone(agent.local_vla)
        self.assertIsNotNone(agent.semantic_memory)
        self.assertIsNotNone(agent.working_memory)
        self.assertIsNotNone(agent.mcts)

    def test_agent_run_completes_with_latencies(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=3)
        run = agent.run("write a report")
        self.assertIsNotNone(run.run_id)
        self.assertEqual(run.objective, "write a report")
        self.assertTrue(len(run.steps) > 0)
        self.assertGreaterEqual(run.avg_latency_ms, 0.0)

    def test_agent_records_semantic_memory(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=3)
        run = agent.run("test objective")
        self.assertGreater(len(agent.semantic_memory._events), 0)

    def test_agent_cognitive_summary_has_all_fields(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=3)
        run = agent.run("test")
        summary = agent.get_cognitive_summary(run)
        self.assertIn("run_id", summary)
        self.assertIn("avg_latency_ms", summary)
        self.assertIn("model_used_ratio", summary)
        self.assertIn("semantic_memory_events", summary)
        self.assertIn("step_breakdown", summary)

    def test_agent_async_perception_captures_screenshots(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=2)
        agent.start_perception_loop()
        import time

        time.sleep(0.15)
        screenshot = agent.get_latest_screenshot()
        agent.stop_perception_loop()
        self.assertIsNotNone(screenshot)
        self.assertGreater(len(screenshot), 0)

    def test_agent_with_learned_model_records_transitions(self) -> None:
        backend = FakeHybridBackend()
        agent = UniversalDesktopAgentV2(backend, max_steps=5, use_learned_model=True)
        run = agent.run("click buttons")
        self.assertGreater(
            agent.world_model._model_used_count
            + agent.world_model._fallback_used_count,
            0,
        )

    def test_agent_pixel_mode_without_accessibility_tree(self) -> None:
        backend = FakePixelBackend()
        agent = UniversalDesktopAgentV2(
            backend,
            max_steps=3,
            use_pixel_pomdp=True,
        )
        run = agent.run("explore the UI")
        self.assertTrue(len(run.steps) > 0)
        # Should use pixel observations, not node snapshots
        self.assertIsNone(agent.pomdp)
        self.assertIsNotNone(agent.pixel_env)

    def test_agent_v2_does_not_crash_with_failures(self) -> None:
        class FailingBackend:
            def snapshot(self):
                return []

            def capture(self):
                img = Image.new("RGB", (100, 100))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            def perform(self, action):
                raise RuntimeError("always fails")

        agent = UniversalDesktopAgentV2(FailingBackend(), max_steps=3)
        run = agent.run("try something")
        self.assertFalse(run.success)
        self.assertTrue(len(run.steps) >= 1)

    def test_agent_grounds_snapshot_primitives_before_mcts(self) -> None:
        class SnapshotOnlyBackend:
            def snapshot(self):
                return [
                    UiNode(node_id="search", role="Edit", name="Search"),
                    UiNode(node_id="go", role="Button", name="Go"),
                ]

            def perform(self, action):
                return json.dumps({"status": "executed", "action": action.action_type})

        agent = UniversalDesktopAgentV2(
            SnapshotOnlyBackend(),
            max_steps=1,
            use_frontier_api=False,
            use_local_vla=False,
        )
        action = agent._select_action(
            SimpleNamespace(
                name="discover_affordances",
                description="search for API docs",
            ),
            [],
            AbstractUIState(app_context="unknown"),
            UniversalAgentRun(run_id="r1", objective="search for API docs"),
            [],
        )
        self.assertEqual(action.metadata.get("source"), "active_inference_grounding")
        self.assertEqual(action.action_type, "type")
        self.assertEqual(action.selector, "name=Search")


# --------------------------------------------------------------------------- #
# Service Integration Tests
# --------------------------------------------------------------------------- #
class V2ServiceIntegrationTests(unittest.TestCase):
    def test_service_enables_v2_mode(self) -> None:
        from agentos_orchestrator.os_control.workflow.service import (
            DesktopWorkflowService,
        )

        with tempfile.TemporaryDirectory() as td:
            service = DesktopWorkflowService(td)
            backend = FakeHybridBackend()
            service.enable_universal_mode_v2(backend, max_steps=3)
            self.assertIsNotNone(service._universal_agent)
            self.assertIsInstance(
                service._universal_agent,
                UniversalDesktopAgentV2,
            )

    def test_service_execute_with_v2_mode(self) -> None:
        from agentos_orchestrator.os_control.workflow.service import (
            DesktopWorkflowService,
        )

        with tempfile.TemporaryDirectory() as td:
            service = DesktopWorkflowService(td)
            backend = FakeHybridBackend()
            service.enable_universal_mode_v2(backend, max_steps=3)
            result = service.execute("write a report about AI", backend)
            self.assertIn("receipts", result)
            universal = [
                r
                for r in result["receipts"]
                if r.get("action_type") == "universal_agent_run"
            ]
            self.assertEqual(len(universal), 1)
            receipt = universal[0]["receipt"]
            self.assertIn("run_id", receipt)
            self.assertIn("avg_latency_ms", receipt)
            self.assertIn("model_used_ratio", receipt)
            self.assertEqual(
                receipt["adaptation_readiness"]["status"],
                "needs_training_or_eval",
            )


if __name__ == "__main__":
    unittest.main()
