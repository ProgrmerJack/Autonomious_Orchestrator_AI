"""Cognitive Architecture for Universal Agent PC Control.

This package implements the research blueprint for bridging the gap between
Generalizing Agents and Universal Agents through:

1. Vision-Language-Action (VLA) affordance grounding
2. Active Inference exploration
3. POMDP state representation
4. System 2 MCTS deliberative reasoning
5. Hierarchical Macro/Micro planning
6. Differentiable Working + Episodic memory
7. Learned generative world model (neural dynamics)
8. Local fast VLA (zero API latency)
9. Semantic dense vector memory
10. Pure pixel-based POMDP
"""

from __future__ import annotations

from .active_inference import ActiveInferenceExplorer
from .differentiable_memory import EpisodicMemoryBank, WorkingMemoryScratchpad
from .hierarchical_planner import MacroPlanner, MicroExecutor
from .learned_world_model import LearnedGenerativeWorldModel, MLPDynamics
from .local_vla import LocalFastVLA
from .mcts_simulator import MCTSWorldModel, MCTSSimulator
from .pixel_pomdp import PurePixelEnvironment, PixelFeatureExtractor
from .pomdp_state import POMDPBeliefState, POMDPEnvironmentModel
from .semantic_memory import SemanticEmbedder, SemanticEpisodicMemory
from .universal_agent import UniversalDesktopAgent
from .vla_affordance import VLAActionSpace, VLAAffordanceGrounding

__all__ = [
    "ActiveInferenceExplorer",
    "EpisodicMemoryBank",
    "WorkingMemoryScratchpad",
    "LearnedGenerativeWorldModel",
    "MLPDynamics",
    "LocalFastVLA",
    "MacroPlanner",
    "MicroExecutor",
    "MCTSWorldModel",
    "MCTSSimulator",
    "PixelFeatureExtractor",
    "PurePixelEnvironment",
    "POMDPBeliefState",
    "POMDPEnvironmentModel",
    "SemanticEmbedder",
    "SemanticEpisodicMemory",
    "UniversalDesktopAgent",
    "VLAActionSpace",
    "VLAAffordanceGrounding",
]
