from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    frontier_openai: str
    frontier_anthropic: str
    frontier_gemini: str
    gemini_fast: str
    gemini_lite: str
    gemini_workflow: str
    gemini_vision: str


def _env_model(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def configured_model_defaults() -> ModelDefaults:
    return ModelDefaults(
        frontier_openai=_env_model("AGENTOS_FRONTIER_OPENAI_MODEL", "gpt-4o-mini"),
        frontier_anthropic=_env_model(
            "AGENTOS_FRONTIER_ANTHROPIC_MODEL",
            "claude-3-5-haiku-latest",
        ),
        frontier_gemini=_env_model(
            "AGENTOS_FRONTIER_GEMINI_MODEL",
            "gemini-2.5-flash",
        ),
        gemini_fast=_env_model("AGENTOS_GEMINI_FAST_MODEL", "gemini-2.5-flash"),
        gemini_lite=_env_model(
            "AGENTOS_GEMINI_LITE_MODEL",
            "gemini-2.5-flash-lite",
        ),
        gemini_workflow=_env_model(
            "AGENTOS_GEMINI_WORKFLOW_MODEL",
            "gemini-2.5-flash",
        ),
        gemini_vision=_env_model(
            "AGENTOS_GEMINI_VISION_MODEL",
            "gemini-2.5-flash",
        ),
    )


def default_frontier_model(provider: str) -> str:
    defaults = configured_model_defaults()
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return defaults.frontier_openai
    if normalized == "anthropic":
        return defaults.frontier_anthropic
    if normalized == "gemini":
        return defaults.frontier_gemini
    raise ValueError(f"Unknown provider for frontier model defaults: {provider!r}")


def gemini_fast_model() -> str:
    return configured_model_defaults().gemini_fast


def gemini_lite_model() -> str:
    return configured_model_defaults().gemini_lite


def gemini_workflow_model() -> str:
    return configured_model_defaults().gemini_workflow


def gemini_vision_model() -> str:
    return configured_model_defaults().gemini_vision
