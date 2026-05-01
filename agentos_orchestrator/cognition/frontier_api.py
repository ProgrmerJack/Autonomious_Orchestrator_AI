"""Frontier multimodal API bridge for OS-level execution.

The local agent owns pixels, safety, and physical execution. Frontier models own
semantic reasoning. This module keeps that boundary explicit: clients receive a
Set-of-Mark screenshot plus a compact mark table, then must return a tiny JSON
decision such as {"action": "click", "target_id": 42}.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


TARGETED_ACTIONS = {
    "click",
    "double_click",
    "type",
    "hover",
    "focus",
    "select",
    "drag",
    "scroll",
}
NON_TARGETED_ACTIONS = {"tool", "explore"}
ALLOWED_FRONTIER_ACTIONS = TARGETED_ACTIONS | NON_TARGETED_ACTIONS


@dataclass(slots=True)
class FrontierDecision:
    """A normalized action decision returned by a frontier model."""

    action: str
    target_id: int | None = None
    text: str | None = None
    rationale: str = ""
    confidence: float = 0.0
    tool: str | None = None
    code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_action_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "target_id": self.target_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }
        payload["metadata"].update(
            {
                "frontier_confidence": self.confidence,
                "frontier_rationale": self.rationale,
            }
        )
        return payload


@dataclass(slots=True)
class FrontierPrompt:
    """Payload sent to a frontier client."""

    objective: str
    annotated_png: bytes
    mark_payload: dict[str, Any]
    documentation_context: str = ""
    memory_context: str = ""
    tool_context: str = ""
    state_context: dict[str, Any] = field(default_factory=dict)
    confidence_floor: float = 0.55

    def instruction_text(self) -> str:
        docs = self.documentation_context.strip()
        memory = self.memory_context.strip()
        tools = self.tool_context.strip()
        context_parts = []
        if docs:
            context_parts.append("Official documentation context:\n" + docs[:6000])
        if memory:
            context_parts.append("Relevant memory:\n" + memory[:2000])
        if tools:
            context_parts.append("Available local tools:\n" + tools[:2000])
        context = "\n\n".join(context_parts)
        mark_table = json.dumps(self.mark_payload, ensure_ascii=True)
        state = json.dumps(
            self.state_context,
            ensure_ascii=True,
            sort_keys=True,
        )
        valid_ids = sorted(_mark_id_set(self.mark_payload))
        return (
            "You are a UI-control state machine for a deterministic Windows "
            "desktop executor. The screenshot has bright numbered Set-of-Mark "
            "boxes over interactable UI elements. Treat this prompt as a "
            "formal contract, not a conversation. Output exactly one JSON "
            "object and no surrounding prose.\n\n"
            f"Objective: {self.objective}\n\n"
            f"Current State JSON: {state}\n\n"
            f"Mark table JSON: {mark_table}\n\n"
            f"Valid target IDs: {valid_ids}\n\n"
            f"{context}\n\n"
            "Required grounding process: identify the objective, locate the "
            "visual element that satisfies it, read the numeric tag overlaid "
            "on that exact element, then encode that observable mapping in "
            "grounding.target_mapping. Example: 'Tag 42 is the Execute Trade "
            "button'. Do not reveal hidden deliberation; provide concise "
            "observable evidence only.\n\n"
            "Schema:\n"
            "{\n"
            '  "orientation": {\n'
            '    "what_changed": "observable temporal/state change",\n'
            '    "current_blocker": "blocking UI fact or null",\n'
            '    "relevant_history": ["recent action/state facts"]\n'
            "  },\n"
            '  "hypothesis": {\n'
            '    "claim": "why this action should help",\n'
            '    "expected_observation": "what should be true next",\n'
            '    "risk": "low|medium|high"\n'
            "  },\n"
            '  "action": "click|double_click|type|hover|focus|select|drag|scroll|tool|explore",\n'
            '  "target_id": 1,\n'
            '  "text": null,\n'
            '  "tool": null,\n'
            '  "code": null,\n'
            '  "confidence": 0.0,\n'
            '  "grounding": {\n'
            '    "state_evidence": "focus/modal/task facts used",\n'
            '    "visual_evidence": "visible UI evidence used",\n'
            '    "target_mapping": "Tag N is the exact target"\n'
            "  },\n"
            '  "rationale": "brief public reason",\n'
            '  "metadata": {}\n'
            "}\n\n"
            "Rules:\n"
            "- orientation and hypothesis are required. They must be based "
            "on visible state, temporal trace, receipts, memory, or mark IDs.\n"
            "- For targeted UI actions, target_id is required and must be "
            "one of the valid target IDs. Never invent IDs or coordinates.\n"
            "- For action='type', include text. For action='tool', include "
            "tool and code, with target_id null.\n"
            "- If the correct tag is ambiguous, missing, occluded, or "
            f"confidence is below {self.confidence_floor:.2f}, output "
            '{"action": "explore", "target_id": null, '
            '"confidence": <score>, "grounding": {...}}.\n'
            "- Explore is the safe escape hatch; it hands control back to "
            "the local ActiveInferenceExplorer for bounded probing.\n"
        )


class FrontierClient(Protocol):
    """Provider-neutral multimodal client protocol."""

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision:
        """Return the next grounded action or local tool request."""


class StaticFrontierClient:
    """Deterministic client for tests and offline development."""

    def __init__(self, decision: FrontierDecision | dict[str, Any]) -> None:
        self._decision = decision
        self.calls: list[FrontierPrompt] = []

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision:
        self.calls.append(prompt)
        return normalize_decision(
            self._decision,
            mark_payload=prompt.mark_payload,
        )


@dataclass(slots=True)
class FrontierProviderConfig:
    provider: str
    model: str
    api_key_env: str
    endpoint: str = ""
    timeout_seconds: int = 60

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class HTTPFrontierClient:
    """No-SDK HTTP client for Gemini, Claude, and OpenAI-compatible APIs."""

    def __init__(self, config: FrontierProviderConfig) -> None:
        self.config = config

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision:
        provider = self.config.provider.lower().strip()
        if provider in {"openai", "gpt", "gpt-4o", "openai_compatible"}:
            raw = self._call_openai_compatible(prompt)
        elif provider in {"anthropic", "claude", "claude-3.5-sonnet"}:
            raw = self._call_anthropic(prompt)
        elif provider in {"gemini", "google"}:
            raw = self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported frontier provider: {self.config.provider}")
        return normalize_decision(
            extract_json_object(raw),
            mark_payload=prompt.mark_payload,
            confidence_floor=prompt.confidence_floor,
        )

    def _call_openai_compatible(self, prompt: FrontierPrompt) -> str:
        endpoint = self.config.endpoint or "https://api.openai.com/v1/chat/completions"
        image_b64 = base64.b64encode(prompt.annotated_png).decode("ascii")
        body = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.instruction_text()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }
        data = self._post_json(endpoint, body, headers)
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: FrontierPrompt) -> str:
        endpoint = self.config.endpoint or "https://api.anthropic.com/v1/messages"
        image_b64 = base64.b64encode(prompt.annotated_png).decode("ascii")
        body = {
            "model": self.config.model,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt.instruction_text()},
                    ],
                }
            ],
        }
        headers = {
            "x-api-key": self._require_api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post_json(endpoint, body, headers)
        parts = data.get("content", [])
        return "\n".join(
            part.get("text", "") for part in parts if part.get("type") == "text"
        )

    def _call_gemini(self, prompt: FrontierPrompt) -> str:
        api_key = self._require_api_key()
        endpoint = self.config.endpoint or (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={api_key}"
        )
        image_b64 = base64.b64encode(prompt.annotated_png).decode("ascii")
        body = {
            "generationConfig": {"temperature": 0},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt.instruction_text()},
                        {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                    ],
                }
            ],
        }
        data = self._post_json(endpoint, body, {"Content-Type": "application/json"})
        candidates = data.get("candidates", [])
        if not candidates:
            return "{}"
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts)

    def _require_api_key(self) -> str:
        key = self.config.api_key
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.config.api_key_env}")
        return key

    def _post_json(
        self,
        endpoint: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout_seconds
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Frontier API HTTP {exc.code}: {detail}") from exc


def extract_json_object(text: str | dict[str, Any]) -> dict[str, Any]:
    """Extract the first JSON object from model text."""
    if isinstance(text, dict):
        return text
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    parsed = _scan_first_json_object(stripped)
    if parsed is None:
        raise ValueError(f"No JSON object found in frontier response: {stripped[:200]}")
    return parsed


def normalize_decision(
    raw: FrontierDecision | dict[str, Any],
    mark_payload: dict[str, Any] | None = None,
    confidence_floor: float | None = None,
) -> FrontierDecision:
    if isinstance(raw, FrontierDecision):
        return raw
    metadata = (
        dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    )
    grounding = raw.get("grounding")
    if isinstance(grounding, dict):
        metadata["frontier_grounding"] = grounding
    orientation = raw.get("orientation")
    if isinstance(orientation, dict):
        metadata["frontier_orientation"] = orientation
    hypothesis = raw.get("hypothesis")
    if isinstance(hypothesis, dict):
        metadata["frontier_hypothesis"] = hypothesis
        expected = hypothesis.get("expected_observation")
        if expected is not None:
            metadata["expected_observation"] = str(expected)
    action = str(raw.get("action", "click")).strip().lower() or "click"
    if action not in ALLOWED_FRONTIER_ACTIONS:
        raise ValueError(f"Unsupported frontier action: {action}")

    target_id = raw.get("target_id", raw.get("id"))
    parsed_target: int | None
    if target_id is None or target_id == "":
        parsed_target = None
    else:
        parsed_target = int(target_id)
    confidence = float(raw.get("confidence", 0.0) or 0.0)

    if _should_explore(action, confidence, confidence_floor):
        metadata["frontier_original_action"] = action
        metadata["frontier_original_target_id"] = parsed_target
        action = "explore"
        parsed_target = None

    if action in TARGETED_ACTIONS and parsed_target is None:
        raise ValueError(f"Frontier action {action!r} requires target_id")
    valid_ids = _mark_id_set(mark_payload or {})
    if action in TARGETED_ACTIONS and valid_ids and parsed_target not in valid_ids:
        raise ValueError(
            f"Unknown Set-of-Mark target_id: {parsed_target}; "
            f"valid IDs are {sorted(valid_ids)}"
        )

    return FrontierDecision(
        action=action,
        target_id=parsed_target,
        text=str(raw["text"]) if raw.get("text") is not None else None,
        rationale=str(raw.get("rationale", "")),
        confidence=confidence,
        tool=str(raw["tool"]) if raw.get("tool") is not None else None,
        code=str(raw["code"]) if raw.get("code") is not None else None,
        metadata=metadata,
    )


def _should_explore(
    action: str,
    confidence: float,
    confidence_floor: float | None,
) -> bool:
    if action in NON_TARGETED_ACTIONS or confidence_floor is None:
        return False
    return confidence < confidence_floor


def _mark_id_set(mark_payload: dict[str, Any]) -> set[int]:
    marks = mark_payload.get("marks", [])
    ids: set[int] = set()
    if not isinstance(marks, list):
        return ids
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        try:
            ids.add(int(mark["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def _scan_first_json_object(text: str) -> dict[str, Any] | None:
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = _json_object_from_start(text, start)
        if candidate is not None:
            return candidate
    return None


def _json_object_from_start(text: str, start: int) -> dict[str, Any] | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def default_provider_from_env() -> HTTPFrontierClient | None:
    """Create a frontier client from env vars, or None if no provider is configured."""
    provider = os.environ.get("AGENTOS_FRONTIER_PROVIDER", "").strip().lower()
    if not provider:
        return None
    if provider in {"openai", "gpt", "gpt-4o"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="openai",
                model=os.environ.get("AGENTOS_FRONTIER_MODEL", "gpt-4o"),
                api_key_env="OPENAI_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
            )
        )
    if provider in {"anthropic", "claude"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="anthropic",
                model=os.environ.get(
                    "AGENTOS_FRONTIER_MODEL", "claude-3-5-sonnet-latest"
                ),
                api_key_env="ANTHROPIC_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
            )
        )
    if provider in {"gemini", "google"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="gemini",
                model=os.environ.get("AGENTOS_FRONTIER_MODEL", "gemini-1.5-pro"),
                api_key_env="GEMINI_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
            )
        )
    raise ValueError(f"Unknown AGENTOS_FRONTIER_PROVIDER={provider!r}")
