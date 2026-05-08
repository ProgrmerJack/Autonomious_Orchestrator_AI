"""Frontier multimodal API bridge for OS-level execution.

The local agent owns pixels, safety, and physical execution. Frontier models own
semantic reasoning. This module keeps that boundary explicit: clients receive a
Set-of-Mark screenshot plus a compact mark table, then must return a tiny JSON
decision such as {"action": "click", "target_id": 42}.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentos_orchestrator.config import default_frontier_model


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

FRONTIER_STATIC_INSTRUCTION = (
    "You are a UI-control state machine for a deterministic Windows "
    "desktop executor. The screenshot has bright numbered Set-of-Mark "
    "boxes over interactable UI elements. Treat this prompt as a "
    "formal contract, not a conversation. Output exactly one JSON "
    "object and no surrounding prose.\n\n"
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
    "confidence is below the runtime confidence floor, output "
    '{"action": "explore", "target_id": null, '
    '"confidence": <score>, "grounding": {...}}.\n'
    "- Explore is the safe escape hatch; it hands control back to "
    "the local ActiveInferenceExplorer for bounded probing.\n"
)


@dataclass(frozen=True, slots=True)
class FrontierContextBudget:
    """Approximate token budget for a single frontier control prompt."""

    total_tokens: int = 3600
    documentation_tokens: int = 900
    memory_tokens: int = 600
    tool_tokens: int = 260
    state_tokens: int = 900
    mark_tokens: int = 700
    objective_tokens: int = 240

    @classmethod
    def from_env(cls) -> "FrontierContextBudget":
        return cls(
            total_tokens=_env_int("AGENTOS_FRONTIER_PROMPT_TOKENS", 3600),
            documentation_tokens=_env_int("AGENTOS_FRONTIER_DOC_TOKENS", 900),
            memory_tokens=_env_int("AGENTOS_FRONTIER_MEMORY_TOKENS", 600),
            tool_tokens=_env_int("AGENTOS_FRONTIER_TOOL_TOKENS", 260),
            state_tokens=_env_int("AGENTOS_FRONTIER_STATE_TOKENS", 900),
            mark_tokens=_env_int("AGENTOS_FRONTIER_MARK_TOKENS", 700),
            objective_tokens=_env_int("AGENTOS_FRONTIER_OBJECTIVE_TOKENS", 240),
        )


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
    context_budget: FrontierContextBudget | None = None

    def instruction_text(self) -> str:
        return f"{self.static_instruction_text()}\n\n{self.runtime_instruction_text()}"

    def static_instruction_text(self) -> str:
        return (
            FRONTIER_STATIC_INSTRUCTION
            + "\n\nObjective-adaptive focus:\n"
            + _objective_adaptive_instruction(self.objective)
        )

    def runtime_instruction_text(self) -> str:
        budget = self._budget()
        docs = _compact_text(
            self.documentation_context,
            budget.documentation_tokens,
            self.objective,
        )
        memory = _compact_text(
            self.memory_context,
            budget.memory_tokens,
            self.objective,
        )
        tools = _compact_text(self.tool_context, budget.tool_tokens, self.objective)
        state_payload = _compact_state_context(self.state_context)
        state = _compact_json(state_payload, budget.state_tokens)
        prompt_mark_payload = self.prompt_mark_payload()
        mark_table = _compact_json(prompt_mark_payload, budget.mark_tokens)
        valid_ids = sorted(_mark_id_set(prompt_mark_payload))
        context_parts = []
        if docs:
            context_parts.append("Official documentation context:\n" + docs)
        if memory:
            context_parts.append("Relevant memory:\n" + memory)
        if tools:
            context_parts.append("Available local tools:\n" + tools)
        context = "\n\n".join(context_parts)
        objective = _truncate_tokens(self.objective.strip(), budget.objective_tokens)
        estimate = self.token_estimate()
        runtime = (
            f"Objective: {objective}\n\n"
            f"Runtime constraints: confidence_floor={self.confidence_floor:.2f}\n\n"
            f"Current State JSON: {state}\n\n"
            f"Mark table JSON: {mark_table}\n\n"
            f"Valid target IDs: {valid_ids}\n\n"
        )
        if context:
            runtime += f"{context}\n\n"
        runtime += (
            "Token budget estimate: "
            f"text_tokens~{estimate['text_tokens']}, "
            f"image_bytes={estimate['image_bytes']}, "
            f"prompt_digest={estimate['prompt_digest']}."
        )
        return _truncate_tokens(runtime, budget.total_tokens)

    def prompt_mark_payload(self) -> dict[str, Any]:
        budget = self._budget()
        return _compact_mark_payload(self.mark_payload, budget.mark_tokens)

    def token_estimate(self) -> dict[str, Any]:
        text = f"{self.static_instruction_text()}\n\n{self.runtime_instruction_text_without_estimate()}"
        prompt_digest = _stable_digest(text)
        return {
            "text_tokens": estimate_tokens(text),
            "image_bytes": len(self.annotated_png),
            "prompt_digest": prompt_digest,
            "static_digest": _stable_digest(self.static_instruction_text()),
            "mark_count": len(_mark_id_set(self.mark_payload)),
            "prompt_mark_count": len(_mark_id_set(self.prompt_mark_payload())),
        }

    def runtime_instruction_text_without_estimate(self) -> str:
        budget = self._budget()
        docs = _compact_text(
            self.documentation_context,
            budget.documentation_tokens,
            self.objective,
        )
        memory = _compact_text(
            self.memory_context,
            budget.memory_tokens,
            self.objective,
        )
        tools = _compact_text(self.tool_context, budget.tool_tokens, self.objective)
        state = _compact_json(
            _compact_state_context(self.state_context), budget.state_tokens
        )
        prompt_mark_payload = self.prompt_mark_payload()
        mark_table = _compact_json(prompt_mark_payload, budget.mark_tokens)
        valid_ids = sorted(_mark_id_set(prompt_mark_payload))
        context_parts = []
        if docs:
            context_parts.append("Official documentation context:\n" + docs)
        if memory:
            context_parts.append("Relevant memory:\n" + memory)
        if tools:
            context_parts.append("Available local tools:\n" + tools)
        context = "\n\n".join(context_parts)
        objective = _truncate_tokens(self.objective.strip(), budget.objective_tokens)
        runtime = (
            f"Objective: {objective}\n\n"
            f"Runtime constraints: confidence_floor={self.confidence_floor:.2f}\n\n"
            f"Current State JSON: {state}\n\n"
            f"Mark table JSON: {mark_table}\n\n"
            f"Valid target IDs: {valid_ids}\n\n"
        )
        if context:
            runtime += f"{context}\n\n"
        return _truncate_tokens(runtime, budget.total_tokens)

    def _budget(self) -> FrontierContextBudget:
        if self.context_budget is not None:
            return self.context_budget
        return _objective_budget(FrontierContextBudget.from_env(), self.objective)


class FrontierClient(Protocol):
    """Provider-neutral multimodal client protocol."""

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision: ...


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
    max_output_tokens: int = 512
    enable_prompt_cache: bool = True
    image_detail: str = "low"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class HTTPFrontierClient:
    """No-SDK HTTP client for Gemini, Claude, and OpenAI-compatible APIs."""

    def __init__(self, config: FrontierProviderConfig) -> None:
        self.config = config
        self.last_usage: dict[str, Any] = {}

    def choose_action(self, prompt: FrontierPrompt) -> FrontierDecision:
        provider = self.config.provider.lower().strip()
        self.last_usage = {}
        if provider in {"openai", "gpt", "gpt-4o", "openai_compatible"}:
            raw = self._call_openai_compatible(prompt)
        elif provider in {"anthropic", "claude", "claude-3.5-sonnet"}:
            raw = self._call_anthropic(prompt)
        elif provider in {"gemini", "google"}:
            raw = self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported frontier provider: {self.config.provider}")
        decision = normalize_decision(
            extract_json_object(raw),
            mark_payload=prompt.prompt_mark_payload(),
            confidence_floor=prompt.confidence_floor,
        )
        decision.metadata["frontier_usage"] = dict(self.last_usage)
        decision.metadata["frontier_prompt"] = prompt.token_estimate()
        return decision

    def _call_openai_compatible(self, prompt: FrontierPrompt) -> str:
        endpoint = self.config.endpoint or "https://api.openai.com/v1/chat/completions"
        image_b64 = base64.b64encode(prompt.annotated_png).decode("ascii")
        image_url: dict[str, Any] = {"url": f"data:image/png;base64,{image_b64}"}
        if self.config.image_detail:
            image_url["detail"] = self.config.image_detail
        body = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "messages": [
                {"role": "system", "content": prompt.static_instruction_text()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.runtime_instruction_text()},
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }
        data = self._post_json(endpoint, body, headers)
        self.last_usage = _usage_payload(
            "openai",
            data,
            prompt,
            model=self.config.model,
        )
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: FrontierPrompt) -> str:
        endpoint = self.config.endpoint or "https://api.anthropic.com/v1/messages"
        image_b64 = base64.b64encode(prompt.annotated_png).decode("ascii")
        system_block: dict[str, Any] = {
            "type": "text",
            "text": prompt.static_instruction_text(),
        }
        if self.config.enable_prompt_cache:
            system_block["cache_control"] = {"type": "ephemeral"}
        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": 0,
            "system": [system_block],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.runtime_instruction_text()},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        }
        if self.config.enable_prompt_cache:
            body["messages"][0]["content"][1]["cache_control"] = {"type": "ephemeral"}
        headers = {
            "x-api-key": self._require_api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post_json(endpoint, body, headers)
        self.last_usage = _usage_payload(
            "anthropic",
            data,
            prompt,
            model=self.config.model,
        )
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
            "systemInstruction": {
                "parts": [{"text": prompt.static_instruction_text()}]
            },
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.config.max_output_tokens,
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt.runtime_instruction_text()},
                        {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                    ],
                }
            ],
        }
        data = self._post_json(endpoint, body, {"Content-Type": "application/json"})
        self.last_usage = _usage_payload(
            "gemini",
            data,
            prompt,
            model=self.config.model,
        )
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


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _truncate_tokens(text: str, max_tokens: int) -> str:
    max_chars = max(1, max_tokens) * 4
    if len(text) <= max_chars:
        return text
    if max_chars < 96:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 48
    return f"{text[:head]}\n[... token-budget omitted middle ...]\n{text[-tail:]}"


def _compact_text(text: str, max_tokens: int, objective: str = "") -> str:
    normalized = _dedupe_lines(text)
    if estimate_tokens(normalized) <= max_tokens:
        return normalized
    lines = [line for line in normalized.splitlines() if line.strip()]
    if not lines:
        return _truncate_tokens(normalized, max_tokens)
    compacted, omitted = _ranked_context_excerpt(
        lines,
        _objective_terms(objective),
        max_tokens * 4,
    )
    prefix = f"[compacted_context omitted_lines={omitted}]\n" if omitted else ""
    return _truncate_tokens(prefix + compacted, max_tokens)


def _ranked_context_excerpt(
    lines: list[str],
    terms: set[str],
    char_budget: int,
) -> tuple[str, int]:
    ranked = sorted(
        enumerate(lines),
        key=lambda item: _line_relevance(item[1], terms),
        reverse=True,
    )
    selected_indexes: set[int] = set()
    used = 0
    for index, line in ranked:
        line_cost = len(line) + 1
        if selected_indexes and used + line_cost > char_budget:
            continue
        selected_indexes.add(index)
        used += line_cost
        if used >= char_budget:
            break
    compacted = "\n".join(lines[index] for index in sorted(selected_indexes))
    omitted = max(0, len(lines) - len(selected_indexes))
    return compacted, omitted


def _dedupe_lines(text: str) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in text.strip().splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        fingerprint = line.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        lines.append(line)
    return "\n".join(lines)


def _objective_terms(objective: str) -> set[str]:
    return {
        term.lower()
        for term in objective.split()
        if len(term) >= 4 and any(char.isalpha() for char in term)
    }


def _line_relevance(line: str, terms: set[str]) -> tuple[int, int, int]:
    lower = line.lower()
    term_hits = sum(1 for term in terms if term in lower)
    structure_bonus = int(line.startswith(("#", "-", "*"))) + int("http" in lower)
    return (term_hits, structure_bonus, min(len(line), 240))


def _objective_adaptive_instruction(objective: str) -> str:
    lower = (objective or "").lower()
    lines: list[str] = []
    if any(
        token in lower
        for token in {"research", "search", "find", "compare", "analyze", "analyse"}
    ):
        lines.append(
            "- Prioritize navigation, reading, and evidence collection controls before destructive actions."
        )
    if any(token in lower for token in {"write", "create", "edit", "draft", "compose"}):
        lines.append(
            "- Prioritize editable surfaces and verify focus before typing or submitting."
        )
    if any(
        token in lower
        for token in {
            "save",
            "export",
            "download",
            "upload",
            "move",
            "copy",
            "rename",
            "delete",
        }
    ):
        lines.append(
            "- Prioritize path/file controls and confirm destination visibility before committing file operations."
        )
    if any(
        token in lower
        for token in {"terminal", "script", "python", "code", "run", "execute"}
    ):
        lines.append(
            "- Favor tool/editor-oriented controls and avoid unrelated browser-only actions."
        )
    if not lines:
        lines.append(
            "- Prefer the most semantically aligned visible control and use explore when grounding is uncertain."
        )
    return "\n".join(lines)


def _objective_budget(
    base: FrontierContextBudget,
    objective: str,
) -> FrontierContextBudget:
    lower = (objective or "").lower()
    docs_boost = 0
    tool_boost = 0
    state_boost = 0

    if any(
        token in lower
        for token in {"research", "search", "find", "compare", "analyze", "analyse"}
    ):
        docs_boost += 180
        state_boost += 60
    if any(
        token in lower for token in {"script", "python", "code", "terminal", "execute"}
    ):
        tool_boost += 120
    if any(
        token in lower
        for token in {"click", "type", "fill", "drag", "scroll", "ui", "desktop"}
    ):
        state_boost += 100

    doc_tokens = max(120, base.documentation_tokens + docs_boost)
    tool_tokens = max(120, base.tool_tokens + tool_boost)
    state_tokens = max(300, base.state_tokens + state_boost)

    # Keep within the total budget by proportionally trimming mutable sections.
    objective_tokens = base.objective_tokens
    memory_tokens = base.memory_tokens
    mark_tokens = base.mark_tokens
    fixed_total = objective_tokens + memory_tokens + mark_tokens
    mutable_total = doc_tokens + tool_tokens + state_tokens
    allowed_mutable = max(300, base.total_tokens - fixed_total)
    if mutable_total > allowed_mutable:
        scale = allowed_mutable / mutable_total
        doc_tokens = max(120, int(doc_tokens * scale))
        tool_tokens = max(120, int(tool_tokens * scale))
        state_tokens = max(300, int(state_tokens * scale))

    return FrontierContextBudget(
        total_tokens=base.total_tokens,
        documentation_tokens=doc_tokens,
        memory_tokens=memory_tokens,
        tool_tokens=tool_tokens,
        state_tokens=state_tokens,
        mark_tokens=mark_tokens,
        objective_tokens=objective_tokens,
    )


def _compact_state_context(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "[omitted:max_depth]"
    if isinstance(value, dict):
        return _compact_state_dict(value, depth)
    if isinstance(value, list):
        return _compact_state_list(value, depth)
    if isinstance(value, str):
        return _truncate_tokens(" ".join(value.split()), 180)
    return value


def _compact_state_dict(value: dict[Any, Any], depth: int) -> dict[str, Any]:
    return {
        str(key): _compact_state_context(item, depth + 1)
        for key, item in value.items()
        if not _is_empty_context_value(item)
    }


def _compact_state_list(value: list[Any], depth: int) -> list[Any]:
    max_items = 8
    items = value[-max_items:]
    compacted = [_compact_state_context(item, depth + 1) for item in items]
    omitted = len(value) - len(items)
    if omitted > 0:
        compacted.insert(0, {"omitted_prior_items": omitted})
    return compacted


def _is_empty_context_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _compact_json(value: Any, max_tokens: int) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if estimate_tokens(text) <= max_tokens:
        return text
    fallback = {
        "compacted": True,
        "digest": _stable_digest(text),
        "excerpt": _truncate_tokens(text, max(1, max_tokens - 80)),
    }
    return json.dumps(
        fallback,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _compact_mark_payload(
    mark_payload: dict[str, Any], max_tokens: int
) -> dict[str, Any]:
    marks = mark_payload.get("marks", [])
    if not isinstance(marks, list):
        return {"marks": []}
    compact_marks = [_compact_mark(mark) for mark in marks if isinstance(mark, dict)]
    payload: dict[str, Any] = {
        "image_size": mark_payload.get("image_size"),
        "marks": compact_marks,
    }
    if estimate_tokens(json.dumps(payload, separators=(",", ":"))) <= max_tokens:
        return payload
    kept: list[dict[str, Any]] = []
    for mark in compact_marks:
        candidate = {**payload, "marks": [*kept, mark]}
        if estimate_tokens(json.dumps(candidate, separators=(",", ":"))) > max_tokens:
            break
        kept.append(mark)
    return {
        "image_size": mark_payload.get("image_size"),
        "marks": kept,
        "omitted_mark_count": max(0, len(compact_marks) - len(kept)),
    }


def _compact_mark(mark: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key in ("id", "bbox", "center", "type", "text_like"):
        if key in mark:
            compacted[key] = mark[key]
    return compacted


def _stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _usage_payload(
    provider: str,
    data: dict[str, Any],
    prompt: FrontierPrompt,
    *,
    model: str = "",
) -> dict[str, Any]:
    usage = (
        data.get("usage")
        or data.get("usageMetadata")
        or data.get("usage_metadata")
        or {}
    )
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "estimated_prompt": prompt.token_estimate(),
    }
    if isinstance(usage, dict):
        payload["raw"] = usage
    return payload


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
    max_output_tokens = _env_int("AGENTOS_FRONTIER_MAX_OUTPUT_TOKENS", 512)
    enable_prompt_cache = _env_bool("AGENTOS_FRONTIER_PROMPT_CACHE", True)
    image_detail = os.environ.get("AGENTOS_FRONTIER_IMAGE_DETAIL", "low").strip()
    if provider in {"openai", "gpt", "gpt-4o"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="openai",
                model=os.environ.get(
                    "AGENTOS_FRONTIER_MODEL",
                    default_frontier_model("openai"),
                ),
                api_key_env="OPENAI_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
                max_output_tokens=max_output_tokens,
                enable_prompt_cache=enable_prompt_cache,
                image_detail=image_detail,
            )
        )
    if provider in {"anthropic", "claude"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="anthropic",
                model=os.environ.get(
                    "AGENTOS_FRONTIER_MODEL",
                    default_frontier_model("anthropic"),
                ),
                api_key_env="ANTHROPIC_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
                max_output_tokens=max_output_tokens,
                enable_prompt_cache=enable_prompt_cache,
                image_detail=image_detail,
            )
        )
    if provider in {"gemini", "google"}:
        return HTTPFrontierClient(
            FrontierProviderConfig(
                provider="gemini",
                model=os.environ.get(
                    "AGENTOS_FRONTIER_MODEL",
                    default_frontier_model("gemini"),
                ),
                api_key_env="GEMINI_API_KEY",
                endpoint=os.environ.get("AGENTOS_FRONTIER_ENDPOINT", ""),
                max_output_tokens=max_output_tokens,
                enable_prompt_cache=enable_prompt_cache,
                image_detail=image_detail,
            )
        )
    raise ValueError(f"Unknown AGENTOS_FRONTIER_PROVIDER={provider!r}")
