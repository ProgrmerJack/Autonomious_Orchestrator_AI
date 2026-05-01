"""Generic discovery of higher-level control surfaces in unknown apps.

Instead of assuming every unfamiliar application must be manipulated through
raw clicks, this module looks for exposed API, JSON, DOM, and developer
surfaces that can provide stronger control channels.
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from agentos_orchestrator.os_control.base import UiNode

from .capability_profile import CapabilityProfile


@dataclass(slots=True)
class ControlSurfaceCandidate:
    kind: str
    channel: str
    confidence: float
    rationale: str
    action_type: str = ""
    selector: str = ""
    value: str | None = None
    endpoint: str = ""
    workflow: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class GenericControlSurfaceDiscoverer:
    DEFAULT_LOOPBACK_PORTS = (
        8000,
        8080,
        5173,
        3000,
        3001,
        4000,
        4200,
        5000,
        5001,
        8765,
        9000,
        9001,
    )
    LOOPBACK_PROBE_PATHS = (
        "/openapi.json",
        "/swagger.json",
        "/swagger/v1/swagger.json",
        "/api/openapi.json",
        "/api/swagger.json",
        "/",
    )
    LOOPBACK_GRAPHQL_PATHS = (
        "/graphql",
        "/api/graphql",
    )
    LOOPBACK_PROBE_TIMEOUT_SECONDS = 0.35
    LOOPBACK_CACHE_TTL_SECONDS = 10.0
    GRAPHQL_INTROSPECTION_QUERY = (
        "query AgentOSIntrospection { __schema { queryType { name } mutationType { name } "
        "types { kind name fields { name args { name defaultValue type { kind name ofType { kind name ofType { kind name } } } } "
        "type { kind name ofType { kind name ofType { kind name } } } } inputFields { name defaultValue type { kind name ofType { kind name ofType { kind name } } } } } } }"
    )
    LOCAL_ARTIFACT_NAME_HINTS = {
        "openapi",
        "swagger",
        "postman",
        "collection",
        "manifest",
        "schema",
        "graphql",
        "api",
        "readme",
    }
    LOCAL_ARTIFACT_FILE_NAMES = {
        "package.json",
        "manifest.json",
        "readme.md",
        "readme.txt",
    }
    LOCAL_ARTIFACT_SUFFIXES = {
        ".json",
        ".yaml",
        ".yml",
        ".http",
        ".rest",
        ".md",
        ".txt",
        ".toml",
        ".ini",
        ".cfg",
        ".config",
    }
    LOCAL_ARTIFACT_SKIP_DIRS = {
        ".git",
        ".venv",
        "node_modules",
        ".agentos",
        "runs",
        "artifacts",
        "__pycache__",
    }
    MAX_LOCAL_ARTIFACT_FILES = 64
    MAX_LOCAL_ARTIFACT_SIZE = 512_000
    MAX_LOCAL_ARTIFACT_CONTENT = 4_000
    OBJECTIVE_QUERY_KEYWORDS = {"search", "find", "query", "lookup", "list", "read"}
    OBJECTIVE_MUTATION_KEYWORDS = {
        "create",
        "new",
        "add",
        "submit",
        "post",
        "send",
        "write",
        "update",
        "edit",
        "change",
        "delete",
        "remove",
    }
    API_SURFACE_KEYWORDS = {
        "api",
        "endpoint",
        "swagger",
        "openapi",
        "graphql",
        "json-rpc",
        "rpc",
        "webhook",
        "developer",
        "devtools",
        "console",
    }
    JSON_SURFACE_KEYWORDS = {
        "json",
        "payload",
        "schema",
        "request body",
        "response body",
        "manifest",
        "config",
        "configuration",
    }
    ACTIVATION_ROLES = {
        "button",
        "menuitem",
        "hyperlink",
        "tabitem",
        "tab",
        "listitem",
        "treeitem",
    }
    EDIT_ROLES = {"edit", "document", "pane"}
    ENDPOINT_RE = re.compile(
        r"https?://(?:localhost|127\.0\.0\.1|[a-z0-9.-]+)(?::\d+)?(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?",
        flags=re.I,
    )
    METHOD_PATH_RE = re.compile(
        r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS)\s+((?:https?://[^\s)\]}]+)|(?:/[A-Za-z0-9._~/{}/:-]+))",
        flags=re.I,
    )
    RELATIVE_API_PATH_RE = re.compile(
        r"(?:^|[\s(])(/(?:api|v\d+|graphql|rpc|webhooks?)[A-Za-z0-9._~/{}/:-]*)",
        flags=re.I,
    )
    PORT_HINT_RE = re.compile(
        r"(?:localhost|127\.0\.0\.1)[:/](\d{2,5})|\b(?:api_port|port)\b[^0-9]{0,5}(\d{2,5})",
        flags=re.I,
    )

    def __init__(
        self,
        workspace_root: str | Path = ".",
        loopback_ports: list[int] | tuple[int, ...] | None = None,
        loopback_cache_ttl_seconds: float = LOOPBACK_CACHE_TTL_SECONDS,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.loopback_ports = [
            int(port)
            for port in (loopback_ports or self.DEFAULT_LOOPBACK_PORTS)
            if 1 <= int(port) <= 65535
        ]
        self.loopback_cache_ttl_seconds = float(loopback_cache_ttl_seconds)
        self._workspace_artifact_cache: list[dict[str, Any]] | None = None
        self._loopback_probe_cache: dict[str, Any] | None = None

    def discover(
        self,
        profile: CapabilityProfile,
        nodes: list[UiNode],
        objective: str,
        documentation_context: str = "",
        preferred_channels: list[str] | None = None,
        active_fingerprinting: bool = False,
    ) -> list[ControlSurfaceCandidate]:
        preferred_channels = preferred_channels or []
        text_blob = self._text_blob(nodes, documentation_context)
        candidates: list[ControlSurfaceCandidate] = []

        candidates.extend(
            self._workflow_candidates_from_docs(
                profile,
                objective,
                documentation_context,
                preferred_channels,
            )
        )
        candidates.extend(
            self._workflow_candidates_from_workspace(
                profile,
                objective,
                preferred_channels,
            )
        )
        if active_fingerprinting:
            candidates.extend(
                self._workflow_candidates_from_loopback(
                    profile,
                    objective,
                    preferred_channels,
                    text_blob,
                )
            )

        for endpoint in self.ENDPOINT_RE.findall(text_blob):
            candidates.append(
                ControlSurfaceCandidate(
                    kind="api_endpoint",
                    channel="api",
                    confidence=self._endpoint_confidence(endpoint),
                    rationale=f"Found explicit API endpoint {endpoint}",
                    endpoint=endpoint,
                    metadata={"endpoint": endpoint},
                )
            )

        for node in nodes:
            node_text = self._node_text(node)
            selector = self._selector_for_node(node)
            role = node.role.lower()
            if any(keyword in node_text for keyword in self.API_SURFACE_KEYWORDS):
                candidates.append(
                    ControlSurfaceCandidate(
                        kind="api_surface",
                        channel="api",
                        confidence=0.7,
                        rationale=f"Visible API-oriented control '{node.name or node.role}'",
                        action_type="click"
                        if role in self.ACTIVATION_ROLES
                        else "focus",
                        selector=selector,
                        metadata={"node_id": node.node_id},
                    )
                )
            if any(keyword in node_text for keyword in self.JSON_SURFACE_KEYWORDS):
                candidates.append(
                    ControlSurfaceCandidate(
                        kind="json_surface",
                        channel="api",
                        confidence=0.68,
                        rationale=f"Visible JSON-oriented control '{node.name or node.role}'",
                        action_type="focus" if role in self.EDIT_ROLES else "click",
                        selector=selector,
                        metadata={"node_id": node.node_id},
                    )
                )

        objective_lower = objective.lower()
        if any(
            keyword in objective_lower
            for keyword in {"json", "api", "endpoint", "payload"}
        ):
            for candidate in candidates:
                candidate.confidence = min(1.0, candidate.confidence + 0.08)

        if profile.app_family in {"browser", "terminal", "electron_app"}:
            for candidate in candidates:
                if candidate.channel == "api":
                    candidate.confidence = min(1.0, candidate.confidence + 0.05)

        for candidate in candidates:
            if candidate.channel in preferred_channels:
                candidate.confidence = min(1.0, candidate.confidence + 0.06)
            if candidate.channel in profile.control_channels:
                candidate.confidence = min(1.0, candidate.confidence + 0.04)

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return self._dedupe(candidates)

    @staticmethod
    def _node_text(node: UiNode) -> str:
        return " ".join([node.role, node.name, str(node.metadata)]).lower()

    @staticmethod
    def _selector_for_node(node: UiNode) -> str:
        if node.name:
            return f"name={node.name}"
        return node.node_id

    @staticmethod
    def _text_blob(nodes: list[UiNode], documentation_context: str) -> str:
        chunks = [documentation_context.lower()]
        for node in nodes:
            chunks.append(GenericControlSurfaceDiscoverer._node_text(node))
        return "\n".join(chunks)

    @staticmethod
    def _endpoint_confidence(endpoint: str) -> float:
        lower = endpoint.lower()
        if "localhost" in lower or "127.0.0.1" in lower:
            return 0.92
        if any(token in lower for token in {"/api", "graphql", "swagger", "openapi"}):
            return 0.82
        return 0.72

    @staticmethod
    def _dedupe(
        candidates: list[ControlSurfaceCandidate],
    ) -> list[ControlSurfaceCandidate]:
        deduped: dict[tuple[str, str, str], ControlSurfaceCandidate] = {}
        for candidate in candidates:
            key = (candidate.kind, candidate.selector, candidate.endpoint)
            current = deduped.get(key)
            if current is None or candidate.confidence > current.confidence:
                deduped[key] = candidate
        return list(deduped.values())

    def _workflow_candidates_from_docs(
        self,
        profile: CapabilityProfile,
        objective: str,
        documentation_context: str,
        preferred_channels: list[str],
    ) -> list[ControlSurfaceCandidate]:
        if not documentation_context.strip():
            return []
        candidates: list[ControlSurfaceCandidate] = []
        for source in self._documentation_sources(documentation_context):
            step_specs = self._documented_step_specs(source["url"], source["excerpt"])
            if not step_specs:
                continue
            workflow = self._build_workflow(step_specs, objective, source["excerpt"])
            if not workflow:
                continue
            auth_env_keys = self._auth_env_keys(source["excerpt"])
            confidence = self._workflow_confidence(
                workflow,
                source["excerpt"],
                profile,
                preferred_channels,
            )
            endpoint = workflow[0].get("url", "")
            candidates.append(
                ControlSurfaceCandidate(
                    kind="api_workflow",
                    channel="api",
                    confidence=confidence,
                    rationale=(
                        f"Synthesized documented API workflow from {source['url']} "
                        f"with {len(workflow)} step(s)"
                    ),
                    endpoint=endpoint,
                    workflow=workflow,
                    metadata={
                        "documentation_url": source["url"],
                        "auth_env_keys": auth_env_keys,
                        "workflow": workflow,
                        "payload_hint": workflow[1].get("json_body")
                        if len(workflow) > 1
                        else {},
                        "response_hint": self._response_hint(step_specs),
                    },
                )
            )
        return candidates

    def _workflow_candidates_from_workspace(
        self,
        profile: CapabilityProfile,
        objective: str,
        preferred_channels: list[str],
    ) -> list[ControlSurfaceCandidate]:
        candidates: list[ControlSurfaceCandidate] = []
        for source in self._workspace_artifact_sources():
            step_specs = list(source.get("step_specs", []))
            excerpt = str(source.get("excerpt", ""))
            if not step_specs:
                step_specs = self._documented_step_specs("", excerpt)
            if not step_specs:
                continue
            workflow = self._build_workflow(step_specs, objective, excerpt)
            if not workflow:
                continue
            auth_env_keys = self._auth_env_keys(excerpt)
            confidence = min(
                0.98,
                self._workflow_confidence(
                    workflow, excerpt, profile, preferred_channels
                )
                + float(source.get("confidence_bonus", 0.0)),
            )
            endpoint = workflow[0].get("url", "")
            artifact_path = str(source.get("path", ""))
            artifact_kind = str(source.get("kind", "workspace_artifact"))
            candidates.append(
                ControlSurfaceCandidate(
                    kind="api_workflow",
                    channel="api",
                    confidence=confidence,
                    rationale=(
                        f"Synthesized local {artifact_kind} workflow from {artifact_path} "
                        f"with {len(workflow)} step(s)"
                    ),
                    endpoint=endpoint,
                    workflow=workflow,
                    metadata={
                        "artifact_kind": artifact_kind,
                        "artifact_path": artifact_path,
                        "auth_env_keys": auth_env_keys,
                        "workflow": workflow,
                        "payload_hint": workflow[1].get("json_body")
                        if len(workflow) > 1
                        else {},
                        "response_hint": self._response_hint(step_specs),
                        "discovery_source": "workspace_artifact",
                    },
                )
            )
        return candidates

    def _workflow_candidates_from_loopback(
        self,
        profile: CapabilityProfile,
        objective: str,
        preferred_channels: list[str],
        text_blob: str,
    ) -> list[ControlSurfaceCandidate]:
        candidates: list[ControlSurfaceCandidate] = []
        for source in self._active_loopback_service_sources(text_blob):
            step_specs = list(source.get("step_specs", []))
            excerpt = str(source.get("excerpt", ""))
            endpoint = str(source.get("endpoint", ""))
            auth_env_keys = self._auth_env_keys(excerpt)
            if step_specs:
                workflow = self._build_workflow(step_specs, objective, excerpt)
                if workflow:
                    confidence = min(
                        0.99,
                        self._workflow_confidence(
                            workflow,
                            excerpt,
                            profile,
                            preferred_channels,
                        )
                        + float(source.get("confidence_bonus", 0.0)),
                    )
                    candidates.append(
                        ControlSurfaceCandidate(
                            kind="api_workflow",
                            channel="api",
                            confidence=confidence,
                            rationale=(
                                f"Synthesized active loopback workflow from {endpoint} "
                                f"with {len(workflow)} step(s)"
                            ),
                            endpoint=endpoint,
                            workflow=workflow,
                            metadata={
                                "artifact_kind": str(
                                    source.get("kind", "loopback_service")
                                ),
                                "artifact_path": endpoint,
                                "auth_env_keys": auth_env_keys,
                                "workflow": workflow,
                                "payload_hint": workflow[1].get("json_body")
                                if len(workflow) > 1
                                else {},
                                "response_hint": self._response_hint(step_specs),
                                "discovery_source": "loopback_service",
                                "service_fingerprint": dict(
                                    source.get("fingerprint", {})
                                ),
                            },
                        )
                    )
                    continue
            if endpoint:
                candidates.append(
                    ControlSurfaceCandidate(
                        kind="api_endpoint",
                        channel="api",
                        confidence=min(
                            0.97,
                            self._endpoint_confidence(endpoint)
                            + float(source.get("confidence_bonus", 0.0)),
                        ),
                        rationale=f"Fingerprint suggests active loopback API at {endpoint}",
                        endpoint=endpoint,
                        metadata={
                            "endpoint": endpoint,
                            "discovery_source": "loopback_service",
                            "service_fingerprint": dict(source.get("fingerprint", {})),
                        },
                    )
                )
        return candidates

    def _workspace_artifact_sources(self) -> list[dict[str, Any]]:
        if self._workspace_artifact_cache is not None:
            return self._workspace_artifact_cache
        if not self.workspace_root.exists():
            self._workspace_artifact_cache = []
            return self._workspace_artifact_cache

        sources: list[dict[str, Any]] = []
        for artifact_path in self._candidate_artifact_files():
            try:
                text = artifact_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            excerpt = text[: self.MAX_LOCAL_ARTIFACT_CONTENT].strip()
            if not excerpt:
                continue
            structured = self._structured_artifact_source(artifact_path, excerpt)
            if structured is not None:
                sources.append(structured)
                continue
            if (
                self.ENDPOINT_RE.search(excerpt)
                or self.METHOD_PATH_RE.search(excerpt)
                or self.RELATIVE_API_PATH_RE.search(excerpt)
            ):
                sources.append(
                    {
                        "kind": "text_artifact",
                        "path": str(artifact_path),
                        "excerpt": excerpt,
                        "confidence_bonus": 0.08,
                    }
                )
        self._workspace_artifact_cache = sources[:16]
        return self._workspace_artifact_cache

    def _candidate_artifact_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.workspace_root.rglob("*"):
            if len(files) >= self.MAX_LOCAL_ARTIFACT_FILES:
                break
            if not path.is_file():
                continue
            if any(part in self.LOCAL_ARTIFACT_SKIP_DIRS for part in path.parts):
                continue
            name = path.name.lower()
            suffix = path.suffix.lower()
            is_candidate_name = (
                name in self.LOCAL_ARTIFACT_FILE_NAMES
                or name.startswith(".env")
                or any(hint in name for hint in self.LOCAL_ARTIFACT_NAME_HINTS)
            )
            if not is_candidate_name and suffix not in self.LOCAL_ARTIFACT_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.MAX_LOCAL_ARTIFACT_SIZE:
                continue
            files.append(path)
        return files

    def _structured_artifact_source(
        self,
        artifact_path: Path,
        excerpt: str,
    ) -> dict[str, Any] | None:
        payload = self._try_parse_json_object(excerpt)
        if payload is None:
            return None
        structured = self._structured_payload_source(
            payload,
            str(artifact_path),
        )
        if structured is None:
            return None
        structured["path"] = str(artifact_path)
        return structured

    def _structured_payload_source(
        self,
        payload: dict[str, Any],
        source_id: str,
        endpoint_url: str = "",
    ) -> dict[str, Any] | None:
        step_specs = self._openapi_step_specs(payload)
        kind = "openapi_spec"
        confidence_bonus = 0.18
        if not step_specs:
            step_specs = self._graphql_step_specs(payload, endpoint_url or "/graphql")
            kind = "graphql_introspection"
            confidence_bonus = 0.2
        if not step_specs:
            step_specs = self._postman_step_specs(payload)
            kind = "postman_collection"
            confidence_bonus = 0.14
        if not step_specs:
            return None
        inferred_endpoint = endpoint_url or str(step_specs[0].get("url", ""))
        return {
            "kind": kind,
            "path": source_id,
            "endpoint": inferred_endpoint,
            "excerpt": self._structured_excerpt(payload, step_specs),
            "step_specs": step_specs,
            "confidence_bonus": confidence_bonus,
        }

    @staticmethod
    def _try_parse_json_object(text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def _openapi_step_specs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        paths = payload.get("paths")
        if not isinstance(paths, dict):
            return []
        base_url = self._openapi_base_url(payload)
        step_specs: list[dict[str, Any]] = []
        for raw_path, path_item in paths.items():
            if not isinstance(raw_path, str) or not isinstance(path_item, dict):
                continue
            shared_parameters = (
                path_item.get("parameters")
                if isinstance(path_item.get("parameters"), list)
                else []
            )
            for method, operation in path_item.items():
                if method.lower() not in {
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                    "options",
                }:
                    continue
                if not isinstance(operation, dict):
                    continue
                parameters = list(shared_parameters)
                if isinstance(operation.get("parameters"), list):
                    parameters.extend(operation["parameters"])
                step_specs.append(
                    {
                        "method": method.upper(),
                        "url": self._apply_parameter_examples(
                            self._normalize_url(raw_path, base_url),
                            parameters,
                            payload,
                        ),
                        "source": "openapi",
                        "json_example": self._operation_json_example(
                            operation, payload
                        ),
                        "response_example": self._operation_response_example(
                            operation, payload
                        ),
                    }
                )
        return step_specs[:12]

    def _graphql_step_specs(
        self,
        payload: dict[str, Any],
        endpoint_url: str,
    ) -> list[dict[str, Any]]:
        schema = self._graphql_schema(payload)
        if not isinstance(schema, dict):
            return []
        type_map = {
            str(item.get("name")): item
            for item in schema.get("types", [])
            if isinstance(item, dict) and item.get("name")
        }
        step_specs: list[dict[str, Any]] = []
        for operation_kind, type_name in (
            ("query", self._graphql_named_type(schema.get("queryType"))),
            ("mutation", self._graphql_named_type(schema.get("mutationType"))),
        ):
            if not type_name:
                continue
            type_payload = type_map.get(type_name)
            if not isinstance(type_payload, dict):
                continue
            for field in type_payload.get("fields", [])[:6]:
                if not isinstance(field, dict):
                    continue
                payload_example = self._graphql_operation_payload(
                    field,
                    operation_kind,
                    type_map,
                )
                if not payload_example:
                    continue
                response_example = self._graphql_response_example(field, type_map)
                step_specs.append(
                    {
                        "method": "POST",
                        "url": endpoint_url,
                        "source": "graphql_introspection",
                        "operation_kind": operation_kind,
                        "field_name": str(field.get("name") or ""),
                        "json_example": payload_example,
                        "response_example": response_example,
                    }
                )
        return step_specs[:12]

    def _postman_step_specs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        items = payload.get("item")
        if not isinstance(items, list):
            return []
        step_specs: list[dict[str, Any]] = []

        def walk(entries: list[Any]) -> None:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                request = entry.get("request")
                if isinstance(request, dict):
                    url = self._postman_request_url(request.get("url"))
                    if url:
                        step_specs.append(
                            {
                                "method": str(request.get("method", "GET")).upper(),
                                "url": url,
                                "source": "postman",
                                "json_example": self._postman_json_body(
                                    request.get("body")
                                ),
                            }
                        )
                child_items = entry.get("item")
                if isinstance(child_items, list):
                    walk(child_items)

        walk(items)
        return step_specs[:12]

    def _structured_excerpt(
        self,
        payload: dict[str, Any],
        step_specs: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for step in step_specs[:8]:
            line = f"{step['method']} {step['url']}"
            if step.get("json_example"):
                line += " JSON " + json.dumps(step["json_example"], ensure_ascii=True)
            if step.get("response_example") not in (None, {}, []):
                line += " RESPONSE " + json.dumps(
                    step["response_example"], ensure_ascii=True
                )
            lines.append(line)
        lines.extend(self._security_requirement_phrases(payload))
        return "\n".join(lines)

    def _security_requirement_phrases(self, payload: dict[str, Any]) -> list[str]:
        phrases: list[str] = []
        schemes: dict[str, Any] = {}
        components = payload.get("components")
        if isinstance(components, dict) and isinstance(
            components.get("securitySchemes"), dict
        ):
            schemes.update(components["securitySchemes"])
        if isinstance(payload.get("securityDefinitions"), dict):
            schemes.update(payload["securityDefinitions"])
        for scheme in schemes.values():
            if not isinstance(scheme, dict):
                continue
            scheme_type = str(scheme.get("type", "")).lower()
            scheme_name = str(scheme.get("scheme", "")).lower()
            header_name = str(scheme.get("name", ""))
            if scheme_type == "http" and scheme_name == "bearer":
                phrases.append("Authorization: Bearer token required.")
            elif scheme_type == "oauth2":
                phrases.append("OAuth access token required.")
            elif scheme_type == "apikey":
                if header_name:
                    phrases.append(f"{header_name}: API key required.")
                else:
                    phrases.append("API key required.")
        return phrases

    def _openapi_base_url(self, payload: dict[str, Any]) -> str:
        servers = payload.get("servers")
        if isinstance(servers, list):
            for server in servers:
                if isinstance(server, dict) and isinstance(server.get("url"), str):
                    return str(server["url"]).strip()
        host = payload.get("host")
        if isinstance(host, str) and host:
            schemes = (
                payload.get("schemes")
                if isinstance(payload.get("schemes"), list)
                else []
            )
            scheme = next(
                (entry for entry in schemes if isinstance(entry, str) and entry), "http"
            )
            base_path = (
                payload.get("basePath")
                if isinstance(payload.get("basePath"), str)
                else ""
            )
            return f"{scheme}://{host}{base_path}"
        return ""

    def _operation_json_example(
        self,
        operation: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        request_body = operation.get("requestBody")
        if isinstance(request_body, dict):
            content = request_body.get("content")
            if isinstance(content, dict):
                json_payload = self._json_content_example(content, payload)
                if json_payload not in (None, {}, []):
                    return json_payload
        parameters = operation.get("parameters")
        if isinstance(parameters, list):
            for parameter in parameters:
                resolved_parameter = self._resolve_parameter(parameter, payload)
                if not isinstance(resolved_parameter, dict):
                    continue
                if str(resolved_parameter.get("in", "")).lower() != "body":
                    continue
                schema = resolved_parameter.get("schema")
                if isinstance(schema, dict):
                    example = self._schema_example(schema, payload)
                    if example not in (None, {}, []):
                        return example
        return None

    def _operation_response_example(
        self,
        operation: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        responses = operation.get("responses")
        if not isinstance(responses, dict):
            return None
        preferred_statuses = ("200", "201", "202", "default")
        ordered_items = [
            (status, responses[status])
            for status in preferred_statuses
            if status in responses
        ]
        ordered_items.extend(
            (status, value)
            for status, value in responses.items()
            if status not in preferred_statuses
        )
        for _status, response in ordered_items:
            resolved = self._resolve_parameter(response, payload)
            if not isinstance(resolved, dict):
                continue
            content = resolved.get("content")
            if isinstance(content, dict):
                example = self._json_content_example(content, payload)
                if example not in (None, {}, []):
                    return example
            examples = resolved.get("examples")
            if isinstance(examples, dict):
                for entry in examples.values():
                    if isinstance(entry, dict) and "value" in entry:
                        return entry.get("value")
            schema = resolved.get("schema")
            if isinstance(schema, dict):
                example = self._schema_example(schema, payload)
                if example not in (None, {}, []):
                    return example
        return None

    def _json_content_example(
        self,
        content: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        for content_type, media_type in content.items():
            if "json" not in str(content_type).lower():
                continue
            if not isinstance(media_type, dict):
                continue
            example = media_type.get("example")
            if example not in (None, {}, []):
                return example
            examples = media_type.get("examples")
            if isinstance(examples, dict):
                for entry in examples.values():
                    if isinstance(entry, dict) and entry.get("value") not in (
                        None,
                        {},
                        [],
                    ):
                        return entry["value"]
            schema = media_type.get("schema")
            if isinstance(schema, dict):
                schema_example = self._schema_example(schema, payload)
                if schema_example not in (None, {}, []):
                    return schema_example
        return None

    def _schema_example(
        self,
        schema: dict[str, Any],
        payload: dict[str, Any],
        depth: int = 0,
        hint: str = "value",
    ) -> Any:
        if depth > 6:
            return None
        if "$ref" in schema:
            resolved = self._resolve_ref(str(schema.get("$ref") or ""), payload)
            if isinstance(resolved, dict):
                return self._schema_example(resolved, payload, depth + 1, hint=hint)
        example = schema.get("example")
        if example not in (None, {}, []):
            return example
        examples = schema.get("examples")
        if isinstance(examples, dict):
            for entry in examples.values():
                if isinstance(entry, dict) and entry.get("value") not in (None, {}, []):
                    return entry.get("value")
        if "default" in schema:
            return schema.get("default")
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        for composite_key in ("allOf", "oneOf", "anyOf"):
            entries = schema.get(composite_key)
            if not isinstance(entries, list):
                continue
            if composite_key == "allOf":
                merged: dict[str, Any] = {}
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    example_value = self._schema_example(
                        entry, payload, depth + 1, hint=hint
                    )
                    if isinstance(example_value, dict):
                        merged.update(example_value)
                    elif example_value not in (None, {}, []):
                        return example_value
                if merged:
                    return merged
            else:
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    example_value = self._schema_example(
                        entry, payload, depth + 1, hint=hint
                    )
                    if example_value not in (None, {}, []):
                        return example_value
        schema_type = str(schema.get("type", "")).lower()
        if schema_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                item_example = self._schema_example(
                    items, payload, depth + 1, hint=hint
                )
                return [item_example] if item_example is not None else []
            return []
        if schema_type in {"integer"}:
            return 0
        if schema_type in {"number"}:
            return 0.0
        if schema_type in {"boolean"}:
            return False
        if schema_type in {"string"}:
            return f"<{schema.get('title') or schema.get('name') or hint or 'value'}>"
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            if isinstance(schema.get("additionalProperties"), dict):
                additional = self._schema_example(
                    schema["additionalProperties"],
                    payload,
                    depth + 1,
                    hint=hint,
                )
                return {"example": additional}
            return None
        materialized: dict[str, Any] = {}
        for name, property_schema in list(properties.items())[:8]:
            if not isinstance(property_schema, dict):
                continue
            materialized[name] = self._schema_example(
                property_schema,
                payload,
                depth + 1,
                hint=name,
            )
            if materialized[name] is None:
                materialized[name] = f"<{name}>"
        return materialized

    def _postman_request_url(self, url_field: Any) -> str:
        if isinstance(url_field, str):
            return url_field
        if isinstance(url_field, dict):
            raw = url_field.get("raw")
            if isinstance(raw, str):
                return raw
            protocol = str(url_field.get("protocol", "http"))
            host = url_field.get("host")
            path = url_field.get("path")
            host_text = ""
            if isinstance(host, list):
                host_text = ".".join(str(part) for part in host)
            elif isinstance(host, str):
                host_text = host
            path_text = ""
            if isinstance(path, list):
                path_text = "/" + "/".join(str(part) for part in path)
            elif isinstance(path, str):
                path_text = path if path.startswith("/") else f"/{path}"
            if host_text:
                return f"{protocol}://{host_text}{path_text}"
        return ""

    def _postman_json_body(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            return {}
        raw = body.get("raw")
        if isinstance(raw, str):
            parsed = self._try_parse_json_object(raw)
            if parsed is not None:
                return parsed
        return {}

    def _active_loopback_service_sources(self, text_blob: str) -> list[dict[str, Any]]:
        ports = tuple(self._candidate_loopback_ports(text_blob))
        cache = self._loopback_probe_cache
        if (
            cache is not None
            and cache.get("ports") == ports
            and (time.monotonic() - float(cache.get("ts", 0.0)))
            <= self.loopback_cache_ttl_seconds
        ):
            return list(cache.get("services", []))
        services: list[dict[str, Any]] = []
        for port in ports:
            if not self._is_loopback_port_open(port):
                continue
            service = self._fingerprint_loopback_service(port)
            if service is not None:
                services.append(service)
        self._loopback_probe_cache = {
            "ports": ports,
            "services": services,
            "ts": time.monotonic(),
        }
        return services

    def _candidate_loopback_ports(self, text_blob: str) -> list[int]:
        ports = list(self.loopback_ports)
        ports.extend(self._extract_ports(text_blob))
        for source in self._workspace_artifact_sources():
            ports.extend(self._extract_ports(str(source.get("excerpt", ""))))
        deduped: list[int] = []
        seen: set[int] = set()
        for port in ports:
            if port in seen or port < 1 or port > 65535:
                continue
            seen.add(port)
            deduped.append(port)
        return deduped[:12]

    def _extract_ports(self, text: str) -> list[int]:
        ports: list[int] = []
        for match in self.PORT_HINT_RE.findall(text or ""):
            for group in match:
                if group:
                    ports.append(int(group))
        return ports

    @staticmethod
    def _is_loopback_port_open(port: int) -> bool:
        try:
            with socket.create_connection(
                ("127.0.0.1", int(port)),
                timeout=0.12,
            ):
                return True
        except OSError:
            return False

    def _fingerprint_loopback_service(self, port: int) -> dict[str, Any] | None:
        base_url = f"http://127.0.0.1:{port}"
        for path in self.LOOPBACK_PROBE_PATHS[:-1]:
            response = self._http_probe("GET", base_url + path)
            if response is None:
                continue
            payload = response.get("json_payload")
            if isinstance(payload, dict):
                structured = self._structured_payload_source(
                    payload,
                    base_url + path,
                    endpoint_url=base_url + path,
                )
                if structured is not None:
                    structured["fingerprint"] = self._fingerprint_payload(response)
                    return structured
        for path in self.LOOPBACK_GRAPHQL_PATHS:
            response = self._http_probe(
                "POST",
                base_url + path,
                json_body={"query": self.GRAPHQL_INTROSPECTION_QUERY},
            )
            if response is None:
                continue
            payload = response.get("json_payload")
            if isinstance(payload, dict):
                structured = self._structured_payload_source(
                    payload,
                    base_url + path,
                    endpoint_url=base_url + path,
                )
                if structured is not None:
                    structured["fingerprint"] = self._fingerprint_payload(response)
                    return structured
            if self._looks_like_graphql_surface(response):
                return {
                    "kind": "graphql_endpoint",
                    "endpoint": base_url + path,
                    "excerpt": f"POST {base_url + path} GraphQL endpoint",
                    "step_specs": [
                        {
                            "method": "POST",
                            "url": base_url + path,
                            "source": "graphql_endpoint",
                            "operation_kind": "query",
                            "json_example": {
                                "query": "query AgentOSProbe { __typename }"
                            },
                            "response_example": {"data": {"__typename": "Query"}},
                        }
                    ],
                    "confidence_bonus": 0.12,
                    "fingerprint": self._fingerprint_payload(response),
                }
        response = self._http_probe("GET", base_url + "/")
        if response is not None and self._looks_like_api_surface(response):
            return {
                "kind": "loopback_endpoint",
                "endpoint": base_url,
                "excerpt": self._response_excerpt(response),
                "confidence_bonus": 0.1,
                "fingerprint": self._fingerprint_payload(response),
            }
        return None

    def _http_probe(
        self,
        method: str,
        url: str,
        json_body: Any | None = None,
    ) -> dict[str, Any] | None:
        headers = {
            "User-Agent": "AgentOS/loopback-fingerprint",
            "Accept": "application/json, text/plain, text/html;q=0.8, */*;q=0.2",
        }
        data = None
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.LOOPBACK_PROBE_TIMEOUT_SECONDS,
            ) as response:
                raw = response.read(16000)
                status = int(response.status)
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read(16000)
            status = int(exc.code)
            content_type = exc.headers.get("Content-Type", "")
        except Exception:
            return None
        body_text = raw.decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": status,
            "content_type": content_type,
            "body_preview": body_text[:1000],
            "json_payload": self._try_parse_json_object(body_text),
        }

    @staticmethod
    def _fingerprint_payload(response: dict[str, Any]) -> dict[str, Any]:
        return {
            "url": response.get("url", ""),
            "status": response.get("status", 0),
            "content_type": response.get("content_type", ""),
            "body_preview": str(response.get("body_preview", ""))[:400],
        }

    @staticmethod
    def _response_excerpt(response: dict[str, Any]) -> str:
        return (
            f"GET {response.get('url', '')} status={response.get('status', 0)} "
            f"type={response.get('content_type', '')} {str(response.get('body_preview', ''))[:240]}"
        )

    @staticmethod
    def _looks_like_api_surface(response: dict[str, Any]) -> bool:
        lower = " ".join(
            [
                str(response.get("url", "")),
                str(response.get("content_type", "")),
                str(response.get("body_preview", "")),
            ]
        ).lower()
        return any(
            token in lower for token in {"api", "swagger", "openapi", "json", "graphql"}
        )

    @staticmethod
    def _looks_like_graphql_surface(response: dict[str, Any]) -> bool:
        lower = " ".join(
            [
                str(response.get("url", "")),
                str(response.get("content_type", "")),
                str(response.get("body_preview", "")),
            ]
        ).lower()
        return "graphql" in lower or "graphiql" in lower

    def _resolve_parameter(
        self,
        parameter: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(parameter, dict):
            return None
        if "$ref" in parameter:
            resolved = self._resolve_ref(str(parameter.get("$ref") or ""), payload)
            if isinstance(resolved, dict):
                return resolved
        return parameter

    def _resolve_ref(
        self,
        ref: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not ref.startswith("#/"):
            return None
        current: Any = payload
        for part in ref[2:].split("/"):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        if isinstance(current, dict):
            return current
        return None

    def _apply_parameter_examples(
        self,
        url: str,
        parameters: list[Any],
        payload: dict[str, Any],
    ) -> str:
        if not parameters:
            return url
        parsed = urlparse(url)
        path = parsed.path
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        for parameter in parameters:
            resolved = self._resolve_parameter(parameter, payload)
            if not isinstance(resolved, dict):
                continue
            name = str(resolved.get("name") or "")
            location = str(resolved.get("in") or "").lower()
            if not name or location not in {"path", "query"}:
                continue
            example = self._parameter_example(resolved, payload)
            if example is None:
                continue
            if location == "path":
                path = path.replace("{" + name + "}", str(example))
            else:
                if isinstance(example, list):
                    for item in example:
                        query_pairs.append((name, str(item)))
                else:
                    query_pairs.append((name, str(example)))
        return urlunparse(
            parsed._replace(path=path, query=urlencode(query_pairs, doseq=True))
        )

    def _parameter_example(
        self,
        parameter: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        if "example" in parameter and parameter.get("example") not in (None, {}, []):
            return parameter.get("example")
        if "default" in parameter:
            return parameter.get("default")
        enum = parameter.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        content = parameter.get("content")
        if isinstance(content, dict):
            example = self._json_content_example(content, payload)
            if example not in (None, {}, []):
                return example
        schema = parameter.get("schema")
        if isinstance(schema, dict):
            example = self._schema_example(schema, payload)
            if example not in (None, {}, []):
                return example
        schema_type = str(parameter.get("type", "string")).lower()
        if schema_type == "integer":
            return 0
        if schema_type == "number":
            return 0.0
        if schema_type == "boolean":
            return False
        return f"sample_{parameter.get('name') or 'value'}"

    @staticmethod
    def _graphql_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("__schema"), dict):
            return data["__schema"]
        if isinstance(payload.get("__schema"), dict):
            return payload["__schema"]
        return None

    @staticmethod
    def _graphql_named_type(type_ref: Any) -> str:
        current = type_ref
        while isinstance(current, dict):
            name = current.get("name")
            if isinstance(name, str) and name:
                return name
            current = current.get("ofType")
        return ""

    def _graphql_operation_payload(
        self,
        field: dict[str, Any],
        operation_kind: str,
        type_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        field_name = str(field.get("name") or "")
        if not field_name:
            return {}
        variables: dict[str, Any] = {}
        variable_defs: list[str] = []
        arg_bindings: list[str] = []
        for argument in field.get("args", [])[:4]:
            if not isinstance(argument, dict):
                continue
            argument_name = str(argument.get("name") or "")
            type_ref = argument.get("type")
            if not argument_name or not isinstance(type_ref, dict):
                continue
            graphql_type = self._graphql_type_signature(type_ref)
            if not graphql_type:
                continue
            variable_defs.append(f"${argument_name}: {graphql_type}")
            arg_bindings.append(f"{argument_name}: ${argument_name}")
            variables[argument_name] = self._graphql_type_example(
                type_ref,
                type_map,
                hint=argument_name,
            )
        selection_set = self._graphql_selection_set(field.get("type"), type_map)
        signature = f"({', '.join(variable_defs)})" if variable_defs else ""
        arguments = f"({', '.join(arg_bindings)})" if arg_bindings else ""
        payload = {
            "query": (
                f"{operation_kind} AgentOS{operation_kind.title()}{signature} "
                f"{{ {field_name}{arguments}{selection_set} }}"
            )
        }
        if variables:
            payload["variables"] = variables
        return payload

    def _graphql_response_example(
        self,
        field: dict[str, Any],
        type_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        field_name = str(field.get("name") or "result")
        return {
            "data": {
                field_name: self._graphql_type_example(
                    field.get("type"),
                    type_map,
                    hint=field_name,
                )
            }
        }

    def _graphql_selection_set(
        self,
        type_ref: Any,
        type_map: dict[str, dict[str, Any]],
        depth: int = 0,
    ) -> str:
        if depth > 2:
            return ""
        type_name = self._graphql_named_type(type_ref)
        type_payload = type_map.get(type_name)
        if not isinstance(type_payload, dict):
            return ""
        fields = type_payload.get("fields")
        if not isinstance(fields, list):
            return ""
        scalar_fields: list[str] = []
        nested_fields: list[str] = []
        for field in fields[:5]:
            if not isinstance(field, dict):
                continue
            field_name = str(field.get("name") or "")
            if not field_name:
                continue
            nested = self._graphql_selection_set(field.get("type"), type_map, depth + 1)
            if nested:
                nested_fields.append(f"{field_name}{nested}")
            else:
                scalar_fields.append(field_name)
        selected = scalar_fields[:3] or nested_fields[:2]
        if not selected:
            return ""
        return " { " + " ".join(selected) + " }"

    def _graphql_type_signature(self, type_ref: Any) -> str:
        if not isinstance(type_ref, dict):
            return ""
        kind = str(type_ref.get("kind") or "")
        name = str(type_ref.get("name") or "")
        of_type = type_ref.get("ofType")
        if kind == "NON_NULL":
            nested = self._graphql_type_signature(of_type)
            return f"{nested}!" if nested else ""
        if kind == "LIST":
            nested = self._graphql_type_signature(of_type)
            return f"[{nested}]" if nested else "[]"
        return name

    def _graphql_type_example(
        self,
        type_ref: Any,
        type_map: dict[str, dict[str, Any]],
        hint: str = "value",
        depth: int = 0,
    ) -> Any:
        if depth > 3 or not isinstance(type_ref, dict):
            return None
        kind = str(type_ref.get("kind") or "")
        name = str(type_ref.get("name") or "")
        of_type = type_ref.get("ofType")
        if kind == "NON_NULL":
            return self._graphql_type_example(
                of_type, type_map, hint=hint, depth=depth + 1
            )
        if kind == "LIST":
            item = self._graphql_type_example(
                of_type, type_map, hint=hint, depth=depth + 1
            )
            return [item] if item is not None else []
        if kind in {"SCALAR", "ENUM"}:
            scalar_name = name.upper()
            if scalar_name in {"INT", "LONG"}:
                return 0
            if scalar_name in {"FLOAT", "DECIMAL"}:
                return 0.0
            if scalar_name in {"BOOLEAN", "BOOL"}:
                return False
            if scalar_name in {"ID"}:
                return f"<{hint}_id>"
            return f"<{hint}>"
        type_payload = type_map.get(name)
        if not isinstance(type_payload, dict):
            return f"<{hint}>"
        input_fields = type_payload.get("inputFields")
        if isinstance(input_fields, list):
            materialized: dict[str, Any] = {}
            for item in input_fields[:6]:
                if not isinstance(item, dict):
                    continue
                item_name = str(item.get("name") or "")
                if not item_name:
                    continue
                materialized[item_name] = self._graphql_type_example(
                    item.get("type"),
                    type_map,
                    hint=item_name,
                    depth=depth + 1,
                )
            return materialized
        fields = type_payload.get("fields")
        if isinstance(fields, list):
            materialized = {}
            for item in fields[:4]:
                if not isinstance(item, dict):
                    continue
                item_name = str(item.get("name") or "")
                if not item_name:
                    continue
                materialized[item_name] = self._graphql_type_example(
                    item.get("type"),
                    type_map,
                    hint=item_name,
                    depth=depth + 1,
                )
            return materialized
        return f"<{hint}>"

    def _documentation_sources(
        self, documentation_context: str
    ) -> list[dict[str, str]]:
        pattern = re.compile(
            r"\[(?P<index>\d+)\]\s+(?P<title>.*?)\nURL:\s+(?P<url>.*?)\nOfficial score:\s+.*?\nExcerpt:\s+(?P<excerpt>.*?)(?=\n\n\[\d+\]\s+|\Z)",
            flags=re.S,
        )
        sources: list[dict[str, str]] = []
        for match in pattern.finditer(documentation_context):
            sources.append(
                {
                    "title": match.group("title").strip(),
                    "url": match.group("url").strip(),
                    "excerpt": match.group("excerpt").strip(),
                }
            )
        if sources:
            return sources
        return [{"title": "", "url": "", "excerpt": documentation_context.strip()}]

    def _documented_step_specs(
        self,
        source_url: str,
        excerpt: str,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        base_url = self._base_url(source_url)
        if not base_url:
            endpoint_match = self.ENDPOINT_RE.search(excerpt)
            if endpoint_match is not None:
                base_url = self._base_url(endpoint_match.group(0))
        for method, raw_path in self.METHOD_PATH_RE.findall(excerpt):
            specs.append(
                {
                    "method": method.upper(),
                    "url": self._normalize_url(raw_path, base_url),
                    "source": "method_path",
                }
            )
        if specs:
            return specs
        for path in self.RELATIVE_API_PATH_RE.findall(excerpt):
            method = self._primary_method_from_excerpt(excerpt)
            specs.append(
                {
                    "method": method,
                    "url": self._normalize_url(path, base_url),
                    "source": "relative_path",
                }
            )
        return specs

    def _build_workflow(
        self,
        step_specs: list[dict[str, Any]],
        objective: str,
        excerpt: str,
    ) -> list[dict[str, Any]]:
        if not step_specs:
            return []
        objective_lower = objective.lower()
        primary = dict(self._primary_step_spec(step_specs, objective_lower))
        workflow: list[dict[str, Any]] = []
        is_graphql = str(primary.get("source", "")).startswith("graphql")

        workflow.append(self._probe_step(primary, is_graphql))

        primary_method = primary["method"]
        if (
            is_graphql
            or primary_method in {"POST", "PUT", "PATCH", "DELETE"}
            or any(
                keyword in objective_lower
                for keyword in self.OBJECTIVE_MUTATION_KEYWORDS
            )
        ):
            payload: Any = None
            if primary.get("json_example") not in (None, {}, []):
                payload = self._materialize_example(primary["json_example"], objective)
            if payload in (None, {}, []):
                payload = self._payload_template(excerpt, objective)
            workflow.append(
                {
                    "name": "execute_objective",
                    "method": primary_method,
                    "url": primary["url"],
                    "json_body": payload or None,
                    "expected_response": primary.get("response_example"),
                }
            )
        else:
            workflow.append(
                {
                    "name": "execute_objective",
                    "method": primary_method,
                    "url": primary["url"],
                    "json_body": None,
                    "expected_response": primary.get("response_example"),
                }
            )

        workflow.append(self._verification_step(step_specs, primary, objective))
        return workflow

    def _primary_step_spec(
        self,
        step_specs: list[dict[str, Any]],
        objective_lower: str,
    ) -> dict[str, Any]:
        if any(
            keyword in objective_lower for keyword in self.OBJECTIVE_MUTATION_KEYWORDS
        ):
            for spec in step_specs:
                if spec.get("operation_kind") == "mutation":
                    return spec
                if spec.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
                    return spec
        return step_specs[0]

    def _probe_step(
        self,
        primary: dict[str, Any],
        is_graphql: bool,
    ) -> dict[str, Any]:
        if is_graphql:
            return {
                "name": "probe_surface",
                "method": "POST",
                "url": primary["url"],
                "json_body": {"query": "query AgentOSProbe { __typename }"},
            }
        return {
            "name": "probe_surface",
            "method": "OPTIONS",
            "url": primary["url"],
            "json_body": None,
        }

    def _verification_step(
        self,
        step_specs: list[dict[str, Any]],
        primary: dict[str, Any],
        objective: str,
    ) -> dict[str, Any]:
        if str(primary.get("source", "")).startswith("graphql"):
            for spec in step_specs:
                if spec.get("operation_kind") == "query":
                    return {
                        "name": "verify_result",
                        "method": "POST",
                        "url": spec["url"],
                        "json_body": self._materialize_example(
                            spec.get("json_example")
                            or {"query": "query AgentOSProbe { __typename }"},
                            objective,
                        ),
                        "expected_response": spec.get("response_example"),
                    }
            return {
                "name": "verify_result",
                "method": "POST",
                "url": primary["url"],
                "json_body": {"query": "query AgentOSProbe { __typename }"},
                "expected_response": primary.get("response_example"),
            }
        verify_target = self._verification_target(step_specs, primary["url"])
        return {
            "name": "verify_result",
            "method": "GET",
            "url": verify_target,
            "json_body": None,
            "expected_response": self._response_hint(step_specs),
        }

    def _workflow_confidence(
        self,
        workflow: list[dict[str, Any]],
        excerpt: str,
        profile: CapabilityProfile,
        preferred_channels: list[str],
    ) -> float:
        confidence = 0.56
        if len(workflow) >= 3:
            confidence += 0.1
        if any(step.get("json_body") for step in workflow):
            confidence += 0.09
        if any(env for env in self._auth_env_keys(excerpt)):
            confidence += 0.05
        if "api" in preferred_channels:
            confidence += 0.06
        if "api" in profile.control_channels:
            confidence += 0.05
        if profile.app_family in {"browser", "terminal", "electron_app", "unknown"}:
            confidence += 0.04
        return min(0.95, confidence)

    @staticmethod
    def _base_url(source_url: str) -> str:
        parsed = urlparse(source_url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _normalize_url(path_or_url: str, base_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if path_or_url.startswith("/") and base_url:
            return base_url + path_or_url
        return path_or_url

    def _primary_method_from_excerpt(self, excerpt: str) -> str:
        lower = excerpt.lower()
        if any(keyword in lower for keyword in {"post", "create", "submit"}):
            return "POST"
        if any(keyword in lower for keyword in {"patch", "update", "edit"}):
            return "PATCH"
        if any(keyword in lower for keyword in {"put", "replace"}):
            return "PUT"
        if any(keyword in lower for keyword in {"delete", "remove"}):
            return "DELETE"
        return "GET"

    def _payload_template(self, excerpt: str, objective: str) -> dict[str, Any]:
        example = self._json_example(excerpt)
        if example:
            return self._materialize_example(example, objective)
        objective_lower = objective.lower()
        payload: dict[str, Any] = {}
        if any(keyword in objective_lower for keyword in self.OBJECTIVE_QUERY_KEYWORDS):
            payload["query"] = objective.strip()[:160]
        if any(
            keyword in objective_lower
            for keyword in {"message", "reply", "comment", "write"}
        ):
            payload["text"] = objective.strip()[:160]
        if any(keyword in objective_lower for keyword in {"create", "new", "add"}):
            payload["name"] = objective.strip()[:120]
        return payload

    def _json_example(self, excerpt: str) -> dict[str, Any]:
        for candidate in self._extract_json_objects(excerpt):
            if isinstance(candidate, dict) and candidate:
                return candidate
        return {}

    def _extract_json_objects(self, text: str) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        start = None
        depth = 0
        in_string = False
        escape = False
        for index, char in enumerate(text):
            if char == '"' and not escape:
                in_string = not in_string
            if in_string:
                escape = char == "\\" and not escape
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    raw = text[start : index + 1]
                    if len(raw) <= 400:
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            objects.append(parsed)
                    start = None
            escape = char == "\\" and not escape
        return objects[:3]

    def _materialize_example(self, example: Any, objective: str) -> Any:
        if isinstance(example, dict):
            return {
                key: self._materialize_example(value, objective)
                for key, value in example.items()
            }
        if isinstance(example, list):
            return [self._materialize_example(value, objective) for value in example]
        if isinstance(example, str):
            lower = example.strip().lower()
            if (
                example.startswith("<")
                or example.startswith("{")
                or lower
                in {
                    "example",
                    "placeholder",
                    "query",
                    "text",
                    "message",
                    "<query>",
                    "<text>",
                    "<message>",
                }
                or lower.endswith(" placeholder")
            ):
                return objective.strip()[:160]
            return example
        return example

    @staticmethod
    def _response_hint(step_specs: list[dict[str, Any]]) -> Any:
        for spec in step_specs:
            response_example = spec.get("response_example")
            if response_example not in (None, {}, []):
                return response_example
        return {}

    @staticmethod
    def _verification_target(
        step_specs: list[dict[str, Any]],
        primary_url: str,
    ) -> str:
        for spec in step_specs[1:]:
            if spec.get("method") == "GET":
                return str(spec.get("url") or primary_url)
        return primary_url

    @staticmethod
    def _auth_env_keys(excerpt: str) -> list[str]:
        lower = excerpt.lower()
        keys: list[str] = []
        if "bearer" in lower or "access token" in lower or "oauth" in lower:
            keys.extend(["ACCESS_TOKEN", "API_TOKEN", "BEARER_TOKEN"])
        if "api key" in lower or "x-api-key" in lower:
            keys.extend(["API_KEY", "X_API_KEY"])
        if "authorization" in lower and "bearer" not in lower:
            keys.append("AUTHORIZATION")
        return list(dict.fromkeys(keys))
