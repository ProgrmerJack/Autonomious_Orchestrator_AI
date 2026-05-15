from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .base import UiAction


def perform_real_file_operation(action: UiAction) -> dict[str, Any]:
    operation = _operation_name(action)
    metadata = dict(action.metadata or {})
    source = str(metadata.get("source") or "").strip()
    destination = str(metadata.get("destination") or "").strip()
    new_name = str(metadata.get("new_name") or "").strip()

    if operation in {"copy", "move"} and (not source or not destination):
        source, destination = _parse_arrow_value(str(action.value or ""))
    if operation == "rename" and (not source or not new_name):
        source, new_name = _parse_arrow_value(str(action.value or ""))

    if not source:
        raise ValueError(f"{operation}_file requires a source path")

    source_path = _resolve_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if operation == "copy":
        if not destination:
            raise ValueError("copy_file requires a destination path")
        destination_path = _resolve_path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)
        resulting_path = destination_path
    elif operation == "move":
        if not destination:
            raise ValueError("move_file requires a destination path")
        destination_path = _resolve_path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
        resulting_path = destination_path
    elif operation == "rename":
        if not new_name:
            raise ValueError("rename_file requires a new_name payload")
        resulting_path = _rename_target(source_path, new_name)
        resulting_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(resulting_path)
    else:
        raise ValueError(f"Unsupported real file operation: {action.action_type}")

    return {
        "status": "file-op-executed",
        "action_type": action.action_type,
        "selector": action.selector,
        "file_op": {
            "operation": operation,
            "source": source,
            "destination": destination if operation in {"copy", "move"} else None,
            "new_name": new_name if operation == "rename" else None,
            "resulting_path": str(resulting_path),
            "result_exists": resulting_path.exists(),
        },
    }


def _operation_name(action: UiAction) -> str:
    return str(action.action_type or "").removesuffix("_file")


def _parse_arrow_value(value: str) -> tuple[str, str]:
    if "->" not in value:
        return "", ""
    source, target = value.split("->", 1)
    return source.strip(), target.strip()


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (Path.cwd() / candidate).resolve(strict=False)


def _rename_target(source_path: Path, new_name: str) -> Path:
    target_candidate = Path(new_name).expanduser()
    if target_candidate.is_absolute():
        return target_candidate.resolve(strict=False)
    if target_candidate.parent != Path("."):
        return (Path.cwd() / target_candidate).resolve(strict=False)
    return source_path.with_name(new_name)
