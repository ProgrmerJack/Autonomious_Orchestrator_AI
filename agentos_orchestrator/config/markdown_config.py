from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MarkdownAgentConfig:
    soul: str
    agents: str
    heartbeat: dict[str, str]
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, root: str | Path = ".") -> "MarkdownAgentConfig":
        base = Path(root)
        return cls(
            soul=cls._read(base / "SOUL.md"),
            agents=cls._read(base / "AGENTS.md"),
            heartbeat=cls._parse_heartbeat(base / "HEARTBEAT.md"),
            root=base,
        )

    def heartbeat_enabled(self) -> bool:
        value = self.heartbeat.get("enabled", "false").lower()
        return value in {"1", "true", "yes", "on"}

    def heartbeat_interval(self) -> int:
        return int(self.heartbeat.get("interval_seconds", "300"))

    @staticmethod
    def _read(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    @classmethod
    def _parse_heartbeat(cls, path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in cls._read(path).splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                values[key] = value.strip()
        return values
