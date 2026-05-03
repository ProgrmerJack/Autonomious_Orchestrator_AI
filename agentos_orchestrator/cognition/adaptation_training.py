"""Unknown-app adaptation training from external GUI datasets and local traces."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image

from agentos_orchestrator.os_control.base import UiAction

from .learned_world_model import LearnedGenerativeWorldModel
from .local_vla import LocalFastVLA
from .trajectory_training import TrajectoryTrainingBuilder


ROWS_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={dataset}&config={config}&split={split}"
    "&offset={offset}&length={length}"
)
SCREENSPOT_ROWS_URL = ROWS_API_URL.format(
    dataset="rootsautomation/ScreenSpot",
    config="default",
    split="test",
    offset="{offset}",
    length="{length}",
)
CLICK100K_ROWS_URL = ROWS_API_URL.format(
    dataset="mlfoundations/Click-100k",
    config="default",
    split="train",
    offset="{offset}",
    length="{length}",
)
OSWORLD_DATASET_REPO = "xlangai/ubuntu_osworld_verified_trajs"
GUI_ACTOR_DATASET_REPO = "cckevinn/GUI-Actor-Data"
OSWORLD_MANIFEST_URL = (
    f"https://huggingface.co/datasets/{OSWORLD_DATASET_REPO}"
    "/resolve/main/all_result.json"
)
OSWORLD_TREE_URL = (
    f"https://huggingface.co/api/datasets/{OSWORLD_DATASET_REPO}/tree/main?recursive=1"
)
GUI_ACTOR_TREE_URL = (
    f"https://huggingface.co/api/datasets/{GUI_ACTOR_DATASET_REPO}"
    "/tree/main?recursive=1"
)
DATASET_TREE_URLS: dict[str, str] = {
    OSWORLD_DATASET_REPO: OSWORLD_TREE_URL,
    GUI_ACTOR_DATASET_REPO: GUI_ACTOR_TREE_URL,
}
HF_DATASET_RESOLVE_URL = (
    "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}?download=true"
)
DATASET_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "name": "OS-Atlas-data",
        "repo_id": "OS-Copilot/OS-Atlas-data",
        "mode": "archive-backed annotations",
        "scale": "more than 270k webpage screenshots, over 3M webpage elements, and at least 1.6M FineWeb images",
    },
    {
        "name": "GUI-Actor-Data",
        "repo_id": GUI_ACTOR_DATASET_REPO,
        "mode": "archive-backed grounding",
        "scale": "approximately 1M screenshots and 10M elements",
    },
    {
        "name": "Click-100k",
        "repo_id": "mlfoundations/Click-100k",
        "mode": "rows-based grounding",
        "scale": "100K<n<1M rows",
    },
    {
        "name": "OSWorld verified trajectories",
        "repo_id": OSWORLD_DATASET_REPO,
        "mode": "archive-backed transitions",
        "scale": "100K<n<1M verified archive trajectories",
    },
)


@dataclass(slots=True)
class AdaptationTrainingConfig:
    run_id: str = ""
    screenspot_limit: int = 64
    screenspot_offset: int = 0
    click100k_limit: int = 0
    click100k_offset: int = 0
    gui_actor_limit: int = 0
    gui_actor_offset: int = 0
    gui_actor_sources: tuple[str, ...] = ()
    screenspot_sources: tuple[str, ...] = ("windows", "web", "macos")
    include_internal_trajectories: bool = True
    trajectory_paths: tuple[str, ...] = ()
    download_osworld_manifest: bool = True
    osworld_archive_limit: int = 0
    osworld_archive_transition_limit: int = 0
    osworld_archive_paths: tuple[str, ...] = ()
    cache_dir: str = ""
    cache_budget_bytes: int = 0
    stage_remote_archives: bool = False
    output_dir: str = ""


@dataclass(slots=True)
class AdaptationLongRunConfig:
    run_id: str = ""
    shard_count: int = 1
    screenspot_limit_per_shard: int = 0
    click100k_limit_per_shard: int = 0
    gui_actor_limit_per_shard: int = 0
    screenspot_sources: tuple[str, ...] = ("windows", "web", "macos")
    include_internal_trajectories_first_shard: bool = True
    trajectory_paths: tuple[str, ...] = ()
    download_osworld_manifest: bool = True
    osworld_archives_per_shard: int = 0
    osworld_archive_transition_limit_per_shard: int = 0
    osworld_archive_candidate_multiplier: int = 0
    cache_dir: str = ""
    cache_budget_bytes: int = 0
    stage_remote_archives: bool = False
    output_dir: str = ""
    state_path: str = ""
    resume: bool = True


@dataclass(slots=True)
class AdaptationTrainingResult:
    run_id: str
    success: bool
    screenspot_rows_used: int
    click100k_rows_used: int
    gui_actor_rows_used: int
    world_model_transitions: int
    osworld_archive_transitions: int
    osworld_archives_parsed: int
    osworld_domains: list[str]
    output_dir: str
    local_vla_model_path: str = ""
    world_model_checkpoint: str = ""
    screenspot_rows_path: str = ""
    click100k_rows_path: str = ""
    gui_actor_rows_path: str = ""
    internal_dataset_path: str = ""
    osworld_manifest_path: str = ""
    osworld_archive_dataset_path: str = ""
    dataset_inventory_path: str = ""
    dataset_summaries: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    requested: dict[str, int] = field(default_factory=dict)
    underfill: dict[str, Any] = field(default_factory=dict)
    scale_report: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptationLongRunResult:
    run_id: str
    success: bool
    output_dir: str
    state_path: str
    cache_dir: str
    shards_completed: int
    total_shards: int
    screenspot_rows_used: int
    click100k_rows_used: int
    gui_actor_rows_used: int
    world_model_transitions: int
    osworld_archive_transitions: int
    osworld_archives_parsed: int
    shard_results: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    requested: dict[str, int] = field(default_factory=dict)
    underfill: dict[str, Any] = field(default_factory=dict)
    scale_report: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class UnknownAppAdaptationTrainer:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.models_root = self.workspace_root / ".agentos"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._dataset_tree_cache: dict[str, list[dict[str, Any]]] = {}
        self._dataset_tree_index: dict[str, dict[str, dict[str, Any]]] = {}
        self.local_vla = LocalFastVLA(
            model_path=str(self.models_root / "local_vla.pkl")
        )
        self.world_model = LearnedGenerativeWorldModel(
            checkpoint_path=self.models_root / "world_model.pkl"
        )
        self.trajectory_builder = TrajectoryTrainingBuilder(self.workspace_root)

    def train(
        self,
        config: AdaptationTrainingConfig | None = None,
    ) -> AdaptationTrainingResult:
        config = config or AdaptationTrainingConfig()
        run_id = config.run_id or f"adapt_{int(time.time())}"
        output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else (self.models_root / "adaptation" / run_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root = self._prepare_cache_root(
            config.cache_dir,
            stage_remote_archives=config.stage_remote_archives,
        )

        dataset_summaries: list[dict[str, Any]] = []
        total_vla_feedback_samples = 0

        screenspot_rows_path = output_dir / "screenspot_rows.jsonl"
        screenspot_rows_used = 0
        if config.screenspot_limit > 0:
            screenspot_rows_used = self._train_vla_from_screenspot(
                limit=config.screenspot_limit,
                start_offset=config.screenspot_offset,
                sources=set(source.lower() for source in config.screenspot_sources),
                output_path=screenspot_rows_path,
            )
            total_vla_feedback_samples += screenspot_rows_used
            dataset_summaries.append(
                {
                    "name": "ScreenSpot",
                    "mode": "rows-based grounding",
                    "records_requested": config.screenspot_limit,
                    "records_used": screenspot_rows_used,
                    "path": str(screenspot_rows_path),
                }
            )

        click100k_rows_path = output_dir / "click100k_rows.jsonl"
        click100k_rows_used = 0
        if config.click100k_limit > 0:
            click100k_rows_used = self._train_vla_from_click100k(
                limit=config.click100k_limit,
                start_offset=config.click100k_offset,
                output_path=click100k_rows_path,
            )
            total_vla_feedback_samples += click100k_rows_used
            dataset_summaries.append(
                {
                    "name": "Click-100k",
                    "mode": "rows-based grounding",
                    "records_requested": config.click100k_limit,
                    "records_used": click100k_rows_used,
                    "path": str(click100k_rows_path),
                }
            )

        gui_actor_rows_path = output_dir / "gui_actor_rows.jsonl"
        gui_actor_rows_used = 0
        if config.gui_actor_limit > 0:
            gui_actor_rows_used = self._train_vla_from_gui_actor(
                limit=config.gui_actor_limit,
                source_offset=config.gui_actor_offset,
                source_names=set(config.gui_actor_sources)
                if config.gui_actor_sources
                else None,
                cache_root=cache_root,
                cache_budget_bytes=config.cache_budget_bytes,
                stage_remote_archives=config.stage_remote_archives,
                output_path=gui_actor_rows_path,
            )
            total_vla_feedback_samples += gui_actor_rows_used
            dataset_summaries.append(
                {
                    "name": "GUI-Actor-Data",
                    "mode": "archive-backed grounding",
                    "records_requested": config.gui_actor_limit,
                    "records_used": gui_actor_rows_used,
                    "path": str(gui_actor_rows_path),
                }
            )

        world_model_transitions = 0
        internal_dataset_path = output_dir / "trajectory_training.jsonl"
        if config.include_internal_trajectories:
            internal_transitions = self._train_world_model_from_trajectories(
                paths=list(config.trajectory_paths) or None,
            )
            world_model_transitions += internal_transitions
            dataset_summary = self.trajectory_builder.write_dataset(
                output_path=internal_dataset_path,
                paths=list(config.trajectory_paths) or None,
            )
            internal_dataset_path = Path(dataset_summary["path"])
            dataset_summaries.append(
                {
                    "name": "internal_trajectories",
                    "mode": "local replay",
                    "records_used": internal_transitions,
                    "path": str(internal_dataset_path),
                }
            )

        osworld_archive_transitions = 0
        osworld_archives_parsed = 0
        osworld_archive_dataset_path = output_dir / "osworld_archive_transitions.jsonl"
        if (
            config.osworld_archive_limit > 0
            or config.osworld_archive_transition_limit > 0
        ):
            archive_summary = self._train_world_model_from_osworld_archives(
                archive_limit=config.osworld_archive_limit,
                transition_limit=config.osworld_archive_transition_limit,
                archive_paths=list(config.osworld_archive_paths) or None,
                cache_root=cache_root,
                cache_budget_bytes=config.cache_budget_bytes,
                stage_remote_archives=config.stage_remote_archives,
                output_path=osworld_archive_dataset_path,
            )
            osworld_archive_transitions = int(archive_summary["transitions"])
            osworld_archives_parsed = int(archive_summary["archives_parsed"])
            world_model_transitions += osworld_archive_transitions
            dataset_summaries.append(
                {
                    "name": "OSWorld verified trajectories",
                    "mode": "remote archive replay",
                    "records_requested": config.osworld_archive_transition_limit,
                    "records_used": osworld_archive_transitions,
                    "archives_requested": config.osworld_archive_limit,
                    "archives_parsed": osworld_archives_parsed,
                    "path": str(osworld_archive_dataset_path),
                }
            )

        osworld_domains: list[str] = []
        osworld_manifest_path = output_dir / "osworld_verified_manifest.json"
        if config.download_osworld_manifest:
            manifest = self._fetch_json(OSWORLD_MANIFEST_URL)
            osworld_manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            if isinstance(manifest, dict):
                osworld_domains = sorted(str(key) for key in manifest.keys())

        dataset_inventory_path = output_dir / "dataset_inventory.json"
        dataset_inventory_path.write_text(
            json.dumps(list(DATASET_INVENTORY), indent=2),
            encoding="utf-8",
        )

        if total_vla_feedback_samples >= 10:
            self.local_vla.fit_feedback_classifier(min_samples=10)
        if world_model_transitions > 0:
            self.world_model.finalize_training()

        local_vla_model_path = self.local_vla.save_model(
            self.models_root / "local_vla.pkl"
        )
        world_model_checkpoint = self.world_model.save_checkpoint()

        requested = {
            "screenspot_rows": max(0, config.screenspot_limit),
            "click100k_rows": max(0, config.click100k_limit),
            "gui_actor_rows": max(0, config.gui_actor_limit),
            "osworld_archive_transitions": max(
                0,
                config.osworld_archive_transition_limit,
            ),
            "osworld_archives": max(0, config.osworld_archive_limit),
        }
        actual = {
            "screenspot_rows": screenspot_rows_used,
            "click100k_rows": click100k_rows_used,
            "gui_actor_rows": gui_actor_rows_used,
            "osworld_archive_transitions": osworld_archive_transitions,
            "osworld_archives": osworld_archives_parsed,
        }
        underfill = _underfill_report(requested, actual)
        scale_report = _scale_report(
            grounding_examples=total_vla_feedback_samples,
            world_model_transitions=world_model_transitions,
            app_families=osworld_domains,
        )

        notes: list[str] = []
        if screenspot_rows_used == 0:
            notes.append("No ScreenSpot rows were ingested.")
        if click100k_rows_used == 0:
            notes.append("No Click-100k rows were ingested.")
        if gui_actor_rows_used == 0:
            notes.append("No GUI-Actor rows were ingested.")
        if world_model_transitions == 0:
            notes.append(
                "No trajectory transitions were replayed into the world model."
            )
        if osworld_domains:
            notes.append(
                "OSWorld manifest coverage includes: " + ", ".join(osworld_domains[:8])
            )
        notes.append(
            "Large-source inventory includes OS-Atlas-data, GUI-Actor-Data, Click-100k, and OSWorld verified trajectories."
        )
        if cache_root is not None:
            notes.append(f"Dataset cache root: {cache_root}")
        notes.extend(_underfill_notes(underfill))
        if not scale_report["meets_minimum_scale"]:
            notes.append(
                "Scale target not yet met: continue resumable long-run training "
                "toward 100K+ grounding examples and trajectory transitions."
            )
        elif not scale_report["meets_production_scale"]:
            notes.append(
                "Bootstrap scale is met; continue resumable UI long-run training "
                "toward the 10M+ production target."
            )

        result = AdaptationTrainingResult(
            run_id=run_id,
            success=bool(
                screenspot_rows_used
                or click100k_rows_used
                or gui_actor_rows_used
                or world_model_transitions
            ),
            screenspot_rows_used=screenspot_rows_used,
            click100k_rows_used=click100k_rows_used,
            gui_actor_rows_used=gui_actor_rows_used,
            world_model_transitions=world_model_transitions,
            osworld_archive_transitions=osworld_archive_transitions,
            osworld_archives_parsed=osworld_archives_parsed,
            osworld_domains=osworld_domains,
            output_dir=str(output_dir),
            local_vla_model_path=str(local_vla_model_path),
            world_model_checkpoint=str(world_model_checkpoint or ""),
            screenspot_rows_path=str(screenspot_rows_path)
            if screenspot_rows_path.exists()
            else "",
            click100k_rows_path=str(click100k_rows_path)
            if click100k_rows_path.exists()
            else "",
            gui_actor_rows_path=str(gui_actor_rows_path)
            if gui_actor_rows_path.exists()
            else "",
            internal_dataset_path=str(internal_dataset_path)
            if internal_dataset_path.exists()
            else "",
            osworld_manifest_path=str(osworld_manifest_path)
            if osworld_manifest_path.exists()
            else "",
            osworld_archive_dataset_path=(
                str(osworld_archive_dataset_path)
                if osworld_archive_dataset_path.exists()
                else ""
            ),
            dataset_inventory_path=str(dataset_inventory_path),
            dataset_summaries=dataset_summaries,
            notes=notes,
            requested=requested,
            underfill=underfill,
            scale_report=scale_report,
        )
        (output_dir / "training_result.json").write_text(
            json.dumps(result.asdict(), indent=2),
            encoding="utf-8",
        )
        return result

    def train_long_run(
        self,
        config: AdaptationLongRunConfig | None = None,
    ) -> AdaptationLongRunResult:
        config = config or AdaptationLongRunConfig()
        run_id = config.run_id or f"adapt_long_{int(time.time())}"
        output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else (self.models_root / "adaptation_longrun" / run_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root = self._prepare_cache_root(
            config.cache_dir,
            stage_remote_archives=config.stage_remote_archives,
        )
        state_path = (
            Path(config.state_path)
            if config.state_path
            else (output_dir / "long_run_state.json")
        )
        state = (
            self._load_long_run_state(state_path)
            if config.resume
            else {
                "run_id": run_id,
                "completed_shards": [],
                "shard_results": [],
            }
        )

        gui_sources = [source["name"] for source in self._gui_actor_sources()]
        gui_source_stride = max(
            _distributed_targets(config.gui_actor_limit_per_shard, len(gui_sources))
            or [0]
        )
        archive_pool_limit = 0
        if config.osworld_archives_per_shard > 0 and config.shard_count > 0:
            archive_pool_limit = (
                config.osworld_archives_per_shard
                * config.shard_count
                * self._osworld_candidate_multiplier(config)
            )
        osworld_archives = self._list_osworld_archives(limit=archive_pool_limit)
        osworld_batches = _osworld_shard_batches(
            osworld_archives,
            shard_count=max(0, config.shard_count),
            archives_per_shard=max(0, config.osworld_archives_per_shard),
            candidate_multiplier=self._osworld_candidate_multiplier(config),
        )

        totals = {
            "screenspot_rows_used": 0,
            "click100k_rows_used": 0,
            "gui_actor_rows_used": 0,
            "world_model_transitions": 0,
            "osworld_archive_transitions": 0,
            "osworld_archives_parsed": 0,
        }
        for item in list(state.get("shard_results") or []):
            totals["screenspot_rows_used"] += _safe_int(
                item.get("screenspot_rows_used")
            )
            totals["click100k_rows_used"] += _safe_int(item.get("click100k_rows_used"))
            totals["gui_actor_rows_used"] += _safe_int(item.get("gui_actor_rows_used"))
            totals["world_model_transitions"] += _safe_int(
                item.get("world_model_transitions")
            )
            totals["osworld_archive_transitions"] += _safe_int(
                item.get("osworld_archive_transitions")
            )
            totals["osworld_archives_parsed"] += _safe_int(
                item.get("osworld_archives_parsed")
            )

        completed_shards = {
            int(index) for index in list(state.get("completed_shards") or [])
        }
        for shard_index in range(max(0, config.shard_count)):
            if shard_index in completed_shards:
                continue
            shard_output_dir = output_dir / f"shard_{shard_index:04d}"
            shard_output_dir.mkdir(parents=True, exist_ok=True)
            osworld_batch: tuple[str, ...] = ()
            if shard_index < len(osworld_batches):
                osworld_batch = tuple(osworld_batches[shard_index])
            shard_result = self.train(
                AdaptationTrainingConfig(
                    run_id=f"{run_id}_shard_{shard_index:04d}",
                    screenspot_limit=config.screenspot_limit_per_shard,
                    screenspot_offset=shard_index
                    * max(0, config.screenspot_limit_per_shard),
                    click100k_limit=config.click100k_limit_per_shard,
                    click100k_offset=shard_index
                    * max(0, config.click100k_limit_per_shard),
                    gui_actor_limit=config.gui_actor_limit_per_shard,
                    gui_actor_offset=shard_index * gui_source_stride,
                    gui_actor_sources=tuple(gui_sources),
                    screenspot_sources=config.screenspot_sources,
                    include_internal_trajectories=(
                        config.include_internal_trajectories_first_shard
                        and shard_index == 0
                    ),
                    trajectory_paths=config.trajectory_paths,
                    download_osworld_manifest=(
                        config.download_osworld_manifest and shard_index == 0
                    ),
                    osworld_archive_limit=len(osworld_batch),
                    osworld_archive_transition_limit=(
                        config.osworld_archive_transition_limit_per_shard
                    ),
                    osworld_archive_paths=osworld_batch,
                    cache_dir=str(cache_root) if cache_root is not None else "",
                    cache_budget_bytes=config.cache_budget_bytes,
                    stage_remote_archives=config.stage_remote_archives,
                    output_dir=str(shard_output_dir),
                )
            )
            shard_payload = shard_result.asdict()
            state.setdefault("completed_shards", []).append(shard_index)
            state.setdefault("shard_results", []).append(shard_payload)
            self._write_long_run_state(state_path, state)
            totals["screenspot_rows_used"] += shard_result.screenspot_rows_used
            totals["click100k_rows_used"] += shard_result.click100k_rows_used
            totals["gui_actor_rows_used"] += shard_result.gui_actor_rows_used
            totals["world_model_transitions"] += shard_result.world_model_transitions
            totals["osworld_archive_transitions"] += (
                shard_result.osworld_archive_transitions
            )
            totals["osworld_archives_parsed"] += shard_result.osworld_archives_parsed

        notes = [
            f"Cache root: {cache_root}"
            if cache_root is not None
            else "No local dataset cache configured.",
            f"Completed {len(list(state.get('completed_shards') or []))} of {max(0, config.shard_count)} shards.",
        ]
        if config.stage_remote_archives:
            notes.append(
                "Remote archive staging enabled for resumable long-run training."
            )
        if config.osworld_archives_per_shard > 0:
            notes.append(
                f"OSWorld batches planned from {len(osworld_archives)} candidate archives."
            )
            if self._osworld_candidate_multiplier(config) > 1:
                notes.append(
                    "OSWorld candidate overfetch enabled so failed or exhausted "
                    "archives can be replaced inside each shard budget."
                )
        if config.gui_actor_limit_per_shard > 0:
            notes.append(f"GUI-Actor source stride per shard: {gui_source_stride}.")
        requested = {
            "screenspot_rows": max(0, config.shard_count)
            * max(0, config.screenspot_limit_per_shard),
            "click100k_rows": max(0, config.shard_count)
            * max(0, config.click100k_limit_per_shard),
            "gui_actor_rows": max(0, config.shard_count)
            * max(0, config.gui_actor_limit_per_shard),
            "osworld_archives": max(0, config.shard_count)
            * max(0, config.osworld_archives_per_shard),
            "osworld_archive_transitions": max(0, config.shard_count)
            * max(0, config.osworld_archive_transition_limit_per_shard),
        }
        actual = {
            "screenspot_rows": totals["screenspot_rows_used"],
            "click100k_rows": totals["click100k_rows_used"],
            "gui_actor_rows": totals["gui_actor_rows_used"],
            "osworld_archives": totals["osworld_archives_parsed"],
            "osworld_archive_transitions": totals["osworld_archive_transitions"],
        }
        underfill = _underfill_report(requested, actual)
        scale_report = _scale_report(
            grounding_examples=(
                totals["screenspot_rows_used"]
                + totals["click100k_rows_used"]
                + totals["gui_actor_rows_used"]
            ),
            world_model_transitions=totals["world_model_transitions"],
            app_families=[],
        )
        notes.extend(_underfill_notes(underfill))
        if not scale_report["meets_minimum_scale"]:
            notes.append(
                "Scale target not yet met: keep resuming with fresh cache batches "
                "until the 100K+ minimum is reached."
            )
        elif not scale_report["meets_production_scale"]:
            notes.append(
                "Minimum scale is met; keep resuming shard training until the "
                "10M+ production target is reached."
            )
        result = AdaptationLongRunResult(
            run_id=run_id,
            success=bool(
                totals["screenspot_rows_used"]
                or totals["click100k_rows_used"]
                or totals["gui_actor_rows_used"]
                or totals["world_model_transitions"]
            ),
            output_dir=str(output_dir),
            state_path=str(state_path),
            cache_dir=str(cache_root) if cache_root is not None else "",
            shards_completed=len(list(state.get("completed_shards") or [])),
            total_shards=max(0, config.shard_count),
            screenspot_rows_used=totals["screenspot_rows_used"],
            click100k_rows_used=totals["click100k_rows_used"],
            gui_actor_rows_used=totals["gui_actor_rows_used"],
            world_model_transitions=totals["world_model_transitions"],
            osworld_archive_transitions=totals["osworld_archive_transitions"],
            osworld_archives_parsed=totals["osworld_archives_parsed"],
            shard_results=list(state.get("shard_results") or []),
            notes=notes,
            requested=requested,
            underfill=underfill,
            scale_report=scale_report,
        )
        (output_dir / "long_run_result.json").write_text(
            json.dumps(result.asdict(), indent=2),
            encoding="utf-8",
        )
        return result

    def _osworld_candidate_multiplier(self, config: AdaptationLongRunConfig) -> int:
        if config.osworld_archives_per_shard <= 0:
            return 1
        if config.osworld_archive_candidate_multiplier > 0:
            return max(1, config.osworld_archive_candidate_multiplier)
        if config.cache_dir or config.stage_remote_archives:
            return 2
        return 1

    def _train_vla_from_screenspot(
        self,
        *,
        limit: int,
        start_offset: int,
        sources: set[str],
        output_path: Path,
    ) -> int:
        return self._train_vla_from_rows_dataset(
            url_pattern=SCREENSPOT_ROWS_URL,
            limit=limit,
            start_offset=start_offset,
            output_path=output_path,
            row_filter=lambda row: str(row.get("data_source") or "").lower() in sources,
            row_mapper=self._screenspot_training_sample,
        )

    def _train_vla_from_click100k(
        self,
        *,
        limit: int,
        start_offset: int,
        output_path: Path,
    ) -> int:
        return self._train_vla_from_rows_dataset(
            url_pattern=CLICK100K_ROWS_URL,
            limit=limit,
            start_offset=start_offset,
            output_path=output_path,
            row_filter=lambda row: True,
            row_mapper=self._click100k_training_sample,
        )

    def _train_vla_from_gui_actor(
        self,
        *,
        limit: int,
        source_offset: int,
        source_names: set[str] | None,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
        output_path: Path,
    ) -> int:
        used = 0
        lines: list[str] = []
        sources = self._gui_actor_sources(source_names=source_names)
        targets = _distributed_targets(limit, len(sources))
        carry_forward = 0
        for index, source in enumerate(sources):
            source_target = targets[index] if index < len(targets) else 0
            source_target += carry_forward
            carry_forward = 0
            if used >= limit:
                break
            if source_target <= 0:
                continue
            source_start = used
            try:
                with self._open_dataset_archive(
                    GUI_ACTOR_DATASET_REPO,
                    source["archive"],
                    cache_root=cache_root,
                    cache_budget_bytes=cache_budget_bytes,
                    stage_remote_archives=stage_remote_archives,
                ) as archive:
                    image_prefixes = self._archive_image_prefixes(archive)
                    remaining = min(limit - used, source_target)
                    scan_limit = max(remaining, remaining * 4)
                    for row in self._iter_json_array_source(
                        GUI_ACTOR_DATASET_REPO,
                        source["annotation"],
                        limit=scan_limit,
                        skip=source_offset,
                        cache_root=cache_root,
                        cache_budget_bytes=cache_budget_bytes,
                        stage_remote_archives=stage_remote_archives,
                    ):
                        sample = self._gui_actor_training_sample(
                            dict(row),
                            archive,
                            source["name"],
                            image_prefixes,
                        )
                        if sample is None:
                            continue
                        screenshot, x, y, label, payload = sample
                        self.local_vla.collect_feedback(screenshot, x, y, label)
                        lines.append(json.dumps(payload, sort_keys=True))
                        used += 1
                        if used >= limit or used - source_start >= remaining:
                            break
            except (OSError, ValueError, zipfile.BadZipFile):
                carry_forward += min(limit - used, source_target)
                continue
            source_used = used - source_start
            source_missing = min(limit - source_start, source_target) - source_used
            if source_missing > 0:
                carry_forward += source_missing
        if lines:
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return used

    def _train_vla_from_rows_dataset(
        self,
        *,
        url_pattern: str,
        limit: int,
        start_offset: int,
        output_path: Path,
        row_filter: Callable[[dict[str, Any]], bool],
        row_mapper: Callable[
            [dict[str, Any]],
            tuple[bytes, int, int, str, dict[str, Any]] | None,
        ],
    ) -> int:
        used = 0
        offset = max(0, start_offset)
        page_size = min(100, max(1, limit))
        lines: list[str] = []
        while used < limit:
            payload = self._fetch_json(
                url_pattern.format(offset=offset, length=page_size)
            )
            rows = list((payload or {}).get("rows") or [])
            if not rows:
                break
            for item in rows:
                row = dict(item.get("row") or {})
                if not row or not row_filter(row):
                    continue
                try:
                    sample = row_mapper(row)
                except (OSError, TimeoutError, urllib.error.URLError):
                    continue
                if sample is None:
                    continue
                screenshot, x, y, label, payload = sample
                self.local_vla.collect_feedback(screenshot, x, y, label)
                lines.append(json.dumps(payload, sort_keys=True))
                used += 1
                if used >= limit:
                    break
            offset += len(rows)
        if lines:
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return used

    def _screenspot_training_sample(
        self,
        row: dict[str, Any],
    ) -> tuple[bytes, int, int, str, dict[str, Any]] | None:
        image = dict(row.get("image") or {})
        image_src = str(image.get("src") or "")
        if not image_src:
            return None
        screenshot = self._fetch_bytes(image_src)
        x, y = self._center_from_bbox(
            row.get("bbox") or [],
            width=int(image.get("width") or 0),
            height=int(image.get("height") or 0),
            normalized=True,
        )
        label = self._label_from_screenspot_row(row)
        return (
            screenshot,
            x,
            y,
            label,
            {
                "dataset": "ScreenSpot",
                "instruction": row.get("instruction"),
                "data_type": row.get("data_type"),
                "data_source": row.get("data_source"),
                "label": label,
                "center": [x, y],
                "file_name": row.get("file_name"),
            },
        )

    def _click100k_training_sample(
        self,
        row: dict[str, Any],
    ) -> tuple[bytes, int, int, str, dict[str, Any]] | None:
        images = list(row.get("images") or [])
        if not images:
            return None
        image = dict(images[0] or {})
        image_src = str(image.get("src") or "")
        if not image_src:
            return None
        screenshot = self._fetch_bytes(image_src)
        x, y = self._center_from_bbox(
            row.get("bbox") or [],
            width=int(row.get("image_width") or image.get("width") or 0),
            height=int(row.get("image_height") or image.get("height") or 0),
            normalized=False,
        )
        instruction = self._click100k_instruction(row)
        label = self._label_from_instruction(instruction, default="button")
        return (
            screenshot,
            x,
            y,
            label,
            {
                "dataset": "Click-100k",
                "instruction": instruction,
                "label": label,
                "center": [x, y],
                "image_path": row.get("image_path"),
            },
        )

    def _gui_actor_training_sample(
        self,
        row: dict[str, Any],
        archive: zipfile.ZipFile,
        source_name: str,
        image_prefixes: list[str],
    ) -> tuple[bytes, int, int, str, dict[str, Any]] | None:
        image_name = str(row.get("image") or "")
        if not image_name:
            return None
        archive_member = self._resolve_archive_member_name(
            archive,
            image_name,
            image_prefixes,
        )
        if not archive_member:
            return None
        try:
            screenshot = archive.read(archive_member)
        except KeyError:
            return None
        instruction = self._gui_actor_instruction(row)
        bbox = self._gui_actor_bbox(row)
        if not instruction or len(bbox) != 4:
            return None
        with Image.open(io.BytesIO(screenshot)) as screenshot_image:
            width, height = screenshot_image.size
        x, y = self._center_from_bbox(
            bbox,
            width=width,
            height=height,
            normalized=True,
        )
        label = self._label_from_instruction(instruction, default="button")
        return (
            screenshot,
            x,
            y,
            label,
            {
                "dataset": "GUI-Actor-Data",
                "source": source_name,
                "instruction": instruction,
                "label": label,
                "center": [x, y],
                "image": image_name,
                "archive_member": archive_member,
            },
        )

    def _train_world_model_from_trajectories(
        self,
        paths: list[str] | None = None,
    ) -> int:
        transitions = 0
        trajectory_paths: list[str | Path] | None = (
            [Path(path) for path in paths] if paths else None
        )
        for event in self.trajectory_builder.iter_step_events(trajectory_paths):
            before = _safe_dict(event.get("before"))
            after = _safe_dict(event.get("after"))
            action = _action_from_payload(event.get("action"))
            if not before or not after or action is None:
                continue
            self.world_model.record_transition(before, action, after)
            transitions += 1
        return transitions

    def _train_world_model_from_osworld_archives(
        self,
        *,
        archive_limit: int,
        transition_limit: int,
        archive_paths: list[str] | None,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
        output_path: Path,
    ) -> dict[str, int]:
        transitions = 0
        archives_parsed = 0
        lines: list[str] = []
        selected_archive_paths = archive_paths or self._list_osworld_archives(
            limit=archive_limit
        )
        archive_targets = _distributed_targets(
            transition_limit, len(selected_archive_paths)
        )
        for archive_index, archive_path in enumerate(selected_archive_paths):
            archive_transitions = 0
            archive_target = 0
            if transition_limit > 0 and archive_index < len(archive_targets):
                archive_target = archive_targets[archive_index]
            if transition_limit > 0 and archive_target <= 0:
                continue
            for payload in self._iter_osworld_archive_payloads(
                archive_path,
                cache_root=cache_root,
                cache_budget_bytes=cache_budget_bytes,
                stage_remote_archives=stage_remote_archives,
            ):
                for before, action, after in self._iter_transition_candidates(payload):
                    self.world_model.record_transition(before, action, after)
                    lines.append(
                        json.dumps(
                            {
                                "archive": archive_path,
                                "before": before,
                                "action": asdict(action),
                                "after": after,
                            },
                            sort_keys=True,
                        )
                    )
                    transitions += 1
                    archive_transitions += 1
                    if transition_limit > 0 and archive_transitions >= archive_target:
                        break
                if transition_limit > 0 and archive_transitions >= archive_target:
                    break
            if archive_transitions > 0:
                archives_parsed += 1
            if transition_limit > 0 and transitions >= transition_limit:
                break
        if lines:
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"transitions": transitions, "archives_parsed": archives_parsed}

    def _list_osworld_archives(self, *, limit: int) -> list[str]:
        payload = self._dataset_tree(OSWORLD_DATASET_REPO)
        if not isinstance(payload, list):
            return []
        archives: list[tuple[int, str]] = []
        for item in payload:
            path = str(item.get("path") or "")
            if not path.endswith(".zip"):
                continue
            if "results_only" in path.lower():
                continue
            size = _safe_int(item.get("size"))
            archives.append((size if size > 0 else 1 << 62, path))
        archives.sort(key=lambda item: (item[0], item[1]))
        paths = [path for _, path in archives]
        if limit > 0:
            return paths[:limit]
        return paths

    def _iter_osworld_archive_payloads(
        self,
        archive_path: str,
        *,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
    ) -> Iterator[Any]:
        payloads: list[Any] = []
        try:
            with self._open_dataset_archive(
                OSWORLD_DATASET_REPO,
                archive_path,
                cache_root=cache_root,
                cache_budget_bytes=cache_budget_bytes,
                stage_remote_archives=stage_remote_archives,
            ) as archive:
                for info in archive.infolist():
                    if info.is_dir() or info.file_size <= 0:
                        continue
                    lower = info.filename.lower()
                    if not lower.endswith((".json", ".jsonl")):
                        continue
                    if info.file_size > 5_000_000:
                        continue
                    with archive.open(info) as handle:
                        raw = handle.read().decode("utf-8", errors="replace")
                    if lower.endswith("traj.jsonl"):
                        records: list[dict[str, Any]] = []
                        for line in raw.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(record, dict):
                                records.append(record)
                        if records:
                            payloads.append(
                                {
                                    "kind": "osworld_traj",
                                    "archive_member": info.filename,
                                    "app_context": info.filename.split("/", 1)[0],
                                    "records": records,
                                }
                            )
                        continue
                    if lower.endswith(".jsonl"):
                        for line in raw.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                payloads.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                        continue
                    try:
                        payloads.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except (OSError, zipfile.BadZipFile):
            return
        yield from payloads

    def _iter_transition_candidates(
        self,
        payload: Any,
    ) -> Iterator[tuple[dict[str, Any], UiAction, dict[str, Any]]]:
        if isinstance(payload, dict):
            if payload.get("kind") == "osworld_traj":
                records = payload.get("records")
                if isinstance(records, list):
                    yield from self._iter_osworld_traj_transitions(
                        records,
                        app_context=str(payload.get("app_context") or "os"),
                        archive_member=str(payload.get("archive_member") or ""),
                    )
                return
            before = _safe_dict(payload.get("before"))
            after = _safe_dict(payload.get("after"))
            action = _action_from_payload(payload.get("action"))
            if before and after and action is not None:
                yield before, action, after
            for value in payload.values():
                yield from self._iter_transition_candidates(value)
            return
        if isinstance(payload, list):
            for item in payload:
                yield from self._iter_transition_candidates(item)

    @staticmethod
    def _gui_actor_instruction(row: dict[str, Any]) -> str:
        for turn in list(row.get("conversations") or []):
            if str(turn.get("from") or "") != "human":
                continue
            return str(turn.get("value") or "").replace("<image>", "").strip()
        return ""

    @staticmethod
    def _gui_actor_bbox(row: dict[str, Any]) -> list[Any]:
        for turn in list(row.get("conversations") or []):
            bbox = turn.get("bbox_gt")
            if isinstance(bbox, list) and len(bbox) == 4:
                return bbox
        return []

    @staticmethod
    def _click100k_instruction(row: dict[str, Any]) -> str:
        prompt = str(row.get("easyr1_prompt") or "")
        if "<image>" in prompt:
            return prompt.split("<image>", 1)[1].strip()
        return prompt.strip()

    @staticmethod
    def _label_from_instruction(instruction: str, *, default: str) -> str:
        lowered = instruction.lower()
        if any(
            token in lowered for token in {"type", "enter", "write", "input", "search"}
        ):
            return "text_field"
        if any(token in lowered for token in {"menu", "tab", "dropdown"}):
            return "menu"
        return default

    @staticmethod
    def _label_from_screenspot_row(row: dict[str, Any]) -> str:
        instruction = str(row.get("instruction") or "").lower()
        data_type = str(row.get("data_type") or "").lower()
        if data_type == "text":
            return "text_field"
        if any(
            token in instruction
            for token in {"type", "enter", "write", "input", "search"}
        ):
            return "text_field"
        if any(token in instruction for token in {"menu", "tab", "dropdown"}):
            return "menu"
        return "icon" if data_type == "icon" else "button"

    @staticmethod
    def _center_from_bbox(
        bbox: list[Any],
        *,
        width: int,
        height: int,
        normalized: bool,
    ) -> tuple[int, int]:
        if len(bbox) != 4 or width <= 0 or height <= 0:
            return width // 2, height // 2
        if normalized:
            x0 = float(bbox[0]) * width
            y0 = float(bbox[1]) * height
            x1 = float(bbox[2]) * width
            y1 = float(bbox[3]) * height
        else:
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            x1 = float(bbox[2])
            y1 = float(bbox[3])
        return int((x0 + x1) / 2), int((y0 + y1) / 2)

    def _gui_actor_sources(
        self,
        source_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        payload = self._dataset_tree(GUI_ACTOR_DATASET_REPO)
        if not isinstance(payload, list):
            return []
        annotations: dict[str, str] = {}
        archives: dict[str, dict[str, Any]] = {}
        for item in payload:
            path = str(item.get("path") or "")
            filename = Path(path).name
            lower = filename.lower()
            if lower.endswith("_bbox.json"):
                annotations[_normalize_gui_actor_key(filename)] = path
            if lower.endswith(".zip"):
                archives[_normalize_gui_actor_key(filename)] = {
                    "path": path,
                    "size": _safe_int(item.get("size")),
                }
        pairs: list[dict[str, str]] = []
        sorted_annotations = sorted(
            annotations.items(),
            key=lambda item: (
                archives.get(item[0], {}).get("size", 1 << 62),
                item[0],
            ),
        )
        for key, annotation in sorted_annotations:
            if source_names and key not in source_names:
                continue
            archive = archives.get(key)
            if not archive:
                continue
            pairs.append(
                {
                    "name": key,
                    "annotation": annotation,
                    "archive": str(archive.get("path") or ""),
                }
            )
        return pairs

    @staticmethod
    def _archive_image_prefixes(
        archive: zipfile.ZipFile, *, sample_size: int = 256
    ) -> list[str]:
        prefixes: list[str] = []
        seen: set[str] = set()
        for info in archive.infolist()[:sample_size]:
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if not lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            prefix = (
                info.filename.rsplit("/", 1)[0] + "/" if "/" in info.filename else ""
            )
            if prefix in seen:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
        return prefixes

    @staticmethod
    def _resolve_archive_member_name(
        archive: zipfile.ZipFile,
        image_name: str,
        prefixes: list[str],
    ) -> str:
        candidates = [image_name, Path(image_name).name]
        basename = Path(image_name).name
        for prefix in prefixes:
            candidates.append(f"{prefix}{basename}")
        for candidate in candidates:
            try:
                archive.getinfo(candidate)
                return candidate
            except KeyError:
                continue
        return ""

    def _iter_osworld_traj_transitions(
        self,
        records: list[dict[str, Any]],
        *,
        app_context: str,
        archive_member: str,
    ) -> Iterator[tuple[dict[str, Any], UiAction, dict[str, Any]]]:
        for index, record in enumerate(records):
            action = self._action_from_osworld_record(record, app_context=app_context)
            if action is None:
                continue
            before = self._state_from_osworld_record(
                record,
                app_context=app_context,
                archive_member=archive_member,
            )
            next_record = record
            if index + 1 < len(records) and isinstance(records[index + 1], dict):
                next_record = records[index + 1]
            after = self._state_from_osworld_record(
                next_record,
                app_context=app_context,
                archive_member=archive_member,
            )
            after["last_action"] = action.action_type
            yield before, action, after

    @staticmethod
    def _state_from_osworld_record(
        record: dict[str, Any],
        *,
        app_context: str,
        archive_member: str,
    ) -> dict[str, Any]:
        action_text = str(record.get("action") or "")
        response_text = str(record.get("response") or "")
        screenshot_file = str(record.get("screenshot_file") or "")
        return {
            "app_context": app_context,
            "archive_member": archive_member,
            "step_num": _safe_int(record.get("step_num")),
            "reward": _safe_float(record.get("reward")),
            "done": bool(record.get("done")),
            "last_action": _infer_action_type_from_script(action_text),
            "focused_element": _script_selector_hint(action_text) or screenshot_file,
            "response_length": len(response_text),
            "screenshot_file": screenshot_file,
        }

    @staticmethod
    def _action_from_osworld_record(
        record: dict[str, Any],
        *,
        app_context: str,
    ) -> UiAction | None:
        action_text = str(record.get("action") or "").strip()
        if not action_text:
            return None
        action_type = _infer_action_type_from_script(action_text)
        selector_hint = _script_selector_hint(action_text) or f"{app_context}:step"
        value = _script_value_hint(action_text)
        return UiAction(
            action_type=action_type,
            selector=selector_hint,
            value=value,
            metadata={
                "source": "osworld_traj",
                "app_context": app_context,
                "raw_action": action_text,
            },
        )

    def _prepare_cache_root(
        self,
        cache_dir: str,
        *,
        stage_remote_archives: bool,
    ) -> Path | None:
        if not cache_dir and not stage_remote_archives:
            return None
        root = Path(cache_dir) if cache_dir else (self.models_root / "dataset_cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _load_long_run_state(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"completed_shards": [], "shard_results": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"completed_shards": [], "shard_results": []}

    @staticmethod
    def _write_long_run_state(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _dataset_tree(self, repo_id: str) -> list[dict[str, Any]]:
        cached = self._dataset_tree_cache.get(repo_id)
        if cached is not None:
            return cached
        url = DATASET_TREE_URLS.get(repo_id)
        if not url:
            self._dataset_tree_cache[repo_id] = []
            self._dataset_tree_index[repo_id] = {}
            return []
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            payload = []
        self._dataset_tree_cache[repo_id] = payload
        self._dataset_tree_index[repo_id] = {
            str(item.get("path") or ""): item
            for item in payload
            if str(item.get("path") or "")
        }
        return payload

    def _dataset_tree_item(
        self,
        repo_id: str,
        path: str,
    ) -> dict[str, Any]:
        self._dataset_tree(repo_id)
        return dict(self._dataset_tree_index.get(repo_id, {}).get(path) or {})

    @contextlib.contextmanager
    def _open_dataset_archive(
        self,
        repo_id: str,
        path: str,
        *,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
    ) -> Iterator[zipfile.ZipFile]:
        staged_path = self._maybe_stage_archive(
            repo_id,
            path,
            cache_root=cache_root,
            cache_budget_bytes=cache_budget_bytes,
            stage_remote_archives=stage_remote_archives,
        )
        if staged_path is not None and staged_path.exists():
            with zipfile.ZipFile(staged_path) as archive:
                yield archive
            return
        if self._split_archive_members(repo_id, path):
            raise OSError(f"Split archive requires staging: {path}")
        archive_url = self._hf_resolve_url(repo_id, path)
        with zipfile.ZipFile(_HttpRangeReader(archive_url)) as archive:
            yield archive

    def _maybe_stage_archive(
        self,
        repo_id: str,
        path: str,
        *,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
    ) -> Path | None:
        split_members = self._split_archive_members(repo_id, path)
        if split_members:
            if cache_root is None:
                return None
            combined_path = self._cache_file_path(
                cache_root,
                repo_id,
                path,
                suffix_override=".combined.zip",
            )
            if combined_path.exists():
                return combined_path
            if not stage_remote_archives:
                return None
            total_size = sum(
                _safe_int(self._dataset_tree_item(repo_id, member).get("size"))
                for member in split_members
            )
            if not self._cache_space_available(
                cache_root,
                required_bytes=total_size,
                cache_budget_bytes=cache_budget_bytes,
            ):
                return None
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = combined_path.with_suffix(combined_path.suffix + ".tmp")
            with temp_path.open("wb") as handle:
                for member in split_members:
                    self._download_url_to_handle(
                        self._hf_resolve_url(repo_id, member),
                        handle,
                    )
            temp_path.replace(combined_path)
            return combined_path
        return self._maybe_stage_remote_file(
            repo_id,
            path,
            cache_root=cache_root,
            cache_budget_bytes=cache_budget_bytes,
            stage_remote_archives=stage_remote_archives,
        )

    def _maybe_stage_remote_file(
        self,
        repo_id: str,
        path: str,
        *,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
    ) -> Path | None:
        if cache_root is None:
            return None
        local_path = self._cache_file_path(cache_root, repo_id, path)
        if local_path.exists():
            return local_path
        if not stage_remote_archives:
            return None
        size = _safe_int(self._dataset_tree_item(repo_id, path).get("size"))
        if not self._cache_space_available(
            cache_root,
            required_bytes=size,
            cache_budget_bytes=cache_budget_bytes,
        ):
            return None
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        self._download_url_to_path(self._hf_resolve_url(repo_id, path), temp_path)
        temp_path.replace(local_path)
        return local_path

    def _split_archive_members(self, repo_id: str, path: str) -> list[str]:
        filename = Path(path).name
        lowered = filename.lower()
        if "split.zip" not in lowered:
            return []
        stem = Path(filename).stem
        parts: list[str] = []
        for item in self._dataset_tree(repo_id):
            candidate = str(item.get("path") or "")
            name = Path(candidate).name
            candidate_lower = name.lower()
            if candidate_lower == lowered:
                continue
            if not candidate_lower.startswith(stem.lower() + ".z"):
                continue
            parts.append(candidate)
        if not parts:
            return []
        parts.sort(key=_split_archive_order_key)
        return parts + [path]

    @staticmethod
    def _cache_file_path(
        cache_root: Path,
        repo_id: str,
        path: str,
        *,
        suffix_override: str = "",
    ) -> Path:
        repo_root = cache_root / repo_id.replace("/", "__")
        target = repo_root / Path(path)
        if suffix_override:
            return target.with_suffix(suffix_override)
        return target

    @staticmethod
    def _download_url_to_path(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            UnknownAppAdaptationTrainer._download_url_to_handle(url, handle)

    @staticmethod
    def _download_url_to_handle(url: str, handle: Any) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "AgentOS/1.0"})
        with _urlopen_with_retries(request, timeout=60.0) as response:
            shutil.copyfileobj(response, handle, length=4 << 20)

    @staticmethod
    def _cache_space_available(
        cache_root: Path,
        *,
        required_bytes: int,
        cache_budget_bytes: int,
    ) -> bool:
        reserve_bytes = 5 << 30
        root = cache_root if cache_root.exists() else cache_root.parent
        free_bytes = shutil.disk_usage(root).free
        if required_bytes > 0 and free_bytes - reserve_bytes < required_bytes:
            return False
        if cache_budget_bytes > 0:
            current_bytes = sum(
                file_path.stat().st_size
                for file_path in cache_root.rglob("*")
                if file_path.is_file()
            )
            if current_bytes + max(required_bytes, 0) > cache_budget_bytes:
                return False
        return True

    @staticmethod
    def _hf_resolve_url(repo_id: str, path: str) -> str:
        return HF_DATASET_RESOLVE_URL.format(
            repo_id=urllib.parse.quote(repo_id, safe="/"),
            path=urllib.parse.quote(path, safe="/[]_-"),
        )

    def _iter_json_array_source(
        self,
        repo_id: str,
        path: str,
        *,
        limit: int = 0,
        skip: int = 0,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
    ) -> Iterator[Any]:
        local_path = self._maybe_stage_remote_file(
            repo_id,
            path,
            cache_root=cache_root,
            cache_budget_bytes=cache_budget_bytes,
            stage_remote_archives=stage_remote_archives,
        )
        if local_path is not None and local_path.exists():
            yield from self._iter_json_array_from_file(
                local_path, limit=limit, skip=skip
            )
            return
        yield from self._iter_json_array_from_url(
            self._hf_resolve_url(repo_id, path),
            limit=limit,
            skip=skip,
        )

    def _iter_json_array_from_url(
        self,
        url: str,
        *,
        limit: int = 0,
        skip: int = 0,
    ) -> Iterator[Any]:
        decoder = json.JSONDecoder()
        request = urllib.request.Request(url, headers={"User-Agent": "AgentOS/1.0"})
        with _urlopen_with_retries(request, timeout=60.0) as response:
            buffer = ""
            in_array = False
            yielded = 0
            skipped = 0
            while True:
                chunk = response.read(65536)
                if chunk:
                    buffer += chunk.decode("utf-8", errors="replace")
                index = 0
                while True:
                    while index < len(buffer) and buffer[index] in " \r\n\t,":
                        index += 1
                    if not in_array:
                        if index >= len(buffer):
                            break
                        if buffer[index] != "[":
                            raise ValueError(f"Expected a JSON array at {url}")
                        in_array = True
                        index += 1
                        continue
                    while index < len(buffer) and buffer[index] in " \r\n\t,":
                        index += 1
                    if index >= len(buffer):
                        break
                    if buffer[index] == "]":
                        return
                    try:
                        value, next_index = decoder.raw_decode(buffer, index)
                    except json.JSONDecodeError:
                        break
                    if skipped < skip:
                        skipped += 1
                    else:
                        yield value
                        yielded += 1
                        if limit > 0 and yielded >= limit:
                            return
                    index = next_index
                if index > 0:
                    buffer = buffer[index:]
                if not chunk:
                    return

    def _iter_json_array_from_file(
        self,
        path: Path,
        *,
        limit: int = 0,
        skip: int = 0,
    ) -> Iterator[Any]:
        decoder = json.JSONDecoder()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            buffer = ""
            in_array = False
            yielded = 0
            skipped = 0
            while True:
                chunk = handle.read(65536)
                if chunk:
                    buffer += chunk
                index = 0
                while True:
                    while index < len(buffer) and buffer[index] in " \r\n\t,":
                        index += 1
                    if not in_array:
                        if index >= len(buffer):
                            break
                        if buffer[index] != "[":
                            raise ValueError(f"Expected a JSON array at {path}")
                        in_array = True
                        index += 1
                        continue
                    while index < len(buffer) and buffer[index] in " \r\n\t,":
                        index += 1
                    if index >= len(buffer):
                        break
                    if buffer[index] == "]":
                        return
                    try:
                        value, next_index = decoder.raw_decode(buffer, index)
                    except json.JSONDecodeError:
                        break
                    if skipped < skip:
                        skipped += 1
                    else:
                        yield value
                        yielded += 1
                        if limit > 0 and yielded >= limit:
                            return
                    index = next_index
                if index > 0:
                    buffer = buffer[index:]
                if not chunk:
                    return

    @staticmethod
    def _fetch_json(url: str) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": "AgentOS/1.0"})
        with _urlopen_with_retries(request, timeout=30.0) as response:
            payload = response.read().decode("utf-8")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return ast.literal_eval(payload)

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "AgentOS/1.0"})
        with _urlopen_with_retries(request, timeout=30.0) as response:
            return response.read()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _distributed_targets(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return [0] * max(count, 0)
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _underfill_report(
    requested: dict[str, int],
    actual: dict[str, int],
) -> dict[str, Any]:
    items: dict[str, dict[str, int | float]] = {}
    missing_total = 0
    for name, requested_count in requested.items():
        requested_value = max(0, _safe_int(requested_count))
        actual_value = max(0, _safe_int(actual.get(name)))
        missing = max(0, requested_value - actual_value)
        if requested_value <= 0 and actual_value <= 0:
            continue
        missing_total += missing
        fill_rate = 1.0 if requested_value <= 0 else actual_value / requested_value
        items[name] = {
            "requested": requested_value,
            "used": actual_value,
            "missing": missing,
            "fill_rate": round(min(1.0, fill_rate), 4),
        }
    return {
        "underfilled": any(item["missing"] for item in items.values()),
        "missing_total": missing_total,
        "items": items,
    }


def _underfill_notes(report: dict[str, Any]) -> list[str]:
    if not report.get("underfilled"):
        return []
    notes: list[str] = []
    for name, item in dict(report.get("items") or {}).items():
        missing = _safe_int(dict(item).get("missing"))
        if missing <= 0:
            continue
        notes.append(
            f"Underfilled {name}: missing {missing} of {dict(item).get('requested', 0)} requested."
        )
    return notes


def _scale_report(
    *,
    grounding_examples: int,
    world_model_transitions: int,
    app_families: list[str],
) -> dict[str, Any]:
    minimum_target = 100_000
    production_target = 10_000_000
    grounding = max(0, _safe_int(grounding_examples))
    transitions = max(0, _safe_int(world_model_transitions))
    families = sorted({str(item) for item in app_families if str(item)})
    meets_minimum = grounding >= minimum_target and transitions >= minimum_target
    meets_production = (
        grounding >= production_target and transitions >= production_target
    )
    return {
        "grounding_examples": grounding,
        "world_model_transitions": transitions,
        "app_family_count": len(families),
        "app_families": families,
        "minimum_target": minimum_target,
        "stretch_target": production_target,
        "production_target": production_target,
        "grounding_remaining_to_minimum": max(0, minimum_target - grounding),
        "transition_remaining_to_minimum": max(0, minimum_target - transitions),
        "grounding_remaining_to_stretch": max(0, production_target - grounding),
        "transition_remaining_to_stretch": max(0, production_target - transitions),
        "grounding_remaining_to_production": max(0, production_target - grounding),
        "transition_remaining_to_production": max(
            0,
            production_target - transitions,
        ),
        "meets_minimum_scale": meets_minimum,
        "meets_stretch_scale": meets_production,
        "meets_production_scale": meets_production,
        "scale_stage": _scale_stage(meets_minimum, meets_production),
    }


def _scale_stage(meets_minimum: bool, meets_production: bool) -> str:
    if meets_production:
        return "production"
    if meets_minimum:
        return "minimum"
    return "bootstrap"


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return []
    return [
        items[index : index + chunk_size] for index in range(0, len(items), chunk_size)
    ]


def _osworld_shard_batches(
    archives: list[str],
    *,
    shard_count: int,
    archives_per_shard: int,
    candidate_multiplier: int,
) -> list[list[str]]:
    if shard_count <= 0 or archives_per_shard <= 0:
        return []
    multiplier = max(1, candidate_multiplier)
    requested_archive_count = shard_count * archives_per_shard
    extra_per_shard = archives_per_shard * (multiplier - 1)
    batches: list[list[str]] = []
    for shard_index in range(shard_count):
        core_start = shard_index * archives_per_shard
        core_end = core_start + archives_per_shard
        batch = list(archives[core_start:core_end])
        if extra_per_shard > 0:
            extra_start = requested_archive_count + shard_index * extra_per_shard
            extra_end = extra_start + extra_per_shard
            batch.extend(archives[extra_start:extra_end])
        batches.append(batch)
    return batches


def _split_archive_order_key(path: str) -> tuple[int, str]:
    suffix = Path(path).suffix.lower()
    if suffix.startswith(".z"):
        return (_safe_int(suffix[2:]), path)
    return (1 << 30, path)


def _urlopen_with_retries(
    request: urllib.request.Request,
    *,
    timeout: float,
    attempts: int = 4,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 >= max(1, attempts):
            assert last_error is not None
            raise last_error
        time.sleep(min(8.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def _action_from_payload(payload: Any) -> UiAction | None:
    data = payload if isinstance(payload, dict) else None
    if not data:
        return None
    action_type = str(data.get("action_type") or "").strip()
    selector = str(data.get("selector") or "").strip()
    if not action_type or not selector:
        return None
    return UiAction(
        action_type=action_type,
        selector=selector,
        value=(None if data.get("value") is None else str(data.get("value"))),
        metadata=dict(data.get("metadata") or {}),
    )


def _normalize_gui_actor_key(filename: str) -> str:
    lowered = filename.lower()
    suffixes = (
        "_bbox.json",
        "_images_split.zip",
        "_images.zip",
        "-images.zip",
        ".zip",
    )
    base = lowered
    for suffix in suffixes:
        if lowered.endswith(suffix):
            base = lowered[: -len(suffix)]
            break
    return "".join(ch for ch in base if ch.isalnum())


def _infer_action_type_from_script(script: str) -> str:
    lowered = script.strip().lower()
    if lowered == "done":
        return "invoke"
    if "click(" in lowered:
        return "click"
    if any(
        token in lowered
        for token in {"write(", "typewrite(", "insert_text", "set_text"}
    ):
        return "type"
    if any(token in lowered for token in {"hotkey(", "press(", "key("}):
        return "hotkey"
    if "scroll(" in lowered:
        return "scroll"
    if any(token in lowered for token in {"focus(", "activate("}):
        return "focus"
    return "invoke"


def _script_selector_hint(script: str) -> str:
    raw = script.strip()
    if not raw:
        return ""
    head = raw.split(";", 1)[0].strip()
    if "(" in head:
        head = head.split("(", 1)[0]
    return head[:120]


def _script_value_hint(script: str) -> str | None:
    marker = "'"
    if marker in script:
        parts = script.split(marker)
        if len(parts) >= 3:
            value = parts[1].strip()
            return value or None
    marker = '"'
    if marker in script:
        parts = script.split(marker)
        if len(parts) >= 3:
            value = parts[1].strip()
            return value or None
    return None


class _HttpRangeReader(io.RawIOBase):
    def __init__(self, url: str, *, block_size: int = 4 << 20) -> None:
        self.url = url
        self.block_size = block_size
        self.position = 0
        self.length = self._probe_length()
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self.position = max(0, min(self.length, new_position))
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.length - self.position
        if size <= 0 or self.position >= self.length:
            return b""
        end = min(self.length, self.position + size)
        start = self.position
        chunks: list[bytes] = []
        while start < end:
            block_index = start // self.block_size
            block = self._fetch_block(block_index)
            block_start = block_index * self.block_size
            offset = start - block_start
            take = min(end - start, len(block) - offset)
            chunks.append(block[offset : offset + take])
            start += take
        self.position = end
        return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size

    def _probe_length(self) -> int:
        request = urllib.request.Request(
            self.url,
            headers={"Range": "bytes=0-0", "User-Agent": "AgentOS/1.0"},
        )
        with _urlopen_with_retries(request, timeout=30.0) as response:
            content_range = str(response.headers.get("Content-Range") or "")
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                return int(content_length)
            return len(response.read())

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            return cached
        start = block_index * self.block_size
        end = min(self.length, start + self.block_size) - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "AgentOS/1.0"},
        )
        with _urlopen_with_retries(request, timeout=60.0) as response:
            payload = response.read()
        if len(self._cache) >= 32:
            self._cache.clear()
        self._cache[block_index] = payload
        return payload
