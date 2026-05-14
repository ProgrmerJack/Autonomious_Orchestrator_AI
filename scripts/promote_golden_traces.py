"""Promote successful research runs into the benchmark golden-trace corpus.

Scans ``runs/`` for completed runs (those with non-empty
``research/sources.json`` + ``research/analysis_report.md`` + optional
``pc/frontier_graph.json``), scores them by signal richness, and copies
the top-N into ``benchmarks/golden_traces/<run_id>/``.

These golden traces become the regression set the orchestrator replays
during evaluation, so promotion is gated: only runs that actually
finished and produced meaningful artifacts are eligible.

Usage::

    python scripts/promote_golden_traces.py --top 50
    python scripts/promote_golden_traces.py --top 50 --dry-run
    python scripts/promote_golden_traces.py --runs-root runs \\
        --golden-root benchmarks/golden_traces --top 25
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CandidateScore:
    run_id: str
    run_path: Path
    score: float
    sources: int
    has_frontier_graph: bool
    has_evidence_graph: bool
    analysis_bytes: int
    sources_bytes: int


_REQUIRED_FILES: tuple[str, ...] = (
    "research/sources.json",
    "research/analysis_report.md",
)
_OPTIONAL_SIGNALS: tuple[str, ...] = (
    "pc/frontier_graph.json",
    "research/evidence_graph.json",
    "research/synthesis_packet.json",
    "research/claim_trace.json",
    "research/research_plan.json",
)


def _is_eligible(run_path: Path) -> bool:
    for relative in _REQUIRED_FILES:
        full = run_path / relative
        if not full.is_file():
            return False
        try:
            if full.stat().st_size < 256:
                return False
        except OSError:
            return False
    return True


def _count_sources(run_path: Path) -> int:
    sources_path = run_path / "research" / "sources.json"
    try:
        data = json.loads(sources_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("sources", "items", "records"):
            inner = data.get(key)
            if isinstance(inner, list):
                return len(inner)
    return 0


def _score_run(run_path: Path) -> CandidateScore | None:
    if not _is_eligible(run_path):
        return None
    sources_count = _count_sources(run_path)
    analysis = run_path / "research" / "analysis_report.md"
    sources_file = run_path / "research" / "sources.json"
    try:
        analysis_bytes = analysis.stat().st_size
        sources_bytes = sources_file.stat().st_size
    except OSError:
        return None
    score = 0.0
    score += min(sources_count, 1000) * 1.0
    score += min(analysis_bytes / 1024.0, 200.0)
    score += min(sources_bytes / 1024.0, 200.0)
    has_frontier = (run_path / "pc" / "frontier_graph.json").is_file()
    has_evidence = (run_path / "research" / "evidence_graph.json").is_file()
    for relative in _OPTIONAL_SIGNALS:
        if (run_path / relative).is_file():
            score += 25.0
    return CandidateScore(
        run_id=run_path.name,
        run_path=run_path,
        score=score,
        sources=sources_count,
        has_frontier_graph=has_frontier,
        has_evidence_graph=has_evidence,
        analysis_bytes=analysis_bytes,
        sources_bytes=sources_bytes,
    )


def _copy_run(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # Copy only artifact-bearing subdirectories to keep golden traces tight.
    for subdir in ("research", "pc", "planning", "workflows"):
        src_dir = src / subdir
        if src_dir.is_dir():
            shutil.copytree(src_dir, dst / subdir, dirs_exist_ok=True)
    for filename in ("heartbeat.json", "run_progress.json", "final_report.json"):
        src_file = src / filename
        if src_file.is_file():
            shutil.copy2(src_file, dst / filename)


def promote(
    runs_root: Path,
    golden_root: Path,
    top: int,
    dry_run: bool,
) -> list[CandidateScore]:
    if not runs_root.is_dir():
        raise SystemExit(f"runs root does not exist: {runs_root}")
    candidates: list[CandidateScore] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        scored = _score_run(run_dir)
        if scored is not None:
            candidates.append(scored)
    candidates.sort(key=lambda c: c.score, reverse=True)
    selected = candidates[:top]
    if not dry_run:
        golden_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "promoted_at_utc": None,
            "count": len(selected),
            "runs": [],
        }
        import datetime as _dt

        manifest["promoted_at_utc"] = _dt.datetime.now(_dt.UTC).isoformat()
        for cand in selected:
            dst = golden_root / cand.run_id
            _copy_run(cand.run_path, dst)
            manifest["runs"].append(
                {
                    "run_id": cand.run_id,
                    "score": round(cand.score, 2),
                    "sources": cand.sources,
                    "analysis_bytes": cand.analysis_bytes,
                    "sources_bytes": cand.sources_bytes,
                    "has_frontier_graph": cand.has_frontier_graph,
                    "has_evidence_graph": cand.has_evidence_graph,
                }
            )
        manifest_path = golden_root / "promotion_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=Path("benchmarks") / "golden_traces",
    )
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = promote(
        runs_root=args.runs_root,
        golden_root=args.golden_root,
        top=args.top,
        dry_run=args.dry_run,
    )
    print(f"{'[dry-run] ' if args.dry_run else ''}selected {len(selected)} runs")
    for cand in selected:
        print(
            f"  {cand.run_id:48s} "
            f"score={cand.score:8.2f} "
            f"sources={cand.sources:5d} "
            f"frontier={'Y' if cand.has_frontier_graph else 'N'} "
            f"evidence={'Y' if cand.has_evidence_graph else 'N'}"
        )


if __name__ == "__main__":
    main()
