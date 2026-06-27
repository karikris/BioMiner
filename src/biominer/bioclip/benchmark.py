from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Iterable
from uuid import uuid4

import polars as pl

from biominer.bioclip.candidate_sets import CandidateMode, CandidateStrategy, parse_candidate_mode, parse_candidate_strategy
from biominer.bioclip.diagnostics import mps_memory_metrics
from biominer.storage.parquet import write_parquet


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_candidate_modes(value: str) -> list[CandidateMode]:
    return [parse_candidate_mode(part) for part in value.split(",") if part.strip()]


def build_benchmark_report(
    *,
    input_path: str | Path,
    species_candidates_path: str | Path,
    output_path: str | Path,
    device: str,
    register_sizes: Iterable[int],
    register_counts: Iterable[int],
    candidate_limits: Iterable[int],
    classification_modes: Iterable[CandidateMode],
    candidate_strategy: str | CandidateStrategy,
    download_workers: int,
) -> dict[str, object]:
    started_at = datetime.now(UTC)
    run_id = f"bioclip-benchmark-{uuid4().hex[:12]}"
    rows_in = _parquet_row_count(Path(input_path))
    species_candidate_rows = _parquet_or_table_row_count(Path(species_candidates_path))
    strategy = parse_candidate_strategy(candidate_strategy)
    mps_metrics = mps_memory_metrics()
    runs: list[dict[str, object]] = []
    for register_count in register_counts:
        for register_size in register_sizes:
            for candidate_limit in candidate_limits:
                for mode in classification_modes:
                    runs.append(
                        {
                            "run_id": run_id,
                            "git_sha": _git_sha(),
                            "started_at": started_at.isoformat(),
                            "ended_at": None,
                            "device": device,
                            "mps_available": mps_metrics["mps_available"],
                            "register_count": register_count,
                            "register_size": register_size,
                            "download_workers": download_workers,
                            "candidate_limit": candidate_limit,
                            "classification_mode": mode.value,
                            "candidate_strategy": strategy.value,
                            "rows_in": rows_in,
                            "rows_out": None,
                            "images_classified": 0,
                            "images_per_second": "not_instrumented",
                            "seconds_per_image": "not_instrumented",
                            "rss_peak_memory_bytes": None,
                            "mps_current_allocated_memory_bytes": mps_metrics["mps_current_allocated_memory_bytes"],
                            "mps_driver_allocated_memory_bytes": mps_metrics["mps_driver_allocated_memory_bytes"],
                            "mps_recommended_max_memory_bytes": mps_metrics["mps_recommended_max_memory_bytes"],
                            "download_failure_count": 0,
                            "bioclip_failure_count": 0,
                            "bucket_counts": "not_instrumented",
                            "category_counts": "not_instrumented",
                            "life_stage_counts": "not_instrumented",
                            "species_top1_score_distribution": "not_instrumented",
                            "species_top1_top2_margin_distribution": "not_instrumented",
                            "species_entropy_distribution": "not_instrumented",
                            "species_candidate_rows": species_candidate_rows,
                        }
                    )
    ended_at = datetime.now(UTC).isoformat()
    for row in runs:
        row["ended_at"] = ended_at
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at,
        "input": str(input_path),
        "species_candidates": str(species_candidates_path),
        "output": str(output_path),
        "runs": runs,
        "notes": [
            "Benchmark skeleton enumerates requested configurations without running BioCLIP inference.",
            "Unsupported runtime metrics are null or not_instrumented.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def write_benchmark_runs_parquet(report: dict[str, object], output_path: str | Path) -> Path:
    runs = report.get("runs")
    frame = pl.DataFrame(runs if isinstance(runs, list) else [])
    return write_parquet(frame, output_path)


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _parquet_or_table_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.casefold() == ".parquet":
        return _parquet_row_count(path)
    if path.suffix.casefold() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.casefold() == ".tsv" else ","
        return pl.scan_csv(path, separator=separator).select(pl.len()).collect().item()
    return 0


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=False, text=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"
