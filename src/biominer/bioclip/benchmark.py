from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import resource
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable
from uuid import uuid4

import polars as pl

from biominer.bioclip.candidate_sets import CandidateMode, CandidateStrategy, parse_candidate_mode, parse_candidate_strategy
from biominer.bioclip.diagnostics import mps_memory_metrics
from biominer.storage.parquet import write_parquet


@dataclass(frozen=True)
class BenchmarkRunConfig:
    run_id: str
    output_path: Path
    device: str
    register_count: int
    register_size: int
    download_workers: int
    candidate_limit: int
    classification_mode: CandidateMode
    candidate_strategy: CandidateStrategy


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
    dry_run: bool = True,
    screen_runner: Callable[[BenchmarkRunConfig], object] | None = None,
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
                    config = BenchmarkRunConfig(
                        run_id=run_id,
                        output_path=_classified_output_path(Path(output_path), mode, register_count, register_size, candidate_limit),
                        device=device,
                        register_count=register_count,
                        register_size=register_size,
                        download_workers=download_workers,
                        candidate_limit=candidate_limit,
                        classification_mode=mode,
                        candidate_strategy=strategy,
                    )
                    if dry_run:
                        runs.append(_dry_run_row(config, rows_in=rows_in, species_candidate_rows=species_candidate_rows, mps_metrics=mps_metrics))
                    else:
                        if screen_runner is None:
                            raise ValueError("screen_runner is required when dry_run is false")
                        runs.append(_real_run_row(config, rows_in=rows_in, species_candidate_rows=species_candidate_rows, screen_runner=screen_runner))
    ended_at = datetime.now(UTC).isoformat()
    for row in runs:
        row["ended_at"] = row["ended_at"] or ended_at
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
            "Dry-run configurations enumerate requested settings without running BioCLIP inference."
            if dry_run
            else "Real benchmark configurations ran bounded BioCLIP screen passes.",
            "Unsupported runtime metrics are null or not_instrumented.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_benchmark_runs_parquet(report, output.with_suffix(output.suffix + ".runs.parquet"))
    return report


def write_benchmark_runs_parquet(report: dict[str, object], output_path: str | Path) -> Path:
    runs = report.get("runs")
    frame = pl.DataFrame([_parquet_safe_row(row) for row in runs] if isinstance(runs, list) else [])
    return write_parquet(frame, output_path)


def _dry_run_row(
    config: BenchmarkRunConfig,
    *,
    rows_in: int,
    species_candidate_rows: int,
    mps_metrics: dict[str, object],
) -> dict[str, object]:
    return {
        **_base_run_row(config, rows_in=rows_in, species_candidate_rows=species_candidate_rows, started_at=datetime.now(UTC).isoformat()),
        "ended_at": None,
        "mps_available": mps_metrics["mps_available"],
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
        "classified_output": None,
    }


def _real_run_row(
    config: BenchmarkRunConfig,
    *,
    rows_in: int,
    species_candidate_rows: int,
    screen_runner: Callable[[BenchmarkRunConfig], object],
) -> dict[str, object]:
    started_at = datetime.now(UTC)
    before_mps = mps_memory_metrics()
    start = time.perf_counter()
    result = screen_runner(config)
    elapsed = max(0.0, time.perf_counter() - start)
    after_mps = mps_memory_metrics()
    frame = getattr(result, "frame", pl.DataFrame())
    if not isinstance(frame, pl.DataFrame):
        frame = pl.DataFrame()
    images_classified = int(getattr(result, "records_classified", 0) or 0)
    row = {
        **_base_run_row(config, rows_in=rows_in, species_candidate_rows=species_candidate_rows, started_at=started_at.isoformat()),
        "ended_at": datetime.now(UTC).isoformat(),
        "mps_available": after_mps["mps_available"],
        "rows_out": frame.height,
        "images_classified": images_classified,
        "images_per_second": images_classified / elapsed if images_classified and elapsed > 0 else 0.0,
        "seconds_per_image": elapsed / images_classified if images_classified else None,
        "rss_peak_memory_bytes": _rss_peak_memory_bytes(),
        "mps_current_allocated_memory_bytes": after_mps["mps_current_allocated_memory_bytes"],
        "mps_driver_allocated_memory_bytes": after_mps["mps_driver_allocated_memory_bytes"],
        "mps_recommended_max_memory_bytes": after_mps["mps_recommended_max_memory_bytes"],
        "download_failure_count": int(getattr(result, "download_failures", 0) or 0),
        "bioclip_failure_count": int(getattr(result, "bioclip_failures", 0) or 0),
        "bucket_counts": _count_map(frame, "occurrence_bin"),
        "category_counts": _count_map(frame, "image_category"),
        "life_stage_counts": _count_map(frame, "life_stage"),
        "species_top1_score_distribution": _numeric_distribution(frame, "species_top1_score"),
        "species_top1_top2_margin_distribution": _numeric_distribution(frame, "species_top1_top2_margin"),
        "species_entropy_distribution": _numeric_distribution(frame, "species_entropy"),
        "mps_before": before_mps,
        "classified_output": str(config.output_path),
    }
    return row


def _base_run_row(
    config: BenchmarkRunConfig,
    *,
    rows_in: int,
    species_candidate_rows: int,
    started_at: str,
) -> dict[str, object]:
    return {
        "run_id": config.run_id,
        "git_sha": _git_sha(),
        "started_at": started_at,
        "device": config.device,
        "register_count": config.register_count,
        "register_size": config.register_size,
        "download_workers": config.download_workers,
        "candidate_limit": config.candidate_limit,
        "classification_mode": config.classification_mode.value,
        "candidate_strategy": config.candidate_strategy.value,
        "rows_in": rows_in,
        "species_candidate_rows": species_candidate_rows,
    }


def _count_map(frame: pl.DataFrame, column: str) -> dict[str, int] | str:
    if frame.is_empty() or column not in frame.columns:
        return "not_instrumented"
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).to_dicts()
        if row[column] is not None
    }


def _numeric_distribution(frame: pl.DataFrame, column: str) -> dict[str, float | int | None] | str:
    if frame.is_empty() or column not in frame.columns:
        return "not_instrumented"
    values = frame.select(pl.col(column).cast(pl.Float64, strict=False).drop_nulls().alias(column))[column]
    if values.is_empty():
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(values.min()),
        "p50": float(values.quantile(0.5)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _rss_peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _parquet_safe_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        return {}
    return {
        str(key): json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def _classified_output_path(
    output_path: Path,
    mode: CandidateMode,
    register_count: int,
    register_size: int,
    candidate_limit: int,
) -> Path:
    stem = f"{output_path.stem}.{mode.value}.r{register_count}.s{register_size}.c{candidate_limit}.classified"
    return output_path.with_name(stem).with_suffix(".parquet")


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
