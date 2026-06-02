from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
import tracemalloc
from typing import Callable

import polars as pl

from flickr_bio_occurrence.dwc.exporter import export_dwc_records
from flickr_bio_occurrence.flickr.client import FlickrSearchResult
from flickr_bio_occurrence.flickr.work_items import WorkItem, build_monthly_work_items
from flickr_bio_occurrence.pipeline.transforms import build_dwc_rows, build_silver_candidates, flatten_search_payloads
from flickr_bio_occurrence.storage.duckdb_index import create_qa_views
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset
from flickr_bio_occurrence.taxonomy.species_mapper import get_seed_species


SearchPhotos = Callable[[WorkItem], FlickrSearchResult]


def run_live_search_benchmark(
    *,
    search_photos: SearchPhotos,
    output_dir: str | Path,
    target_records: int = 1000,
    max_calls: int = 3000,
    species_name: str = "Papilio demoleus",
    regions: list[tuple[str, str, str]] | None = None,
    years: list[int] | None = None,
    months: range = range(1, 13),
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tracemalloc.start()
    work_items = build_monthly_work_items(
        species=get_seed_species(species_name),
        regions=regions or [("AU_ALL", "Australia", "112.92,-43.74,153.64,-10.05")],
        years=years or [2024, 2023, 2022, 2021, 2020],
        months=months,
    )
    payloads: list[tuple[WorkItem, dict[str, object]]] = []
    seen: set[str] = set()
    errors: list[dict[str, str]] = []
    fetch_start = time.perf_counter()
    for item in work_items[:max_calls]:
        if len(seen) >= target_records:
            break
        try:
            result = search_photos(item)
        except Exception as exc:  # noqa: BLE001 - benchmark report must preserve failures and continue.
            errors.append({"work_item_id": item.work_item_id, "error": type(exc).__name__, "message": str(exc)[:300]})
            continue
        payloads.append((item, result.payload))
        seen.update(result.photo_ids)
    timings["flickr_fetch"] = time.perf_counter() - fetch_start
    bronze = _timed(timings, "bronze_flattening_dedup", lambda: _build_bronze(payloads, species_name, target_records))
    silver = _timed(timings, "silver_candidate_build", lambda: build_silver_candidates(bronze))
    gold = _timed(timings, "dwc_mapping", lambda: build_dwc_rows(silver))
    bronze_paths, silver_paths, gold_paths, duckdb_path = _timed(
        timings,
        "artifact_write",
        lambda: _write_outputs(output_path, bronze, silver, gold),
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    files = [path for path in output_path.rglob("*") if path.is_file()]
    report = {
        "species": species_name,
        "region": "Australia",
        "target_record_count": target_records,
        "actual_unique_records": bronze.height,
        "work_items_planned": len(work_items),
        "work_items_called": len(payloads) + len(errors),
        "max_calls": max_calls,
        "errors": errors[:20],
        "step_timings_seconds": timings,
        "storage_artifacts": {
            "root": str(output_path),
            "bronze_parquet_files": len(bronze_paths),
            "silver_parquet_files": len(silver_paths),
            "gold_parquet_files": len(gold_paths),
            "duckdb_path": str(duckdb_path) if duckdb_path else None,
            "total_artifact_bytes": sum(path.stat().st_size for path in files),
            "files": {str(path): path.stat().st_size for path in files if path.suffix in {".parquet", ".duckdb", ".json"}},
        },
        "memory_artifacts": {
            "current_traced_bytes": current,
            "peak_traced_bytes": peak,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "compute_artifacts": {
            "worker_count": min(16, os.cpu_count() or 1),
            "cpu_count": os.cpu_count() or 1,
            "gpu_used": False,
            "vision_model_loaded": False,
            "http_client": "httpx",
        },
    }
    report_path = output_path / "live_search_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _build_bronze(payloads: list[tuple[WorkItem, dict[str, object]]], species_name: str, target_records: int) -> pl.DataFrame:
    frames = [
        flatten_search_payloads([payload], species_name=species_name, region_id=item.region_id, work_item_id=item.work_item_id)
        for item, payload in payloads
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(subset=["flickr_photo_id"], keep="first").head(target_records)


def _write_outputs(output_path: Path, bronze: pl.DataFrame, silver: pl.DataFrame, gold: pl.DataFrame) -> tuple[list[Path], list[Path], list[Path], Path | None]:
    if not bronze.height:
        return [], [], [], None
    bronze_paths = write_parquet_dataset(bronze, output_path / "bronze" / "bronze_flickr_photo")
    silver_paths = write_parquet_dataset(silver, output_path / "silver" / "silver_occurrence_candidate")
    gold_outputs = export_dwc_records(gold, output_path / "gold")
    duckdb_path = create_qa_views(db_path=output_path / "live_search_benchmark.duckdb", data_root=output_path)
    return bronze_paths, silver_paths, gold_outputs.parquet_paths, duckdb_path


def _timed[T](timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = fn()
    timings[name] = time.perf_counter() - start
    return result
