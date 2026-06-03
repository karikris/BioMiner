from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
import tracemalloc
from typing import Iterable

import polars as pl

from flickr_bio_occurrence.benchmark.live_run import VisionClassifier, _timed
from flickr_bio_occurrence.dwc.exporter import export_dwc_records
from flickr_bio_occurrence.pipeline.transforms import build_dwc_rows, build_silver_candidates, flatten_search_payloads
from flickr_bio_occurrence.storage.duckdb_index import create_qa_views
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset


def run_existing_payload_benchmark(
    *,
    payload_paths: Iterable[str | Path],
    output_dir: str | Path,
    species_name: str = "Papilio demoleus",
    region_id: str = "AU_ALL",
    target_records: int = 1000,
    vision_classifier: VisionClassifier | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tracemalloc.start()

    payload_items = _timed(timings, "raw_payload_read", lambda: _read_payloads(payload_paths))
    bronze = _timed(timings, "bronze_flattening_dedup", lambda: _build_bronze(payload_items, species_name, region_id, target_records))
    silver = _timed(timings, "silver_candidate_build", lambda: build_silver_candidates(bronze))
    vision_predictions = _timed(timings, "vision_classification", lambda: _build_vision_predictions(bronze, vision_classifier))
    gold = _timed(timings, "dwc_mapping", lambda: build_dwc_rows(silver))
    bronze_paths, silver_paths, vision_paths, gold_paths, duckdb_path = _timed(
        timings,
        "artifact_write",
        lambda: _write_outputs(output_path, bronze, silver, vision_predictions, gold),
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    files = [path for path in output_path.rglob("*") if path.is_file()]
    report = {
        "species": species_name,
        "region": region_id,
        "target_record_count": target_records,
        "actual_unique_records": bronze.height,
        "api_calls_made": 0,
        "raw_payload_files": len(payload_items),
        "step_timings_seconds": timings,
        "storage_artifacts": {
            "root": str(output_path),
            "bronze_parquet_files": len(bronze_paths),
            "silver_parquet_files": len(silver_paths),
            "silver_vision_prediction_parquet_files": len(vision_paths),
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
            "worker_count": 1,
            "cpu_count": os.cpu_count() or 1,
            "gpu_used": False,
            "vision_model_loaded": vision_classifier is not None,
        },
    }
    report_path = output_path / "existing_payload_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _read_payloads(payload_paths: Iterable[str | Path]) -> list[tuple[str, dict[str, object]]]:
    return [
        (Path(path).stem, json.loads(Path(path).read_text(encoding="utf-8")))
        for path in payload_paths
    ]


def _build_bronze(payload_items: list[tuple[str, dict[str, object]]], species_name: str, region_id: str, target_records: int) -> pl.DataFrame:
    frames = [
        flatten_search_payloads([payload], species_name=species_name, region_id=region_id, work_item_id=work_item_id)
        for work_item_id, payload in payload_items
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(subset=["flickr_photo_id"], keep="first").head(target_records)


def _build_vision_predictions(bronze: pl.DataFrame, vision_classifier: VisionClassifier | None) -> pl.DataFrame:
    if vision_classifier is None or not bronze.height:
        return pl.DataFrame()
    rows = [vision_classifier(row) for row in bronze.to_dicts() if row.get("image_url")]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _write_outputs(
    output_path: Path,
    bronze: pl.DataFrame,
    silver: pl.DataFrame,
    vision_predictions: pl.DataFrame,
    gold: pl.DataFrame,
) -> tuple[list[Path], list[Path], list[Path], list[Path], Path | None]:
    if not bronze.height:
        return [], [], [], [], None
    bronze_paths = write_parquet_dataset(bronze, output_path / "bronze" / "bronze_flickr_photo")
    silver_paths = write_parquet_dataset(silver, output_path / "silver" / "silver_occurrence_candidate")
    vision_paths = (
        write_parquet_dataset(vision_predictions, output_path / "silver" / "silver_vision_prediction")
        if vision_predictions.height
        else []
    )
    gold_outputs = export_dwc_records(gold, output_path / "gold")
    duckdb_path = create_qa_views(db_path=output_path / "existing_payload_benchmark.duckdb", data_root=output_path)
    return bronze_paths, silver_paths, vision_paths, gold_outputs.parquet_paths, duckdb_path
