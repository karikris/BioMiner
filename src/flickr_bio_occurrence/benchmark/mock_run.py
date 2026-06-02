from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
import tracemalloc
from typing import Callable, TypeVar

import polars as pl

from flickr_bio_occurrence.dwc.exporter import export_dwc_records
from flickr_bio_occurrence.dwc.mapper import map_candidate_to_dwc
from flickr_bio_occurrence.storage.duckdb_index import create_qa_views
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset


T = TypeVar("T")


def run_mock_1000_record_benchmark(*, output_dir: str | Path, species: str, region: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tracemalloc.start()

    records = _timed(timings, "mock_metadata_generation", lambda: _mock_records(species, 1000))
    bronze = _timed(timings, "bronze_flattening", lambda: [dict(record) for record in records])
    silver = _timed(timings, "silver_candidate_build", lambda: [_to_candidate(record, species) for record in bronze])
    dwc_rows = _timed(timings, "dwc_mapping", lambda: [map_candidate_to_dwc(candidate) for candidate in silver])
    bronze_paths = write_parquet_dataset(pl.DataFrame(bronze), output_path / "bronze" / "bronze_flickr_photo")
    silver_paths = write_parquet_dataset(pl.DataFrame(silver), output_path / "silver" / "silver_occurrence_candidate")
    gold_outputs = export_dwc_records(pl.DataFrame(dwc_rows), output_path / "gold")
    duckdb_path = create_qa_views(
        db_path=output_path / "mock_1000_record_benchmark.duckdb",
        data_root=output_path,
    )
    report_path = output_path / "mock_1000_record_benchmark.json"

    write_start = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    timings["artifact_write"] = 0.0
    report = {
        "species": species,
        "region": region,
        "record_count": len(dwc_rows),
        "step_timings_seconds": timings,
            "storage_artifacts": {
                "metrics_json": str(report_path),
                "mock_dwc_rows_in_memory": len(dwc_rows),
                "bronze_parquet_files": len(bronze_paths),
                "silver_parquet_files": len(silver_paths),
                "gold_parquet_files": len(gold_outputs.parquet_paths),
                "duckdb_path": str(duckdb_path),
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
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    timings["artifact_write"] = time.perf_counter() - write_start
    report["step_timings_seconds"] = timings
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tracemalloc.stop()
    return report_path


def _timed(timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = fn()
    timings[name] = time.perf_counter() - start
    return result


def _mock_records(species: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "flickr_photo_id": str(index),
            "title": species,
            "date_taken": "2024-01-15",
            "latitude": -27.0 - (index % 100) / 1000,
            "longitude": 153.0 + (index % 100) / 1000,
            "photo_url": f"https://www.flickr.com/photos/example/{index}",
            "license": "cc-by",
            "owner_name": "mock_owner",
        }
        for index in range(count)
    ]


def _to_candidate(record: dict[str, object], species: str) -> dict[str, object]:
    return {
        "flickr_photo_id": record["flickr_photo_id"],
        "resolved_scientific_name": species,
        "eventDate": record["date_taken"],
        "decimalLatitude": record["latitude"],
        "decimalLongitude": record["longitude"],
        "verbatimIdentification": record["title"],
        "identificationVerificationStatus": "machine_suggested",
        "associatedReferences": record["photo_url"],
        "license": record["license"],
        "rightsHolder": record["owner_name"],
        "human_evidence": True,
        "dynamicProperties": {"mock_benchmark": True},
    }
