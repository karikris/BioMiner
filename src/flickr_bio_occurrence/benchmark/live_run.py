from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
import tracemalloc
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date
from typing import Callable

import polars as pl

from flickr_bio_occurrence.dwc.exporter import export_dwc_records
from flickr_bio_occurrence.flickr.client import FlickrSearchResult
from flickr_bio_occurrence.flickr.work_items import WorkItem, build_monthly_work_items
from flickr_bio_occurrence.pipeline.transforms import build_dwc_rows, build_silver_candidates, flatten_search_payloads
from flickr_bio_occurrence.storage.duckdb_index import create_qa_views
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset
from flickr_bio_occurrence.taxonomy.species_mapper import get_seed_species
from flickr_bio_occurrence.vision.pipeline import build_bioclip_row_classifier


SearchPhotos = Callable[[WorkItem], FlickrSearchResult]
VisionClassifier = Callable[[dict[str, object]], dict[str, object]]
DEFAULT_LIVE_TEST_API_CALL_CAP = 100
DEFAULT_SOFT_API_CALLS_PER_HOUR = 3200
DEFAULT_HARD_API_CALLS_PER_HOUR = 3600
DEFAULT_HARD_PHOTO_RECORDS_PER_HOUR = 3600


def run_live_search_benchmark(
    *,
    search_photos: SearchPhotos,
    output_dir: str | Path,
    target_records: int = 1000,
    max_calls: int = DEFAULT_LIVE_TEST_API_CALL_CAP,
    species_name: str = "Papilio demoleus",
    regions: list[tuple[str, str, str]] | None = None,
    years: list[int] | None = None,
    months: range = range(1, 13),
    max_workers: int = 1,
    query_variants: list[str] | None = None,
    pages: range | None = None,
    end_date: date | None = None,
    excluded_work_item_ids: set[str] | None = None,
    vision_classifier: VisionClassifier | None = None,
    use_bioclip_vision: bool = False,
    model_registry_path: str | Path = "config/model_registry.toml",
    image_cache_root: str | Path = "data/cache/images",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tracemalloc.start()
    effective_vision_classifier = vision_classifier
    if effective_vision_classifier is None and use_bioclip_vision:
        effective_vision_classifier = build_bioclip_row_classifier(
            model_registry_path=model_registry_path,
            cache_root=image_cache_root,
        )
    work_items = build_monthly_work_items(
        species=get_seed_species(species_name),
        regions=regions or [("AU_ALL", "Australia", "112.92,-43.74,153.64,-10.05")],
        years=years or [2024, 2023, 2022, 2021, 2020],
        months=months,
        query_variants=query_variants,
        pages=pages,
        end_date=end_date,
    )
    excluded_ids = excluded_work_item_ids or set()
    scheduled_work_items = [
        item
        for item in work_items
        if item.work_item_id not in excluded_ids
    ]
    payloads: list[tuple[WorkItem, dict[str, object]]] = []
    seen: set[str] = set()
    errors: list[dict[str, str]] = []
    timings["flickr_fetch"] = _fetch_payloads(
        search_photos=search_photos,
        work_items=scheduled_work_items[:max_calls],
        target_records=target_records,
        max_workers=max_workers,
        payloads=payloads,
        seen=seen,
        errors=errors,
    )
    bronze = _timed(timings, "bronze_flattening_dedup", lambda: _build_bronze(payloads, species_name, target_records))
    silver = _timed(timings, "silver_candidate_build", lambda: build_silver_candidates(bronze))
    vision_predictions = _timed(timings, "vision_classification", lambda: _build_vision_predictions(bronze, effective_vision_classifier))
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
        "region": "Australia",
        "target_record_count": target_records,
        "actual_unique_records": bronze.height,
        "work_items_planned": len(work_items),
        "work_items_skipped_as_existing": len(work_items) - len(scheduled_work_items),
        "work_items_called": len(payloads) + len(errors),
        "max_calls": max_calls,
        "api_policy": {
            "per_test_api_call_cap": DEFAULT_LIVE_TEST_API_CALL_CAP,
            "soft_api_calls_per_hour": DEFAULT_SOFT_API_CALLS_PER_HOUR,
            "hard_api_calls_per_hour": DEFAULT_HARD_API_CALLS_PER_HOUR,
            "hard_photo_records_per_hour": DEFAULT_HARD_PHOTO_RECORDS_PER_HOUR,
            "effective_max_calls_for_run": max_calls,
            "end_date": end_date.isoformat() if end_date else None,
        },
        "errors": errors[:20],
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
            "worker_count": max_workers,
            "cpu_count": os.cpu_count() or 1,
            "gpu_used": False,
            "vision_model_loaded": effective_vision_classifier is not None,
            "http_client": "httpx",
            "rate_limiter_scope": "caller_supplied_global_limiter_required",
        },
    }
    report_path = output_path / "live_search_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _fetch_payloads(
    *,
    search_photos: SearchPhotos,
    work_items: list[WorkItem],
    target_records: int,
    max_workers: int,
    payloads: list[tuple[WorkItem, dict[str, object]]],
    seen: set[str],
    errors: list[dict[str, str]],
) -> float:
    start = time.perf_counter()
    if max_workers <= 1:
        for item in work_items:
            if len(seen) >= target_records:
                break
            _fetch_one(search_photos, item, payloads, seen, errors)
        return time.perf_counter() - start

    pending: set[Future[tuple[WorkItem, FlickrSearchResult]]] = set()
    iterator = iter(work_items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while len(pending) < max_workers and len(seen) < target_records:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(_call_search, search_photos, item))
        while pending and len(seen) < target_records:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    item, result = future.result()
                except Exception as exc:  # noqa: BLE001 - benchmark report must preserve failures and continue.
                    errors.append({"work_item_id": "unknown", "error": type(exc).__name__, "message": str(exc)[:300]})
                    continue
                payloads.append((item, _payload_for_reserved_photo_ids(result.payload, result.photo_ids)))
                seen.update(result.photo_ids)
            while len(pending) < max_workers and len(seen) < target_records:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.add(executor.submit(_call_search, search_photos, item))
        for future in pending:
            future.cancel()
    return time.perf_counter() - start


def _fetch_one(
    search_photos: SearchPhotos,
    item: WorkItem,
    payloads: list[tuple[WorkItem, dict[str, object]]],
    seen: set[str],
    errors: list[dict[str, str]],
) -> None:
    try:
        result = search_photos(item)
    except Exception as exc:  # noqa: BLE001 - benchmark report must preserve failures and continue.
        errors.append({"work_item_id": item.work_item_id, "error": type(exc).__name__, "message": str(exc)[:300]})
        return
    payloads.append((item, _payload_for_reserved_photo_ids(result.payload, result.photo_ids)))
    seen.update(result.photo_ids)


def _call_search(search_photos: SearchPhotos, item: WorkItem) -> tuple[WorkItem, FlickrSearchResult]:
    return item, search_photos(item)


def _build_bronze(payloads: list[tuple[WorkItem, dict[str, object]]], species_name: str, target_records: int) -> pl.DataFrame:
    frames = [
        flatten_search_payloads([payload], species_name=species_name, region_id=item.region_id, work_item_id=item.work_item_id)
        for item, payload in payloads
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(subset=["flickr_photo_id"], keep="first").head(target_records)


def _payload_for_reserved_photo_ids(payload: dict[str, object], photo_ids: list[str]) -> dict[str, object]:
    allowed = set(photo_ids)
    photos = payload.get("photos")
    if not isinstance(photos, dict):
        return payload
    photo_rows = photos.get("photo")
    if not isinstance(photo_rows, list):
        return payload
    filtered_photos = [
        photo
        for photo in photo_rows
        if isinstance(photo, dict) and str(photo.get("id", "")) in allowed
    ]
    return {
        **payload,
        "photos": {
            **photos,
            "photo": filtered_photos,
        },
    }


def _build_vision_predictions(bronze: pl.DataFrame, vision_classifier: VisionClassifier | None) -> pl.DataFrame:
    if vision_classifier is None or not bronze.height:
        return pl.DataFrame()
    rows = [
        vision_classifier(row)
        for row in bronze.to_dicts()
        if row.get("image_url")
    ]
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
    duckdb_path = create_qa_views(db_path=output_path / "live_search_benchmark.duckdb", data_root=output_path)
    return bronze_paths, silver_paths, vision_paths, gold_outputs.parquet_paths, duckdb_path


def _timed[T](timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = fn()
    timings[name] = time.perf_counter() - start
    return result
