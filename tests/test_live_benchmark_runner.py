from __future__ import annotations

import json
from pathlib import Path
import time

from flickr_bio_occurrence.benchmark.live_run import run_live_search_benchmark
from flickr_bio_occurrence.flickr.client import FlickrSearchResult
from flickr_bio_occurrence.flickr.work_items import WorkItem


def test_live_search_benchmark_writes_report_for_actual_records(tmp_path) -> None:
    calls: list[str] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        calls.append(work_item.work_item_id)
        photo_id = str(len(calls))
        payload = {"stat": "ok", "photos": {"photo": [{"id": photo_id, "title": "Papilio demoleus", "latitude": "-27", "longitude": "153"}]}}
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    report_path = run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "run",
        target_records=3,
        max_calls=10,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["target_record_count"] == 3
    assert report["actual_unique_records"] == 3
    assert report["work_items_called"] == 3
    assert report["storage_artifacts"]["bronze_parquet_files"] == 1
    assert report["storage_artifacts"]["duckdb_path"].endswith(".duckdb")


def test_live_search_benchmark_can_run_parallel_and_stop_at_target(tmp_path) -> None:
    calls: list[str] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        time.sleep(0.01)
        calls.append(work_item.work_item_id)
        photo_id = str(len(calls))
        payload = {"stat": "ok", "photos": {"photo": [{"id": photo_id, "title": "Papilio demoleus"}]}}
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    report_path = run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "parallel-run",
        target_records=8,
        max_calls=40,
        max_workers=4,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["actual_unique_records"] == 8
    assert report["work_items_called"] <= 12
    assert report["compute_artifacts"]["worker_count"] == 4


def test_live_search_benchmark_can_filter_query_variants(tmp_path) -> None:
    variants: list[str] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        variants.append(work_item.query_variant)
        photo_id = str(len(variants))
        payload = {"stat": "ok", "photos": {"photo": [{"id": photo_id, "title": "swallowtail"}]}}
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "variant-run",
        target_records=3,
        max_calls=10,
        query_variants=["swallowtail"],
    )

    assert variants == ["swallowtail", "swallowtail", "swallowtail"]


def test_live_search_benchmark_can_plan_pages(tmp_path) -> None:
    pages: list[int] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        pages.append(work_item.page)
        photo_id = str(len(pages))
        payload = {"stat": "ok", "photos": {"photo": [{"id": photo_id, "title": "swallowtail"}]}}
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "page-run",
        target_records=3,
        max_calls=10,
        query_variants=["swallowtail"],
        pages=range(1, 4),
    )

    assert pages == [1, 2, 3]
