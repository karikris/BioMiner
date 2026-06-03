from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

import flickr_bio_occurrence.benchmark.live_run as live_run
from flickr_bio_occurrence.benchmark.live_run import DEFAULT_LIVE_TEST_API_CALL_CAP, run_live_search_benchmark
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


def test_live_search_benchmark_defaults_to_100_api_calls_per_test(tmp_path) -> None:
    calls: list[str] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        calls.append(work_item.work_item_id)
        photo_id = str(len(calls))
        payload = {"stat": "ok", "photos": {"photo": [{"id": photo_id, "title": "Papilio demoleus"}]}}
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    report_path = run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "default-cap-run",
        target_records=1000,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert DEFAULT_LIVE_TEST_API_CALL_CAP == 100
    assert report["max_calls"] == 100
    assert report["work_items_called"] == 100


def test_live_search_benchmark_writes_optional_vision_predictions(tmp_path) -> None:
    calls: list[str] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        calls.append(work_item.work_item_id)
        photo_id = str(len(calls))
        payload = {
            "stat": "ok",
            "photos": {
                "photo": [
                    {
                        "id": photo_id,
                        "title": "Papilio demoleus",
                        "url_m": "https://live.staticflickr.com/example.jpg",
                    }
                ]
            },
        }
        raw_path = tmp_path / f"{photo_id}.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    def fake_classifier(row: dict[str, object]) -> dict[str, object]:
        return {
            "flickr_photo_id": row["flickr_photo_id"],
            "model_family": "bioclip",
            "model_name": "imageomics/bioclip-2",
            "model_version": "bioclip2_5_huge",
            "model_checkpoint": "checkpoint",
            "model_hash": "sha256:test",
            "image_hash": "sha256:image",
            "image_url_used": row["image_url"],
            "top1_label": "a photo of Papilio demoleus",
            "top1_score": 0.9,
            "topk_json": [{"label": "a photo of Papilio demoleus", "score": 0.9}],
            "species_agreement_status": "exact_species_agreement",
            "vision_review_required": False,
            "created_at": "2026-06-03T00:00:00+00:00",
        }

    report_path = run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "vision-run",
        target_records=2,
        max_calls=5,
        vision_classifier=fake_classifier,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["storage_artifacts"]["silver_vision_prediction_parquet_files"] == 1
    assert report["compute_artifacts"]["vision_model_loaded"] is True
    assert "vision_classification" in report["step_timings_seconds"]
    assert list((tmp_path / "vision-run" / "silver" / "silver_vision_prediction").rglob("*.parquet"))


def test_live_search_benchmark_can_build_bioclip_vision_classifier(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_search(work_item: WorkItem) -> FlickrSearchResult:
        photo_id = "1"
        payload = {
            "stat": "ok",
            "photos": {"photo": [{"id": photo_id, "title": "Papilio demoleus", "url_m": "https://live.staticflickr.com/example.jpg"}]},
        }
        raw_path = tmp_path / "1.json"
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=[photo_id])

    def fake_factory(*, model_registry_path: str | Path, cache_root: str | Path):
        calls.append({"model_registry_path": model_registry_path, "cache_root": cache_root})

        def classify(row: dict[str, object]) -> dict[str, object]:
            return {
                "flickr_photo_id": row["flickr_photo_id"],
                "model_family": "bioclip",
                "model_name": "imageomics/bioclip-2",
                "model_version": "bioclip2_5_huge",
                "model_checkpoint": "checkpoint",
                "model_hash": "sha256:test",
                "image_hash": "sha256:image",
                "image_url_used": row["image_url"],
                "top1_label": "a photo of Papilio demoleus",
                "top1_score": 0.9,
                "topk_json": [{"label": "a photo of Papilio demoleus", "score": 0.9}],
                "species_agreement_status": "exact_species_agreement",
                "vision_review_required": False,
                "created_at": "2026-06-03T00:00:00+00:00",
            }

        return classify

    monkeypatch.setattr(live_run, "build_bioclip_row_classifier", fake_factory)

    report_path = run_live_search_benchmark(
        search_photos=fake_search,
        output_dir=tmp_path / "bioclip-run",
        target_records=1,
        max_calls=1,
        use_bioclip_vision=True,
        model_registry_path="config/model_registry.toml",
        image_cache_root=tmp_path / "cache",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == [{"model_registry_path": "config/model_registry.toml", "cache_root": tmp_path / "cache"}]
    assert report["compute_artifacts"]["vision_model_loaded"] is True
    assert report["storage_artifacts"]["silver_vision_prediction_parquet_files"] == 1


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
