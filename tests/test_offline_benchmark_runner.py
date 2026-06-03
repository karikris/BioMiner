from __future__ import annotations

import json

import polars as pl

from flickr_bio_occurrence.benchmark.offline_run import run_existing_payload_benchmark


def test_existing_payload_benchmark_reprocesses_raw_json_without_api_calls(tmp_path) -> None:
    payload_path = tmp_path / "raw" / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "stat": "ok",
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "title": "Papilio demoleus",
                            "latitude": "-27",
                            "longitude": "153",
                            "url_m": "https://live.staticflickr.com/example.jpg",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

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

    report_path = run_existing_payload_benchmark(
        payload_paths=[payload_path],
        output_dir=tmp_path / "offline-run",
        vision_classifier=fake_classifier,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["api_calls_made"] == 0
    assert report["actual_unique_records"] == 1
    assert report["storage_artifacts"]["silver_vision_prediction_parquet_files"] == 1
    assert report["compute_artifacts"]["vision_model_loaded"] is True


def test_existing_payload_benchmark_checkpoints_vision_predictions_and_resumes(tmp_path) -> None:
    payload_path = tmp_path / "raw" / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "stat": "ok",
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "title": "Papilio demoleus",
                            "latitude": "-27",
                            "longitude": "153",
                            "url_m": "https://live.staticflickr.com/1.jpg",
                        },
                        {
                            "id": "2",
                            "title": "Papilio demoleus",
                            "latitude": "-28",
                            "longitude": "152",
                            "url_m": "https://live.staticflickr.com/2.jpg",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_classifier(row: dict[str, object]) -> dict[str, object]:
        photo_id = str(row["flickr_photo_id"])
        calls.append(photo_id)
        return {
            "flickr_photo_id": photo_id,
            "model_family": "bioclip",
            "model_name": "imageomics/bioclip-2",
            "model_version": "bioclip2_5_huge",
            "model_checkpoint": "checkpoint",
            "model_hash": "sha256:test",
            "image_hash": f"sha256:image-{photo_id}",
            "image_url_used": row["image_url"],
            "top1_label": "a photo of Papilio demoleus",
            "top1_score": 0.9,
            "topk_json": [{"label": "a photo of Papilio demoleus", "score": 0.9}],
            "species_agreement_status": "exact_species_agreement",
            "vision_review_required": False,
            "created_at": "2026-06-03T00:00:00+00:00",
        }

    output_dir = tmp_path / "offline-run"
    first_report_path = run_existing_payload_benchmark(
        payload_paths=[payload_path],
        output_dir=output_dir,
        vision_classifier=fake_classifier,
    )
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    second_report_path = run_existing_payload_benchmark(
        payload_paths=[payload_path],
        output_dir=output_dir,
        vision_classifier=fake_classifier,
    )

    assert sorted(calls) == ["1", "2"]
    vision_files = sorted((output_dir / "silver" / "silver_vision_prediction").glob("*.parquet"))
    assert [path.name for path in vision_files] == ["flickr_photo_id=1.parquet", "flickr_photo_id=2.parquet"]
    predictions = pl.read_parquet(vision_files)
    assert predictions.select(pl.col("flickr_photo_id").n_unique()).item() == 2
    second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
    assert first_report["compute_artifacts"]["vision_predictions_completed"] == 2
    assert first_report["compute_artifacts"]["vision_predictions_skipped_existing"] == 0
    assert second_report["compute_artifacts"]["vision_predictions_completed"] == 2
    assert second_report["compute_artifacts"]["vision_predictions_skipped_existing"] == 2
