from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from flickr_bio_occurrence.cli import build_parser, run
from flickr_bio_occurrence.flickr.rate_limiter import FlickrRateLimiter
from flickr_bio_occurrence.pipeline.dry_run import build_dry_run_summary


def test_fetch_dry_run_reports_required_fields() -> None:
    summary = build_dry_run_summary(
        species="Papilio demoleus",
        region="AU_QLD",
        year=2024,
        month=1,
        config_path="config/pipeline.toml",
    )

    assert summary["planned_api_calls"] == 5
    assert summary["planned_maximum_photo_records"] == 1250
    assert summary["hourly_limit_status"] == "within_soft_cap"
    assert summary["work_item_count"] == 5
    assert summary["output_paths"]["raw"] == "data/raw/flickr/photos_search/"
    assert summary["vision_package"] == "BioCLIP functionality now lives in karikris/BioCLIPMiner"


def test_fetch_dry_run_can_plan_multiple_pages() -> None:
    summary = build_dry_run_summary(
        species="Papilio demoleus",
        region="AU_ALL",
        year=2024,
        month=1,
        config_path="config/pipeline.toml",
        pages=range(1, 4),
    )

    assert summary["planned_api_calls"] == 15
    assert summary["planned_maximum_photo_records"] == 3600
    assert summary["work_item_count"] == 15


def test_fetch_dry_run_cli_outputs_json(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch", "--species", "Papilio demoleus", "--region", "AU_QLD", "--year", "2024", "--month", "1", "--dry-run"])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["species"] == "Papilio demoleus"
    assert payload["planned_maximum_photo_records"] == 1250


def test_poll_once_cli_accepts_bounded_cycle_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["poll-once", "--max-api-calls", "3400"])

    assert args.command == "poll-once"
    assert args.max_api_calls == 3400


def test_cli_help_does_not_describe_old_gold_silver_bronze_logic(capsys) -> None:
    parser = build_parser()

    parser.print_help()
    help_text = capsys.readouterr().out

    assert "human_verified_bioclip_positive" not in help_text
    assert "human verification" not in help_text.casefold()
    assert "bioclip_positive_without_human_verification" not in help_text


def test_cli_module_execution_outputs_dry_run_json() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flickr_bio_occurrence.cli",
            "fetch",
            "--species",
            "Papilio demoleus",
            "--region",
            "AU_QLD",
            "--year",
            "2024",
            "--month",
            "1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout)["planned_api_calls"] == 5


def test_qa_rate_limit_outputs_limiter_status_json(tmp_path, capsys) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite")
    limiter.acquire_api_token("flickr.photos.search", "work-1")
    limiter.log_call("flickr.photos.search", "work-1", "ok")
    limiter.log_photo_records(["1", "2"], "work-1")
    parser = build_parser()
    args = parser.parse_args(["qa-rate-limit", "--ledger-path", str(tmp_path / "limits.sqlite")])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["api_calls_in_window"] == 1
    assert payload["photo_records_in_window"] == 2
    assert payload["soft_api_calls_per_hour"] == 3200
    assert payload["hard_api_calls_per_hour"] == 3600


def test_qa_summary_outputs_benchmark_report_summary(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "species": "Papilio demoleus",
                "actual_unique_records": 16,
                "api_calls_made": 0,
                "step_timings_seconds": {"vision_classification": 84.9},
                "storage_artifacts": {"total_artifact_bytes": 1234},
                "memory_artifacts": {"peak_traced_bytes": 4567},
                "compute_artifacts": {"vision_model_loaded": True},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["qa-summary", "--report", str(report_path)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["species"] == "Papilio demoleus"
    assert payload["actual_unique_records"] == 16
    assert payload["vision_model_loaded"] is True
    assert payload["total_artifact_bytes"] == 1234


def test_benchmark_existing_payloads_cli_runs_offline_benchmark(tmp_path, capsys) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "payload.json").write_text(
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
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-existing-payloads",
            "--raw-root",
            str(raw_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--target-records",
            "1000",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["api_calls_made"] == 0
    assert payload["actual_unique_records"] == 1
    assert payload["target_record_count"] == 1000
    assert Path(payload["report"]).exists()


def test_evidence_first_cli_commands_build_classify_apply_and_compact(tmp_path, capsys) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "payload.json").write_text(
        json.dumps(
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "title": "Papilio demoleus verified by expert",
                            "url_l": "https://live.staticflickr.com/large.jpg",
                            "comments": {"comment": [{"_content": "confirmed by reviewer"}]},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    evidence_path = tmp_path / "evidence.parquet"
    queue_path = tmp_path / "queue.sqlite"
    args = parser.parse_args(
        [
            "build-evidence",
            "--raw-root",
            str(raw_root),
            "--output",
            str(evidence_path),
            "--queue-path",
            str(queue_path),
        ]
    )

    assert run(args) == 0
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["evidence_rows"] == 1
    assert build_payload["classification_job_id"]

    args = parser.parse_args(
        [
            "classify-once",
            "--queue-path",
            str(queue_path),
            "--prediction-output-dir",
            str(tmp_path / "predictions"),
            "--fake-classifier",
        ]
    )
    assert run(args) == 0
    classify_payload = json.loads(capsys.readouterr().out)
    assert classify_payload["processed_jobs"] == 1
    assert classify_payload["prediction_rows"] == 1

    classified_path = tmp_path / "classified.parquet"
    args = parser.parse_args(["apply-rules", "--evidence", str(evidence_path), "--output", str(classified_path)])
    assert run(args) == 0
    rules_payload = json.loads(capsys.readouterr().out)
    assert rules_payload["rows"] == 1
    assert sum(rules_payload["publication_state_counts"].values()) == 1
    assert rules_payload["in_review_without_reason"] == 0

    compacted_path = tmp_path / "compacted.parquet"
    args = parser.parse_args(["compact-parquet", "--input-root", str(tmp_path / "predictions"), "--output", str(compacted_path)])
    assert run(args) == 0
    compact_payload = json.loads(capsys.readouterr().out)
    assert compact_payload["input_parquet_files"] == 1
    assert compact_payload["rows"] == 1
    assert compacted_path.exists()


def test_fetch_live_dry_run_and_comments_audit_cli(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "fetch-live",
            "--species",
            "Papilio demoleus",
            "--region",
            "AU_QLD",
            "--year",
            "2024",
            "--month",
            "1",
            "--dry-run",
        ]
    )

    assert run(args) == 0
    assert json.loads(capsys.readouterr().out)["planned_api_calls"] == 5

    args = parser.parse_args(["fetch-comments", "--photo-id", "1"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["implemented"] is False
    assert payload["photo_ids_requested"] == ["1"]


def test_classify_watch_skips_completed_jobs_and_gc_cache_reports(tmp_path, capsys) -> None:
    parser = build_parser()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "payload.json").write_text(
        json.dumps({"photos": {"photo": [{"id": "1", "title": "Papilio demoleus", "url_l": "https://live.staticflickr.com/large.jpg"}]}}),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.sqlite"
    run(
        parser.parse_args(
            [
                "build-evidence",
                "--raw-root",
                str(raw_root),
                "--output",
                str(tmp_path / "evidence.parquet"),
                "--queue-path",
                str(queue_path),
            ]
        )
    )
    capsys.readouterr()
    args = parser.parse_args(
        [
            "classify-watch",
            "--queue-path",
            str(queue_path),
            "--prediction-output-dir",
            str(tmp_path / "predictions"),
            "--limit",
            "10",
            "--fake-classifier",
        ]
    )
    assert run(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["processed_jobs"] == 1
    assert run(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["processed_jobs"] == 0

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "image.jpg").write_bytes(b"abc")
    args = parser.parse_args(["gc-cache", "--cache-root", str(cache_root), "--delete"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_seen"] == 1
    assert payload["deleted_files"] == 1
