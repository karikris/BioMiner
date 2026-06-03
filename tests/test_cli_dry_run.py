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
        model_registry_path="config/model_registry.toml",
    )

    assert summary["planned_api_calls"] == 5
    assert summary["planned_maximum_photo_records"] == 1250
    assert summary["hourly_limit_status"] == "within_soft_cap"
    assert summary["work_item_count"] == 5
    assert summary["output_paths"]["raw"] == "data/raw/flickr/photos_search/"
    assert summary["selected_bioclip_model"] == "bioclip2_5_huge"
    assert summary["selected_bioclip_runtime"]["package_name"] == "open_clip_torch"
    assert "available" in summary["selected_bioclip_runtime"]


def test_fetch_dry_run_can_plan_multiple_pages() -> None:
    summary = build_dry_run_summary(
        species="Papilio demoleus",
        region="AU_ALL",
        year=2024,
        month=1,
        config_path="config/pipeline.toml",
        model_registry_path="config/model_registry.toml",
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
    assert payload["soft_api_calls_per_hour"] == 3000
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
