from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from flickr_bio_occurrence.cli import build_parser, run
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
    assert summary["selected_bioclip_model"] == "bioclip2"


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
