from __future__ import annotations

import json
from pathlib import Path

import pytest

from biominer.gbif_final.telemetry import (
    BoundedRunTelemetry,
    FAILURE_RECEIPT,
    SUCCESS_RECEIPT,
    validate_run_receipt,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fixture/v1",
                "counts": {"rows": 3},
                "validation": {"pass": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_bounded_run_telemetry_seals_success_and_detects_tampering(
    tmp_path: Path,
) -> None:
    output_manifest = _manifest(tmp_path / "manifest.json")
    observed: list[dict[str, object]] = []
    telemetry = BoundedRunTelemetry(
        root_directory=tmp_path / "telemetry",
        producer_git_sha="deadbeef",
        config={"memory_limit": "8GB", "threads": 2},
        run_id="fixture-success",
        event_sink=lambda event: observed.append(dict(event)),
    )
    telemetry.emit(
        "partition_completed",
        stage="global_sidecar",
        partition=0,
        rows_read=3,
        rows_written=3,
        rows_passed=3,
        rows_failed=0,
        rows_unresolved=0,
        rows_skipped_from_cache=0,
        requests_completed=0,
        retries=0,
        rate_limit_events=0,
        bytes_downloaded=0,
        network_scope="NOT_APPLICABLE",
        checkpoint_path="/tmp/checkpoint",
    )
    receipt = telemetry.finish(
        output_manifest=output_manifest,
        rows=3,
        resumed_output=False,
    )
    receipt_path = (
        telemetry.invocation_directory / SUCCESS_RECEIPT
    )

    assert receipt["status"] == "completed"
    assert receipt["event_log"]["event_count"] == 3
    assert receipt["output"]["rows"] == 3
    assert receipt["metrics"]["peak_rss_bytes"] > 0
    assert [event["event"] for event in observed] == [
        "run_started",
        "partition_completed",
        "run_completed",
    ]
    assert (
        validate_run_receipt(
            receipt_path,
            expected_output_manifest=output_manifest,
        )
        == receipt
    )

    event_log = telemetry.event_log
    event_log.write_text(
        event_log.read_text(encoding="utf-8").replace(
            "partition_completed",
            "partition_corrupted",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="event log checksum mismatch",
    ):
        validate_run_receipt(receipt_path)


def test_bounded_run_telemetry_seals_failure_create_only(
    tmp_path: Path,
) -> None:
    telemetry = BoundedRunTelemetry(
        root_directory=tmp_path / "telemetry",
        producer_git_sha="deadbeef",
        config={"memory_limit": "8GB"},
        run_id="fixture-failure",
    )
    failure = telemetry.fail(
        RuntimeError("fixture failure"),
        stage="species_dimension",
        partition=7,
    )
    failure_path = (
        telemetry.invocation_directory / FAILURE_RECEIPT
    )

    assert failure["status"] == "failed"
    assert failure["failure"] == {
        "stage": "species_dimension",
        "partition": 7,
        "error_type": "RuntimeError",
        "error_message": "fixture failure",
    }
    assert validate_run_receipt(failure_path) == failure
    with pytest.raises(RuntimeError, match="already closed"):
        telemetry.emit("late_event")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BoundedRunTelemetry(
            root_directory=tmp_path / "telemetry",
            producer_git_sha="deadbeef",
            config={},
            run_id="fixture-failure",
        )
