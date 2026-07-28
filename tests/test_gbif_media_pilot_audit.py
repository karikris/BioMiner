from pathlib import Path
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_media_resolution.pipeline import PILOT_SELECTION_SCHEMA
from biominer.gbif_media_resolution.models import ATTEMPT_SCHEMA, RESULT_SCHEMA
from biominer.gbif_media_resolution.pilot_audit import (
    REVIEW_SCHEMA,
    prepare_pilot_execution_review,
    publish_pilot_execution_audit,
    publish_pilot_preflight_audit,
    write_pilot_execution_review,
    wilson_interval,
)


def test_wilson_interval_is_bounded_and_empty_aware() -> None:
    assert wilson_interval(0,0)==(None,None)
    low,high=wilson_interval(99,100)
    assert 0 < low < 0.99 < high <= 1


def test_pilot_preflight_retains_every_row_without_network_claims(tmp_path: Path) -> None:
    selection=tmp_path/"pilot.parquet"
    rows=[_row("1",False),_row("2",True)]
    pq.write_table(pa.Table.from_pylist(rows,schema=PILOT_SELECTION_SCHEMA),selection)
    import hashlib
    sha=hashlib.sha256(selection.read_bytes()).hexdigest()
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps({"work_rows":2,"source_artifact_sha256":"sha256:source","pilot_selection_artifact":{"physical_sha256":"sha256:"+sha}}))
    manifest=publish_pilot_preflight_audit(prepare_receipt=receipt,pilot_selection=selection,output_directory=tmp_path/"out",expected_rows=2,code_commit="deadbeef")
    assert manifest["overall_acceptance_status"]=="NOT_TESTED"
    assert manifest["counts"]["pending_manual_reviews"]==1
    assert manifest["network_requests"]==0


def test_pilot_execution_review_binds_results_to_selection(tmp_path: Path) -> None:
    selection = tmp_path / "pilot.parquet"
    results = tmp_path / "resolution_results.parquet"
    pq.write_table(
        pa.Table.from_pylist([_row("1", False), _row("2", True)], schema=PILOT_SELECTION_SCHEMA),
        selection,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _result("1", "resolved", terminal_reason=None),
                _result("2", "rights_blocked", terminal_reason="rights_policy"),
            ],
            schema=RESULT_SCHEMA,
        ),
        results,
    )

    review = prepare_pilot_execution_review(
        pilot_selection=selection,
        resolution_results=results,
    )

    assert review.num_rows == 2
    by_id = {row["gbifID"]: row for row in review.to_pylist()}
    assert by_id["1"]["resolver_status"] == "resolved"
    assert by_id["1"]["review_status"] == "PENDING"
    assert by_id["2"]["resolver_status"] == "rights_blocked"
    assert by_id["2"]["review_status"] == "NOT_APPLICABLE"

    output = tmp_path / "review.parquet"
    inventory = write_pilot_execution_review(
        pilot_selection=selection,
        resolution_results=results,
        output_path=output,
    )
    assert inventory["row_count"] == 2
    assert inventory["physical_sha256"] == "sha256:" + _sha256(output)
    with pytest.raises(FileExistsError):
        write_pilot_execution_review(
            pilot_selection=selection,
            resolution_results=results,
            output_path=output,
        )


def test_pilot_execution_audit_requires_review_and_reports_wilson_rates(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "pilot.parquet"
    resolution = tmp_path / "resolution"
    resolution.mkdir()
    selection_rows = [_row("1", False), _row("2", False), _row("3", True)]
    result_rows = [
        _result("1", "resolved", terminal_reason=None),
        _result("2", "unresolved_not_found", terminal_reason="http_status_404"),
        _result("3", "rights_blocked", terminal_reason="rights_policy"),
    ]
    pq.write_table(
        pa.Table.from_pylist(selection_rows, schema=PILOT_SELECTION_SCHEMA),
        selection,
    )
    pq.write_table(
        pa.Table.from_pylist(result_rows, schema=RESULT_SCHEMA),
        resolution / "resolution_results.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_attempt("1"), _attempt("2")],
            schema=ATTEMPT_SCHEMA,
        ),
        resolution / "resolution_attempts.parquet",
    )
    _write_resolution_manifest(resolution, result_rows, attempt_rows=2)
    review = prepare_pilot_execution_review(
        pilot_selection=selection,
        resolution_results=resolution / "resolution_results.parquet",
    )
    review_rows = review.to_pylist()
    review_rows[0].update(
        {
            "manual_category": "correctly_resolved_direct_image",
            "manual_direct_image_valid": True,
            "wrong_occurrence": False,
            "review_status": "REVIEWED",
            "reviewer": "fixture-reviewer",
            "reviewed_at": "2026-07-29T00:00:00Z",
        }
    )
    reviewed = tmp_path / "reviewed.parquet"
    pq.write_table(pa.Table.from_pylist(review_rows, schema=REVIEW_SCHEMA), reviewed)
    test_receipt = _write_test_receipt(tmp_path)

    manifest = publish_pilot_execution_audit(
        prepare_receipt=_write_prepare_receipt(tmp_path, selection, work_rows=3),
        pilot_selection=selection,
        resolution_directory=resolution,
        reviewed_pilot=reviewed,
        output_directory=tmp_path / "audit",
        expected_rows=3,
        code_commit="deadbeef",
        adapter_test_receipt=test_receipt,
    )

    assert manifest["overall_acceptance_status"] == "PASS"
    assert manifest["counts"]["reviewed_resolved_rows"] == 1
    assert manifest["metrics"]["manual_direct_image_precision"] == 1.0
    assert manifest["metrics"]["manual_direct_image_precision_wilson_95"][0] < 1.0
    assert manifest["validation"]["all_resolved_rows_reviewed"]
    assert (tmp_path / "audit" / "pilot_rates_by_provider.parquet").is_file()

    bad_rows = review.to_pylist()
    bad_rows[0].update(
        {
            "manual_category": "wrong_occurrence",
            "manual_direct_image_valid": False,
            "wrong_occurrence": True,
            "review_status": "REVIEWED",
            "reviewer": "fixture-reviewer",
            "reviewed_at": "2026-07-29T00:00:00Z",
        }
    )
    bad_review = tmp_path / "bad-reviewed.parquet"
    pq.write_table(pa.Table.from_pylist(bad_rows, schema=REVIEW_SCHEMA), bad_review)
    failed = publish_pilot_execution_audit(
        prepare_receipt=_write_prepare_receipt(tmp_path, selection, work_rows=3),
        pilot_selection=selection,
        resolution_directory=resolution,
        reviewed_pilot=bad_review,
        output_directory=tmp_path / "bad-audit",
        expected_rows=3,
        code_commit="deadbeef",
        adapter_test_receipt=test_receipt,
    )
    assert failed["overall_acceptance_status"] == "FAIL"
    assert (tmp_path / "bad-audit" / "manifest.json").is_file()


def _row(gbif_id,blocked):
    return {"source_row_id":"r"+gbif_id,"gbifID":gbif_id,"media_references":"https://example.org/"+gbif_id,"media_host":"example.org","host_population_rows":2,"host_size_band":"small","provider":"p","publisher":"p","dataset_name":"d","url_pattern":"extensionless_reference","license_state":"explicitly_restricted" if blocked else "item_media_license","reference_type":"html_or_unknown_reference","taxon_rank":"SPECIES","country_code":"AU","expected_adapter":"generic_structured_or_gbif","rights_blocked":blocked,"selection_stratum":"s","selection_hash":"sha256:h"+gbif_id}


def _result(
    gbif_id: str,
    status: str,
    *,
    terminal_reason: str | None,
) -> dict[str, object]:
    resolved = status == "resolved"
    return {
        "source_row_id": "r" + gbif_id,
        "source_artifact_sha256": "sha256:source",
        "gbif_id": gbif_id,
        "media_references": "https://example.org/" + gbif_id,
        "reference_host": "example.org",
        "media_type": "StillImage",
        "media_format": None,
        "media_license": "CC0",
        "occurrence_license": "CC0",
        "license_basis": "item_media_license",
        "status": status,
        "method": "structured_metadata" if resolved else "resolution_exhausted",
        "stable_candidate_url": (
            "https://example.org/image-" + gbif_id + ".jpg" if resolved else None
        ),
        "validated_final_url": (
            "https://example.org/image-" + gbif_id + ".jpg" if resolved else None
        ),
        "redirect_count": 0,
        "declared_content_type": "image/jpeg" if resolved else None,
        "detected_content_type": "image/jpeg" if resolved else None,
        "bytes_sampled": 32 if resolved else 0,
        "probe_prefix_sha256": "sha256:probe" if resolved else None,
        "content_sha256": None,
        "content_hash_status": "prefix_only" if resolved else "not_downloaded",
        "adapter_version": "fixture/v1",
        "attempt_count": 1 if status != "rights_blocked" else 0,
        "terminal_reason": terminal_reason,
        "resolved_at": "2026-07-29T00:00:00Z",
        "provenance_fingerprint": "sha256:provenance-" + gbif_id,
    }


def _attempt(gbif_id: str) -> dict[str, object]:
    return {
        "attempt_id": "attempt-" + gbif_id,
        "source_row_id": "r" + gbif_id,
        "sequence": 1,
        "phase": "reference",
        "method": "reference_probe",
        "requested_url": "https://example.org/" + gbif_id,
        "response_url": "https://example.org/" + gbif_id,
        "redirect_from": None,
        "status_code": 200,
        "outcome": "received",
        "error": None,
        "declared_content_type": "text/html",
        "response_prefix_sha256": "sha256:prefix",
        "response_byte_count": 10,
        "etag": None,
        "last_modified": None,
        "retry_number": 0,
        "started_at": "2026-07-29T00:00:00Z",
        "ended_at": "2026-07-29T00:00:01Z",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_resolution_manifest(
    directory: Path,
    result_rows: list[dict[str, object]],
    *,
    attempt_rows: int,
) -> None:
    result_path = directory / "resolution_results.parquet"
    attempt_path = directory / "resolution_attempts.parquet"
    status_counts: dict[str, int] = {}
    for row in result_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "schema_version": "biominer-gbif-media-url-resolution/v1",
        "input": {
            "mode": "pilot",
            "work_rows": len(result_rows),
            "source_artifact_sha256": "sha256:source",
        },
        "counts": {
            "result_rows": len(result_rows),
            "attempt_rows": attempt_rows,
            "status_counts": status_counts,
        },
        "artifacts": {
            "resolution_results.parquet": {
                "physical_sha256": "sha256:" + _sha256(result_path),
                "row_count": len(result_rows),
            },
            "resolution_attempts.parquet": {
                "physical_sha256": "sha256:" + _sha256(attempt_path),
                "row_count": attempt_rows,
            },
        },
        "validation": {
            "one_result_per_input": True,
            "unique_source_row_ids": True,
            "every_work_item_completed": True,
            "rights_blocked_zero_attempts": True,
            "all_parquet_row_groups_complete": True,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_prepare_receipt(
    directory: Path,
    selection: Path,
    *,
    work_rows: int,
) -> Path:
    path = directory / "prepare.json"
    path.write_text(
        json.dumps(
            {
                "work_rows": work_rows,
                "source_artifact_sha256": "sha256:source",
                "pilot_selection_artifact": {
                    "physical_sha256": "sha256:" + _sha256(selection),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_test_receipt(directory: Path) -> Path:
    path = directory / "test-receipt.json"
    path.write_text(
        json.dumps(
            {
                "command": "uv run pytest -q tests/test_gbif_media_url_resolution.py",
                "exit_code": 0,
                "tests_passed": 17,
            }
        ),
        encoding="utf-8",
    )
    return path
