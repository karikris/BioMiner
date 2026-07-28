from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.pipeline import (
    FINAL_FILENAME,
    FINAL_SCHEMA_VERSION,
    MANIFEST_FILENAME,
)
from biominer.gbif_final.resolution_enrichment import (
    RESOLUTION_ENRICHMENT_VERSION,
    enrich_final_with_resolutions,
    validate_resolution_enriched_publication,
)
from biominer.gbif_media_resolution.models import (
    ATTEMPT_SCHEMA,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    source_row_id,
)


SOURCE_SHA256 = "sha256:" + "a" * 64


def test_enrichment_retains_resolved_unresolved_and_rights_blocked_rows(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)

    manifest = enrich_final_with_resolutions(**values)
    output = Path(str(values["output_directory"]))
    table = pq.read_table(output / FINAL_FILENAME)
    rows = table.to_pylist()

    assert manifest["enrichment_contract"] == RESOLUTION_ENRICHMENT_VERSION
    assert manifest["counts"]["base_rows"] == 5
    assert manifest["counts"]["output_rows"] == 5
    assert manifest["counts"]["resolution_rows"] == 4
    assert manifest["counts"]["matched_resolution_rows"] == 3
    assert manifest["counts"]["unmatched_resolution_rows"] == 1
    assert all(manifest["acceptance_gate"].values())
    assert [row["gbifID"] for row in rows] == [
        "direct",
        "resolved",
        "unresolved",
        "blocked",
        "no-reference",
    ]
    assert [row["media_identifier_resolution_status"] for row in rows] == [
        "source_identifier",
        "resolved",
        "unresolved_not_found",
        "rights_blocked",
        "missing_reference_not_selected",
    ]
    assert rows[0]["effective_media_identifier"] == (
        "https://images.example/direct.jpg"
    )
    assert rows[1]["resolved_media_identifier"] == (
        "https://images.example/resolved.jpg"
    )
    assert rows[1]["effective_media_identifier"] == (
        "https://images.example/resolved.jpg"
    )
    assert rows[2]["effective_media_identifier"] is None
    assert rows[3]["effective_media_identifier"] is None
    assert rows[3]["media_identifier_resolution_id"] is not None
    assert rows[4]["media_identifier_resolution_id"] is None
    assert {
        path.name for path in output.iterdir() if path.is_file()
    } == {FINAL_FILENAME, MANIFEST_FILENAME}
    assert (
        validate_resolution_enriched_publication(
            output,
            base_publication_directory=values[
                "base_publication_directory"
            ],
            resolution_directory=values["resolution_directory"],
            repository_root=values["repository_root"],
        )
        == manifest
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        enrich_final_with_resolutions(**values)


def test_enrichment_fails_closed_when_base_reference_has_no_result(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    resolution = Path(str(values["resolution_directory"]))
    table = pq.read_table(
        resolution / "resolution_results.parquet"
    ).filter(
        pa.compute.not_equal(
            pq.read_table(
                resolution / "resolution_results.parquet",
                columns=["gbif_id"],
            )["gbif_id"],
            "unresolved",
        )
    )
    _rewrite_resolution_publication(resolution, table)
    values["expected_resolution_rows"] = 3

    with pytest.raises(
        RuntimeError,
        match="missing terminal resolution result",
    ):
        enrich_final_with_resolutions(**values)
    assert not Path(str(values["output_directory"])).exists()


def test_enrichment_rejects_tampered_resolution_publication(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    result_path = (
        Path(str(values["resolution_directory"]))
        / "resolution_results.parquet"
    )
    result_path.write_bytes(result_path.read_bytes() + b"tampered")

    with pytest.raises(
        RuntimeError,
        match="cannot inspect Parquet artifact",
    ):
        enrich_final_with_resolutions(**values)


def test_enrichment_rejects_non_http_resolved_candidate(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    resolution = Path(str(values["resolution_directory"]))
    table = pq.read_table(resolution / "resolution_results.parquet")
    rows = table.to_pylist()
    for row in rows:
        if row["gbif_id"] == "resolved":
            row["stable_candidate_url"] = "file:///tmp/not-an-image"
    _rewrite_resolution_publication(
        resolution,
        pa.Table.from_pylist(rows, schema=RESULT_SCHEMA),
    )

    with pytest.raises(
        RuntimeError,
        match="invalid direct URL",
    ):
        enrich_final_with_resolutions(**values)


def test_validator_rejects_changed_final_parquet(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    enrich_final_with_resolutions(**values)
    output = Path(str(values["output_directory"]))
    final_path = output / FINAL_FILENAME
    final_path.write_bytes(final_path.read_bytes() + b"changed")

    with pytest.raises(
        RuntimeError,
        match="cannot inspect Parquet artifact",
    ):
        validate_resolution_enriched_publication(output)


def _fixture(tmp_path: Path) -> dict[str, object]:
    repository = Path(__file__).parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base = tmp_path / "base"
    base.mkdir()
    base_table = pa.table(
        {
            "gbifID": [
                "direct",
                "resolved",
                "unresolved",
                "blocked",
                "no-reference",
            ],
            "license": [
                "CC BY 4.0",
                "CC BY 4.0",
                "CC0",
                "CC BY-NC",
                None,
            ],
            "media_identifier": [
                "https://images.example/direct.jpg",
                None,
                None,
                None,
                None,
            ],
            "media_references": [
                "https://records.example/direct",
                "https://records.example/resolved",
                "https://records.example/unresolved",
                "https://records.example/blocked",
                None,
            ],
            "media_license": [
                "CC BY 4.0",
                None,
                "CC0",
                "All rights reserved",
                None,
            ],
            "source_row_id": [
                "source-direct",
                "source-resolved",
                "source-unresolved",
                "source-blocked",
                "source-no-reference",
            ],
            "species": [
                "One",
                "Two",
                "Three",
                "Four",
                "Five",
            ],
        }
    )
    base_path = base / FINAL_FILENAME
    pq.write_table(
        base_table,
        base_path,
        compression="zstd",
        row_group_size=2,
    )
    base_manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "producer_git_sha": commit,
        "artifact": _legacy_inventory(base_path),
        "inputs": {
            "temporal_parquet": {
                "path": "/fixture/temporal.parquet",
                "sha256": "sha256:temporal",
            },
            "pre_temporal_parquet": {
                "path": "/fixture/pre-temporal.parquet",
                "sha256": "sha256:pre-temporal",
            },
            "registry_dir": "/fixture/registry",
            "quality_dir": "/fixture/quality",
        },
        "acceptance_gate": {
            "row_count_preserved": True,
            "stable_media_identity_complete": True,
            "row_groups_complete": True,
            "manifest_written_last": True,
        },
    }
    (base / MANIFEST_FILENAME).write_text(
        json.dumps(base_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolution = tmp_path / "resolution"
    resolution.mkdir()
    result_rows = [
        _result(
            gbif_id="resolved",
            reference="https://records.example/resolved",
            status="resolved",
            stable_url="https://images.example/resolved.jpg",
        ),
        _result(
            gbif_id="unresolved",
            reference="https://records.example/unresolved",
            status="unresolved_not_found",
        ),
        _result(
            gbif_id="blocked",
            reference="https://records.example/blocked",
            status="rights_blocked",
            media_license="All rights reserved",
        ),
        _result(
            gbif_id="pre-1960-unmatched",
            reference="https://records.example/pre-1960",
            status="resolved",
            stable_url="https://images.example/pre-1960.jpg",
        ),
    ]
    result_table = pa.Table.from_pylist(
        result_rows,
        schema=RESULT_SCHEMA,
    )
    _rewrite_resolution_publication(resolution, result_table)
    return {
        "base_publication_directory": base,
        "resolution_directory": resolution,
        "output_directory": tmp_path / "output",
        "repository_root": repository,
        "producer_git_sha": commit,
        "expected_resolution_rows": 4,
        "batch_rows": 2,
        "row_group_rows": 2,
    }


def _result(
    *,
    gbif_id: str,
    reference: str,
    status: str,
    stable_url: str | None = None,
    media_license: str | None = None,
) -> dict[str, object]:
    identity = source_row_id(SOURCE_SHA256, gbif_id, reference)
    return {
        "source_row_id": identity,
        "source_artifact_sha256": SOURCE_SHA256,
        "gbif_id": gbif_id,
        "media_references": reference,
        "reference_host": "records.example",
        "media_type": "StillImage",
        "media_format": "image/jpeg",
        "media_license": media_license,
        "occurrence_license": "CC BY 4.0",
        "license_basis": (
            "item_media_license"
            if media_license
            else "occurrence_license_fallback"
        ),
        "status": status,
        "method": "fixture",
        "stable_candidate_url": stable_url,
        "validated_final_url": stable_url,
        "redirect_count": 0,
        "declared_content_type": "image/jpeg" if stable_url else None,
        "detected_content_type": "image/jpeg" if stable_url else None,
        "bytes_sampled": 128 if stable_url else 0,
        "probe_prefix_sha256": (
            "sha256:" + "b" * 64 if stable_url else None
        ),
        "content_sha256": None,
        "content_hash_status": "deferred",
        "adapter_version": "fixture/v1",
        "attempt_count": 0 if status == "rights_blocked" else 1,
        "terminal_reason": None if stable_url else status,
        "resolved_at": "2026-07-29T00:00:00Z",
        "provenance_fingerprint": "sha256:" + "c" * 64,
    }


def _rewrite_resolution_publication(
    resolution: Path,
    result_table: pa.Table,
) -> None:
    for path in resolution.iterdir():
        path.unlink()
    unresolved = result_table.filter(
        pa.compute.not_equal(result_table["status"], "resolved")
    )
    attempts = pa.Table.from_pylist([], schema=ATTEMPT_SCHEMA)
    artifacts = {
        "resolution_results.parquet": result_table,
        "resolution_attempts.parquet": attempts,
        "unresolved_rows.parquet": unresolved,
    }
    inventory: dict[str, dict[str, object]] = {}
    for name, table in artifacts.items():
        path = resolution / name
        pq.write_table(
            table,
            path,
            compression="zstd",
            row_group_size=2,
        )
        inventory[name] = _resolution_inventory(path)
    status_counts: dict[str, int] = {}
    for value in result_table["status"].to_pylist():
        status = str(value)
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "fixture-full-run",
        "input": {
            "mode": "full",
            "work_rows": result_table.num_rows,
            "input_rows": result_table.num_rows,
            "source_artifact_sha256": SOURCE_SHA256,
        },
        "counts": {
            "result_rows": result_table.num_rows,
            "attempt_rows": 0,
            "unresolved_rows": unresolved.num_rows,
            "resolved_rows": status_counts.get("resolved", 0),
            "rights_blocked_rows": status_counts.get(
                "rights_blocked",
                0,
            ),
            "eligible_resolution_rows": (
                result_table.num_rows
                - status_counts.get("rights_blocked", 0)
            ),
            "status_counts": status_counts,
        },
        "artifacts": inventory,
        "validation": {
            "one_result_per_input": True,
            "unique_source_row_ids": True,
            "every_work_item_completed": True,
            "rights_blocked_zero_attempts": True,
            "result_shard_checksums_match_registry": True,
            "registered_result_membership_bounded_to_queue": True,
            "selected_result_membership_exact": True,
            "attempt_shard_checksums_match_registry": True,
            "registered_attempt_membership_bounded_to_result_shards": True,
            "selected_attempt_membership_bounded_to_queue": True,
            "no_unreferenced_registered_shards": True,
            "all_parquet_row_groups_complete": True,
        },
        "manifest_policy": {"written_last": True},
    }
    (resolution / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _legacy_inventory(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": path.name,
        "rows": parquet.metadata.num_rows,
        "columns": parquet.metadata.num_columns,
        "row_groups": parquet.metadata.num_row_groups,
        "row_group_rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolution_inventory(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": parquet.metadata.num_columns,
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": rows,
        "row_groups_complete": (
            sum(rows) == parquet.metadata.num_rows
            and (
                parquet.metadata.num_rows == 0
                or all(value > 0 for value in rows)
            )
        ),
    }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
