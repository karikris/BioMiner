from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.evidence import audit_evidence_manifest


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_manifest(
    root: Path,
    *,
    artifact: Path,
    source_snapshot_id: str = "sha256:source",
) -> Path:
    metadata = pq.read_metadata(artifact)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fixture/v1",
                "code_commit": "HEAD",
                "source_snapshot_id": source_snapshot_id,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "physical_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                        "row_count": metadata.num_rows,
                        "column_count": metadata.num_columns,
                        "row_group_count": metadata.num_row_groups,
                    }
                ],
                "manifest_policy": {"written_last": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_evidence_audit_recalculates_parquet_and_dependency_fingerprint(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]}),
        artifact,
        row_group_size=2,
    )
    manifest = _write_manifest(tmp_path, artifact=artifact)

    result = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:source",
    )

    assert result["status"] == "PASS"
    assert result["artifact_count"] == 1
    assert result["artifacts"][0]["checksum_status"] == "PASS"
    assert result["artifacts"][0]["row_group_status"] == "PASS"
    assert result["artifacts"][0]["row_group_rows"] == [2, 1]
    assert result["producer_commit_status"] == "PASS"
    assert result["source_snapshot_status"] == "PASS"
    assert result["manifest_last_status"] == "PASS"
    assert result["dependency_fingerprint"].startswith("sha256:")


def test_evidence_audit_fails_closed_for_checksum_or_truncation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3]}), artifact)
    manifest = _write_manifest(tmp_path, artifact=artifact)
    artifact.write_bytes(artifact.read_bytes()[:-8])

    result = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:source",
    )

    assert result["status"] == "FAIL"
    assert result["artifacts"][0]["checksum_status"] == "FAIL"
    assert result["artifacts"][0]["row_group_status"] == "FAIL"
    assert "artifact_checksum_mismatch" in result["failure_reasons"]
    assert "parquet_metadata_unreadable" in result["failure_reasons"]


def test_evidence_audit_requires_manifest_to_be_newer_than_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1]}), artifact)
    manifest = _write_manifest(tmp_path, artifact=artifact)
    manifest_time = manifest.stat().st_mtime_ns
    os.utime(
        artifact,
        ns=(manifest_time + 1_000_000_000, manifest_time + 1_000_000_000),
    )

    result = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:source",
    )

    assert result["status"] == "FAIL"
    assert result["manifest_last_status"] == "FAIL"
    assert "manifest_not_written_last" in result["failure_reasons"]


def test_evidence_audit_is_invalidated_by_dependencies_not_current_head(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1]}), artifact)
    manifest = _write_manifest(tmp_path, artifact=artifact)

    first = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:source",
    )
    second = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:source",
        current_git_commit="a-different-head-does-not-invalidate-artifacts",
    )

    assert first["dependency_fingerprint"] == second["dependency_fingerprint"]
    assert first["status"] == second["status"] == "PASS"


def test_evidence_audit_rejects_wrong_source_snapshot(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1]}), artifact)
    manifest = _write_manifest(tmp_path, artifact=artifact)

    result = audit_evidence_manifest(
        manifest,
        repository_root=Path.cwd(),
        expected_source_snapshot_id="sha256:other",
    )

    assert result["status"] == "FAIL"
    assert result["source_snapshot_status"] == "FAIL"
    assert "source_snapshot_mismatch" in result["failure_reasons"]
