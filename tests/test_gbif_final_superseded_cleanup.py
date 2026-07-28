from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.publication_audit import (
    IDENTITY_AUDIT_SCHEMA,
    INPUT_INVENTORY_SCHEMA,
    PUBLICATION_AUDIT_VERSION,
)
from biominer.gbif_final.superseded_cleanup import (
    CLEANUP_MANIFEST_VERSION,
    PROTECTED_RELATIVE_PATHS,
    SUPERSEDED_RELATIVE_PATHS,
    execute_superseded_cleanup,
    plan_superseded_cleanup,
    prepare_superseded_cleanup,
    validate_superseded_cleanup,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(
    path: Path,
    values: dict[str, list[object]],
    *,
    schema: pa.Schema | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = (
        pa.Table.from_pylist(
            [
                {
                    name: values[name][index]
                    for name in values
                }
                for index in range(len(next(iter(values.values()))))
            ],
            schema=schema,
        )
        if schema is not None
        else pa.table(values)
    )
    pq.write_table(table, path, compression="zstd", row_group_size=1)
    return path


def _inventory(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": str(path.resolve()),
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "physical_column_count": parquet.metadata.num_columns,
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "schema_fingerprint": "sha256:"
        + hashlib.sha256(
            parquet.schema_arrow.serialize().to_pybytes()
        ).hexdigest(),
        "row_groups_complete": (
            bool(row_group_rows)
            and sum(row_group_rows) == parquet.metadata.num_rows
            and all(rows > 0 for rows in row_group_rows)
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "BioMiner tests"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "fixture"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    for relative in PROTECTED_RELATIVE_PATHS:
        path = repository / relative
        if path.suffix in {".zip", ".sqlite"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"protected")
        else:
            path.mkdir(parents=True)
            (path / "protected.bin").write_bytes(b"protected")

    temporal = (
        repository
        / "data/derived/gbif_media_temporal/v1/"
        "gbif_media_temporal.parquet"
    )
    for index, relative in enumerate(SUPERSEDED_RELATIVE_PATHS):
        target = repository / relative
        target.mkdir(parents=True, exist_ok=True)
        if target != temporal.parent:
            (target / f"obsolete-{index:02d}.bin").write_bytes(
                f"obsolete-{index}".encode()
            )
    _write_parquet(temporal, {"gbifID": ["1", "2"]})

    publication = (
        repository / "data/derived/gbif_media_final/current"
    )
    final_path = _write_parquet(
        publication / "gbif_media_final_enriched.parquet",
        {"gbifID": ["1", "2"]},
    )
    primary_manifest_path = publication / "manifest.json"
    _write_json(
        primary_manifest_path,
        {"schema_version": "fixture", "producer_git_sha": commit},
    )

    audit = repository / "data/derived/gbif_media_final/audit-v1"
    temporal_inventory = _inventory(temporal)
    input_row = {
        "input_role": "temporal",
        "path": temporal_inventory["path"],
        "physical_bytes": temporal_inventory["physical_bytes"],
        "physical_sha256": temporal_inventory["physical_sha256"],
        "row_count": temporal_inventory["row_count"],
        "column_count": temporal_inventory["column_count"],
        "row_group_count": temporal_inventory["row_group_count"],
        "schema_fingerprint": temporal_inventory["schema_fingerprint"],
        "row_groups_complete": True,
        "not_newer_than_final_artifact": True,
        "primary_manifest_binding_status": "PRIMARY_MANIFEST_MATCH",
    }
    input_path = _write_parquet(
        audit / "input_inventory.parquet",
        {name: [value] for name, value in input_row.items()},
        schema=INPUT_INVENTORY_SCHEMA,
    )
    identity_path = _write_parquet(
        audit / "identity_audit.parquet",
        {"metric": ["rows"], "value": [2], "status": ["PASS"]},
        schema=IDENTITY_AUDIT_SCHEMA,
    )
    audit_artifacts: dict[str, dict[str, object]] = {}
    for path in (input_path, identity_path):
        evidence = _inventory(path)
        evidence["path"] = path.name
        audit_artifacts[path.name] = evidence
    audit_manifest: dict[str, object] = {
        "schema_version": PUBLICATION_AUDIT_VERSION,
        "generated_at": "2026-07-29T00:00:00Z",
        "producer_git_sha": commit,
        "audit_git_commit": commit,
        "primary_publication": {
            "directory": str(publication.resolve()),
            "manifest_path": str(primary_manifest_path.resolve()),
            "manifest_sha256": _sha256(primary_manifest_path),
            "manifest_schema_version": "fixture",
            "final_artifact": _inventory(final_path),
        },
        "counts": {
            "rows": 2,
            "columns": 1,
            "input_artifacts": 1,
        },
        "identity_audit": {"rows": 2},
        "artifacts": audit_artifacts,
        "validation": {"fixture_complete": True},
        "manifest_policy": {
            "create_only": True,
            "manifest_written_last": True,
            "primary_publication_unchanged": True,
        },
    }
    audit_manifest["manifest_fingerprint"] = (
        canonical_semantic_fingerprint(audit_manifest)
    )
    _write_json(audit / "manifest.json", audit_manifest)
    return {
        "repository_root": repository,
        "publication_audit_directory": audit,
        "state_directory": (
            repository
            / "data/state/gbif-final-superseded-cleanup-v1"
        ),
    }


def test_cleanup_is_dry_run_by_default_then_removes_exact_allowlist(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    plan = plan_superseded_cleanup(**arguments)
    repository = arguments["repository_root"]
    protected = [
        repository / relative for relative in PROTECTED_RELATIVE_PATHS
    ]

    assert plan["counts"]["target_directories"] == len(
        SUPERSEDED_RELATIVE_PATHS
    )
    assert all(
        (repository / relative).exists()
        for relative in SUPERSEDED_RELATIVE_PATHS
    )
    prepare_superseded_cleanup(**arguments)
    assert not (
        arguments["state_directory"] / "manifest.json"
    ).exists()
    assert all(path.exists() for path in protected)

    manifest = execute_superseded_cleanup(**arguments)

    assert manifest["schema_version"] == CLEANUP_MANIFEST_VERSION
    assert all(manifest["validation"].values())
    assert all(
        not (repository / relative).exists()
        for relative in SUPERSEDED_RELATIVE_PATHS
    )
    assert all(path.exists() for path in protected)
    assert (
        validate_superseded_cleanup(**arguments)
        == manifest
    )
    assert execute_superseded_cleanup(**arguments) == manifest


def test_cleanup_rejects_file_changed_after_intent(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    intent = prepare_superseded_cleanup(**arguments)
    first = (
        arguments["repository_root"]
        / intent["files"][0]["relative_path"]
    )
    first.write_bytes(b"x" * first.stat().st_size)

    with pytest.raises(
        RuntimeError,
        match="cleanup file checksum changed",
    ):
        execute_superseded_cleanup(**arguments)
    assert first.exists()
    assert not (
        arguments["state_directory"] / "manifest.json"
    ).exists()


def test_cleanup_resume_records_file_absent_after_intent(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    intent = prepare_superseded_cleanup(**arguments)
    first = (
        arguments["repository_root"]
        / intent["files"][0]["relative_path"]
    )
    first.unlink()

    manifest = execute_superseded_cleanup(**arguments)

    assert manifest["counts"]["already_absent_after_intent"] == 1
    assert all(manifest["validation"].values())


def test_cleanup_rejects_symlink_and_unexpected_post_intent_file(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    repository = arguments["repository_root"]
    target = repository / SUPERSEDED_RELATIVE_PATHS[0]
    protected = repository / PROTECTED_RELATIVE_PATHS[0]
    (target / "unsafe-link").symlink_to(protected / "protected.bin")

    with pytest.raises(RuntimeError, match="cleanup refuses symlink"):
        plan_superseded_cleanup(**arguments)

    (target / "unsafe-link").unlink()
    prepare_superseded_cleanup(**arguments)
    (target / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(
        RuntimeError,
        match="cleanup directory contains unexpected entries",
    ):
        execute_superseded_cleanup(**arguments)
    assert (target / "unexpected.bin").exists()
    assert not (
        arguments["state_directory"] / "manifest.json"
    ).exists()


def test_cleanup_requires_valid_publication_audit(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    audit_manifest_path = (
        arguments["publication_audit_directory"] / "manifest.json"
    )
    manifest = json.loads(audit_manifest_path.read_text())
    manifest["validation"]["fixture_complete"] = False
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_fingerprint"
    }
    manifest["manifest_fingerprint"] = canonical_semantic_fingerprint(body)
    _write_json(audit_manifest_path, manifest)

    with pytest.raises(
        RuntimeError,
        match="publication audit validation is not PASS",
    ):
        plan_superseded_cleanup(**arguments)
