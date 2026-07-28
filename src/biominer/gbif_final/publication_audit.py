from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import (
    ASSEMBLY_MANIFEST_VERSION,
    FINAL_FILENAME,
    MANIFEST_FILENAME,
    validate_assembled_output,
)
from biominer.gbif_final.pipeline import FINAL_SCHEMA_VERSION


PUBLICATION_AUDIT_VERSION = "gbif-final-publication-audit/v1"

INPUT_INVENTORY_SCHEMA = pa.schema(
    [
        pa.field("input_role", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("physical_bytes", pa.int64(), nullable=False),
        pa.field("physical_sha256", pa.string(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("column_count", pa.int64(), nullable=False),
        pa.field("row_group_count", pa.int64(), nullable=False),
        pa.field("schema_fingerprint", pa.string(), nullable=False),
        pa.field("row_groups_complete", pa.bool_(), nullable=False),
        pa.field(
            "not_newer_than_final_artifact",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "primary_manifest_binding_status",
            pa.string(),
            nullable=False,
        ),
    ]
)

IDENTITY_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("metric", pa.string(), nullable=False),
        pa.field("value", pa.int64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ]
)


def audit_final_publication(
    *,
    publication_directory: str | Path,
    temporal_parquet: str | Path,
    pre_temporal_parquet: str | Path,
    registry_directory: str | Path,
    source_assertions: str | Path | None,
    quality_directory: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    expected_producer_git_sha: str,
    memory_limit: str = "8GB",
    threads: int = 4,
) -> dict[str, Any]:
    """Independently bind and audit a legacy or bounded final publication."""

    if not expected_producer_git_sha.strip():
        raise ValueError("expected producer Git SHA must be non-empty")
    if not memory_limit.strip():
        raise ValueError("memory_limit must be non-empty")
    if threads <= 0:
        raise ValueError("threads must be positive")
    publication = Path(publication_directory).resolve()
    destination = Path(output_directory).resolve()
    repository = Path(repository_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite publication audit: {destination}"
        )
    if (
        destination == publication
        or destination.is_relative_to(publication)
        or publication.is_relative_to(destination)
    ):
        raise ValueError("publication and audit directories must not overlap")

    manifest_path = publication / MANIFEST_FILENAME
    final_path = publication / FINAL_FILENAME
    if not manifest_path.is_file() or not final_path.is_file():
        raise FileNotFoundError(
            "final publication requires both Parquet and manifest"
        )
    primary_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    primary_schema = str(primary_manifest.get("schema_version") or "")
    if primary_schema == ASSEMBLY_MANIFEST_VERSION:
        validate_assembled_output(
            publication,
            expected_code_commit=expected_producer_git_sha,
        )
        producer_git_sha = str(primary_manifest["code_commit"])
        recorded_artifact = primary_manifest["artifacts"][0]
    elif primary_schema == FINAL_SCHEMA_VERSION:
        producer_git_sha = str(primary_manifest.get("producer_git_sha") or "")
        recorded_artifact = primary_manifest.get("artifact") or {}
        acceptance_gate = primary_manifest.get("acceptance_gate") or {}
        if not acceptance_gate or not all(acceptance_gate.values()):
            raise RuntimeError(
                "legacy final publication acceptance gate is not PASS"
            )
    else:
        raise RuntimeError(
            f"unsupported final publication schema: {primary_schema!r}"
        )
    if producer_git_sha != expected_producer_git_sha:
        raise RuntimeError("final publication producer Git SHA differs")
    _require_git_commit(repository, producer_git_sha)
    audit_git_commit = _git_head(repository)

    final_inventory = _parquet_inventory(final_path)
    _validate_recorded_artifact(
        recorded_artifact,
        final_inventory=final_inventory,
        schema_version=primary_schema,
    )
    if manifest_path.stat().st_mtime_ns < final_path.stat().st_mtime_ns:
        raise RuntimeError("final publication manifest was not written last")
    observed_publication_files = {
        path.resolve()
        for path in publication.rglob("*")
        if path.is_file()
    }
    publication_files_exact = observed_publication_files == {
        manifest_path,
        final_path,
    }
    if not publication_files_exact:
        raise RuntimeError("final publication file inventory is not exact")

    temporal = Path(temporal_parquet).resolve()
    pre_temporal = Path(pre_temporal_parquet).resolve()
    registry = Path(registry_directory).resolve()
    quality = Path(quality_directory).resolve()
    inputs = _input_paths(
        temporal=temporal,
        pre_temporal=pre_temporal,
        registry=registry,
        source_assertions=source_assertions,
        quality=quality,
    )
    final_mtime_ns = final_path.stat().st_mtime_ns
    input_rows: list[dict[str, object]] = []
    for role, path in inputs:
        inventory = _parquet_inventory(path)
        input_rows.append(
            {
                "input_role": role,
                **inventory,
                "not_newer_than_final_artifact": (
                    path.stat().st_mtime_ns <= final_mtime_ns
                ),
                "primary_manifest_binding_status": (
                    _primary_binding_status(
                        primary_manifest,
                        schema_version=primary_schema,
                        role=role,
                        inventory=inventory,
                    )
                ),
            }
        )
    temporal_inventory = next(
        row for row in input_rows if row["input_role"] == "temporal"
    )
    expected_rows = int(temporal_inventory["row_count"])
    if int(final_inventory["row_count"]) != expected_rows:
        raise RuntimeError("final and temporal row counts differ")

    required_columns = {
        "source_row_id",
        "media_assertion_id",
        "occurrence_quality",
        "media_quality",
        "rights_quality",
        "duplicate_quality",
        "ai_readiness",
        "derived_quality_assertions",
        "registry_match_status",
        "registry_match_method",
        "registry_taxon_key",
        "keyword_evidence",
        "keyword_source_assertions",
        "flickr_query_terms",
    }
    final_schema = pq.ParquetFile(final_path).schema_arrow
    missing_columns = sorted(required_columns - set(final_schema.names))
    if missing_columns:
        raise RuntimeError(
            "final publication lacks enrichment columns: "
            + ", ".join(missing_columns)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        destination.parent
        / f".{destination.name}.{uuid4().hex}.staging"
    )
    staging.mkdir()
    temporary = staging / ".duckdb_tmp"
    temporary.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET memory_limit=?", [memory_limit])
        connection.execute("SET temp_directory=?", [str(temporary)])
        (
            rows,
            null_source_ids,
            null_media_ids,
            unique_source_ids,
            unique_media_ids,
            enrichment_status_rows,
        ) = connection.execute(
            """
            SELECT
              count(*),
              count(*) FILTER (WHERE source_row_id IS NULL),
              count(*) FILTER (WHERE media_assertion_id IS NULL),
              count(DISTINCT source_row_id),
              count(DISTINCT media_assertion_id),
              count(*) FILTER (WHERE registry_match_status IS NOT NULL)
            FROM read_parquet(?)
            """,
            [str(final_path)],
        ).fetchone()
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)

    identity_values = {
        "rows": int(rows),
        "null_source_row_ids": int(null_source_ids),
        "null_media_assertion_ids": int(null_media_ids),
        "unique_source_row_ids": int(unique_source_ids),
        "unique_media_assertion_ids": int(unique_media_ids),
        "rows_with_registry_match_status": int(enrichment_status_rows),
    }
    expected_values = {
        "rows": expected_rows,
        "null_source_row_ids": 0,
        "null_media_assertion_ids": 0,
        "unique_source_row_ids": expected_rows,
        "unique_media_assertion_ids": expected_rows,
        "rows_with_registry_match_status": expected_rows,
    }
    identity_rows = [
        {
            "metric": metric,
            "value": value,
            "status": "PASS" if value == expected_values[metric] else "FAIL",
        }
        for metric, value in identity_values.items()
    ]
    validation = {
        "primary_manifest_revalidated": True,
        "producer_commit_addressable": True,
        "final_checksum_recalculated": True,
        "final_row_groups_complete": bool(
            final_inventory["row_groups_complete"]
        ),
        "all_dependencies_checksummed": len(input_rows) == len(inputs),
        "recorded_dependency_bindings_match": not any(
            row["primary_manifest_binding_status"]
            == "PRIMARY_MANIFEST_MISMATCH"
            for row in input_rows
        ),
        "all_dependencies_not_newer_than_final": all(
            bool(row["not_newer_than_final_artifact"])
            for row in input_rows
        ),
        "final_rows_match_temporal_rows": int(rows) == expected_rows,
        "stable_identities_non_null": (
            int(null_source_ids) == 0 and int(null_media_ids) == 0
        ),
        "stable_identities_unique": (
            int(unique_source_ids) == expected_rows
            and int(unique_media_ids) == expected_rows
        ),
        "enrichment_status_covers_every_row": (
            int(enrichment_status_rows) == expected_rows
        ),
        "required_enrichment_columns_present": not missing_columns,
        "primary_publication_file_inventory_exact": (
            publication_files_exact
        ),
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        raise RuntimeError(
            f"final publication audit failed: {validation}"
        )

    try:
        input_table = pa.Table.from_pylist(
            input_rows,
            schema=INPUT_INVENTORY_SCHEMA,
        )
        identity_table = pa.Table.from_pylist(
            identity_rows,
            schema=IDENTITY_AUDIT_SCHEMA,
        )
        input_path = staging / "input_inventory.parquet"
        identity_path = staging / "identity_audit.parquet"
        pq.write_table(
            input_table,
            input_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            identity_table,
            identity_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        artifacts = {}
        for artifact_path in (input_path, identity_path):
            inventory = _parquet_inventory(artifact_path)
            inventory["path"] = artifact_path.name
            artifacts[artifact_path.name] = inventory
        audit_manifest: dict[str, Any] = {
            "schema_version": PUBLICATION_AUDIT_VERSION,
            "generated_at": _timestamp(),
            "producer_git_sha": producer_git_sha,
            "audit_git_commit": audit_git_commit,
            "configuration": {
                "memory_limit": memory_limit,
                "threads": threads,
                "dependency_roles": [role for role, _ in inputs],
            },
            "primary_publication": {
                "directory": str(publication),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "manifest_schema_version": primary_schema,
                "final_artifact": final_inventory,
            },
            "counts": {
                "rows": expected_rows,
                "columns": int(final_inventory["column_count"]),
                "input_artifacts": len(input_rows),
            },
            "identity_audit": identity_values,
            "artifacts": artifacts,
            "validation": validation,
            "manifest_policy": {
                "create_only": True,
                "manifest_written_last": True,
                "primary_publication_unchanged": True,
            },
        }
        audit_manifest["manifest_fingerprint"] = (
            canonical_semantic_fingerprint(audit_manifest)
        )
        _write_json(staging / MANIFEST_FILENAME, audit_manifest)
        if (staging / MANIFEST_FILENAME).stat().st_mtime_ns < max(
            input_path.stat().st_mtime_ns,
            identity_path.stat().st_mtime_ns,
        ):
            raise RuntimeError("publication audit manifest was not written last")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return audit_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _input_paths(
    *,
    temporal: Path,
    pre_temporal: Path,
    registry: Path,
    source_assertions: str | Path | None,
    quality: Path,
) -> list[tuple[str, Path]]:
    paths = [
        ("temporal", temporal),
        ("pre_temporal", pre_temporal),
        ("registry_taxa", registry / "taxa.parquet"),
        ("registry_names", registry / "names.parquet"),
        ("registry_species_paths", registry / "species_paths.parquet"),
        (
            "media_quality",
            quality
            / "media_assertion_quality"
            / "media_assertion_quality.parquet",
        ),
        (
            "occurrence_quality",
            quality / "occurrence_quality" / "occurrence_quality.parquet",
        ),
        (
            "rights_quality",
            quality / "rights_and_attribution" / "media_rights.parquet",
        ),
        (
            "duplicate_quality",
            quality / "duplicates" / "duplicate_membership.parquet",
        ),
        (
            "derived_assertions",
            quality
            / "quality_results"
            / "phase3"
            / "derived_assertions.parquet",
        ),
    ]
    readiness = sorted(
        (quality / "ai_readiness" / "parts").glob("*.parquet")
    )
    if not readiness:
        raise FileNotFoundError(
            quality / "ai_readiness" / "parts" / "*.parquet"
        )
    paths.extend(
        (f"ai_readiness_part_{index:05d}", path)
        for index, path in enumerate(readiness)
    )
    if source_assertions is not None:
        paths.append(
            ("keyword_source_assertions", Path(source_assertions).resolve())
        )
    resolved = [(role, path.resolve()) for role, path in paths]
    for _, path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


def _primary_binding_status(
    manifest: Mapping[str, object],
    *,
    schema_version: str,
    role: str,
    inventory: Mapping[str, object],
) -> str:
    if schema_version == FINAL_SCHEMA_VERSION:
        key = {
            "temporal": "temporal_parquet",
            "pre_temporal": "pre_temporal_parquet",
        }.get(role)
        if key is None:
            return "INDEPENDENT_AUDIT_BINDING"
        recorded = (manifest.get("inputs") or {}).get(key) or {}
        return (
            "PRIMARY_MANIFEST_MATCH"
            if recorded.get("sha256") == inventory["physical_sha256"]
            else "PRIMARY_MANIFEST_MISMATCH"
        )
    source_scope = manifest.get("source_scope") or {}
    recorded = source_scope.get("input_inventory") or {}
    dimension = {
        "temporal": "temporal",
        "media_quality": "media_quality",
        "occurrence_quality": "occurrence_quality",
        "rights_quality": "rights_quality",
        "duplicate_quality": "duplicate_quality",
        "derived_assertions": "derived_assertions",
    }.get(role)
    if dimension is None:
        return "EMBEDDED_UPSTREAM_OR_INDEPENDENT_BINDING"
    evidence = recorded.get(dimension) or {}
    return (
        "PRIMARY_MANIFEST_MATCH"
        if evidence.get("physical_sha256") == inventory["physical_sha256"]
        else "PRIMARY_MANIFEST_MISMATCH"
    )


def _validate_recorded_artifact(
    artifact: Mapping[str, object],
    *,
    final_inventory: Mapping[str, object],
    schema_version: str,
) -> None:
    if str(artifact.get("path")) != FINAL_FILENAME:
        raise RuntimeError("final manifest artifact path differs")
    fields = (
        {
            "rows": "row_count",
            "columns": "physical_column_count",
            "row_groups": "row_group_count",
            "row_group_rows": "row_group_rows",
            "bytes": "physical_bytes",
            "sha256": "physical_sha256",
        }
        if schema_version == FINAL_SCHEMA_VERSION
        else {
            "row_count": "row_count",
            "column_count": "column_count",
            "row_group_count": "row_group_count",
            "row_group_rows": "row_group_rows",
            "physical_bytes": "physical_bytes",
            "physical_sha256": "physical_sha256",
            "schema_fingerprint": "schema_fingerprint",
        }
    )
    mismatches = [
        recorded
        for recorded, observed in fields.items()
        if artifact.get(recorded) != final_inventory.get(observed)
    ]
    if mismatches:
        raise RuntimeError(
            "final manifest artifact inventory mismatch: "
            + ", ".join(mismatches)
        )


def _parquet_inventory(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    complete = (
        parquet.metadata.num_rows > 0
        and parquet.metadata.num_row_groups > 0
        and sum(row_group_rows) == parquet.metadata.num_rows
        and all(rows > 0 for rows in row_group_rows)
    )
    if not complete:
        raise RuntimeError(f"Parquet row groups are incomplete: {path}")
    return {
        "path": str(path),
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
        "row_groups_complete": complete,
    }


def _require_git_commit(repository: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"producer Git commit is not addressable: {commit}"
        )


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "IDENTITY_AUDIT_SCHEMA",
    "INPUT_INVENTORY_SCHEMA",
    "PUBLICATION_AUDIT_VERSION",
    "audit_final_publication",
]
