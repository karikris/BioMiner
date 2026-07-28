from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


PROVIDER_REVIEW_VERSION = "biominer-gbif-provider-review/v1"
PROVIDER_REVIEW_SCHEMA = pa.schema(
    [
        ("provider_review_version", pa.string()),
        ("review_sample_id", pa.string()),
        ("review_stratum", pa.string()),
        ("provider", pa.string()),
        ("datasetKey", pa.string()),
        ("evidence_scope", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("occurrenceID", pa.string()),
        ("media_identifier", pa.string()),
        ("before_values_json", pa.string()),
        ("provider_values_json", pa.string()),
        ("evidence_reference", pa.string()),
        ("review_reason", pa.string()),
        ("expected_review_decision", pa.string()),
        ("review_status", pa.string()),
        ("reviewer", pa.string()),
        ("reviewed_at", pa.string()),
        ("review_notes", pa.string()),
    ]
)


def publish_provider_review_sample(
    *,
    provider_enrichment_directory: str | Path,
    output_directory: str | Path,
    code_commit: str,
    sample_per_stratum: int = 10,
) -> dict[str, object]:
    """Publish deterministic review rows without changing provider evidence."""

    source = Path(provider_enrichment_directory).resolve()
    destination = Path(output_directory).resolve()
    if sample_per_stratum < 1:
        raise ValueError("sample_per_stratum must be positive")
    manifest_path = source / "manifest.json"
    for path in (
        manifest_path,
        source / "provider_item_evidence.parquet",
        source / "provider_occurrence_context.parquet",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_source_artifacts(source, source_manifest)
    context_path = source / "provider_occurrence_context.parquet"
    denied_contexts = connection_count(
        context_path,
        "explicit_denied_license_items > 0",
    )

    connection = duckdb.connect()
    try:
        item_path = source / "provider_item_evidence.parquet"
        table = connection.execute(
            f"""
            WITH candidates AS (
                SELECT
                    provider,
                    datasetKey,
                    'item' AS evidence_scope,
                    source_row_id,
                    media_assertion_id,
                    gbifID,
                    occurrenceID,
                    media_identifier,
                    json_object(
                        'media_identifier', media_identifier
                    )::VARCHAR AS before_values_json,
                    json_object(
                        'media_license', archive_media_license,
                        'creator', archive_creator,
                        'rightsHolder', archive_rightsHolder,
                        'format', archive_format,
                        'type', archive_type
                    )::VARCHAR AS provider_values_json,
                    archive_member || ':' ||
                        archive_source_row_number::VARCHAR AS evidence_reference,
                    CASE
                        WHEN item_binding_status = 'CONFLICT'
                            THEN 'ambiguous_exact_item_key'
                        ELSE 'verify_exact_item_binding_and_fields'
                    END AS review_reason,
                    'VERIFY_ITEM_BINDING_AND_FIELDS'
                        AS expected_review_decision,
                    item_binding_status AS status,
                    0::BIGINT AS denied_items
                FROM read_parquet({_literal(str(item_path))})
                UNION ALL
                SELECT
                    provider,
                    datasetKey,
                    'occurrence_media_ensemble' AS evidence_scope,
                    NULL AS source_row_id,
                    NULL AS media_assertion_id,
                    array_to_string(target_gbif_ids, ',') AS gbifID,
                    occurrenceID,
                    NULL AS media_identifier,
                    json_object(
                        'current_media_license_values',
                        current_media_license_values
                    )::VARCHAR AS before_values_json,
                    json_object(
                        'archive_media_license_values',
                        archive_media_license_values,
                        'archive_creator_values', archive_creator_values,
                        'archive_rightsHolder_values',
                        archive_rightsHolder_values,
                        'archive_media_rows', archive_media_rows
                    )::VARCHAR AS provider_values_json,
                    archive_member || ':occurrenceID=' ||
                        occurrenceID AS evidence_reference,
                    CASE
                        WHEN explicit_denied_license_items > 0
                            THEN 'current_archive_contains_explicitly_denied_item'
                        WHEN license_context_status = 'CONFLICT'
                            THEN 'current_archive_has_multiple_occurrence_licences'
                        ELSE 'occurrence_context_is_not_item_bound'
                    END AS review_reason,
                    'DO_NOT_AUTOMATICALLY_REPAIR'
                        AS expected_review_decision,
                    license_context_status AS status,
                    explicit_denied_license_items AS denied_items
                FROM read_parquet({_literal(str(context_path))})
            ),
            ranked AS (
                SELECT *,
                    provider || '|' || evidence_scope || '|' || status
                        AS review_stratum,
                    sha256(
                        provider || '|' || datasetKey || '|' ||
                        coalesce(source_row_id, '') || '|' ||
                        coalesce(occurrenceID, '') || '|' ||
                        evidence_scope
                    ) AS selection_hash,
                    row_number() OVER (
                        PARTITION BY provider, evidence_scope, status
                        ORDER BY sha256(
                            provider || '|' || datasetKey || '|' ||
                            coalesce(source_row_id, '') || '|' ||
                            coalesce(occurrenceID, '') || '|' ||
                            evidence_scope
                        )
                    ) AS sample_rank
                FROM candidates
            )
            SELECT
                {_literal(PROVIDER_REVIEW_VERSION)}
                    AS provider_review_version,
                'sha256:' || sha256(
                    'provider-review|' || selection_hash
                ) AS review_sample_id,
                review_stratum,
                provider,
                datasetKey,
                evidence_scope,
                source_row_id,
                media_assertion_id,
                gbifID,
                occurrenceID,
                media_identifier,
                before_values_json,
                provider_values_json,
                evidence_reference,
                review_reason,
                expected_review_decision,
                'PENDING' AS review_status,
                NULL::VARCHAR AS reviewer,
                NULL::VARCHAR AS reviewed_at,
                NULL::VARCHAR AS review_notes
            FROM ranked
            WHERE sample_rank <= {int(sample_per_stratum)}
               OR denied_items > 0
            ORDER BY review_stratum, review_sample_id
            """
        ).to_arrow_table()
    finally:
        connection.close()
    table = table.cast(PROVIDER_REVIEW_SCHEMA)
    if table.num_rows == 0:
        raise ValueError("provider review sample is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    output = staging / "provider_review_sample.parquet"
    pq.write_table(
        table,
        output,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    artifact = _artifact(output)
    denied_rows = connection_count(
        output,
        "review_reason = 'current_archive_contains_explicitly_denied_item'",
    )
    manifest = {
        "schema_version": PROVIDER_REVIEW_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_manifest["source_snapshot_id"],
        "input": {
            "provider_enrichment_manifest": str(manifest_path),
            "provider_enrichment_manifest_sha256": _sha256(manifest_path),
        },
        "configuration": {"sample_per_stratum": sample_per_stratum},
        "counts": {
            "review_rows": table.num_rows,
            "review_strata": len(set(table["review_stratum"].to_pylist())),
            "explicit_denied_context_review_rows": denied_rows,
            "explicit_denied_context_items": int(
                source_manifest["counts"]["explicit_denied_context_items"]
            ),
            "pending_reviews": table.num_rows,
        },
        "validation": {
            "source_artifact_checksums_match": True,
            "all_explicit_denied_contexts_included": denied_rows
            == denied_contexts,
            "all_reviews_pending": set(table["review_status"].to_pylist())
            == {"PENDING"},
            "occurrence_context_not_automatically_repaired": all(
                decision == "DO_NOT_AUTOMATICALLY_REPAIR"
                for scope, decision in zip(
                    table["evidence_scope"].to_pylist(),
                    table["expected_review_decision"].to_pylist(),
                    strict=True,
                )
                if scope == "occurrence_media_ensemble"
            ),
            "manifest_written_last": True,
        },
        "artifacts": [artifact],
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    if not all(manifest["validation"].values()):
        raise ValueError(f"provider review validation failed: {manifest['validation']}")
    _write_json(staging / "manifest.json", manifest)
    if _sha256(output) != artifact["sha256"]:
        raise ValueError("provider review checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _validate_source_artifacts(root: Path, manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("provider enrichment manifest has no artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("provider enrichment artifact is invalid")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("provider enrichment artifact has no checksum")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"provider enrichment artifact mismatch: {path}")


def connection_count(path: Path, predicate: str) -> int:
    connection = duckdb.connect()
    try:
        return int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet(?) WHERE {predicate}",
                [str(path)],
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = [
    "PROVIDER_REVIEW_SCHEMA",
    "PROVIDER_REVIEW_VERSION",
    "publish_provider_review_sample",
]
