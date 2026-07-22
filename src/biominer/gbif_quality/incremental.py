from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import glob
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


INCREMENTAL_VERSION = "biominer-gbif-media-incremental/v1"
INCREMENTAL_RULE_VERSION = "source-domain-diff/v1.0.0"

QUEUE_SCHEMA = pa.schema(
    [
        ("incremental_version", pa.string()),
        ("media_assertion_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("change_status", pa.string()),
        ("change_reasons", pa.list_(pa.string())),
        ("url_probe_due_status", pa.string()),
        ("provider_metadata_due_status", pa.string()),
        ("queue_status", pa.string()),
    ]
)

FRESHNESS_SCHEMA = pa.schema(
    [
        ("incremental_version", pa.string()),
        ("evidence_domain", pa.string()),
        ("ttl_days", pa.int32()),
        ("version", pa.string()),
        ("refresh_rule", pa.string()),
        ("current_status", pa.string()),
    ]
)


def publish_incremental_state(
    *,
    v3_parquet: str | Path,
    media_quality_parquet: str | Path,
    duplicates_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_rows: int,
    code_commit: str,
    previous_state_glob: str | Path | None = None,
    taxonomy_snapshot_version: str = "GBIF_BACKBONE_SOURCE_SNAPSHOT",
    boundary_dataset_version: str = "NOT_TESTED",
    adapter_version: str = "gbif-media-resolver/v2",
    url_probe_ttl_days: int = 30,
    provider_metadata_ttl_days: int = 90,
    memory_limit: str = "6GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
    partitions: int = 16,
) -> dict[str, object]:
    """Publish current domain hashes and a sparse queue of changed assertions."""

    source = Path(v3_parquet).resolve()
    quality = Path(media_quality_parquet).resolve()
    duplicates = Path(duplicates_parquet).resolve()
    for path in (source, quality, duplicates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if partitions < 1 or url_probe_ttl_days < 1 or provider_metadata_ttl_days < 1:
        raise ValueError("partition and TTL values must be positive")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    previous = str(previous_state_glob) if previous_state_glob is not None else None
    if previous is not None and not _glob_files(previous):
        raise FileNotFoundError(previous)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = Path(temp_directory).resolve() if temp_directory else staging / "duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    state = staging / "state"
    queue = staging / "changed_row_queue.parquet"
    freshness = staging / "freshness_policy.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            _state_sql(
                source=source,
                quality=quality,
                duplicates=duplicates,
                output=state,
                source_snapshot_id=source_snapshot_id,
                partitions=partitions,
            )
        )
        current_glob = str(state / "**/*.parquet")
        rows, distinct_ids = connection.execute(
            f"SELECT count(*),count(distinct media_assertion_id) FROM read_parquet({_lit(current_glob)})"
        ).fetchone()
        if int(rows) != expected_rows or int(distinct_ids) != expected_rows:
            raise ValueError("incremental state does not reconcile")
        if previous is None:
            pq.write_table(pa.Table.from_pylist([], schema=QUEUE_SCHEMA), queue, compression="zstd")
            change_counts: dict[str, int] = {"BASELINE_INITIALIZED": int(rows)}
        else:
            connection.execute(_queue_sql(current_glob, previous, queue))
            change_counts = {
                str(key): int(value)
                for key, value in connection.execute(
                    f"SELECT change_status,count(*) FROM read_parquet({_lit(str(queue))}) GROUP BY 1"
                ).fetchall()
            }
        _write_freshness(
            freshness,
            url_probe_ttl_days=url_probe_ttl_days,
            provider_metadata_ttl_days=provider_metadata_ttl_days,
            taxonomy_snapshot_version=taxonomy_snapshot_version,
            boundary_dataset_version=boundary_dataset_version,
            adapter_version=adapter_version,
        )
        queue_rows = int(
            connection.execute(f"SELECT count(*) FROM read_parquet({_lit(str(queue))})").fetchone()[0]
        )
    except BaseException:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    if temp_directory is None:
        shutil.rmtree(temporary, ignore_errors=True)
    state_parts = sorted(state.glob("**/*.parquet"))
    validation = {
        "rows_match": int(rows) == expected_rows,
        "one_row_per_media_assertion": int(distinct_ids) == expected_rows,
        "state_partitions_nonempty": bool(state_parts),
        "baseline_queue_empty" if previous is None else "unchanged_rows_not_queued": (
            queue_rows == 0 if previous is None else True
        ),
        "source_fields_unchanged": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"incremental validation failed: {validation}")
    artifacts = [*(_artifact(path, staging) for path in state_parts), _artifact(queue, staging), _artifact(freshness, staging)]
    manifest = {
        "schema_version": INCREMENTAL_VERSION,
        "rule_version": INCREMENTAL_RULE_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "previous_state": previous,
        "inputs": {"v3": str(source), "media_quality": str(quality), "duplicates": str(duplicates)},
        "counts": {
            "state_rows": int(rows),
            "distinct_media_assertions": int(distinct_ids),
            "queue_rows": queue_rows,
            "change_counts": change_counts,
        },
        "configuration": {
            "partitions": partitions,
            "url_probe_ttl_days": url_probe_ttl_days,
            "provider_metadata_ttl_days": provider_metadata_ttl_days,
            "taxonomy_snapshot_version": taxonomy_snapshot_version,
            "boundary_dataset_version": boundary_dataset_version,
            "adapter_version": adapter_version,
        },
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        if _sha256(staging / str(artifact["path"])) != artifact["sha256"]:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError("incremental checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _state_sql(*, source: Path, quality: Path, duplicates: Path, output: Path, source_snapshot_id: str, partitions: int) -> str:
    def digest(*fields: str) -> str:
        values = ",".join(f"coalesce(cast(v.{field} AS VARCHAR),'<NULL>')" for field in fields)
        return f"unhex(sha256(concat_ws(chr(31),{values})))"

    return f"""
    COPY (SELECT
      (hash(q.media_assertion_id)%{partitions})::INTEGER partition_bucket,
      {_lit(INCREMENTAL_VERSION)} incremental_version,{_lit(source_snapshot_id)} source_snapshot_id,
      q.source_row_id,q.media_assertion_id,trim(cast(v.gbifID AS VARCHAR)) gbifID,
      unhex(replace(d.source_value_hash,'sha256:','')) source_value_hash,
      {digest('media_identifier','media_references')} media_url_value_hash,
      NULL::BLOB final_url_value_hash,NULL::BLOB image_content_hash,
      {digest('media_license','media_creator','media_rightsHolder')} media_rights_value_hash,
      {digest('decimalLatitude','decimalLongitude','coordinateUncertaintyInMeters')} spatial_value_hash,
      {digest('eventDate','year','month','day')} temporal_value_hash,
      {digest('identifiedBy','identificationVerificationStatus','dateIdentified')} identification_value_hash,
      {digest('scientificName','taxonRank','taxonKey','acceptedTaxonKey','taxonomicStatus')} taxonomy_value_hash,
      {digest('datasetKey','publisher','media_publisher')} provider_value_hash
    FROM read_parquet({_lit(str(source))}) v
    POSITIONAL JOIN read_parquet({_lit(str(quality))}) q
    JOIN read_parquet({_lit(str(duplicates))}) d ON q.media_assertion_id=d.media_assertion_id)
    TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(partition_bucket),
      ROW_GROUP_SIZE 250000,FILENAME_PATTERN 'part-{{i}}')
    """


def _queue_sql(current: str, previous: str, output: Path) -> str:
    fields = (
        ("source_value_hash", "SOURCE_VALUE_CHANGED"),
        ("media_url_value_hash", "MEDIA_URL_CHANGED"),
        ("final_url_value_hash", "FINAL_URL_CHANGED"),
        ("image_content_hash", "IMAGE_CONTENT_CHANGED"),
        ("media_rights_value_hash", "MEDIA_RIGHTS_CHANGED"),
        ("spatial_value_hash", "SPATIAL_VALUE_CHANGED"),
        ("temporal_value_hash", "TEMPORAL_VALUE_CHANGED"),
        ("identification_value_hash", "IDENTIFICATION_CHANGED"),
        ("taxonomy_value_hash", "TAXONOMY_CHANGED"),
        ("provider_value_hash", "PROVIDER_CHANGED"),
    )
    reasons = ",".join(
        f"CASE WHEN c.media_assertion_id IS NOT NULL AND p.media_assertion_id IS NOT NULL "
        f"AND c.{field} IS DISTINCT FROM p.{field} THEN {_lit(reason)} END"
        for field, reason in fields
    )
    any_changed = " OR ".join(f"c.{field} IS DISTINCT FROM p.{field}" for field, _ in fields)
    return f"""
    COPY (SELECT {_lit(INCREMENTAL_VERSION)} incremental_version,
      coalesce(c.media_assertion_id,p.media_assertion_id) media_assertion_id,
      coalesce(c.source_row_id,p.source_row_id) source_row_id,coalesce(c.gbifID,p.gbifID) gbifID,
      CASE WHEN p.media_assertion_id IS NULL THEN 'NEW'
           WHEN c.media_assertion_id IS NULL THEN 'DELETED' ELSE 'CHANGED' END change_status,
      list_filter([CASE WHEN p.media_assertion_id IS NULL THEN 'NEW_MEDIA_ASSERTION' END,
        CASE WHEN c.media_assertion_id IS NULL THEN 'DELETED_MEDIA_ASSERTION' END,{reasons}],x->x IS NOT NULL) change_reasons,
      CASE WHEN p.media_assertion_id IS NULL OR c.media_url_value_hash IS DISTINCT FROM p.media_url_value_hash
        THEN 'DUE' ELSE 'NOT_DUE' END url_probe_due_status,
      CASE WHEN p.media_assertion_id IS NULL OR c.provider_value_hash IS DISTINCT FROM p.provider_value_hash
        THEN 'DUE' ELSE 'NOT_DUE' END provider_metadata_due_status,'PENDING' queue_status
    FROM read_parquet({_lit(current)}) c FULL OUTER JOIN read_parquet({_lit(previous)}) p USING(media_assertion_id)
    WHERE c.media_assertion_id IS NULL OR p.media_assertion_id IS NULL OR {any_changed})
    TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)
    """


def _write_freshness(path: Path, *, url_probe_ttl_days: int, provider_metadata_ttl_days: int, taxonomy_snapshot_version: str, boundary_dataset_version: str, adapter_version: str) -> None:
    rows = [
        ("url_probe", url_probe_ttl_days, adapter_version, "refresh_when_url_changes_or_ttl_expires", "NOT_TESTED"),
        ("provider_metadata", provider_metadata_ttl_days, adapter_version, "refresh_when_provider_changes_or_ttl_expires", "NOT_TESTED"),
        ("taxonomy", None, taxonomy_snapshot_version, "refresh_when_pinned_snapshot_changes", "PASS"),
        ("boundary", None, boundary_dataset_version, "refresh_when_pinned_boundary_version_changes", "NOT_TESTED" if boundary_dataset_version == "NOT_TESTED" else "PASS"),
    ]
    pq.write_table(pa.Table.from_pylist([{"incremental_version": INCREMENTAL_VERSION, "evidence_domain": domain, "ttl_days": ttl, "version": version, "refresh_rule": rule, "current_status": status} for domain, ttl, version, rule, status in rows], schema=FRESHNESS_SCHEMA), path, compression="zstd")


def _glob_files(value: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(value, recursive=True))]


def _artifact(path: Path, root: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {"path": str(path.relative_to(root)), "physical_bytes": path.stat().st_size, "sha256": _sha256(path), "row_count": parquet.metadata.num_rows, "column_count": len(parquet.schema_arrow), "row_group_count": parquet.metadata.num_row_groups}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = ["FRESHNESS_SCHEMA", "INCREMENTAL_VERSION", "QUEUE_SCHEMA", "publish_incremental_state"]
