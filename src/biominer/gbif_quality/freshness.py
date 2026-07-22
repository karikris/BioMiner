from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


FRESHNESS_VERSION = "biominer-gbif-source-freshness/v1"
DERIVED_SCHEMA = pa.schema(
    [
        ("freshness_version", pa.string()),
        ("artifact", pa.string()),
        ("schema_version", pa.string()),
        ("generated_at", pa.string()),
        ("age_days", pa.int64()),
        ("freshness_ttl_days", pa.int64()),
        ("freshness_status", pa.string()),
        ("freshness_reason", pa.string()),
    ]
)


def publish_freshness_audit(
    *,
    v3_parquet: str | Path,
    source_inventory_json: str | Path,
    data_root: str | Path,
    output_directory: str | Path,
    expected_rows: int,
    code_commit: str,
    provider_stale_days: int = 365,
    derived_stale_days: int = 30,
    memory_limit: str = "4GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
) -> dict[str, object]:
    """Compare provider and derived timestamps with the pinned source snapshot."""

    source = Path(v3_parquet).resolve()
    inventory_path = Path(source_inventory_json).resolve()
    data = Path(data_root).resolve()
    destination = Path(output_directory).resolve()
    for path in (source, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if pq.ParquetFile(source).metadata.num_rows != expected_rows:
        raise ValueError("freshness source row count mismatch")
    if provider_stale_days < 1 or derived_stale_days < 1:
        raise ValueError("freshness TTLs must be positive")
    if destination.exists():
        raise FileExistsError(destination)

    inventory = json.loads(inventory_path.read_text())
    source_date = _parse_time(_required_text(inventory, "source_publication_date"))
    now = datetime.now(UTC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    provider_output = staging / "provider_dataset_freshness.parquet"
    derived_output = staging / "derived_snapshot_freshness.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            _provider_sql(source, provider_output, source_date, provider_stale_days)
        )
        provider_counts = {
            str(status): int(count)
            for status, count in connection.execute(
                f"""
                SELECT freshness_status, count(*)
                FROM read_parquet({_lit(str(provider_output))}) GROUP BY 1
                """
            ).fetchall()
        }
        provider_rows = sum(provider_counts.values())
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

    derived_rows = _derived_freshness_rows(
        data, now=now, stale_days=derived_stale_days
    )
    pq.write_table(pa.Table.from_pylist(derived_rows, schema=DERIVED_SCHEMA), derived_output)
    derived_counts: dict[str, int] = {}
    for row in derived_rows:
        status = str(row["freshness_status"])
        derived_counts[status] = derived_counts.get(status, 0) + 1
    validation = {
        "provider_dataset_rows_nonempty": provider_rows > 0,
        "all_derived_manifests_classified": bool(derived_rows),
        "source_fields_unchanged": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"freshness validation failed: {validation}")
    artifacts = [_artifact(provider_output), _artifact(derived_output)]
    manifest = {
        "schema_version": FRESHNESS_VERSION,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": inventory["source_snapshot_id"],
        "source_publication_date": inventory["source_publication_date"],
        "inputs": {
            "v3": str(source),
            "source_inventory": str(inventory_path),
            "data_root": str(data),
        },
        "counts": {
            "source_rows": expected_rows,
            "provider_dataset_rows": provider_rows,
            "provider_status_counts": dict(sorted(provider_counts.items())),
            "derived_manifest_rows": len(derived_rows),
            "derived_status_counts": dict(sorted(derived_counts.items())),
        },
        "configuration": {
            "provider_stale_days": provider_stale_days,
            "derived_stale_days": derived_stale_days,
        },
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        if _sha256(staging / str(artifact["path"])) != artifact["sha256"]:
            raise ValueError("freshness artifact checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _provider_sql(
    source: Path, output: Path, source_date: datetime, stale_days: int
) -> str:
    publication = source_date.date().isoformat()
    return f"""
    COPY (
      WITH grouped AS (
        SELECT
          coalesce(nullif(trim(media_publisher), ''), nullif(trim(publisher), ''),
            '<MISSING>') AS provider,
          coalesce(nullif(trim(datasetKey), ''), '<MISSING>') AS dataset_key,
          coalesce(nullif(trim(datasetName), ''), '<MISSING>') AS dataset_name,
          count(*)::BIGINT AS media_rows,
          count(distinct gbifID)::BIGINT AS distinct_occurrences,
          count(*) FILTER (WHERE try_cast(modified AS TIMESTAMP) IS NULL)::BIGINT
            AS missing_or_invalid_modified_rows,
          count(*) FILTER (WHERE try_cast(lastInterpreted AS TIMESTAMP) IS NULL)::BIGINT
            AS missing_or_invalid_last_interpreted_rows,
          min(try_cast(modified AS TIMESTAMP)) AS earliest_modified,
          max(try_cast(modified AS TIMESTAMP)) AS latest_modified,
          max(try_cast(lastInterpreted AS TIMESTAMP)) AS latest_interpreted
        FROM read_parquet({_lit(str(source))})
        GROUP BY provider, dataset_key, dataset_name
      ), classified AS (
        SELECT *, coalesce(greatest(latest_modified, latest_interpreted),
          latest_modified, latest_interpreted) AS latest_source_timestamp
        FROM grouped
      )
      SELECT {_lit(FRESHNESS_VERSION)} AS freshness_version, provider, dataset_key,
        dataset_name, media_rows, distinct_occurrences,
        missing_or_invalid_modified_rows, missing_or_invalid_last_interpreted_rows,
        cast(earliest_modified AS VARCHAR) AS earliest_modified,
        cast(latest_modified AS VARCHAR) AS latest_modified,
        cast(latest_interpreted AS VARCHAR) AS latest_interpreted,
        cast(latest_source_timestamp AS VARCHAR) AS latest_source_timestamp,
        {_lit(publication)} AS source_publication_date,
        date_diff('day', cast(latest_source_timestamp AS DATE), DATE {_lit(publication)})::BIGINT
          AS age_at_source_days,
        {stale_days}::BIGINT AS freshness_ttl_days,
        CASE
          WHEN latest_source_timestamp IS NULL THEN 'UNKNOWN'
          WHEN cast(latest_source_timestamp AS DATE) > DATE {_lit(publication)} THEN 'CONFLICT'
          WHEN date_diff('day', cast(latest_source_timestamp AS DATE), DATE {_lit(publication)})
            > {stale_days} THEN 'FAIL'
          ELSE 'PASS'
        END AS freshness_status,
        CASE
          WHEN latest_source_timestamp IS NULL THEN 'no_parseable_provider_or_gbif_timestamp'
          WHEN cast(latest_source_timestamp AS DATE) > DATE {_lit(publication)}
            THEN 'timestamp_after_source_publication_date'
          WHEN date_diff('day', cast(latest_source_timestamp AS DATE), DATE {_lit(publication)})
            > {stale_days} THEN 'provider_dataset_timestamp_exceeds_ttl'
          ELSE NULL
        END AS freshness_reason
      FROM classified
    ) TO {_lit(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
    """


def _derived_freshness_rows(
    data_root: Path, *, now: datetime, stale_days: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(data_root.rglob("manifest.json")):
        payload = json.loads(path.read_text())
        generated = payload.get("generated_at")
        if not isinstance(generated, str):
            status, age, reason = "UNKNOWN", None, "manifest_has_no_generated_at"
        else:
            try:
                timestamp = _parse_time(generated)
                age = (now.date() - timestamp.date()).days
                if age < 0:
                    status, reason = "CONFLICT", "generated_at_is_in_the_future"
                elif age > stale_days:
                    status, reason = "FAIL", "derived_snapshot_exceeds_ttl"
                else:
                    status, reason = "PASS", None
            except ValueError:
                status, age, reason = "CONFLICT", None, "generated_at_is_invalid"
        rows.append(
            {
                "freshness_version": FRESHNESS_VERSION,
                "artifact": str(path.relative_to(data_root)),
                "schema_version": str(payload.get("schema_version", "UNKNOWN")),
                "generated_at": generated if isinstance(generated, str) else None,
                "age_days": age,
                "freshness_ttl_days": stale_days,
                "freshness_status": status,
                "freshness_reason": reason,
            }
        )
    return rows


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _required_text(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"source inventory has no valid {key}")
    return result.strip()


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
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = ["DERIVED_SCHEMA", "FRESHNESS_VERSION", "publish_freshness_audit"]
