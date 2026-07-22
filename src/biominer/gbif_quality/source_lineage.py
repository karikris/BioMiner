from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq


SOURCE_LINEAGE_VERSION = "biominer-gbif-source-assertion-lineage/v2"


def publish_source_assertion_lineage(
    *,
    multimedia_parquet: str | Path,
    source_status_parquet: str | Path,
    source_inventory_json: str | Path,
    output_directory: str | Path,
    expected_rows: int,
    code_commit: str,
    partition_rows: int = 1_000_000,
    memory_limit: str = "6GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
) -> dict[str, object]:
    """Publish immutable source-row identity, location, and value hashes."""

    multimedia = Path(multimedia_parquet).resolve()
    status = Path(source_status_parquet).resolve()
    inventory_path = Path(source_inventory_json).resolve()
    destination = Path(output_directory).resolve()
    for path in (multimedia, status, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)
    if partition_rows < 1 or threads < 1 or expected_rows < 1:
        raise ValueError("row, partition, and thread counts must be positive")
    if pq.ParquetFile(multimedia).metadata.num_rows != expected_rows:
        raise ValueError("multimedia row count mismatch")
    if pq.ParquetFile(status).metadata.num_rows != expected_rows:
        raise ValueError("source status row count mismatch")

    inventory = json.loads(inventory_path.read_text())
    snapshot = _required_text(inventory, "source_snapshot_id")
    download_key = _required_text(inventory, "source_download_key")
    source_doi = inventory.get("source_doi")
    ingestion_timestamp = _required_text(inventory, "generated_at")
    multimedia_entry = next(
        row
        for row in inventory["artifacts"]
        if row["artifact_role"] == "multimedia_extension"
    )
    source_file_sha256 = _required_text(multimedia_entry, "sha256")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    parts = staging / "parts"
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    columns = pq.ParquetFile(multimedia).schema_arrow.names
    value_hash = _row_hash_expression(columns)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"""
            COPY (
              SELECT
                cast(floor(s.source_sort_position / {partition_rows}) AS INTEGER)
                  AS source_partition,
                {_lit(SOURCE_LINEAGE_VERSION)} AS lineage_version,
                {_lit(snapshot)} AS source_snapshot_id,
                {_lit(download_key)} AS source_download_key,
                {_lit(source_doi)} AS source_doi,
                'multimedia.txt' AS source_file,
                {_lit(source_file_sha256)} AS source_file_sha256,
                s.source_sort_position AS source_row_number,
                s.source_row_id,
                'sha256:' || sha256({value_hash}) AS source_value_hash,
                {_lit(ingestion_timestamp)} AS ingestion_timestamp,
                s.gbifID,
                s.media_join_status,
                s.v3_funnel_status,
                s.exclusion_reason
              FROM read_parquet({_lit(str(multimedia))}) m
              POSITIONAL JOIN read_parquet({_lit(str(status))}) s
            ) TO {_lit(str(parts))} (
              FORMAT PARQUET,
              COMPRESSION ZSTD,
              PARTITION_BY(source_partition),
              ROW_GROUP_SIZE 250000,
              FILENAME_PATTERN 'part-{{i}}'
            )
            """
        )
        output_glob = str(parts / "**/*.parquet")
        observed = connection.execute(
            f"""
            SELECT count(*), count(distinct source_row_id),
              count(*) FILTER (WHERE source_value_hash IS NULL),
              count(*) FILTER (
                WHERE source_row_number < 0 OR source_snapshot_id <> {_lit(snapshot)}
              )
            FROM read_parquet({_lit(output_glob)})
            """
        ).fetchone()
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

    part_files = sorted(parts.glob("**/*.parquet"))
    validation = {
        "rows_match": int(observed[0]) == expected_rows,
        "source_row_ids_unique": int(observed[1]) == expected_rows,
        "all_rows_have_source_value_hash": int(observed[2]) == 0,
        "locations_and_snapshot_valid": int(observed[3]) == 0,
        "source_fields_unchanged": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"source lineage validation failed: {validation}")
    artifacts = [_artifact(path, staging) for path in part_files]
    manifest = {
        "schema_version": SOURCE_LINEAGE_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": snapshot,
        "source_download_key": download_key,
        "source_doi": source_doi,
        "source_ingestion_timestamp": ingestion_timestamp,
        "inputs": {
            "multimedia": str(multimedia),
            "source_status": str(status),
            "source_inventory": str(inventory_path),
        },
        "counts": {"rows": expected_rows, "parts": len(part_files)},
        "configuration": {
            "partition_rows": partition_rows,
            "value_hash_columns": columns,
        },
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        if _sha256(staging / str(artifact["path"])) != artifact["sha256"]:
            raise ValueError("source lineage artifact checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _row_hash_expression(columns: list[str]) -> str:
    values = [
        f"coalesce(cast(m.{_ident(name)} AS VARCHAR), '<NULL>')" for name in columns
    ]
    return "concat_ws(chr(31), " + ", ".join(values) + ")"


def _required_text(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"source inventory has no valid {key}")
    return result.strip()


def _artifact(path: Path, root: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.relative_to(root)),
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


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _lit(value: object | None) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


__all__ = ["SOURCE_LINEAGE_VERSION", "publish_source_assertion_lineage"]
