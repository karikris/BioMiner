from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Mapping
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.pipeline import FINAL_FILENAME, MANIFEST_FILENAME
from biominer.gbif_final.publication_audit import (
    validate_publication_audit,
)


LOCATOR_INDEX_VERSION = "gbif-final-locator-index/v1"
DATABASE_FILENAME = "gbif_media_locator.duckdb"
TABLE_NAME = "media_locator"
LOCATOR_COLUMNS = (
    "source_row_id",
    "media_assertion_id",
    "gbifID",
    "media_identifier",
    "media_references",
    "speciesKey",
    "species",
    "registry_taxon_key",
)
EXPECTED_INDEXES = {
    "idx_media_locator_gbif_id",
    "idx_media_locator_media_identifier",
    "idx_media_locator_registry_taxon_key",
    "idx_media_locator_species_key",
}


def build_final_locator_index(
    *,
    publication_directory: str | Path,
    publication_audit_directory: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    memory_limit: str = "8GB",
    threads: int = 4,
) -> dict[str, Any]:
    """Build a slim indexed locator without duplicating the enriched table."""

    publication = Path(publication_directory).resolve()
    audit = Path(publication_audit_directory).resolve()
    output = Path(output_directory).resolve()
    repository = Path(repository_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not memory_limit.strip():
        raise ValueError("memory_limit must not be empty")
    for protected in (publication, audit):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise RuntimeError(
                "locator index output overlaps publication evidence"
            )
    audit_manifest = validate_publication_audit(
        audit,
        repository_root=repository,
        require_dependencies=True,
        primary_publication_directory=publication,
    )
    final_path = publication / FINAL_FILENAME
    final_file = pq.ParquetFile(final_path)
    missing = sorted(
        set(LOCATOR_COLUMNS) - set(final_file.schema_arrow.names)
    )
    if missing:
        raise RuntimeError(
            "final publication lacks locator columns: "
            + ", ".join(missing)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid4().hex}.staging"
    temporary = output.parent / f".{output.name}.{uuid4().hex}.duckdb-tmp"
    staging.mkdir()
    temporary.mkdir()
    database = staging / DATABASE_FILENAME
    try:
        connection = duckdb.connect(str(database))
        try:
            connection.execute("SET memory_limit=?", [memory_limit])
            connection.execute(f"SET threads={int(threads)}")
            connection.execute("SET temp_directory=?", [str(temporary)])
            projected = ", ".join(
                _quoted(column) for column in LOCATOR_COLUMNS
            )
            connection.execute(
                f"""
                CREATE TABLE {TABLE_NAME} AS
                SELECT {projected}
                FROM read_parquet(?)
                """,
                [str(final_path)],
            )
            connection.execute(
                f"""
                CREATE INDEX idx_media_locator_media_identifier
                ON {TABLE_NAME}(media_identifier)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_media_locator_gbif_id
                ON {TABLE_NAME}(gbifID)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_media_locator_species_key
                ON {TABLE_NAME}(speciesKey)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_media_locator_registry_taxon_key
                ON {TABLE_NAME}(registry_taxon_key)
                """
            )
            connection.execute(f"ANALYZE {TABLE_NAME}")
            connection.execute("CHECKPOINT")
            evidence = _database_evidence(
                connection,
                expected_rows=final_file.metadata.num_rows,
            )
        finally:
            connection.close()
        _fsync_file(database)

        reopened = duckdb.connect(str(database), read_only=True)
        try:
            reopened_evidence = _database_evidence(
                reopened,
                expected_rows=final_file.metadata.num_rows,
            )
        finally:
            reopened.close()
        if _stable_database_evidence(
            reopened_evidence
        ) != _stable_database_evidence(evidence):
            raise RuntimeError("locator evidence changed after reopen")

        recorded_final = audit_manifest["primary_publication"][
            "final_artifact"
        ]
        manifest: dict[str, Any] = {
            "schema_version": LOCATOR_INDEX_VERSION,
            "generated_at": _timestamp(),
            "audit_git_commit": audit_manifest["audit_git_commit"],
            "producer_git_sha": audit_manifest["producer_git_sha"],
            "input": {
                "publication_directory": str(publication),
                "publication_manifest_path": str(
                    publication / MANIFEST_FILENAME
                ),
                "publication_manifest_sha256": _sha256(
                    publication / MANIFEST_FILENAME
                ),
                "final_artifact_path": str(final_path),
                "final_artifact_sha256": recorded_final[
                    "physical_sha256"
                ],
                "final_rows": recorded_final["row_count"],
                "publication_audit_directory": str(audit),
                "publication_audit_manifest_sha256": _sha256(
                    audit / MANIFEST_FILENAME
                ),
                "publication_audit_fingerprint": audit_manifest[
                    "manifest_fingerprint"
                ],
            },
            "database": {
                "path": DATABASE_FILENAME,
                "physical_bytes": database.stat().st_size,
                "physical_sha256": _sha256(database),
                "engine": "DuckDB",
                "engine_version": duckdb.__version__,
                "table": TABLE_NAME,
                **evidence,
            },
            "validation": {
                "publication_audit_revalidated": True,
                "final_rows_preserved": (
                    evidence["row_count"]
                    == final_file.metadata.num_rows
                ),
                "stable_identifiers_non_null": (
                    evidence["null_source_row_ids"] == 0
                    and evidence["null_media_assertion_ids"] == 0
                ),
                "stable_identifiers_unique": (
                    evidence["distinct_source_row_ids"]
                    == final_file.metadata.num_rows
                    and evidence["distinct_media_assertion_ids"]
                    == final_file.metadata.num_rows
                ),
                "locator_columns_exact": (
                    evidence["columns"] == list(LOCATOR_COLUMNS)
                ),
                "all_expected_indexes_present": (
                    set(evidence["index_names"]) == EXPECTED_INDEXES
                ),
                "sample_url_lookup_returns_rows": (
                    evidence["benchmarks"]["url_lookup"]["result_rows"] > 0
                ),
                "sample_gbif_lookup_returns_rows": (
                    evidence["benchmarks"]["gbif_lookup"]["result_rows"] > 0
                ),
                "warm_url_lookup_below_100ms": (
                    evidence["benchmarks"]["url_lookup"]["median_ms"]
                    < 100.0
                ),
                "database_reopened_unchanged": True,
                "manifest_written_last": True,
            },
            "query_examples": {
                "gbif_ids_for_url": (
                    "SELECT gbifID FROM media_locator "
                    "WHERE media_identifier = ?"
                ),
                "urls_for_occurrence": (
                    "SELECT media_identifier FROM media_locator "
                    "WHERE gbifID = ?"
                ),
                "urls_for_species_key": (
                    "SELECT media_identifier FROM media_locator "
                    "WHERE speciesKey = ?"
                ),
            },
            "policy": {
                "full_enriched_rows_remain_only_in_parquet": True,
                "database_contains_locator_columns_only": True,
                "create_only": True,
                "manifest_written_last": True,
            },
        }
        if not all(manifest["validation"].values()):
            raise RuntimeError(
                f"locator index validation failed: {manifest['validation']}"
            )
        manifest["manifest_fingerprint"] = (
            canonical_semantic_fingerprint(manifest)
        )
        _write_json_create_only(staging / MANIFEST_FILENAME, manifest)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_final_locator_index(
    *,
    index_directory: str | Path,
    publication_audit_directory: str | Path,
    repository_root: str | Path,
    publication_directory: str | Path | None = None,
    require_dependencies: bool = False,
) -> dict[str, Any]:
    """Revalidate a local or transferred locator and its final publication."""

    index = Path(index_directory).resolve()
    audit = Path(publication_audit_directory).resolve()
    expected_files = {
        index / DATABASE_FILENAME,
        index / MANIFEST_FILENAME,
    }
    observed_files = {
        path.resolve() for path in index.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError("locator index file inventory is not exact")
    manifest_path = index / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != LOCATOR_INDEX_VERSION:
        raise RuntimeError("locator index schema version differs")
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_fingerprint"
    }
    if manifest.get(
        "manifest_fingerprint"
    ) != canonical_semantic_fingerprint(body):
        raise RuntimeError("locator index manifest fingerprint mismatch")
    if not all(
        value is True for value in manifest["validation"].values()
    ):
        raise RuntimeError("locator index validation is not PASS")
    database = index / DATABASE_FILENAME
    if manifest["database"]["physical_sha256"] != _sha256(database):
        raise RuntimeError("locator database checksum mismatch")
    if manifest["database"]["physical_bytes"] != database.stat().st_size:
        raise RuntimeError("locator database byte count mismatch")
    if manifest_path.stat().st_mtime_ns < database.stat().st_mtime_ns:
        raise RuntimeError("locator manifest was not written last")

    audit_manifest = validate_publication_audit(
        audit,
        repository_root=repository_root,
        require_dependencies=require_dependencies,
        primary_publication_directory=publication_directory,
    )
    if manifest["input"][
        "publication_audit_fingerprint"
    ] != audit_manifest["manifest_fingerprint"]:
        raise RuntimeError("locator publication audit binding differs")
    if manifest["input"][
        "publication_audit_manifest_sha256"
    ] != _sha256(audit / MANIFEST_FILENAME):
        raise RuntimeError("locator publication audit checksum differs")

    connection = duckdb.connect(str(database), read_only=True)
    try:
        evidence = _database_evidence(
            connection,
            expected_rows=int(manifest["database"]["row_count"]),
        )
    finally:
        connection.close()
    recorded = manifest["database"]
    for field in (
        "row_count",
        "columns",
        "null_source_row_ids",
        "null_media_assertion_ids",
        "distinct_source_row_ids",
        "distinct_media_assertion_ids",
        "nonblank_image_url_rows",
        "distinct_image_urls",
        "distinct_gbif_ids",
        "index_names",
    ):
        if evidence[field] != recorded[field]:
            raise RuntimeError(
                f"locator database evidence differs: {field}"
            )
    return manifest


def _database_evidence(
    connection: duckdb.DuckDBPyConnection,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    counts = connection.execute(
        f"""
        SELECT
          count(*),
          count(*) FILTER (WHERE source_row_id IS NULL),
          count(*) FILTER (WHERE media_assertion_id IS NULL),
          count(DISTINCT source_row_id),
          count(DISTINCT media_assertion_id),
          count(NULLIF(trim(CAST(media_identifier AS VARCHAR)), '')),
          count(DISTINCT NULLIF(trim(CAST(media_identifier AS VARCHAR)), '')),
          count(DISTINCT gbifID)
        FROM {TABLE_NAME}
        """
    ).fetchone()
    if int(counts[0]) != expected_rows:
        raise RuntimeError("locator row count differs from final publication")
    columns = [
        str(row[0])
        for row in connection.execute(
            f"DESCRIBE {TABLE_NAME}"
        ).fetchall()
    ]
    indexes = sorted(
        str(row[0])
        for row in connection.execute(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = ?
            ORDER BY index_name
            """,
            [TABLE_NAME],
        ).fetchall()
    )
    sample = connection.execute(
        f"""
        SELECT media_identifier, gbifID
        FROM {TABLE_NAME}
        WHERE NULLIF(trim(CAST(media_identifier AS VARCHAR)), '') IS NOT NULL
          AND gbifID IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if sample is None:
        raise RuntimeError("locator database has no benchmarkable URL")
    benchmarks = {
        "url_lookup": _benchmark(
            connection,
            f"""
            SELECT gbifID
            FROM {TABLE_NAME}
            WHERE media_identifier = ?
            """,
            str(sample[0]),
        ),
        "gbif_lookup": _benchmark(
            connection,
            f"""
            SELECT media_identifier
            FROM {TABLE_NAME}
            WHERE gbifID = ?
            """,
            str(sample[1]),
        ),
    }
    return {
        "row_count": int(counts[0]),
        "columns": columns,
        "null_source_row_ids": int(counts[1]),
        "null_media_assertion_ids": int(counts[2]),
        "distinct_source_row_ids": int(counts[3]),
        "distinct_media_assertion_ids": int(counts[4]),
        "nonblank_image_url_rows": int(counts[5]),
        "distinct_image_urls": int(counts[6]),
        "distinct_gbif_ids": int(counts[7]),
        "index_names": indexes,
        "benchmarks": benchmarks,
    }


def _benchmark(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    value: str,
) -> dict[str, object]:
    connection.execute(query, [value]).fetchall()
    timings: list[float] = []
    result_rows = 0
    for _ in range(7):
        started = time.perf_counter()
        result_rows = len(connection.execute(query, [value]).fetchall())
        timings.append(1_000 * (time.perf_counter() - started))
    return {
        "runs": len(timings),
        "result_rows": result_rows,
        "minimum_ms": min(timings),
        "median_ms": statistics.median(timings),
        "maximum_ms": max(timings),
    }


def _stable_database_evidence(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in evidence.items()
        if key != "benchmarks"
    }


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_create_only(
    path: Path,
    value: Mapping[str, object],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "DATABASE_FILENAME",
    "LOCATOR_COLUMNS",
    "LOCATOR_INDEX_VERSION",
    "TABLE_NAME",
    "build_final_locator_index",
    "validate_final_locator_index",
]
