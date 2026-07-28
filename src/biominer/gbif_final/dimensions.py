from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import (
    seal_record_batches,
    validate_part_receipt,
)


DERIVED_ASSERTION_DIMENSION_VERSION = (
    "gbif-final-derived-assertion-dimension/v1"
)
ASSERTION_FIELDS = (
    "assertion_id",
    "target_field",
    "original_value",
    "derived_value",
    "evidence_source",
    "derivation_method",
    "derivation_rule_version",
    "confidence_class",
    "validation_status",
    "conflict_status",
    "reviewer_status",
)


def build_derived_assertion_dimension(
    *,
    source_assertions: str | Path,
    output_path: str | Path,
    producer_git_sha: str,
    batch_rows: int = 65_536,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> dict[str, Any]:
    """Aggregate the sparse assertion ledger to one ordered row per gbifID."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not producer_git_sha.strip():
        raise ValueError("producer_git_sha must be non-empty")
    source = Path(source_assertions).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    parquet = pq.ParquetFile(source)
    required = ("gbifID", *ASSERTION_FIELDS)
    missing = [
        name for name in required if name not in parquet.schema_arrow.names
    ]
    if missing:
        raise RuntimeError(
            "assertion ledger is missing required columns: "
            + ", ".join(missing)
        )
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    if (
        sum(row_group_rows) != parquet.metadata.num_rows
        or any(rows <= 0 for rows in row_group_rows)
    ):
        raise RuntimeError("assertion ledger row groups are incomplete")

    contract = {
        "schema_version": DERIVED_ASSERTION_DIMENSION_VERSION,
        "producer_git_sha": producer_git_sha,
        "source_physical_sha256": _sha256(source),
        "source_rows": parquet.metadata.num_rows,
        "source_schema_fingerprint": _schema_fingerprint(
            parquet.schema_arrow
        ),
        "group_key": "gbifID",
        "assertion_fields": list(ASSERTION_FIELDS),
        "ordering": ["gbifID", "target_field", "assertion_id"],
    }
    dependencies = {
        **contract,
        "contract_fingerprint": canonical_semantic_fingerprint(contract),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() != receipt_path.exists():
        raise RuntimeError(
            f"assertion dimension is only partially sealed: {output}"
        )
    if output.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=dependencies,
        )
        if pq.ParquetFile(output).schema_arrow.names != [
            "dimension_ordinal",
            "gbifID",
            "derived_quality_assertions",
        ]:
            raise RuntimeError(
                f"assertion dimension has a stale schema: {output}"
            )
        return receipt

    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET memory_limit=?", [memory_limit])
        group_count = int(
            connection.execute(
                """
                SELECT count(*)
                FROM (
                  SELECT gbifID
                  FROM read_parquet(?)
                  GROUP BY gbifID
                )
                """,
                [str(source)],
            ).fetchone()[0]
        )
        if group_count <= 0:
            raise RuntimeError("assertion dimension would be empty")
        struct_fields = ", ".join(
            f"{_quoted(field)} := {_quoted(field)}"
            for field in ASSERTION_FIELDS
        )
        query = f"""
            WITH grouped AS (
              SELECT
                gbifID,
                list(
                  struct_pack({struct_fields})
                  ORDER BY target_field, assertion_id
                ) AS derived_quality_assertions
              FROM read_parquet(?)
              GROUP BY gbifID
            )
            SELECT
              row_number() OVER (
                ORDER BY gbifID NULLS LAST
              ) - 1 AS dimension_ordinal,
              gbifID,
              derived_quality_assertions
            FROM grouped
            ORDER BY gbifID NULLS LAST
        """
        reader = connection.execute(
            query,
            [str(source)],
        ).to_arrow_reader(batch_size=batch_rows)
        return seal_record_batches(
            batches=reader,
            schema=reader.schema,
            part_path=output,
            source_start_ordinal=0,
            source_stop_ordinal=group_count,
            dependencies=dependencies,
            row_group_size=batch_rows,
        )
    finally:
        connection.close()


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _schema_fingerprint(schema: pa.Schema) -> str:
    digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "DERIVED_ASSERTION_DIMENSION_VERSION",
    "build_derived_assertion_dimension",
]
