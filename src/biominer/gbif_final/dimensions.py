from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

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
SPECIES_ENRICHMENT_DIMENSION_VERSION = (
    "gbif-final-species-enrichment-dimension/v1"
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


def build_species_enrichment_dimension(
    *,
    source_parquet: str | Path,
    registry_dir: str | Path,
    output_path: str | Path,
    source_assertions_path: str | Path | None,
    producer_git_sha: str,
    row_group_size: int = 65_536,
) -> dict[str, Any]:
    """Build, verify, and seal the unique species/keyword dimension."""

    if not producer_git_sha.strip():
        raise ValueError("producer_git_sha must be non-empty")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    source = Path(source_parquet).resolve()
    registry = Path(registry_dir).resolve()
    output = Path(output_path).resolve()
    source_file = pq.ParquetFile(source)
    required_source = {"speciesKey", "species"}
    if not required_source.issubset(source_file.schema_arrow.names):
        raise RuntimeError(
            "species source lacks speciesKey or species"
        )
    registry_paths = {
        name: registry / name
        for name in (
            "taxa.parquet",
            "names.parquet",
            "species_paths.parquet",
        )
    }
    for path in registry_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    assertions = (
        Path(source_assertions_path).resolve()
        if source_assertions_path is not None
        else None
    )
    if assertions is not None and not assertions.is_file():
        raise FileNotFoundError(assertions)

    contract = {
        "schema_version": SPECIES_ENRICHMENT_DIMENSION_VERSION,
        "producer_git_sha": producer_git_sha,
        "source": _parquet_inventory(source),
        "registry": {
            name: _parquet_inventory(path)
            for name, path in registry_paths.items()
        },
        "source_assertions": (
            _parquet_inventory(assertions)
            if assertions is not None
            else {"present": False}
        ),
        "key_columns": ["dataset_species_key", "dataset_species"],
        "null_policy": "null_species_components_normalized_to_empty_string",
    }
    dependencies = {
        **contract,
        "contract_fingerprint": canonical_semantic_fingerprint(contract),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() != receipt_path.exists():
        raise RuntimeError(
            f"species enrichment dimension is partially sealed: {output}"
        )
    if output.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=dependencies,
        )
        _validate_species_dimension(output)
        return receipt

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    try:
        from biominer.gbif_final.pipeline import build_species_enrichments

        result = build_species_enrichments(
            source_parquet=source,
            registry_dir=registry,
            output_path=temporary,
            source_assertions_path=assertions,
        )
        if result.height <= 0:
            raise RuntimeError("species enrichment dimension would be empty")
        _validate_species_dimension(temporary)
        table = pq.read_table(temporary)
        return seal_record_batches(
            batches=table.to_batches(max_chunksize=row_group_size),
            schema=table.schema,
            part_path=output,
            source_start_ordinal=0,
            source_stop_ordinal=table.num_rows,
            dependencies=dependencies,
            row_group_size=row_group_size,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _validate_species_dimension(path: Path) -> None:
    connection = duckdb.connect()
    try:
        (
            rows,
            unique_keys,
            null_components,
        ) = connection.execute(
            """
            SELECT
              count(*),
              count(
                DISTINCT struct_pack(
                  speciesKey := dataset_species_key,
                  species := dataset_species
                )
              ),
              count(*) FILTER (
                WHERE dataset_species_key IS NULL
                   OR dataset_species IS NULL
              )
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    finally:
        connection.close()
    if int(rows) <= 0 or int(rows) != int(unique_keys):
        raise RuntimeError(
            "species enrichment dimension keys are not unique"
        )
    if int(null_components):
        raise RuntimeError(
            "species enrichment dimension retains null key components"
        )


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


def _parquet_inventory(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    if (
        parquet.metadata.num_rows <= 0
        or parquet.metadata.num_row_groups <= 0
        or sum(row_group_rows) != parquet.metadata.num_rows
        or any(rows <= 0 for rows in row_group_rows)
    ):
        raise RuntimeError(f"Parquet row groups are incomplete: {path}")
    return {
        "path": str(path),
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
    }


__all__ = [
    "DERIVED_ASSERTION_DIMENSION_VERSION",
    "SPECIES_ENRICHMENT_DIMENSION_VERSION",
    "build_derived_assertion_dimension",
    "build_species_enrichment_dimension",
]
