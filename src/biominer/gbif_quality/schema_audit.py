from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_quality.policy import FieldPolicy


SCHEMA_AUDIT_VERSION = "biominer-gbif-media-schema-audit/v1"
SCHEMA_INVENTORY_SCHEMA = pa.schema(
    [
        ("audit_version", pa.string()),
        ("schema_fingerprint", pa.string()),
        ("field_index", pa.int32()),
        ("field_name", pa.string()),
        ("nullable", pa.bool_()),
        ("arrow_type", pa.string()),
        ("parquet_physical_types", pa.list_(pa.string())),
        ("parquet_logical_types", pa.list_(pa.string())),
        ("encodings", pa.list_(pa.string())),
        ("row_group_count", pa.int32()),
        ("row_groups_with_statistics", pa.int32()),
        ("metadata_null_count", pa.int64()),
        ("compressed_bytes", pa.int64()),
        ("uncompressed_bytes", pa.int64()),
        ("recommended_valid_type", pa.string()),
        ("typed_derivative_recommended", pa.bool_()),
        ("original_preservation_policy", pa.string()),
        ("type_drift_status", pa.string()),
        ("full_value_scan_status", pa.string()),
    ]
)
ROW_GROUP_INVENTORY_SCHEMA = pa.schema(
    [
        ("audit_version", pa.string()),
        ("row_group_index", pa.int32()),
        ("row_count", pa.int64()),
        ("column_count", pa.int32()),
        ("compressed_bytes", pa.int64()),
        ("uncompressed_bytes", pa.int64()),
        ("minimum_chunk_offset", pa.int64()),
        ("maximum_chunk_end", pa.int64()),
        ("nonempty_status", pa.string()),
        ("column_chunk_bounds_status", pa.string()),
        ("schema_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class SchemaAudit:
    schema_version: str
    schema_fingerprint: str
    columns: tuple[dict[str, Any], ...]
    row_groups: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    validation: dict[str, bool]

    def column_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            list(self.columns), schema=SCHEMA_INVENTORY_SCHEMA
        )

    def row_group_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            list(self.row_groups), schema=ROW_GROUP_INVENTORY_SCHEMA
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_fingerprint": self.schema_fingerprint,
            "counts": self.counts,
            "validation": self.validation,
        }


def audit_parquet_schema(
    parquet_path: str | Path,
    policies: Iterable[FieldPolicy],
    *,
    full_value_scan_status: str = "NOT_TESTED",
) -> SchemaAudit:
    """Audit physical Parquet metadata without coercing or rewriting source data."""

    path = Path(parquet_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    policy_rows = tuple(policies)
    if tuple(item.field_name for item in policy_rows) != tuple(schema.names):
        raise ValueError("field policies do not exactly match the Parquet schema")
    if full_value_scan_status not in {"PASS", "FAIL", "NOT_TESTED"}:
        raise ValueError("invalid full value scan status")
    fingerprint = canonical_semantic_fingerprint(
        {
            "contract": SCHEMA_AUDIT_VERSION,
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in schema
            ],
        }
    )
    columns = tuple(
        _column_inventory(
            parquet,
            index,
            field,
            policy_rows[index],
            fingerprint,
            full_value_scan_status,
        )
        for index, field in enumerate(schema)
    )
    row_groups = tuple(
        _row_group_inventory(parquet, path.stat().st_size, index, len(schema))
        for index in range(parquet.metadata.num_row_groups)
    )
    row_group_rows = sum(row["row_count"] for row in row_groups)
    validation = {
        "schema_policy_coverage_complete": len(columns) == len(schema),
        "row_group_rows_match_metadata": row_group_rows == parquet.metadata.num_rows,
        "all_row_groups_nonempty": all(
            row["nonempty_status"] == "PASS" for row in row_groups
        ),
        "all_column_chunks_within_file": all(
            row["column_chunk_bounds_status"] == "PASS" for row in row_groups
        ),
        "all_row_group_schemas_match": all(
            row["schema_status"] == "PASS" for row in row_groups
        ),
        "no_physical_type_drift": all(
            row["type_drift_status"] == "PASS" for row in columns
        ),
        "full_value_scan_completed": full_value_scan_status == "PASS",
    }
    if not all(validation.values()):
        raise ValueError(f"schema audit validation failed: {validation}")
    return SchemaAudit(
        schema_version=SCHEMA_AUDIT_VERSION,
        schema_fingerprint=fingerprint,
        columns=columns,
        row_groups=row_groups,
        counts={
            "rows": parquet.metadata.num_rows,
            "columns": len(schema),
            "row_groups": parquet.metadata.num_row_groups,
            "row_group_rows": row_group_rows,
        },
        validation=validation,
    )


def _column_inventory(
    parquet: pq.ParquetFile,
    index: int,
    field: pa.Field,
    policy: FieldPolicy,
    fingerprint: str,
    full_value_scan_status: str,
) -> dict[str, Any]:
    physical_types: set[str] = set()
    logical_types: set[str] = set()
    encodings: set[str] = set()
    statistics_count = 0
    null_count = 0
    compressed = 0
    uncompressed = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        chunk = parquet.metadata.row_group(row_group_index).column(index)
        physical_types.add(str(chunk.physical_type))
        logical_types.add(str(chunk.statistics.logical_type) if chunk.statistics else "NONE")
        encodings.update(map(str, chunk.encodings))
        compressed += int(chunk.total_compressed_size)
        uncompressed += int(chunk.total_uncompressed_size)
        if chunk.statistics is not None:
            statistics_count += 1
            if chunk.statistics.null_count is not None:
                null_count += int(chunk.statistics.null_count)
    arrow_type = str(field.type)
    recommended = policy.valid_type
    typed_recommendation = recommended not in {
        "string",
        "controlled_vocabulary",
        "multi_value_vocabulary",
        "json_or_preserved_text",
    }
    return {
        "audit_version": SCHEMA_AUDIT_VERSION,
        "schema_fingerprint": fingerprint,
        "field_index": index,
        "field_name": field.name,
        "nullable": field.nullable,
        "arrow_type": arrow_type,
        "parquet_physical_types": sorted(physical_types),
        "parquet_logical_types": sorted(logical_types),
        "encodings": sorted(encodings),
        "row_group_count": parquet.metadata.num_row_groups,
        "row_groups_with_statistics": statistics_count,
        "metadata_null_count": null_count,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "recommended_valid_type": recommended,
        "typed_derivative_recommended": typed_recommendation,
        "original_preservation_policy": "PRESERVE_RAW_AND_DERIVE_SEPARATELY",
        "type_drift_status": "PASS" if len(physical_types) == 1 else "FAIL",
        "full_value_scan_status": full_value_scan_status,
    }


def _row_group_inventory(
    parquet: pq.ParquetFile,
    file_size: int,
    index: int,
    expected_columns: int,
) -> dict[str, Any]:
    row_group = parquet.metadata.row_group(index)
    starts: list[int] = []
    ends: list[int] = []
    compressed = 0
    uncompressed = 0
    chunks_valid = True
    for column_index in range(row_group.num_columns):
        chunk = row_group.column(column_index)
        candidates = [
            value
            for value in (
                chunk.dictionary_page_offset,
                chunk.data_page_offset,
                chunk.file_offset,
            )
            if value is not None and int(value) >= 0
        ]
        start = min(map(int, candidates)) if candidates else -1
        end = start + int(chunk.total_compressed_size)
        starts.append(start)
        ends.append(end)
        compressed += int(chunk.total_compressed_size)
        uncompressed += int(chunk.total_uncompressed_size)
        chunks_valid = chunks_valid and start >= 0 and start < end <= file_size
    return {
        "audit_version": SCHEMA_AUDIT_VERSION,
        "row_group_index": index,
        "row_count": row_group.num_rows,
        "column_count": row_group.num_columns,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "minimum_chunk_offset": min(starts) if starts else None,
        "maximum_chunk_end": max(ends) if ends else None,
        "nonempty_status": "PASS" if row_group.num_rows > 0 else "FAIL",
        "column_chunk_bounds_status": "PASS" if chunks_valid else "FAIL",
        "schema_status": (
            "PASS" if row_group.num_columns == expected_columns else "FAIL"
        ),
    }


__all__ = [
    "ROW_GROUP_INVENTORY_SCHEMA",
    "SCHEMA_AUDIT_VERSION",
    "SCHEMA_INVENTORY_SCHEMA",
    "SchemaAudit",
    "audit_parquet_schema",
]
