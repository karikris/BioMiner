from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
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


MATERIALIZED_WINDOW_VERSION = "gbif-final-materialized-window/v1"


def seal_temporal_enriched_window(
    *,
    connection: duckdb.DuckDBPyConnection,
    temporal_parquet: str | Path,
    aligned_part: str | Path,
    output_part: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    expanded_struct_fields: Mapping[str, Sequence[str]],
    dependencies: Mapping[str, object],
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Zip one temporal row range with its ordinal-aligned enrichments."""

    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not dependencies:
        raise ValueError("dependencies must be non-empty")
    if "materialized_window_contract" in dependencies:
        raise ValueError(
            "dependencies may not override materialized_window_contract"
        )

    start = source_start_ordinal
    stop = source_stop_ordinal
    expected_rows = stop - start
    temporal_path = Path(temporal_parquet).resolve()
    aligned_path = Path(aligned_part).resolve()
    output_path = Path(output_part).resolve()
    temporal = _open_parquet(temporal_path)
    aligned = _open_parquet(aligned_path)
    if stop > temporal.metadata.num_rows:
        raise RuntimeError(
            "materialized window exceeds temporal source rows: "
            f"stop={stop}, source_rows={temporal.metadata.num_rows}"
        )
    _validate_aligned_part(
        connection=connection,
        path=aligned_path,
        expected_rows=expected_rows,
        start=start,
        stop=stop,
    )

    expansions = {
        str(column): tuple(str(field) for field in fields)
        for column, fields in expanded_struct_fields.items()
    }
    if any(not column for column in expansions):
        raise ValueError("expanded struct column names must be non-empty")
    if any(
        not fields or len(fields) != len(set(fields))
        for fields in expansions.values()
    ):
        raise ValueError(
            "expanded struct field lists must be non-empty and unique"
        )
    aligned_output_fields = _aligned_output_fields(
        temporal_schema=temporal.schema_arrow,
        aligned_schema=aligned.schema_arrow,
        expansions=expansions,
    )
    output_schema = pa.schema(
        [*temporal.schema_arrow, *aligned_output_fields],
        metadata=temporal.schema_arrow.metadata,
    )
    contract = {
        "schema_version": MATERIALIZED_WINDOW_VERSION,
        "source_start_ordinal": start,
        "source_stop_ordinal": stop,
        "batch_rows": batch_rows,
        "expanded_struct_fields": {
            column: list(fields)
            for column, fields in sorted(expansions.items())
        },
        "temporal_schema_fingerprint": _schema_fingerprint(
            temporal.schema_arrow
        ),
        "aligned_schema_fingerprint": _schema_fingerprint(
            aligned.schema_arrow
        ),
        "output_schema_fingerprint": _schema_fingerprint(output_schema),
    }
    effective_dependencies = {
        **dict(dependencies),
        "materialized_window_contract": {
            **contract,
            "contract_fingerprint": canonical_semantic_fingerprint(contract),
        },
    }

    receipt_path = output_path.with_suffix(
        output_path.suffix + ".receipt.json"
    )
    if output_path.exists() != receipt_path.exists():
        raise RuntimeError(
            f"materialized window is only partially sealed: {output_path}"
        )
    if output_path.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=effective_dependencies,
        )
        if (
            int(receipt["source_start_ordinal"]) != start
            or int(receipt["source_stop_ordinal"]) != stop
            or int(receipt["artifact"]["row_count"]) != expected_rows
            or pq.ParquetFile(output_path).schema_arrow != output_schema
        ):
            raise RuntimeError(
                f"materialized window receipt is stale: {output_path}"
            )
        return receipt

    temporal_batches = _iter_parquet_range(
        temporal,
        start=start,
        stop=stop,
        batch_rows=batch_rows,
    )
    aligned_batches = aligned.iter_batches(batch_size=batch_rows)
    batches = _zip_enrichment_batches(
        temporal_batches=temporal_batches,
        aligned_batches=aligned_batches,
        aligned_schema=aligned.schema_arrow,
        output_schema=output_schema,
        expansions=expansions,
    )
    return seal_record_batches(
        batches=batches,
        schema=output_schema,
        part_path=output_path,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        dependencies=effective_dependencies,
        row_group_size=batch_rows,
    )


def _aligned_output_fields(
    *,
    temporal_schema: pa.Schema,
    aligned_schema: pa.Schema,
    expansions: Mapping[str, tuple[str, ...]],
) -> list[pa.Field]:
    if "source_ordinal" not in aligned_schema.names:
        raise RuntimeError("aligned enrichment lacks source_ordinal")
    unknown = sorted(set(expansions) - set(aligned_schema.names))
    if unknown:
        raise RuntimeError(
            f"expanded struct columns are absent: {unknown}"
        )

    output: list[pa.Field] = []
    for field in aligned_schema:
        if field.name == "source_ordinal":
            continue
        expanded = expansions.get(field.name)
        if expanded is None:
            output.append(field)
            continue
        if not pa.types.is_struct(field.type):
            raise RuntimeError(
                f"expanded aligned column is not a struct: {field.name}"
            )
        struct_type = field.type
        for child_name in expanded:
            child_index = struct_type.get_field_index(child_name)
            if child_index < 0:
                raise RuntimeError(
                    f"aligned struct {field.name} lacks field {child_name}"
                )
            child = struct_type[child_index]
            output.append(
                pa.field(
                    child.name,
                    child.type,
                    nullable=field.nullable or child.nullable,
                    metadata=child.metadata,
                )
            )

    names = [*temporal_schema.names, *(field.name for field in output)]
    duplicates = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicates:
        raise RuntimeError(
            f"materialized output column names collide: {duplicates}"
        )
    return output


def _iter_parquet_range(
    parquet: pq.ParquetFile,
    *,
    start: int,
    stop: int,
    batch_rows: int,
) -> Iterator[pa.RecordBatch]:
    row_group_start = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group_rows = parquet.metadata.row_group(
            row_group_index
        ).num_rows
        row_group_stop = row_group_start + row_group_rows
        overlap_start = max(start, row_group_start)
        overlap_stop = min(stop, row_group_stop)
        if overlap_start < overlap_stop:
            table = parquet.read_row_group(
                row_group_index,
                use_threads=True,
            ).slice(
                overlap_start - row_group_start,
                overlap_stop - overlap_start,
            )
            yield from table.to_batches(max_chunksize=batch_rows)
        row_group_start = row_group_stop
        if row_group_start >= stop:
            return


def _zip_enrichment_batches(
    *,
    temporal_batches: Iterable[pa.RecordBatch],
    aligned_batches: Iterable[pa.RecordBatch],
    aligned_schema: pa.Schema,
    output_schema: pa.Schema,
    expansions: Mapping[str, tuple[str, ...]],
) -> Iterator[pa.RecordBatch]:
    temporal = _BatchCursor(temporal_batches, label="temporal")
    aligned = _BatchCursor(aligned_batches, label="aligned enrichment")
    while temporal.remaining or aligned.remaining:
        if not temporal.remaining or not aligned.remaining:
            raise RuntimeError(
                "temporal and aligned batch streams have different lengths"
            )
        rows = min(temporal.remaining, aligned.remaining)
        temporal_batch = temporal.take(rows)
        aligned_batch = aligned.take(rows)
        arrays: list[pa.Array] = list(temporal_batch.columns)
        for field in aligned_schema:
            if field.name == "source_ordinal":
                continue
            column = aligned_batch.column(
                aligned_schema.get_field_index(field.name)
            )
            expanded = expansions.get(field.name)
            if expanded is None:
                arrays.append(column)
                continue
            for child_name in expanded:
                arrays.append(
                    column.field(field.type.get_field_index(child_name))
                )
        yield pa.RecordBatch.from_arrays(arrays, schema=output_schema)


class _BatchCursor:
    def __init__(
        self,
        batches: Iterable[pa.RecordBatch],
        *,
        label: str,
    ) -> None:
        self._batches = iter(batches)
        self._label = label
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self._exhausted = False
        self._advance()

    @property
    def remaining(self) -> int:
        if self._batch is None:
            return 0
        return self._batch.num_rows - self._offset

    def take(self, rows: int) -> pa.RecordBatch:
        if rows <= 0 or rows > self.remaining or self._batch is None:
            raise RuntimeError(
                f"{self._label} cannot provide requested rows: {rows}"
            )
        result = self._batch.slice(self._offset, rows)
        self._offset += rows
        if self._offset == self._batch.num_rows:
            self._advance()
        return result

    def _advance(self) -> None:
        self._batch = None
        self._offset = 0
        while not self._exhausted:
            try:
                candidate = next(self._batches)
            except StopIteration:
                self._exhausted = True
                return
            if candidate.num_rows:
                self._batch = candidate
                return


def _validate_aligned_part(
    *,
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    expected_rows: int,
    start: int,
    stop: int,
) -> None:
    row = connection.execute(
        """
        SELECT
          count(*) AS rows,
          count(DISTINCT source_ordinal) AS ordinals,
          min(source_ordinal) AS minimum_ordinal,
          max(source_ordinal) AS maximum_ordinal,
          sum(source_ordinal) AS ordinal_sum,
          count(*) FILTER (
            WHERE source_ordinal != ? + row_offset
          ) AS out_of_order
        FROM (
          SELECT
            source_ordinal,
            row_number() OVER () - 1 AS row_offset
          FROM read_parquet(?)
        )
        """,
        [start, str(path)],
    ).fetchone()
    expected_sum = (start + stop - 1) * expected_rows // 2
    if row != (
        expected_rows,
        expected_rows,
        start,
        stop - 1,
        expected_sum,
        0,
    ):
        raise RuntimeError(
            f"aligned enrichment source ordinals are incomplete: {path}"
        )


def _open_parquet(path: Path) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows <= 0:
        raise RuntimeError(f"Parquet input has no rows: {path}")
    if parquet.metadata.num_row_groups <= 0:
        raise RuntimeError(f"Parquet input has no row groups: {path}")
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    if (
        sum(row_group_rows) != parquet.metadata.num_rows
        or any(rows <= 0 for rows in row_group_rows)
    ):
        raise RuntimeError(f"Parquet input has incomplete row groups: {path}")
    return parquet


def _schema_fingerprint(schema: pa.Schema) -> str:
    digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "MATERIALIZED_WINDOW_VERSION",
    "seal_temporal_enriched_window",
]
