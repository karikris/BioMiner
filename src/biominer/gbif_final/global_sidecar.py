from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import (
    seal_record_batches,
    validate_part_receipt,
)


GLOBAL_SIDECAR_VERSION = "gbif-final-global-sidecar/v1"


def seal_global_keyed_dimension(
    *,
    connection: duckdb.DuckDBPyConnection,
    spine_parts: Sequence[str | Path],
    dimension: str | Path | Sequence[str | Path],
    output_part: str | Path,
    expected_rows: int,
    spine_key: str,
    dimension_key: str,
    output_column: str,
    excluded_dimension_columns: set[str] | frozenset[str],
    required_match: bool,
    dependencies: Mapping[str, object],
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Scan one dimension once into a complete source-ordinal sidecar."""

    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not spine_key or not dimension_key or not output_column:
        raise ValueError("join and output columns must be non-empty")
    if not dependencies:
        raise ValueError("dependencies must be non-empty")
    if "global_sidecar_contract" in dependencies:
        raise ValueError(
            "dependencies may not override global_sidecar_contract"
        )

    resolved_spine = _resolve_paths(spine_parts, label="source spine")
    resolved_dimension = _resolve_paths(
        dimension if _is_path_sequence(dimension) else [dimension],
        label="dimension",
    )
    spine_files = [pq.ParquetFile(path) for path in resolved_spine]
    dimension_files = [pq.ParquetFile(path) for path in resolved_dimension]
    spine_schema = spine_files[0].schema_arrow
    dimension_schema = dimension_files[0].schema_arrow
    _require_identical_schemas(
        resolved_spine,
        spine_files,
        label="source spine",
    )
    _require_identical_schemas(
        resolved_dimension,
        dimension_files,
        label="dimension",
    )
    _require_columns(
        spine_schema,
        ("source_ordinal", spine_key),
        label="source spine",
    )
    _require_columns(
        dimension_schema,
        (dimension_key,),
        label="dimension",
    )
    if (
        spine_schema.field(spine_key).type
        != dimension_schema.field(dimension_key).type
    ):
        raise RuntimeError("global sidecar join key types differ")
    _validate_spine_scope(
        connection=connection,
        paths=resolved_spine,
        expected_rows=expected_rows,
    )

    excluded = set(excluded_dimension_columns)
    dimension_fields = [
        field
        for field in dimension_schema
        if field.name not in excluded
    ]
    if not dimension_fields:
        raise ValueError("global sidecar dimension has no output fields")
    contract = {
        "schema_version": GLOBAL_SIDECAR_VERSION,
        "operation": "global_keyed_dimension",
        "expected_rows": expected_rows,
        "spine_key": spine_key,
        "dimension_key": dimension_key,
        "output_column": output_column,
        "excluded_dimension_columns": sorted(excluded),
        "required_match": required_match,
        "spine_schema_fingerprint": _schema_fingerprint(spine_schema),
        "dimension_schema_fingerprint": _schema_fingerprint(
            dimension_schema
        ),
        "spine_part_count": len(resolved_spine),
        "dimension_part_count": len(resolved_dimension),
        "dimension_output_fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in dimension_fields
        ],
    }
    effective_dependencies = {
        **dict(dependencies),
        "global_sidecar_contract": {
            **contract,
            "contract_fingerprint": canonical_semantic_fingerprint(contract),
        },
    }

    output = Path(output_part).resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() != receipt_path.exists():
        raise RuntimeError(
            f"global sidecar is only partially sealed: {output}"
        )
    if output.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=effective_dependencies,
        )
        if (
            int(receipt["source_start_ordinal"]) != 0
            or int(receipt["source_stop_ordinal"]) != expected_rows
            or pq.ParquetFile(output).schema_arrow.names
            != ["source_ordinal", output_column]
        ):
            raise RuntimeError(f"global sidecar receipt is stale: {output}")
        return receipt

    spine_source = _source_parameter(resolved_spine)
    dimension_source = _source_parameter(resolved_dimension)
    duplicate = connection.execute(
        f"""
        SELECT {_quoted(dimension_key)}, count(*)
        FROM read_parquet(?)
        GROUP BY {_quoted(dimension_key)}
        HAVING count(*) > 1
        LIMIT 1
        """,
        [dimension_source],
    ).fetchone()
    if duplicate is not None:
        raise RuntimeError(
            "duplicate global dimension key: "
            f"key={duplicate[0]!r}, rows={int(duplicate[1])}"
        )
    missing = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet(?) AS s
            LEFT JOIN read_parquet(?) AS d
              ON s.{_quoted(spine_key)} = d.{_quoted(dimension_key)}
            WHERE d.{_quoted(dimension_key)} IS NULL
            """,
            [spine_source, dimension_source],
        ).fetchone()[0]
    )
    if required_match and missing:
        raise RuntimeError(
            f"required global dimension match missing for {missing} rows"
        )

    struct_fields = ", ".join(
        f"{_quoted(field.name)} := d.{_quoted(field.name)}"
        for field in dimension_fields
    )
    query = f"""
        SELECT
          s.source_ordinal,
          struct_pack({struct_fields}) AS {_quoted(output_column)}
        FROM read_parquet(?) AS s
        LEFT JOIN read_parquet(?) AS d
          ON s.{_quoted(spine_key)} = d.{_quoted(dimension_key)}
        ORDER BY s.source_ordinal
    """
    reader = connection.execute(
        query,
        [spine_source, dimension_source],
    ).to_arrow_reader(batch_size=batch_rows)
    batches = _validate_ordered_batches(
        reader,
        expected_rows=expected_rows,
    )
    return seal_record_batches(
        batches=batches,
        schema=reader.schema,
        part_path=output,
        source_start_ordinal=0,
        source_stop_ordinal=expected_rows,
        dependencies=effective_dependencies,
        row_group_size=batch_rows,
    )


def seal_global_sidecar_window(
    *,
    global_sidecar: str | Path,
    output_part: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    dependencies: Mapping[str, object],
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Slice one sealed global sidecar into a restartable ordinal window."""

    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not dependencies:
        raise ValueError("dependencies must be non-empty")
    source = Path(global_sidecar).resolve()
    source_receipt_path = source.with_suffix(
        source.suffix + ".receipt.json"
    )
    source_receipt = validate_part_receipt(source_receipt_path)
    if int(source_receipt["source_start_ordinal"]) != 0:
        raise RuntimeError("global sidecar does not begin at source ordinal zero")
    stop = source_stop_ordinal
    if stop > int(source_receipt["source_stop_ordinal"]):
        raise RuntimeError("sidecar window exceeds global source scope")

    contract = {
        "schema_version": GLOBAL_SIDECAR_VERSION,
        "operation": "global_sidecar_window",
        "global_sidecar_part_id": source_receipt["part_id"],
        "global_sidecar_sha256": source_receipt["artifact"][
            "physical_sha256"
        ],
        "source_start_ordinal": source_start_ordinal,
        "source_stop_ordinal": stop,
    }
    effective_dependencies = {
        **dict(dependencies),
        "global_sidecar_window_contract": {
            **contract,
            "contract_fingerprint": canonical_semantic_fingerprint(contract),
        },
    }
    output = Path(output_part).resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() != receipt_path.exists():
        raise RuntimeError(
            f"global sidecar window is only partially sealed: {output}"
        )
    if output.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=effective_dependencies,
        )
        if (
            int(receipt["source_start_ordinal"])
            != source_start_ordinal
            or int(receipt["source_stop_ordinal"]) != stop
            or pq.ParquetFile(output).schema_arrow
            != pq.ParquetFile(source).schema_arrow
        ):
            raise RuntimeError(
                f"global sidecar window receipt is stale: {output}"
            )
        return receipt

    parquet = pq.ParquetFile(source)
    batches = _iter_parquet_range(
        parquet,
        start=source_start_ordinal,
        stop=stop,
        batch_rows=batch_rows,
    )
    return seal_record_batches(
        batches=batches,
        schema=parquet.schema_arrow,
        part_path=output,
        source_start_ordinal=source_start_ordinal,
        source_stop_ordinal=stop,
        dependencies=effective_dependencies,
        row_group_size=batch_rows,
    )


def _validate_ordered_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    expected_rows: int,
) -> Iterator[pa.RecordBatch]:
    expected_ordinal = 0
    for batch in batches:
        if not batch.num_rows:
            continue
        ordinal_index = batch.schema.get_field_index("source_ordinal")
        if ordinal_index < 0:
            raise RuntimeError("global sidecar lacks source_ordinal")
        observed = batch.column(ordinal_index)
        expected = pa.array(
            range(expected_ordinal, expected_ordinal + batch.num_rows),
            type=pa.int64(),
        )
        if observed.type != pa.int64() or not pc.all(
            pc.equal(observed, expected)
        ).as_py():
            raise RuntimeError(
                "global sidecar source ordinals are not ordered and complete"
            )
        expected_ordinal += batch.num_rows
        yield batch
    if expected_ordinal != expected_rows:
        raise RuntimeError(
            "global sidecar row stream does not cover its source scope"
        )


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


def _validate_spine_scope(
    *,
    connection: duckdb.DuckDBPyConnection,
    paths: list[Path],
    expected_rows: int,
) -> None:
    row = connection.execute(
        """
        SELECT
          count(*),
          count(DISTINCT source_ordinal),
          min(source_ordinal),
          max(source_ordinal),
          sum(source_ordinal)
        FROM read_parquet(?)
        """,
        [_source_parameter(paths)],
    ).fetchone()
    expected_sum = expected_rows * (expected_rows - 1) // 2
    if row != (
        expected_rows,
        expected_rows,
        0,
        expected_rows - 1,
        expected_sum,
    ):
        raise RuntimeError("global source spine ordinals are incomplete")


def _resolve_paths(
    values: Sequence[str | Path],
    *,
    label: str,
) -> list[Path]:
    paths = [Path(value).resolve() for value in values]
    if not paths:
        raise ValueError(f"{label} paths must be non-empty")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths must be unique")
    return paths


def _is_path_sequence(
    value: str | Path | Sequence[str | Path],
) -> bool:
    return not isinstance(value, (str, Path))


def _source_parameter(paths: list[Path]) -> str | list[str]:
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


def _require_identical_schemas(
    paths: list[Path],
    files: list[pq.ParquetFile],
    *,
    label: str,
) -> None:
    expected = files[0].schema_arrow
    for path, parquet in zip(paths[1:], files[1:]):
        if parquet.schema_arrow != expected:
            raise RuntimeError(f"{label} schemas differ: {path}")


def _require_columns(
    schema: pa.Schema,
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [name for name in names if name not in schema.names]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _schema_fingerprint(schema: pa.Schema) -> str:
    digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "GLOBAL_SIDECAR_VERSION",
    "seal_global_keyed_dimension",
    "seal_global_sidecar_window",
]
