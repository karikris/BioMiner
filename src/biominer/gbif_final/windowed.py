from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any, Mapping

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import (
    seal_record_batches,
    validate_part_receipt,
)


WINDOWED_DIMENSION_VERSION = "gbif-final-windowed-dimension/v1"
_KEYS_TABLE = "_gbif_final_window_keys"
_DIMENSION_TABLE = "_gbif_final_window_dimension"


def seal_keyed_dimension_window(
    *,
    connection: duckdb.DuckDBPyConnection,
    spine_part: str | Path,
    dimension: str | Path | Sequence[str | Path],
    output_part: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    spine_key: str,
    dimension_key: str,
    output_column: str,
    excluded_dimension_columns: set[str] | frozenset[str],
    required_match: bool,
    dependencies: Mapping[str, object],
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Join one source-ordinal window to a slim keyed dimension and seal it."""

    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not spine_key:
        raise ValueError("spine_key must be non-empty")
    if not dimension_key:
        raise ValueError("dimension_key must be non-empty")
    if not output_column:
        raise ValueError("output_column must be non-empty")
    if not dependencies:
        raise ValueError("dependencies must be non-empty")
    if "windowed_dimension_contract" in dependencies:
        raise ValueError(
            "dependencies may not override windowed_dimension_contract"
        )

    spine_path = Path(spine_part).resolve()
    dimension_paths = _resolve_dimension_paths(dimension)
    dimension_source: str | list[str] = (
        str(dimension_paths[0])
        if len(dimension_paths) == 1
        else [str(path) for path in dimension_paths]
    )
    output_path = Path(output_part).resolve()
    spine_file = _open_parquet(spine_path)
    dimension_files = [_open_parquet(path) for path in dimension_paths]
    dimension_file = dimension_files[0]
    for path, parquet in zip(dimension_paths[1:], dimension_files[1:]):
        if parquet.schema_arrow != dimension_file.schema_arrow:
            raise RuntimeError(
                f"windowed dimension schemas differ: {path}"
            )
    _require_columns(
        spine_file.schema_arrow,
        ("source_ordinal", spine_key),
        label="source spine",
    )
    _require_columns(
        dimension_file.schema_arrow,
        (dimension_key,),
        label="dimension",
    )
    expected_rows = source_stop_ordinal - source_start_ordinal
    if spine_file.metadata.num_rows != expected_rows:
        raise RuntimeError(
            "source-spine part row count does not match its ordinal range: "
            f"rows={spine_file.metadata.num_rows}, expected={expected_rows}"
        )
    spine_type = spine_file.schema_arrow.field(spine_key).type
    dimension_type = dimension_file.schema_arrow.field(dimension_key).type
    if spine_type != dimension_type:
        raise RuntimeError(
            "windowed join key types differ: "
            f"spine={spine_type}, dimension={dimension_type}"
        )

    excluded = set(excluded_dimension_columns)
    dimension_fields = [
        field
        for field in dimension_file.schema_arrow
        if field.name not in excluded
    ]
    if not dimension_fields:
        raise ValueError("windowed dimension has no output fields")
    contract = {
        "schema_version": WINDOWED_DIMENSION_VERSION,
        "spine_key": spine_key,
        "dimension_key": dimension_key,
        "output_column": output_column,
        "excluded_dimension_columns": sorted(excluded),
        "required_match": required_match,
        "spine_schema_fingerprint": _schema_fingerprint(
            spine_file.schema_arrow
        ),
        "dimension_schema_fingerprint": _schema_fingerprint(
            dimension_file.schema_arrow
        ),
        "dimension_part_count": len(dimension_paths),
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
        "windowed_dimension_contract": {
            **contract,
            "contract_fingerprint": canonical_semantic_fingerprint(contract),
        },
    }

    receipt_path = output_path.with_suffix(
        output_path.suffix + ".receipt.json"
    )
    if output_path.exists() != receipt_path.exists():
        raise RuntimeError(
            f"windowed dimension is only partially sealed: {output_path}"
        )
    if output_path.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=effective_dependencies,
        )
        if (
            int(receipt["source_start_ordinal"]) != source_start_ordinal
            or int(receipt["source_stop_ordinal"]) != source_stop_ordinal
            or int(receipt["artifact"]["row_count"]) != expected_rows
        ):
            raise RuntimeError(
                f"windowed dimension receipt has a stale range: {output_path}"
            )
        output_schema = pq.ParquetFile(output_path).schema_arrow
        if output_schema.names != ["source_ordinal", output_column]:
            raise RuntimeError(
                f"windowed dimension receipt has a stale schema: {output_path}"
            )
        return receipt

    _drop_temporary_tables(connection)
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE {_quoted(_KEYS_TABLE)} AS
            SELECT
              source_ordinal,
              {_quoted(spine_key)} AS join_key
            FROM read_parquet(?)
            """,
            [str(spine_path)],
        )
        _validate_source_window(
            connection=connection,
            expected_rows=expected_rows,
            source_start_ordinal=source_start_ordinal,
            source_stop_ordinal=source_stop_ordinal,
        )
        selected_dimension_columns = [
            dimension_key,
            *[
                field.name
                for field in dimension_fields
                if field.name != dimension_key
            ],
        ]
        dimension_select = ", ".join(
            f"d.{_quoted(name)}"
            for name in selected_dimension_columns
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE {_quoted(_DIMENSION_TABLE)} AS
            SELECT {dimension_select}
            FROM read_parquet(?) AS d
            INNER JOIN (
              SELECT DISTINCT join_key
              FROM {_quoted(_KEYS_TABLE)}
              WHERE join_key IS NOT NULL
            ) AS k
              ON d.{_quoted(dimension_key)} = k.join_key
            """,
            [dimension_source],
        )
        duplicate = connection.execute(
            f"""
            SELECT {_quoted(dimension_key)}, count(*)
            FROM {_quoted(_DIMENSION_TABLE)}
            GROUP BY {_quoted(dimension_key)}
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise RuntimeError(
                "duplicate dimension key in source window: "
                f"key={duplicate[0]!r}, rows={int(duplicate[1])}"
            )
        missing = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM {_quoted(_KEYS_TABLE)} AS k
                LEFT JOIN {_quoted(_DIMENSION_TABLE)} AS d
                  ON k.join_key = d.{_quoted(dimension_key)}
                WHERE d.{_quoted(dimension_key)} IS NULL
                """
            ).fetchone()[0]
        )
        if required_match and missing:
            raise RuntimeError(
                "required dimension match missing for "
                f"{missing} source rows in range "
                f"[{source_start_ordinal}, {source_stop_ordinal})"
            )

        struct_fields = ", ".join(
            f"{_quoted(field.name)} := d.{_quoted(field.name)}"
            for field in dimension_fields
        )
        query = f"""
            SELECT
              k.source_ordinal,
              struct_pack({struct_fields}) AS {_quoted(output_column)}
            FROM {_quoted(_KEYS_TABLE)} AS k
            LEFT JOIN {_quoted(_DIMENSION_TABLE)} AS d
              ON k.join_key = d.{_quoted(dimension_key)}
            ORDER BY k.source_ordinal
        """
        reader = connection.execute(query).to_arrow_reader(
            batch_size=batch_rows
        )
        return seal_record_batches(
            batches=reader,
            schema=reader.schema,
            part_path=output_path,
            source_start_ordinal=source_start_ordinal,
            source_stop_ordinal=source_stop_ordinal,
            dependencies=effective_dependencies,
            row_group_size=batch_rows,
        )
    finally:
        _drop_temporary_tables(connection)


def seal_ordinal_aligned_window(
    *,
    connection: duckdb.DuckDBPyConnection,
    spine_part: str | Path,
    sidecar_parts: Mapping[str, str | Path],
    output_part: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    spine_columns: tuple[str, ...],
    dependencies: Mapping[str, object],
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Stream one exact source-ordinal window from validated sidecar columns."""

    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not sidecar_parts:
        raise ValueError("sidecar_parts must be non-empty")
    if not dependencies:
        raise ValueError("dependencies must be non-empty")
    if "ordinal_alignment_contract" in dependencies:
        raise ValueError(
            "dependencies may not override ordinal_alignment_contract"
        )
    if len(spine_columns) != len(set(spine_columns)):
        raise ValueError("spine_columns must be unique")
    if "source_ordinal" in spine_columns:
        raise ValueError("source_ordinal is added automatically")
    if "source_ordinal" in sidecar_parts:
        raise ValueError("source_ordinal cannot be a sidecar output column")

    start = source_start_ordinal
    stop = source_stop_ordinal
    expected_rows = stop - start
    spine_path = Path(spine_part).resolve()
    output_path = Path(output_part).resolve()
    spine_file = _open_parquet(spine_path)
    _require_columns(
        spine_file.schema_arrow,
        ("source_ordinal", *spine_columns),
        label="source spine",
    )
    sidecars: list[tuple[str, Path, pq.ParquetFile]] = []
    for output_column, raw_path in sidecar_parts.items():
        if not output_column:
            raise ValueError("sidecar output columns must be non-empty")
        path = Path(raw_path).resolve()
        parquet = _open_parquet(path)
        _require_columns(
            parquet.schema_arrow,
            ("source_ordinal", output_column),
            label=f"sidecar {output_column}",
        )
        sidecars.append((output_column, path, parquet))

    _validate_ordinal_file(
        connection=connection,
        path=spine_path,
        expected_rows=expected_rows,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        label="source spine",
    )
    for output_column, path, _ in sidecars:
        _validate_ordinal_file(
            connection=connection,
            path=path,
            expected_rows=expected_rows,
            source_start_ordinal=start,
            source_stop_ordinal=stop,
            label=f"sidecar {output_column}",
        )

    contract = {
        "schema_version": WINDOWED_DIMENSION_VERSION,
        "operation": "ordinal_aligned_window",
        "spine_columns": list(spine_columns),
        "spine_schema_fingerprint": _schema_fingerprint(
            spine_file.schema_arrow
        ),
        "sidecars": [
            {
                "output_column": output_column,
                "schema_fingerprint": _schema_fingerprint(
                    parquet.schema_arrow
                ),
                "output_type": str(
                    parquet.schema_arrow.field(output_column).type
                ),
            }
            for output_column, _, parquet in sidecars
        ],
    }
    effective_dependencies = {
        **dict(dependencies),
        "ordinal_alignment_contract": {
            **contract,
            "contract_fingerprint": canonical_semantic_fingerprint(contract),
        },
    }

    receipt_path = output_path.with_suffix(
        output_path.suffix + ".receipt.json"
    )
    if output_path.exists() != receipt_path.exists():
        raise RuntimeError(
            f"ordinal-aligned window is only partially sealed: {output_path}"
        )
    expected_names = [
        "source_ordinal",
        *spine_columns,
        *(output_column for output_column, _, _ in sidecars),
    ]
    if output_path.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=effective_dependencies,
        )
        if (
            int(receipt["source_start_ordinal"]) != start
            or int(receipt["source_stop_ordinal"]) != stop
            or int(receipt["artifact"]["row_count"]) != expected_rows
        ):
            raise RuntimeError(
                f"ordinal-aligned receipt has a stale range: {output_path}"
            )
        if pq.ParquetFile(output_path).schema_arrow.names != expected_names:
            raise RuntimeError(
                f"ordinal-aligned receipt has a stale schema: {output_path}"
            )
        return receipt

    select_values = [
        "s.source_ordinal",
        *(
            f"s.{_quoted(column)}"
            for column in spine_columns
        ),
        *(
            f"d{index}.{_quoted(output_column)} AS "
            f"{_quoted(output_column)}"
            for index, (output_column, _, _) in enumerate(sidecars)
        ),
    ]
    joins = [
        f"INNER JOIN read_parquet(?) AS d{index} USING (source_ordinal)"
        for index in range(len(sidecars))
    ]
    query = f"""
        SELECT {", ".join(select_values)}
        FROM read_parquet(?) AS s
        {" ".join(joins)}
        ORDER BY s.source_ordinal
    """
    reader = connection.execute(
        query,
        [str(spine_path), *(str(path) for _, path, _ in sidecars)],
    ).to_arrow_reader(batch_size=batch_rows)
    return seal_record_batches(
        batches=reader,
        schema=reader.schema,
        part_path=output_path,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        dependencies=effective_dependencies,
        row_group_size=batch_rows,
    )


def _validate_source_window(
    *,
    connection: duckdb.DuckDBPyConnection,
    expected_rows: int,
    source_start_ordinal: int,
    source_stop_ordinal: int,
) -> None:
    (
        rows,
        distinct_ordinals,
        minimum_ordinal,
        maximum_ordinal,
        ordinal_sum,
    ) = connection.execute(
        f"""
        SELECT
          count(*),
          count(DISTINCT source_ordinal),
          min(source_ordinal),
          max(source_ordinal),
          sum(source_ordinal)
        FROM {_quoted(_KEYS_TABLE)}
        """
    ).fetchone()
    expected_sum = (
        source_start_ordinal + source_stop_ordinal - 1
    ) * expected_rows // 2
    valid = (
        int(rows) == expected_rows
        and int(distinct_ordinals) == expected_rows
        and int(minimum_ordinal) == source_start_ordinal
        and int(maximum_ordinal) == source_stop_ordinal - 1
        and int(ordinal_sum) == expected_sum
    )
    if not valid:
        raise RuntimeError(
            "source-spine ordinal window is incomplete or duplicated"
        )


def _validate_ordinal_file(
    *,
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    expected_rows: int,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    label: str,
) -> None:
    (
        rows,
        distinct_ordinals,
        minimum_ordinal,
        maximum_ordinal,
        ordinal_sum,
    ) = connection.execute(
        """
        SELECT
          count(*),
          count(DISTINCT source_ordinal),
          min(source_ordinal),
          max(source_ordinal),
          sum(source_ordinal)
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    expected_sum = (
        source_start_ordinal + source_stop_ordinal - 1
    ) * expected_rows // 2
    valid = (
        int(rows) == expected_rows
        and int(distinct_ordinals) == expected_rows
        and int(minimum_ordinal) == source_start_ordinal
        and int(maximum_ordinal) == source_stop_ordinal - 1
        and int(ordinal_sum) == expected_sum
    )
    if not valid:
        raise RuntimeError(
            f"{label} source ordinals are incomplete or duplicated"
        )


def _drop_temporary_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute(
        f"DROP TABLE IF EXISTS {_quoted(_DIMENSION_TABLE)}"
    )
    connection.execute(
        f"DROP TABLE IF EXISTS {_quoted(_KEYS_TABLE)}"
    )


def _open_parquet(path: Path) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        parquet = pq.ParquetFile(path)
        row_group_rows = [
            parquet.metadata.row_group(index).num_rows
            for index in range(parquet.metadata.num_row_groups)
        ]
        if sum(row_group_rows) != parquet.metadata.num_rows:
            raise RuntimeError(f"Parquet row groups are incomplete: {path}")
        if any(rows <= 0 for rows in row_group_rows):
            raise RuntimeError(f"Parquet has an empty row group: {path}")
        return parquet
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"cannot reopen Parquet input: {path}") from error


def _resolve_dimension_paths(
    value: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    raw_values: Sequence[str | Path]
    if isinstance(value, (str, Path)):
        raw_values = [value]
    else:
        raw_values = value
    paths = tuple(Path(item).resolve() for item in raw_values)
    if not paths:
        raise ValueError("dimension must contain at least one Parquet path")
    if len(paths) != len(set(paths)):
        raise ValueError("dimension contains duplicate Parquet paths")
    return paths


def _require_columns(
    schema: pa.Schema,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [name for name in columns if name not in schema.names]
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
    "WINDOWED_DIMENSION_VERSION",
    "seal_keyed_dimension_window",
    "seal_ordinal_aligned_window",
]
