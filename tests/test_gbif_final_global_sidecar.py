from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.global_sidecar import (
    seal_global_keyed_dimension,
    seal_global_sidecar_window,
)


def _write(path: Path, values: dict[str, list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(values), path, row_group_size=2)
    return path


def test_global_sidecar_scans_dimension_once_then_slices_windows(
    tmp_path: Path,
) -> None:
    spine_one = _write(
        tmp_path / "spine-1.parquet",
        {
            "source_ordinal": [0, 1],
            "media_assertion_id": ["m0", "m1"],
        },
    )
    spine_two = _write(
        tmp_path / "spine-2.parquet",
        {
            "source_ordinal": [2, 3],
            "media_assertion_id": ["m2", "m3"],
        },
    )
    dimension_one = _write(
        tmp_path / "dimension-1.parquet",
        {
            "media_assertion_id": ["m3", "m0"],
            "status": ["three", "zero"],
        },
    )
    dimension_two = _write(
        tmp_path / "dimension-2.parquet",
        {
            "media_assertion_id": ["m2", "m1"],
            "status": ["two", "one"],
        },
    )
    global_part = tmp_path / "global.parquet"
    connection = duckdb.connect()
    try:
        receipt = seal_global_keyed_dimension(
            connection=connection,
            spine_parts=[spine_two, spine_one],
            dimension=[dimension_one, dimension_two],
            output_part=global_part,
            expected_rows=4,
            spine_key="media_assertion_id",
            dimension_key="media_assertion_id",
            output_column="quality",
            excluded_dimension_columns={"media_assertion_id"},
            required_match=True,
            dependencies={"dimension_sha256": "sha256:fixture"},
            batch_rows=2,
        )
    finally:
        connection.close()

    table = pq.read_table(global_part)
    assert receipt["artifact"]["row_count"] == 4
    assert table["source_ordinal"].to_pylist() == [0, 1, 2, 3]
    assert [
        value["status"] for value in table["quality"].to_pylist()
    ] == ["zero", "one", "two", "three"]

    window = tmp_path / "window.parquet"
    window_receipt = seal_global_sidecar_window(
        global_sidecar=global_part,
        output_part=window,
        source_start_ordinal=1,
        source_stop_ordinal=3,
        dependencies={"global_part_id": receipt["part_id"]},
        batch_rows=1,
    )
    assert window_receipt["artifact"]["row_count"] == 2
    window_table = pq.read_table(window)
    assert window_table["source_ordinal"].to_pylist() == [1, 2]
    assert [
        value["status"]
        for value in window_table["quality"].to_pylist()
    ] == ["one", "two"]


def test_global_sidecar_rejects_duplicates_and_required_misses(
    tmp_path: Path,
) -> None:
    spine = _write(
        tmp_path / "spine.parquet",
        {
            "source_ordinal": [0, 1],
            "media_assertion_id": ["m0", "m1"],
        },
    )
    duplicate = _write(
        tmp_path / "duplicate.parquet",
        {
            "media_assertion_id": ["m0", "m0"],
            "status": ["first", "second"],
        },
    )
    connection = duckdb.connect()
    try:
        with pytest.raises(
            RuntimeError,
            match="duplicate global dimension key",
        ):
            seal_global_keyed_dimension(
                connection=connection,
                spine_parts=[spine],
                dimension=duplicate,
                output_part=tmp_path / "duplicate-output.parquet",
                expected_rows=2,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="quality",
                excluded_dimension_columns={"media_assertion_id"},
                required_match=True,
                dependencies={"dimension_sha256": "sha256:duplicate"},
            )
    finally:
        connection.close()

    missing = _write(
        tmp_path / "missing.parquet",
        {
            "media_assertion_id": ["m0"],
            "status": ["first"],
        },
    )
    connection = duckdb.connect()
    try:
        with pytest.raises(
            RuntimeError,
            match="required global dimension match missing for 1 rows",
        ):
            seal_global_keyed_dimension(
                connection=connection,
                spine_parts=[spine],
                dimension=missing,
                output_part=tmp_path / "missing-output.parquet",
                expected_rows=2,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="quality",
                excluded_dimension_columns={"media_assertion_id"},
                required_match=True,
                dependencies={"dimension_sha256": "sha256:missing"},
            )
    finally:
        connection.close()
