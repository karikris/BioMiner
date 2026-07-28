from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.windowed import (
    seal_keyed_dimension_window,
    seal_ordinal_aligned_window,
)


def _write(path: Path, values: dict[str, list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(values),
        path,
        compression="zstd",
        row_group_size=2,
    )
    return path


def _spine(root: Path) -> Path:
    return _write(
        root / "spine.parquet",
        {
            "source_ordinal": [0, 1, 2, 3],
            "media_assertion_id": ["m1", "m2", "m3", "m4"],
            "gbifID": ["g1", "g1", "g2", "g3"],
        },
    )


def test_windowed_dimension_join_is_ordered_bounded_and_structured(
    tmp_path: Path,
) -> None:
    spine = _spine(tmp_path)
    dimension = _write(
        tmp_path / "dimension.parquet",
        {
            "media_assertion_id": ["m3", "m1", "m4", "m2", "unused"],
            "status": ["C", "A", "D", "B", "unused"],
            "score": [3, 1, 4, 2, 99],
        },
    )
    output = tmp_path / "quality.parquet"
    connection = duckdb.connect()
    try:
        receipt = seal_keyed_dimension_window(
            connection=connection,
            spine_part=spine,
            dimension=dimension,
            output_part=output,
            source_start_ordinal=0,
            source_stop_ordinal=4,
            spine_key="media_assertion_id",
            dimension_key="media_assertion_id",
            output_column="media_quality",
            excluded_dimension_columns={"media_assertion_id"},
            required_match=True,
            dependencies={"dimension_sha256": "sha256:dimension"},
            batch_rows=2,
        )
    finally:
        connection.close()

    assert receipt["artifact"]["row_count"] == 4
    assert receipt["artifact"]["row_group_rows"] == [2, 2]
    table = pq.read_table(output)
    assert table["source_ordinal"].to_pylist() == [0, 1, 2, 3]
    assert table["media_quality"].to_pylist() == [
        {"status": "A", "score": 1},
        {"status": "B", "score": 2},
        {"status": "C", "score": 3},
        {"status": "D", "score": 4},
    ]


@pytest.mark.parametrize(
    ("dimension_values", "message"),
    [
        (
            {
                "media_assertion_id": ["m1", "m1", "m2", "m3", "m4"],
                "status": ["A", "duplicate", "B", "C", "D"],
            },
            "duplicate dimension key",
        ),
        (
            {
                "media_assertion_id": ["m1", "m2", "m3"],
                "status": ["A", "B", "C"],
            },
            "required dimension match missing",
        ),
    ],
)
def test_windowed_dimension_join_fails_closed(
    tmp_path: Path,
    dimension_values: dict[str, list[object]],
    message: str,
) -> None:
    spine = _spine(tmp_path)
    dimension = _write(tmp_path / "dimension.parquet", dimension_values)
    output = tmp_path / "quality.parquet"
    connection = duckdb.connect()
    try:
        with pytest.raises(RuntimeError, match=message):
            seal_keyed_dimension_window(
                connection=connection,
                spine_part=spine,
                dimension=dimension,
                output_part=output,
                source_start_ordinal=0,
                source_stop_ordinal=4,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="media_quality",
                excluded_dimension_columns={"media_assertion_id"},
                required_match=True,
                dependencies={"dimension_sha256": "sha256:dimension"},
                batch_rows=2,
            )
    finally:
        connection.close()

    assert not output.exists()
    assert not output.with_suffix(".parquet.receipt.json").exists()


def test_windowed_optional_dimension_retains_unmatched_source_row(
    tmp_path: Path,
) -> None:
    spine = _spine(tmp_path)
    dimension = _write(
        tmp_path / "dimension.parquet",
        {
            "gbifID": ["g1", "g2"],
            "assertions": [["a"], ["b"]],
        },
    )
    output = tmp_path / "assertions.parquet"
    connection = duckdb.connect()
    try:
        seal_keyed_dimension_window(
            connection=connection,
            spine_part=spine,
            dimension=dimension,
            output_part=output,
            source_start_ordinal=0,
            source_stop_ordinal=4,
            spine_key="gbifID",
            dimension_key="gbifID",
            output_column="derived",
            excluded_dimension_columns={"gbifID"},
            required_match=False,
            dependencies={"dimension_sha256": "sha256:dimension"},
            batch_rows=3,
        )
    finally:
        connection.close()

    table = pq.read_table(output)
    assert table["source_ordinal"].to_pylist() == [0, 1, 2, 3]
    assert table["derived"].to_pylist() == [
        {"assertions": ["a"]},
        {"assertions": ["a"]},
        {"assertions": ["b"]},
        {"assertions": None},
    ]


def test_windowed_dimension_join_resumes_verified_part(
    tmp_path: Path,
) -> None:
    spine = _spine(tmp_path)
    dimension = _write(
        tmp_path / "dimension.parquet",
        {
            "media_assertion_id": ["m1", "m2", "m3", "m4"],
            "status": ["A", "B", "C", "D"],
        },
    )
    output = tmp_path / "quality.parquet"
    arguments = {
        "spine_part": spine,
        "dimension": dimension,
        "output_part": output,
        "source_start_ordinal": 0,
        "source_stop_ordinal": 4,
        "spine_key": "media_assertion_id",
        "dimension_key": "media_assertion_id",
        "output_column": "media_quality",
        "excluded_dimension_columns": {"media_assertion_id"},
        "required_match": True,
        "dependencies": {"dimension_sha256": "sha256:dimension"},
        "batch_rows": 2,
    }
    connection = duckdb.connect()
    try:
        first = seal_keyed_dimension_window(
            connection=connection,
            **arguments,
        )
        mtime = output.stat().st_mtime_ns
        second = seal_keyed_dimension_window(
            connection=connection,
            **arguments,
        )
    finally:
        connection.close()

    assert first == second
    assert output.stat().st_mtime_ns == mtime


def test_windowed_dimension_join_reads_multiple_parquet_parts(
    tmp_path: Path,
) -> None:
    spine = _spine(tmp_path)
    first = _write(
        tmp_path / "dimension" / "part-0.parquet",
        {
            "media_assertion_id": ["m4", "m1"],
            "status": ["D", "A"],
        },
    )
    second = _write(
        tmp_path / "dimension" / "part-1.parquet",
        {
            "media_assertion_id": ["m3", "m2"],
            "status": ["C", "B"],
        },
    )
    output = tmp_path / "quality.parquet"
    connection = duckdb.connect()
    try:
        seal_keyed_dimension_window(
            connection=connection,
            spine_part=spine,
            dimension=[second, first],
            output_part=output,
            source_start_ordinal=0,
            source_stop_ordinal=4,
            spine_key="media_assertion_id",
            dimension_key="media_assertion_id",
            output_column="media_quality",
            excluded_dimension_columns={"media_assertion_id"},
            required_match=True,
            dependencies={"dimension_sha256": ["sha256:second", "sha256:first"]},
            batch_rows=2,
        )
    finally:
        connection.close()

    assert [
        value["status"]
        for value in pq.read_table(output)["media_quality"].to_pylist()
    ] == ["A", "B", "C", "D"]


def test_ordinal_aligned_window_combines_sealed_sidecars(
    tmp_path: Path,
) -> None:
    spine = _write(
        tmp_path / "spine.parquet",
        {
            "source_ordinal": [4, 5, 6],
            "source_row_id": ["s4", "s5", "s6"],
            "media_assertion_id": ["m4", "m5", "m6"],
        },
    )
    media = _write(
        tmp_path / "media.parquet",
        {
            "source_ordinal": [4, 5, 6],
            "media_quality": [{"status": "A"}, {"status": "B"}, {"status": "C"}],
        },
    )
    rights = _write(
        tmp_path / "rights.parquet",
        {
            "source_ordinal": [4, 5, 6],
            "rights_quality": [{"allowed": True}, {"allowed": False}, None],
        },
    )
    output = tmp_path / "enrichment.parquet"
    connection = duckdb.connect()
    try:
        receipt = seal_ordinal_aligned_window(
            connection=connection,
            spine_part=spine,
            sidecar_parts={
                "media_quality": media,
                "rights_quality": rights,
            },
            output_part=output,
            source_start_ordinal=4,
            source_stop_ordinal=7,
            spine_columns=("source_row_id", "media_assertion_id"),
            dependencies={"run": "sha256:run"},
            batch_rows=2,
        )
    finally:
        connection.close()

    assert receipt["artifact"]["row_group_rows"] == [2, 1]
    table = pq.read_table(output)
    assert table.column_names == [
        "source_ordinal",
        "source_row_id",
        "media_assertion_id",
        "media_quality",
        "rights_quality",
    ]
    assert table["source_ordinal"].to_pylist() == [4, 5, 6]
    assert table["media_quality"].to_pylist() == [
        {"status": "A"},
        {"status": "B"},
        {"status": "C"},
    ]


def test_ordinal_aligned_window_rejects_sidecar_gap_even_if_count_matches(
    tmp_path: Path,
) -> None:
    spine = _write(
        tmp_path / "spine.parquet",
        {
            "source_ordinal": [0, 1, 2],
            "source_row_id": ["s0", "s1", "s2"],
        },
    )
    invalid = _write(
        tmp_path / "invalid.parquet",
        {
            "source_ordinal": [0, 0, 2],
            "quality": ["a", "duplicate", "c"],
        },
    )
    output = tmp_path / "enrichment.parquet"
    connection = duckdb.connect()
    try:
        with pytest.raises(
            RuntimeError,
            match="sidecar quality source ordinals are incomplete or duplicated",
        ):
            seal_ordinal_aligned_window(
                connection=connection,
                spine_part=spine,
                sidecar_parts={"quality": invalid},
                output_part=output,
                source_start_ordinal=0,
                source_stop_ordinal=3,
                spine_columns=("source_row_id",),
                dependencies={"run": "sha256:run"},
            )
    finally:
        connection.close()

    assert not output.exists()
