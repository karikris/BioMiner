from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.materialize import seal_temporal_enriched_window


def test_temporal_enriched_window_reads_only_requested_range(
    tmp_path: Path,
) -> None:
    temporal = tmp_path / "temporal.parquet"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["0", "1", "2", "3", "4"],
                "species": ["A", "B", "C", "D", "E"],
            }
        ),
        temporal,
        row_group_size=2,
    )
    aligned = tmp_path / "aligned.parquet"
    pq.write_table(
        pa.table(
            {
                "source_ordinal": [1, 2, 3],
                "source_row_id": ["row-1", "row-2", "row-3"],
                "media_quality": [
                    {"status": "ok"},
                    {"status": "review"},
                    {"status": "ok"},
                ],
                "derived_quality": [
                    {"derived_quality_assertions": ["a"]},
                    {"derived_quality_assertions": None},
                    {"derived_quality_assertions": ["b", "c"]},
                ],
                "species_enrichment": [
                    {
                        "registry_match_status": "matched",
                        "flickr_query_terms": ["one"],
                    },
                    {
                        "registry_match_status": "unmatched",
                        "flickr_query_terms": [],
                    },
                    {
                        "registry_match_status": "matched",
                        "flickr_query_terms": ["three"],
                    },
                ],
            }
        ),
        aligned,
        row_group_size=1,
    )
    output = tmp_path / "final-part.parquet"
    connection = duckdb.connect()
    try:
        receipt = seal_temporal_enriched_window(
            connection=connection,
            temporal_parquet=temporal,
            aligned_part=aligned,
            output_part=output,
            source_start_ordinal=1,
            source_stop_ordinal=4,
            expanded_struct_fields={
                "derived_quality": ("derived_quality_assertions",),
                "species_enrichment": (
                    "registry_match_status",
                    "flickr_query_terms",
                ),
            },
            dependencies={"temporal_sha256": "sha256:temporal"},
            batch_rows=2,
        )
    finally:
        connection.close()

    table = pq.read_table(output)
    assert receipt["artifact"]["row_count"] == 3
    assert table.schema.names == [
        "gbifID",
        "species",
        "source_row_id",
        "media_quality",
        "derived_quality_assertions",
        "registry_match_status",
        "flickr_query_terms",
    ]
    assert table["gbifID"].to_pylist() == ["1", "2", "3"]
    assert table["derived_quality_assertions"].to_pylist() == [
        ["a"],
        None,
        ["b", "c"],
    ]

    connection = duckdb.connect()
    try:
        resumed = seal_temporal_enriched_window(
            connection=connection,
            temporal_parquet=temporal,
            aligned_part=aligned,
            output_part=output,
            source_start_ordinal=1,
            source_stop_ordinal=4,
            expanded_struct_fields={
                "derived_quality": ("derived_quality_assertions",),
                "species_enrichment": (
                    "registry_match_status",
                    "flickr_query_terms",
                ),
            },
            dependencies={"temporal_sha256": "sha256:temporal"},
            batch_rows=2,
        )
    finally:
        connection.close()
    assert resumed["part_id"] == receipt["part_id"]


@pytest.mark.parametrize(
    "ordinals",
    ([0, 2], [1, 0]),
)
def test_temporal_enriched_window_rejects_incomplete_ordinals(
    tmp_path: Path,
    ordinals: list[int],
) -> None:
    temporal = tmp_path / "temporal.parquet"
    pq.write_table(pa.table({"gbifID": ["0", "1"]}), temporal)
    aligned = tmp_path / "aligned.parquet"
    pq.write_table(
        pa.table(
            {
                "source_ordinal": ordinals,
                "source_row_id": ["row-0", "row-1"],
            }
        ),
        aligned,
    )

    connection = duckdb.connect()
    try:
        with pytest.raises(
            RuntimeError,
            match="source ordinals are incomplete",
        ):
            seal_temporal_enriched_window(
                connection=connection,
                temporal_parquet=temporal,
                aligned_part=aligned,
                output_part=tmp_path / "output.parquet",
                source_start_ordinal=0,
                source_stop_ordinal=2,
                expanded_struct_fields={},
                dependencies={"temporal_sha256": "sha256:temporal"},
            )
    finally:
        connection.close()
