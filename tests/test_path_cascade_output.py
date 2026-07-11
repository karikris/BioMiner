from __future__ import annotations

import polars as pl
import pyarrow.parquet as pq
import pytest

from biominer.bioclip.path_cascade_output import (
    PATH_CASCADE_OUTPUT_SCHEMA,
    PATH_CASCADE_OUTPUT_SCHEMA_VERSION,
    PATH_CASCADE_PRUNING_TRACE_VERSION,
    empty_path_cascade_output_frame,
    path_cascade_output_frame,
    validate_path_cascade_output_frame,
    write_path_cascade_output,
)
from biominer.registry.classification_v3 import CLASSIFICATION_RANKS


def test_cascade_output_schema_is_six_rank_and_has_no_legacy_identity_fields() -> None:
    assert "subtribe_top3" in PATH_CASCADE_OUTPUT_SCHEMA
    assert "genus_top3" in PATH_CASCADE_OUTPUT_SCHEMA
    assert "genus_top8" not in PATH_CASCADE_OUTPUT_SCHEMA
    assert "family_top3_node_ids" in PATH_CASCADE_OUTPUT_SCHEMA
    assert "family_top3_accepted_taxon_keys" not in PATH_CASCADE_OUTPUT_SCHEMA
    assert PATH_CASCADE_OUTPUT_SCHEMA["species_top20_first_pass_scores"] == pl.List(
        pl.Float32
    )
    assert PATH_CASCADE_OUTPUT_SCHEMA["species_top5_rerank_scores"] == pl.List(pl.Float32)
    assert PATH_CASCADE_OUTPUT_SCHEMA["candidate_counts_by_rank"] == pl.Struct(
        {rank: pl.UInt32 for rank in CLASSIFICATION_RANKS}
    )


def test_empty_cascade_output_preserves_exact_physical_schema(tmp_path) -> None:
    frame = empty_path_cascade_output_frame()

    assert frame.shape == (0, len(PATH_CASCADE_OUTPUT_SCHEMA))
    assert frame.columns == list(PATH_CASCADE_OUTPUT_SCHEMA)
    assert dict(frame.schema) == PATH_CASCADE_OUTPUT_SCHEMA
    assert validate_path_cascade_output_frame(frame).schema == frame.schema
    path = write_path_cascade_output(frame, tmp_path / "empty-cascade.parquet")
    restored = pl.read_parquet(path)
    assert restored.is_empty()
    assert dict(restored.schema) == PATH_CASCADE_OUTPUT_SCHEMA


def test_minimal_row_uses_typed_empty_lists_zero_count_structs_and_null_scalars() -> None:
    frame = path_cascade_output_frame([{}])
    row = frame.row(0, named=True)

    assert row["classifier_schema_version"] == PATH_CASCADE_OUTPUT_SCHEMA_VERSION
    assert row["pruning_trace_version"] == PATH_CASCADE_PRUNING_TRACE_VERSION
    assert row["family_top3"] == []
    assert row["subtribe_top3_node_ids"] == []
    assert row["species_top20"] == []
    assert row["family_top1"] is None
    assert row["family_top1_score"] is None
    assert row["candidate_counts_by_rank"] == {rank: 0 for rank in CLASSIFICATION_RANKS}


def test_zstd_parquet_roundtrip_preserves_nested_schema(tmp_path) -> None:
    counts = {rank: index for index, rank in enumerate(CLASSIFICATION_RANKS, start=1)}
    frame = path_cascade_output_frame(
        [
            {
                "classification_version": "butterfly-classification-v3.0.0",
                "family_top3": ["Papilionidae"],
                "family_top3_node_ids": ["reviewed:family:papilionidae"],
                "family_top3_scores": [0.75],
                "species_top20": ["Papilio demoleus"],
                "species_top20_node_ids": ["reviewed:species:papilio-demoleus"],
                "species_top20_accepted_taxon_keys": ["gbif:1938069"],
                "species_top20_first_pass_scores": [0.5],
                "candidate_counts_by_rank": counts,
            }
        ]
    )
    path = write_path_cascade_output(frame, tmp_path / "cascade.parquet")

    restored = pl.read_parquet(path)
    parquet = pq.ParquetFile(path)
    try:
        compression = parquet.metadata.row_group(0).column(0).compression
    finally:
        parquet.close()

    assert compression == "ZSTD"
    assert dict(restored.schema) == PATH_CASCADE_OUTPUT_SCHEMA
    assert restored.to_dicts() == frame.to_dicts()


def test_output_rejects_version_and_physical_schema_drift() -> None:
    with pytest.raises(ValueError, match="classifier schema version mismatch"):
        path_cascade_output_frame([{"classifier_schema_version": "legacy"}])

    frame = path_cascade_output_frame([{}]).with_columns(
        pl.col("rank_beam_width").cast(pl.UInt16)
    )
    with pytest.raises(ValueError, match="physical schema mismatch"):
        validate_path_cascade_output_frame(frame)
