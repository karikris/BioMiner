from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.funnel import FunnelConfig, build_source_funnel


def test_source_funnel_reconciles_every_transition(tmp_path: Path) -> None:
    config = _fixture(tmp_path)

    funnel = build_source_funnel(config)

    assert all(funnel.validation.values())
    assert funnel.counts == {
        "raw_occurrence_rows": 10,
        "raw_multimedia_rows": 7,
        "occurrences_with_media": 4,
        "occurrences_without_media": 6,
        "unresolved_multimedia_rows": 0,
        "pre_1960_media_rows_excluded": 1,
        "legacy_cohort_rows_excluded": 2,
        "explicit_rights_rows_excluded": 1,
        "v3_media_rows": 3,
        "v3_occurrences": 2,
        "unexplained_media_residual_rows": 0,
    }
    stages = {row["stage_id"]: row for row in funnel.stages}
    assert stages["YEAR_COHORT"]["excluded_row_count"] == 1
    assert stages["LEGACY_COLUMN_PROJECTION"]["excluded_row_count"] == 0
    assert stages["EXPLICIT_MEDIA_RIGHTS_FILTER"]["exclusion_reason"] == (
        "EXPLICIT_ALL_RIGHTS_RESERVED_OR_COPYRIGHT"
    )
    assert funnel.stage_table().num_rows == 8


def test_source_funnel_fails_on_unexplained_residual(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    value = json.loads(config.v3_filter_manifest.read_text(encoding="utf-8"))
    value["counts"]["removed_rows"] = 999
    config.v3_filter_manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="source funnel validation failed"):
        build_source_funnel(config)


def _fixture(root: Path) -> FunnelConfig:
    paths = {
        name: root / f"{name}.json"
        for name in (
            "dwca",
            "join",
            "year",
            "completeness",
            "grouped",
            "normalized",
            "cohort",
            "v3",
        )
    }
    paths["dwca"].write_text(
        json.dumps(
            {
                "outputs": [
                    {"member": "occurrence.txt", "row_count": 10},
                    {"member": "multimedia.txt", "row_count": 7},
                    {"member": "verbatim.txt", "row_count": 10},
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["join"].write_text(
        json.dumps({"counts": {"joined_rows": 7}}), encoding="utf-8"
    )
    paths["year"].write_text(
        json.dumps(
            {
                "counts": {
                    "output_rows": 6,
                    "output_distinct_gbif_ids": 3,
                    "removed_pre_cutoff_rows": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    paths["completeness"].write_text(
        json.dumps({"output": {"row_count": 6}}), encoding="utf-8"
    )
    paths["grouped"].write_text(
        json.dumps({"counts": {"total_rows": 6}}), encoding="utf-8"
    )
    paths["normalized"].write_text(
        json.dumps({"counts": {"total_rows": 6}}), encoding="utf-8"
    )
    paths["cohort"].write_text(
        json.dumps(
            {
                "counts": {
                    "retained_rows": 4,
                    "retained_distinct_occurrences": 2,
                    "removed_rows": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    paths["v3"].write_text(
        json.dumps(
            {
                "counts": {
                    "retained_rows": 3,
                    "retained_distinct_occurrences": 2,
                    "restricted_image_rows_excluded": 1,
                    "removed_rows": 3,
                },
                "output": {"row_count": 3},
            }
        ),
        encoding="utf-8",
    )
    coverage = root / "coverage.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "coverage_dimension": "occurrence_media_cardinality",
                    "coverage_category": "zero_media",
                    "occurrence_record_count": 6,
                    "multimedia_row_count": 0,
                },
                {
                    "coverage_dimension": "occurrence_media_cardinality",
                    "coverage_category": "one_media",
                    "occurrence_record_count": 2,
                    "multimedia_row_count": 2,
                },
                {
                    "coverage_dimension": "occurrence_media_cardinality",
                    "coverage_category": "multiple_media",
                    "occurrence_record_count": 2,
                    "multimedia_row_count": 5,
                },
                {
                    "coverage_dimension": "multimedia_foreign_key",
                    "coverage_category": "resolved_occurrence",
                    "occurrence_record_count": None,
                    "multimedia_row_count": 7,
                },
                {
                    "coverage_dimension": "multimedia_foreign_key",
                    "coverage_category": "unresolved_occurrence",
                    "occurrence_record_count": None,
                    "multimedia_row_count": 0,
                },
            ]
        ),
        coverage,
    )
    return FunnelConfig(
        repository_root=root,
        dwca_manifest=paths["dwca"],
        join_manifest=paths["join"],
        join_coverage=coverage,
        year_manifest=paths["year"],
        completeness_manifest=paths["completeness"],
        verification_group_manifest=paths["grouped"],
        verification_normalized_manifest=paths["normalized"],
        cohort_manifest=paths["cohort"],
        v3_filter_manifest=paths["v3"],
    )
