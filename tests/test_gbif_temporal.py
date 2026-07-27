from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_temporal.parser import (
    PARSER_VERSION,
    derive_temporal_components,
)
from biominer.gbif_temporal.pipeline import publish_temporal_enrichment
from biominer.cli import build_parser


def test_interval_start_derives_only_missing_components() -> None:
    result = derive_temporal_components(
        event_date="2010-03-01/2010-03-31",
        year="2010",
        month="3",
        day=None,
    )

    assert result.status == "derived"
    assert result.derived_year is None
    assert result.derived_month is None
    assert result.derived_day == 1
    assert result.derived_components == "day"
    assert result.method == "event_date_interval_start_date"
    assert result.interval_start == "2010-03-01"
    assert result.interval_end == "2010-03-31"


def test_timestamp_interval_preserves_lexical_calendar_date() -> None:
    result = derive_temporal_components(
        event_date="2021-08-06T23:30-10:00/2021-08-07T13:00-10:00",
        year=None,
        month=None,
        day=None,
    )

    assert (
        result.derived_year,
        result.derived_month,
        result.derived_day,
    ) == (2021, 8, 6)
    assert result.method == "event_date_interval_start_timestamp"


@pytest.mark.parametrize(
    ("event_date", "status"),
    [
        ("2023-02-29/2023-03-01", "invalid_calendar_date"),
        ("2024-02-29/2024-03-01", "derived"),
        ("2024-04-31/2024-05-01", "invalid_calendar_date"),
        ("2024-03-02/2024-03-01", "reversed_interval"),
        ("2024-03-01T12:00/2024-03-01T11:00", "reversed_interval"),
        ("03/04/2024", "unsupported_format"),
        ("2024-03", "insufficient_precision"),
        ("2024", "insufficient_precision"),
        ("2024-13", "invalid_calendar_date"),
        ("0000", "invalid_calendar_date"),
    ],
)
def test_calendar_and_format_validation(event_date: str, status: str) -> None:
    assert (
        derive_temporal_components(
            event_date=event_date,
            year=None,
            month=None,
            day=None,
        ).status
        == status
    )


def test_existing_component_conflict_fails_closed() -> None:
    result = derive_temporal_components(
        event_date="2024-05-06/2024-05-07",
        year="2023",
        month=None,
        day=None,
    )

    assert result.status == "existing_component_conflict"
    assert result.derived_year is None
    assert result.derived_month is None
    assert result.derived_day is None


def test_single_date_is_supported_without_inventing_partial_precision() -> None:
    exact = derive_temporal_components(
        event_date="2024-02-29",
        year=None,
        month=None,
        day=None,
    )
    partial = derive_temporal_components(
        event_date="2024-02",
        year="2024",
        month="2",
        day=None,
    )

    assert exact.method == "event_date_single_date"
    assert (exact.derived_year, exact.derived_month, exact.derived_day) == (
        2024,
        2,
        29,
    )
    assert partial.status == "insufficient_precision"
    assert partial.derived_day is None


def test_temporal_command_is_registered_with_source_bound_defaults() -> None:
    args = build_parser().parse_args(
        [
            "gbif-temporal-enrich",
            "--source",
            "source.parquet",
            "--source-manifest",
            "source.json",
            "--output-directory",
            "output",
        ]
    )

    assert args.command == "gbif-temporal-enrich"
    assert args.expected_source_rows == 16_612_063
    assert args.expected_derived_day_rows == 18_741
    assert args.expected_pre_1960_excluded_rows == 2_236


def test_publish_temporal_enrichment_is_create_only_and_auditable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source-manifest.json"
    output = tmp_path / "temporal-v1"
    table = pa.table(
        {
            "gbifID": ["1", "1", "2", "3", "4", "5", "6"],
            "eventDate": [
                "2010-03-01/2010-03-31",
                "2010-03-01/2010-03-31",
                "1959-12-31/1960-01-02",
                "2023-02-29/2023-03-01",
                "2024-02",
                "2024-05-06/2024-05-07",
                "2024-06-01/2024-06-02",
            ],
            "year": ["2010", "2010", None, None, "2024", "2023", "2024"],
            "month": ["3", "3", None, None, "2", None, "6"],
            "day": [None, None, None, None, None, None, "1"],
            "media_identifier": [f"https://example.test/{i}.jpg" for i in range(7)],
            "payload": ["a", "a", "b", "c", "d", "e", "f"],
        }
    )
    pq.write_table(table, source, row_group_size=2)
    source_manifest.write_text('{"fixture": true}\n', encoding="utf-8")

    manifest = publish_temporal_enrichment(
        source=source,
        source_manifest=source_manifest,
        output_directory=output,
        expected_source_sha256=None,
        expected_source_rows=7,
        expected_derived_year_rows=1,
        expected_derived_month_rows=1,
        expected_derived_day_rows=3,
        expected_pre_1960_excluded_rows=1,
        batch_rows=2,
        duckdb_memory_limit="1GB",
        duckdb_threads=1,
    )

    assert manifest["counts"]["source_rows"] == 7
    assert manifest["counts"]["output_rows"] == 6
    assert manifest["counts"]["excluded_pre_1960_rows"] == 1
    assert manifest["counts"]["derived_year_rows"] == 1
    assert manifest["counts"]["derived_month_rows"] == 1
    assert manifest["counts"]["derived_day_rows"] == 3
    assert manifest["parser_version"] == PARSER_VERSION
    assert manifest["manifest_policy"]["written_last"] is True

    enriched = pq.read_table(output / "gbif_media_temporal.parquet")
    assert enriched.schema.names[-4:] == [
        "derived_year",
        "derived_month",
        "derived_day",
        "temporal_derivation_method",
    ]
    assert enriched.column("eventDate").to_pylist() == [
        "2010-03-01/2010-03-31",
        "2010-03-01/2010-03-31",
        "2023-02-29/2023-03-01",
        "2024-02",
        "2024-05-06/2024-05-07",
        "2024-06-01/2024-06-02",
    ]
    assert enriched.column("derived_day").to_pylist() == [1, 1, None, None, None, None]

    audit = pq.read_table(output / "temporal_derivations.parquet")
    assert audit.num_rows == 5
    by_id = {row["gbifID"]: row for row in audit.to_pylist()}
    assert by_id["2"]["temporal_derivation_status"] == "excluded_pre_1960"
    assert by_id["3"]["temporal_derivation_status"] == "invalid_calendar_date"
    assert by_id["4"]["temporal_derivation_status"] == "insufficient_precision"
    assert by_id["5"]["temporal_derivation_status"] == "existing_component_conflict"

    with duckdb.connect(str(output / "gbif_media_temporal.duckdb"), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM gbif_media").fetchone()[0] == 6
        assert con.execute("SELECT count(*) FROM temporal_derivations").fetchone()[0] == 5
        indexes = {row[0] for row in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_gbif_media_gbif_id" in indexes
    assert "idx_temporal_derivations_gbif_id" in indexes

    on_disk_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest == manifest
    with pytest.raises(FileExistsError):
        publish_temporal_enrichment(
            source=source,
            source_manifest=source_manifest,
            output_directory=output,
            expected_source_sha256=None,
        )


def test_publication_rejects_inconsistent_occurrence_temporal_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    manifest = tmp_path / "source.json"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "1"],
                "eventDate": ["2024-01-01/2024-01-02", "2024-02-01/2024-02-02"],
                "year": ["2024", "2024"],
                "month": pa.array([None, None], type=pa.string()),
                "day": pa.array([None, None], type=pa.string()),
            }
        ),
        source,
    )
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent temporal values"):
        publish_temporal_enrichment(
            source=source,
            source_manifest=manifest,
            output_directory=tmp_path / "out",
            expected_source_sha256=None,
            expected_source_rows=None,
            expected_derived_year_rows=None,
            expected_derived_month_rows=None,
            expected_derived_day_rows=None,
            expected_pre_1960_excluded_rows=None,
        )
