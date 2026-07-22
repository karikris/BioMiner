from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.assertions import DERIVED_ASSERTION_SCHEMA
from biominer.gbif_quality.temporal import (
    TEMPORAL_QUALITY_SCHEMA,
    parse_event_date,
    publish_temporal_quality_v2,
)


@pytest.mark.parametrize(
    ("raw", "status", "precision", "start", "end"),
    [
        (None, "UNKNOWN", None, None, None),
        ("2020", "PASS", "YEAR", "2020-01-01", "2020-12-31"),
        ("2020-02", "PASS", "MONTH", "2020-02-01", "2020-02-29"),
        ("2020-02-29", "PASS", "DAY", "2020-02-29", "2020-02-29"),
        (
            "2020-02-29T23:59:01Z",
            "PASS",
            "DATETIME",
            "2020-02-29",
            "2020-02-29",
        ),
        (
            "2020-02-29/2020-03-02",
            "PASS",
            "INTERVAL",
            "2020-02-29",
            "2020-03-02",
        ),
        ("2021-02-29", "FAIL", None, None, None),
        (
            "2020-03-02/2020-02-29",
            "FAIL",
            "INTERVAL",
            "2020-03-02",
            "2020-02-29",
        ),
    ],
)
def test_parse_event_date_is_strict_and_precision_aware(
    raw: str | None,
    status: str,
    precision: str | None,
    start: str | None,
    end: str | None,
) -> None:
    parsed = parse_event_date(raw)

    assert parsed.status == status
    assert parsed.precision == precision
    assert (parsed.start.isoformat() if parsed.start else None) == start
    assert (parsed.end.isoformat() if parsed.end else None) == end


def test_temporal_publication_retains_flags_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    rows = [
        _row("1", "2020-02-29", None, None, None),
        _row("1", "2020-02-29", None, None, None),
        _row("2", "1959-12-31/1960-01-02", None, "12", "31"),
        _row("3", "2024-05", "2024", None, None),
        _row("4", "2030-01-01", None, None, None),
        _row("5", "2021-02-29", None, None, None),
        _row("6", "2021-06-02/2021-06-03", "2021", "6", None),
        _row("7", "2020-01-01", None, "1", "1"),
        _row("7", "2020-01-01", "2020", "1", "1"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)

    first = _publish(source, tmp_path / "first")
    second = _publish(source, tmp_path / "second")

    quality = pq.read_table(first.quality_path)
    assertions = pq.read_table(first.assertion_path)
    assert quality.schema == TEMPORAL_QUALITY_SCHEMA
    assert assertions.schema == DERIVED_ASSERTION_SCHEMA
    assert quality.num_rows == 7
    assert assertions.num_rows == 9
    by_id = {row["gbifID"]: row for row in quality.to_pylist()}
    assert by_id["2"]["derived_year"] == 1959
    assert by_id["2"]["ancient_record_status"] == "FLAGGED"
    assert by_id["4"]["future_record_status"] == "FLAGGED"
    assert by_id["5"]["temporal_parse_status"] == "FAIL"
    assert by_id["7"]["temporal_conflict_status"] == "CONFLICT"
    assert first.manifest["counts"]["media_rows"] == 9
    assert first.manifest["counts"]["derived_year_media_rows"] == 4
    assert first.manifest["counts"]["derived_month_media_rows"] == 4
    assert first.manifest["counts"]["derived_day_media_rows"] == 4
    assert json.loads((first.output_directory / "manifest.json").read_text())[
        "manifest_policy"
    ]["written_last"] is True
    first_ids = assertions.column("assertion_id").to_pylist()
    second_ids = pq.read_table(second.assertion_path).column("assertion_id").to_pylist()
    assert first_ids == second_ids
    with pytest.raises(FileExistsError):
        _publish(source, first.output_directory)


def _row(
    gbif_id: str,
    event_date: str | None,
    year: str | None,
    month: str | None,
    day: str | None,
) -> dict[str, str | None]:
    return {
        "gbifID": gbif_id,
        "eventDate": event_date,
        "year": year,
        "month": month,
        "day": day,
    }


def _publish(source: Path, output: Path):
    return publish_temporal_quality_v2(
        v3_parquet=source,
        output_directory=output,
        source_snapshot_id="sha256:test-snapshot",
        source_publication_date="2026-07-19",
        expected_media_rows=9,
        expected_occurrences=7,
        expected_derived_year_media_rows=4,
        expected_derived_month_media_rows=4,
        expected_derived_day_media_rows=4,
        code_commit="deadbeef",
        expected_ancient_media_rows=1,
        expected_ancient_occurrences=1,
        batch_rows=2,
    )
