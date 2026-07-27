from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.policy import build_field_policy
from biominer.gbif_quality.profile import profile_completeness


def test_profile_reports_physical_and_applicable_denominators(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "gbifID": "1",
                "taxonRank": "SPECIES",
                "species": "Papilio x",
                "eventDate": "2020-01",
                "day": None,
                "informationWithheld": None,
                "dataGeneralizations": None,
                "media_identifier": "https://example/1.jpg",
                "media_format": "image/jpeg",
            },
            {
                "gbifID": "1",
                "taxonRank": "SPECIES",
                "species": "Papilio x",
                "eventDate": "2020-01",
                "day": None,
                "informationWithheld": None,
                "dataGeneralizations": None,
                "media_identifier": "https://example/2.jpg",
                "media_format": "unknown",
            },
            {
                "gbifID": "2",
                "taxonRank": "GENUS",
                "species": None,
                "eventDate": "2019",
                "day": None,
                "informationWithheld": "locality withheld",
                "dataGeneralizations": None,
                "media_identifier": None,
                "media_format": None,
            },
            {
                "gbifID": "3",
                "taxonRank": "SUBSPECIES",
                "species": None,
                "eventDate": "2018-03-04",
                "day": "4",
                "informationWithheld": None,
                "dataGeneralizations": "coordinates generalized",
                "media_identifier": "https://example/3.jpg",
                "media_format": None,
            },
        ]
    )
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)

    result = profile_completeness(
        path, build_field_policy(table.schema), occurrence_batch_size=2
    )

    assert all(result.validation.values())
    assert result.denominators == {
        "media_rows": 4,
        "distinct_occurrences": 3,
        "columns": 9,
    }
    rows = {row["field_name"]: row for row in result.rows}
    assert rows["species"]["applicable_media_rows"] == 3
    assert rows["species"]["not_applicable_media_rows"] == 1
    assert rows["species"]["applicable_filled_occurrences"] == 1
    assert rows["day"]["applicable_media_rows"] == 1
    assert rows["day"]["applicable_filled_media_rows"] == 1
    assert rows["media_format"]["applicable_media_rows"] == 3
    assert rows["media_format"]["semantic_sentinel_media_rows"] == 1
    assert rows["media_format"]["applicable_filled_media_rows"] == 1
    assert result.table().num_rows == 9


def test_profile_rejects_policy_schema_drift(tmp_path: Path) -> None:
    table = pa.table({"gbifID": ["1"], "value": ["x"]})
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)
    policies = build_field_policy(pa.schema([("gbifID", pa.string())]))

    try:
        profile_completeness(path, policies)
    except ValueError as exc:
        assert "exactly match" in str(exc)
    else:
        raise AssertionError("schema drift must fail closed")
