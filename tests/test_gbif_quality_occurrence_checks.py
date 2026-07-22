from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.occurrence_checks import publish_occurrence_quality


def test_occurrence_checks_are_one_row_per_gbif_and_keep_context(tmp_path: Path) -> None:
    rows = [
        _row("1", "occ:1", "SPECIES", "Papilio alpha", "10", "20"),
        _row("1", "occ:1", "SPECIES", "Papilio alpha", "10", "20"),
        {
            **_row("2", "occ:shared", "GENUS", None, None, None),
            "informationWithheld": "coordinates withheld",
            "identifiedBy": None,
            "eventDate": "2020-02",
            "year": "2020",
            "month": "2",
            "day": None,
        },
        {
            **_row("3", "occ:shared", "SPECIES", None, "0", "0"),
            "occurrenceStatus": "ABSENT",
            "individualCount": "2",
            "eventDate": "2021-02-30",
        },
        {
            **_row("4", "occ:4", "GENUS", None, "12", "21"),
            "datasetKey": "not-a-uuid",
        },
    ]
    table = pa.Table.from_pylist(rows)
    source = tmp_path / "v3.parquet"
    pq.write_table(table, source)

    result = publish_occurrence_quality(
        v3_parquet=source,
        output_directory=tmp_path / "quality",
        source_snapshot_id="sha256:test",
        expected_media_rows=5,
        expected_occurrences=4,
        code_commit="abc",
        memory_limit="1GB",
        threads=1,
    )

    assert all(result.manifest["validation"].values())
    output = {row["gbifID"]: row for row in pq.read_table(result.quality_path).to_pylist()}
    assert output["1"]["media_assertion_count"] == 2
    assert output["1"]["coordinate_pair_status"] == "PASS"
    assert output["1"]["accepted_taxon_key_status"] == "PASS"
    assert output["2"]["coordinate_pair_status"] == "WITHHELD"
    assert output["2"]["rank_name_consistency_status"] == "PASS"
    assert output["4"]["dataset_key_status"] == "FAIL"
    assert output["3"]["rank_name_consistency_status"] == "FAIL"
    assert output["3"]["zero_coordinate_status"] == "FAIL"
    assert output["3"]["event_date_status"] == "FAIL"
    assert output["3"]["occurrence_count_consistency_status"] == "CONFLICT"
    assert output["2"]["occurrence_identity_conflict_status"] == "CONFLICT"
    assert output["3"]["occurrence_identity_conflict_status"] == "CONFLICT"
    issue = pq.read_table(result.output_directory / "gbif_issue_summary.parquet")
    assert set(issue["gbif_issue_flag"].to_pylist()) == {
        "COORDINATE_ROUNDED",
        "TAXON_ID_NOT_FOUND",
    }


def _row(
    gbif_id: str,
    occurrence_id: str,
    rank: str,
    species: str | None,
    latitude: str | None,
    longitude: str | None,
) -> dict[str, str | None]:
    return {
        "gbifID": gbif_id,
        "issue": "COORDINATE_ROUNDED;TAXON_ID_NOT_FOUND",
        "datasetKey": "123e4567-e89b-12d3-a456-426614174000",
        "occurrenceID": occurrence_id,
        "basisOfRecord": "HUMAN_OBSERVATION",
        "occurrenceStatus": "PRESENT",
        "sex": "Female",
        "eventDate": "2020-02-03",
        "year": "2020",
        "month": "2",
        "day": "3",
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
        "coordinateUncertaintyInMeters": "10" if latitude else None,
        "informationWithheld": None,
        "dataGeneralizations": None,
        "taxonRank": rank,
        "species": species,
        "specificEpithet": "alpha" if species else None,
        "genus": "Papilio",
        "taxonKey": "1",
        "acceptedTaxonKey": "1",
        "taxonomicStatus": "ACCEPTED",
        "identifiedBy": "Expert",
        "identificationVerificationStatus": "accepted",
        "individualCount": "1",
    }
