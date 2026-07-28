import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.geography import publish_geographic_enrichment


def test_geography_derives_only_safe_country_mappings(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    rows = [
        _row("1", "AU", None, None, "-33", "151"),
        _row("2", "AU", "OCEANIA", "OCEANIA", "-34", "150"),
        _row("3", None, None, None, "10", "20"),
        _row("4", "US", "NORTH_AMERICA", "NORTH_AMERICA", "30", "-90"),
        _row("5", "US", "OCEANIA", "NORTH_AMERICA", "20", "-150"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    result = publish_geographic_enrichment(
        v3_parquet=source,
        output_directory=tmp_path / "out",
        source_snapshot_id="sha256:test",
        expected_coordinate_country_media_rows=1,
        expected_coordinate_country_occurrences=1,
        expected_missing_continent_media_rows=1,
        expected_missing_continent_occurrences=1,
        expected_missing_region_media_rows=1,
        expected_missing_region_occurrences=1,
        code_commit="deadbeef",
        minimum_mapping_confidence=0.6,
    )
    outcomes = {row["gbifID"]: row for row in pq.read_table(result.outcome_path).to_pylist()}
    assert outcomes["1"]["derived_continent"] == "OCEANIA"
    assert outcomes["1"]["derived_gbifRegion"] == "OCEANIA"
    assert outcomes["3"]["country_derivation_status"] == "NOT_TESTED"
    assert outcomes["3"]["derived_countryCode"] is None
    assertions = pq.read_table(result.assertion_path)
    assert assertions.num_rows == 2


def test_geography_uses_pinned_boundary_and_retains_ambiguous_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row("1", "AU", "OCEANIA", "OCEANIA", "-33", "151"),
                _row(
                    "2",
                    "US",
                    "NORTH_AMERICA",
                    "NORTH_AMERICA",
                    "30",
                    "-90",
                ),
                _row("3", None, None, None, "1", "1"),
                _row("4", None, None, None, "1", "2"),
                _row("5", None, None, None, "10", "10"),
            ]
        ),
        source,
    )
    boundary = tmp_path / "countries.geojson"
    boundary.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "ISO_A2": "AU",
                            "ISO_A2_EH": "AU",
                            "ADMIN": "Fixture Australia",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "ISO_A2": "US",
                            "ISO_A2_EH": "US",
                            "ADMIN": "Fixture United States",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[2, 0], [4, 0], [4, 2], [2, 2], [2, 0]]
                            ],
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "boundary-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "boundary_dataset": "Fixture countries",
                "boundary_version": "fixture-v1",
                "files": {
                    boundary.name: hashlib.sha256(boundary.read_bytes()).hexdigest()
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = publish_geographic_enrichment(
        v3_parquet=source,
        output_directory=tmp_path / "out-boundary",
        source_snapshot_id="sha256:test",
        expected_coordinate_country_media_rows=3,
        expected_coordinate_country_occurrences=3,
        expected_missing_continent_media_rows=0,
        expected_missing_continent_occurrences=0,
        expected_missing_region_media_rows=0,
        expected_missing_region_occurrences=0,
        code_commit="deadbeef",
        boundary_manifest=manifest,
        minimum_mapping_confidence=0.99,
    )

    outcomes = {
        row["gbifID"]: row
        for row in pq.read_table(result.outcome_path).to_pylist()
    }
    assert outcomes["3"]["derived_countryCode"] == "AU"
    assert outcomes["3"]["derived_country"] == "Fixture Australia"
    assert outcomes["3"]["derived_continent"] == "OCEANIA"
    assert outcomes["3"]["border_ambiguity_status"] == "PASS"
    assert outcomes["4"]["derived_countryCode"] is None
    assert outcomes["4"]["border_ambiguity_status"] == "AMBIGUOUS"
    assert outcomes["4"]["boundary_candidate_countryCodes"] == ["AU", "US"]
    assert outcomes["5"]["derived_countryCode"] is None
    assert outcomes["5"]["border_ambiguity_status"] == "OUTSIDE_OR_UNMAPPED"
    assert result.manifest["counts"]["derived_country_occurrences"] == 1
    assert result.manifest["counts"]["ambiguous_border_occurrences"] == 1
    assert result.manifest["counts"]["outside_or_unmapped_occurrences"] == 1
    assert pq.read_table(result.assertion_path).num_rows == 4


def _row(
    gbif_id: str,
    country_code: str | None,
    continent: str | None,
    region: str | None,
    latitude: str | None,
    longitude: str | None,
) -> dict[str, str | None]:
    return {
        "gbifID": gbif_id,
        "countryCode": country_code,
        "continent": continent,
        "gbifRegion": region,
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
    }
