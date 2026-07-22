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
