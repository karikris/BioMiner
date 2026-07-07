from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.registry.language_targets import generate_language_targets, write_language_targets


def test_country_with_occurrence_evidence_produces_expected_language_targets(tmp_path) -> None:
    range_countries = _range_frame(
        [
            {
                "country_code": "IN",
                "country_name": "India",
                "range_status": "native_or_long_established",
                "taxonomic_caution": False,
                "region": "South Asia",
            }
        ]
    )

    targets = generate_language_targets(
        range_countries,
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    rows = targets.sort(["priority", "language_code"]).to_dicts()
    assert ("IN", "", "eng", "Latn", True) in {
        (row["country_code"], row["admin1_code"], row["language_code"], row["script"], row["enabled"]) for row in rows
    }
    assert ("IN", "", "hin", "Deva", True) in {
        (row["country_code"], row["admin1_code"], row["language_code"], row["script"], row["enabled"]) for row in rows
    }
    assert ("IN", "", "kan", "Knda", True) in {
        (row["country_code"], row["admin1_code"], row["language_code"], row["script"], row["enabled"]) for row in rows
    }
    assert all(row["reason"] in {"country_language_override", "species_region_language_target"} for row in rows)


def test_india_admin_region_occurrence_preserves_state_target_capacity(tmp_path) -> None:
    range_countries = _range_frame(
        [
            {
                "country_code": "IN",
                "country_name": "India",
                "admin1_code": "IN-KA",
                "admin1_name": "Karnataka",
                "range_status": "native_or_long_established",
                "taxonomic_caution": False,
                "region": "South Asia",
            }
        ]
    )

    targets = generate_language_targets(
        range_countries,
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    karnataka = targets.filter((pl.col("admin1_code") == "IN-KA") & (pl.col("language_code") == "kan"))
    assert karnataka.height == 1
    row = karnataka.to_dicts()[0]
    assert row["admin1_name"] == "Karnataka"
    assert row["region"] == "South Asia"
    assert row["enabled"] is True


def test_duplicate_language_targets_collapse_cleanly(tmp_path) -> None:
    range_countries = _range_frame(
        [
            {
                "country_code": "SG",
                "country_name": "Singapore",
                "range_status": "native_or_long_established",
                "taxonomic_caution": False,
                "region": "Maritime Southeast Asia",
            }
        ]
    )
    country_config = tmp_path / "country_languages.json"
    country_config.write_text(
        json.dumps(
            {
                "schema_version": "country-language-overrides-v1",
                "countries": [{"country_code": "SG", "languages": [{"language": "en"}, {"language": "eng"}]}],
            }
        ),
        encoding="utf-8",
    )
    region_config = tmp_path / "region_languages.json"
    region_config.write_text(
        json.dumps(
            {
                "schema_version": "species-region-language-targets-v1",
                "accepted_taxon_key": "gbif:1938069",
                "scientific_name": "Papilio demoleus",
                "regions": [{"region": "Maritime Southeast Asia", "languages": [{"language": "eng"}]}],
            }
        ),
        encoding="utf-8",
    )

    targets = generate_language_targets(
        range_countries,
        country_language_overrides_json=country_config,
        species_region_language_targets_json=region_config,
    )

    assert targets.filter(pl.col("language_code") == "eng").height == 1
    assert targets.height == 1


def test_taxonomic_caution_regions_create_disabled_language_targets() -> None:
    range_countries = _range_frame(
        [
            {
                "country_code": "AU",
                "country_name": "Australia",
                "range_status": "taxonomically_cautionary",
                "taxonomic_caution": True,
                "region": "Australia/New Guinea taxonomic caution",
            }
        ]
    )

    targets = generate_language_targets(
        range_countries,
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    assert targets.height > 0
    assert targets.select("enabled").to_series().to_list() == [False]
    assert targets.select("disabled_reason").to_series().to_list() == ["taxonomic_caution_range"]


def test_unknown_country_without_language_config_does_not_fail() -> None:
    range_countries = _range_frame(
        [
            {
                "country_code": "ZZ",
                "country_name": "Unknown",
                "range_status": "occurrence_supported",
                "taxonomic_caution": False,
                "region": "",
            }
        ]
    )

    targets = generate_language_targets(
        range_countries,
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    assert targets.is_empty()
    assert targets.schema == _target_schema()


def test_language_targets_output_schema_and_parquet_writer(tmp_path) -> None:
    targets = generate_language_targets(
        _range_frame(
            [
                {
                    "country_code": "IN",
                    "country_name": "India",
                    "range_status": "native_or_long_established",
                    "taxonomic_caution": False,
                    "region": "South Asia",
                }
            ]
        ),
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    assert targets.schema == _target_schema()
    path = write_language_targets(targets, tmp_path / "registry")
    assert path == tmp_path / "registry" / "country_language_targets.parquet"
    assert pl.read_parquet(path).height == targets.height


def test_papilio_region_language_config_resolves_language_and_script_metadata() -> None:
    range_rows = []
    for country_code, country_name, region, range_status, taxonomic_caution in [
        ("IN", "India", "South Asia", "native_or_long_established", False),
        ("SA", "Saudi Arabia", "Middle East / West Asia", "occurrence_supported", False),
        ("TW", "Taiwan", "East Asia", "occurrence_supported", False),
        ("MM", "Myanmar", "Mainland Southeast Asia", "native_or_long_established", False),
        ("ID", "Indonesia", "Maritime Southeast Asia", "native_or_long_established", False),
        ("DO", "Dominican Republic", "Caribbean introduced range", "introduced_established", False),
        ("SC", "Seychelles", "Seychelles / Indian Ocean spread watch", "occurrence_supported", False),
        ("AU", "Australia", "Australia/New Guinea taxonomic caution", "taxonomically_cautionary", True),
    ]:
        range_rows.append(
            {
                "country_code": country_code,
                "country_name": country_name,
                "range_status": range_status,
                "taxonomic_caution": taxonomic_caution,
                "region": region,
            }
        )

    targets = generate_language_targets(
        _range_frame(range_rows),
        country_language_overrides_json=Path("config/language_targets/country_language_overrides.json"),
        species_region_language_targets_json=Path("config/language_targets/papilio_demoleus_region_language_targets.json"),
    )

    assert targets.filter((pl.col("language_code") == "") | (pl.col("script") == "")).is_empty()


def _range_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "accepted_taxon_key": row.get("accepted_taxon_key", "gbif:1938069"),
                "scientific_name": row.get("scientific_name", "Papilio demoleus"),
                "source": "GBIF",
                "source_taxon_key": row.get("accepted_taxon_key", "gbif:1938069"),
                "country_code": row["country_code"],
                "country_name": row["country_name"],
                "admin1_code": row.get("admin1_code", ""),
                "admin1_name": row.get("admin1_name", ""),
                "occurrence_count": row.get("occurrence_count", 10),
                "georeferenced_count": row.get("georeferenced_count", 8),
                "basis_of_record_counts_json": "{}",
                "first_year": row.get("first_year", 2010),
                "last_year": row.get("last_year", 2026),
                "has_recent_records": row.get("has_recent_records", True),
                "range_status": row["range_status"],
                "confidence": row.get("confidence", "high"),
                "taxonomic_caution": row["taxonomic_caution"],
                "retrieved_at": "2026-07-07T00:00:00+00:00",
                "source_query_hash": "sha256:test",
                "region": row.get("region", ""),
            }
        )
    return pl.DataFrame(normalized)


def _target_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "country_code": pl.String,
        "country_name": pl.String,
        "admin1_code": pl.String,
        "admin1_name": pl.String,
        "language_code": pl.String,
        "language_name": pl.String,
        "script": pl.String,
        "region": pl.String,
        "priority": pl.Int64,
        "reason": pl.String,
        "source": pl.String,
        "source_version": pl.String,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }
