from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from biominer.candidates.regional_occurrence import (
    REGIONAL_TAXON_OCCURRENCE_FILE,
    REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION,
    build_flickr_cluster_scope_memberships,
    build_regional_taxon_occurrence_index,
    regional_taxon_occurrence_schema,
    write_regional_taxon_occurrence,
)
from biominer.geography import GeographicCoordinate, default_cell_grid
from biominer.registry.geographic_spread import geographic_occurrence_evidence_schema


TARGET_KEY = "gbif:1938069"
CONGENER_KEY = "gbif:100"
OTHER_KEY = "gbif:200"
REGISTRY_VERSION = "butterflies-test-v1"
EVIDENCE_VERSION = "gbif-test-v1+accepted-reconciliation-v1"
OUTPUT_EVIDENCE_VERSION = (
    EVIDENCE_VERSION + "+inverse-uncertainty-100km-v1.0.0"
)


def _taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": TARGET_KEY,
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "in_scope": True,
            },
            {
                "accepted_taxon_key": CONGENER_KEY,
                "scientific_name": "Papilio polytes",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "in_scope": True,
            },
            {
                "accepted_taxon_key": OTHER_KEY,
                "scientific_name": "Danaus plexippus",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "in_scope": True,
            },
        ]
    )


def _classification_paths() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": TARGET_KEY,
                "species": "Papilio demoleus",
                "family": "Papilionidae",
                "subfamily": "Papilioninae",
                "tribe": "Papilionini",
                "genus": "Papilio",
                "enabled": True,
            },
            {
                "accepted_taxon_key": CONGENER_KEY,
                "species": "Papilio polytes",
                "family": "Papilionidae",
                "subfamily": "Papilioninae",
                "tribe": "Papilionini",
                "genus": "Papilio",
                "enabled": True,
            },
        ]
    )


def _cell(latitude: float, longitude: float) -> str:
    return default_cell_grid().coordinate_to_cell(
        GeographicCoordinate(latitude=latitude, longitude=longitude),
        resolution=5,
    )


def _evidence(
    record_id: str,
    taxon_key: str,
    cell_id: str,
    *,
    dataset: str = "dataset-a",
    event_date: date = date(2025, 1, 1),
    uncertainty_m: float | None = 1_000.0,
    country_code: str | None = "AU",
    bioregion: str | None = "australasia",
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "gbif_id": record_id,
        "source": "GBIF",
        "source_dataset_key": dataset,
        "accepted_taxon_key": taxon_key,
        "scientific_name": "untrusted supplied name",
        "spatial_cell_id": cell_id,
        "spatial_resolution": 5,
        "country_code": country_code,
        "bioregion": bioregion,
        "coordinate_uncertainty_m": uncertainty_m,
        "event_date": event_date,
        "occurrence_status": "PRESENT",
        "has_geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "range_inference_eligible": eligible,
        "taxon_key_match": True,
        "coordinate_valid": True,
    }


def _scope(
    scope_id: str,
    overlap_type: str,
    *,
    cell_id: str | None = None,
    country_code: str | None = None,
    bioregion: str | None = None,
) -> dict[str, object]:
    return {
        "regional_scope_id": scope_id,
        "regional_scope_type": "geo_cluster",
        "overlap_type": overlap_type,
        "spatial_cell_id": cell_id,
        "spatial_resolution": 5 if cell_id else None,
        "country_code": country_code,
        "bioregion": bioregion,
    }


def test_builds_reconciled_occurrence_index_with_dataset_and_rank_provenance() -> None:
    brisbane = _cell(-27.4705, 153.026)
    evidence = pl.DataFrame(
        [
            _evidence(
                "1",
                TARGET_KEY,
                brisbane,
                dataset="dataset-a",
                event_date=date(2024, 1, 2),
                uncertainty_m=1_000,
            ),
            _evidence(
                "2",
                TARGET_KEY,
                brisbane,
                dataset="dataset-b",
                event_date=date(2025, 3, 4),
                uncertainty_m=10_000,
            ),
        ]
    )

    result = build_regional_taxon_occurrence_index(
        evidence,
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
        classification_paths=_classification_paths(),
    )

    assert result.schema == regional_taxon_occurrence_schema()
    assert result.height == 1
    row = result.to_dicts()[0]
    assert row["schema_version"] == REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION
    assert row["regional_scope_id"] == brisbane
    assert row["regional_scope_type"] == "spatial_cell"
    assert row["accepted_taxon_key"] == TARGET_KEY
    assert row["scientific_name"] == "Papilio demoleus"
    assert row["family"] == "Papilionidae"
    assert row["subfamily"] == "Papilioninae"
    assert row["tribe"] == "Papilionini"
    assert row["genus"] == "Papilio"
    assert row["occurrence_count"] == 2
    assert row["independent_dataset_count"] == 2
    assert row["earliest_occurrence_date"] == date(2024, 1, 2)
    assert row["latest_occurrence_date"] == date(2025, 3, 4)
    assert row["coordinate_confidence"] == pytest.approx(0.94959497)
    assert row["overlap_type"] == "exact"
    assert row["source"] == "GBIF"
    assert row["source_dataset_keys"] == ["dataset-a", "dataset-b"]
    assert row["evidence_version"] == OUTPUT_EVIDENCE_VERSION
    assert row["registry_version"] == REGISTRY_VERSION


def test_uses_only_strongest_overlap_tier_without_double_counting() -> None:
    brisbane = _cell(-27.4705, 153.026)
    sydney = _cell(-33.8688, 151.2093)
    evidence = pl.DataFrame(
        [
            _evidence("target-local", TARGET_KEY, brisbane),
            _evidence("target-country", TARGET_KEY, sydney, dataset="dataset-b"),
            _evidence("congener-country", CONGENER_KEY, sydney),
        ]
    )
    memberships = pl.DataFrame(
        [
            _scope("brisbane-cluster", "exact", cell_id=brisbane),
            _scope("brisbane-cluster", "country", country_code="AU"),
            _scope("no_geo", "global"),
        ]
    )

    result = build_regional_taxon_occurrence_index(
        evidence,
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
        scope_memberships=memberships,
    )
    rows = {
        (row["regional_scope_id"], row["accepted_taxon_key"]): row
        for row in result.to_dicts()
    }

    assert rows[("brisbane-cluster", TARGET_KEY)]["overlap_type"] == "exact"
    assert rows[("brisbane-cluster", TARGET_KEY)]["occurrence_count"] == 1
    assert rows[("brisbane-cluster", CONGENER_KEY)]["overlap_type"] == "country"
    assert rows[("brisbane-cluster", CONGENER_KEY)]["occurrence_count"] == 1
    assert rows[("no_geo", TARGET_KEY)]["overlap_type"] == "global"
    assert rows[("no_geo", TARGET_KEY)]["occurrence_count"] == 2
    assert rows[("no_geo", TARGET_KEY)]["independent_dataset_count"] == 2


def test_cluster_scope_memberships_include_exact_buffer_country_and_no_geo_global() -> None:
    grid = default_cell_grid()
    core = _cell(-27.4705, 153.026)
    adjacent = grid.neighbours(core)[0]
    clusters = pl.DataFrame(
        [
            {
                "geo_cluster_id": "brisbane",
                "member_cell_ids": [core],
                "source_resolution": 5,
                "countries": ["AU"],
                "candidate_distribution_only": True,
            },
            {
                "geo_cluster_id": "no_geo",
                "member_cell_ids": [],
                "source_resolution": 5,
                "countries": [],
                "candidate_distribution_only": True,
            },
        ]
    )

    memberships = build_flickr_cluster_scope_memberships(clusters)

    assert memberships.filter(
        (pl.col("regional_scope_id") == "brisbane")
        & (pl.col("overlap_type") == "exact")
    )["spatial_cell_id"].to_list() == [core]
    assert adjacent in memberships.filter(
        (pl.col("regional_scope_id") == "brisbane")
        & (pl.col("overlap_type") == "buffer")
    )["spatial_cell_id"].to_list()
    assert memberships.filter(pl.col("overlap_type") == "country")[
        "country_code"
    ].to_list() == ["AU"]
    no_geo = memberships.filter(pl.col("regional_scope_id") == "no_geo").to_dicts()
    assert len(no_geo) == 1
    assert no_geo[0]["overlap_type"] == "global"


def test_rejects_unreconciled_taxa_and_contradictory_eligible_evidence() -> None:
    cell_id = _cell(-27.4705, 153.026)
    unknown = pl.DataFrame([_evidence("unknown", "gbif:missing", cell_id)])
    with pytest.raises(ValueError, match="not accepted species in the registry"):
        build_regional_taxon_occurrence_index(
            unknown,
            _taxa(),
            evidence_version=EVIDENCE_VERSION,
            registry_version=REGISTRY_VERSION,
        )

    malformed_resolution = pl.DataFrame(
        [{**_evidence("fractional", TARGET_KEY, cell_id), "spatial_resolution": 5.5}]
    )
    with pytest.raises(TypeError, match="spatial_resolution must be an integer"):
        build_regional_taxon_occurrence_index(
            malformed_resolution,
            _taxa(),
            evidence_version=EVIDENCE_VERSION,
            registry_version=REGISTRY_VERSION,
        )

    contradictory_row = _evidence("preserved", TARGET_KEY, cell_id)
    contradictory_row["preserved_specimen"] = True
    contradictory = pl.DataFrame([contradictory_row])
    with pytest.raises(ValueError, match="contradicts range_inference_eligible"):
        build_regional_taxon_occurrence_index(
            contradictory,
            _taxa(),
            evidence_version=EVIDENCE_VERSION,
            registry_version=REGISTRY_VERSION,
        )


def test_deduplicates_identical_source_records_and_rejects_conflicts() -> None:
    cell_id = _cell(-27.4705, 153.026)
    row = _evidence("duplicate", TARGET_KEY, cell_id)
    deduped = build_regional_taxon_occurrence_index(
        pl.DataFrame([row, dict(row)]),
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
    )
    assert deduped["occurrence_count"].to_list() == [1]

    conflict = dict(row)
    conflict["accepted_taxon_key"] = CONGENER_KEY
    with pytest.raises(ValueError, match="conflicting occurrence evidence"):
        build_regional_taxon_occurrence_index(
            pl.DataFrame([row, conflict]),
            _taxa(),
            evidence_version=EVIDENCE_VERSION,
            registry_version=REGISTRY_VERSION,
        )


def test_output_is_deterministic_and_writes_exact_physical_schema(tmp_path: Path) -> None:
    brisbane = _cell(-27.4705, 153.026)
    sydney = _cell(-33.8688, 151.2093)
    evidence = pl.DataFrame(
        [
            _evidence("2", CONGENER_KEY, sydney),
            _evidence("1", TARGET_KEY, brisbane),
            _evidence("ignored", TARGET_KEY, brisbane, eligible=False),
        ]
    )
    first = build_regional_taxon_occurrence_index(
        evidence,
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
    )
    second = build_regional_taxon_occurrence_index(
        evidence.reverse(),
        _taxa().reverse(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
    )
    assert first.equals(second)

    output = write_regional_taxon_occurrence(first, tmp_path)
    assert output == tmp_path / REGIONAL_TAXON_OCCURRENCE_FILE
    restored = pl.read_parquet(output)
    assert restored.equals(first)
    assert restored.schema == regional_taxon_occurrence_schema()


def test_coordinate_confidence_is_unknown_when_any_selected_uncertainty_is_missing() -> None:
    cell_id = _cell(-27.4705, 153.026)
    result = build_regional_taxon_occurrence_index(
        pl.DataFrame(
            [
                _evidence("known", TARGET_KEY, cell_id),
                _evidence("unknown", TARGET_KEY, cell_id, uncertainty_m=None),
            ]
        ),
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
    )

    assert result["occurrence_count"].to_list() == [2]
    assert result["coordinate_confidence"].to_list() == [None]


def test_consumes_phase1_occurrence_evidence_physical_schema() -> None:
    cell_id = _cell(-27.4705, 153.026)
    row = {name: None for name in geographic_occurrence_evidence_schema()}
    row.update(
        {
            "schema_version": "geographic-occurrence-evidence-v1.0.0",
            "gbif_id": "phase1-record",
            "registry_version": REGISTRY_VERSION,
            "accepted_taxon_key": TARGET_KEY,
            "gbif_species_key": 1_938_069,
            "scientific_name": "Papilio demoleus",
            "source": "GBIF",
            "source_dataset_key": "phase1-dataset",
            "source_dataset_citation": "Phase 1 fixture",
            "source_query_hash": "sha256:" + "1" * 64,
            "spatial_cell_id": cell_id,
            "spatial_resolution": 5,
            "country_code": "AU",
            "admin1": "Queensland",
            "bioregion": "australasia",
            "centroid_latitude": -27.47,
            "centroid_longitude": 153.03,
            "coordinate_uncertainty_m": 250.0,
            "event_date": date(2025, 2, 3),
            "basis_of_record": "HUMAN_OBSERVATION",
            "establishment_means": "NATIVE",
            "occurrence_status": "PRESENT",
            "known_range_role": "current",
            "has_geospatial_issue": False,
            "preserved_specimen": False,
            "fossil": False,
            "range_inference_eligible": True,
            "taxon_key_match": True,
            "coordinate_valid": True,
            "retrieved_at": datetime(2026, 7, 13, tzinfo=UTC),
            "source_snapshot_version": "gbif-phase1-fixture",
        }
    )
    evidence = pl.DataFrame([row], schema=geographic_occurrence_evidence_schema())

    result = build_regional_taxon_occurrence_index(
        evidence,
        _taxa(),
        evidence_version=EVIDENCE_VERSION,
        registry_version=REGISTRY_VERSION,
        classification_paths=_classification_paths(),
    )

    assert result.height == 1
    assert result["accepted_taxon_key"].to_list() == [TARGET_KEY]
    assert result["occurrence_count"].to_list() == [1]
    assert result["source_dataset_keys"].to_list() == [["phase1-dataset"]]
