from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from biominer.geography import (
    GeographicCoordinate,
    GeographicResolutions,
    default_cell_grid,
)
from biominer.registry.geographic_spread import (
    OccurrenceBatch,
    build_taxon_geographic_spread,
    geographic_occurrence_evidence_schema,
    geographic_spread_schema,
)
from biominer.registry.geographic_summary import (
    GEOGRAPHIC_QA_FINDINGS_FILE,
    TAXON_GEOGRAPHIC_SUMMARY_FILE,
    GeographicSummaryPolicy,
    build_geographic_summary,
    geographic_summary_schema,
)


TARGET_KEY = "gbif:1938069"
REGISTRY_VERSION = "butterflies-v1"
SNAPSHOT_VERSION = "gbif-download:fixture-1"
RETRIEVED_AT = "2026-07-13T00:00:00Z"
RESOLUTIONS = GeographicResolutions(coarse=3, regional=5, local=7)


class BatchSource:
    source = "GBIF"
    source_query_hash = "sha256:" + ("2" * 64)
    source_snapshot_version = SNAPSHOT_VERSION

    def __init__(self, records: tuple[dict[str, object], ...]) -> None:
        self.records = records

    def iter_batches(self, *, start_cursor: int = 0) -> Iterator[OccurrenceBatch]:
        selected = self.records[start_cursor:]
        yield OccurrenceBatch(
            cursor=start_cursor,
            next_cursor=len(self.records),
            records=selected,
            end_of_records=True,
            total_records=len(self.records),
        )


def test_geographic_summary_uses_eligible_cells_and_versions_the_policy(
    tmp_path: Path,
) -> None:
    grid = default_cell_grid()
    origin = grid.coordinate_to_cell(
        GeographicCoordinate(latitude=-27.4705, longitude=153.0260),
        resolution=5,
    )
    neighbour = grid.neighbours(origin)[0]
    neighbour_center = grid.center(neighbour)
    records = (
        _occurrence("1", latitude=-27.4705, longitude=153.0260, admin1="Queensland"),
        _occurrence(
            "2",
            latitude=float(neighbour_center.latitude),
            longitude=float(neighbour_center.longitude),
            admin1="Queensland",
        ),
        _occurrence("3", latitude=-12.4634, longitude=130.8456, admin1="Northern Territory"),
        _occurrence(
            "4",
            latitude=-31.9523,
            longitude=115.8613,
            admin1="Western Australia",
            establishment_means="INTRODUCED",
        ),
        _occurrence(
            "5",
            latitude=-33.8688,
            longitude=151.2093,
            admin1="New South Wales",
            basis_of_record="PRESERVED_SPECIMEN",
            event_date="1980-01-01",
        ),
    )
    spread, evidence = _build_spread(tmp_path / "spread", records)
    policy = GeographicSummaryPolicy(
        component_resolution=5,
        min_eligible_occurrences=3,
        min_occupied_cells=2,
        outlier_max_eligible_occurrences=1,
        outlier_neighbour_distance=3,
        current_window_years=20,
    )

    result = build_geographic_summary(
        spread=spread,
        occurrence_evidence=evidence,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        policy=policy,
        output_dir=tmp_path / "summary",
        created_at=RETRIEVED_AT,
        grid=grid,
    )

    assert result.summary.schema == geographic_summary_schema()
    assert result.summary.height == 1
    row = result.summary.row(0, named=True)
    assert row["accepted_taxon_key"] == TARGET_KEY
    assert row["cell_counts_by_resolution"] == [
        {"resolution": 3, "count": 3},
        {"resolution": 5, "count": 4},
        {"resolution": 7, "count": 4},
    ]
    assert row["countries"] == ["AU"]
    assert row["disconnected_range_component_count"] == 3
    assert row["occurrence_density_summary"] == {
        "min": 1.0,
        "p50": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }
    assert row["data_deficient"] is False
    assert row["suspicious_outlier_cell_count"] == 1
    assert row["range_source_coverage"] == [
        {"source": "GBIF", "dataset_count": 1, "eligible_occurrence_count": 4}
    ]
    assert row["known_introduced_regions"] == ["admin1:AU:Western Australia"]
    assert row["current_evidence_count"] == 4
    assert row["historical_evidence_count"] == 1
    assert row["geographic_evidence_version"].startswith("sha256:")
    assert row["spread_fingerprint"].startswith("sha256:")
    assert set(result.qa["code"].to_list()) == {"geographic_extreme_isolated_outlier"}
    assert (tmp_path / "summary" / TAXON_GEOGRAPHIC_SUMMARY_FILE).exists()
    assert (tmp_path / "summary" / GEOGRAPHIC_QA_FINDINGS_FILE).exists()
    assert result.manifest["summary_row_count"] == 1
    assert result.manifest["qa_warning_count"] == 1


def test_geographic_qa_retains_invalid_mismatch_and_preserved_only_evidence(
    tmp_path: Path,
) -> None:
    records = (
        _occurrence("preserved", basis_of_record="PRESERVED_SPECIMEN"),
        _occurrence("invalid", latitude=91.0),
        _occurrence("mismatch", species_key=999),
    )
    spread, evidence = _build_spread(tmp_path / "spread", records)

    result = build_geographic_summary(
        spread=spread,
        occurrence_evidence=evidence,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        policy=GeographicSummaryPolicy(component_resolution=5),
        output_dir=tmp_path / "summary",
        created_at=RETRIEVED_AT,
    )

    row = result.summary.row(0, named=True)
    assert row["data_deficient"] is True
    assert "no_range_inference_eligible_occurrences" in row["data_deficient_reasons"]
    assert row["cell_counts_by_resolution"] == []
    assert row["current_evidence_count"] == 0
    codes = set(result.qa["code"].to_list())
    assert codes == {
        "geographic_invalid_coordinate",
        "geographic_occurrences_only_from_preserved_specimens",
        "geographic_taxon_key_mismatch",
    }


def test_geographic_qa_detects_impossible_cells_and_range_role_conflicts(
    tmp_path: Path,
) -> None:
    spread, evidence = _build_spread(tmp_path / "spread", (_occurrence("1"),))
    regional = spread.filter(pl.col("spatial_resolution") == 5).row(0, named=True)
    impossible = {**regional, "spatial_cell_id": "not-a-cell"}
    introduced = {**regional, "known_range_role": "introduced"}
    corrupted_spread = pl.concat(
        [
            spread,
            pl.DataFrame([impossible, introduced], schema=geographic_spread_schema()),
        ],
        how="vertical",
    )

    result = build_geographic_summary(
        spread=corrupted_spread,
        occurrence_evidence=evidence,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        policy=GeographicSummaryPolicy(component_resolution=5),
        output_dir=tmp_path / "summary",
        created_at=RETRIEVED_AT,
    )

    findings = {(row["severity"], row["code"]) for row in result.qa.to_dicts()}
    assert ("fatal", "geographic_impossible_cell_identifier") in findings
    assert ("warning", "geographic_conflicting_native_introduced_evidence") in findings
    assert result.manifest["qa_status"] == "failed"


def test_geographic_summary_emits_data_deficient_rows_for_taxa_without_evidence(
    tmp_path: Path,
) -> None:
    result = build_geographic_summary(
        spread=pl.DataFrame(schema=geographic_spread_schema()),
        occurrence_evidence=pl.DataFrame(schema=geographic_occurrence_evidence_schema()),
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        policy=GeographicSummaryPolicy(component_resolution=5),
        output_dir=tmp_path / "summary",
        created_at=RETRIEVED_AT,
    )

    row = result.summary.row(0, named=True)
    assert row["data_deficient"] is True
    assert row["data_deficient_reasons"] == [
        "no_georeferenced_evidence",
        "no_range_inference_eligible_occurrences",
    ]
    assert result.qa.to_dicts() == [
        {
            "severity": "warning",
            "code": "geographic_no_georeferenced_evidence",
            "subject": TARGET_KEY,
        }
    ]


def test_geographic_summary_and_fingerprints_are_input_order_independent(
    tmp_path: Path,
) -> None:
    spread, evidence = _build_spread(
        tmp_path / "spread",
        (
            _occurrence("2", latitude=-31.9523, longitude=115.8613),
            _occurrence("1", latitude=-27.4705, longitude=153.0260),
        ),
    )
    kwargs = {
        "taxa": _taxa(),
        "registry_version": REGISTRY_VERSION,
        "policy": GeographicSummaryPolicy(component_resolution=5),
        "created_at": RETRIEVED_AT,
    }

    ordered = build_geographic_summary(
        spread=spread,
        occurrence_evidence=evidence,
        output_dir=tmp_path / "ordered",
        **kwargs,
    )
    reversed_input = build_geographic_summary(
        spread=spread.reverse(),
        occurrence_evidence=evidence.reverse(),
        output_dir=tmp_path / "reversed",
        **kwargs,
    )

    assert ordered.summary.equals(reversed_input.summary)
    assert ordered.qa.equals(reversed_input.qa)


def _build_spread(
    output: Path,
    records: tuple[dict[str, object], ...],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    result = build_taxon_geographic_spread(
        accepted_taxon_key=TARGET_KEY,
        scientific_name="Papilio demoleus",
        registry_version=REGISTRY_VERSION,
        source=BatchSource(records),
        resolutions=RESOLUTIONS,
        output_dir=output / "artifacts",
        checkpoint_dir=output / "checkpoint",
        retrieved_at=RETRIEVED_AT,
    )
    return result.spread, pl.read_parquet(result.evidence_path)


def _taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": TARGET_KEY,
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
            }
        ]
    )


def _occurrence(
    gbif_id: str,
    *,
    latitude: float = -27.4705,
    longitude: float = 153.0260,
    admin1: str = "Queensland",
    basis_of_record: str = "HUMAN_OBSERVATION",
    event_date: str = "2025-01-01",
    establishment_means: str = "NATIVE",
    species_key: int = 1_938_069,
) -> dict[str, object]:
    return {
        "key": gbif_id,
        "acceptedTaxonKey": species_key,
        "speciesKey": species_key,
        "acceptedScientificName": "Papilio demoleus",
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
        "coordinateUncertaintyInMeters": 25.0,
        "countryCode": "AU",
        "stateProvince": admin1,
        "datasetKey": "dataset-1",
        "datasetCitation": "Fixture observations. GBIF.org occurrence dataset.",
        "basisOfRecord": basis_of_record,
        "establishmentMeans": establishment_means,
        "occurrenceStatus": "PRESENT",
        "eventDate": event_date,
        "issues": [],
    }
