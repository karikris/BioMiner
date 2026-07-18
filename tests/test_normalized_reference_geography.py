"""Tests for observation-grained reference geography normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.geography import GeographicCoordinate, GeographicResolutions
from biominer.references.normalized_geography import (
    NORMALIZED_REFERENCE_GEOGRAPHY_FILE,
    NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION,
    ReferenceGeographyPrecisionPolicy,
    build_normalized_reference_geography,
    normalized_reference_geography_artifact_fingerprint,
    normalized_reference_geography_schema,
    validate_normalized_reference_geography,
    write_normalized_reference_geography,
)
from biominer.references.schemas import (
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_observation_id,
    reference_observations_frame,
)


NOW = datetime(2026, 7, 18, tzinfo=UTC)
RESOLUTIONS = GeographicResolutions(coarse=3, regional=5, local=7)
POLICY = ReferenceGeographyPrecisionPolicy(
    local_max_uncertainty_m=1_000,
    regional_max_uncertainty_m=10_000,
    coarse_max_uncertainty_m=100_000,
)


class _Grid:
    name = "fixture_hierarchical_grid"
    version = "fixture-grid:v1"

    def coordinate_to_cell(
        self, coordinate: GeographicCoordinate, *, resolution: int
    ) -> str:
        return (
            f"fixture-r{resolution}:"
            f"{float(coordinate.latitude):.4f}:{float(coordinate.longitude):.4f}"
        )

    def parent(self, cell_id: str, *, resolution: int | None = None) -> str:
        return f"{cell_id}:parent:{resolution}"

    def neighbours(
        self,
        cell_id: str,
        *,
        grid_distance: int = 1,
        include_origin: bool = False,
    ) -> tuple[str, ...]:
        return (cell_id,) if include_origin else ()

    def center(self, cell_id: str) -> GeographicCoordinate:
        return GeographicCoordinate(-33.87, 151.21)

    def is_valid(self, cell_id: object) -> bool:
        return isinstance(cell_id, str) and cell_id.startswith("fixture-r")


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _observation(source_id: str = "1", **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": make_reference_observation_id("GBIF", source_id),
        "source": "GBIF",
        "source_observation_id": source_id,
        "source_taxon_id": "1938069",
        "supplied_scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "reconciled_scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v2-20260718",
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": f"observer-{source_id}",
        "locality": "Sydney",
        "life_stage": "adult",
        "sex": None,
        "observed_at": datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        "latitude": -33.87,
        "longitude": 151.21,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-au",
        "distance_to_cluster_medoid_km": 4.2,
        "source_dataset_key": "dataset-1",
        "source_dataset_doi": "10.15468/example",
        "source_record_url": f"https://example.test/occurrence/{source_id}",
        "source_record_hash": _sha("a"),
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-occurrence-2026-07-18",
        "source_query_fingerprint": _sha("b"),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": False,
        "basis_of_record_suitable": True,
    }
    row.update(changes)
    if row["latitude"] is None or row["longitude"] is None:
        row["distance_to_cluster_medoid_km"] = None
    return row


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return reference_observations_frame(list(rows))


def _context(observation_id: str) -> dict[str, object]:
    return {
        "reference_observation_id": observation_id,
        "continent_code": "oc",
        "admin1": "New South Wales",
        "bioregion": "Sydney Basin",
    }


def test_local_geography_preserves_source_snapshot_context_and_cells() -> None:
    observation = _observation()
    result = build_normalized_reference_geography(
        _frame(observation),
        resolutions=RESOLUTIONS,
        context_rows=[_context(str(observation["reference_observation_id"]))],
        policy=POLICY,
        grid=_Grid(),
    )

    row = result.row(0, named=True)
    assert row["schema_version"] == NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION
    assert row["source_snapshot_version"] == "gbif-occurrence-2026-07-18"
    assert row["coordinate_quality"] == "local"
    assert row["latitude"] == -33.87
    assert row["longitude"] == 151.21
    assert row["country_code"] == "AU"
    assert row["continent_code"] == "OC"
    assert row["admin1"] == "New South Wales"
    assert row["bioregion"] == "Sydney Basin"
    assert row["supported_cell_resolution"] == 7
    assert row["coarse_cell_id"].startswith("fixture-r3:")
    assert row["regional_cell_id"].startswith("fixture-r5:")
    assert row["local_cell_id"].startswith("fixture-r7:")
    assert row["geography_unavailable_reason"] is None
    assert row["observer_id_hash"].startswith("sha256:")
    assert row["row_fingerprint"].startswith("sha256:")


@pytest.mark.parametrize(
    ("uncertainty", "quality", "supported", "present_cells"),
    [
        (1_000.0, "local", 7, (True, True, True)),
        (1_001.0, "regional", 5, (True, True, False)),
        (10_001.0, "coarse", 3, (True, False, False)),
        (100_001.0, "unknown_precision", None, (False, False, False)),
        (None, "unknown_precision", None, (False, False, False)),
    ],
)
def test_uncertainty_never_manufactures_finer_cell_support(
    uncertainty: float | None,
    quality: str,
    supported: int | None,
    present_cells: tuple[bool, bool, bool],
) -> None:
    result = build_normalized_reference_geography(
        _frame(_observation(coordinate_uncertainty=uncertainty)),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )

    row = result.row(0, named=True)
    assert row["coordinate_quality"] == quality
    assert row["supported_cell_resolution"] == supported
    assert (
        tuple(
            row[field] is not None
            for field in ("coarse_cell_id", "regional_cell_id", "local_cell_id")
        )
        == present_cells
    )
    if quality == "unknown_precision":
        assert row["latitude"] == -33.87
        assert row["geography_unavailable_reason"]


@pytest.mark.parametrize(
    ("changes", "quality", "reason", "country_code"),
    [
        (
            {"coordinates_obscured": True},
            "withheld",
            "source_coordinates_obscured",
            "AU",
        ),
        (
            {"geospatial_issue": True},
            "invalid",
            "source_geospatial_issue",
            "AU",
        ),
        (
            {"latitude": None, "longitude": None},
            "country_only",
            "coordinates_missing_country_available",
            "AU",
        ),
        (
            {
                "latitude": None,
                "longitude": None,
                "country": None,
                "country_code": None,
            },
            "missing",
            "coordinates_missing",
            None,
        ),
    ],
)
def test_unusable_coordinates_are_explicit_and_never_local(
    changes: dict[str, object],
    quality: str,
    reason: str,
    country_code: str | None,
) -> None:
    result = build_normalized_reference_geography(
        _frame(_observation(**changes)),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )

    row = result.row(0, named=True)
    assert row["coordinate_quality"] == quality
    assert row["geography_unavailable_reason"] == reason
    assert row["country_code"] == country_code
    assert row["latitude"] is None
    assert row["longitude"] is None
    assert row["supported_cell_resolution"] is None
    assert row["coarse_cell_id"] is None
    assert row["regional_cell_id"] is None
    assert row["local_cell_id"] is None


def test_build_and_artifact_identity_are_order_independent() -> None:
    first = _observation("1")
    second = _observation("2", coordinate_uncertainty=20_000.0)

    forward = build_normalized_reference_geography(
        _frame(first, second),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )
    reverse = build_normalized_reference_geography(
        _frame(second, first),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )

    assert forward.equals(reverse)
    assert normalized_reference_geography_artifact_fingerprint(forward) == (
        normalized_reference_geography_artifact_fingerprint(reverse)
    )


def test_context_must_be_closed_unique_and_bound_to_an_observation() -> None:
    observation = _observation()
    context = _context(str(observation["reference_observation_id"]))
    extra = dict(context)
    extra["unexpected"] = "value"
    with pytest.raises(ValueError, match="context fields differ"):
        build_normalized_reference_geography(
            _frame(observation),
            resolutions=RESOLUTIONS,
            context_rows=[extra],
            policy=POLICY,
            grid=_Grid(),
        )
    with pytest.raises(ValueError, match="duplicates an observation"):
        build_normalized_reference_geography(
            _frame(observation),
            resolutions=RESOLUTIONS,
            context_rows=[context, context],
            policy=POLICY,
            grid=_Grid(),
        )
    unknown = _context(make_reference_observation_id("GBIF", "unknown"))
    with pytest.raises(ValueError, match="unknown observations"):
        build_normalized_reference_geography(
            _frame(observation),
            resolutions=RESOLUTIONS,
            context_rows=[unknown],
            policy=POLICY,
            grid=_Grid(),
        )


def test_validator_detects_precision_and_fingerprint_tampering() -> None:
    result = build_normalized_reference_geography(
        _frame(_observation()),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )
    false_local = result.with_columns(
        pl.lit(None, dtype=pl.String).alias("regional_cell_id")
    )
    with pytest.raises(ValueError, match="local geography requires"):
        validate_normalized_reference_geography(false_local)
    tampered = result.with_columns(pl.lit("Asia").alias("bioregion"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_normalized_reference_geography(tampered)

    policy_conflict_row = result.row(0, named=True)
    policy_conflict_row["coordinate_uncertainty_m"] = 2_000.0
    policy_conflict_row["row_fingerprint"] = canonical_semantic_fingerprint(
        {
            key: value
            for key, value in policy_conflict_row.items()
            if key != "row_fingerprint"
        }
    )
    policy_conflict = pl.DataFrame(
        [policy_conflict_row],
        schema=normalized_reference_geography_schema(),
        orient="row",
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicts with precision policy"):
        validate_normalized_reference_geography(policy_conflict)


def test_writer_uses_contract_filename_and_round_trips(tmp_path: Path) -> None:
    result = build_normalized_reference_geography(
        _frame(_observation()),
        resolutions=RESOLUTIONS,
        policy=POLICY,
        grid=_Grid(),
    )

    path = write_normalized_reference_geography(result, tmp_path / "geography")
    loaded = pl.read_parquet(path)

    assert path.name == NORMALIZED_REFERENCE_GEOGRAPHY_FILE
    assert loaded.schema == normalized_reference_geography_schema()
    validate_normalized_reference_geography(loaded)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_max_uncertainty_m": -1},
        {"regional_max_uncertainty_m": float("nan")},
        {
            "local_max_uncertainty_m": 10,
            "regional_max_uncertainty_m": 5,
        },
    ],
)
def test_precision_policy_rejects_invalid_thresholds(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="uncertainty thresholds"):
        ReferenceGeographyPrecisionPolicy(**kwargs)  # type: ignore[arg-type]
