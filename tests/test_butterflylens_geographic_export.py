"""Tests for the ButterflyLens geographic candidate-evidence export."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from biominer.integration.butterflylens_geographic_export import (
    BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION,
    BUTTERFLYLENS_TARGET_CELL_VERSION,
    build_butterflylens_geographic_impact,
    export_butterflylens_geographic_impact,
    validate_butterflylens_geographic_export,
)
from helpers.butterflylens_handoff_fixture import (
    build_butterflylens_model_fixture,
    sha,
)


def _record(flickr_record_id: str, *, no_geo: bool = False) -> dict[str, object]:
    return {
        "flickr_record_id": flickr_record_id,
        "geography_availability": "no_geo" if no_geo else "h3",
        "h3_cell": None if no_geo else "8928308280fffff",
        "h3_version": None if no_geo else "4.3.0",
        "h3_resolution": None if no_geo else 9,
        "source_precision_metres": None if no_geo else 20.0,
        "published_h3_resolution": None if no_geo else 9,
        "public_geometry_status": "withheld" if no_geo else "available",
        "public_geometry_reason": (
            "source has no publishable geography" if no_geo else None
        ),
        "latest_flickr_event_date": "2026-07-17",
        "geographic_evidence_fingerprint": sha("1"),
    }


def _frame(*, no_geo: bool = False) -> pl.DataFrame:
    fixture = build_butterflylens_model_fixture()
    source_id = fixture["layer"].flickr_source_records["flickr_record_id"][0]
    return build_butterflylens_geographic_impact(
        model_layer=fixture["layer"],
        geographic_records=[_record(source_id, no_geo=no_geo)],
        source_commit="2" * 40,
    )


def test_located_projection_exports_only_supported_candidate_counts() -> None:
    frame = _frame()

    assert frame.height == 2
    assert frame["schema_version"].unique().to_list() == [
        BUTTERFLYLENS_GEOGRAPHIC_IMPACT_VERSION
    ]
    assert frame["target_cell_schema_version"].unique().to_list() == [
        BUTTERFLYLENS_TARGET_CELL_VERSION
    ]
    assert frame["geography_availability"].unique().to_list() == ["h3"]
    assert frame["cell_id"].unique().to_list() == ["8928308280fffff"]
    assert frame["flickr_candidate_count"].to_list() == [1, 1]
    assert frame["bioclip_species_candidate_count"].to_list() == [1, 1]
    assert frame["flickr_candidate_count_state"].unique().to_list() == ["available"]
    for name in (
        "ala_baseline",
        "yoloe_butterfly",
        "community_reviewed",
        "human_supported",
        "release_ready",
    ):
        assert frame[f"{name}_count"].null_count() == frame.height
        assert frame[f"{name}_count_state"].unique().to_list() == ["unavailable"]
    assert frame["provider_union_fingerprint"].null_count() == frame.height
    assert frame["potential_coverage_gap"].null_count() == frame.height
    assert frame["candidate_only_is_occurrence"].to_list() == [False, False]
    assert frame["scientific_claim_allowed"].to_list() == [False, False]


def test_no_geo_is_an_explicit_exclusion_not_zero_or_absence() -> None:
    frame = _frame(no_geo=True)

    assert frame["geography_availability"].unique().to_list() == ["no_geo"]
    assert frame["cell_id"].null_count() == frame.height
    assert frame["h3_resolution"].null_count() == frame.height
    assert frame["flickr_candidate_count"].null_count() == frame.height
    assert frame["bioclip_species_candidate_count"].null_count() == frame.height
    assert frame["data_deficiency_state"].unique().to_list() == [
        "insufficient_precision"
    ]
    assert frame["no_geo_is_biological_absence"].to_list() == [False, False]
    assert frame["public_geometry_status"].unique().to_list() == ["withheld"]


def test_geographic_input_requires_exact_source_coverage_and_valid_h3() -> None:
    fixture = build_butterflylens_model_fixture()
    layer = fixture["layer"]
    source_id = layer.flickr_source_records["flickr_record_id"][0]

    with pytest.raises(ValueError, match="must be nonempty"):
        build_butterflylens_geographic_impact(
            model_layer=layer,
            geographic_records=[],
            source_commit="2" * 40,
        )

    extra = _record(source_id)
    extra["latitude"] = -33.8
    with pytest.raises(ValueError, match="input fields differ"):
        build_butterflylens_geographic_impact(
            model_layer=layer,
            geographic_records=[extra],
            source_commit="2" * 40,
        )

    malformed = _record(source_id)
    malformed["h3_cell"] = "not-an-h3-cell"
    with pytest.raises(ValueError, match="H3 identity"):
        build_butterflylens_geographic_impact(
            model_layer=layer,
            geographic_records=[malformed],
            source_commit="2" * 40,
        )


def test_geographic_export_is_deterministic_create_only_and_source_bound(
    tmp_path: Path,
) -> None:
    frame = _frame()
    first = export_butterflylens_geographic_impact(
        frame=frame, output_root=tmp_path / "first"
    )
    second = export_butterflylens_geographic_impact(
        frame=frame, output_root=tmp_path / "second"
    )

    validate_butterflylens_geographic_export(first.root, first.artifact)
    assert first.artifact["role"] == "geographic_impact"
    assert first.artifact["relative_path"] == (
        "artifacts/geographic/butterflylens_geographic_impact_cells.parquet"
    )
    assert first.artifact["sha256"] == second.artifact["sha256"]
    assert (
        first.artifact["semantic_fingerprint"]
        == second.artifact["semantic_fingerprint"]
    )
    assert sha("1") in first.artifact["parent_fingerprints"]
    with pytest.raises(FileExistsError, match="create-only"):
        export_butterflylens_geographic_impact(frame=frame, output_root=first.root)


def test_geographic_export_rejects_descriptor_and_content_tampering(
    tmp_path: Path,
) -> None:
    exported = export_butterflylens_geographic_impact(
        frame=_frame(), output_root=tmp_path
    )
    descriptor = deepcopy(exported.artifact)
    descriptor["parent_fingerprints"] = []
    with pytest.raises(ValueError, match="semantic identity differs"):
        validate_butterflylens_geographic_export(exported.root, descriptor)

    tampered = pl.read_parquet(exported.path).with_columns(
        pl.lit(True).alias("candidate_only_is_occurrence")
    )
    tampered.write_parquet(exported.path)
    with pytest.raises(ValueError, match="physical identity differs"):
        validate_butterflylens_geographic_export(exported.root, exported.artifact)
