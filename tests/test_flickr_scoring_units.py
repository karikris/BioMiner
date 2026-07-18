from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.flickr_fetch.scoring_units import (
    FLICKR_PHOTO_EMBEDDING_UNITS_FILE,
    FLICKR_SCORING_ASSOCIATIONS_FILE,
    FLICKR_SCORING_CANDIDATES_FILE,
    FLICKR_SCORING_UNITS_FILE,
    FlickrScoringUnitArtifacts,
    build_flickr_scoring_unit_artifacts,
    validate_flickr_scoring_unit_artifacts,
    write_flickr_scoring_unit_artifacts,
)
from biominer.vision.target_full_frame import build_target_full_frame_plan


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64


def test_separates_photo_organism_association_and_candidate_grains() -> None:
    plan = _two_route_plan()
    adult, larval = plan.scoring_units

    artifacts = build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-3.1.1",
        associations=[
            _query_association(),
            _target_association(route="adult_field"),
        ],
        candidate_species=[
            _candidate(adult.scoring_unit_id, "7724053", priority=1),
            _candidate(adult.scoring_unit_id, "1938687", priority=2),
            _candidate(larval.scoring_unit_id, "7724053", priority=1),
        ],
    )

    assert artifacts.photo_embedding_units.height == 1
    assert artifacts.scoring_units.height == 2
    assert artifacts.associations.height == 3
    assert artifacts.candidate_species.height == 3
    assert artifacts.scoring_units["organism_unit_id"].to_list() == [
        adult.scoring_unit_id,
        larval.scoring_unit_id,
    ]
    assert artifacts.scoring_units["route"].to_list() == ["adult_field", "larval"]
    assert artifacts.associations.group_by("association_kind").len().sort(
        "association_kind"
    ).to_dicts() == [
        {"association_kind": "query", "len": 2},
        {"association_kind": "target", "len": 1},
    ]
    assert artifacts.candidate_species.group_by("organism_unit_id").len().sort(
        "organism_unit_id"
    )["len"].to_list() == [2, 1]

    photo = artifacts.photo_embedding_units.row(0, named=True)
    assert set(artifacts.scoring_units["photo_embedding_unit_id"]) == {
        photo["photo_embedding_unit_id"]
    }
    assert set(artifacts.scoring_units["model_input_signature"]) == {
        photo["model_input_signature"]
    }
    all_columns = {
        *artifacts.photo_embedding_units.columns,
        *artifacts.scoring_units.columns,
        *artifacts.associations.columns,
        *artifacts.candidate_species.columns,
    }
    assert "embedding" not in all_columns
    assert "embedding_vector" not in all_columns
    assert "image_bytes" not in all_columns


def test_artifacts_are_deterministic_and_round_trip_as_closed_parquet(tmp_path) -> None:
    plan = _two_route_plan()
    adult, larval = plan.scoring_units
    associations = [_query_association(), _target_association(route="adult_field")]
    candidates = [
        _candidate(adult.scoring_unit_id, "7724053", priority=1),
        _candidate(larval.scoring_unit_id, "7724053", priority=1),
    ]

    first = build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-3.1.1",
        associations=associations,
        candidate_species=candidates,
    )
    second = build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-3.1.1",
        associations=list(reversed(associations)),
        candidate_species=list(reversed(candidates)),
    )

    assert first.photo_embedding_units.equals(second.photo_embedding_units)
    assert first.scoring_units.equals(second.scoring_units)
    assert first.associations.equals(second.associations)
    assert first.candidate_species.equals(second.candidate_species)

    paths = write_flickr_scoring_unit_artifacts(first, tmp_path)
    assert {path.name for path in paths.values()} == {
        FLICKR_PHOTO_EMBEDDING_UNITS_FILE,
        FLICKR_SCORING_UNITS_FILE,
        FLICKR_SCORING_ASSOCIATIONS_FILE,
        FLICKR_SCORING_CANDIDATES_FILE,
    }
    round_trip = FlickrScoringUnitArtifacts(
        photo_embedding_units=pl.read_parquet(paths["photo_embedding_units"]),
        scoring_units=pl.read_parquet(paths["scoring_units"]),
        associations=pl.read_parquet(paths["associations"]),
        candidate_species=pl.read_parquet(paths["candidate_species"]),
    )
    validate_flickr_scoring_unit_artifacts(round_trip)
    assert round_trip.photo_embedding_units.equals(first.photo_embedding_units)
    assert round_trip.scoring_units.equals(first.scoring_units)
    assert round_trip.associations.equals(first.associations)
    assert round_trip.candidate_species.equals(first.candidate_species)


def test_duplicate_content_keeps_photo_provenance_but_one_model_input_signature() -> None:
    image = DecodedImage(width=2, height=2, mode="RGB", data=bytes(range(12)))
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row("photo-1", "det-1"),
            _detection_row("photo-2", "det-2"),
        ],
        image_loader=lambda _row: image,
    )

    artifacts = build_flickr_scoring_unit_artifacts(plan, run_id="run-duplicate")

    assert len(plan.visual_inputs) == 1
    assert artifacts.photo_embedding_units.height == 2
    assert artifacts.photo_embedding_units["visual_input_id"].n_unique() == 1
    assert artifacts.photo_embedding_units["model_input_signature"].n_unique() == 1
    assert artifacts.photo_embedding_units["photo_embedding_unit_id"].n_unique() == 2


def test_fails_closed_on_invalid_many_to_many_rows_and_tampering() -> None:
    plan = _two_route_plan()
    adult = plan.scoring_units[0]
    query = _query_association()
    with pytest.raises(ValueError, match="association identity"):
        build_flickr_scoring_unit_artifacts(
            plan,
            run_id="run-3.1.1",
            associations=[query, query],
        )
    with pytest.raises(ValueError, match="unknown organism unit"):
        build_flickr_scoring_unit_artifacts(
            plan,
            run_id="run-3.1.1",
            candidate_species=[
                _candidate("sha256:" + "9" * 64, "7724053", priority=1)
            ],
        )
    with pytest.raises(ValueError, match="query association requires"):
        build_flickr_scoring_unit_artifacts(
            plan,
            run_id="run-3.1.1",
            associations=[
                {
                    **query,
                    "flickr_query_id": None,
                    "query_hash": None,
                }
            ],
        )

    artifacts = build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-3.1.1",
        candidate_species=[
            _candidate(adult.scoring_unit_id, "7724053", priority=1)
        ],
    )
    tampered = FlickrScoringUnitArtifacts(
        photo_embedding_units=artifacts.photo_embedding_units,
        scoring_units=artifacts.scoring_units,
        associations=artifacts.associations,
        candidate_species=artifacts.candidate_species.with_columns(
            pl.lit("Papilio fabricated").alias("candidate_scientific_name")
        ),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_flickr_scoring_unit_artifacts(tampered)


def _two_route_plan():
    image = DecodedImage(width=2, height=2, mode="RGB", data=bytes(range(12)))
    return build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                "photo-1",
                "adult-1",
                route="adult_field",
                detection_route="adult_butterfly_field",
                bbox=(0.0, 0.0, 1.0, 2.0),
            ),
            _detection_row(
                "photo-1",
                "larval-1",
                route="larval",
                detection_route="caterpillar_field",
                bbox=(1.0, 0.0, 2.0, 2.0),
            ),
        ],
        image_loader=lambda _row: image,
    )


def _query_association() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "association_kind": "query",
        "association_source": "flickr_query_hits",
        "association_source_id": "query-hash-1",
        "flickr_query_id": "query-definition-1",
        "query_hash": "query-hash-1",
        "query_tier": "species_scientific:high:tags",
        "search_term": "Papilio demoleus",
        "accepted_taxon_key": "7724053",
        "scientific_name": "Papilio demoleus",
    }


def _target_association(*, route: str) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "route": route,
        "association_kind": "target",
        "association_source": "configured_target",
        "association_source_id": "target:7724053",
        "accepted_taxon_key": "7724053",
        "scientific_name": "Papilio demoleus",
    }


def _candidate(
    organism_unit_id: str,
    taxon_key: str,
    *,
    priority: int,
) -> dict[str, object]:
    scientific_name = (
        "Papilio demoleus" if taxon_key == "7724053" else "Papilio polytes"
    )
    return {
        "organism_unit_id": organism_unit_id,
        "candidate_accepted_taxon_key": taxon_key,
        "candidate_scientific_name": scientific_name,
        "family_key": "1933990",
        "family_name": "Papilionidae",
        "genus_key": "1938686",
        "genus_name": "Papilio",
        "candidate_priority": priority,
        "candidate_reasons": ["regional", "query_associated"],
        "candidate_source_ids": ["regional-union-1", "query-definition-1"],
    }


def _detection_row(
    photo_id: str,
    detection_id: str,
    *,
    route: str = "adult_field",
    detection_route: str = "adult_butterfly_field",
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 2.0),
) -> dict[str, Any]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": "sha256:" + "f" * 64,
        "detection_id": detection_id,
        "detection_status": "detected",
        "detector_score": 0.9,
        "detector_label": "butterfly_like",
        "bbox_xyxy": list(bbox),
        "bbox_xyxyn": [bbox[0] / 2, bbox[1] / 2, bbox[2] / 2, bbox[3] / 2],
        "mask_polygon_xyn": None,
        "detection_route": detection_route,
        "routing_action": "score",
        "bioclip_route": route,
        "routing_policy_version": "detection-routing-policy-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
        "schema_version": "object-detection-v2",
    }
