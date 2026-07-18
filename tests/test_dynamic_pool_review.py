"""Tests for dynamic-pool Flickr audit and review contracts."""

from __future__ import annotations

import pytest

from biominer.evaluation.dynamic_pool_review import (
    DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
    RAW_SCORE_SEMANTICS,
    DynamicPoolAuditStrataPolicy,
    build_dynamic_pool_audit_frame,
    empty_dynamic_pool_audit_frame,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _candidate(**changes: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "sampling_unit_id": "review-unit-1",
        "source_record_hash": _sha("1"),
        "source_artifact_fingerprint": _sha("2"),
        "flickr_photo_id": "photo-1",
        "organism_unit_id": "organism-1",
        "candidate_family_accepted_taxon_key": "col:Papilionidae",
        "candidate_family_scientific_name": "Papilionidae",
        "candidate_genus_accepted_taxon_key": "col:Papilio",
        "candidate_genus_scientific_name": "Papilio",
        "candidate_species_accepted_taxon_key": "col:Papilio-demoleus",
        "candidate_species_scientific_name": "Papilio demoleus",
        "geographic_cluster_id": "geo-au-sydney",
        "no_geo": False,
        "primary_query_tier": "T2",
        "raw_fusion_score": 0.72,
        "raw_competitor_margin": 0.04,
        "pool_disagreement": 0.18,
        "route": "adult_field",
        "visual_domain": "field_photo",
        "subject_area_ratio": 0.08,
        "owner_group_id": "owner-1",
        "duplicate_group_id": "duplicate-1",
        "observation_group_id": "observation-1",
        "final_release_candidate": True,
    }
    candidate.update(changes)
    return candidate


def test_audit_strata_include_every_required_dimension() -> None:
    frame = build_dynamic_pool_audit_frame([_candidate()])
    row = frame.row(0, named=True)

    assert frame.schema == DYNAMIC_POOL_AUDIT_FRAME_SCHEMA
    assert row["candidate_family_accepted_taxon_key"] == "col:Papilionidae"
    assert row["candidate_genus_accepted_taxon_key"] == "col:Papilio"
    assert row["candidate_species_accepted_taxon_key"] == "col:Papilio-demoleus"
    assert row["geography_stratum"] == "geo:geo-au-sydney"
    assert row["primary_query_tier"] == "T2"
    assert row["raw_score_band"] == "band_02_lt_0.75"
    assert row["raw_margin_band"] == "band_01_lt_0.05"
    assert row["pool_disagreement_band"] == "band_02_gte_0.15"
    assert row["route_domain_stratum"] == "adult_field|field_photo"
    assert row["subject_size_band"] == "band_01_lt_0.1"
    assert row["owner_group_id"] == "owner-1"
    assert row["duplicate_group_id"] == "duplicate-1"
    assert row["observation_group_id"] == "observation-1"
    assert row["score_semantics"] == RAW_SCORE_SEMANTICS
    assert row["probability_available"] is False


def test_no_geo_is_explicit_without_claiming_biological_absence() -> None:
    frame = build_dynamic_pool_audit_frame(
        [_candidate(geographic_cluster_id=None, no_geo=True, pool_disagreement=None)]
    )
    row = frame.row(0, named=True)

    assert row["geography_stratum"] == "no_geo"
    assert row["geographic_cluster_id"] is None
    assert row["pool_disagreement_band"] == "unavailable"


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"owner_group_id": ""}, "owner_group_id"),
        ({"duplicate_group_id": None}, "duplicate_group_id"),
        ({"observation_group_id": ""}, "observation_group_id"),
        (
            {"geographic_cluster_id": "geo-au-sydney", "no_geo": True},
            "cannot claim",
        ),
        ({"geographic_cluster_id": None, "no_geo": False}, "require"),
    ],
)
def test_audit_frame_fails_closed_on_missing_or_inconsistent_groups(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        build_dynamic_pool_audit_frame([_candidate(**changes)])


def test_audit_frame_is_order_independent_and_requires_unique_units() -> None:
    second = _candidate(
        sampling_unit_id="review-unit-2",
        source_record_hash=_sha("3"),
        flickr_photo_id="photo-2",
        organism_unit_id="organism-2",
        owner_group_id="owner-2",
        duplicate_group_id="duplicate-2",
        observation_group_id="observation-2",
    )
    first = build_dynamic_pool_audit_frame([_candidate(), second])
    reversed_frame = build_dynamic_pool_audit_frame([second, _candidate()])

    assert first.to_dicts() == reversed_frame.to_dicts()
    assert first["frame_fingerprint"].n_unique() == 1
    with pytest.raises(ValueError, match="sampling_unit_id must be unique"):
        build_dynamic_pool_audit_frame([_candidate(), _candidate()])


def test_policy_cutpoints_are_versioned_and_strictly_increasing() -> None:
    policy = DynamicPoolAuditStrataPolicy(score_cutpoints=(0.2, 0.6))
    frame = build_dynamic_pool_audit_frame([_candidate()], policy=policy)

    assert frame["strata_policy_fingerprint"].item() == policy.fingerprint
    with pytest.raises(ValueError, match="strictly increasing"):
        DynamicPoolAuditStrataPolicy(score_cutpoints=(0.5, 0.5))


def test_empty_audit_frame_preserves_the_contract_schema() -> None:
    assert empty_dynamic_pool_audit_frame().schema == DYNAMIC_POOL_AUDIT_FRAME_SCHEMA
