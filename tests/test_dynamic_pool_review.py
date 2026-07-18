"""Tests for dynamic-pool Flickr audit and review contracts."""

from __future__ import annotations

import pytest

from biominer.evaluation.dynamic_pool_review import (
    DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
    DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA,
    DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA,
    RAW_SCORE_SEMANTICS,
    DynamicPoolAuditStrataPolicy,
    ProbabilityAuditSamplingPolicy,
    build_dynamic_pool_audit_frame,
    build_probability_audit_sample,
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


def _probability_frame():
    candidates: list[dict[str, object]] = []
    for index in range(6):
        candidates.append(
            _candidate(
                sampling_unit_id=f"review-unit-{index}",
                source_record_hash=_sha(str(index + 3)),
                flickr_photo_id=f"photo-{index}",
                organism_unit_id=f"organism-{index}",
                primary_query_tier="T2" if index < 4 else "T3",
                owner_group_id=f"owner-{index // 2}",
                duplicate_group_id=(
                    "duplicate-shared" if index < 2 else f"duplicate-{index}"
                ),
                observation_group_id=f"observation-{index}",
            )
        )
    return build_dynamic_pool_audit_frame(candidates)


def test_probability_sample_publishes_exact_all_unit_inclusion_design() -> None:
    selection = build_probability_audit_sample(
        _probability_frame(),
        policy=ProbabilityAuditSamplingPolicy(review_budget=3, random_seed=7),
    )

    assert selection.population_count == 5
    assert selection.selected_count == 3
    assert selection.register.schema == DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA
    assert selection.sample.schema == DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA
    assert selection.register["sampling_register_fingerprint"].n_unique() == 1
    assert selection.register.filter(~selection.register["selected"]).height == 2
    for row in selection.register.to_dicts():
        assert row["inclusion_probability"] == pytest.approx(
            row["stratum_sample_count"] / row["stratum_population_count"]
        )
        assert row["sampling_weight"] == pytest.approx(
            1.0 / row["inclusion_probability"]
        )
    assert selection.sample["representative_estimation_eligible"].all()


def test_probability_population_collapses_duplicate_observation_components() -> None:
    selection = build_probability_audit_sample(
        _probability_frame(),
        policy=ProbabilityAuditSamplingPolicy(review_budget=10),
    )
    grouped = selection.register.filter(
        selection.register["sampling_population_member_count"] == 2
    ).row(0, named=True)

    assert selection.population_count == 5
    assert grouped["sampling_population_member_unit_ids"] == [
        "review-unit-0",
        "review-unit-1",
    ]
    assert grouped["sampling_population_owner_group_ids"] == ["owner-0"]
    assert grouped["inclusion_probability"] == 1.0
    assert grouped["sampling_weight"] == 1.0


def test_probability_sample_is_input_order_independent() -> None:
    frame = _probability_frame()
    policy = ProbabilityAuditSamplingPolicy(review_budget=3, random_seed=91)
    first = build_probability_audit_sample(frame, policy=policy)
    reversed_frame = frame.reverse()
    second = build_probability_audit_sample(reversed_frame, policy=policy)

    assert first.register.to_dicts() == second.register.to_dicts()
    assert first.sample.to_dicts() == second.sample.to_dicts()


def test_probability_budget_must_cover_configured_stratum_minimum() -> None:
    with pytest.raises(ValueError, match="represent every nonempty"):
        build_probability_audit_sample(
            _probability_frame(),
            policy=ProbabilityAuditSamplingPolicy(review_budget=1),
        )
