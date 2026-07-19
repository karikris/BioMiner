"""Tests for escalation-bound additional Flickr audit work."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_audit_expansion import (
    DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA,
    DynamicPoolFlickrAuditCandidate,
    build_additional_flickr_audit_queue,
    validate_additional_flickr_audit_queue,
)
from biominer.evaluation.dynamic_pool_escalation import (
    DynamicPoolEscalationPolicy,
    define_pooling_escalations,
)
from biominer.evaluation.dynamic_pool_quality import (
    DynamicPoolQualityPolicy,
    report_family_pooling_quality,
    report_geographic_pooling_quality,
    report_overall_pooling_quality,
)
from test_dynamic_pool_quality import _observation


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _candidate(index: int) -> DynamicPoolFlickrAuditCandidate:
    return DynamicPoolFlickrAuditCandidate(
        item_id=f"item-{index}",
        source_record_id=f"flickr:{index}",
        source_image_sha256=_sha(index),
        audit_unit_fingerprint=_sha(index + 1),
        independence_component_id=f"component-{index}",
        family_key=f"family-{index % 2}",
        family_name=f"Family {index % 2}",
        genus_key=f"genus-{index % 4}",
        genus_name=f"Genus {index % 4}",
        species_key=f"species-{index % 8}",
        scientific_name=f"Species {index % 8}",
        route="adult_field",
        review_priority=0.4 + index * 0.05,
        global_local_disagreement=0.1 * index,
        local_support_available=True,
        out_of_distribution=False,
        calibrated_supported_probability=0.5 + index * 0.02,
        country_code="AU",
        admin1="NSW",
        bioregion="Sydney Basin",
        geographic_cluster_id=f"geo-{index % 2}",
    )


def _quality_policy() -> DynamicPoolQualityPolicy:
    return DynamicPoolQualityPolicy(
        minimum_group_items=2,
        minimum_group_components=2,
        minimum_group_effective_sample_size=2.0,
        minimum_metric_denominator_items=1,
        minimum_metric_denominator_components=1,
        minimum_metric_effective_sample_size=1.0,
    )


def _permissive_escalation() -> DynamicPoolEscalationPolicy:
    return DynamicPoolEscalationPolicy(
        minimum_precision_lower_bound=0.0,
        maximum_family_routing_error_rate=1.0,
        maximum_global_local_disagreement_rate=1.0,
        maximum_local_support_insufficiency_rate=1.0,
        maximum_reference_outlier_error_influence_rate=1.0,
        maximum_weighted_brier_score=1.0,
        maximum_weighted_ece=1.0,
        maximum_ood_false_positive_incidence=1.0,
    )


def test_insufficient_evidence_expansion_requires_sampling_design() -> None:
    quality = report_family_pooling_quality([_observation(0)])
    escalations = define_pooling_escalations([quality])

    projection = build_additional_flickr_audit_queue([_candidate(0)], escalations)

    assert projection.queue.schema == DYNAMIC_POOL_FLICKR_AUDIT_EXPANSION_SCHEMA
    row = projection.queue.row(0, named=True)
    assert row["queue_kinds"] == ["representative_audit_expansion_candidate"]
    assert row["sampling_design_status"] == (
        "required_before_representative_estimation"
    )
    assert row["sampling_purpose"] == "pending_probability_design"
    assert row["inclusion_probability"] is None
    assert row["sampling_weight"] is None
    assert row["representative_estimation_eligible"] is False


def test_diagnostic_followup_remains_targeted_and_nonrepresentative() -> None:
    observations = [
        replace(_observation(index), global_local_disagreed=True) for index in range(4)
    ]
    quality = report_overall_pooling_quality(observations, policy=_quality_policy())
    policy = replace(
        _permissive_escalation(),
        maximum_global_local_disagreement_rate=0.0,
    )
    escalations = define_pooling_escalations([quality], policy=policy)

    row = build_additional_flickr_audit_queue(
        [_candidate(0), _candidate(1)], escalations
    ).queue.row(0, named=True)

    assert row["queue_kinds"] == ["targeted_diagnostic_followup"]
    assert row["sampling_design_status"] == "not_applicable_targeted_followup"
    assert row["sampling_purpose"] == "targeted_failure_discovery"
    assert row["representative_estimation_eligible"] is False


def test_taxon_and_geography_escalations_bind_only_matching_candidates() -> None:
    family_quality = report_family_pooling_quality([_observation(0)])
    family_escalations = define_pooling_escalations([family_quality])
    family_projection = build_additional_flickr_audit_queue(
        [_candidate(0), _candidate(1)], family_escalations
    )

    assert family_projection.queued_candidate_count == 1
    assert family_projection.queue["item_id"].to_list() == ["item-0"]

    geographic_quality = report_geographic_pooling_quality(
        [
            replace(_observation(index), global_local_disagreed=True)
            for index in range(4)
        ],
        policy=_quality_policy(),
    )
    policy = replace(
        _permissive_escalation(),
        maximum_global_local_disagreement_rate=0.0,
    )
    geographic_escalations = define_pooling_escalations(
        [geographic_quality], policy=policy
    )
    nz_candidate = replace(
        _candidate(1),
        country_code="NZ",
        admin1="Auckland",
    )

    geographic_projection = build_additional_flickr_audit_queue(
        [_candidate(0), nz_candidate], geographic_escalations
    )
    au_row = geographic_projection.queue.filter(pl.col("item_id") == "item-0").row(
        0, named=True
    )
    assert "geography:country:AU" in au_row["matched_escalation_group_ids"]
    assert "geography:country:NZ" not in au_row["matched_escalation_group_ids"]


def test_followup_queue_has_zero_occurrence_or_release_authority() -> None:
    quality = report_family_pooling_quality([_observation(0)])
    escalations = define_pooling_escalations([quality])
    queue = build_additional_flickr_audit_queue([_candidate(0)], escalations).queue

    assert not queue["occurrence_claim_supported"].any()
    assert not queue["eligible_for_final_occurrence_dataset"].any()
    assert not queue["release_authorized"].any()
    assert queue["review_status"].unique().to_list() == ["pending"]
    assert queue["human_review_required"].all()


def test_audit_expansion_is_deterministic_and_tamper_evident() -> None:
    observations = [
        replace(_observation(index), global_local_disagreed=True) for index in range(4)
    ]
    quality = report_overall_pooling_quality(observations, policy=_quality_policy())
    escalations = define_pooling_escalations(
        [quality],
        policy=replace(
            _permissive_escalation(),
            maximum_global_local_disagreement_rate=0.0,
        ),
    )
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    first = build_additional_flickr_audit_queue(candidates, escalations)
    second = build_additional_flickr_audit_queue(
        list(reversed(candidates)), escalations
    )

    assert first.projection_fingerprint == second.projection_fingerprint
    assert first.queue.equals(second.queue)
    assert first.queue["priority_rank"].to_list() == [1, 2, 3]
    tampered = first.queue.with_columns(pl.lit(True).alias("release_authorized"))
    with pytest.raises(ValueError, match="evidence boundary"):
        validate_additional_flickr_audit_queue(tampered)


def test_audit_candidate_rejects_non_flickr_or_inconsistent_geography() -> None:
    with pytest.raises(ValueError, match="requires Flickr source"):
        replace(_candidate(0), source="gbif")
    with pytest.raises(ValueError, match="requires geographic cluster"):
        replace(_candidate(0), geographic_cluster_id=None)
