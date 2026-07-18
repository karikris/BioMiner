"""Tests for escalation-bound GBIF reference review targeting."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

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
from biominer.references.dynamic_pool_targeted_review import (
    DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA,
    DynamicPoolReferenceReviewCandidate,
    build_dynamic_pool_targeted_reference_review_queue,
    validate_dynamic_pool_targeted_reference_review_queue,
)
from test_dynamic_pool_quality import _observation


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _candidate(
    index: int, *, family_index: int | None = None
) -> DynamicPoolReferenceReviewCandidate:
    family = index % 2 if family_index is None else family_index
    return DynamicPoolReferenceReviewCandidate(
        reference_media_id=f"gbif-media-{index}",
        reference_observation_id=f"gbif-observation-{index}",
        source_dataset_key="gbif-dataset",
        source_media_sha256=_sha(index),
        reference_bank_version="bank-v1",
        admission_policy_fingerprint=_sha(13),
        embedding_fingerprint=_sha(14),
        family_key=f"family-{family}",
        family_name=f"Family {family}",
        genus_key=f"genus-{index % 4}",
        genus_name=f"Genus {index % 4}",
        species_key=f"species-{index % 8}",
        scientific_name=f"Species {index % 8}",
        route="adult_field",
        embedding_outlier_score=0.1 + index * 0.1,
        prototype_influence=0.2,
        repeated_error_involvement_count=index,
        route_domain_mismatch=False,
        local_scope_member=True,
        current_support_disposition="support_eligible",
        country_code="AU",
        admin1="NSW",
        bioregion="Sydney Basin",
        geographic_cluster_id=f"geo-{index % 2}",
        reference_quality_flags=("fixture",) if index == 1 else (),
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


def _reference_trigger_policy() -> DynamicPoolEscalationPolicy:
    return DynamicPoolEscalationPolicy(
        minimum_precision_lower_bound=0.0,
        maximum_family_routing_error_rate=0.0,
        maximum_global_local_disagreement_rate=1.0,
        maximum_local_support_insufficiency_rate=1.0,
        maximum_reference_outlier_error_influence_rate=1.0,
        maximum_weighted_brier_score=1.0,
        maximum_weighted_ece=1.0,
        maximum_ood_false_positive_incidence=1.0,
    )


def test_family_escalation_targets_only_matching_gbif_references() -> None:
    observations = [
        replace(_observation(index), family_routing_correct=index % 2 == 1)
        for index in range(4)
    ]
    quality = report_family_pooling_quality(observations, policy=_quality_policy())
    escalations = define_pooling_escalations(
        [quality],
        policy=_reference_trigger_policy(),
    )

    projection = build_dynamic_pool_targeted_reference_review_queue(
        [_candidate(0), _candidate(1)],
        escalations,
    )

    assert projection.queue.schema == DYNAMIC_POOL_TARGETED_REFERENCE_QUEUE_SCHEMA
    assert projection.source_candidate_count == 2
    assert projection.targeted_candidate_count == 1
    row = projection.queue.row(0, named=True)
    assert row["reference_media_id"] == "gbif-media-0"
    assert row["matched_escalation_group_ids"] == ["family:family-0"]
    assert "family_misrouting_above_objective" in row["trigger_reasons"]
    assert row["source"] == "gbif"


def test_targeted_reference_queue_never_automates_bad_reference_claim() -> None:
    observations = [
        replace(_observation(index), family_routing_correct=False) for index in (0, 2)
    ]
    escalations = define_pooling_escalations(
        [report_family_pooling_quality(observations, policy=_quality_policy())],
        policy=_reference_trigger_policy(),
    )

    queue = build_dynamic_pool_targeted_reference_review_queue(
        [_candidate(0)],
        escalations,
    ).queue
    row = queue.row(0, named=True)

    assert row["reference_identity_conclusion"] == "not_assessed"
    assert row["review_status"] == "pending"
    assert row["human_review_required"] is True
    assert row["automatic_reference_exclusion"] is False
    assert (
        row["support_disposition_after_targeting"] == row["current_support_disposition"]
    )
    assert row["authorizes_occurrence_release"] is False


def test_unbound_overall_reference_trigger_remains_explicitly_unmatched() -> None:
    observations = [
        replace(_observation(index), family_routing_correct=False) for index in range(4)
    ]
    escalation = define_pooling_escalations(
        [report_overall_pooling_quality(observations, policy=_quality_policy())],
        policy=_reference_trigger_policy(),
    )

    projection = build_dynamic_pool_targeted_reference_review_queue(
        [_candidate(0), _candidate(1)],
        escalation,
    )

    assert projection.queue.is_empty()
    assert projection.matched_escalation_group_ids == ()
    assert projection.unmatched_escalation_group_ids == ("overall",)


def test_geography_trigger_binds_only_matching_reference_geography() -> None:
    observations = [
        replace(_observation(index), local_support_available=False)
        for index in range(4)
    ]
    quality = report_geographic_pooling_quality(observations, policy=_quality_policy())
    policy = replace(
        _reference_trigger_policy(),
        maximum_family_routing_error_rate=1.0,
        maximum_local_support_insufficiency_rate=0.0,
    )
    escalations = define_pooling_escalations([quality], policy=policy)

    projection = build_dynamic_pool_targeted_reference_review_queue(
        [_candidate(0), replace(_candidate(1), country_code="NZ", admin1="Auckland")],
        escalations,
    )

    assert projection.targeted_candidate_count == 2
    first = projection.queue.filter(pl.col("reference_media_id") == "gbif-media-0").row(
        0, named=True
    )
    assert "geography:country:AU" in first["matched_escalation_group_ids"]
    assert "geography:country:NZ" not in first["matched_escalation_group_ids"]


def test_priority_is_transparent_deterministic_and_tamper_evident() -> None:
    observations = [
        replace(_observation(index), family_routing_correct=False) for index in (0, 2)
    ]
    escalations = define_pooling_escalations(
        [report_family_pooling_quality(observations, policy=_quality_policy())],
        policy=_reference_trigger_policy(),
    )
    candidates = [_candidate(0), _candidate(2, family_index=0)]

    first = build_dynamic_pool_targeted_reference_review_queue(candidates, escalations)
    second = build_dynamic_pool_targeted_reference_review_queue(
        list(reversed(candidates)), escalations
    )

    assert first.projection_fingerprint == second.projection_fingerprint
    assert first.queue.equals(second.queue)
    assert first.queue["priority_rank"].to_list() == [1, 2]
    assert first.queue["priority_score_semantics"].unique().to_list() == [
        "transparent_review_heuristic_not_probability"
    ]
    tampered = first.queue.with_columns(
        pl.lit(True).alias("automatic_reference_exclusion")
    )
    with pytest.raises(ValueError, match="authority contract"):
        validate_dynamic_pool_targeted_reference_review_queue(tampered)


def test_reference_candidate_rejects_non_gbif_or_inconsistent_geography() -> None:
    with pytest.raises(ValueError, match="must have GBIF source"):
        replace(_candidate(0), source="flickr")
    with pytest.raises(ValueError, match="requires geographic_cluster_id"):
        replace(_candidate(0), geographic_cluster_id=None)
