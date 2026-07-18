"""Tests for grouped and weighted hierarchical dynamic-pool quality audits."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_quality import (
    GROUPED_WEIGHTED_INTERVAL_METHOD,
    QUALITY_REPORT_SCHEMA,
    TARGETED_FAILURE_PURPOSE,
    DynamicPoolQualityObservation,
    DynamicPoolQualityPolicy,
    report_overall_pooling_quality,
    validate_dynamic_pool_quality_report,
)


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _observation(
    index: int,
    *,
    component: str | None = None,
    weight: float | None = 1.0,
    human_supported: bool = True,
    selected: bool = True,
    abstained: bool = False,
    probability: float | None = 0.9,
    targeted: bool = False,
) -> DynamicPoolQualityObservation:
    return DynamicPoolQualityObservation(
        item_id=f"item-{index}",
        source_record_id=f"flickr:{index}",
        source_image_sha256=_sha(index),
        independence_component_id=component or f"component-{index}",
        family_key=f"family-{index % 2}",
        family_name=f"Family {index % 2}",
        genus_key=f"genus-{index % 3}",
        genus_name=f"Genus {index % 3}",
        species_key=f"species-{index % 4}",
        scientific_name=f"Species {index % 4}",
        sampling_purpose=(
            TARGETED_FAILURE_PURPOSE if targeted else "representative_audit"
        ),
        representative_estimation_eligible=not targeted,
        sampling_weight=None if targeted else weight,
        human_supported=human_supported,
        screening_selected=selected,
        model_abstained=abstained,
        family_routing_correct=True,
        global_local_disagreed=index % 3 == 0,
        local_support_available=index % 4 != 0,
        reference_outlier_influenced_error=False,
        out_of_distribution=False,
        calibrated_supported_probability=probability,
        country_code="AU",
        admin1="NSW",
        bioregion="Sydney Basin",
        geographic_cluster_id=f"geo-{index % 2}",
    )


def _permissive_policy() -> DynamicPoolQualityPolicy:
    return DynamicPoolQualityPolicy(
        minimum_group_items=2,
        minimum_group_components=2,
        minimum_group_effective_sample_size=2.0,
        minimum_metric_denominator_items=1,
        minimum_metric_denominator_components=1,
        minimum_metric_effective_sample_size=1.0,
    )


def _metric(table: pl.DataFrame, name: str) -> dict[str, object]:
    return table.filter(pl.col("metric_name") == name).row(0, named=True)


def test_overall_quality_reports_weighted_metrics_and_grouped_intervals() -> None:
    observations = [
        _observation(0, weight=2.0),
        _observation(1, weight=1.0),
        _observation(2, weight=1.0, human_supported=False, selected=True),
        _observation(3, weight=2.0, human_supported=False, selected=False),
    ]

    report = report_overall_pooling_quality(
        observations,
        policy=_permissive_policy(),
    )

    assert report.schema == QUALITY_REPORT_SCHEMA
    assert report["group_status"].unique().to_list() == ["complete"]
    precision = _metric(report, "selection_precision")
    recall = _metric(report, "selection_recall")
    accuracy = _metric(report, "decision_accuracy")
    assert precision["estimate"] == pytest.approx(0.75)
    assert recall["estimate"] == pytest.approx(1.0)
    assert accuracy["estimate"] == pytest.approx(5.0 / 6.0)
    assert precision["confidence_interval_method"] == GROUPED_WEIGHTED_INTERVAL_METHOD
    assert precision["confidence_interval_lower"] < precision["estimate"]
    assert precision["confidence_interval_upper"] > precision["estimate"]
    assert not report["authorizes_occurrence_release"].any()


def test_component_grouping_caps_effective_sample_size() -> None:
    observations = [
        _observation(index, component="shared-component") for index in range(6)
    ]
    policy = replace(
        _permissive_policy(),
        minimum_group_components=2,
        minimum_group_effective_sample_size=2.0,
    )

    report = report_overall_pooling_quality(observations, policy=policy)

    row = _metric(report, "model_coverage")
    assert row["independence_component_count"] == 1
    assert row["group_effective_sample_size"] == pytest.approx(1.0)
    assert row["group_status"] == "insufficient_sample"
    assert row["estimate"] is None
    assert "group_components_below_minimum" in row["metric_insufficiency_reasons"]
    assert (
        "group_effective_sample_size_below_minimum"
        in row["metric_insufficiency_reasons"]
    )


def test_default_policy_emits_explicit_insufficient_sample_states() -> None:
    report = report_overall_pooling_quality([_observation(0)])

    assert report["group_status"].unique().to_list() == ["insufficient_sample"]
    assert report["estimate"].null_count() == report.height
    reasons = _metric(report, "selection_precision")["group_insufficiency_reasons"]
    assert "group_items_below_minimum" in reasons
    assert "group_components_below_minimum" in reasons
    assert "group_effective_sample_size_below_minimum" in reasons


def test_metric_denominator_floor_is_separate_from_group_floor() -> None:
    observations = [
        _observation(0, selected=True),
        _observation(1, selected=False, human_supported=False),
    ]
    policy = replace(
        _permissive_policy(),
        minimum_metric_denominator_items=2,
        minimum_metric_denominator_components=2,
        minimum_metric_effective_sample_size=2.0,
    )

    report = report_overall_pooling_quality(observations, policy=policy)

    precision = _metric(report, "selection_precision")
    coverage = _metric(report, "model_coverage")
    assert precision["group_status"] == "complete"
    assert precision["metric_status"] == "insufficient_metric_sample"
    assert precision["estimate"] is None
    assert coverage["metric_status"] == "complete"
    assert coverage["estimate"] == pytest.approx(1.0)


def test_targeted_failures_are_visible_but_excluded_from_estimation() -> None:
    report = report_overall_pooling_quality(
        [_observation(0), _observation(1), _observation(2, targeted=True)],
        policy=_permissive_policy(),
    )

    row = _metric(report, "model_coverage")
    assert row["source_item_count"] == 3
    assert row["evaluated_item_count"] == 2
    assert row["excluded_targeted_item_count"] == 1
    assert row["denominator_item_count"] == 2


def test_calibration_diagnostics_use_only_available_probabilities() -> None:
    report = report_overall_pooling_quality(
        [
            _observation(0, probability=0.8),
            _observation(1, probability=0.2, human_supported=False, selected=False),
            _observation(2, probability=None),
        ],
        policy=_permissive_policy(),
    )

    brier = _metric(report, "weighted_brier_score")
    ece = _metric(report, "weighted_ece")
    availability = _metric(report, "calibrated_probability_availability")
    assert brier["denominator_item_count"] == 2
    assert brier["estimate"] == pytest.approx(0.04)
    assert ece["estimate"] == pytest.approx(0.2)
    assert availability["estimate"] == pytest.approx(2.0 / 3.0)


def test_quality_report_is_deterministic_and_tamper_evident() -> None:
    observations = [_observation(index) for index in range(4)]
    first = report_overall_pooling_quality(observations, policy=_permissive_policy())
    second = report_overall_pooling_quality(
        list(reversed(observations)),
        policy=_permissive_policy(),
    )

    assert first.equals(second)
    authorized = first.with_columns(pl.lit(True).alias("authorizes_occurrence_release"))
    with pytest.raises(ValueError, match="authority contract"):
        validate_dynamic_pool_quality_report(authorized)


def test_quality_observation_rejects_invalid_review_design_or_claims() -> None:
    with pytest.raises(ValueError, match="requires sampling_weight"):
        _observation(0, weight=None)
    with pytest.raises(ValueError, match="abstained model"):
        _observation(0, selected=True, abstained=True)
    with pytest.raises(ValueError, match="requires an observed decision error"):
        replace(_observation(0), reference_outlier_influenced_error=True)
