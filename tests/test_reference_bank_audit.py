from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.evaluation.reference_bank_audit import (
    AUDIT_DIMENSIONS,
    ReferenceBankQualityPolicy,
    empty_reference_bank_quality_audit,
    empty_reference_bank_quality_summary,
    write_reference_bank_audit_contract,
)
from biominer.evaluation.reference_bank_performance import (
    HUMAN_REVIEWED_FLICKR_BASIS,
    measure_reference_bank_performance,
)


def test_reference_bank_audit_contract_publishes_four_typed_artifacts(tmp_path) -> None:
    publication = write_reference_bank_audit_contract(
        tmp_path,
        audit=empty_reference_bank_quality_audit(),
        summary=empty_reference_bank_quality_summary(),
    )

    assert publication.audit_path.name == "reference_bank_quality_audit.parquet"
    assert publication.summary_path.name == "reference_bank_quality_summary.parquet"
    assert publication.policy_path.name == "reference_bank_quality_policy.json"
    assert publication.report_path.name == "reference_bank_quality_report.md"
    assert pl.read_parquet(publication.audit_path).schema == (
        empty_reference_bank_quality_audit().schema
    )
    assert set(AUDIT_DIMENSIONS) <= set(pl.read_parquet(publication.summary_path).columns)
    policy = json.loads(publication.policy_path.read_text(encoding="utf-8"))
    assert policy["require_sampling_weights_for_targeted_queues"] is True
    assert "Targeted queues require sampling weights" in publication.report_path.read_text(
        encoding="utf-8"
    )


def test_reference_bank_audit_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        write_reference_bank_audit_contract(
            ".",
            audit=pl.DataFrame({"audit_record_id": []}, schema={"audit_record_id": pl.String}),
            summary=empty_reference_bank_quality_summary(),
        )


def test_reference_bank_quality_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="positive"):
        ReferenceBankQualityPolicy(minimum_group_sample_size=0)
    with pytest.raises(ValueError, match="confidence_level"):
        ReferenceBankQualityPolicy(confidence_level=1.0)


def _performance_audit(
    *,
    campaign: str = "representative_quality_audit",
    sampling_weights: list[float | None] | None = None,
    probabilities: list[float | None] | None = None,
    calibrator_validated: bool = False,
) -> pl.DataFrame:
    truth = [True, True, False, False]
    predicted = [True, False, True, False]
    margins = [0.8, -0.1, 0.4, -0.7]
    weights = sampling_weights or [1.0, 1.0, 1.0, 1.0]
    probability_values = probabilities or [None, None, None, None]
    rows = []
    for index in range(4):
        rows.append(
            {
                "audit_record_id": f"review:{index}",
                "target_species": "Papilio demoleus",
                "competitor_species": "Papilio polytes",
                "region": "AU-QLD",
                "route": "adult_field",
                "life_stage": "adult",
                "visual_domain": "live_field",
                "source_dataset": "flickr",
                "admission_basis": "gbif_provider_asserted",
                "verification_basis": HUMAN_REVIEWED_FLICKR_BASIS,
                "sampling_campaign": campaign,
                "sampling_stratum": "audit",
                "human_target_supported": truth[index],
                "predicted_target": predicted[index],
                "prediction_abstained": False,
                "predicted_competitor_species": (
                    "Papilio polytes" if index == 1 else None
                ),
                "provisional_margin": margins[index],
                "calibrated_probability": probability_values[index],
                "calibrator_validated": calibrator_validated,
                "inclusion_probability": 1.0,
                "sampling_weight": weights[index],
            }
        )
    return pl.DataFrame(rows, schema=empty_reference_bank_quality_audit().schema)


def test_species_performance_uses_weighted_human_reviewed_labels() -> None:
    result = measure_reference_bank_performance(
        _performance_audit(),
        policy=ReferenceBankQualityPolicy(minimum_group_sample_size=4),
    )
    row = result.row(0, named=True)

    assert row["metric_status"] == "complete"
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["false_positive_rate"] == pytest.approx(0.5)
    assert row["false_negative_rate"] == pytest.approx(0.5)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["abstention_rate"] == pytest.approx(0.0)
    assert row["competitor_confusion_rate"] == pytest.approx(0.5)
    assert 0.0 <= row["pr_auc"] <= 1.0
    assert 0.0 <= row["precision_ci_lower"] <= row["precision_ci_upper"] <= 1.0
    assert row["probability_available"] is False
    assert row["brier_score"] is None
    assert row["margin_median"] is not None


def test_calibration_metrics_require_complete_attested_probabilities() -> None:
    result = measure_reference_bank_performance(
        _performance_audit(
            probabilities=[0.8, 0.4, 0.7, 0.1],
            calibrator_validated=True,
        ),
        policy=ReferenceBankQualityPolicy(minimum_group_sample_size=4),
    )
    row = result.row(0, named=True)

    assert row["probability_available"] is True
    assert row["brier_score"] is not None
    assert row["expected_calibration_error"] is not None
    assert row["margin_median"] is None


def test_targeted_queue_without_weights_returns_unavailable_metrics() -> None:
    result = measure_reference_bank_performance(
        _performance_audit(
            campaign="failure_discovery",
            sampling_weights=[None, None, None, None],
        ),
        policy=ReferenceBankQualityPolicy(minimum_group_sample_size=4),
    )
    row = result.row(0, named=True)

    assert row["metric_status"] == "unavailable_missing_sampling_weights"
    assert row["quality_approval_state"] == "unavailable"
    assert row["precision"] is None
    assert row["weights_applied"] is False


def test_underpowered_group_returns_explicit_insufficient_sample() -> None:
    result = measure_reference_bank_performance(
        _performance_audit(),
        policy=ReferenceBankQualityPolicy(minimum_group_sample_size=5),
    )

    assert result["metric_status"].item() == "insufficient_sample"
    assert result["precision"].item() is None


def test_species_performance_rejects_nonhuman_evaluation_labels() -> None:
    audit = _performance_audit().with_columns(
        pl.lit("gbif_provider_asserted").alias("verification_basis")
    )

    with pytest.raises(ValueError, match="human-reviewed Flickr"):
        measure_reference_bank_performance(audit)
