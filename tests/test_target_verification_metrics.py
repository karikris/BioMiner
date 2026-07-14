from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.target_metrics import (
    STRATIFICATION_FIELDS,
    TARGET_MARGIN_DISTRIBUTION_SCHEMA,
    TARGET_MARGIN_DISTRIBUTION_FILE,
    TARGET_VERIFICATION_EVALUATION_SCHEMA,
    TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION,
    TARGET_VERIFICATION_METRIC_SCHEMA,
    TARGET_VERIFICATION_METRICS_FILE,
    TARGET_VERIFICATION_REPORT_FILE,
    TARGET_VERIFICATION_REPORT_MARKDOWN_FILE,
    TargetVerificationMetricsConfig,
    compute_target_verification_metrics,
    empty_target_verification_evaluation_frame,
    evaluate_target_verification,
    publish_target_verification_metric_report,
    target_verification_evaluation_frame,
    validate_target_verification_evaluation_frame,
    validate_target_verification_metric_report,
)
from test_evaluation_holdouts import _frozen_holdout_pair, _leakage_register


def test_weighted_target_metrics_penalize_abstention_and_measure_selective_risk() -> (
    None
):
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(_natural_rows())
    )

    assert report.metrics.schema == TARGET_VERIFICATION_METRIC_SCHEMA
    assert report.margin_distribution.schema == TARGET_MARGIN_DISTRIBUTION_SCHEMA
    assert _metric(report.metrics, "precision") == pytest.approx(2.0 / 5.0)
    assert _metric(report.metrics, "recall") == pytest.approx(2.0 / 4.0)
    assert _metric(report.metrics, "f1") == pytest.approx(4.0 / 9.0)
    assert _metric(report.metrics, "specificity") == pytest.approx(5.0 / 8.0)
    assert _metric(report.metrics, "false_positive_rate") == pytest.approx(3.0 / 8.0)
    assert _metric(report.metrics, "false_negative_rate") == pytest.approx(0.5)
    assert _metric(report.metrics, "coverage") == pytest.approx(10.0 / 12.0)
    assert _metric(report.metrics, "abstention_rate") == pytest.approx(2.0 / 12.0)
    assert _metric(report.metrics, "selective_risk") == pytest.approx(4.0 / 10.0)
    assert _metric(report.metrics, "brier_score") == pytest.approx(2.56 / 12.0)
    assert 0.0 <= _metric(report.metrics, "pr_auc") <= 1.0
    assert 0.0 <= _metric(report.metrics, "roc_auc") <= 1.0
    assert _metric(report.metrics, "recall_at_precision_9000bp") == pytest.approx(0.5)
    assert _metric(report.metrics, "recall_at_precision_9500bp") == pytest.approx(0.5)
    assert _metric(report.metrics, "recall_at_precision_9900bp") == pytest.approx(0.5)
    assert _metric(report.metrics, "ood_false_positive_rate") == pytest.approx(
        3.0 / 7.0
    )
    assert _metric(report.metrics, "detector_gate_recall") == pytest.approx(0.75)


def test_reports_every_required_stratum_without_pooling_frozen_sets() -> None:
    rows = _natural_rows()
    rows.append(
        _row(
            "balanced-1",
            evaluation_set="balanced_challenge",
            target_present=True,
            probability=0.95,
            decision="target_confirmed",
            sampling_weight=1.0,
            country_code="IN",
        )
    )
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(rows)
    )

    overall = report.metrics.filter(pl.col("scope") == "overall")
    assert set(overall["evaluation_set"]) == {
        "balanced_challenge",
        "natural_stream",
    }
    dimensions = set(
        report.metrics.filter(
            (pl.col("evaluation_set") == "natural_stream")
            & (pl.col("scope") == "stratum")
        )["stratum_dimension"]
    )
    assert dimensions == {dimension for dimension, _ in STRATIFICATION_FIELDS}
    balanced_precision = _metric(
        report.metrics,
        "precision",
        evaluation_set="balanced_challenge",
    )
    assert balanced_precision == 1.0


def test_single_class_strata_are_explicitly_undefined() -> None:
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(_natural_rows())
    )

    row = _metric_row(
        report.metrics,
        "roc_auc",
        dimension="country",
        value="positive-country",
    )
    assert row["metric_value"] is None
    assert row["undefined_reason"] == "single_class"
    pr_row = _metric_row(
        report.metrics,
        "pr_auc",
        dimension="country",
        value="negative-country",
    )
    assert pr_row["metric_value"] is None
    assert pr_row["undefined_reason"] == "single_class"


def test_incomplete_probability_coverage_does_not_silently_subset_metrics() -> None:
    rows = _natural_rows()
    rows[0]["calibrated_target_probability"] = None
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(rows)
    )

    assert _metric(report.metrics, "probability_coverage") == pytest.approx(5.0 / 6.0)
    assert _metric(report.metrics, "precision") == pytest.approx(2.0 / 5.0)
    for metric in (
        "pr_auc",
        "roc_auc",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "recall_at_precision_9000bp",
    ):
        row = _metric_row(report.metrics, metric)
        assert row["metric_value"] is None
        assert row["undefined_reason"] == "incomplete_probability_coverage"


def test_margin_distribution_and_diagnostic_recalls_are_weighted() -> None:
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(_natural_rows())
    )

    distribution = report.margin_distribution.filter(
        (pl.col("evaluation_set") == "natural_stream")
        & (pl.col("scope") == "overall")
        & (pl.col("population") == "all")
    ).row(0, named=True)
    assert distribution["item_count"] == 6
    assert distribution["weighted_item_count"] == 12.0
    assert distribution["margin_mean"] == pytest.approx(-0.8 / 12.0)
    assert distribution["margin_median"] == pytest.approx(-0.1)
    assert _metric(report.metrics, "family_recall_at_1") == pytest.approx(0.5)
    assert _metric(report.metrics, "genus_recall_at_3") == pytest.approx(11.0 / 12.0)
    assert _metric(report.metrics, "species_recall_at_5") == pytest.approx(11.0 / 12.0)
    assert _metric(
        report.metrics,
        "old_classifier_target_pruning_rate",
    ) == pytest.approx(0.5)


def test_evaluation_and_report_are_deterministic() -> None:
    rows = _natural_rows()
    config = TargetVerificationMetricsConfig(ece_bin_count=7)

    first = compute_target_verification_metrics(
        target_verification_evaluation_frame(rows),
        config,
    )
    second = compute_target_verification_metrics(
        target_verification_evaluation_frame(list(reversed(rows))),
        config,
    )

    assert_frame_equal(first.metrics, second.metrics)
    assert_frame_equal(first.margin_distribution, second.margin_distribution)
    assert first.input_fingerprint == second.input_fingerprint
    assert first.report_fingerprint == second.report_fingerprint
    assert first.report_fingerprint.startswith("sha256:")


def test_metric_report_publication_is_immutable_and_audited(tmp_path: Path) -> None:
    report = compute_target_verification_metrics(
        target_verification_evaluation_frame(_natural_rows())
    )
    validate_target_verification_metric_report(report)

    publication = publish_target_verification_metric_report(
        report,
        tmp_path / "published",
        run_id="target-metric-test",
    )

    assert publication.metrics_path.name == TARGET_VERIFICATION_METRICS_FILE
    assert publication.margin_distribution_path.name == TARGET_MARGIN_DISTRIBUTION_FILE
    assert publication.report_json_path.name == TARGET_VERIFICATION_REPORT_FILE
    assert (
        publication.report_markdown_path.name
        == TARGET_VERIFICATION_REPORT_MARKDOWN_FILE
    )
    payload = json.loads(publication.report_json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["report_fingerprint"] == report.report_fingerprint
    assert payload["artifacts"]["metrics"]["sha256"].startswith("sha256:")
    with pytest.raises(FileExistsError):
        publish_target_verification_metric_report(report, publication.output_dir)

    tampered = replace(report, report_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="report_fingerprint"):
        validate_target_verification_metric_report(tampered)


def test_official_evaluation_revalidates_leakage_and_frozen_item_coverage() -> None:
    challenge, natural = _frozen_holdout_pair()
    leakage_register = _leakage_register(challenge, natural)
    frame = target_verification_evaluation_frame(
        _rows_for_frozen_holdouts(challenge, natural)
    )

    report = evaluate_target_verification(
        frame,
        challenge,
        natural,
        leakage_register,
    )

    assert set(report.metrics["evaluation_set"]) == {
        "balanced_challenge",
        "natural_stream",
    }
    incomplete = frame.filter(
        pl.col("evaluation_item_id") != frame["evaluation_item_id"][0]
    )
    with pytest.raises(ValueError, match="does not cover the frozen holdouts"):
        evaluate_target_verification(
            incomplete,
            challenge,
            natural,
            leakage_register,
        )


def test_input_validation_is_fail_closed() -> None:
    assert empty_target_verification_evaluation_frame().schema == (
        TARGET_VERIFICATION_EVALUATION_SCHEMA
    )
    base = target_verification_evaluation_frame(_natural_rows())

    missing_label = base.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-1")
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("target_present"))
        .alias("target_present")
    )
    with pytest.raises(ValueError, match="target_present"):
        validate_target_verification_evaluation_frame(missing_label)

    confirmed_abstention = base.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-1")
        .then(pl.lit(True))
        .otherwise(pl.col("abstained"))
        .alias("abstained")
    )
    with pytest.raises(ValueError, match="target_confirmed"):
        validate_target_verification_evaluation_frame(confirmed_abstention)

    bad_geo = base.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-1")
        .then(pl.lit(True))
        .otherwise(pl.col("no_geo"))
        .alias("no_geo")
    )
    with pytest.raises(ValueError, match="no_geo"):
        validate_target_verification_evaluation_frame(bad_geo)

    bad_rank = base.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-1")
        .then(pl.lit(False))
        .otherwise(pl.col("family_evaluable"))
        .alias("family_evaluable")
    )
    with pytest.raises(ValueError, match="true_family_rank"):
        validate_target_verification_evaluation_frame(bad_rank)

    unexpected = dict(_natural_rows()[0])
    unexpected["species_top1_score"] = 0.99
    with pytest.raises(ValueError, match="unexpected=.*species_top1_score"):
        target_verification_evaluation_frame([unexpected])


def _natural_rows() -> list[dict[str, object]]:
    definitions = (
        (
            "item-1",
            True,
            0.9,
            "target_confirmed",
            False,
            2.0,
            0.5,
            False,
            True,
            True,
            1,
            1,
            1,
            False,
        ),
        (
            "item-2",
            True,
            0.7,
            "abstain",
            True,
            1.0,
            0.2,
            False,
            True,
            False,
            2,
            2,
            4,
            True,
        ),
        (
            "item-3",
            False,
            0.8,
            "target_confirmed",
            False,
            3.0,
            0.1,
            True,
            False,
            None,
            None,
            3,
            3,
            None,
        ),
        (
            "item-4",
            False,
            0.2,
            "other_butterfly",
            False,
            4.0,
            -0.4,
            True,
            False,
            None,
            1,
            1,
            1,
            None,
        ),
        (
            "item-5",
            True,
            0.4,
            "other_butterfly",
            False,
            1.0,
            -0.1,
            False,
            True,
            True,
            4,
            2,
            2,
            True,
        ),
        (
            "item-6",
            False,
            0.1,
            "abstain",
            True,
            1.0,
            -0.6,
            False,
            False,
            None,
            None,
            8,
            8,
            None,
        ),
    )
    return [
        _row(
            item_id,
            target_present=target,
            probability=probability,
            decision=decision,
            abstained=abstained,
            sampling_weight=weight,
            margin=margin,
            ground_truth_ood=ood,
            detector_gate_required=gate_required,
            detector_gate_passed=gate_passed,
            family_rank=family_rank,
            genus_rank=genus_rank,
            species_rank=species_rank,
            old_classifier_target_pruned=old_pruned,
            country_code=("positive-country" if target else "negative-country"),
        )
        for (
            item_id,
            target,
            probability,
            decision,
            abstained,
            weight,
            margin,
            ood,
            gate_required,
            gate_passed,
            family_rank,
            genus_rank,
            species_rank,
            old_pruned,
        ) in definitions
    ]


def _rows_for_frozen_holdouts(
    challenge: pl.DataFrame,
    natural: pl.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation_set, holdout in (
        ("balanced_challenge", challenge),
        ("natural_stream", natural),
    ):
        for item in holdout.iter_rows(named=True):
            target_present = bool(item["target_present"])
            family_evaluable = item["family_key"] is not None
            genus_evaluable = item["genus_key"] is not None
            species_evaluable = item["accepted_taxon_key"] is not None
            rows.append(
                {
                    "schema_version": TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION,
                    "evaluation_item_id": item["evaluation_item_id"],
                    "evaluation_set": evaluation_set,
                    "sampling_weight": (
                        1.0
                        if evaluation_set == "balanced_challenge"
                        else item["sampling_weight"]
                    ),
                    "target_present": target_present,
                    "calibrated_target_probability": (0.9 if target_present else 0.1),
                    "classification_decision": (
                        "target_confirmed" if target_present else "other_butterfly"
                    ),
                    "abstained": False,
                    "target_competitor_margin": (0.5 if target_present else -0.5),
                    "ground_truth_out_of_distribution": (
                        item["evaluation_class"] == "artifacts"
                    ),
                    "detector_gate_required": target_present,
                    "detector_gate_passed": True if target_present else None,
                    "family_evaluable": family_evaluable,
                    "true_family_rank": 1 if family_evaluable else None,
                    "genus_evaluable": genus_evaluable,
                    "true_genus_rank": 1 if genus_evaluable else None,
                    "species_evaluable": species_evaluable,
                    "true_species_rank": 1 if species_evaluable else None,
                    "old_classifier_target_pruned": (False if target_present else None),
                    "geo_cluster_id": item["geo_cluster_id"],
                    "country_code": "unknown",
                    "no_geo": item["geo_cluster_id"] == "no_geo",
                    "route": item["route"] or "not_applicable",
                    "life_stage": item["life_stage"],
                    "visual_domain": item["visual_domain"],
                    "subject_area_band": "not_measured",
                    "source_query_tier": item["source_query_tier"],
                    "source_query_term": item["source_query_term"],
                    "source_provider": str(item["source"]).casefold(),
                    "visual_input_kind": "whole_image_reference_ensemble",
                }
            )
    return rows


def _row(
    item_id: str,
    *,
    evaluation_set: str = "natural_stream",
    target_present: bool,
    probability: float,
    decision: str,
    abstained: bool = False,
    sampling_weight: float = 1.0,
    margin: float = 0.0,
    ground_truth_ood: bool = False,
    detector_gate_required: bool = True,
    detector_gate_passed: bool | None = True,
    family_rank: int | None = 1,
    genus_rank: int | None = 1,
    species_rank: int | None = 1,
    old_classifier_target_pruned: bool | None = False,
    country_code: str = "AU",
) -> dict[str, object]:
    return {
        "schema_version": TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION,
        "evaluation_item_id": item_id,
        "evaluation_set": evaluation_set,
        "sampling_weight": sampling_weight,
        "target_present": target_present,
        "calibrated_target_probability": probability,
        "classification_decision": decision,
        "abstained": abstained,
        "target_competitor_margin": margin,
        "ground_truth_out_of_distribution": ground_truth_ood,
        "detector_gate_required": detector_gate_required,
        "detector_gate_passed": detector_gate_passed,
        "family_evaluable": True,
        "true_family_rank": family_rank,
        "genus_evaluable": True,
        "true_genus_rank": genus_rank,
        "species_evaluable": True,
        "true_species_rank": species_rank,
        "old_classifier_target_pruned": old_classifier_target_pruned,
        "geo_cluster_id": "geo:fixture",
        "country_code": country_code,
        "no_geo": False,
        "route": "adult_field",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "subject_area_band": "medium",
        "source_query_tier": "T1",
        "source_query_term": "Papilio demoleus",
        "source_provider": "flickr",
        "visual_input_kind": "whole_image_reference_ensemble",
    }


def _metric(
    frame: pl.DataFrame,
    metric_name: str,
    *,
    evaluation_set: str = "natural_stream",
) -> float:
    row = _metric_row(frame, metric_name, evaluation_set=evaluation_set)
    value = row["metric_value"]
    assert isinstance(value, float)
    return value


def _metric_row(
    frame: pl.DataFrame,
    metric_name: str,
    *,
    evaluation_set: str = "natural_stream",
    dimension: str = "overall",
    value: str = "all",
) -> dict[str, object]:
    selected = frame.filter(
        (pl.col("evaluation_set") == evaluation_set)
        & (pl.col("stratum_dimension") == dimension)
        & (pl.col("stratum_value") == value)
        & (pl.col("metric_name") == metric_name)
    )
    assert selected.height == 1
    return deepcopy(selected.row(0, named=True))
