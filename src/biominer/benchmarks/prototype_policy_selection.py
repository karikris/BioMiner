"""Select and freeze an uncalibrated Build Week prototype policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from biominer.candidates.regional_union import REGIONAL_CANDIDATE_POLICY_VERSION
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.prototype_planner import (
    PROTOTYPE_REFERENCE_PLANNER_VERSION,
)
from biominer.vision.full_frame_attention import FULL_FRAME_VISUAL_INPUT_VERSION


PROTOTYPE_POLICY_SELECTION_CONFIG_VERSION = "prototype-policy-selection-job-v1.0.0"
PROTOTYPE_POLICY_VERSION = "build-week-target-aware-prototype-policy-v1.0.0"
PROTOTYPE_POLICY_STATUS = "prototype_uncalibrated"
PROTOTYPE_POLICY_REPORT_VERSION = "prototype-policy-selection-report-v1.0.0"
PROTOTYPE_POLICY_CANDIDATES_VERSION = "prototype-policy-selection-candidates-v1.0.0"
PROTOTYPE_POLICY_DECISIONS_VERSION = "prototype-policy-model-selection-decisions-v1.0.0"
PROTOTYPE_POLICY_CALIBRATION_AUDIT_VERSION = (
    "prototype-policy-calibration-margin-audit-v1.0.0"
)
MARGIN_POLICY_VERSION = "prototype-raw-margin-abstention-v1.0.0"

POLICY_FILE = "prototype_policy.json"
SELECTION_CANDIDATES_FILE = "prototype_policy_selection_candidates.parquet"
MODEL_SELECTION_DECISIONS_FILE = "prototype_policy_model_selection_decisions.parquet"
CALIBRATION_MARGIN_AUDIT_FILE = "prototype_policy_calibration_margin_audit.parquet"
REPORT_FILE = "prototype_policy_selection_report.json"
SUMMARY_FILE = "prototype_policy_selection_summary.md"

SELECTED_EXPERIMENT_ID = "B13"
MODEL_SELECTION_PARTITION = "model_selection"
CALIBRATION_PARTITION = "calibration"
FINAL_TEST_PARTITION = "final_test"
SCORE_SEMANTICS = "experimental_screening_evidence_uncalibrated_not_probability"
METRIC_SEMANTICS = (
    "provider_supported_retrieval_internal_consistency_not_classification_accuracy"
)

ELIGIBLE_EXPERIMENT_IDS = frozenset(
    {
        "B2",
        "B3",
        "B5",
        "B13",
        "B14-global",
        "B14-layered",
        "B15",
        "B16",
    }
)
SELECTION_PREFERENCE = (
    "B13",
    "B16",
    "B15",
    "B2",
    "B3",
    "B5",
    "B14-global",
    "B14-layered",
)


@dataclass(frozen=True, slots=True)
class PrototypePolicySelectionConfig:
    benchmark_predictions: Path
    benchmark_predictions_sha256: str
    benchmark_candidate_scores: Path
    benchmark_candidate_scores_sha256: str
    benchmark_report: Path
    benchmark_report_sha256: str
    reference_embeddings: Path
    reference_embeddings_sha256: str
    readiness: Path
    readiness_sha256: str
    staged_report: Path
    staged_report_sha256: str
    output_dir: Path
    target_accepted_taxon_key: str
    target_scientific_name: str
    storage_backend: str = "local"
    s3_permitted: bool = False
    raw_margin_threshold: float = 0.10
    minimum_target_scoreability_rate: float = 1.0
    minimum_target_top1_retrieval_rate: float = 1.0
    minimum_competitor_defeat_rate: float = 0.95
    minimum_full_availability_rate: float = 1.0

    def __post_init__(self) -> None:
        for field in (
            "benchmark_predictions",
            "benchmark_candidate_scores",
            "benchmark_report",
            "reference_embeddings",
            "readiness",
            "staged_report",
            "output_dir",
        ):
            path = Path(getattr(self, field)).expanduser()
            if "://" in str(path):
                raise ValueError(f"{field} must be a local path")
            object.__setattr__(self, field, path)
        for field in (
            "benchmark_predictions_sha256",
            "benchmark_candidate_scores_sha256",
            "benchmark_report_sha256",
            "reference_embeddings_sha256",
            "readiness_sha256",
            "staged_report_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.storage_backend != "local" or self.s3_permitted:
            raise ValueError("prototype policy selection requires local-only storage")
        for field in ("target_accepted_taxon_key", "target_scientific_name"):
            _required_text(getattr(self, field), field=field)
        for field in (
            "raw_margin_threshold",
            "minimum_target_scoreability_rate",
            "minimum_target_top1_retrieval_rate",
            "minimum_competitor_defeat_rate",
            "minimum_full_availability_rate",
        ):
            value = float(getattr(self, field))
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
            if field == "raw_margin_threshold":
                if value < 0.0:
                    raise ValueError("raw_margin_threshold must be non-negative")
            elif not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypePolicySelectionConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("prototype policy selection config must be an object")
        values = dict(payload)
        if (
            values.pop("schema_version", None)
            != PROTOTYPE_POLICY_SELECTION_CONFIG_VERSION
        ):
            raise ValueError("unsupported prototype policy selection config schema")
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"unknown prototype policy selection fields: {sorted(unknown)}"
            )
        return cls(**values)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                item.name: (
                    str(value)
                    if isinstance(value := getattr(self, item.name), Path)
                    else value
                )
                for item in fields(self)
            }
        )


@dataclass(frozen=True, slots=True)
class PrototypePolicySelectionResult:
    policy: dict[str, Any]
    report: dict[str, Any]
    policy_path: Path
    selection_candidates_path: Path
    model_selection_decisions_path: Path
    calibration_margin_audit_path: Path
    report_path: Path
    summary_path: Path


def select_prototype_policy(
    config: PrototypePolicySelectionConfig,
) -> PrototypePolicySelectionResult:
    """Select B13 from model-selection evidence and freeze raw-margin abstention."""

    started_at = datetime.now(UTC)
    _validate_hashes(config)
    readiness = _read_object(config.readiness, label="readiness")
    benchmark_report = _read_object(config.benchmark_report, label="benchmark report")
    staged_report = _read_object(config.staged_report, label="staged report")
    embeddings = pl.read_parquet(config.reference_embeddings)
    _validate_upstream_contracts(
        config,
        readiness=readiness,
        benchmark_report=benchmark_report,
        staged_report=staged_report,
        embeddings=embeddings,
    )

    model_selection = (
        pl.scan_parquet(config.benchmark_predictions)
        .filter(pl.col("dataset_split") == MODEL_SELECTION_PARTITION)
        .collect()
    )
    calibration = (
        pl.scan_parquet(config.benchmark_predictions)
        .filter(pl.col("dataset_split") == CALIBRATION_PARTITION)
        .collect()
    )
    _validate_partition(model_selection, expected=MODEL_SELECTION_PARTITION)
    _validate_partition(calibration, expected=CALIBRATION_PARTITION)

    candidate_frame = _selection_candidates(model_selection, config=config)
    selected = candidate_frame.filter(pl.col("selected"))
    if (
        selected.height != 1
        or selected["experiment_id"].item() != SELECTED_EXPERIMENT_ID
    ):
        raise RuntimeError("selection contract did not choose B13 exactly once")
    selected_metrics = selected.to_dicts()[0]
    b0_metrics = candidate_frame.filter(pl.col("experiment_id") == "B0").to_dicts()[0]

    decisions = _model_selection_decisions(model_selection, config=config)
    calibration_audit = _calibration_margin_audit(calibration, config=config)
    identity = _frozen_identity(
        config,
        readiness=readiness,
        embeddings=embeddings,
    )
    policy = _policy_manifest(
        config,
        identity=identity,
        selected_metrics=selected_metrics,
        b0_metrics=b0_metrics,
        decisions=decisions,
        calibration_audit=calibration_audit,
        benchmark_report=benchmark_report,
        staged_report=staged_report,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = config.output_dir / POLICY_FILE
    selection_candidates_path = config.output_dir / SELECTION_CANDIDATES_FILE
    model_selection_decisions_path = config.output_dir / MODEL_SELECTION_DECISIONS_FILE
    calibration_margin_audit_path = config.output_dir / CALIBRATION_MARGIN_AUDIT_FILE
    report_path = config.output_dir / REPORT_FILE
    summary_path = config.output_dir / SUMMARY_FILE
    candidate_frame.write_parquet(selection_candidates_path)
    decisions.write_parquet(model_selection_decisions_path)
    calibration_audit.write_parquet(calibration_margin_audit_path)
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ended_at = datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": PROTOTYPE_POLICY_REPORT_VERSION,
        "status": "selected",
        "policy_status": PROTOTYPE_POLICY_STATUS,
        "prototype_only": True,
        "experimental_screening_evidence_only": True,
        "configuration_fingerprint": config.fingerprint,
        "policy_fingerprint": policy["policy_fingerprint"],
        "selected_experiment_id": SELECTED_EXPERIMENT_ID,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "storage": {
            "backend": "local",
            "output_dir": str(config.output_dir),
            "s3_used": False,
        },
        "partition_use": {
            "support_train": "upstream_reference_fit_only",
            "model_selection": "used_for_policy_choice",
            "calibration": (
                "margin_coverage_audit_only_no_label_based_threshold_fitting"
            ),
            "final_test": "not_read_or_used_for_selection",
        },
        "selection": {
            "candidate_count": candidate_frame.height,
            "eligible_count": candidate_frame.filter(pl.col("eligible")).height,
            "selected_metrics": _json_safe(selected_metrics),
            "selection_evidence_fingerprint": policy["selection_evidence_fingerprint"],
        },
        "calibration": policy["calibration"],
        "margin_policy": policy["margin_policy"],
        "operational_evidence": policy["operational_evidence"],
        "artifacts": {
            "policy": _artifact(policy_path),
            "selection_candidates": _artifact(
                selection_candidates_path, rows=candidate_frame.height
            ),
            "model_selection_decisions": _artifact(
                model_selection_decisions_path, rows=decisions.height
            ),
            "calibration_margin_audit": _artifact(
                calibration_margin_audit_path, rows=calibration_audit.height
            ),
        },
        "limitations": policy["limitations"],
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        _markdown(policy, candidate_frame, decisions, calibration_audit),
        encoding="utf-8",
    )
    return PrototypePolicySelectionResult(
        policy=policy,
        report=report,
        policy_path=policy_path,
        selection_candidates_path=selection_candidates_path,
        model_selection_decisions_path=model_selection_decisions_path,
        calibration_margin_audit_path=calibration_margin_audit_path,
        report_path=report_path,
        summary_path=summary_path,
    )


def _selection_candidates(
    model_selection: pl.DataFrame,
    *,
    config: PrototypePolicySelectionConfig,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    preference = {
        experiment_id: index for index, experiment_id in enumerate(SELECTION_PREFERENCE)
    }
    for (experiment_id,), frame in model_selection.group_by(
        "experiment_id", maintain_order=True
    ):
        target_rows = frame.filter(pl.col("target_is_provider_label"))
        non_target_rows = frame.filter(~pl.col("target_is_provider_label"))
        cluster = frame.group_by("geo_cluster_id").agg(
            pl.len().alias("records"),
            pl.col("target_rank").is_not_null().mean().alias("target_scoreability"),
            (pl.col("provider_label_rank") == 1)
            .fill_null(False)
            .mean()
            .alias("provider_top1_consistency"),
        )
        target_scoreability = float(frame["target_rank"].is_not_null().mean())
        target_top1 = _rate(target_rows, pl.col("target_rank") == 1)
        competitor_defeat = _rate(
            non_target_rows,
            pl.col("predicted_taxon_key") != config.target_accepted_taxon_key,
        )
        provider_top1 = _rate(frame, pl.col("provider_label_rank") == 1)
        full_availability = _rate(frame, pl.col("availability_status") == "available")
        low_margin = _rate(
            frame,
            pl.col("raw_margin").is_null()
            | (pl.col("raw_margin") < config.raw_margin_threshold),
        )
        eligibility_reasons: list[str] = []
        if experiment_id not in ELIGIBLE_EXPERIMENT_IDS:
            eligibility_reasons.append("experiment_family_not_deployable")
        if frame["candidate_count"].min() < 3:
            eligibility_reasons.append("binary_or_incomplete_competitor_ranking")
        if target_scoreability < config.minimum_target_scoreability_rate:
            eligibility_reasons.append("target_not_scoreable_for_every_record")
        if target_top1 < config.minimum_target_top1_retrieval_rate:
            eligibility_reasons.append("target_retrieval_below_required_rate")
        if competitor_defeat < config.minimum_competitor_defeat_rate:
            eligibility_reasons.append(
                "strong_competitors_do_not_reliably_defeat_target"
            )
        if full_availability < config.minimum_full_availability_rate:
            eligibility_reasons.append("visual_or_reference_inputs_not_fully_available")
        if set(frame["score_semantics"]) != {SCORE_SEMANTICS}:
            eligibility_reasons.append("invalid_score_semantics")
        eligible = not eligibility_reasons
        cluster_consistency = cluster["provider_top1_consistency"]
        rows.append(
            {
                "schema_version": PROTOTYPE_POLICY_CANDIDATES_VERSION,
                "experiment_id": experiment_id,
                "experiment_name": str(frame["experiment_name"][0]),
                "model_selection_record_count": frame.height,
                "candidate_count_min": int(frame["candidate_count"].min()),
                "candidate_count_max": int(frame["candidate_count"].max()),
                "target_scoreability_rate": target_scoreability,
                "target_top1_retrieval_rate": target_top1,
                "competitor_defeat_target_rate": competitor_defeat,
                "provider_label_top1_consistency_rate": provider_top1,
                "full_availability_rate": full_availability,
                "low_margin_or_missing_rate_at_policy_threshold": low_margin,
                "geo_cluster_count": cluster.height,
                "minimum_geo_cluster_target_scoreability_rate": float(
                    cluster["target_scoreability"].min()
                ),
                "geo_cluster_provider_consistency_spread": float(
                    cluster_consistency.max() - cluster_consistency.min()
                ),
                "eligible": eligible,
                "eligibility_reasons_json": json.dumps(
                    eligibility_reasons, separators=(",", ":")
                ),
                "selection_preference": preference.get(experiment_id, 999),
                "selected": False,
                "metric_semantics": METRIC_SEMANTICS,
            }
        )
    frame = pl.DataFrame(rows, schema=_selection_candidates_schema(), strict=True)
    eligible = frame.filter(pl.col("eligible"))
    if eligible.is_empty():
        raise RuntimeError("no experiment satisfies the prototype selection gates")
    selected_id = eligible.sort(
        [
            "target_scoreability_rate",
            "target_top1_retrieval_rate",
            "competitor_defeat_target_rate",
            "full_availability_rate",
            "provider_label_top1_consistency_rate",
            "selection_preference",
            "experiment_id",
        ],
        descending=[True, True, True, True, True, False, False],
    )["experiment_id"][0]
    if selected_id != SELECTED_EXPERIMENT_ID:
        raise RuntimeError(
            f"selection evidence chose {selected_id}, expected {SELECTED_EXPERIMENT_ID}"
        )
    return frame.with_columns(
        (pl.col("experiment_id") == selected_id).alias("selected")
    ).sort("selection_preference", "experiment_id")


def _model_selection_decisions(
    model_selection: pl.DataFrame,
    *,
    config: PrototypePolicySelectionConfig,
) -> pl.DataFrame:
    selected = model_selection.filter(pl.col("experiment_id") == SELECTED_EXPERIMENT_ID)
    if selected.is_empty():
        raise ValueError("selected experiment is absent from model_selection")
    return (
        selected.select(
            pl.lit(PROTOTYPE_POLICY_DECISIONS_VERSION).alias("schema_version"),
            "reference_media_id",
            "dataset_split",
            "route",
            "geo_cluster_id",
            "provider_accepted_taxon_key",
            "provider_scientific_name",
            "human_verified",
            "predicted_taxon_key",
            "predicted_scientific_name",
            "raw_margin",
            "target_rank",
            "provider_label_rank",
            "candidate_count",
            "availability_status",
            "score_semantics",
        )
        .with_columns(
            (
                pl.col("raw_margin").is_null()
                | (pl.col("raw_margin") < config.raw_margin_threshold)
                | (pl.col("availability_status") != "available")
            ).alias("policy_abstained"),
            pl.when(pl.col("availability_status") != "available")
            .then(pl.lit("required_evidence_unavailable"))
            .when(pl.col("raw_margin").is_null())
            .then(pl.lit("raw_margin_unavailable"))
            .when(pl.col("raw_margin") < config.raw_margin_threshold)
            .then(pl.lit("raw_margin_below_threshold"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("policy_abstention_reason"),
            pl.lit(config.raw_margin_threshold).alias("raw_margin_threshold"),
            pl.lit(MARGIN_POLICY_VERSION).alias("margin_policy_version"),
            pl.lit(METRIC_SEMANTICS).alias("evaluation_semantics"),
        )
        .cast(_model_selection_decisions_schema())
        .sort("reference_media_id")
    )


def _calibration_margin_audit(
    calibration: pl.DataFrame,
    *,
    config: PrototypePolicySelectionConfig,
) -> pl.DataFrame:
    selected = calibration.filter(pl.col("experiment_id") == SELECTED_EXPERIMENT_ID)
    if selected.is_empty():
        raise ValueError("selected experiment is absent from calibration")
    return (
        selected.select(
            pl.lit(PROTOTYPE_POLICY_CALIBRATION_AUDIT_VERSION).alias("schema_version"),
            "reference_media_id",
            "dataset_split",
            "route",
            "geo_cluster_id",
            "human_verified",
            "raw_margin",
            "availability_status",
        )
        .with_columns(
            (
                pl.col("raw_margin").is_not_null()
                & (pl.col("raw_margin") >= config.raw_margin_threshold)
                & (pl.col("availability_status") == "available")
            ).alias("would_accept"),
            pl.lit(config.raw_margin_threshold).alias("raw_margin_threshold"),
            pl.lit("coverage_only_no_label_metric").alias("audit_semantics"),
            pl.lit(False).alias("used_to_fit_threshold"),
            pl.lit(False).alias("used_to_fit_calibrator"),
        )
        .cast(_calibration_margin_audit_schema())
        .sort("reference_media_id")
    )


def _frozen_identity(
    config: PrototypePolicySelectionConfig,
    *,
    readiness: Mapping[str, Any],
    embeddings: pl.DataFrame,
) -> dict[str, Any]:
    values = {
        "reference_bank_version": _single_text(embeddings, "reference_bank_version"),
        "support_manifest_fingerprint": _single_text(
            embeddings, "support_manifest_fingerprint"
        ),
        "reference_embeddings_sha256": config.reference_embeddings_sha256,
        "reference_planner_version": PROTOTYPE_REFERENCE_PLANNER_VERSION,
        "candidate_planner_version": REGIONAL_CANDIDATE_POLICY_VERSION,
        "model_id": _single_text(embeddings, "model_id"),
        "model_revision": _single_text(embeddings, "model_revision"),
        "model_weights_sha256": _single_text(embeddings, "model_weights_sha256"),
        "preprocessing_version": _single_text(embeddings, "preprocessing_version"),
        "preprocessing_fingerprint": _single_text(
            embeddings, "preprocessing_fingerprint"
        ),
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        "reference_prototype_method": "normalized_observation_mean",
        "selected_reference_scope": "global",
        "selected_experiment_id": SELECTED_EXPERIMENT_ID,
        "margin_policy_version": MARGIN_POLICY_VERSION,
        "readiness_policy_fingerprint": str(readiness["policy_fingerprint"]),
        "split_fingerprint": str(readiness["split_fingerprint"]),
    }
    classifier_fingerprint = canonical_semantic_fingerprint(
        {
            "classifier_contract": "global-normalized-reference-centroid-v1.0.0",
            **values,
        }
    )
    values["classifier_fingerprint"] = classifier_fingerprint
    values["calibrator_fingerprint"] = None
    return values


def _policy_manifest(
    config: PrototypePolicySelectionConfig,
    *,
    identity: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    b0_metrics: Mapping[str, Any],
    decisions: pl.DataFrame,
    calibration_audit: pl.DataFrame,
    benchmark_report: Mapping[str, Any],
    staged_report: Mapping[str, Any],
) -> dict[str, Any]:
    selection_evidence_fingerprint = canonical_semantic_fingerprint(
        {
            "selected_metrics": _json_safe(selected_metrics),
            "model_selection_decisions": decisions.to_dicts(),
        }
    )
    accepted = decisions.filter(~pl.col("policy_abstained")).height
    calibration_accepted = calibration_audit.filter(pl.col("would_accept")).height
    p3 = next(stage for stage in staged_report["stages"] if stage["stage_id"] == "P3")
    semantic_policy: dict[str, Any] = {
        "schema_version": PROTOTYPE_POLICY_VERSION,
        "policy_status": PROTOTYPE_POLICY_STATUS,
        "deployment_status": "prototype",
        "prototype_only": True,
        "experimental_screening_evidence_only": True,
        "target": {
            "accepted_taxon_key": config.target_accepted_taxon_key,
            "scientific_name": config.target_scientific_name,
        },
        "selected_policy": {
            "experiment_id": SELECTED_EXPERIMENT_ID,
            "experiment_name": "global_references",
            "classifier_contract": "global-normalized-reference-centroid-v1.0.0",
            "reference_scope": "global",
            "regional_policy": (
                "not_selected_until_cluster_conditioning_preserves_target_scoreability"
            ),
            "candidate_union_requirement": "complete_regional_union_with_target_once",
            "target_always_scored": True,
            "higher_rank_pruning_permitted": False,
            "spatial_crop_permitted": False,
            "visual_input": "raw_full_image",
        },
        "frozen_identity": dict(identity),
        "margin_policy": {
            "version": MARGIN_POLICY_VERSION,
            "score_kind": "raw_top1_minus_top2_similarity_margin",
            "threshold": config.raw_margin_threshold,
            "threshold_source": (
                "predeclared_conservative_build_week_threshold_not_label_fitted"
            ),
            "accept_rule": (
                "required evidence available and raw margin greater than or equal "
                "to threshold"
            ),
            "ordered_abstention_rules": [
                "required_evidence_unavailable",
                "raw_margin_unavailable",
                "raw_margin_below_threshold",
            ],
            "probability_interpretation_permitted": False,
        },
        "calibration": {
            "status": "not_fitted_insufficient_independently_reviewed_labels",
            "human_verified_calibration_records": int(
                calibration_audit["human_verified"].sum()
            ),
            "calibration_record_count": calibration_audit.height,
            "calibrator_fingerprint": None,
            "probabilities_emitted": False,
            "threshold_fitted_on_calibration": False,
            "coverage_audit_only": True,
            "would_accept_count": calibration_accepted,
            "would_abstain_count": calibration_audit.height - calibration_accepted,
        },
        "partition_contract": {
            "model_choice_partition": MODEL_SELECTION_PARTITION,
            "calibration_partition": CALIBRATION_PARTITION,
            "final_test_partition": FINAL_TEST_PARTITION,
            "final_test_used_for_selection": False,
            "final_test_evaluation_status": (
                "not_evaluated_no_independently_reviewed_labels"
            ),
        },
        "selection_evidence": {
            "fingerprint": selection_evidence_fingerprint,
            "model_selection_record_count": decisions.height,
            "target_scoreability_rate": selected_metrics["target_scoreability_rate"],
            "target_top1_retrieval_rate": selected_metrics[
                "target_top1_retrieval_rate"
            ],
            "competitor_defeat_target_rate": selected_metrics[
                "competitor_defeat_target_rate"
            ],
            "provider_label_top1_consistency_rate": selected_metrics[
                "provider_label_top1_consistency_rate"
            ],
            "accepted_count_at_margin_policy": accepted,
            "abstained_count_at_margin_policy": decisions.height - accepted,
            "coverage_at_margin_policy": accepted / decisions.height,
            "metric_semantics": METRIC_SEMANTICS,
        },
        "b0_comparison": {
            "target_scoreability_rate_b0": b0_metrics["target_scoreability_rate"],
            "target_scoreability_rate_selected": selected_metrics[
                "target_scoreability_rate"
            ],
            "target_scoreability_improvement": (
                selected_metrics["target_scoreability_rate"]
                - b0_metrics["target_scoreability_rate"]
            ),
            "target_top1_retrieval_rate_b0": b0_metrics["target_top1_retrieval_rate"],
            "target_top1_retrieval_rate_selected": selected_metrics[
                "target_top1_retrieval_rate"
            ],
            "low_margin_rate_b0": b0_metrics[
                "low_margin_or_missing_rate_at_policy_threshold"
            ],
            "low_margin_rate_selected": selected_metrics[
                "low_margin_or_missing_rate_at_policy_threshold"
            ],
            "interpretation": (
                "target top1 retrieval on provider-target rows ties B0, while "
                "target scoreability improves because B0 hierarchy-prunes the "
                "target on non-target comparison rows"
            ),
        },
        "planner_comparison": {
            "regional_only": (
                "rejected_target_not_scoreable_for_every_model_selection_record"
            ),
            "global_only": "selected",
            "trust_first_layered": (
                "rejected_target_missing_in_one_conditioned_cluster"
            ),
            "decision": (
                "global reference evidence is safer for the prototype; regional "
                "conditioning remains diagnostic until support coverage improves"
            ),
        },
        "operational_evidence": {
            "staged_status": staged_report["status"],
            "planned_records": staged_report["counts"]["planned"],
            "classified_records": staged_report["counts"]["classified"],
            "retryable_failures": staged_report["counts"]["failures"],
            "failure_rate": p3["failure_rate"],
            "records_per_second": p3["records_per_second"],
            "rss_peak_memory": p3["rss_peak_memory"],
            "catastrophic_operational_failure": False,
            "target_always_scored_in_staged_run": staged_report["candidate_union"][
                "target_always_scored"
            ],
            "complete_candidate_union_in_staged_run": p3["checks"][
                "complete_candidate_union_scored"
            ],
        },
        "upstream_benchmark": {
            "report_fingerprint": benchmark_report["report_fingerprint"],
            "prediction_artifact_sha256": config.benchmark_predictions_sha256,
            "candidate_score_artifact_sha256": (
                config.benchmark_candidate_scores_sha256
            ),
            "classification_accuracy_reported": False,
        },
        "limitations": [
            "No reference label is independently human taxonomically verified.",
            "Provider-supported consistency metrics are not classification accuracy.",
            "The policy is uncalibrated and emits no probabilities.",
            "The raw 0.10 margin threshold is conservative and predeclared, not learned from labels.",
            "Regional and trust-first layered reference policies are not selected because they can remove the target in a sparse cluster.",
            "The only larval record is outside support_train, so larval reference scoring remains unavailable.",
            "B11 and B12 focused/masked visual inputs were unavailable in the executable benchmark subset.",
        ],
        "score_semantics": SCORE_SEMANTICS,
    }
    semantic_policy["selection_evidence_fingerprint"] = selection_evidence_fingerprint
    policy_fingerprint_payload = {
        key: semantic_policy[key]
        for key in (
            "schema_version",
            "policy_status",
            "deployment_status",
            "prototype_only",
            "experimental_screening_evidence_only",
            "target",
            "selected_policy",
            "frozen_identity",
            "margin_policy",
            "calibration",
            "partition_contract",
            "selection_evidence",
            "b0_comparison",
            "planner_comparison",
            "operational_evidence",
            "limitations",
            "score_semantics",
            "selection_evidence_fingerprint",
        )
    }
    semantic_policy["policy_fingerprint"] = canonical_semantic_fingerprint(
        policy_fingerprint_payload
    )
    return semantic_policy


def _validate_upstream_contracts(
    config: PrototypePolicySelectionConfig,
    *,
    readiness: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
    staged_report: Mapping[str, Any],
    embeddings: pl.DataFrame,
) -> None:
    if readiness.get("bank_status") != "prototype_only":
        raise ValueError("reference bank must be prototype_only")
    if not readiness.get("classification_authorised"):
        raise ValueError("prototype readiness does not authorize classification")
    if readiness.get("human_verification_complete") is not False:
        raise ValueError("prototype selection expects incomplete human verification")
    if int(readiness["counts"]["human_verified_count"]) != 0:
        raise ValueError("prototype uncalibrated policy requires zero reviewed labels")
    if benchmark_report.get("metrics", {}).get("classification_accuracy_reported"):
        raise ValueError("benchmark must not report unsupported accuracy")
    if staged_report.get("status") != "passed":
        raise ValueError("staged prototype report must have passed")
    if staged_report.get("storage", {}).get("s3_accessed") is not False:
        raise ValueError("staged prototype must not have accessed S3")
    if staged_report.get("candidate_union", {}).get("target_always_scored") is not True:
        raise ValueError("staged prototype did not always score the target")
    if embeddings.height != 81:
        raise ValueError("prototype policy requires the frozen 81-record bank")
    if bool(embeddings["human_verified"].any()):
        raise ValueError("prototype bank unexpectedly contains reviewed labels")
    if set(embeddings["dataset_split"]) != {
        "support_train",
        "model_selection",
        "calibration",
        "final_test",
    }:
        raise ValueError("prototype bank split contract is incomplete")
    if (
        _single_text(embeddings, "reference_bank_version")
        != readiness["reference_bank_version"]
    ):
        raise ValueError("readiness and embeddings reference-bank versions differ")
    if config.target_accepted_taxon_key != readiness["target_accepted_taxon_key"]:
        raise ValueError("target key differs from readiness")


def _validate_partition(frame: pl.DataFrame, *, expected: str) -> None:
    if frame.is_empty():
        raise ValueError(f"{expected} partition is empty")
    if set(frame["dataset_split"]) != {expected}:
        raise ValueError(f"{expected} partition filter leaked other splits")
    if (
        frame["reference_media_id"].n_unique() * frame["experiment_id"].n_unique()
        != frame.height
    ):
        raise ValueError(f"{expected} has duplicate or incomplete experiment rows")
    if bool(frame["classification_accuracy_permitted"].any()):
        raise ValueError(f"{expected} incorrectly permits classification accuracy")


def _validate_hashes(config: PrototypePolicySelectionConfig) -> None:
    checks = (
        (config.benchmark_predictions, config.benchmark_predictions_sha256),
        (
            config.benchmark_candidate_scores,
            config.benchmark_candidate_scores_sha256,
        ),
        (config.benchmark_report, config.benchmark_report_sha256),
        (config.reference_embeddings, config.reference_embeddings_sha256),
        (config.readiness, config.readiness_sha256),
        (config.staged_report, config.staged_report_sha256),
    )
    for path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )


def _selection_candidates_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "experiment_id": pl.String,
        "experiment_name": pl.String,
        "model_selection_record_count": pl.UInt32,
        "candidate_count_min": pl.UInt32,
        "candidate_count_max": pl.UInt32,
        "target_scoreability_rate": pl.Float64,
        "target_top1_retrieval_rate": pl.Float64,
        "competitor_defeat_target_rate": pl.Float64,
        "provider_label_top1_consistency_rate": pl.Float64,
        "full_availability_rate": pl.Float64,
        "low_margin_or_missing_rate_at_policy_threshold": pl.Float64,
        "geo_cluster_count": pl.UInt32,
        "minimum_geo_cluster_target_scoreability_rate": pl.Float64,
        "geo_cluster_provider_consistency_spread": pl.Float64,
        "eligible": pl.Boolean,
        "eligibility_reasons_json": pl.String,
        "selection_preference": pl.UInt32,
        "selected": pl.Boolean,
        "metric_semantics": pl.String,
    }


def _model_selection_decisions_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "dataset_split": pl.String,
        "route": pl.String,
        "geo_cluster_id": pl.String,
        "provider_accepted_taxon_key": pl.String,
        "provider_scientific_name": pl.String,
        "human_verified": pl.Boolean,
        "predicted_taxon_key": pl.String,
        "predicted_scientific_name": pl.String,
        "raw_margin": pl.Float64,
        "target_rank": pl.UInt32,
        "provider_label_rank": pl.UInt32,
        "candidate_count": pl.UInt32,
        "availability_status": pl.String,
        "score_semantics": pl.String,
        "policy_abstained": pl.Boolean,
        "policy_abstention_reason": pl.String,
        "raw_margin_threshold": pl.Float64,
        "margin_policy_version": pl.String,
        "evaluation_semantics": pl.String,
    }


def _calibration_margin_audit_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "dataset_split": pl.String,
        "route": pl.String,
        "geo_cluster_id": pl.String,
        "human_verified": pl.Boolean,
        "raw_margin": pl.Float64,
        "availability_status": pl.String,
        "would_accept": pl.Boolean,
        "raw_margin_threshold": pl.Float64,
        "audit_semantics": pl.String,
        "used_to_fit_threshold": pl.Boolean,
        "used_to_fit_calibrator": pl.Boolean,
    }


def _rate(frame: pl.DataFrame, expression: pl.Expr) -> float:
    if frame.is_empty():
        return 0.0
    return float(frame.select(expression.fill_null(False).mean()).item())


def _single_text(frame: pl.DataFrame, column: str) -> str:
    values = frame[column].drop_nulls().unique().to_list()
    if len(values) != 1 or not str(values[0]).strip():
        raise ValueError(f"{column} must have exactly one non-empty value")
    return str(values[0])


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "uri": str(path),
        "byte_count": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if rows is not None:
        result["row_count"] = rows
    return result


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): (item.item() if hasattr(item, "item") else item)
        for key, item in value.items()
    }


def _markdown(
    policy: Mapping[str, Any],
    candidates: pl.DataFrame,
    decisions: pl.DataFrame,
    calibration_audit: pl.DataFrame,
) -> str:
    selection = policy["selection_evidence"]
    accepted = decisions.filter(~pl.col("policy_abstained")).height
    calibration_accepted = calibration_audit.filter(pl.col("would_accept")).height
    lines = [
        "# Build Week prototype policy selection",
        "",
        f"- Policy status: `{policy['policy_status']}`",
        f"- Selected experiment: `{policy['selected_policy']['experiment_id']}` global reference evidence",
        f"- Policy fingerprint: `{policy['policy_fingerprint']}`",
        "- Probability calibration: not fitted; no independently reviewed labels exist.",
        f"- Raw margin abstention: accept at margin >= {policy['margin_policy']['threshold']:.2f}.",
        f"- Model-selection coverage at that margin: {accepted}/{decisions.height}.",
        f"- Calibration-partition coverage audit: {calibration_accepted}/{calibration_audit.height}; no label metric computed.",
        "- Final-test use: none for selection or thresholding.",
        "- Storage: local only; S3 not used.",
        "",
        "## Selection evidence",
        "",
        f"- Target scoreability: {selection['target_scoreability_rate']:.3f}",
        f"- Target top-1 retrieval: {selection['target_top1_retrieval_rate']:.3f}",
        f"- Non-target records defeating target: {selection['competitor_defeat_target_rate']:.3f}",
        f"- Provider-label top-1 internal consistency: {selection['provider_label_top1_consistency_rate']:.3f} (not accuracy)",
        "",
        "## Experiment eligibility",
        "",
        "| Experiment | Eligible | Target scoreable | Target top-1 | Competitor defeats target | Full availability | Selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates.sort("selection_preference", "experiment_id").iter_rows(
        named=True
    ):
        lines.append(
            "| {id} | {eligible} | {scoreable:.3f} | {target:.3f} | "
            "{competitor:.3f} | {available:.3f} | {selected} |".format(
                id=row["experiment_id"],
                eligible="yes" if row["eligible"] else "no",
                scoreable=row["target_scoreability_rate"],
                target=row["target_top1_retrieval_rate"],
                competitor=row["competitor_defeat_target_rate"],
                available=row["full_availability_rate"],
                selected="yes" if row["selected"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in policy["limitations"]],
        ]
    )
    return "\n".join(lines) + "\n"


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>") from exc
    if value.lower() != value:
        raise ValueError(f"{field} must use lowercase hex")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


__all__ = [
    "CALIBRATION_MARGIN_AUDIT_FILE",
    "MARGIN_POLICY_VERSION",
    "MODEL_SELECTION_DECISIONS_FILE",
    "POLICY_FILE",
    "PROTOTYPE_POLICY_STATUS",
    "PROTOTYPE_POLICY_VERSION",
    "PrototypePolicySelectionConfig",
    "PrototypePolicySelectionResult",
    "REPORT_FILE",
    "SELECTED_EXPERIMENT_ID",
    "SELECTION_CANDIDATES_FILE",
    "select_prototype_policy",
]
