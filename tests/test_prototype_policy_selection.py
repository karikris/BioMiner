from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from biominer.benchmarks.prototype_policy_selection import (
    PROTOTYPE_POLICY_STATUS,
    SELECTED_EXPERIMENT_ID,
    PrototypePolicySelectionConfig,
    select_prototype_policy,
)


EXPERIMENT_IDS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14-regional",
    "B14-global",
    "B14-layered",
    "B15",
    "B16",
)
TARGET = "gbif:target"


def test_selects_global_uncalibrated_policy_without_final_test_use(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path / "first", final_test_margin=0.01)

    result = select_prototype_policy(config)

    assert result.policy["policy_status"] == PROTOTYPE_POLICY_STATUS
    assert result.policy["selected_policy"]["experiment_id"] == SELECTED_EXPERIMENT_ID
    assert result.policy["selected_policy"]["reference_scope"] == "global"
    assert result.policy["selected_policy"]["target_always_scored"] is True
    assert result.policy["selected_policy"]["higher_rank_pruning_permitted"] is False
    assert result.policy["calibration"] == {
        "status": "not_fitted_insufficient_independently_reviewed_labels",
        "human_verified_calibration_records": 0,
        "calibration_record_count": 2,
        "calibrator_fingerprint": None,
        "probabilities_emitted": False,
        "threshold_fitted_on_calibration": False,
        "coverage_audit_only": True,
        "would_accept_count": 1,
        "would_abstain_count": 1,
    }
    assert result.policy["partition_contract"]["final_test_used_for_selection"] is False
    assert result.policy["margin_policy"]["threshold"] == 0.10
    assert (
        result.policy["margin_policy"]["probability_interpretation_permitted"] is False
    )
    assert result.policy["b0_comparison"]["target_scoreability_improvement"] == 0.5
    candidates = pl.read_parquet(result.selection_candidates_path)
    assert candidates.filter(pl.col("selected"))["experiment_id"].to_list() == [
        SELECTED_EXPERIMENT_ID
    ]
    assert (
        candidates.filter(pl.col("experiment_id") == "B0")["eligible"].item() is False
    )
    decisions = pl.read_parquet(result.model_selection_decisions_path)
    assert set(decisions["dataset_split"]) == {"model_selection"}
    assert decisions["policy_abstained"].sum() == 1
    calibration = pl.read_parquet(result.calibration_margin_audit_path)
    assert set(calibration["dataset_split"]) == {"calibration"}
    assert calibration["used_to_fit_threshold"].to_list() == [False, False]
    assert calibration["used_to_fit_calibrator"].to_list() == [False, False]


def test_final_test_changes_do_not_change_selected_policy_fingerprint(
    tmp_path: Path,
) -> None:
    first = select_prototype_policy(
        _fixture_config(tmp_path / "first", final_test_margin=0.01)
    )
    second = select_prototype_policy(
        _fixture_config(tmp_path / "second", final_test_margin=99.0)
    )

    assert first.policy["policy_fingerprint"] == second.policy["policy_fingerprint"]
    assert (
        first.policy["selection_evidence_fingerprint"]
        == second.policy["selection_evidence_fingerprint"]
    )


def test_rejects_s3_policy_selection(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path / "fixture", final_test_margin=0.01)
    values = {field: getattr(config, field) for field in config.__dataclass_fields__}
    values["storage_backend"] = "s3"
    values["s3_permitted"] = True

    try:
        PrototypePolicySelectionConfig(**values)
    except ValueError as exc:
        assert "local-only storage" in str(exc)
    else:
        raise AssertionError("S3 policy selection should be rejected")


def _fixture_config(
    root: Path,
    *,
    final_test_margin: float,
) -> PrototypePolicySelectionConfig:
    root.mkdir(parents=True)
    predictions_path = root / "predictions.parquet"
    candidate_scores_path = root / "candidate_scores.parquet"
    benchmark_report_path = root / "benchmark_report.json"
    embeddings_path = root / "embeddings.parquet"
    readiness_path = root / "readiness.json"
    staged_report_path = root / "staged_report.json"
    _predictions(final_test_margin).write_parquet(predictions_path)
    pl.DataFrame({"fixture": [1]}).write_parquet(candidate_scores_path)
    benchmark_report_path.write_text(
        json.dumps(
            {
                "report_fingerprint": _fingerprint("benchmark"),
                "metrics": {"classification_accuracy_reported": False},
            }
        ),
        encoding="utf-8",
    )
    _embeddings().write_parquet(embeddings_path)
    readiness_path.write_text(
        json.dumps(
            {
                "bank_status": "prototype_only",
                "classification_authorised": True,
                "human_verification_complete": False,
                "counts": {"human_verified_count": 0},
                "reference_bank_version": "fixture-bank-v1",
                "policy_fingerprint": _fingerprint("readiness"),
                "split_fingerprint": _fingerprint("split"),
                "target_accepted_taxon_key": TARGET,
            }
        ),
        encoding="utf-8",
    )
    staged_report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "storage": {"s3_accessed": False},
                "candidate_union": {"target_always_scored": True},
                "counts": {"planned": 100, "classified": 99, "failures": 1},
                "stages": [
                    {
                        "stage_id": "P3",
                        "failure_rate": 0.01,
                        "records_per_second": 2.0,
                        "rss_peak_memory": 1024,
                        "checks": {"complete_candidate_union_scored": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return PrototypePolicySelectionConfig(
        benchmark_predictions=predictions_path,
        benchmark_predictions_sha256=_sha256(predictions_path),
        benchmark_candidate_scores=candidate_scores_path,
        benchmark_candidate_scores_sha256=_sha256(candidate_scores_path),
        benchmark_report=benchmark_report_path,
        benchmark_report_sha256=_sha256(benchmark_report_path),
        reference_embeddings=embeddings_path,
        reference_embeddings_sha256=_sha256(embeddings_path),
        readiness=readiness_path,
        readiness_sha256=_sha256(readiness_path),
        staged_report=staged_report_path,
        staged_report_sha256=_sha256(staged_report_path),
        output_dir=root / "output",
        target_accepted_taxon_key=TARGET,
        target_scientific_name="Fixture target",
    )


def _predictions(final_test_margin: float) -> pl.DataFrame:
    rows = []
    for split in ("model_selection", "calibration", "final_test"):
        for item_index, provider_target in enumerate((True, False)):
            media_id = f"{split}:{item_index}"
            for experiment_id in EXPERIMENT_IDS:
                target_scoreable = True
                available = "available"
                candidate_count = 10
                predicted_key = TARGET if provider_target else "gbif:competitor"
                raw_margin = 0.20 if provider_target else 0.05
                if split == "final_test":
                    raw_margin = final_test_margin
                if experiment_id == "B0" and not provider_target:
                    target_scoreable = False
                if experiment_id == "B4" and not provider_target:
                    target_scoreable = False
                    predicted_key = TARGET
                if experiment_id in {"B6", "B7", "B8", "B9"}:
                    candidate_count = 2
                if experiment_id in {"B11", "B12"}:
                    available = "partial"
                if (
                    experiment_id in {"B14-regional", "B14-layered"}
                    and not provider_target
                ):
                    target_scoreable = False
                    available = "regional_cluster_support"
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_id.lower(),
                        "reference_media_id": media_id,
                        "dataset_split": split,
                        "route": "adult_field",
                        "geo_cluster_id": ("geo:a" if provider_target else "geo:b"),
                        "provider_accepted_taxon_key": (
                            TARGET if provider_target else "gbif:competitor"
                        ),
                        "provider_scientific_name": (
                            "Fixture target"
                            if provider_target
                            else "Fixture competitor"
                        ),
                        "human_verified": False,
                        "predicted_taxon_key": predicted_key,
                        "predicted_scientific_name": (
                            "Fixture target"
                            if predicted_key == TARGET
                            else "Fixture competitor"
                        ),
                        "raw_margin": raw_margin,
                        "target_rank": 1 if target_scoreable else None,
                        "provider_label_rank": (
                            1
                            if predicted_key
                            == (TARGET if provider_target else "gbif:competitor")
                            else 2
                        ),
                        "candidate_count": candidate_count,
                        "availability_status": available,
                        "score_semantics": (
                            "experimental_screening_evidence_uncalibrated_not_probability"
                        ),
                        "target_is_provider_label": provider_target,
                        "classification_accuracy_permitted": False,
                    }
                )
    return pl.DataFrame(rows).with_columns(
        pl.col("target_rank").cast(pl.UInt32),
        pl.col("provider_label_rank").cast(pl.UInt32),
        pl.col("candidate_count").cast(pl.UInt32),
    )


def _embeddings() -> pl.DataFrame:
    splits = (
        ["support_train"] * 26
        + ["model_selection"] * 30
        + ["calibration"] * 13
        + ["final_test"] * 12
    )
    return pl.DataFrame(
        {
            "reference_media_id": [
                f"reference-media:{index:03d}" for index in range(81)
            ],
            "reference_bank_version": ["fixture-bank-v1"] * 81,
            "support_manifest_fingerprint": [_fingerprint("support")] * 81,
            "model_id": ["fixture-model"] * 81,
            "model_revision": ["fixture-revision"] * 81,
            "model_weights_sha256": [_fingerprint("weights")] * 81,
            "preprocessing_version": ["fixture-preprocessing-v1"] * 81,
            "preprocessing_fingerprint": [_fingerprint("preprocessing")] * 81,
            "human_verified": [False] * 81,
            "dataset_split": splits,
        }
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()
