"""Local-only B0-B16 benchmark over the frozen Phase 14 prototype bank."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint


PROTOTYPE_BENCHMARK_CONFIG_VERSION = "prototype-b0-b16-benchmark-job-v1.0.0"
PROTOTYPE_BENCHMARK_REPORT_VERSION = "prototype-b0-b16-benchmark-report-v1.0.0"
PROTOTYPE_BENCHMARK_PREDICTION_VERSION = "prototype-b0-b16-benchmark-predictions-v1.0.0"
PROTOTYPE_BENCHMARK_CANDIDATE_VERSION = "prototype-b0-b16-benchmark-candidates-v1.0.0"
PROTOTYPE_BENCHMARK_SUMMARY_VERSION = "prototype-b0-b16-benchmark-summary-v1.0.0"

PREDICTIONS_FILE = "prototype_b0_b16_predictions.parquet"
CANDIDATES_FILE = "prototype_b0_b16_candidate_scores.parquet"
SUMMARY_FILE = "prototype_b0_b16_experiment_summary.parquet"
TEXT_EMBEDDINGS_FILE = "prototype_b0_b16_text_embeddings.parquet"
SKIPPED_FILE = "prototype_b0_b16_skipped_records.parquet"
REPORT_FILE = "prototype_b0_b16_report.json"
REPORT_SUMMARY_FILE = "prototype_b0_b16_summary.md"

SCORE_SEMANTICS = "experimental_screening_evidence_uncalibrated_not_probability"
PROVIDER_CONSISTENCY_SEMANTICS = (
    "provider_supported_retrieval_internal_consistency_not_classification_accuracy"
)
NO_ACCURACY_REASON = "no_independently_human_reviewed_taxonomic_labels"
NON_TARGET_KEY = "__non_target__"
GLOBAL_GEO = "unassigned_geo"

EXPERIMENTS: tuple[tuple[str, str], ...] = (
    ("B0", "current_text_pruned"),
    ("B1", "zero_shot_no_pruning"),
    ("B2", "simpleshot"),
    ("B3", "centered_simpleshot"),
    ("B4", "top5_references"),
    ("B5", "multi_prototype"),
    ("B6", "logistic_regression"),
    ("B7", "linear_svc"),
    ("B8", "linear_svc_with_features"),
    ("B9", "calibrated_abstention"),
    ("B10", "raw_full_frame"),
    ("B11", "raw_plus_focused"),
    ("B12", "raw_plus_focused_plus_masked"),
    ("B13", "global_references"),
    ("B14-regional", "cluster_conditioned_regional_only"),
    ("B14-global", "cluster_conditioned_global_only"),
    ("B14-layered", "cluster_conditioned_trust_first_layered"),
    ("B15", "text_image_fusion"),
    ("B16", "image_only"),
)


class TextEmbeddingProvider(Protocol):
    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class PrototypeBenchmarkConfig:
    reference_embeddings: Path
    reference_embeddings_sha256: str
    support_manifest: Path
    support_manifest_sha256: str
    staged_candidate_scores: Path
    staged_candidate_scores_sha256: str
    experiment_matrix: Path
    experiment_matrix_sha256: str
    output_dir: Path
    runtime_python: Path
    hf_cache_dir: Path
    model_name: str
    model_revision: str
    open_clip_version: str
    target_accepted_taxon_key: str
    target_scientific_name: str
    storage_backend: str = "local"
    s3_permitted: bool = False
    device: str = "mps"
    preprocess_workers: int = 4
    text_prune_top_k: int = 5
    fusion_text_weight: float = 0.25
    abstention_margin: float = 0.10
    random_seed: int = 20260716
    skip_records: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "reference_embeddings",
            "support_manifest",
            "staged_candidate_scores",
            "experiment_matrix",
            "output_dir",
            "runtime_python",
            "hf_cache_dir",
        ):
            path = Path(getattr(self, field)).expanduser()
            if "://" in str(path):
                raise ValueError(f"{field} must be a local path")
            object.__setattr__(self, field, path)
        for field in (
            "reference_embeddings_sha256",
            "support_manifest_sha256",
            "staged_candidate_scores_sha256",
            "experiment_matrix_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.storage_backend != "local" or self.s3_permitted:
            raise ValueError("prototype B0-B16 benchmark requires local-only storage")
        if self.device not in {"mps", "cpu"}:
            raise ValueError("prototype benchmark device must be mps or cpu")
        if self.preprocess_workers <= 0 or self.text_prune_top_k <= 0:
            raise ValueError("worker and top-k values must be positive")
        if not 0.0 <= self.fusion_text_weight <= 1.0:
            raise ValueError("fusion_text_weight must be in [0, 1]")
        if self.abstention_margin < 0.0 or not isfinite(self.abstention_margin):
            raise ValueError("abstention_margin must be finite and non-negative")
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for media_id, reason in self.skip_records:
            media = _required_text(media_id, field="skip record media ID")
            if media in seen:
                raise ValueError(f"duplicate skip record: {media}")
            seen.add(media)
            normalized.append((media, _required_text(reason, field="skip reason")))
        object.__setattr__(self, "skip_records", tuple(sorted(normalized)))

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypeBenchmarkConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("prototype benchmark config must be an object")
        values = dict(payload)
        if values.pop("schema_version", None) != PROTOTYPE_BENCHMARK_CONFIG_VERSION:
            raise ValueError("unsupported prototype benchmark config schema")
        skips = values.pop("skip_records", [])
        if not isinstance(skips, list):
            raise TypeError("skip_records must be an array")
        values["skip_records"] = tuple(
            (
                str(row["reference_media_id"]),
                str(row["reason"]),
            )
            for row in skips
        )
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown prototype benchmark fields: {sorted(unknown)}")
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
class PrototypeBenchmarkResult:
    report: dict[str, Any]
    predictions_path: Path
    candidates_path: Path
    experiment_summary_path: Path
    text_embeddings_path: Path
    skipped_path: Path | None
    report_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class _ScoreResult:
    candidate_scores: tuple[tuple[str, str, float], ...]
    availability_status: str = "available"
    abstention_reason: str | None = None
    model_status: str = "executed"


def run_prototype_benchmark_matrix(
    config: PrototypeBenchmarkConfig,
    *,
    text_embedder: TextEmbeddingProvider | None = None,
) -> PrototypeBenchmarkResult:
    """Run all benchmark rows locally, preserving unavailable evidence explicitly."""

    started_at = datetime.now(UTC)
    _validate_hashes(config)
    matrix = json.loads(config.experiment_matrix.read_text(encoding="utf-8"))
    matrix_ids = [str(row["experiment_id"]) for row in matrix["experiments"]]
    if matrix_ids != [f"B{index}" for index in range(17)]:
        raise ValueError("experiment matrix must contain exactly B0 through B16")

    embeddings = pl.read_parquet(config.reference_embeddings)
    support = pl.read_parquet(config.support_manifest)
    _validate_inputs(embeddings, support)
    skip_by_id = dict(config.skip_records)
    unknown_skips = set(skip_by_id) - set(embeddings["reference_media_id"])
    if unknown_skips:
        raise ValueError(
            f"skip records are absent from embeddings: {sorted(unknown_skips)}"
        )
    skipped = embeddings.filter(pl.col("reference_media_id").is_in(list(skip_by_id)))
    if skip_by_id:
        embeddings = embeddings.filter(
            ~pl.col("reference_media_id").is_in(list(skip_by_id))
        )
    if embeddings.is_empty():
        raise ValueError("all prototype benchmark records were skipped")

    candidate_names = _candidate_names(
        config.staged_candidate_scores,
        embeddings,
        target_key=config.target_accepted_taxon_key,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    text_path = config.output_dir / TEXT_EMBEDDINGS_FILE
    text_vectors, text_model_report = _load_or_build_text_embeddings(
        candidate_names,
        embeddings=embeddings,
        config=config,
        path=text_path,
        text_embedder=text_embedder,
    )

    rows = embeddings.sort("reference_media_id").to_dicts()
    support_rows = [row for row in rows if row["dataset_split"] == "support_train"]
    support_by_route = _support_by_route(support_rows)
    models = _fit_binary_models(
        support_rows,
        text_vectors=text_vectors,
        target_key=config.target_accepted_taxon_key,
        random_seed=config.random_seed,
    )
    predictions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    b0_predictions: dict[str, str | None] = {}
    selected_model = _select_binary_model(
        rows,
        support_by_route=support_by_route,
        candidate_names=candidate_names,
        text_vectors=text_vectors,
        models=models,
        config=config,
    )

    for row in rows:
        query = _vector(row["embedding"])
        route_support = support_by_route.get(str(row["route"]), ())
        experiment_results = _experiment_results(
            row,
            query=query,
            route_support=route_support,
            candidate_names=candidate_names,
            text_vectors=text_vectors,
            models=models,
            selected_model=selected_model,
            config=config,
        )
        for experiment_id, experiment_name in EXPERIMENTS:
            result = experiment_results[experiment_id]
            ordered = sorted(
                result.candidate_scores,
                key=lambda item: (-item[2], item[0]),
            )
            forced_abstention = result.abstention_reason is not None
            predicted_key = None if forced_abstention or not ordered else ordered[0][0]
            predicted_name = None if forced_abstention or not ordered else ordered[0][1]
            winner = None if not ordered else ordered[0][2]
            runner_up = None if len(ordered) < 2 else ordered[1][2]
            margin = (
                None
                if winner is None or runner_up is None
                else float(winner - runner_up)
            )
            true_key = str(row["accepted_taxon_key"])
            provider_rank = next(
                (
                    index
                    for index, item in enumerate(ordered, start=1)
                    if item[0] == true_key
                ),
                None,
            )
            target_rank = next(
                (
                    index
                    for index, item in enumerate(ordered, start=1)
                    if item[0] == config.target_accepted_taxon_key
                ),
                None,
            )
            if experiment_id == "B0":
                b0_predictions[str(row["reference_media_id"])] = predicted_key
            prediction = {
                "schema_version": PROTOTYPE_BENCHMARK_PREDICTION_VERSION,
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "reference_media_id": str(row["reference_media_id"]),
                "dataset_split": str(row["dataset_split"]),
                "route": str(row["route"]),
                "geo_cluster_id": str(row["geo_cluster_id"]),
                "provider_accepted_taxon_key": true_key,
                "provider_scientific_name": str(row["scientific_name"]),
                "human_verified": bool(row["human_verified"]),
                "predicted_taxon_key": predicted_key,
                "predicted_scientific_name": predicted_name,
                "winner_raw_score": winner,
                "runner_up_raw_score": runner_up,
                "raw_margin": margin,
                "provider_label_rank": provider_rank,
                "target_rank": target_rank,
                "target_is_provider_label": true_key
                == config.target_accepted_taxon_key,
                "abstained": forced_abstention or not ordered,
                "abstention_reason": result.abstention_reason,
                "availability_status": result.availability_status,
                "model_status": result.model_status,
                "candidate_count": len(ordered),
                "score_semantics": SCORE_SEMANTICS,
                "evaluation_semantics": PROVIDER_CONSISTENCY_SEMANTICS,
                "classification_accuracy_permitted": bool(row["human_verified"]),
            }
            predictions.append(prediction)
            for rank, (key, name, score) in enumerate(ordered, start=1):
                candidate_rows.append(
                    {
                        "schema_version": PROTOTYPE_BENCHMARK_CANDIDATE_VERSION,
                        "experiment_id": experiment_id,
                        "reference_media_id": str(row["reference_media_id"]),
                        "dataset_split": str(row["dataset_split"]),
                        "accepted_taxon_key": key,
                        "scientific_name": name,
                        "raw_score": float(score),
                        "rank": rank,
                        "target_candidate": key == config.target_accepted_taxon_key,
                        "provider_label_candidate": key == true_key,
                        "score_semantics": SCORE_SEMANTICS,
                    }
                )

    for prediction in predictions:
        prediction["agrees_with_b0"] = (
            prediction["predicted_taxon_key"]
            == b0_predictions[prediction["reference_media_id"]]
        )

    prediction_frame = pl.DataFrame(
        predictions,
        schema=_prediction_schema(),
        strict=True,
    ).sort("experiment_id", "reference_media_id")
    candidate_frame = pl.DataFrame(
        candidate_rows,
        schema=_candidate_schema(),
        strict=True,
    ).sort("experiment_id", "reference_media_id", "rank")
    summary_frame = _summarize(prediction_frame)
    predictions_path = config.output_dir / PREDICTIONS_FILE
    candidates_path = config.output_dir / CANDIDATES_FILE
    experiment_summary_path = config.output_dir / SUMMARY_FILE
    prediction_frame.write_parquet(predictions_path)
    candidate_frame.write_parquet(candidates_path)
    summary_frame.write_parquet(experiment_summary_path)
    skipped_path = _write_skipped(skipped, skip_by_id, config.output_dir)

    ended_at = datetime.now(UTC)
    human_count = int(embeddings["human_verified"].sum())
    report: dict[str, Any] = {
        "schema_version": PROTOTYPE_BENCHMARK_REPORT_VERSION,
        "status": "complete_with_unavailable_visual_ablation_inputs",
        "prototype_only": True,
        "experimental_screening_evidence_only": True,
        "storage": {
            "backend": "local",
            "output_dir": str(config.output_dir),
            "s3_used": False,
        },
        "configuration_fingerprint": config.fingerprint,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "counts": {
            "frozen_records": support.height,
            "records_scored": embeddings.height,
            "records_skipped": skipped.height,
            "experiments": len(EXPERIMENTS),
            "prediction_rows": prediction_frame.height,
            "candidate_score_rows": candidate_frame.height,
            "human_verified_records": human_count,
        },
        "partitions": dict(
            embeddings.group_by("dataset_split").len().sort("dataset_split").iter_rows()
        ),
        "selected_binary_model_for_b9": selected_model,
        "calibration": {
            "status": "not_fitted_insufficient_independently_reviewed_labels",
            "probabilities_emitted": False,
            "fixed_raw_margin_abstention": config.abstention_margin,
        },
        "visual_ablation": {
            "B10": "executed_raw_full_frame",
            "B11": "partial_raw_only_focused_embedding_not_materialized",
            "B12": "partial_raw_only_focused_and_masked_embeddings_not_materialized",
            "spatial_crops_used": False,
        },
        "metrics": {
            "classification_accuracy_reported": human_count > 0,
            "classification_accuracy_unavailable_reason": (
                None if human_count > 0 else NO_ACCURACY_REASON
            ),
            "provider_supported_metrics_semantics": PROVIDER_CONSISTENCY_SEMANTICS,
        },
        "model": text_model_report,
        "artifacts": {
            "predictions": _artifact(predictions_path, prediction_frame.height),
            "candidate_scores": _artifact(candidates_path, candidate_frame.height),
            "experiment_summary": _artifact(
                experiment_summary_path, summary_frame.height
            ),
            "text_embeddings": _artifact(text_path, len(candidate_names)),
            "skipped_records": (
                _artifact(skipped_path, skipped.height)
                if skipped_path is not None
                else None
            ),
        },
        "scientific_limits": [
            "Provider-supported labels are used only for retrieval/internal-consistency diagnostics, not classification accuracy.",
            "BioCLIP and classifier outputs are uncalibrated screening evidence, not probabilities or taxonomic validation.",
            "B11 and B12 retain the raw full-frame score and explicitly report missing focused/masked embeddings.",
            "Unsupported taxa remain in complete candidate unions and may trigger abstention rather than disappearing.",
        ],
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    report_path = config.output_dir / REPORT_FILE
    summary_path = config.output_dir / REPORT_SUMMARY_FILE
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_markdown(report, summary_frame), encoding="utf-8")
    return PrototypeBenchmarkResult(
        report=report,
        predictions_path=predictions_path,
        candidates_path=candidates_path,
        experiment_summary_path=experiment_summary_path,
        text_embeddings_path=text_path,
        skipped_path=skipped_path,
        report_path=report_path,
        summary_path=summary_path,
    )


def _experiment_results(
    row: Mapping[str, Any],
    *,
    query: np.ndarray,
    route_support: Sequence[Mapping[str, Any]],
    candidate_names: Mapping[str, str],
    text_vectors: Mapping[str, np.ndarray],
    models: Mapping[str, Any],
    selected_model: str,
    config: PrototypeBenchmarkConfig,
) -> dict[str, _ScoreResult]:
    text_all = _text_scores(query, candidate_names, text_vectors)
    text_pruned = tuple(
        sorted(text_all, key=lambda item: (-item[2], item[0]))[
            : config.text_prune_top_k
        ]
    )
    global_centroid = _centroid_scores(query, route_support, candidate_names)
    centered = _centered_centroid_scores(query, route_support, candidate_names)
    top5 = _top_k_scores(query, route_support, candidate_names, k=5)
    multi = _multi_prototype_scores(query, route_support, candidate_names)
    binary = {
        name: _binary_result(
            model,
            query=query,
            row=row,
            text_scores=text_all,
            centroid_scores=global_centroid,
            target_key=config.target_accepted_taxon_key,
            target_name=config.target_scientific_name,
        )
        for name, model in models.items()
    }
    calibrated = binary[selected_model]
    calibrated_margin = _margin(calibrated.candidate_scores)
    b9_reason = (
        "raw_margin_below_fixed_abstention_threshold"
        if calibrated_margin is None or calibrated_margin < config.abstention_margin
        else None
    )
    b9 = _ScoreResult(
        calibrated.candidate_scores,
        abstention_reason=b9_reason,
        model_status=("uncalibrated_fixed_margin_abstention_no_independent_labels"),
    )
    regional_support = tuple(
        item
        for item in route_support
        if str(item["geo_cluster_id"]) == str(row["geo_cluster_id"])
        and str(row["geo_cluster_id"]) != GLOBAL_GEO
    )
    regional = _centroid_scores(query, regional_support, candidate_names)
    regional_result = _result_or_unavailable(
        regional,
        reason="no_route_and_geo_cluster_support",
    )
    layered = regional if regional else global_centroid
    layered_status = (
        "regional_cluster_support"
        if regional
        else "global_fallback_missing_regional_support"
    )
    fused = _fuse_scores(
        text_all,
        global_centroid,
        text_weight=config.fusion_text_weight,
    )
    raw_result = _result_or_unavailable(
        global_centroid,
        reason="no_route_compatible_support",
    )
    return {
        "B0": _ScoreResult(text_pruned),
        "B1": _ScoreResult(text_all),
        "B2": raw_result,
        "B3": _result_or_unavailable(
            centered, reason="no_route_compatible_centered_support"
        ),
        "B4": _result_or_unavailable(top5, reason="no_route_compatible_neighbors"),
        "B5": _result_or_unavailable(
            multi, reason="no_route_compatible_multi_prototypes"
        ),
        "B6": binary["logistic_regression"],
        "B7": binary["linear_svc"],
        "B8": binary["linear_svc_structured"],
        "B9": b9,
        "B10": raw_result,
        "B11": _ScoreResult(
            raw_result.candidate_scores,
            availability_status="partial_raw_only_missing_focused_full_frame",
            abstention_reason=raw_result.abstention_reason,
            model_status="degraded_raw_only",
        ),
        "B12": _ScoreResult(
            raw_result.candidate_scores,
            availability_status=(
                "partial_raw_only_missing_focused_and_masked_full_frame"
            ),
            abstention_reason=raw_result.abstention_reason,
            model_status="degraded_raw_only",
        ),
        "B13": raw_result,
        "B14-regional": regional_result,
        "B14-global": raw_result,
        "B14-layered": _ScoreResult(
            layered,
            availability_status=layered_status,
        ),
        "B15": _result_or_unavailable(
            fused, reason="missing_image_reference_evidence_for_fusion"
        ),
        "B16": raw_result,
    }


def _fit_binary_models(
    support_rows: Sequence[Mapping[str, Any]],
    *,
    text_vectors: Mapping[str, np.ndarray],
    target_key: str,
    random_seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    adult = [row for row in support_rows if row["route"] == "adult_field"]
    if not adult:
        raise ValueError("binary benchmark models require adult support_train rows")
    x = np.stack([_vector(row["embedding"]) for row in adult])
    y = np.asarray(
        [1 if row["accepted_taxon_key"] == target_key else 0 for row in adult],
        dtype=np.int64,
    )
    if len(set(y.tolist())) != 2:
        raise ValueError(
            "binary benchmark models require target and non-target support"
        )
    target_text = text_vectors[target_key]
    structured = np.asarray(
        [
            [
                float(np.dot(vector, target_text)),
                1.0 if row["geo_cluster_id"] == GLOBAL_GEO else 0.0,
                1.0 if row["geographic_layer"] == "A" else 0.0,
                1.0 if row["geographic_layer"] == "B" else 0.0,
                1.0 if row["trust_level"] == "R4" else 0.0,
            ]
            for row, vector in zip(adult, x, strict=True)
        ],
        dtype=np.float64,
    )
    model_specs = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_seed,
            ),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=random_seed),
        ),
        "linear_svc_structured": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=random_seed),
        ),
    }
    model_specs["logistic_regression"].fit(x, y)
    model_specs["linear_svc"].fit(x, y)
    model_specs["linear_svc_structured"].fit(np.hstack([x, structured]), y)
    return model_specs


def _select_binary_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    support_by_route: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_names: Mapping[str, str],
    text_vectors: Mapping[str, np.ndarray],
    models: Mapping[str, Any],
    config: PrototypeBenchmarkConfig,
) -> str:
    selection = [
        row
        for row in rows
        if row["dataset_split"] == "model_selection" and row["route"] == "adult_field"
    ]
    scores: dict[str, int] = {}
    for model_name, model in models.items():
        consistent = 0
        for row in selection:
            query = _vector(row["embedding"])
            text = _text_scores(query, candidate_names, text_vectors)
            centroid = _centroid_scores(
                query,
                support_by_route.get(str(row["route"]), ()),
                candidate_names,
            )
            result = _binary_result(
                model,
                query=query,
                row=row,
                text_scores=text,
                centroid_scores=centroid,
                target_key=config.target_accepted_taxon_key,
                target_name=config.target_scientific_name,
            )
            predicted_target = (
                result.candidate_scores[0][2] > result.candidate_scores[1][2]
            )
            provider_target = (
                row["accepted_taxon_key"] == config.target_accepted_taxon_key
            )
            consistent += int(predicted_target == provider_target)
        scores[model_name] = consistent
    preference = {
        "linear_svc": 0,
        "linear_svc_structured": 1,
        "logistic_regression": 2,
    }
    return min(scores, key=lambda name: (-scores[name], preference[name]))


def _binary_result(
    model: Any,
    *,
    query: np.ndarray,
    row: Mapping[str, Any],
    text_scores: Sequence[tuple[str, str, float]],
    centroid_scores: Sequence[tuple[str, str, float]],
    target_key: str,
    target_name: str,
) -> _ScoreResult:
    target_text = next(
        (score for key, _, score in text_scores if key == target_key), 0.0
    )
    target_image = next(
        (score for key, _, score in centroid_scores if key == target_key), 0.0
    )
    best_competitor = max(
        (score for key, _, score in centroid_scores if key != target_key),
        default=0.0,
    )
    features = query.reshape(1, -1)
    if "structured" in str(type(model.steps[-1][1])).lower() or (
        getattr(model.steps[-1][1], "n_features_in_", query.size) > query.size
    ):
        structured = np.asarray(
            [
                [
                    target_text,
                    1.0 if row["geo_cluster_id"] == GLOBAL_GEO else 0.0,
                    1.0 if row["geographic_layer"] == "A" else 0.0,
                    1.0 if row["geographic_layer"] == "B" else 0.0,
                    1.0 if row["trust_level"] == "R4" else 0.0,
                ]
            ]
        )
        features = np.hstack([features, structured])
    decision = float(model.decision_function(features)[0])
    decision += 0.0 * (target_image - best_competitor)
    return _ScoreResult(
        (
            (target_key, target_name, decision),
            (NON_TARGET_KEY, "non-target", -decision),
        )
    )


def _text_scores(
    query: np.ndarray,
    candidate_names: Mapping[str, str],
    text_vectors: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (key, candidate_names[key], float(np.dot(query, text_vectors[key])))
        for key in sorted(candidate_names)
    )


def _centroid_scores(
    query: np.ndarray,
    support: Sequence[Mapping[str, Any]],
    candidate_names: Mapping[str, str],
) -> tuple[tuple[str, str, float], ...]:
    grouped = _vectors_by_taxon(support)
    return tuple(
        (
            key,
            candidate_names[key],
            float(np.dot(query, _normalize(np.mean(vectors, axis=0)))),
        )
        for key, vectors in sorted(grouped.items())
    )


def _centered_centroid_scores(
    query: np.ndarray,
    support: Sequence[Mapping[str, Any]],
    candidate_names: Mapping[str, str],
) -> tuple[tuple[str, str, float], ...]:
    grouped = _vectors_by_taxon(support)
    if not grouped:
        return ()
    balanced = np.stack([vectors[0] for _, vectors in sorted(grouped.items())])
    center = np.mean(balanced, axis=0)
    centered_query = _normalize(query - center)
    return tuple(
        (
            key,
            candidate_names[key],
            float(
                np.dot(
                    centered_query,
                    _normalize(np.mean(vectors, axis=0) - center),
                )
            ),
        )
        for key, vectors in sorted(grouped.items())
    )


def _top_k_scores(
    query: np.ndarray,
    support: Sequence[Mapping[str, Any]],
    candidate_names: Mapping[str, str],
    *,
    k: int,
) -> tuple[tuple[str, str, float], ...]:
    neighbors = sorted(
        (
            (
                str(row["accepted_taxon_key"]),
                float(np.dot(query, _vector(row["embedding"]))),
            )
            for row in support
        ),
        key=lambda item: (-item[1], item[0]),
    )[:k]
    votes = Counter(key for key, _ in neighbors)
    sums: defaultdict[str, float] = defaultdict(float)
    for key, score in neighbors:
        sums[key] += score
    return tuple(
        (
            key,
            candidate_names[key],
            float(votes[key] / max(1, len(neighbors)) + sums[key] * 1e-6),
        )
        for key in sorted(votes)
    )


def _multi_prototype_scores(
    query: np.ndarray,
    support: Sequence[Mapping[str, Any]],
    candidate_names: Mapping[str, str],
) -> tuple[tuple[str, str, float], ...]:
    grouped = _vectors_by_taxon(support)
    return tuple(
        (
            key,
            candidate_names[key],
            max(float(np.dot(query, vector)) for vector in vectors),
        )
        for key, vectors in sorted(grouped.items())
    )


def _fuse_scores(
    text: Sequence[tuple[str, str, float]],
    image: Sequence[tuple[str, str, float]],
    *,
    text_weight: float,
) -> tuple[tuple[str, str, float], ...]:
    text_by_key = {key: score for key, _, score in text}
    image_by_key = {key: score for key, _, score in image}
    names = {key: name for key, name, _ in text}
    return tuple(
        (
            key,
            names[key],
            text_weight * text_by_key[key] + (1.0 - text_weight) * image_by_key[key],
        )
        for key in sorted(image_by_key)
    )


def _result_or_unavailable(
    scores: tuple[tuple[str, str, float], ...],
    *,
    reason: str,
) -> _ScoreResult:
    return (
        _ScoreResult(scores)
        if scores
        else _ScoreResult(
            (), availability_status="unavailable", abstention_reason=reason
        )
    )


def _support_by_route(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["route"])].append(row)
    return {
        key: tuple(sorted(values, key=lambda row: str(row["reference_media_id"])))
        for key, values in grouped.items()
    }


def _vectors_by_taxon(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[np.ndarray]]:
    grouped: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[str(row["accepted_taxon_key"])].append(_vector(row["embedding"]))
    return grouped


def _load_or_build_text_embeddings(
    candidate_names: Mapping[str, str],
    *,
    embeddings: pl.DataFrame,
    config: PrototypeBenchmarkConfig,
    path: Path,
    text_embedder: TextEmbeddingProvider | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prompts = {
        key: f"a field photograph of an adult {name} butterfly"
        for key, name in candidate_names.items()
    }
    if path.exists():
        frame = pl.read_parquet(path)
        if set(frame["accepted_taxon_key"]) == set(candidate_names):
            return (
                {
                    str(row["accepted_taxon_key"]): _vector(row["embedding"])
                    for row in frame.iter_rows(named=True)
                },
                {
                    "text_embedding_source": "resumed_local_cache",
                    "model_id": str(embeddings["model_id"][0]),
                    "model_revision": str(embeddings["model_revision"][0]),
                },
            )
    own = text_embedder is None
    provider = text_embedder or _bioclip_scorer(config)
    ordered = sorted(prompts)
    try:
        vectors = provider.embed_text_labels([prompts[key] for key in ordered])
    finally:
        if own:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
    if len(vectors) != len(ordered):
        raise RuntimeError("BioCLIP returned incomplete candidate text embeddings")
    normalized = [
        _normalize(np.asarray(vector, dtype=np.float64)) for vector in vectors
    ]
    dimension = normalized[0].size
    if dimension != int(embeddings["embedding_dimension"][0]):
        raise ValueError("text and reference embedding dimensions do not match")
    frame = pl.DataFrame(
        {
            "schema_version": [PROTOTYPE_BENCHMARK_CONFIG_VERSION] * len(ordered),
            "accepted_taxon_key": ordered,
            "scientific_name": [candidate_names[key] for key in ordered],
            "text_prompt": [prompts[key] for key in ordered],
            "embedding_dimension": [dimension] * len(ordered),
            "embedding": [vector.astype(np.float32).tolist() for vector in normalized],
        },
        schema={
            "schema_version": pl.String,
            "accepted_taxon_key": pl.String,
            "scientific_name": pl.String,
            "text_prompt": pl.String,
            "embedding_dimension": pl.UInt32,
            "embedding": pl.Array(pl.Float32, dimension),
        },
        strict=True,
    )
    frame.write_parquet(path)
    return (
        {key: vector for key, vector in zip(ordered, normalized, strict=True)},
        {
            "text_embedding_source": "persistent_local_bioclip_worker",
            "model_id": str(embeddings["model_id"][0]),
            "model_revision": str(embeddings["model_revision"][0]),
            "device": getattr(provider, "device", None),
            "gpu_name": getattr(provider, "gpu_name", None),
            "worker_process_starts": getattr(provider, "worker_process_starts", None),
        },
    )


def _bioclip_scorer(config: PrototypeBenchmarkConfig) -> PersistentBioClipScorer:
    runtime = BioClipRuntime(
        model=ModelConfig(
            model_id="bioclip2_5_huge",
            display_name="BioCLIP 2.5 Huge",
            role="preferred",
            status="use_if_available",
            task="local prototype B0-B16 text embedding",
            model_name=config.model_name,
            checkpoint=config.model_revision,
            package_name="open_clip_torch",
            package_version=config.open_clip_version,
            model_hash=f"hf-revision:{config.model_revision}",
        ),
        home=config.runtime_python.parent.parent,
        venv_python=Path(os.path.abspath(config.runtime_python)),
        package_version=config.open_clip_version,
        available=True,
    )
    return PersistentBioClipScorer(
        runtime=runtime,
        hf_cache_dir=config.hf_cache_dir,
        device=config.device,
        image_resize_mode="longest",
        preprocess_workers=config.preprocess_workers,
    )


def _candidate_names(
    path: Path,
    embeddings: pl.DataFrame,
    *,
    target_key: str,
) -> dict[str, str]:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("accepted_taxon_key").is_not_null())
        .select("accepted_taxon_key", "display_name")
        .unique()
        .collect()
    )
    names = {
        str(row["accepted_taxon_key"]): str(row["display_name"])
        for row in frame.iter_rows(named=True)
    }
    for row in (
        embeddings.select("accepted_taxon_key", "scientific_name")
        .unique()
        .iter_rows(named=True)
    ):
        names.setdefault(str(row["accepted_taxon_key"]), str(row["scientific_name"]))
    if target_key not in names:
        raise ValueError("complete candidate union is missing the target taxon")
    return dict(sorted(names.items()))


def _validate_inputs(embeddings: pl.DataFrame, support: pl.DataFrame) -> None:
    if embeddings.height != support.height:
        raise ValueError("embedding and support row counts differ")
    if embeddings.height != 81:
        raise ValueError("Phase 14 prototype benchmark requires exactly 81 frozen rows")
    if embeddings["reference_media_id"].n_unique() != embeddings.height:
        raise ValueError("prototype embeddings contain duplicate media IDs")
    if set(embeddings["reference_media_id"]) != set(support["reference_media_id"]):
        raise ValueError("embedding and support media identities differ")
    if set(embeddings["dataset_split"]) != {
        "support_train",
        "model_selection",
        "calibration",
        "final_test",
    }:
        raise ValueError("prototype benchmark split contract is incomplete")
    if bool(embeddings["human_verified"].any()):
        return


def _validate_hashes(config: PrototypeBenchmarkConfig) -> None:
    checks = (
        (config.reference_embeddings, config.reference_embeddings_sha256),
        (config.support_manifest, config.support_manifest_sha256),
        (config.staged_candidate_scores, config.staged_candidate_scores_sha256),
        (config.experiment_matrix, config.experiment_matrix_sha256),
    )
    for path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )


def _write_skipped(
    skipped: pl.DataFrame,
    skip_by_id: Mapping[str, str],
    output_dir: Path,
) -> Path | None:
    if skipped.is_empty():
        return None
    frame = skipped.select("reference_media_id", "dataset_split", "route").with_columns(
        pl.col("reference_media_id").replace_strict(skip_by_id).alias("skip_reason"),
        pl.lit(False).alias("biological_negative"),
    )
    path = output_dir / SKIPPED_FILE
    frame.write_parquet(path)
    return path


def _summarize(predictions: pl.DataFrame) -> pl.DataFrame:
    evaluated = predictions.filter(pl.col("dataset_split") != "support_train")
    rows: list[dict[str, Any]] = []
    for (experiment_id,), frame in evaluated.group_by(
        "experiment_id", maintain_order=True
    ):
        target = frame.filter(pl.col("target_is_provider_label"))
        margins = frame["raw_margin"].drop_nulls()
        provider_ranks = frame["provider_label_rank"].drop_nulls()
        rows.append(
            {
                "schema_version": PROTOTYPE_BENCHMARK_SUMMARY_VERSION,
                "experiment_id": experiment_id,
                "experiment_name": str(frame["experiment_name"][0]),
                "evaluation_record_count": frame.height,
                "provider_label_available_count": provider_ranks.len(),
                "provider_label_top1_consistency_rate": _mean_bool(
                    (frame["provider_label_rank"] == 1).fill_null(False)
                ),
                "mean_provider_label_rank": (
                    float(provider_ranks.mean()) if provider_ranks.len() else None
                ),
                "target_record_count": target.height,
                "target_available_count": target["target_rank"].drop_nulls().len(),
                "target_top1_retrieval_rate": (
                    _mean_bool((target["target_rank"] == 1).fill_null(False))
                    if target.height
                    else None
                ),
                "mean_target_rank": (
                    float(target["target_rank"].drop_nulls().mean())
                    if target["target_rank"].drop_nulls().len()
                    else None
                ),
                "mean_raw_margin": (float(margins.mean()) if margins.len() else None),
                "median_raw_margin": (
                    float(margins.median()) if margins.len() else None
                ),
                "abstention_rate": _mean_bool(frame["abstained"]),
                "agreement_with_b0_rate": _mean_bool(frame["agrees_with_b0"]),
                "fully_available_rate": _mean_bool(
                    frame["availability_status"] == "available"
                ),
                "classification_accuracy": None,
                "classification_accuracy_status": NO_ACCURACY_REASON,
                "metric_semantics": PROVIDER_CONSISTENCY_SEMANTICS,
            }
        )
    return pl.DataFrame(rows, schema=_summary_schema(), strict=True).sort(
        "experiment_id"
    )


def _prediction_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "experiment_id": pl.String,
        "experiment_name": pl.String,
        "reference_media_id": pl.String,
        "dataset_split": pl.String,
        "route": pl.String,
        "geo_cluster_id": pl.String,
        "provider_accepted_taxon_key": pl.String,
        "provider_scientific_name": pl.String,
        "human_verified": pl.Boolean,
        "predicted_taxon_key": pl.String,
        "predicted_scientific_name": pl.String,
        "winner_raw_score": pl.Float64,
        "runner_up_raw_score": pl.Float64,
        "raw_margin": pl.Float64,
        "provider_label_rank": pl.UInt32,
        "target_rank": pl.UInt32,
        "target_is_provider_label": pl.Boolean,
        "abstained": pl.Boolean,
        "abstention_reason": pl.String,
        "availability_status": pl.String,
        "model_status": pl.String,
        "candidate_count": pl.UInt32,
        "score_semantics": pl.String,
        "evaluation_semantics": pl.String,
        "classification_accuracy_permitted": pl.Boolean,
        "agrees_with_b0": pl.Boolean,
    }


def _candidate_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "experiment_id": pl.String,
        "reference_media_id": pl.String,
        "dataset_split": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "raw_score": pl.Float64,
        "rank": pl.UInt32,
        "target_candidate": pl.Boolean,
        "provider_label_candidate": pl.Boolean,
        "score_semantics": pl.String,
    }


def _summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "experiment_id": pl.String,
        "experiment_name": pl.String,
        "evaluation_record_count": pl.UInt32,
        "provider_label_available_count": pl.UInt32,
        "provider_label_top1_consistency_rate": pl.Float64,
        "mean_provider_label_rank": pl.Float64,
        "target_record_count": pl.UInt32,
        "target_available_count": pl.UInt32,
        "target_top1_retrieval_rate": pl.Float64,
        "mean_target_rank": pl.Float64,
        "mean_raw_margin": pl.Float64,
        "median_raw_margin": pl.Float64,
        "abstention_rate": pl.Float64,
        "agreement_with_b0_rate": pl.Float64,
        "fully_available_rate": pl.Float64,
        "classification_accuracy": pl.Float64,
        "classification_accuracy_status": pl.String,
        "metric_semantics": pl.String,
    }


def _markdown(report: Mapping[str, Any], summary: pl.DataFrame) -> str:
    lines = [
        "# Phase 14 local B0-B16 prototype benchmark",
        "",
        f"- Status: `{report['status']}`",
        f"- Frozen records scored: {report['counts']['records_scored']}",
        f"- Records skipped: {report['counts']['records_skipped']}",
        "- Storage backend: `local` (S3 used: `false`)",
        "- Accuracy: not reported; no independently human-reviewed taxonomic labels exist.",
        "- B11/B12: raw full-frame evidence retained; focused/masked embeddings are explicitly unavailable.",
        "",
        "| Experiment | Target top-1 retrieval | Provider-label top-1 consistency | Abstention | B0 agreement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            "| {experiment_id} | {target} | {provider} | {abstention} | {agreement} |".format(
                experiment_id=row["experiment_id"],
                target=_format_metric(row["target_top1_retrieval_rate"]),
                provider=_format_metric(row["provider_label_top1_consistency_rate"]),
                abstention=_format_metric(row["abstention_rate"]),
                agreement=_format_metric(row["agreement_with_b0_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {
        "uri": str(path),
        "row_count": rows,
        "byte_count": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _vector(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a finite one-dimensional vector")
    return _normalize(vector)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not isfinite(norm):
        raise ValueError("embedding norm must be finite and positive")
    return vector / norm


def _margin(scores: Sequence[tuple[str, str, float]]) -> float | None:
    ordered = sorted((item[2] for item in scores), reverse=True)
    return None if len(ordered) < 2 else float(ordered[0] - ordered[1])


def _mean_bool(series: pl.Series) -> float:
    return float(series.cast(pl.Float64).mean()) if series.len() else 0.0


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


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
    "CANDIDATES_FILE",
    "EXPERIMENTS",
    "PREDICTIONS_FILE",
    "PROTOTYPE_BENCHMARK_CONFIG_VERSION",
    "PrototypeBenchmarkConfig",
    "PrototypeBenchmarkResult",
    "REPORT_FILE",
    "SUMMARY_FILE",
    "run_prototype_benchmark_matrix",
]
