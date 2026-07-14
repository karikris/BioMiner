from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl

from biominer.run import RunStage


REFERENCE_WORKFLOW_SETTINGS_SCHEMA_VERSION = "target-aware-reference-cli-settings-v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReferenceWorkflowRuntimeDefaults:
    runtime_python: str
    hf_cache_dir: str


@dataclass(frozen=True, slots=True)
class ResolvedReferenceWorkflowOptions:
    command: str
    stage: RunStage
    values: dict[str, Any]
    settings_fingerprint: str
    settings_file: str | None


@dataclass(frozen=True, slots=True)
class _CommandSpec:
    stage: RunStage
    fields: frozenset[str]
    required: frozenset[str]
    defaults: Mapping[str, object]


_COMMAND_SPECS: dict[str, _CommandSpec] = {
    "build-geographic-spread": _CommandSpec(
        stage=RunStage.GEOGRAPHIC_SPREAD,
        fields=frozenset(
            {
                "accepted_taxon_key",
                "scientific_name",
                "registry_version",
                "source_snapshot_version",
                "occurrences",
                "output_dir",
                "checkpoint_dir",
                "retrieved_at",
                "coarse_resolution",
                "regional_resolution",
                "local_resolution",
                "page_size",
                "max_retries",
            }
        ),
        required=frozenset(
            {
                "accepted_taxon_key",
                "scientific_name",
                "registry_version",
                "source_snapshot_version",
                "output_dir",
                "checkpoint_dir",
                "retrieved_at",
            }
        ),
        defaults={
            "coarse_resolution": 3,
            "regional_resolution": 5,
            "local_resolution": 7,
            "page_size": 300,
            "max_retries": 5,
        },
    ),
    "cluster-flickr-metadata": _CommandSpec(
        stage=RunStage.FLICKR_GEO_CLUSTERING,
        fields=frozenset(
            {
                "geography",
                "target_accepted_taxon_key",
                "output_dir",
                "created_at",
                "source_cell_field",
                "source_resolution",
                "adjacency_grid_distance",
                "minimum_images_per_cell",
                "minimum_cluster_images",
                "maximum_assignment_distance_km",
                "bioregion_map",
                "overwrite",
            }
        ),
        required=frozenset({"geography", "target_accepted_taxon_key", "output_dir"}),
        defaults={
            "source_cell_field": "regional_cell_id",
            "source_resolution": 5,
            "adjacency_grid_distance": 1,
            "minimum_images_per_cell": 1,
            "minimum_cluster_images": 2,
            "maximum_assignment_distance_km": 250.0,
            "overwrite": False,
        },
    ),
    "plan": _CommandSpec(
        stage=RunStage.REFERENCE_METADATA,
        fields=frozenset(
            {
                "candidate_species",
                "observations",
                "media_candidates",
                "review_metadata",
                "existing_selections",
                "output_dir",
                "created_at",
                "strata",
                "minimum_per_sufficient_cluster",
                "sufficiently_populated_candidate_count",
                "distance_balance_band_km",
                "selection_seed",
                "licence_policy_version",
                "selection_strategy",
                "eligible_download_statuses",
                "eligible_licence_policy_statuses",
                "overwrite",
            }
        ),
        required=frozenset(
            {"candidate_species", "observations", "media_candidates", "output_dir"}
        ),
        defaults={
            "strata": [
                {
                    "life_stage": "adult",
                    "visual_domain": "unreviewed",
                    "requested_per_species": 20,
                }
            ],
            "minimum_per_sufficient_cluster": 2,
            "sufficiently_populated_candidate_count": 10,
            "distance_balance_band_km": 50.0,
            "selection_seed": 42,
            "licence_policy_version": "reference-licences-v1",
            "selection_strategy": "minimum-sqrt-diversity-v1.0.0",
            "eligible_download_statuses": ["pending", "complete"],
            "eligible_licence_policy_statuses": [
                "allowed",
                "research_only",
                "unreviewed",
            ],
            "overwrite": False,
        },
    ),
    "fetch-metadata": _CommandSpec(
        stage=RunStage.REFERENCE_METADATA,
        fields=frozenset(
            {
                "queries",
                "registry_version",
                "checkpoint_dir",
                "output_dir",
                "max_retries",
                "inaturalist_min_request_interval_seconds",
                "accepted_photo_licences",
                "overwrite",
            }
        ),
        required=frozenset(
            {"queries", "registry_version", "checkpoint_dir", "output_dir"}
        ),
        defaults={
            "max_retries": 5,
            "inaturalist_min_request_interval_seconds": 1.0,
            "overwrite": False,
        },
    ),
    "download": _CommandSpec(
        stage=RunStage.REFERENCE_MEDIA,
        fields=frozenset(
            {
                "acquisition_selections",
                "media_candidates",
                "output_prefix",
                "run_id",
                "storage_backend",
                "storage_prefix",
                "workers",
                "max_inflight",
                "max_concurrent_decodes",
                "max_attempts",
                "timeout_seconds",
                "max_download_seconds",
                "download_config",
                "licence_policy",
            }
        ),
        required=frozenset(
            {"acquisition_selections", "media_candidates", "output_prefix"}
        ),
        defaults={
            "workers": 8,
            "max_inflight": 32,
            "max_concurrent_decodes": 1,
            "max_attempts": 5,
            "timeout_seconds": 30.0,
            "max_download_seconds": 300.0,
        },
    ),
    "build-support-embeddings": _CommandSpec(
        stage=RunStage.REFERENCE_EMBEDDINGS,
        fields=frozenset(
            {
                "readiness_dir",
                "readiness_sha256",
                "support_manifest",
                "visual_inputs",
                "output",
                "runtime_python",
                "hf_cache_dir",
                "device",
                "preprocess_workers",
                "batch_size",
                "checkpoint_dir",
                "run_id",
                "embedding_cache",
                "resume",
                "overwrite",
            }
        ),
        required=frozenset(
            {
                "readiness_dir",
                "readiness_sha256",
                "visual_inputs",
                "output",
            }
        ),
        defaults={
            "device": "auto",
            "preprocess_workers": 1,
            "batch_size": 64,
            "resume": True,
            "overwrite": False,
        },
    ),
    "build-prototypes": _CommandSpec(
        stage=RunStage.REFERENCE_PROTOTYPES,
        fields=frozenset(
            {
                "reference_embeddings",
                "output",
                "balanced_sampling_seed",
                "include_mean_centered",
                "multi_prototype",
                "minimum_metadata_observation_count",
                "enable_embedding_clustering",
                "minimum_clustering_observation_count",
                "minimum_embedding_cluster_size",
                "maximum_embedding_cluster_count",
                "maximum_clustering_observation_count",
                "cosine_distance_threshold",
                "overwrite",
            }
        ),
        required=frozenset({"reference_embeddings", "output"}),
        defaults={
            "balanced_sampling_seed": 42,
            "include_mean_centered": True,
            "multi_prototype": True,
            "minimum_metadata_observation_count": 2,
            "enable_embedding_clustering": True,
            "minimum_clustering_observation_count": 8,
            "minimum_embedding_cluster_size": 3,
            "maximum_embedding_cluster_count": 4,
            "maximum_clustering_observation_count": 256,
            "cosine_distance_threshold": 0.20,
            "overwrite": False,
        },
    ),
    "train-classifier": _CommandSpec(
        stage=RunStage.CLASSIFIER_TRAINING,
        fields=frozenset(
            {
                "training_features",
                "output_dir",
                "target_task",
                "target_accepted_taxon_key",
                "route",
                "n_splits",
                "random_seed",
                "class_weight",
                "included_label_certainties",
                "enabled_models",
                "enable_rbf_pilot",
                "rbf_max_fit_samples",
                "n_jobs",
                "preprocessing_fingerprint",
                "reference_bank_version",
                "reference_bank_fingerprint",
                "git_sha",
                "model_name",
            }
        ),
        required=frozenset(
            {
                "training_features",
                "output_dir",
                "target_task",
                "target_accepted_taxon_key",
                "route",
                "preprocessing_fingerprint",
                "reference_bank_version",
                "reference_bank_fingerprint",
            }
        ),
        defaults={
            "n_splits": 3,
            "random_seed": 42,
            "class_weight": "balanced",
            "included_label_certainties": ["high", "medium"],
            "enabled_models": [
                "logistic_regression_embedding",
                "linear_svc_embedding",
                "linear_svc_embedding_structured",
            ],
            "enable_rbf_pilot": False,
            "rbf_max_fit_samples": 2000,
            "n_jobs": 1,
        },
    ),
    "calibrate-classifier": _CommandSpec(
        stage=RunStage.CLASSIFIER_CALIBRATION,
        fields=frozenset(
            {
                "predictions",
                "fold_audits",
                "output_dir",
                "classifier_fingerprint",
                "split_fingerprint",
                "target_task",
                "route",
                "class_labels",
                "method",
                "positive_class_label",
                "reliability_bin_count",
                "minimum_class_group_count",
                "decision_policy",
                "git_sha",
            }
        ),
        required=frozenset(
            {
                "predictions",
                "fold_audits",
                "output_dir",
                "classifier_fingerprint",
                "split_fingerprint",
                "target_task",
                "route",
                "class_labels",
            }
        ),
        defaults={
            "method": "auto",
            "reliability_bin_count": 10,
            "minimum_class_group_count": 2,
        },
    ),
    "score-target-aware": _CommandSpec(
        stage=RunStage.TARGET_AWARE_SCORING,
        fields=frozenset(
            {
                "candidate_set",
                "known_negative_classes",
                "visual_domain_classes",
                "score_map",
                "output",
            }
        ),
        required=frozenset(
            {
                "candidate_set",
                "known_negative_classes",
                "visual_domain_classes",
                "score_map",
                "output",
            }
        ),
        defaults={},
    ),
    "evaluate-target-verifier": _CommandSpec(
        stage=RunStage.EVALUATION,
        fields=frozenset(
            {
                "evaluation_frame",
                "balanced_holdout",
                "natural_holdout",
                "leakage_register",
                "output_dir",
                "ece_bin_count",
            }
        ),
        required=frozenset(
            {
                "evaluation_frame",
                "balanced_holdout",
                "natural_holdout",
                "leakage_register",
                "output_dir",
            }
        ),
        defaults={"ece_bin_count": 10},
    ),
}


def add_reference_workflow_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    runtime_defaults: ReferenceWorkflowRuntimeDefaults,
) -> None:
    geographic = subparsers.add_parser("build-geographic-spread")
    _common(geographic, runtime_defaults)
    geographic.add_argument("--accepted-taxon-key")
    geographic.add_argument("--scientific-name")
    geographic.add_argument("--registry-version")
    geographic.add_argument("--source-snapshot-version")
    geographic.add_argument("--occurrences")
    geographic.add_argument("--output-dir")
    geographic.add_argument("--checkpoint-dir")
    geographic.add_argument("--retrieved-at")
    geographic.add_argument("--coarse-resolution", type=int)
    geographic.add_argument("--regional-resolution", type=int)
    geographic.add_argument("--local-resolution", type=int)
    geographic.add_argument("--page-size", type=int)
    geographic.add_argument("--max-retries", type=int)

    clusters = subparsers.add_parser("cluster-flickr-metadata")
    _common(clusters, runtime_defaults)
    clusters.add_argument("--geography")
    clusters.add_argument("--target-accepted-taxon-key")
    clusters.add_argument("--output-dir")
    clusters.add_argument("--created-at")
    clusters.add_argument(
        "--source-cell-field",
        choices=("coarse_cell_id", "regional_cell_id", "local_cell_id"),
    )
    clusters.add_argument("--source-resolution", type=int)
    clusters.add_argument("--adjacency-grid-distance", type=int)
    clusters.add_argument("--minimum-images-per-cell", type=int)
    clusters.add_argument("--minimum-cluster-images", type=int)
    clusters.add_argument("--maximum-assignment-distance-km", type=float)
    clusters.add_argument("--bioregion-map")
    clusters.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    plan = subparsers.add_parser("plan")
    _common(plan, runtime_defaults)
    plan.add_argument("--candidate-species")
    plan.add_argument("--observations")
    plan.add_argument("--media-candidates")
    plan.add_argument("--review-metadata")
    plan.add_argument("--existing-selections")
    plan.add_argument("--output-dir")
    plan.add_argument("--created-at")
    plan.add_argument("--stratum", dest="strata", action="append", type=_stratum)
    plan.add_argument("--minimum-per-sufficient-cluster", type=int)
    plan.add_argument("--sufficiently-populated-candidate-count", type=int)
    plan.add_argument("--distance-balance-band-km", type=float)
    plan.add_argument("--selection-seed", type=int)
    plan.add_argument("--licence-policy-version")
    plan.add_argument("--selection-strategy")
    plan.add_argument("--eligible-download-statuses", type=_csv)
    plan.add_argument("--eligible-licence-policy-statuses", type=_csv)
    plan.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    fetch = subparsers.add_parser("fetch-metadata")
    _common(fetch, runtime_defaults)
    fetch.add_argument("--queries")
    fetch.add_argument("--registry-version")
    fetch.add_argument("--checkpoint-dir")
    fetch.add_argument("--output-dir")
    fetch.add_argument("--max-retries", type=int)
    fetch.add_argument("--inaturalist-min-request-interval-seconds", type=float)
    fetch.add_argument("--accepted-photo-licences", type=_csv)
    fetch.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    download = subparsers.add_parser("download")
    _common(download, runtime_defaults)
    download.add_argument("--acquisition-selections")
    download.add_argument("--media-candidates")
    download.add_argument("--output-prefix")
    download.add_argument("--run-id")
    download.add_argument("--storage-backend", choices=("local", "s3"))
    download.add_argument("--storage-prefix")
    download.add_argument("--workers", type=int)
    download.add_argument("--max-inflight", type=int)
    download.add_argument("--max-concurrent-decodes", type=int)
    download.add_argument("--max-attempts", type=int)
    download.add_argument("--timeout-seconds", type=float)
    download.add_argument("--max-download-seconds", type=float)

    embeddings = subparsers.add_parser("build-support-embeddings")
    _common(embeddings, runtime_defaults)
    embeddings.add_argument("--readiness-dir")
    embeddings.add_argument("--readiness-sha256")
    embeddings.add_argument("--support-manifest")
    embeddings.add_argument("--visual-inputs")
    embeddings.add_argument("--output")
    embeddings.add_argument("--runtime-python")
    embeddings.add_argument("--hf-cache-dir")
    embeddings.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    embeddings.add_argument("--preprocess-workers", type=int)
    embeddings.add_argument("--batch-size", type=int)
    embeddings.add_argument("--checkpoint-dir")
    embeddings.add_argument("--run-id")
    embeddings.add_argument("--embedding-cache")
    embeddings.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    embeddings.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    prototypes = subparsers.add_parser("build-prototypes")
    _common(prototypes, runtime_defaults)
    prototypes.add_argument("--reference-embeddings")
    prototypes.add_argument("--output")
    prototypes.add_argument("--balanced-sampling-seed", type=int)
    prototypes.add_argument(
        "--include-mean-centered",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prototypes.add_argument(
        "--multi-prototype",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prototypes.add_argument("--minimum-metadata-observation-count", type=int)
    prototypes.add_argument(
        "--enable-embedding-clustering",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prototypes.add_argument("--minimum-clustering-observation-count", type=int)
    prototypes.add_argument("--minimum-embedding-cluster-size", type=int)
    prototypes.add_argument("--maximum-embedding-cluster-count", type=int)
    prototypes.add_argument("--maximum-clustering-observation-count", type=int)
    prototypes.add_argument("--cosine-distance-threshold", type=float)
    prototypes.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    train = subparsers.add_parser("train-classifier")
    _common(train, runtime_defaults)
    train.add_argument("--training-features")
    train.add_argument("--output-dir")
    train.add_argument(
        "--target-task",
        choices=(
            "binary_target_verifier",
            "regional_multiclass",
            "visual_domain",
            "larval_target_verifier",
        ),
    )
    train.add_argument("--target-accepted-taxon-key")
    train.add_argument(
        "--route",
        choices=("adult_field", "larval", "pupal", "egg", "pinned_specimen"),
    )
    train.add_argument("--n-splits", type=int)
    train.add_argument("--random-seed", type=int)
    train.add_argument("--class-weight", choices=("balanced", "none"))
    train.add_argument("--included-label-certainties", type=_csv)
    train.add_argument("--enabled-models", type=_csv)
    train.add_argument(
        "--enable-rbf-pilot",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train.add_argument("--rbf-max-fit-samples", type=int)
    train.add_argument("--n-jobs", type=int)
    train.add_argument("--preprocessing-fingerprint")
    train.add_argument("--reference-bank-version")
    train.add_argument("--reference-bank-fingerprint")
    train.add_argument("--git-sha")
    train.add_argument("--model-name")

    calibrate = subparsers.add_parser("calibrate-classifier")
    _common(calibrate, runtime_defaults)
    calibrate.add_argument("--predictions")
    calibrate.add_argument("--fold-audits")
    calibrate.add_argument("--output-dir")
    calibrate.add_argument("--classifier-fingerprint")
    calibrate.add_argument("--split-fingerprint")
    calibrate.add_argument(
        "--target-task",
        choices=(
            "binary_target_verifier",
            "regional_multiclass",
            "visual_domain",
            "larval_target_verifier",
        ),
    )
    calibrate.add_argument(
        "--route",
        choices=("adult_field", "larval", "pupal", "egg", "pinned_specimen"),
    )
    calibrate.add_argument("--class-labels", type=_csv)
    calibrate.add_argument(
        "--method", choices=("auto", "sigmoid", "isotonic", "temperature")
    )
    calibrate.add_argument("--positive-class-label")
    calibrate.add_argument("--reliability-bin-count", type=int)
    calibrate.add_argument("--minimum-class-group-count", type=int)
    calibrate.add_argument("--decision-policy")
    calibrate.add_argument("--git-sha")

    scoring = subparsers.add_parser("score-target-aware")
    _common(scoring, runtime_defaults)
    scoring.add_argument("--candidate-set")
    scoring.add_argument("--known-negative-classes")
    scoring.add_argument("--visual-domain-classes")
    scoring.add_argument("--score-map")
    scoring.add_argument("--output")

    evaluate = subparsers.add_parser("evaluate-target-verifier")
    _common(evaluate, runtime_defaults)
    evaluate.add_argument("--evaluation-frame")
    evaluate.add_argument("--balanced-holdout")
    evaluate.add_argument("--natural-holdout")
    evaluate.add_argument("--leakage-register")
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--ece-bin-count", type=int)


def is_reference_workflow_command(value: object) -> bool:
    return str(value or "") in _COMMAND_SPECS


def resolve_reference_workflow_options(
    args: argparse.Namespace,
) -> ResolvedReferenceWorkflowOptions:
    command = str(getattr(args, "references_command", "") or "")
    spec = _COMMAND_SPECS.get(command)
    if spec is None:
        raise ValueError(
            f"unsupported reference workflow command: {command or '<none>'}"
        )
    settings_path = getattr(args, "settings_file", None)
    configured = _settings_for_command(settings_path, command=command)
    unexpected = sorted(set(configured) - spec.fields)
    if unexpected:
        raise ValueError(
            f"settings for {command} contain unknown fields: {', '.join(unexpected)}"
        )
    values = {key: _copy_jsonable(value) for key, value in spec.defaults.items()}
    values.update(configured)
    for field_name in spec.fields:
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    runtime_defaults = getattr(args, "reference_workflow_runtime_defaults", None)
    if command == "build-support-embeddings" and isinstance(
        runtime_defaults, ReferenceWorkflowRuntimeDefaults
    ):
        values.setdefault("runtime_python", runtime_defaults.runtime_python)
        values.setdefault("hf_cache_dir", runtime_defaults.hf_cache_dir)
    missing = sorted(
        field_name
        for field_name in spec.required
        if field_name not in values or _blank(values[field_name])
    )
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"{command} requires {options} (CLI or settings file)")
    _validate_effective_options(command, values)
    return ResolvedReferenceWorkflowOptions(
        command=command,
        stage=spec.stage,
        values=values,
        settings_fingerprint=_fingerprint(
            {"command": command, "values": _jsonable(values)}
        ),
        settings_file=str(settings_path) if settings_path is not None else None,
    )


def run_reference_workflow_command(args: argparse.Namespace) -> int:
    try:
        resolved = resolve_reference_workflow_options(args)
        if bool(getattr(args, "dry_run", False)):
            print(
                json.dumps(
                    {
                        "command": f"references {resolved.command}",
                        "dry_run": True,
                        "options": _jsonable(resolved.values),
                        "settings_file": resolved.settings_file,
                        "settings_fingerprint": resolved.settings_fingerprint,
                        "stage": resolved.stage.value,
                        "status": "planned",
                    },
                    sort_keys=True,
                )
            )
            return 0
        payload = _COMMAND_RUNNERS[resolved.command](resolved, args)
    except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


def _common(
    parser: argparse.ArgumentParser,
    runtime_defaults: ReferenceWorkflowRuntimeDefaults,
) -> None:
    parser.add_argument(
        "--settings-file",
        help="JSON settings file; explicit typed options take precedence",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the operation without reading data inputs",
    )
    parser.set_defaults(reference_workflow_runtime_defaults=runtime_defaults)


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected one or more comma-separated values")
    return result


def _stratum(value: str) -> dict[str, object]:
    parts = [item.strip() for item in value.split(":")]
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "stratum must be LIFE_STAGE:VISUAL_DOMAIN:REQUESTED_PER_SPECIES"
        )
    try:
        count = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "stratum requested count must be an integer"
        ) from exc
    return {
        "life_stage": parts[0],
        "visual_domain": parts[1],
        "requested_per_species": count,
    }


def _settings_for_command(path: object, *, command: str) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json_object(path, artifact="reference workflow settings")
    schema = payload.get("schema_version")
    if schema is not None and schema != REFERENCE_WORKFLOW_SETTINGS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported reference workflow settings schema_version: {schema!r}"
        )
    if "commands" in payload:
        commands = payload["commands"]
        if not isinstance(commands, Mapping):
            raise ValueError("reference workflow settings commands must be an object")
        section = commands.get(command, {})
    elif command in payload:
        section = payload[command]
    else:
        section = {
            key: value for key, value in payload.items() if key != "schema_version"
        }
    if not isinstance(section, Mapping):
        raise ValueError(f"settings section {command!r} must be an object")
    return {str(key): value for key, value in section.items()}


def _validate_effective_options(command: str, values: dict[str, Any]) -> None:
    if command == "build-geographic-spread":
        from biominer.geography import GeographicResolutions

        GeographicResolutions(
            coarse=_integer(values["coarse_resolution"], "coarse_resolution"),
            regional=_integer(values["regional_resolution"], "regional_resolution"),
            local=_integer(values["local_resolution"], "local_resolution"),
        )
        _positive_integer(values["page_size"], "page_size")
        _nonnegative_integer(values["max_retries"], "max_retries")
    elif command == "cluster-flickr-metadata":
        _flickr_cluster_config(values)
    elif command == "plan":
        _reference_planner_config(values)
        _boolean(values["overwrite"], "overwrite")
    elif command == "fetch-metadata":
        _nonnegative_integer(values["max_retries"], "max_retries")
        _nonnegative_float(
            values["inaturalist_min_request_interval_seconds"],
            "inaturalist_min_request_interval_seconds",
        )
        _boolean(values["overwrite"], "overwrite")
        if values.get("accepted_photo_licences") is not None:
            _string_tuple(
                values["accepted_photo_licences"],
                field="accepted_photo_licences",
            )
    elif command == "download":
        _reference_download_config(values)
        if values.get("storage_backend") not in {None, "local", "s3"}:
            raise ValueError("storage_backend must be local or s3")
    elif command == "build-support-embeddings":
        _canonical_sha256(values["readiness_sha256"], "readiness_sha256")
        _positive_integer(values["batch_size"], "batch_size")
        _positive_integer(values["preprocess_workers"], "preprocess_workers")
        if values["device"] not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be auto, cuda, mps, or cpu")
        _boolean(values["resume"], "resume")
        _boolean(values["overwrite"], "overwrite")
    elif command == "build-prototypes":
        _multi_prototype_config(values)
        _integer(values["balanced_sampling_seed"], "balanced_sampling_seed")
        _boolean(values["include_mean_centered"], "include_mean_centered")
        _boolean(values["multi_prototype"], "multi_prototype")
        _boolean(values["overwrite"], "overwrite")
    elif command == "train-classifier":
        _classifier_training_config(values)
        _canonical_sha256(
            values["preprocessing_fingerprint"],
            "preprocessing_fingerprint",
        )
        _canonical_sha256(
            values["reference_bank_fingerprint"],
            "reference_bank_fingerprint",
        )
    elif command == "calibrate-classifier":
        _calibration_config(values)
    elif command == "evaluate-target-verifier":
        _positive_integer(values["ece_bin_count"], "ece_bin_count")


def _run_score_target_aware(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.bioclip.target_aware_scoring import (
        build_target_aware_scoring_plan,
        score_target_aware_candidate_union,
    )

    values = resolved.values
    candidate_set = _candidate_set_from_mapping(
        _read_json_object(values["candidate_set"], artifact="candidate set")
    )
    known_negatives = _auxiliary_classes_from_path(
        values["known_negative_classes"], artifact="known-negative classes"
    )
    visual_domains = _auxiliary_classes_from_path(
        values["visual_domain_classes"], artifact="visual-domain classes"
    )
    score_payload = _read_json_object(values["score_map"], artifact="score map")
    raw_scores = score_payload.get("scores", score_payload)
    if not isinstance(raw_scores, Mapping):
        raise ValueError("score map must be an object or contain a scores object")
    scorer = _StaticCompleteSetScorer(
        {
            str(key): _finite_float(value, f"scores[{key}]")
            for key, value in raw_scores.items()
        }
    )
    plan = build_target_aware_scoring_plan(
        candidate_set,
        known_negative_classes=known_negatives,
        visual_domain_classes=visual_domains,
    )
    result = score_target_aware_candidate_union(plan, scorer)
    output = Path(str(values["output"]))
    payload = _jsonable(asdict(result))
    _write_json_atomic(output, payload, overwrite=False)
    return {
        "artifacts": {"complete_set_scores": str(output)},
        "candidate_set_id": result.candidate_set_id,
        "command": "references score-target-aware",
        "scored_class_count": len(result.scored_classes),
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
        "target_regional_rank": result.target_regional_rank,
    }


@dataclass(frozen=True, slots=True)
class _StaticCompleteSetScorer:
    scores: Mapping[str, float]

    def score(self, _plan: object) -> Mapping[str, float]:
        return self.scores


def _run_evaluate_target_verifier(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.evaluation.target_metrics import (
        TargetVerificationMetricsConfig,
        evaluate_target_verification,
        publish_target_verification_metric_report,
    )

    values = resolved.values
    evaluation_frame = pl.read_parquet(str(values["evaluation_frame"]))
    balanced_holdout = pl.read_parquet(str(values["balanced_holdout"]))
    natural_holdout = pl.read_parquet(str(values["natural_holdout"]))
    leakage_register = pl.read_parquet(str(values["leakage_register"]))
    report = evaluate_target_verification(
        evaluation_frame,
        balanced_holdout,
        natural_holdout,
        leakage_register,
        TargetVerificationMetricsConfig(
            ece_bin_count=int(values["ece_bin_count"]),
        ),
    )
    publication = publish_target_verification_metric_report(
        report,
        Path(str(values["output_dir"])),
    )
    return {
        "artifacts": {
            "metrics": str(publication.metrics_path),
            "margin_distribution": str(publication.margin_distribution_path),
            "report": str(publication.report_json_path),
            "summary": str(publication.report_markdown_path),
        },
        "command": "references evaluate-target-verifier",
        "sample_count": evaluation_frame.height,
        "report_fingerprint": report.report_fingerprint,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }




def _candidate_set_from_mapping(payload: Mapping[str, object]) -> object:
    from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon

    allowed = {
        "candidate_set_id",
        "registry_version",
        "target_accepted_taxon_key",
        "target_scientific_name",
        "family_candidates",
        "genus_candidates",
        "species_candidates",
        "prompt_variant_version",
        "geospatial_scope",
        "source_evidence",
        "candidate_contract_version",
        "candidate_set_fingerprint",
    }
    _reject_unknown(payload, allowed, artifact="candidate set")

    def taxa(field_name: str) -> tuple[CandidateTaxon, ...]:
        raw = payload.get(field_name)
        if not isinstance(raw, list):
            raise ValueError(f"candidate set {field_name} must be an array")
        result: list[CandidateTaxon] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(f"candidate set {field_name} entries must be objects")
            candidate = dict(item)
            for tuple_field in ("common_names", "candidate_reasons", "source_versions"):
                candidate[tuple_field] = _string_tuple(
                    candidate.get(tuple_field, ()), field=f"{field_name}.{tuple_field}"
                )
            result.append(CandidateTaxon(**candidate))
        return tuple(result)

    kwargs = dict(payload)
    kwargs["family_candidates"] = taxa("family_candidates")
    kwargs["genus_candidates"] = taxa("genus_candidates")
    kwargs["species_candidates"] = taxa("species_candidates")
    kwargs["source_evidence"] = _string_tuple(
        payload.get("source_evidence", ()), field="source_evidence"
    )
    return CandidateSet(**kwargs)


def _auxiliary_classes_from_path(path: object, *, artifact: str) -> tuple[object, ...]:
    from biominer.bioclip.target_aware_scoring import TargetAwareAuxiliaryClass

    payload = _read_json_value(path, artifact=artifact)
    values = payload.get("classes") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError(f"{artifact} must be an array or contain a classes array")
    result = []
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError(f"{artifact} entries must be objects")
        _reject_unknown(
            item,
            {"class_id", "display_name", "source_versions"},
            artifact=artifact,
        )
        result.append(
            TargetAwareAuxiliaryClass(
                class_id=str(item.get("class_id") or ""),
                display_name=str(item.get("display_name") or ""),
                source_versions=_string_tuple(
                    item.get("source_versions", ()), field="source_versions"
                ),
            )
        )
    return tuple(result)


def _read_json_value(path: object, *, artifact: str) -> object:
    source = Path(str(path))
    if not source.is_file():
        raise FileNotFoundError(f"{artifact} path does not exist: {source}")
    data = source.read_bytes()
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError(f"{artifact} exceeds {_MAX_JSON_BYTES} bytes")
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} is not valid JSON: {exc}") from exc


def _read_json_object(path: object, *, artifact: str) -> dict[str, Any]:
    payload = _read_json_value(path, artifact=artifact)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact} must contain a JSON object")
    return payload


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains unsupported constant: {value}")


def _write_json_atomic(path: Path, payload: object, *, overwrite: bool) -> None:
    _write_text_atomic(
        path,
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def _write_text_atomic(path: Path, value: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(path) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_outputs_available(
    paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing workflow artifacts: " + ", ".join(existing)
        )


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def _copy_jsonable(value: object) -> Any:
    return json.loads(json.dumps(_jsonable(value)))


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _reject_unknown(
    payload: Mapping[str, object], allowed: set[str], *, artifact: str
) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"{artifact} contains unknown fields: {', '.join(unexpected)}")


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field} cannot contain blank values")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be Boolean")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _positive_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_float(value: object, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _canonical_sha256(value: object, field: str) -> str:
    text = str(value or "").strip()
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} must be a canonical sha256 digest")
    return text


def _nonnegative_float(value: object, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


# Config builders are also called during dry-run so JSON values receive the same
# type and invariant checks as explicit argparse values.
def _flickr_cluster_config(values: Mapping[str, object]) -> object:
    from biominer.flickr_fetch.geographic_clustering import FlickrGeoClusterConfig

    bioregions: tuple[tuple[str, str], ...] = ()
    mapping = values.get("bioregion_map")
    if mapping is not None:
        payload = (
            _read_json_object(mapping, artifact="bioregion map")
            if isinstance(mapping, str)
            else mapping
        )
        if not isinstance(payload, Mapping):
            raise TypeError("bioregion_map must be an object or JSON object path")
        bioregions = tuple(
            sorted((str(key), str(value)) for key, value in payload.items())
        )
    return FlickrGeoClusterConfig(
        source_cell_field=str(values["source_cell_field"]),
        source_resolution=values["source_resolution"],
        adjacency_grid_distance=values["adjacency_grid_distance"],
        minimum_images_per_cell=values["minimum_images_per_cell"],
        minimum_cluster_images=values["minimum_cluster_images"],
        maximum_assignment_distance_km=values["maximum_assignment_distance_km"],
        bioregion_by_admin_region=bioregions,
    )


def _reference_planner_config(values: Mapping[str, object]) -> object:
    from biominer.references.planner import (
        ReferencePlannerConfig,
        ReferenceStratumQuota,
    )

    strata_payload = values["strata"]
    if not isinstance(strata_payload, (list, tuple)):
        raise TypeError("strata must be an array")
    strata = []
    for item in strata_payload:
        if not isinstance(item, Mapping):
            raise TypeError("strata entries must be objects")
        _reject_unknown(
            item,
            {"life_stage", "visual_domain", "requested_per_species"},
            artifact="reference stratum",
        )
        strata.append(ReferenceStratumQuota(**dict(item)))
    return ReferencePlannerConfig(
        strata=tuple(strata),
        minimum_per_sufficient_cluster=values["minimum_per_sufficient_cluster"],
        sufficiently_populated_candidate_count=values[
            "sufficiently_populated_candidate_count"
        ],
        distance_balance_band_km=values["distance_balance_band_km"],
        selection_seed=values["selection_seed"],
        licence_policy_version=str(values["licence_policy_version"]),
        selection_strategy=str(values["selection_strategy"]),
        eligible_download_statuses=_string_tuple(
            values["eligible_download_statuses"], field="eligible_download_statuses"
        ),
        eligible_licence_policy_statuses=_string_tuple(
            values["eligible_licence_policy_statuses"],
            field="eligible_licence_policy_statuses",
        ),
    )


def _reference_download_config(values: Mapping[str, object]) -> object:
    from biominer.references.downloader import (
        ProviderMediaDownloadPolicy,
        ReferenceMediaDownloadConfig,
    )

    payload = values.get("download_config", {})
    if not isinstance(payload, Mapping):
        raise TypeError("download_config must be an object")
    config = dict(payload)
    for field_name in (
        "workers",
        "max_inflight",
        "max_concurrent_decodes",
        "max_attempts",
        "timeout_seconds",
        "max_download_seconds",
    ):
        config[field_name] = values[field_name]
    for field_name in ("retry_statuses", "allowed_content_types"):
        if field_name in config:
            config[field_name] = _string_or_integer_tuple(
                config[field_name], field=field_name
            )
    if "temporary_directory" in config and config["temporary_directory"] is not None:
        config["temporary_directory"] = Path(str(config["temporary_directory"]))
    if "provider_policies" in config:
        raw_policies = config["provider_policies"]
        if not isinstance(raw_policies, (list, tuple)):
            raise TypeError("provider_policies must be an array")
        policies = []
        for item in raw_policies:
            if not isinstance(item, Mapping):
                raise TypeError("provider_policies entries must be objects")
            policy = dict(item)
            for field_name in ("allowed_hosts", "allowed_schemes"):
                if field_name in policy:
                    policy[field_name] = _string_tuple(
                        policy[field_name], field=f"provider_policies.{field_name}"
                    )
            policies.append(ProviderMediaDownloadPolicy(**policy))
        config["provider_policies"] = tuple(policies)
    return ReferenceMediaDownloadConfig(**config)


def _string_or_integer_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be an array")
    return tuple(value)


def _multi_prototype_config(values: Mapping[str, object]) -> object:
    from biominer.bioclip.reference_prototypes import MultiPrototypeConfig

    return MultiPrototypeConfig(
        minimum_metadata_observation_count=values["minimum_metadata_observation_count"],
        enable_embedding_clustering=values["enable_embedding_clustering"],
        minimum_clustering_observation_count=values[
            "minimum_clustering_observation_count"
        ],
        minimum_embedding_cluster_size=values["minimum_embedding_cluster_size"],
        maximum_embedding_cluster_count=values["maximum_embedding_cluster_count"],
        maximum_clustering_observation_count=values[
            "maximum_clustering_observation_count"
        ],
        cosine_distance_threshold=values["cosine_distance_threshold"],
    )


def _classifier_training_config(values: Mapping[str, object]) -> object:
    from biominer.ml.classifiers import ClassifierTrainingConfig

    class_weight = values["class_weight"]
    if class_weight == "none":
        class_weight = None
    return ClassifierTrainingConfig(
        target_task=str(values["target_task"]),
        target_accepted_taxon_key=str(values["target_accepted_taxon_key"]),
        route=str(values["route"]),
        n_splits=values["n_splits"],
        random_seed=values["random_seed"],
        class_weight=class_weight,
        included_label_certainties=_string_tuple(
            values["included_label_certainties"],
            field="included_label_certainties",
        ),
        enabled_models=_string_tuple(values["enabled_models"], field="enabled_models"),
        enable_rbf_pilot=values["enable_rbf_pilot"],
        rbf_max_fit_samples=values["rbf_max_fit_samples"],
        n_jobs=values["n_jobs"],
    )


def _calibration_config(values: Mapping[str, object]) -> object:
    from biominer.ml.calibration import CalibrationConfig

    return CalibrationConfig(
        classifier_fingerprint=str(values["classifier_fingerprint"]),
        split_fingerprint=str(values["split_fingerprint"]),
        target_task=str(values["target_task"]),
        route=str(values["route"]),
        class_labels=_string_tuple(values["class_labels"], field="class_labels"),
        method=str(values["method"]),
        positive_class_label=(
            None
            if values.get("positive_class_label") is None
            else str(values["positive_class_label"])
        ),
        reliability_bin_count=values["reliability_bin_count"],
        minimum_class_group_count=values["minimum_class_group_count"],
    )


_COMMAND_RUNNERS = {
    "build-geographic-spread": lambda resolved, args: _run_geographic_spread(
        resolved, args
    ),
    "cluster-flickr-metadata": lambda resolved, args: _run_flickr_clusters(
        resolved, args
    ),
    "plan": lambda resolved, args: _run_reference_plan(resolved, args),
    "fetch-metadata": lambda resolved, args: _run_fetch_metadata(resolved, args),
    "download": lambda resolved, args: _run_reference_download(resolved, args),
    "build-support-embeddings": lambda resolved, args: _run_support_embeddings(
        resolved, args
    ),
    "build-prototypes": lambda resolved, args: _run_build_prototypes(resolved, args),
    "train-classifier": lambda resolved, args: _run_train_classifier(resolved, args),
    "calibrate-classifier": lambda resolved, args: _run_calibrate_classifier(
        resolved, args
    ),
    "score-target-aware": _run_score_target_aware,
    "evaluate-target-verifier": _run_evaluate_target_verifier,
}


def _run_geographic_spread(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.geography import GeographicResolutions
    from biominer.registry.geographic_spread import (
        GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
        GEOGRAPHIC_SPREAD_MANIFEST_FILE,
        TAXON_GEOGRAPHIC_SPREAD_FILE,
        GBIFOccurrenceSearchSource,
        GBIFParquetOccurrenceSource,
        build_taxon_geographic_spread,
    )

    values = resolved.values
    source = (
        GBIFParquetOccurrenceSource(
            str(values["occurrences"]),
            accepted_taxon_key=str(values["accepted_taxon_key"]),
            source_snapshot_version=str(values["source_snapshot_version"]),
        )
        if values.get("occurrences")
        else GBIFOccurrenceSearchSource(
            accepted_taxon_key=str(values["accepted_taxon_key"]),
            source_snapshot_version=str(values["source_snapshot_version"]),
            page_size=int(values["page_size"]),
            max_retries=int(values["max_retries"]),
        )
    )
    output_dir = Path(str(values["output_dir"]))
    try:
        result = build_taxon_geographic_spread(
            accepted_taxon_key=str(values["accepted_taxon_key"]),
            scientific_name=str(values["scientific_name"]),
            registry_version=str(values["registry_version"]),
            source=source,
            resolutions=GeographicResolutions(
                coarse=int(values["coarse_resolution"]),
                regional=int(values["regional_resolution"]),
                local=int(values["local_resolution"]),
            ),
            output_dir=output_dir,
            checkpoint_dir=Path(str(values["checkpoint_dir"])),
            retrieved_at=str(values["retrieved_at"]),
        )
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    return {
        "artifacts": {
            "spread": str(output_dir / TAXON_GEOGRAPHIC_SPREAD_FILE),
            "evidence": str(
                result.evidence_path
                if result.evidence_path.name == GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE
                else output_dir / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE
            ),
            "manifest": str(output_dir / GEOGRAPHIC_SPREAD_MANIFEST_FILE),
        },
        "command": "references build-geographic-spread",
        "resumed": result.resumed,
        "row_count": result.spread.height,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_flickr_clusters(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.flickr_fetch.geographic_clustering import (
        build_flickr_geo_clusters,
        write_flickr_geo_cluster_artifacts,
    )

    values = resolved.values
    output_dir = Path(str(values["output_dir"]))
    _ensure_outputs_available(
        (
            output_dir / "flickr_geo_clusters.parquet",
            output_dir / "flickr_geo_assignments.parquet",
        ),
        overwrite=bool(values["overwrite"]),
    )
    result = build_flickr_geo_clusters(
        pl.read_parquet(str(values["geography"])),
        target_accepted_taxon_key=str(values["target_accepted_taxon_key"]),
        config=_flickr_cluster_config(values),
        created_at=values.get("created_at"),
    )
    artifacts = write_flickr_geo_cluster_artifacts(
        result,
        output_dir,
        overwrite=bool(values["overwrite"]),
    )
    return {
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "assignment_count": result.assignments.height,
        "cluster_configuration_hash": result.cluster_configuration_hash,
        "cluster_count": result.clusters.height,
        "command": "references cluster-flickr-metadata",
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_reference_plan(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.references.planner import (
        plan_geographically_balanced_support_bank,
        write_reference_plan_result,
    )

    values = resolved.values
    output_dir = Path(str(values["output_dir"]))
    _ensure_outputs_available(
        (
            output_dir / "reference_acquisition_plan.parquet",
            output_dir / "reference_acquisition_selections.parquet",
            output_dir / "reference_acquisition_plan.json",
            output_dir / "reference_acquisition_plan.md",
        ),
        overwrite=bool(values["overwrite"]),
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=pl.read_parquet(str(values["candidate_species"])),
        observations=pl.read_parquet(str(values["observations"])),
        media_candidates=pl.read_parquet(str(values["media_candidates"])),
        review_metadata=_optional_parquet(values.get("review_metadata")),
        existing_selections=_optional_parquet(values.get("existing_selections")),
        config=_reference_planner_config(values),
        created_at=values.get("created_at"),
    )
    artifacts = write_reference_plan_result(result, output_dir)
    return {
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "command": "references plan",
        "plan_row_count": result.plan.height,
        "selection_row_count": result.selections.height,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_fetch_metadata(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.references.gbif import (
        GBIFReferenceAdapter,
        load_gbif_reference_checkpoint_frames,
    )
    from biominer.references.inaturalist import (
        DEFAULT_ACCEPTED_PHOTO_LICENCES,
        INaturalistReferenceAdapter,
        load_inaturalist_reference_checkpoint_frames,
        mark_inaturalist_gbif_media_duplicates,
    )
    from biominer.references.schemas import (
        reference_media_candidates_frame,
        reference_observations_frame,
        write_reference_media_candidates,
        write_reference_observations,
    )

    values = resolved.values
    output_dir = Path(str(values["output_dir"]))
    report_path = output_dir / "reference_metadata_report.json"
    _ensure_outputs_available(
        (
            output_dir / "reference_observations.parquet",
            output_dir / "reference_media_candidates.parquet",
            report_path,
        ),
        overwrite=bool(values["overwrite"]),
    )
    queries = _reference_source_queries(values["queries"])
    checkpoint_dir = Path(str(values["checkpoint_dir"]))
    accepted_licences = (
        _string_tuple(
            values["accepted_photo_licences"], field="accepted_photo_licences"
        )
        if values.get("accepted_photo_licences") is not None
        else DEFAULT_ACCEPTED_PHOTO_LICENCES
    )
    adapters: dict[str, object] = {}
    if any(source == "GBIF" for source, _query in queries):
        adapters["GBIF"] = GBIFReferenceAdapter(
            registry_version=str(values["registry_version"]),
            max_retries=int(values["max_retries"]),
        )
    if any(source == "iNaturalist" for source, _query in queries):
        adapters["iNaturalist"] = INaturalistReferenceAdapter(
            registry_version=str(values["registry_version"]),
            accepted_photo_licences=accepted_licences,
            max_retries=int(values["max_retries"]),
            min_request_interval_seconds=float(
                values["inaturalist_min_request_interval_seconds"]
            ),
        )
    observation_frames: list[pl.DataFrame] = []
    media_frames: list[pl.DataFrame] = []
    fetched_pages = 0
    try:
        for source, query in queries:
            adapter = adapters[source]
            for _page in adapter.iter_pages(query, checkpoint_dir=checkpoint_dir):
                fetched_pages += 1
            if source == "GBIF":
                observations, media = load_gbif_reference_checkpoint_frames(
                    query, checkpoint_dir
                )
            else:
                observations, media = load_inaturalist_reference_checkpoint_frames(
                    query, checkpoint_dir
                )
            observation_frames.append(observations)
            media_frames.append(media)
    finally:
        for adapter in adapters.values():
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
    observations = reference_observations_frame(
        _merge_unique_rows(
            observation_frames,
            key="reference_observation_id",
            artifact="reference observations",
        )
    )
    media_candidates = reference_media_candidates_frame(
        _merge_unique_rows(
            media_frames,
            key="reference_media_id",
            artifact="reference media candidates",
        )
    )
    media_candidates = mark_inaturalist_gbif_media_duplicates(
        observations, media_candidates
    )
    observation_path = write_reference_observations(
        observations,
        output_dir,
        overwrite=bool(values["overwrite"]),
    )
    media_path = write_reference_media_candidates(
        media_candidates,
        output_dir,
        overwrite=bool(values["overwrite"]),
    )
    _write_json_atomic(
        report_path,
        {
            "schema_version": "reference-metadata-cli-report-v1",
            "command": "references fetch-metadata",
            "query_count": len(queries),
            "fetched_page_count": fetched_pages,
            "observation_count": observations.height,
            "media_candidate_count": media_candidates.height,
            "settings_fingerprint": resolved.settings_fingerprint,
            "status": "complete",
        },
        overwrite=bool(values["overwrite"]),
    )
    return {
        "artifacts": {
            "observations": str(observation_path),
            "media_candidates": str(media_path),
            "report": str(report_path),
        },
        "command": "references fetch-metadata",
        "media_candidate_count": media_candidates.height,
        "observation_count": observations.height,
        "query_count": len(queries),
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_reference_download(
    resolved: ResolvedReferenceWorkflowOptions,
    args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.config import create_storage_backend, load_biominer_config
    from biominer.references.downloader import download_reference_media

    values = resolved.values
    config = load_biominer_config(getattr(args, "config", None))
    storage_config = config.storage
    if values.get("storage_backend") is not None:
        storage_config = replace(storage_config, backend=str(values["storage_backend"]))
    if values.get("storage_prefix") is not None:
        storage_config = replace(storage_config, prefix=str(values["storage_prefix"]))
    storage = create_storage_backend(storage_config)
    result = download_reference_media(
        pl.read_parquet(str(values["acquisition_selections"])),
        pl.read_parquet(str(values["media_candidates"])),
        storage=storage,
        output_prefix=str(values["output_prefix"]),
        config=_reference_download_config(values),
        licence_policy=_reference_licence_policy(values),
        run_id=(str(values["run_id"]) if values.get("run_id") else None),
    )
    return {
        "artifacts": {
            "media_objects": result.media_objects_uri,
            "report": result.report_uri,
            "summary": result.summary_uri,
        },
        "command": "references download",
        "media_object_count": result.media_objects.height,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_support_embeddings(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.bioclip.bioclip import PersistentBioClipScorer
    from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
    from biominer.bioclip.reference_embeddings import (
        build_reference_embeddings,
        write_reference_embeddings,
    )
    from biominer.references.readiness import (
        REFERENCE_SUPPORT_MANIFEST_FILE,
        load_reference_bank_readiness,
    )

    values = resolved.values
    permit = load_reference_bank_readiness(
        str(values["readiness_dir"]),
        expected_readiness_sha256=str(values["readiness_sha256"]),
    )
    support_path = Path(
        str(
            values.get("support_manifest")
            or Path(str(values["readiness_dir"])) / REFERENCE_SUPPORT_MANIFEST_FILE
        )
    )
    runtime_python = Path(str(values["runtime_python"])).expanduser()
    if not runtime_python.is_file():
        raise FileNotFoundError(f"BioCLIP runtime Python not found: {runtime_python}")
    runtime = BioClipRuntime(
        model=ModelConfig(
            model_id="bioclip2_5_huge",
            display_name="BioCLIP 2.5 Huge",
            role="preferred",
            status="use_if_available",
            task="frozen reference image embedding",
            model_name=permit.model_name,
            checkpoint=permit.model_revision,
            package_name="open_clip_torch",
            package_version=permit.open_clip_version,
            model_hash=permit.checkpoint_sha256,
        ),
        home=runtime_python.parent.parent,
        venv_python=runtime_python,
        package_version=permit.open_clip_version,
        available=True,
    )
    with PersistentBioClipScorer(
        runtime=runtime,
        hf_cache_dir=str(values["hf_cache_dir"]),
        device=str(values["device"]),
        preprocess_workers=int(values["preprocess_workers"]),
    ) as scorer:
        embeddings = build_reference_embeddings(
            pl.read_parquet(support_path),
            _reference_visual_inputs(values["visual_inputs"]),
            readiness_permit=permit,
            scorer=scorer,
            batch_size=int(values["batch_size"]),
            embedding_cache=values.get("embedding_cache"),
            checkpoint_dir=values.get("checkpoint_dir"),
            resume=bool(values["resume"]),
            run_id=(str(values["run_id"]) if values.get("run_id") else None),
        )
        cache_metrics = dict(scorer.cache_metrics)
    output = write_reference_embeddings(
        embeddings,
        str(values["output"]),
        overwrite=bool(values["overwrite"]),
    )
    return {
        "artifacts": {"reference_embeddings": str(output)},
        "cache_metrics": cache_metrics,
        "command": "references build-support-embeddings",
        "row_count": embeddings.height,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_build_prototypes(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.bioclip.reference_prototypes import (
        build_reference_prototypes,
        reference_prototypes_artifact_fingerprint,
        write_reference_prototypes,
    )

    values = resolved.values
    prototypes = build_reference_prototypes(
        str(values["reference_embeddings"]),
        balanced_sampling_seed=int(values["balanced_sampling_seed"]),
        include_mean_centered=bool(values["include_mean_centered"]),
        multi_prototype_config=(
            _multi_prototype_config(values) if values["multi_prototype"] else None
        ),
    )
    output = write_reference_prototypes(
        prototypes,
        str(values["output"]),
        overwrite=bool(values["overwrite"]),
    )
    return {
        "artifact_fingerprint": reference_prototypes_artifact_fingerprint(prototypes),
        "artifacts": {"reference_prototypes": str(output)},
        "command": "references build-prototypes",
        "row_count": prototypes.height,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_train_classifier(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.ml.classifiers import train_frozen_embedding_classifiers
    from biominer.ml.persistence import write_frozen_classifier

    values = resolved.values
    training = train_frozen_embedding_classifiers(
        str(values["training_features"]),
        _classifier_training_config(values),
    )
    git_sha = _resolved_git_sha(values.get("git_sha"))
    artifacts = write_frozen_classifier(
        training,
        str(values["output_dir"]),
        preprocessing_fingerprint=str(values["preprocessing_fingerprint"]),
        reference_bank_version=str(values["reference_bank_version"]),
        reference_bank_fingerprint=str(values["reference_bank_fingerprint"]),
        git_sha=git_sha,
        model_name=(str(values["model_name"]) if values.get("model_name") else None),
    )
    return {
        "artifacts": {
            "directory": str(artifacts.directory),
            "manifest": str(artifacts.manifest_path),
            "arrays": str(artifacts.arrays_path),
        },
        "classifier_fingerprint": artifacts.classifier_fingerprint,
        "command": "references train-classifier",
        "fit_sample_count": training.fit_sample_count,
        "selected_model_name": training.selected_model_name,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _run_calibrate_classifier(
    resolved: ResolvedReferenceWorkflowOptions,
    _args: argparse.Namespace,
) -> dict[str, object]:
    from biominer.ml.calibration import (
        fit_probability_calibrator,
        write_probability_calibrator,
    )

    values = resolved.values
    fit = fit_probability_calibrator(
        _calibration_predictions(values["predictions"]),
        _calibration_fold_audits(values["fold_audits"]),
        _calibration_config(values),
    )
    decision_policy = (
        _read_json_object(values["decision_policy"], artifact="decision policy")
        if values.get("decision_policy")
        else None
    )
    artifacts = write_probability_calibrator(
        fit,
        str(values["output_dir"]),
        git_sha=_resolved_git_sha(values.get("git_sha")),
        decision_policy=decision_policy,
    )
    return {
        "artifacts": {
            "directory": str(artifacts.directory),
            "manifest": str(artifacts.manifest_path),
            "arrays": str(artifacts.arrays_path),
            "report": str(artifacts.report_path),
        },
        "calibration_fingerprint": artifacts.calibration_fingerprint,
        "command": "references calibrate-classifier",
        "group_count": fit.group_count,
        "sample_count": fit.sample_count,
        "settings_fingerprint": resolved.settings_fingerprint,
        "status": "complete",
    }


def _optional_parquet(value: object) -> pl.DataFrame | None:
    return None if value is None else pl.read_parquet(str(value))


def _reference_source_queries(path: object) -> tuple[tuple[str, object], ...]:
    from biominer.references.source_base import ReferenceSourceQuery

    payload = _read_json_value(path, artifact="reference source queries")
    values = payload.get("queries") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("reference source queries must contain a non-empty array")
    allowed = {
        "source",
        "accepted_taxon_key",
        "scientific_name",
        "geo_cluster_id",
        "fallback_level",
        "source_taxon_id",
        "spatial_cell_ids",
        "country_codes",
        "source_place_ids",
        "geometry_wkt",
        "bounding_box",
        "cluster_medoid_latitude",
        "cluster_medoid_longitude",
        "page_size",
        "source_snapshot_version",
    }
    result = []
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("reference source query entries must be objects")
        _reject_unknown(item, allowed, artifact="reference source query")
        query_payload = dict(item)
        source = str(query_payload.pop("source", "")).strip()
        normalized_source = {
            "gbif": "GBIF",
            "inaturalist": "iNaturalist",
        }.get(source.casefold())
        if normalized_source is None:
            raise ValueError(
                "reference source query source must be GBIF or iNaturalist"
            )
        query_payload.setdefault(
            "page_size", 200 if normalized_source == "iNaturalist" else 300
        )
        for field_name in (
            "spatial_cell_ids",
            "country_codes",
            "source_place_ids",
            "bounding_box",
        ):
            if field_name in query_payload and query_payload[field_name] is not None:
                query_payload[field_name] = tuple(query_payload[field_name])
        result.append((normalized_source, ReferenceSourceQuery(**query_payload)))
    fingerprints = [query.query_fingerprint for _source, query in result]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("reference source queries contain duplicate semantic queries")
    return tuple(sorted(result, key=lambda item: (item[0], item[1].query_fingerprint)))


def _merge_unique_rows(
    frames: Sequence[pl.DataFrame],
    *,
    key: str,
    artifact: str,
) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            identity = str(row[key])
            previous = rows.get(identity)
            if previous is not None and previous != row:
                raise ValueError(
                    f"{artifact} contain conflicting rows for {key}={identity}"
                )
            rows[identity] = row
    return [rows[identity] for identity in sorted(rows)]


def _reference_licence_policy(values: Mapping[str, object]) -> object:
    from biominer.references.licensing import ReferenceLicencePolicy

    payload = values.get("licence_policy", {})
    if not isinstance(payload, Mapping):
        raise TypeError("licence_policy must be an object")
    config = dict(payload)
    for field_name in (
        "broadly_reusable",
        "research_only",
        "attribution_required",
    ):
        if field_name in config:
            config[field_name] = _string_tuple(config[field_name], field=field_name)
    if "licence_aliases" in config:
        aliases = config["licence_aliases"]
        if not isinstance(aliases, (list, tuple)):
            raise TypeError("licence_aliases must be an array")
        config["licence_aliases"] = tuple(tuple(item) for item in aliases)
    return ReferenceLicencePolicy(**config)


def _reference_visual_inputs(path: object) -> tuple[object, ...]:
    from biominer.bioclip.reference_embeddings import ReferenceVisualInput

    rows = _records(path, artifact="reference visual inputs")
    allowed = {
        "reference_media_id",
        "source_image_path",
        "image_path",
        "visual_input_id",
        "visual_input_kind",
        "raw_image_content_hash",
        "image_content_hash",
        "transformation_version",
        "transformation_policy_fingerprint",
        "transformation_fingerprint",
    }
    result = []
    for row in rows:
        _reject_unknown(row, allowed, artifact="reference visual input")
        values = dict(row)
        values["source_image_path"] = Path(str(values["source_image_path"]))
        values["image_path"] = Path(str(values["image_path"]))
        result.append(ReferenceVisualInput(**values))
    return tuple(result)


def _calibration_predictions(path: object) -> tuple[object, ...]:
    from biominer.ml.calibration import CalibrationPrediction

    allowed = {
        "prediction_id",
        "source_item_id",
        "leakage_component_id",
        "fold_index",
        "dataset_split",
        "true_class_label",
        "decision_scores",
        "sample_weight",
    }
    result = []
    for row in _records(path, artifact="calibration predictions"):
        _reject_unknown(row, allowed, artifact="calibration prediction")
        values = dict(row)
        values["decision_scores"] = tuple(values["decision_scores"])
        result.append(CalibrationPrediction(**values))
    return tuple(result)


def _calibration_fold_audits(path: object) -> tuple[object, ...]:
    from biominer.ml.calibration import CalibrationFoldAudit

    allowed = {"fold_index", "estimator_fit_group_ids", "validation_group_ids"}
    result = []
    for row in _records(path, artifact="calibration fold audits"):
        _reject_unknown(row, allowed, artifact="calibration fold audit")
        values = dict(row)
        values["estimator_fit_group_ids"] = _string_tuple(
            values["estimator_fit_group_ids"], field="estimator_fit_group_ids"
        )
        values["validation_group_ids"] = _string_tuple(
            values["validation_group_ids"], field="validation_group_ids"
        )
        result.append(CalibrationFoldAudit(**values))
    return tuple(result)


def _records(path: object, *, artifact: str) -> list[dict[str, object]]:
    source = Path(str(path))
    if source.suffix.casefold() == ".parquet":
        if not source.is_file():
            raise FileNotFoundError(f"{artifact} path does not exist: {source}")
        return pl.read_parquet(source).to_dicts()
    payload = _read_json_value(source, artifact=artifact)
    values = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not all(
        isinstance(item, Mapping) for item in values
    ):
        raise ValueError(f"{artifact} must be an array or contain a rows array")
    return [dict(item) for item in values]


def _resolved_git_sha(value: object) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    from biominer.reports.flickr_fetch import current_git_sha

    git_sha = current_git_sha()
    if not git_sha:
        raise ValueError("git_sha is required when the repository SHA is unavailable")
    return git_sha
