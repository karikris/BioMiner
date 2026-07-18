"""TaxaLens anti-corruption manifest for dynamic-pool evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from biominer.bioclip.dynamic_pool_scores import (
    DYNAMIC_POOL_CANDIDATE_SCORES_FILE,
    DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
    DYNAMIC_POOL_PHOTO_SUMMARY_FILE,
    DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION,
)
from biominer.bioclip.dynamic_pool_contracts import (
    DYNAMIC_POOL_MEMBERS_FILE,
    DYNAMIC_POOL_MEMBER_SCHEMA_VERSION,
    DYNAMIC_POOL_PLANS_FILE,
    DYNAMIC_POOL_PLAN_SCHEMA_VERSION,
    DYNAMIC_POOL_SUMMARY_FILE,
    DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION,
)
from biominer.bioclip.family_geo_candidates import (
    FAMILY_GEO_CANDIDATE_FILE,
    FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
)
from biominer.integration.product_handoff import (
    finalize_product_manifest,
    normalize_product_artifacts,
    validate_fingerprint,
    validate_git_sha,
    validate_normalized_product_artifacts,
    validate_product_manifest_identity,
    write_product_manifest,
)


TAXALENS_POOL_HANDOFF_SCHEMA_VERSION = "biominer-taxalens-dynamic-pool-handoff-v1.0.0"
TAXALENS_POOL_HANDOFF_FILE = "taxalens_dynamic_pool_handoff.json"
TAXALENS_REPOSITORY = "karikris/taxalens"
TAXALENS_PINNED_COMMIT = "c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc"
TAXALENS_REQUIRED_ARTIFACT_ROLES = (
    "candidate_scores",
    "photo_summaries",
    "pool_plans",
    "pool_members",
    "pool_summaries",
    "candidate_sets",
    "review_sampling_frame",
    "quality_sidecar",
    "geographic_cells",
)
TAXALENS_TARGET_CONTRACTS = {
    "geographic_impact_export": "taxalens-geographic-impact-export:v1.0.0",
    "verification_campaign": "taxalens-verification-campaign:v1.0.0",
    "flickr_verification_source": "taxalens-flickr-verification-source:v1.1.0",
    "quality_snapshot": "taxalens-verification-quality-snapshot:v1.1.0",
    "reviewed_labels": "reviewed-labels-v2",
    "reviewed_labels_filename": "flickr_reviewed_labels_v2.parquet",
}
TAXALENS_ROLE_DEFAULTS = {
    "candidate_scores": (
        DYNAMIC_POOL_CANDIDATE_SCORES_FILE,
        DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
    ),
    "photo_summaries": (
        DYNAMIC_POOL_PHOTO_SUMMARY_FILE,
        DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION,
    ),
    "pool_plans": (
        DYNAMIC_POOL_PLANS_FILE,
        DYNAMIC_POOL_PLAN_SCHEMA_VERSION,
    ),
    "pool_members": (
        DYNAMIC_POOL_MEMBERS_FILE,
        DYNAMIC_POOL_MEMBER_SCHEMA_VERSION,
    ),
    "pool_summaries": (
        DYNAMIC_POOL_SUMMARY_FILE,
        DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION,
    ),
    "candidate_sets": (
        FAMILY_GEO_CANDIDATE_FILE,
        FAMILY_GEO_CANDIDATE_SCHEMA_VERSION,
    ),
    "review_sampling_frame": (
        "taxalens_review_sampling_frame.parquet",
        "biominer-taxalens-review-sampling-frame-v1.0.0",
    ),
    "quality_sidecar": (
        "taxalens_quality_sidecar.json",
        "biominer-taxalens-quality-sidecar-v1.0.0",
    ),
    "geographic_cells": (
        "taxalens_geographic_impact_cells.parquet",
        "biominer-taxalens-geographic-cells-v1.0.0",
    ),
}

_IDENTITY_PREFIX = "taxalens-pool-handoff:"
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "handoff_id",
        "manifest_fingerprint",
        "created_at",
        "producer",
        "consumer",
        "run_id",
        "registry_version",
        "source_snapshot_fingerprints",
        "model_fingerprint",
        "preprocessing_fingerprint",
        "target_contracts",
        "artifact_roles",
        "artifacts",
        "field_projections",
        "evidence_maturity",
        "authority_boundary",
    }
)


def build_taxalens_pool_handoff(
    *,
    producer_commit: str,
    created_at: str | datetime,
    run_id: str,
    registry_version: str,
    source_snapshot_fingerprints: Sequence[str],
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    artifacts: Sequence[Mapping[str, object]],
    completed_review_count: int,
    quality_estimate_available: bool,
    quality_unavailable_reason: str | None,
) -> dict[str, object]:
    """Build a deterministic, fail-closed TaxaLens product manifest."""

    producer_sha = validate_git_sha(producer_commit, field="producer_commit")
    if (
        isinstance(completed_review_count, bool)
        or not isinstance(completed_review_count, int)
        or completed_review_count < 0
    ):
        raise ValueError("completed_review_count must be a nonnegative integer")
    if quality_estimate_available and completed_review_count == 0:
        raise ValueError("quality estimates require completed human reviews")
    if quality_estimate_available and quality_unavailable_reason is not None:
        raise ValueError("available quality estimate cannot have an unavailable reason")
    if not quality_estimate_available:
        quality_unavailable_reason = _required_text(
            quality_unavailable_reason, field="quality_unavailable_reason"
        )
    snapshots = sorted(
        validate_fingerprint(value, field="source_snapshot_fingerprints")
        for value in source_snapshot_fingerprints
    )
    if not snapshots or len(snapshots) != len(set(snapshots)):
        raise ValueError("source snapshot fingerprints must be nonempty and unique")
    normalized_artifacts = normalize_product_artifacts(
        artifacts,
        required_roles=TAXALENS_REQUIRED_ARTIFACT_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit=producer_sha,
    )
    _validate_role_defaults(normalized_artifacts)
    _validate_maturity_artifacts(
        normalized_artifacts,
        completed_review_count=completed_review_count,
        quality_estimate_available=quality_estimate_available,
    )
    review_available = completed_review_count > 0
    body: dict[str, object] = {
        "schema_version": TAXALENS_POOL_HANDOFF_SCHEMA_VERSION,
        "created_at": _utc_instant(created_at),
        "producer": {
            "repository": "karikris/BioMiner",
            "commit": producer_sha,
        },
        "consumer": {
            "repository": TAXALENS_REPOSITORY,
            "commit": TAXALENS_PINNED_COMMIT,
        },
        "run_id": _required_text(run_id, field="run_id"),
        "registry_version": _required_text(registry_version, field="registry_version"),
        "source_snapshot_fingerprints": snapshots,
        "model_fingerprint": validate_fingerprint(
            model_fingerprint, field="model_fingerprint"
        ),
        "preprocessing_fingerprint": validate_fingerprint(
            preprocessing_fingerprint, field="preprocessing_fingerprint"
        ),
        "target_contracts": dict(TAXALENS_TARGET_CONTRACTS),
        "artifact_roles": list(TAXALENS_REQUIRED_ARTIFACT_ROLES),
        "artifacts": normalized_artifacts,
        "field_projections": _field_projections(),
        "evidence_maturity": {
            "model_scores": {
                "status": "available",
                "label": "provisional_raw_score",
                "unavailable_reason": None,
                "probability_semantics": False,
                "release_authorizing": False,
            },
            "human_review": {
                "status": "available" if review_available else "not_evaluated",
                "label": "human_reviewed_flickr_labels" if review_available else None,
                "completed_review_count": completed_review_count,
                "unavailable_reason": (
                    None
                    if review_available
                    else "no completed TaxaLens human reviews are present in this handoff"
                ),
                "release_authorizing": False,
            },
            "quality_estimate": {
                "status": "available" if quality_estimate_available else "unavailable",
                "label": "human_reviewed_flickr_labels"
                if quality_estimate_available
                else None,
                "unavailable_reason": quality_unavailable_reason,
                "zero_review_is_zero_quality": False,
                "release_authorizing": False,
            },
            "release": {
                "status": "not_evaluated",
                "label": None,
                "release_ready": False,
                "scientific_claim_allowed": False,
                "unavailable_reason": (
                    "BioMiner evidence handoff cannot authorize TaxaLens occurrence release"
                ),
            },
        },
        "authority_boundary": {
            "adapter_owner": TAXALENS_REPOSITORY,
            "biominer_application_writes_allowed": False,
            "taxalens_identifiers_are_assigned_downstream": True,
            "review_occurrence_is_release": False,
            "scientific_claim_allowed": False,
        },
    }
    manifest = finalize_product_manifest(body, identity_prefix=_IDENTITY_PREFIX)
    validate_taxalens_pool_handoff(manifest)
    return manifest


def validate_taxalens_pool_handoff(manifest: Mapping[str, object]) -> None:
    """Validate exact pins, roles, maturity, and content-derived identities."""

    if set(manifest) != _TOP_LEVEL_FIELDS:
        raise ValueError("TaxaLens handoff top-level fields differ from the contract")
    if manifest["schema_version"] != TAXALENS_POOL_HANDOFF_SCHEMA_VERSION:
        raise ValueError("unsupported TaxaLens pool handoff schema version")
    if manifest["consumer"] != {
        "repository": TAXALENS_REPOSITORY,
        "commit": TAXALENS_PINNED_COMMIT,
    }:
        raise ValueError("TaxaLens consumer pin moved without a contract revision")
    producer = _mapping(manifest["producer"], field="producer")
    if producer.get("repository") != "karikris/BioMiner":
        raise ValueError("TaxaLens handoff producer repository is invalid")
    producer_commit = validate_git_sha(producer.get("commit"), field="producer.commit")
    if manifest["target_contracts"] != TAXALENS_TARGET_CONTRACTS:
        raise ValueError("TaxaLens target contract versions differ")
    if manifest["artifact_roles"] != list(TAXALENS_REQUIRED_ARTIFACT_ROLES):
        raise ValueError("TaxaLens artifact role order differs")
    artifacts_value = manifest["artifacts"]
    if not isinstance(artifacts_value, Sequence) or isinstance(
        artifacts_value, (str, bytes)
    ):
        raise ValueError("TaxaLens artifacts must be an array")
    artifacts = [_mapping(value, field="artifacts[]") for value in artifacts_value]
    artifacts = validate_normalized_product_artifacts(
        artifacts,
        required_roles=TAXALENS_REQUIRED_ARTIFACT_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit=producer_commit,
    )
    _validate_role_defaults(artifacts)
    if _utc_instant(manifest["created_at"]) != manifest["created_at"]:
        raise ValueError("TaxaLens created_at is not a canonical UTC instant")
    _required_text(manifest["run_id"], field="run_id")
    _required_text(manifest["registry_version"], field="registry_version")
    snapshot_values = manifest["source_snapshot_fingerprints"]
    if not isinstance(snapshot_values, Sequence) or isinstance(
        snapshot_values, (str, bytes)
    ):
        raise ValueError("source snapshot fingerprints must be an array")
    snapshots = [
        validate_fingerprint(value, field="source_snapshot_fingerprints")
        for value in snapshot_values
    ]
    if not snapshots or snapshots != sorted(set(snapshots)):
        raise ValueError("source snapshot fingerprints are not canonical")
    validate_fingerprint(manifest["model_fingerprint"], field="model_fingerprint")
    validate_fingerprint(
        manifest["preprocessing_fingerprint"], field="preprocessing_fingerprint"
    )
    maturity = _mapping(manifest["evidence_maturity"], field="evidence_maturity")
    review = _mapping(maturity.get("human_review"), field="human_review")
    quality = _mapping(maturity.get("quality_estimate"), field="quality_estimate")
    release = _mapping(maturity.get("release"), field="release")
    completed = review.get("completed_review_count")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ValueError("completed review count is invalid")
    quality_available = quality.get("status") == "available"
    if completed == 0 and quality_available:
        raise ValueError("zero reviews cannot yield an available quality estimate")
    _validate_maturity_artifacts(
        artifacts,
        completed_review_count=completed,
        quality_estimate_available=quality_available,
    )
    model_scores = _mapping(maturity.get("model_scores"), field="model_scores")
    if model_scores != {
        "status": "available",
        "label": "provisional_raw_score",
        "unavailable_reason": None,
        "probability_semantics": False,
        "release_authorizing": False,
    }:
        raise ValueError("TaxaLens model score maturity differs")
    expected_review_status = "available" if completed else "not_evaluated"
    if (
        review.get("status") != expected_review_status
        or review.get("label")
        != ("human_reviewed_flickr_labels" if completed else None)
        or review.get("release_authorizing") is not False
        or (completed == 0 and not review.get("unavailable_reason"))
        or (completed > 0 and review.get("unavailable_reason") is not None)
    ):
        raise ValueError("TaxaLens human-review maturity differs")
    if (
        quality.get("status") != ("available" if quality_available else "unavailable")
        or quality.get("label")
        != ("human_reviewed_flickr_labels" if quality_available else None)
        or quality.get("zero_review_is_zero_quality") is not False
        or quality.get("release_authorizing") is not False
        or (quality_available and quality.get("unavailable_reason") is not None)
        or (not quality_available and not quality.get("unavailable_reason"))
    ):
        raise ValueError("TaxaLens quality-estimate maturity differs")
    if (
        release.get("release_ready") is not False
        or release.get("scientific_claim_allowed") is not False
        or _mapping(manifest["authority_boundary"], field="authority_boundary").get(
            "review_occurrence_is_release"
        )
        is not False
    ):
        raise ValueError("TaxaLens handoff cannot authorize occurrence release")
    if release != {
        "status": "not_evaluated",
        "label": None,
        "release_ready": False,
        "scientific_claim_allowed": False,
        "unavailable_reason": (
            "BioMiner evidence handoff cannot authorize TaxaLens occurrence release"
        ),
    }:
        raise ValueError("TaxaLens release maturity differs")
    if manifest["authority_boundary"] != {
        "adapter_owner": TAXALENS_REPOSITORY,
        "biominer_application_writes_allowed": False,
        "taxalens_identifiers_are_assigned_downstream": True,
        "review_occurrence_is_release": False,
        "scientific_claim_allowed": False,
    }:
        raise ValueError("TaxaLens authority boundary differs")
    if manifest["field_projections"] != _field_projections():
        raise ValueError("TaxaLens field projections differ")
    validate_product_manifest_identity(manifest, identity_prefix=_IDENTITY_PREFIX)


def write_taxalens_pool_handoff(
    manifest: Mapping[str, object], output_dir: str | Path
) -> Path:
    """Validate and atomically write the TaxaLens manifest filename."""

    validate_taxalens_pool_handoff(manifest)
    return write_product_manifest(
        manifest, Path(output_dir) / TAXALENS_POOL_HANDOFF_FILE
    )


def _validate_role_defaults(artifacts: Sequence[Mapping[str, object]]) -> None:
    for artifact in artifacts:
        role = str(artifact["role"])
        filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
        if artifact["schema_version"] != schema_version:
            raise ValueError(f"TaxaLens role {role!r} schema version differs")
        relative_path = artifact["relative_path"]
        if relative_path is not None and Path(str(relative_path)).name != filename:
            raise ValueError(f"TaxaLens role {role!r} filename differs")


def _validate_maturity_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    completed_review_count: int,
    quality_estimate_available: bool,
) -> None:
    by_role = {str(value["role"]): value for value in artifacts}
    if by_role["candidate_scores"]["availability"] != "available":
        raise ValueError("TaxaLens handoff requires available candidate scores")
    if by_role["candidate_scores"]["evidence_maturity_label"] != (
        "provisional_raw_score"
    ):
        raise ValueError("candidate scores must remain provisional raw scores")
    quality_artifact_available = by_role["quality_sidecar"]["availability"] == (
        "available"
    )
    if quality_estimate_available and not quality_artifact_available:
        raise ValueError("available quality estimate requires a quality sidecar")
    if completed_review_count == 0 and quality_artifact_available:
        raise ValueError("quality sidecar cannot be available before human review")


def _field_projections() -> dict[str, object]:
    return {
        "candidate_evidence": {
            "source_role": "candidate_scores",
            "identity_fields": [
                "flickr_photo_id",
                "organism_unit_id",
                "candidate_accepted_taxon_key",
                "candidate_scientific_name",
            ],
            "score_fields": [
                "global_prototype_raw_score",
                "local_prototype_raw_score",
                "nearest_reference_raw_score",
                "top_k_reference_raw_score",
                "fused_raw_score",
                "margin_to_next_raw",
                "candidate_rank",
                "probability_availability",
            ],
            "target_contract": "flickr_verification_source",
            "semantics": "post-decision evidence; raw scores are not probabilities",
        },
        "pool_provenance": {
            "source_roles": ["pool_plans", "pool_members", "pool_summaries"],
            "fields": [
                "plan_id",
                "plan_fingerprint",
                "pool_id",
                "pool_fingerprint",
                "pool_role",
                "reference_observation_id",
                "geographic_scope_id",
                "distance_km",
                "fallback_level",
                "selection_rank",
                "inclusion_reason",
            ],
            "target_contract": "flickr_verification_source",
        },
        "review_sampling": {
            "source_role": "review_sampling_frame",
            "fields": [
                "sampling_plan_id",
                "sampling_purpose",
                "sampling_design",
                "representative",
                "blind_review",
                "selection_seed",
                "independent_unit",
                "grouping_keys",
                "sampling_stratum_id",
                "inclusion_probability",
                "dataset_partition",
            ],
            "target_contract": "verification_campaign",
        },
        "quality": {
            "source_role": "quality_sidecar",
            "availability_is_explicit": True,
            "zero_review_is_zero_quality": False,
            "target_contract": "quality_snapshot",
        },
        "geographic_impact": {
            "source_role": "geographic_cells",
            "fields": [
                "spatial_resolution",
                "spatial_cell_id",
                "country_code",
                "admin1",
                "centroid_latitude",
                "centroid_longitude",
                "baseline_union_count",
                "flickr_candidate_count",
                "reviewed_positive_count",
                "reviewed_negative_count",
                "pending_count",
                "release_ready_count",
                "baseline_only_cell",
                "candidate_only_cell",
                "data_deficient_state",
            ],
            "target_contract": "geographic_impact_export",
            "missing_baseline_is_biological_absence": False,
            "flickr_candidates_are_occurrences": False,
        },
    }


def _utc_instant(value: str | datetime) -> str:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _required_text(value: object, *, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


__all__ = [
    "TAXALENS_PINNED_COMMIT",
    "TAXALENS_POOL_HANDOFF_FILE",
    "TAXALENS_POOL_HANDOFF_SCHEMA_VERSION",
    "TAXALENS_REPOSITORY",
    "TAXALENS_REQUIRED_ARTIFACT_ROLES",
    "TAXALENS_ROLE_DEFAULTS",
    "TAXALENS_TARGET_CONTRACTS",
    "build_taxalens_pool_handoff",
    "validate_taxalens_pool_handoff",
    "write_taxalens_pool_handoff",
]
