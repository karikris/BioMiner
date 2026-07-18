"""ButterflyLens anti-corruption manifest for dynamic-pool evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from biominer.bioclip.dynamic_pool_scores import (
    DYNAMIC_POOL_CANDIDATE_SCORES_FILE,
    DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
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


BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION = (
    "biominer-butterflylens-dynamic-pool-handoff-v1.0.0"
)
BUTTERFLYLENS_POOL_HANDOFF_FILE = "butterflylens_dynamic_pool_handoff.json"
BUTTERFLYLENS_REPOSITORY = "karikris/ButterflyLens"
BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT = (
    "fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3"
)
BUTTERFLYLENS_PINNED_COMMIT = "1cea643623f2f20a2bea72afc754c7b194db3278"
BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES = (
    "project",
    "run",
    "flickr_source_records",
    "media_objects",
    "model_evidence",
    "geographic_impact",
    "review_campaign_inputs",
    "review_assignment_inputs",
    "classification_maturity",
    "release_state",
)
BUTTERFLYLENS_TARGET_CONTRACTS = {
    "project": "butterflylens-project:v1.0.0",
    "run": "butterflylens-run:v1.0.0",
    "evidence_fingerprint": "butterflylens-evidence-fingerprint:v1.1.0",
    "classification_maturity": "butterflylens-classification-maturity:v1.0.0",
    "geographic_impact_cell": "butterflylens-geographic-impact-cell:v1.0.0",
    "geographic_impact_snapshot": (
        "butterflylens-geographic-impact-snapshot:v1.0.0"
    ),
    "verification_campaign": "butterflylens-verification-campaign:v1.0.0",
    "verification_assignment": "butterflylens-verification-assignment:v1.0.0",
    "verification_event": "butterflylens-verification-event:v1.0.0",
    "verification_consensus": "butterflylens-verification-consensus:v1.0.0",
    "model_evidence_migration": "20260717212553_model_evidence_schema.sql",
    "release_candidate_migration": "20260717214211_map_impact_schema.sql",
    "rls_policy_migration": "20260717215002_rls_role_policies.sql",
    "repeated_assignment_migration": (
        "20260718013600_repeated_independent_assignments.sql"
    ),
    "blind_disclosure_migration": "20260718014600_blind_review_disclosure.sql",
    "append_only_review_migration": (
        "20260718015500_append_only_review_submission.sql"
    ),
}
BUTTERFLYLENS_ROLE_DEFAULTS = {
    "project": (
        "butterflylens_project.json",
        "biominer-butterflylens-project-projection-v1.0.0",
    ),
    "run": (
        "butterflylens_run.json",
        "biominer-butterflylens-run-projection-v1.0.0",
    ),
    "flickr_source_records": (
        "butterflylens_flickr_source_records.parquet",
        "biominer-butterflylens-flickr-source-v1.0.0",
    ),
    "media_objects": (
        "butterflylens_media_objects.parquet",
        "biominer-butterflylens-media-object-v1.0.0",
    ),
    "model_evidence": (
        DYNAMIC_POOL_CANDIDATE_SCORES_FILE,
        DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
    ),
    "geographic_impact": (
        "butterflylens_geographic_impact_cells.parquet",
        "biominer-butterflylens-geographic-impact-v1.0.0",
    ),
    "review_campaign_inputs": (
        "butterflylens_review_campaign_inputs.json",
        "biominer-butterflylens-review-campaign-input-v1.0.0",
    ),
    "review_assignment_inputs": (
        "butterflylens_review_assignment_inputs.parquet",
        "biominer-butterflylens-review-assignment-input-v1.0.0",
    ),
    "classification_maturity": (
        "butterflylens_classification_maturity.parquet",
        "biominer-butterflylens-classification-maturity-projection-v1.0.0",
    ),
    "release_state": (
        "butterflylens_release_state.json",
        "biominer-butterflylens-release-state-v1.0.0",
    ),
}

_IDENTITY_PREFIX = "butterflylens-pool-handoff:"
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "handoff_id",
        "manifest_fingerprint",
        "created_at",
        "producer",
        "consumer",
        "consumer_compatibility",
        "project_id",
        "run_id",
        "registry_version",
        "source_snapshot_fingerprints",
        "model_fingerprint",
        "preprocessing_fingerprint",
        "target_contracts",
        "artifact_roles",
        "artifacts",
        "field_projections",
        "review_assignment_policy",
        "release_state",
        "authority_boundary",
    }
)


def build_butterflylens_pool_handoff(
    *,
    producer_commit: str,
    created_at: str | datetime,
    project_id: str,
    run_id: str,
    registry_version: str,
    source_snapshot_fingerprints: Sequence[str],
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a deterministic evidence manifest without database authority."""

    producer_sha = validate_git_sha(producer_commit, field="producer_commit")
    snapshots = sorted(
        validate_fingerprint(value, field="source_snapshot_fingerprints")
        for value in source_snapshot_fingerprints
    )
    if not snapshots or snapshots != sorted(set(snapshots)):
        raise ValueError("source snapshot fingerprints must be nonempty and unique")
    normalized_artifacts = normalize_product_artifacts(
        artifacts,
        required_roles=BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit=producer_sha,
    )
    _validate_role_defaults(normalized_artifacts)
    _validate_evidence_roles(normalized_artifacts)
    body: dict[str, object] = {
        "schema_version": BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION,
        "created_at": _utc_instant(created_at),
        "producer": {
            "repository": "karikris/BioMiner",
            "commit": producer_sha,
        },
        "consumer": {
            "repository": BUTTERFLYLENS_REPOSITORY,
            "commit": BUTTERFLYLENS_PINNED_COMMIT,
        },
        "consumer_compatibility": _consumer_compatibility(),
        "project_id": _required_text(project_id, field="project_id"),
        "run_id": _required_text(run_id, field="run_id"),
        "registry_version": _required_text(
            registry_version, field="registry_version"
        ),
        "source_snapshot_fingerprints": snapshots,
        "model_fingerprint": validate_fingerprint(
            model_fingerprint, field="model_fingerprint"
        ),
        "preprocessing_fingerprint": validate_fingerprint(
            preprocessing_fingerprint, field="preprocessing_fingerprint"
        ),
        "target_contracts": dict(BUTTERFLYLENS_TARGET_CONTRACTS),
        "artifact_roles": list(BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES),
        "artifacts": normalized_artifacts,
        "field_projections": _field_projections(),
        "review_assignment_policy": _review_assignment_policy(),
        "release_state": _release_state(),
        "authority_boundary": _authority_boundary(),
    }
    manifest = finalize_product_manifest(body, identity_prefix=_IDENTITY_PREFIX)
    validate_butterflylens_pool_handoff(manifest)
    return manifest


def validate_butterflylens_pool_handoff(manifest: Mapping[str, object]) -> None:
    """Validate the exact ButterflyLens pin and anti-corruption boundary."""

    if set(manifest) != _TOP_LEVEL_FIELDS:
        raise ValueError("ButterflyLens handoff top-level fields differ")
    if manifest["schema_version"] != BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION:
        raise ValueError("unsupported ButterflyLens pool handoff schema version")
    if manifest["consumer"] != {
        "repository": BUTTERFLYLENS_REPOSITORY,
        "commit": BUTTERFLYLENS_PINNED_COMMIT,
    }:
        raise ValueError("ButterflyLens consumer pin moved without review")
    if manifest["consumer_compatibility"] != _consumer_compatibility():
        raise ValueError("ButterflyLens pin compatibility decision differs")
    producer = _mapping(manifest["producer"], field="producer")
    if producer.get("repository") != "karikris/BioMiner":
        raise ValueError("ButterflyLens producer repository is invalid")
    producer_commit = validate_git_sha(producer.get("commit"), field="producer.commit")
    if manifest["target_contracts"] != BUTTERFLYLENS_TARGET_CONTRACTS:
        raise ValueError("ButterflyLens target contract versions differ")
    if manifest["artifact_roles"] != list(BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES):
        raise ValueError("ButterflyLens artifact role order differs")
    artifacts_value = manifest["artifacts"]
    if not isinstance(artifacts_value, Sequence) or isinstance(
        artifacts_value, (str, bytes)
    ):
        raise ValueError("ButterflyLens artifacts must be an array")
    artifacts = validate_normalized_product_artifacts(
        [_mapping(value, field="artifacts[]") for value in artifacts_value],
        required_roles=BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit=producer_commit,
    )
    _validate_role_defaults(artifacts)
    _validate_evidence_roles(artifacts)
    if _utc_instant(manifest["created_at"]) != manifest["created_at"]:
        raise ValueError("ButterflyLens created_at is not canonical UTC")
    _required_text(manifest["project_id"], field="project_id")
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
    if manifest["field_projections"] != _field_projections():
        raise ValueError("ButterflyLens field projections differ")
    if manifest["review_assignment_policy"] != _review_assignment_policy():
        raise ValueError("ButterflyLens review assignment boundary differs")
    if manifest["release_state"] != _release_state():
        raise ValueError("ButterflyLens release state was escalated")
    if manifest["authority_boundary"] != _authority_boundary():
        raise ValueError("ButterflyLens database authority boundary differs")
    validate_product_manifest_identity(manifest, identity_prefix=_IDENTITY_PREFIX)


def write_butterflylens_pool_handoff(
    manifest: Mapping[str, object], output_dir: str | Path
) -> Path:
    """Validate and atomically write the ButterflyLens manifest filename."""

    validate_butterflylens_pool_handoff(manifest)
    return write_product_manifest(
        manifest, Path(output_dir) / BUTTERFLYLENS_POOL_HANDOFF_FILE
    )


def _validate_role_defaults(artifacts: Sequence[Mapping[str, object]]) -> None:
    for artifact in artifacts:
        role = str(artifact["role"])
        filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
        if artifact["schema_version"] != schema_version:
            raise ValueError(f"ButterflyLens role {role!r} schema version differs")
        relative_path = artifact["relative_path"]
        if relative_path is not None and Path(str(relative_path)).name != filename:
            raise ValueError(f"ButterflyLens role {role!r} filename differs")


def _validate_evidence_roles(artifacts: Sequence[Mapping[str, object]]) -> None:
    by_role = {str(value["role"]): value for value in artifacts}
    for required in ("project", "run", "flickr_source_records", "model_evidence"):
        if by_role[required]["availability"] != "available":
            raise ValueError(f"ButterflyLens handoff requires available {required}")
    if by_role["model_evidence"]["evidence_maturity_label"] != (
        "provisional_raw_score"
    ):
        raise ValueError("ButterflyLens model evidence must remain raw score evidence")
    if by_role["release_state"]["evidence_maturity_label"] == (
        "final_release_status"
    ):
        raise ValueError("BioMiner cannot hand off final ButterflyLens release status")


def _consumer_compatibility() -> dict[str, object]:
    return {
        "previous_audited_commit": BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT,
        "current_audited_commit": BUTTERFLYLENS_PINNED_COMMIT,
        "decision": "compatible_additive_with_stricter_review_controls",
        "wire_schema_breaking_changes": False,
        "database_adapter_changes_required": True,
        "required_adapter_changes": [
            "create repeated independent assignments under downstream policy",
            "keep model evidence and peer decisions blind until review submission",
            "submit append-only review events with correction lineage",
        ],
        "silent_pin_movement": False,
    }


def _review_assignment_policy() -> dict[str, object]:
    return {
        "policy_version": "repeated-independent-v1",
        "minimum_independent_reviewers": 2,
        "blind_until_decision": True,
        "reviewer_identity_assigned_downstream": True,
        "required_reviewer_role_assigned_downstream": True,
        "assignments_created_by_biominer": False,
        "review_events_are_append_only": True,
        "corrections_require_supersedes_lineage": True,
        "model_vote_included_in_consensus": False,
    }


def _release_state() -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "candidate_state": "blocked",
        "all_release_gates_passed": False,
        "release_ready": False,
        "scientific_claim_allowed": False,
        "unavailable_reason": (
            "ButterflyLens must evaluate consensus, quality, rights, geography, "
            "expert and authorization gates downstream"
        ),
    }


def _authority_boundary() -> dict[str, object]:
    return {
        "adapter_owner": BUTTERFLYLENS_REPOSITORY,
        "database_primary_keys_in_handoff": False,
        "reviewer_identity_in_handoff": False,
        "service_role_credentials_in_handoff": False,
        "biominer_database_writes_allowed": False,
        "database_ids_assigned_downstream": True,
        "service_role_ingestion_enforced_downstream": True,
        "table_grants_enforced_downstream": True,
        "row_level_security_enforced_downstream": True,
        "rls_bypass_claimed_by_biominer": False,
    }


def _field_projections() -> dict[str, object]:
    return {
        "project": {
            "source_role": "project",
            "target_contract": "project",
            "fields": [
                "project_id",
                "geographic_scope",
                "taxon_scope",
                "discovery_scope",
                "data_policy_version",
                "consent_policy_version",
            ],
        },
        "run": {
            "source_role": "run",
            "target_contract": "run",
            "fields": [
                "run_id",
                "project_id",
                "run_kind",
                "mode",
                "status",
                "engine.repository",
                "engine.commit",
                "engine.interface_version",
                "engine.command",
                "input_fingerprints",
                "artifacts",
            ],
        },
        "flickr_identity": {
            "source_roles": ["flickr_source_records", "media_objects"],
            "fields": [
                "flickr_record_id",
                "flickr_photo_id",
                "source_record_fingerprint",
                "media_object_id",
                "media_sha256",
                "media_byte_count",
                "media_type",
                "rights_fingerprint",
                "duplicate_group_id",
                "owner_group_id",
                "observation_group_id",
            ],
            "database_primary_keys_excluded": True,
        },
        "model_evidence": {
            "source_role": "model_evidence",
            "evidence_kinds": [
                "yoloe_route",
                "bioclip_embedding",
                "prototype",
                "candidate_score",
            ],
            "statuses": ["completed", "failed", "blocked", "skipped_unfinished"],
            "fields": [
                "model_id",
                "model_revision",
                "model_weights_sha256",
                "model_fingerprint",
                "preprocessing_fingerprint",
                "input_fingerprint",
                "output_fingerprint",
                "fused_raw_score",
                "probability_availability",
                "calibrator_fingerprint",
            ],
            "raw_score_is_probability": False,
        },
        "geographic_impact": {
            "source_role": "geographic_impact",
            "target_contract": "geographic_impact_cell",
            "count_states": [
                "available",
                "unavailable",
                "withheld",
                "not_applicable",
            ],
            "candidate_only_is_occurrence": False,
            "missing_baseline_is_absence": False,
        },
        "review_inputs": {
            "source_roles": [
                "review_campaign_inputs",
                "review_assignment_inputs",
            ],
            "target_contracts": [
                "verification_campaign",
                "verification_assignment",
            ],
            "fields": [
                "campaign_id",
                "item_id",
                "review_round",
                "assignment_reason",
                "independence_group_key",
                "blind",
                "assignment_policy_version",
            ],
            "excluded_fields": [
                "reviewer_id",
                "reviewer_account_id",
                "reviewer_database_pk",
                "assignment_database_pk",
            ],
        },
        "release": {
            "source_roles": ["classification_maturity", "release_state"],
            "target_contract": "classification_maturity",
            "required_maturity_fields": [
                "butterfly_detected",
                "species_candidate_available",
                "community_reviewed",
                "quality_estimate_available",
                "expert_reviewed",
                "release_ready",
            ],
            "producer_release_authority": False,
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
    "BUTTERFLYLENS_PINNED_COMMIT",
    "BUTTERFLYLENS_POOL_HANDOFF_FILE",
    "BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION",
    "BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT",
    "BUTTERFLYLENS_REPOSITORY",
    "BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES",
    "BUTTERFLYLENS_ROLE_DEFAULTS",
    "BUTTERFLYLENS_TARGET_CONTRACTS",
    "build_butterflylens_pool_handoff",
    "validate_butterflylens_pool_handoff",
    "write_butterflylens_pool_handoff",
]
