from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import polars as pl

from biominer.candidates.regional_union import validate_regional_candidate_species
from biominer.common.semantic_hash import (
    canonical_semantic_bytes,
    canonical_semantic_fingerprint,
)
from biominer.references.admission import (
    DEFAULT_REFERENCE_ADMISSION_MODE,
    REFERENCE_ADMISSION_MODES,
    ReferenceAdmissionPolicy,
    strict_reference_admission_policy,
)
from biominer.references.admission_compiler import (
    validate_reference_provisional_support,
)
from biominer.references.deduplication import (
    validate_reference_media_deduplication_artifacts,
)
from biominer.references.planner import make_reference_candidate_union_id
from biominer.references.review import (
    resolve_reference_review_statuses,
    validate_reference_review_queue_source_bindings,
)
from biominer.references.support_admission import (
    SupportAdmissionEvidence,
    evaluate_support_admission,
)
from biominer.references.schemas import (
    REFERENCE_ROUTES,
    validate_reference_acquisition_plan,
    validate_reference_acquisition_selections,
    validate_reference_media_candidates,
    validate_reference_media_duplicate_relationships,
    validate_reference_media_objects,
    validate_reference_observations,
    validate_reference_review_decisions,
    validate_reference_review_queue,
)
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION = (
    "reference-bank-split-assignments-v1.0.0"
)
REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION = "reference-support-manifest-v3.0.0"
REFERENCE_BANK_SUMMARY_SCHEMA_VERSION = "reference-bank-summary-v2.0.0"
REFERENCE_BANK_READINESS_SCHEMA_VERSION = "reference-bank-readiness-v3.0.0"
REFERENCE_BANK_READINESS_POLICY_SCHEMA_VERSION = (
    "reference-bank-readiness-policy-v1.0.0"
)
LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION = (
    "reference-support-manifest-v2.0.0"
)
REFERENCE_SUPPORT_MANIFEST_MIGRATION_SCHEMA_VERSION = (
    "reference-support-manifest-migration-v1.0.0"
)
REFERENCE_MODEL_INPUT_IDENTITY_SCHEMA_VERSION = (
    "reference-model-input-identity-v2.0.0"
)

REFERENCE_SUPPORT_MANIFEST_FILE = "reference_support_manifest.parquet"
REFERENCE_BANK_SUMMARY_FILE = "reference_bank_summary.parquet"
REFERENCE_BANK_READINESS_FILE = "reference_bank_readiness.json"

READINESS_STATUSES = frozenset(
    {
        "ready",
        "ready_provisional",
        "ready_with_documented_shortfalls",
        "awaiting_manual_review",
        "blocked_licence",
        "blocked_missing_target_support",
        "invalid",
    }
)
STRICT_PERMITTING_READINESS_STATUSES = frozenset(
    {"ready", "ready_with_documented_shortfalls"}
)
PERMITTING_READINESS_STATUSES = frozenset(
    {*STRICT_PERMITTING_READINESS_STATUSES, "ready_provisional"}
)
READINESS_CHECK_STATUSES = frozenset({"passed", "failed", "pending", "warning"})
REFERENCE_SUPPORT_SPLITS = frozenset(
    {"support_train", "model_selection", "calibration", "final_test"}
)
REFERENCE_IDENTITY_EVIDENCE_BASES = frozenset(
    {"human_verified", "gbif_provider_asserted", "none"}
)
REFERENCE_ADMISSION_STATUSES = frozenset(
    {"admitted", "excluded", "review_required"}
)
REFERENCE_ROUTE_EVIDENCE_BASES = frozenset(
    {"human_verified_review", "yoloe", "none"}
)
REFERENCE_HUMAN_REVIEW_STATUSES = frozenset(
    {"not_requested", "pending", "in_review", "completed", "rejected", "conflict"}
)
_COMMON_REQUIRED_CHECK_IDS = (
    "artifact_integrity",
    "target_adult_minimum",
    "competitor_minima",
    "geographic_cluster_coverage",
    "larval_route_separation",
    "pinned_specimen_separation",
    "support_admission_policy_satisfied",
    "provider_assertion_integrity",
    "automated_reference_qa_passed",
    "reference_routes_separated",
    "provisional_support_declared",
    "statistical_audit_plan_available",
    "human_rejections_respected",
    "duplicate_groups_resolved",
    "licences_accepted",
    "source_attribution_complete",
    "split_group_separation",
    "model_building_inputs_available",
)


def _required_check_ids(admission_mode: str) -> tuple[str, ...]:
    if admission_mode == "human_verified_strict":
        index = _COMMON_REQUIRED_CHECK_IDS.index("human_rejections_respected") + 1
        return (
            *_COMMON_REQUIRED_CHECK_IDS[:index],
            "strict_support_only",
            *_COMMON_REQUIRED_CHECK_IDS[index:],
        )
    return _COMMON_REQUIRED_CHECK_IDS
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUMMARY_SORT = (
    "reference_bank_version",
    "accepted_taxon_key",
    "geo_cluster_id",
    "route",
    "life_stage",
    "visual_domain",
    "support_split",
)
_SUPPORT_SORT = (
    "accepted_taxon_key",
    "geo_cluster_id",
    "route",
    "support_split",
    "reference_media_id",
)

# Semantic identity is an explicit projection, not a hash of every physical
# column. Operational audit fields remain persisted and validated, but a no-op
# rebuild or publication must not create a different frozen bank.
_READINESS_INPUT_SEMANTIC_FIELDS: dict[str, tuple[str, ...]] = {
    "candidate_species": (
        "schema_version",
        "candidate_set_id",
        "target_accepted_taxon_key",
        "geo_cluster_id",
        "candidate_accepted_taxon_key",
        "scientific_name",
        "family",
        "genus",
        "candidate_reason",
        "geographic_evidence_score",
        "occurrence_support",
        "same_genus",
        "same_family",
        "known_mimic",
        "historical_false_positive",
        "visually_nearest",
        "target_candidate",
        "candidate_priority",
        "source_versions",
        "candidate_set_fingerprint",
    ),
    "acquisition_plan": (
        "schema_version",
        "acquisition_plan_id",
        "target_accepted_taxon_key",
        "candidate_set_id",
        "candidate_accepted_taxon_key",
        "scientific_name",
        "geo_cluster_id",
        "life_stage",
        "visual_domain",
        "source",
        "requested_count",
        "existing_support_count",
        "available_candidate_count",
        "selected_candidate_count",
        "shortfall_count",
        "fallback_level",
        "selection_strategy",
        "selection_seed",
        "max_distance_km",
        "licence_policy_version",
        "source_snapshot_version",
        "plan_configuration_fingerprint",
    ),
    "acquisition_selections": (
        "schema_version",
        "reference_selection_id",
        "acquisition_plan_id",
        "target_accepted_taxon_key",
        "candidate_set_id",
        "source_candidate_set_id",
        "candidate_accepted_taxon_key",
        "scientific_name",
        "geo_cluster_id",
        "life_stage",
        "visual_domain",
        "reference_media_id",
        "reference_observation_id",
        "source",
        "fallback_level",
        "selection_rank",
        "selection_round",
        "distance_to_cluster_medoid_km",
        "observer_id",
        "observed_date",
        "locality",
        "background_group_id",
        "licence",
        "source_snapshot_version",
        "selection_strategy",
        "selection_seed",
        "plan_configuration_fingerprint",
    ),
    "observations": (
        "schema_version",
        "reference_observation_id",
        "source",
        "source_observation_id",
        "source_taxon_id",
        "supplied_scientific_name",
        "accepted_taxon_key",
        "reconciled_scientific_name",
        "registry_version",
        "taxon_reconciliation_status",
        "identification_quality",
        "community_taxon_status",
        "identification_disagreement",
        "captive_or_cultivated",
        "observer_id",
        "locality",
        "life_stage",
        "sex",
        "observed_at",
        "latitude",
        "longitude",
        "coordinate_uncertainty",
        "coordinates_obscured",
        "country",
        "country_code",
        "geo_cluster_id",
        "distance_to_cluster_medoid_km",
        "source_dataset_key",
        "source_dataset_doi",
        "source_record_hash",
        "source_snapshot_version",
        "source_query_fingerprint",
        "fallback_level",
        "geospatial_issue",
        "preserved_specimen",
        "fossil",
        "occurrence_absent",
        "uncertain_taxon_match",
        "basis_of_record_suitable",
    ),
    "media_candidates": (
        "schema_version",
        "reference_media_id",
        "reference_observation_id",
        "provider_media_id",
        "source",
        "media_type",
        "width",
        "height",
        "creator",
        "rights_holder",
        "licence",
        "attribution",
        "occurrence_licence",
        "original_provider",
        "media_position",
        "source_checksum",
        "source_checksum_algorithm",
        "download_status",
        "verification_status",
        "exclusion_reason",
        "licence_policy_status",
        "source_snapshot_version",
    ),
    "media_objects": (
        "schema_version",
        "reference_media_id",
        "content_type",
        "decoded_width",
        "decoded_height",
        "sha256",
        "perceptual_hash",
        "duplicate_group_id",
        "duplicate_type",
        "canonical_reference_media_id",
        "provider_mirror_ids",
        "licence_policy_status",
        "decode_status",
        "quarantine_reason",
    ),
    "duplicate_relationships": (
        "schema_version",
        "duplicate_relationship_id",
        "duplicate_group_id",
        "canonical_reference_media_id",
        "left_reference_media_id",
        "right_reference_media_id",
        "left_reference_observation_id",
        "right_reference_observation_id",
        "left_source",
        "right_source",
        "left_provider_media_id",
        "right_provider_media_id",
        "relationship_type",
        "evidence_types",
        "sha256_equal",
        "perceptual_hash_distance",
        "same_observation",
        "provider_mirror",
        "resolution_status",
        "policy_version",
        "policy_fingerprint",
    ),
    "review_queue": (
        "schema_version",
        "reference_media_id",
        "reference_observation_id",
        "canonical_reference_media_id",
        "accepted_taxon_key",
        "scientific_name",
        "duplicate_group_id",
        "source",
        "provider_media_id",
        "provider_verification_status",
        "creator",
        "rights_holder",
        "licence",
        "licence_policy_status",
        "attribution",
        "life_stage",
        "visual_domain",
        "view",
        "required_review_count",
        "review_status",
        "reference_bank_version",
    ),
    "queue_provenance": (
        "schema_version",
        "reference_media_id",
    ),
    "review_decisions": (
        "schema_version",
        "reference_media_id",
        "review_round",
        "target_identity_verified",
        "verification_status",
        "life_stage",
        "visual_domain",
        "view",
        "review_confidence",
        "review_notes",
        "exclusion_reason",
        "second_review_required",
    ),
    "split_assignments": (
        "schema_version",
        "reference_media_id",
        "split_version",
        "support_split",
        "included",
        "exclusion_reason",
    ),
}

_SUPPORT_ROW_SEMANTIC_FIELDS = (
    "schema_version",
    "reference_bank_version",
    "registry_version",
    "reference_media_id",
    "canonical_reference_media_id",
    "reference_observation_id",
    "reference_admission_mode",
    "reference_admission_policy_version",
    "reference_admission_policy_fingerprint",
    "identity_evidence_basis",
    "provider_asserted_identity",
    "provider_asserted_taxon_key",
    "provider_asserted_scientific_name",
    "provider_dataset_key",
    "provider_quality_status",
    "human_review_status",
    "human_verified_identity",
    "provisional_support",
    "statistical_audit_required",
    "admission_status",
    "admission_reasons",
    "reference_quality_flags",
    "route_evidence_basis",
    "geographic_prototype_eligible",
    "source",
    "source_observation_id",
    "provider_media_id",
    "source_snapshot_version",
    "source_dataset_key",
    "accepted_taxon_key",
    "scientific_name",
    "target_candidate",
    "geo_cluster_id",
    "observer_id",
    "observed_at",
    "latitude",
    "longitude",
    "image_sha256",
    "perceptual_hash",
    "duplicate_group_id",
    "duplicate_type",
    "creator",
    "rights_holder",
    "licence",
    "licence_policy_status",
    "attribution",
    "review_status",
    "verification_status",
    "target_identity_verified",
    "life_stage",
    "visual_domain",
    "view",
    "route",
    "support_split",
    "support_eligible",
    "exclusion_reasons",
    "reference_bank_fingerprint",
)
_REFERENCE_SUPPORT_V3_EVIDENCE_FIELDS = frozenset(
    {
        "reference_admission_mode",
        "reference_admission_policy_version",
        "reference_admission_policy_fingerprint",
        "identity_evidence_basis",
        "provider_asserted_identity",
        "provider_asserted_taxon_key",
        "provider_asserted_scientific_name",
        "provider_dataset_key",
        "provider_quality_status",
        "human_review_status",
        "human_verified_identity",
        "provisional_support",
        "statistical_audit_required",
        "admission_status",
        "admission_reasons",
        "reference_quality_flags",
        "route_evidence_basis",
        "geographic_prototype_eligible",
    }
)
_LEGACY_SUPPORT_ROW_SEMANTIC_FIELDS = tuple(
    field
    for field in _SUPPORT_ROW_SEMANTIC_FIELDS
    if field not in _REFERENCE_SUPPORT_V3_EVIDENCE_FIELDS
)


@dataclass(frozen=True, slots=True)
class ReferenceBankRequirement:
    accepted_taxon_key: str
    route: str
    minimum_count: int
    geo_cluster_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_taxon_key",
            _required_text(self.accepted_taxon_key, field="accepted_taxon_key"),
        )
        route = _required_text(self.route, field="route").casefold()
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported reference route: {route}")
        object.__setattr__(self, "route", route)
        object.__setattr__(
            self,
            "minimum_count",
            _positive_int(self.minimum_count, field="minimum_count"),
        )
        object.__setattr__(
            self,
            "geo_cluster_id",
            _optional_text(self.geo_cluster_id, field="geo_cluster_id"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ReferenceBankRequirement:
        _exact_mapping_keys(
            value,
            required={"accepted_taxon_key", "route", "minimum_count"},
            optional={"geo_cluster_id"},
            artifact="reference bank requirement",
        )
        return cls(
            accepted_taxon_key=value["accepted_taxon_key"],  # type: ignore[arg-type]
            route=value["route"],  # type: ignore[arg-type]
            minimum_count=value["minimum_count"],  # type: ignore[arg-type]
            geo_cluster_id=value.get("geo_cluster_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DocumentedReferenceShortfall:
    shortfall_id: str
    accepted_taxon_key: str
    route: str
    approved_minimum_count: int
    reason: str
    approved_by: str
    approved_at: datetime
    plan_configuration_fingerprint: str
    geo_cluster_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("shortfall_id", "accepted_taxon_key", "reason", "approved_by"):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), field=name),
            )
        route = _required_text(self.route, field="route").casefold()
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported reference route: {route}")
        object.__setattr__(self, "route", route)
        object.__setattr__(
            self,
            "approved_minimum_count",
            _nonnegative_int(
                self.approved_minimum_count,
                field="approved_minimum_count",
            ),
        )
        object.__setattr__(
            self,
            "approved_at",
            _utc_datetime(self.approved_at, field="approved_at"),
        )
        object.__setattr__(
            self,
            "plan_configuration_fingerprint",
            _fingerprint(
                self.plan_configuration_fingerprint,
                field="plan_configuration_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "geo_cluster_id",
            _optional_text(self.geo_cluster_id, field="geo_cluster_id"),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> DocumentedReferenceShortfall:
        _exact_mapping_keys(
            value,
            required={
                "shortfall_id",
                "accepted_taxon_key",
                "route",
                "approved_minimum_count",
                "reason",
                "approved_by",
                "approved_at",
                "plan_configuration_fingerprint",
            },
            optional={"geo_cluster_id"},
            artifact="documented reference shortfall",
        )
        approved_at = value["approved_at"]
        if not isinstance(approved_at, str):
            raise ValueError("documented shortfall approved_at must be an ISO timestamp")
        return cls(
            shortfall_id=value["shortfall_id"],  # type: ignore[arg-type]
            accepted_taxon_key=value["accepted_taxon_key"],  # type: ignore[arg-type]
            route=value["route"],  # type: ignore[arg-type]
            approved_minimum_count=value["approved_minimum_count"],  # type: ignore[arg-type]
            reason=value["reason"],  # type: ignore[arg-type]
            approved_by=value["approved_by"],  # type: ignore[arg-type]
            approved_at=_parse_datetime(approved_at, field="approved_at"),
            plan_configuration_fingerprint=value["plan_configuration_fingerprint"],  # type: ignore[arg-type]
            geo_cluster_id=value.get("geo_cluster_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ReferenceBankReadinessPolicy:
    policy_version: str
    target_accepted_taxon_key: str
    requirements: tuple[ReferenceBankRequirement, ...]
    documented_shortfalls: tuple[DocumentedReferenceShortfall, ...] = ()
    accepted_licence_policy_statuses: tuple[str, ...] = ("allowed",)
    schema_version: str = REFERENCE_BANK_READINESS_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_BANK_READINESS_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported reference readiness policy schema version")
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, field="policy_version"),
        )
        object.__setattr__(
            self,
            "target_accepted_taxon_key",
            _required_text(
                self.target_accepted_taxon_key,
                field="target_accepted_taxon_key",
            ),
        )
        requirements = tuple(self.requirements)
        if not requirements or not all(
            isinstance(item, ReferenceBankRequirement) for item in requirements
        ):
            raise ValueError("requirements must contain ReferenceBankRequirement values")
        requirement_keys = [_requirement_key(item) for item in requirements]
        if len(set(requirement_keys)) != len(requirement_keys):
            raise ValueError("reference bank requirements must be unique")
        if not any(
            item.accepted_taxon_key == self.target_accepted_taxon_key
            and item.route == "adult_field"
            and item.minimum_count > 0
            for item in requirements
        ):
            raise ValueError("policy must contain a positive target adult_field requirement")
        object.__setattr__(self, "requirements", requirements)

        shortfalls = tuple(self.documented_shortfalls)
        if not all(isinstance(item, DocumentedReferenceShortfall) for item in shortfalls):
            raise TypeError(
                "documented_shortfalls must contain DocumentedReferenceShortfall values"
            )
        if len({item.shortfall_id for item in shortfalls}) != len(shortfalls):
            raise ValueError("documented shortfall IDs must be unique")
        shortfall_keys = [_shortfall_key(item) for item in shortfalls]
        if len(set(shortfall_keys)) != len(shortfall_keys):
            raise ValueError("documented shortfall scopes must be unique")
        requirement_by_key = dict(zip(requirement_keys, requirements, strict=True))
        for shortfall, key in zip(shortfalls, shortfall_keys, strict=True):
            requirement = requirement_by_key.get(key)
            if requirement is None:
                raise ValueError("documented shortfall does not match a requirement")
            if shortfall.accepted_taxon_key == self.target_accepted_taxon_key:
                raise ValueError("target support shortfalls cannot be documented")
            if shortfall.approved_minimum_count >= requirement.minimum_count:
                raise ValueError(
                    "documented shortfall approved minimum must be below the requirement"
                )
        object.__setattr__(self, "documented_shortfalls", shortfalls)

        statuses = tuple(
            sorted(
                {
                    _required_text(value, field="accepted_licence_policy_statuses").casefold()
                    for value in self.accepted_licence_policy_statuses
                }
            )
        )
        if not statuses:
            raise ValueError("accepted licence policy statuses must not be empty")
        object.__setattr__(self, "accepted_licence_policy_statuses", statuses)

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "target_accepted_taxon_key": self.target_accepted_taxon_key,
            "requirements": [
                _dataclass_payload(item)
                for item in sorted(self.requirements, key=_requirement_key)
            ],
            "documented_shortfalls": [
                _dataclass_payload(item)
                for item in sorted(
                    self.documented_shortfalls,
                    key=lambda item: (*_shortfall_key(item), item.shortfall_id),
                )
            ],
            "accepted_licence_policy_statuses": list(
                self.accepted_licence_policy_statuses
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> ReferenceBankReadinessPolicy:
        _exact_mapping_keys(
            value,
            required={
                "schema_version",
                "policy_version",
                "target_accepted_taxon_key",
                "requirements",
            },
            optional={
                "documented_shortfalls",
                "accepted_licence_policy_statuses",
            },
            artifact="reference bank readiness policy",
        )
        requirements = _mapping_sequence(value["requirements"], field="requirements")
        shortfalls = _mapping_sequence(
            value.get("documented_shortfalls", []),
            field="documented_shortfalls",
        )
        statuses = value.get("accepted_licence_policy_statuses", ["allowed"])
        if not isinstance(statuses, list) or not all(
            isinstance(item, str) for item in statuses
        ):
            raise ValueError("accepted_licence_policy_statuses must be a string list")
        return cls(
            schema_version=str(value["schema_version"]),
            policy_version=str(value["policy_version"]),
            target_accepted_taxon_key=str(value["target_accepted_taxon_key"]),
            requirements=tuple(
                ReferenceBankRequirement.from_mapping(item) for item in requirements
            ),
            documented_shortfalls=tuple(
                DocumentedReferenceShortfall.from_mapping(item) for item in shortfalls
            ),
            accepted_licence_policy_statuses=tuple(statuses),
        )


@dataclass(frozen=True, slots=True)
class ReferenceModelInputIdentity:
    model_name: str
    model_version: str
    model_revision: str
    checkpoint_uri: str
    checkpoint_sha256: str
    open_clip_version: str
    open_clip_config_sha256: str
    preprocessing_version: str
    preprocessing_contract_fingerprint: str
    preprocessing_attestation_fingerprint: str
    input_contract_version: str
    schema_version: str = REFERENCE_MODEL_INPUT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_MODEL_INPUT_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported reference model identity schema version")
        for name in (
            "model_name",
            "model_version",
            "model_revision",
            "open_clip_version",
            "preprocessing_version",
            "input_contract_version",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), field=name),
            )
        object.__setattr__(
            self,
            "checkpoint_uri",
            _absolute_uri(self.checkpoint_uri, field="checkpoint_uri"),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _fingerprint(self.checkpoint_sha256, field="checkpoint_sha256"),
        )
        for name in (
            "open_clip_config_sha256",
            "preprocessing_contract_fingerprint",
            "preprocessing_attestation_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _fingerprint(getattr(self, name), field=name),
            )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                key: value
                for key, value in self.to_dict().items()
                if key != "checkpoint_uri"
            }
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_revision": self.model_revision,
            "checkpoint_uri": self.checkpoint_uri,
            "checkpoint_sha256": self.checkpoint_sha256,
            "open_clip_version": self.open_clip_version,
            "open_clip_config_sha256": self.open_clip_config_sha256,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_contract_fingerprint": (
                self.preprocessing_contract_fingerprint
            ),
            "preprocessing_attestation_fingerprint": (
                self.preprocessing_attestation_fingerprint
            ),
            "input_contract_version": self.input_contract_version,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> ReferenceModelInputIdentity:
        _exact_mapping_keys(
            value,
            required={
                "schema_version",
                "model_name",
                "model_version",
                "model_revision",
                "checkpoint_uri",
                "checkpoint_sha256",
                "open_clip_version",
                "open_clip_config_sha256",
                "preprocessing_version",
                "preprocessing_contract_fingerprint",
                "preprocessing_attestation_fingerprint",
                "input_contract_version",
            },
            optional=set(),
            artifact="reference model input identity",
        )
        return cls(**{key: str(item) for key, item in value.items()})


@dataclass(frozen=True, slots=True)
class ReferenceBankReadinessResult:
    support_manifest: pl.DataFrame
    summary: pl.DataFrame
    readiness: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReferenceSupportManifestMigration:
    manifest: pl.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReferenceBankRequirementStatus:
    accepted_taxon_key: str
    route: str
    minimum_count: int
    observed_count: int
    geo_cluster_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceBankReadinessPermit:
    status: str
    registry_version: str
    reference_bank_version: str
    target_accepted_taxon_key: str
    policy_fingerprint: str
    bank_fingerprint: str
    support_manifest_fingerprint: str
    summary_fingerprint: str
    split_assignments_fingerprint: str
    model_input_schema_version: str
    model_name: str
    model_version: str
    model_revision: str
    checkpoint_uri: str
    checkpoint_sha256: str
    open_clip_version: str
    open_clip_config_sha256: str
    preprocessing_version: str
    preprocessing_contract_fingerprint: str
    preprocessing_attestation_fingerprint: str
    input_contract_version: str
    model_input_fingerprint: str
    readiness_sha256: str
    support_manifest_sha256: str
    summary_sha256: str
    permits_reference_embedding: bool
    permits_provisional_scoring: bool
    permits_calibrated_scoring: bool
    permits_scientific_release: bool
    reference_admission_mode: str
    admission_policy_fingerprint: str
    provisional_support_count: int
    human_verified_support_count: int
    statistical_audit_required: bool
    permits_prototype_creation: bool = False
    requires_downstream_flickr_review: bool = True
    candidate_set_fingerprints: tuple[str, ...] = ()
    target_adult_requirements: tuple[ReferenceBankRequirementStatus, ...] = ()

    def model_input_identity(self) -> ReferenceModelInputIdentity:
        return ReferenceModelInputIdentity(
            schema_version=self.model_input_schema_version,
            model_name=self.model_name,
            model_version=self.model_version,
            model_revision=self.model_revision,
            checkpoint_uri=self.checkpoint_uri,
            checkpoint_sha256=self.checkpoint_sha256,
            open_clip_version=self.open_clip_version,
            open_clip_config_sha256=self.open_clip_config_sha256,
            preprocessing_version=self.preprocessing_version,
            preprocessing_contract_fingerprint=(
                self.preprocessing_contract_fingerprint
            ),
            preprocessing_attestation_fingerprint=(
                self.preprocessing_attestation_fingerprint
            ),
            input_contract_version=self.input_contract_version,
        )


def reference_bank_split_assignments_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "split_version": pl.String,
        "support_split": pl.String,
        "included": pl.Boolean,
        "exclusion_reason": pl.String,
        "assigned_by": pl.String,
        "assigned_at": pl.Datetime("us", "UTC"),
        "assignment_fingerprint": pl.String,
    }


def reference_support_manifest_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "registry_version": pl.String,
        "reference_media_id": pl.String,
        "canonical_reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "reference_admission_mode": pl.String,
        "reference_admission_policy_version": pl.String,
        "reference_admission_policy_fingerprint": pl.String,
        "identity_evidence_basis": pl.String,
        "provider_asserted_identity": pl.Boolean,
        "provider_asserted_taxon_key": pl.String,
        "provider_asserted_scientific_name": pl.String,
        "provider_dataset_key": pl.String,
        "provider_quality_status": pl.String,
        "human_review_status": pl.String,
        "human_verified_identity": pl.Boolean,
        "provisional_support": pl.Boolean,
        "statistical_audit_required": pl.Boolean,
        "admission_status": pl.String,
        "admission_reasons": pl.List(pl.String),
        "reference_quality_flags": pl.List(pl.String),
        "route_evidence_basis": pl.String,
        "geographic_prototype_eligible": pl.Boolean,
        "review_request_id": pl.String,
        "review_decision_ids": pl.List(pl.String),
        "reviewer_ids": pl.List(pl.String),
        "source": pl.String,
        "source_observation_id": pl.String,
        "provider_media_id": pl.String,
        "source_record_url": pl.String,
        "source_snapshot_version": pl.String,
        "source_dataset_key": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "target_candidate": pl.Boolean,
        "geo_cluster_id": pl.String,
        "observer_id": pl.String,
        "observed_at": pl.Datetime("us", "UTC"),
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "source_object_uri": pl.String,
        "image_sha256": pl.String,
        "perceptual_hash": pl.String,
        "object_fingerprint": pl.String,
        "duplicate_group_id": pl.String,
        "duplicate_type": pl.String,
        "creator": pl.String,
        "rights_holder": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "licence_policy_status": pl.String,
        "attribution": pl.String,
        "review_status": pl.String,
        "verification_status": pl.String,
        "target_identity_verified": pl.Boolean,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "route": pl.String,
        "support_split": pl.String,
        "support_eligible": pl.Boolean,
        "exclusion_reasons": pl.List(pl.String),
        "split_assignment_fingerprint": pl.String,
        "support_row_fingerprint": pl.String,
        "reference_bank_fingerprint": pl.String,
    }


def legacy_reference_support_manifest_v2_schema() -> dict[str, pl.DataType]:
    """Return the exact pre-admission strict support schema."""

    return {
        field: dtype
        for field, dtype in reference_support_manifest_schema().items()
        if field not in _REFERENCE_SUPPORT_V3_EVIDENCE_FIELDS
    }


def migrate_strict_reference_support_manifest_v2(
    frame: pl.DataFrame,
) -> ReferenceSupportManifestMigration:
    """Explicitly migrate a validated v2 strict manifest without weakening trust."""

    _validate_legacy_reference_support_manifest_v2(frame)
    policy = strict_reference_admission_policy()
    source_fingerprint = _semantic_frame_fingerprint(
        frame,
        fields=_LEGACY_SUPPORT_ROW_SEMANTIC_FIELDS,
    )
    rows: list[dict[str, object]] = []
    for legacy in frame.iter_rows(named=True):
        row = dict(legacy)
        eligible = bool(row["support_eligible"])
        row.update(
            {
                "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
                "reference_admission_mode": policy.mode,
                "reference_admission_policy_version": policy.policy_version,
                "reference_admission_policy_fingerprint": policy.fingerprint,
                "identity_evidence_basis": "human_verified",
                "provider_asserted_identity": False,
                "provider_asserted_taxon_key": None,
                "provider_asserted_scientific_name": None,
                "provider_dataset_key": row["source_dataset_key"],
                "provider_quality_status": None,
                "human_review_status": "completed",
                "human_verified_identity": True,
                "provisional_support": False,
                "statistical_audit_required": False,
                "admission_status": "admitted" if eligible else "excluded",
                "admission_reasons": ["legacy_strict_human_review_verified"]
                if eligible
                else sorted(
                    set(row["exclusion_reasons"])
                    or {"legacy_strict_support_excluded"}
                ),
                "reference_quality_flags": [],
                "route_evidence_basis": "human_verified_review",
                "geographic_prototype_eligible": bool(
                    eligible
                    and _present(row["geo_cluster_id"])
                    and row["latitude"] is not None
                    and row["longitude"] is not None
                ),
            }
        )
        row["support_row_fingerprint"] = _support_row_fingerprint(row)
        rows.append(row)
    migrated = _strict_frame(
        rows,
        schema=reference_support_manifest_schema(),
        sort_by=_SUPPORT_SORT,
    )
    validate_reference_support_manifest(migrated)
    report: dict[str, Any] = {
        "schema_version": REFERENCE_SUPPORT_MANIFEST_MIGRATION_SCHEMA_VERSION,
        "source_schema_version": LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
        "target_schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
        "migration_mode": "explicit_human_verified_strict",
        "reference_admission_mode": policy.mode,
        "admission_policy_fingerprint": policy.fingerprint,
        "source_manifest_fingerprint": source_fingerprint,
        "target_manifest_fingerprint": _support_manifest_fingerprint(migrated),
        "row_count": migrated.height,
        "provider_assertion_backfilled": False,
        "requires_downstream_rebuild": [
            "readiness",
            "reference_embeddings",
            "reference_prototypes",
            "models",
            "scores",
        ],
    }
    report["migration_fingerprint"] = canonical_semantic_fingerprint(report)
    return ReferenceSupportManifestMigration(manifest=migrated, report=report)


def _validate_legacy_reference_support_manifest_v2(frame: pl.DataFrame) -> None:
    _validate_exact_frame(
        frame,
        schema=legacy_reference_support_manifest_v2_schema(),
        artifact="legacy reference support manifest v2",
        sort_by=_SUPPORT_SORT,
    )
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("legacy support manifest contains duplicate canonical media")
    if not frame.is_empty() and frame["reference_bank_fingerprint"].n_unique() != 1:
        raise ValueError("legacy support manifest spans multiple bank fingerprints")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != LEGACY_REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported legacy support manifest schema version")
        if row["canonical_reference_media_id"] != row["reference_media_id"]:
            raise ValueError("legacy support manifest row is not canonical media")
        if (
            row["review_status"] != "completed"
            or row["verification_status"] != "verified"
            or not row["target_identity_verified"]
        ):
            raise ValueError("legacy support manifest is not human-verified strict")
        expected_route = _reference_route(
            life_stage=str(row["life_stage"]),
            visual_domain=str(row["visual_domain"]),
        )
        if expected_route != row["route"] or row["route"] not in REFERENCE_ROUTES:
            raise ValueError("legacy support manifest route is invalid")
        _fingerprint(row["image_sha256"], field="image_sha256")
        _fingerprint(row["object_fingerprint"], field="object_fingerprint")
        if row["support_row_fingerprint"] != _legacy_support_row_fingerprint_v2(row):
            raise ValueError("legacy support manifest row fingerprint is invalid")


def reference_bank_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "target_candidate": pl.Boolean,
        "geo_cluster_id": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "support_split": pl.String,
        "required_count": pl.UInt64,
        "candidate_count": pl.UInt64,
        "downloaded_count": pl.UInt64,
        "deduplicated_count": pl.UInt64,
        "reviewed_count": pl.UInt64,
        "verified_count": pl.UInt64,
        "eligible_count": pl.UInt64,
        "excluded_count": pl.UInt64,
        "pending_review_count": pl.UInt64,
        "provider_asserted_count": pl.UInt64,
        "provider_asserted_eligible_count": pl.UInt64,
        "human_verified_count": pl.UInt64,
        "human_verified_eligible_count": pl.UInt64,
        "provisional_support_count": pl.UInt64,
        "strict_support_count": pl.UInt64,
        "flagged_for_review_count": pl.UInt64,
        "excluded_by_automated_qa_count": pl.UInt64,
        "excluded_by_human_review_count": pl.UInt64,
        "shortfall_count": pl.UInt64,
        "documented_shortfall_count": pl.UInt64,
        "source_count": pl.UInt64,
        "licence_count": pl.UInt64,
        "creator_count": pl.UInt64,
        "observer_count": pl.UInt64,
        "observation_count": pl.UInt64,
        "geographic_cluster_count": pl.UInt64,
        "reference_bank_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "split_assignments_fingerprint": pl.String,
        "summary_row_fingerprint": pl.String,
    }


def reference_bank_split_assignments_frame(
    rows: Sequence[Mapping[str, object]] = (),
) -> pl.DataFrame:
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault(
            "schema_version",
            REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION,
        )
        normalized.append(item)
    return _strict_frame(
        normalized,
        schema=reference_bank_split_assignments_schema(),
        sort_by=("reference_media_id",),
    )


def validate_reference_bank_split_assignments(frame: pl.DataFrame) -> None:
    _validate_exact_frame(
        frame,
        schema=reference_bank_split_assignments_schema(),
        artifact="reference bank split assignments",
        sort_by=("reference_media_id",),
    )
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("split assignments contain duplicate reference media IDs")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION:
            raise ValueError("unsupported split assignment schema version")
        if row["support_split"] not in REFERENCE_SUPPORT_SPLITS:
            raise ValueError("split assignment contains an unsupported split")
        for field_name in ("reference_media_id", "split_version", "assigned_by"):
            _required_text(row[field_name], field=field_name)
        assigned_at = _utc_datetime(row["assigned_at"], field="assigned_at")
        exclusion_reason = _optional_text(
            row["exclusion_reason"],
            field="exclusion_reason",
        )
        if bool(row["included"]) == bool(exclusion_reason):
            raise ValueError(
                "included split assignments cannot have an exclusion reason and "
                "excluded assignments require one"
            )
        expected = _assignment_fingerprint(
            reference_media_id=str(row["reference_media_id"]),
            split_version=str(row["split_version"]),
            support_split=str(row["support_split"]),
            included=bool(row["included"]),
            exclusion_reason=exclusion_reason,
            assigned_by=str(row["assigned_by"]),
            assigned_at=assigned_at,
        )
        if row["assignment_fingerprint"] != expected:
            raise ValueError("split assignment fingerprint is invalid")


def validate_reference_support_manifest(frame: pl.DataFrame) -> None:
    _validate_exact_frame(
        frame,
        schema=reference_support_manifest_schema(),
        artifact="reference support manifest",
        sort_by=_SUPPORT_SORT,
    )
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("support manifest contains duplicate canonical media")
    if not frame.is_empty() and frame["reference_bank_fingerprint"].n_unique() != 1:
        raise ValueError("support manifest spans multiple bank fingerprints")
    if not frame.is_empty() and (
        frame["reference_admission_mode"].n_unique() != 1
        or frame["reference_admission_policy_fingerprint"].n_unique() != 1
    ):
        raise ValueError("support manifest spans multiple admission policies")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported reference support manifest schema version")
        if row["canonical_reference_media_id"] != row["reference_media_id"]:
            raise ValueError("support manifest row is not canonical media")
        route = str(row["route"])
        if route not in REFERENCE_ROUTES:
            raise ValueError("support manifest contains an unsupported route")
        admission_mode = str(row["reference_admission_mode"])
        if admission_mode not in REFERENCE_ADMISSION_MODES:
            raise ValueError("support manifest contains an unsupported admission mode")
        _required_text(
            row["reference_admission_policy_version"],
            field="reference_admission_policy_version",
        )
        _fingerprint(
            row["reference_admission_policy_fingerprint"],
            field="reference_admission_policy_fingerprint",
        )
        identity_basis = str(row["identity_evidence_basis"])
        if identity_basis not in REFERENCE_IDENTITY_EVIDENCE_BASES:
            raise ValueError("support manifest contains an unsupported identity basis")
        admission_status = str(row["admission_status"])
        if admission_status not in REFERENCE_ADMISSION_STATUSES:
            raise ValueError("support manifest contains an unsupported admission status")
        route_evidence = str(row["route_evidence_basis"])
        if route_evidence not in REFERENCE_ROUTE_EVIDENCE_BASES:
            raise ValueError("support manifest contains an unsupported route evidence basis")
        human_review_status = str(row["human_review_status"])
        if human_review_status not in REFERENCE_HUMAN_REVIEW_STATUSES:
            raise ValueError("support manifest contains an unsupported human review status")
        admission_reasons = list(row["admission_reasons"])
        if not admission_reasons or admission_reasons != sorted(set(admission_reasons)):
            raise ValueError("support manifest row must record canonical admission reasons")
        quality_flags = list(row["reference_quality_flags"])
        if quality_flags != sorted(set(quality_flags)):
            raise ValueError("support manifest reference quality flags are not canonical")
        provider_asserted = bool(row["provider_asserted_identity"])
        human_verified = bool(row["human_verified_identity"])
        provisional = bool(row["provisional_support"])
        if provider_asserted and (
            not _present(row["provider_asserted_taxon_key"])
            or not _present(row["provider_asserted_scientific_name"])
        ):
            raise ValueError("provider assertion lacks taxon identity evidence")
        if human_verified != bool(row["target_identity_verified"]):
            raise ValueError("human-verified identity fields disagree")
        if identity_basis == "human_verified":
            if (
                not human_verified
                or human_review_status != "completed"
                or row["verification_status"] != "verified"
                or provisional
                or route_evidence != "human_verified_review"
            ):
                raise ValueError("human-verified identity basis is inconsistent")
        elif identity_basis == "gbif_provider_asserted":
            if (
                str(row["source"]).casefold() != "gbif"
                or not provider_asserted
                or human_verified
                or bool(row["target_identity_verified"])
                or row["verification_status"] == "verified"
                or not provisional
                or not bool(row["statistical_audit_required"])
                or admission_mode != "adaptive_gbif_fast_start"
                or route_evidence != "yoloe"
                or human_review_status == "completed"
            ):
                raise ValueError(
                    "GBIF provider assertion is inconsistent with provisional support"
                )
        elif human_verified or provisional:
            raise ValueError("identity basis none cannot claim identity support")
        if provisional and identity_basis != "gbif_provider_asserted":
            raise ValueError("provisional support must use GBIF provider assertion")
        if bool(row["geographic_prototype_eligible"]) and (
            not bool(row["support_eligible"])
            or not _present(row["geo_cluster_id"])
            or row["latitude"] is None
            or row["longitude"] is None
        ):
            raise ValueError("geographic prototype eligibility lacks usable geography")
        if row["support_split"] is not None and (
            row["support_split"] not in REFERENCE_SUPPORT_SPLITS
        ):
            raise ValueError("support manifest contains an unsupported split")
        expected_route = _reference_route(
            life_stage=str(row["life_stage"]),
            visual_domain=str(row["visual_domain"]),
        )
        if expected_route != route:
            raise ValueError("support manifest route conflicts with resolved review")
        if row["support_eligible"]:
            required_text_fields = (
                "source",
                "provider_media_id",
                "source_record_url",
                "source_snapshot_version",
                "source_object_uri",
                "creator",
                "rights_holder",
                "licence",
                "licence_uri",
                "attribution",
                "support_split",
            )
            if any(not _present(row[field_name]) for field_name in required_text_fields):
                raise ValueError("eligible support row has incomplete attribution or source data")
            if admission_status != "admitted":
                raise ValueError("eligible support row is not admitted")
            if identity_basis == "none":
                raise ValueError("eligible support row lacks an identity evidence basis")
            if identity_basis == "human_verified":
                if row["review_status"] != "completed":
                    raise ValueError("eligible strict support row has incomplete review")
                if row["verification_status"] != "verified":
                    raise ValueError("eligible strict support row is not verified")
                if not row["target_identity_verified"]:
                    raise ValueError("eligible strict support row lacks verified identity")
            if row["exclusion_reasons"]:
                raise ValueError("eligible support row has exclusion reasons")
        _fingerprint(row["image_sha256"], field="image_sha256")
        _fingerprint(row["object_fingerprint"], field="object_fingerprint")
        _fingerprint(
            row["split_assignment_fingerprint"],
            field="split_assignment_fingerprint",
            allow_none=not bool(row["support_eligible"]),
        )
        expected = _support_row_fingerprint(row)
        if row["support_row_fingerprint"] != expected:
            raise ValueError("support manifest row fingerprint is invalid")


def reference_support_manifest_fingerprint(frame: pl.DataFrame) -> str:
    validate_reference_support_manifest(frame)
    return _support_manifest_fingerprint(frame)


def reference_support_manifest_frame(
    rows: Sequence[Mapping[str, object]],
    *,
    reference_bank_version: str,
    reference_bank_fingerprint: str,
) -> pl.DataFrame:
    """Bind support rows to a bank identity and recompute semantic row hashes."""

    bank_version = _required_text(
        reference_bank_version,
        field="reference_bank_version",
    )
    bank_fingerprint = _fingerprint(
        reference_bank_fingerprint,
        field="reference_bank_fingerprint",
    )
    assert bank_fingerprint is not None
    normalized: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        row["reference_bank_version"] = bank_version
        row["reference_bank_fingerprint"] = bank_fingerprint
        row["support_row_fingerprint"] = ""
        row["support_row_fingerprint"] = _support_row_fingerprint(row)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    ).sort(list(_SUPPORT_SORT))
    validate_reference_support_manifest(frame)
    return frame


def reference_support_split_leakage(
    frame: pl.DataFrame,
) -> tuple[dict[str, object], ...]:
    validate_reference_support_manifest(frame)
    assigned_support = list(
        frame.filter(pl.col("support_eligible")).iter_rows(named=True)
    )
    return tuple(_support_split_leakage(assigned_support))


def validate_reference_bank_summary(frame: pl.DataFrame) -> None:
    _validate_exact_frame(
        frame,
        schema=reference_bank_summary_schema(),
        artifact="reference bank summary",
        sort_by=_SUMMARY_SORT,
    )
    identity = list(_SUMMARY_SORT)
    if frame.select(identity).unique().height != frame.height:
        raise ValueError("reference bank summary contains duplicate grain rows")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_BANK_SUMMARY_SCHEMA_VERSION:
            raise ValueError("unsupported reference bank summary schema version")
        if row["shortfall_count"] != max(
            0,
            int(row["required_count"]) - int(row["eligible_count"]),
        ):
            raise ValueError("reference bank summary shortfall count is inconsistent")
        if int(row["provider_asserted_eligible_count"]) > int(
            row["provider_asserted_count"]
        ):
            raise ValueError("provider-asserted summary counts are inconsistent")
        if int(row["human_verified_eligible_count"]) > int(
            row["human_verified_count"]
        ):
            raise ValueError("human-verified summary counts are inconsistent")
        if int(row["strict_support_count"]) != int(
            row["human_verified_eligible_count"]
        ):
            raise ValueError("strict support summary count is inconsistent")
        if int(row["provisional_support_count"]) + int(
            row["strict_support_count"]
        ) != int(row["eligible_count"]):
            raise ValueError("provisional and strict support counts are inconsistent")
        if row["summary_row_fingerprint"] != _summary_row_fingerprint(row):
            raise ValueError("reference bank summary row fingerprint is invalid")


def build_reference_bank_readiness(
    *,
    candidate_species: pl.DataFrame,
    acquisition_plan: pl.DataFrame,
    acquisition_selections: pl.DataFrame,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    deduplication_report: Mapping[str, object],
    review_queue: pl.DataFrame,
    queue_provenance: pl.DataFrame,
    review_decisions: pl.DataFrame,
    split_assignments: pl.DataFrame,
    policy: ReferenceBankReadinessPolicy,
    registry_version: str,
    reference_bank_version: str,
    model_identity: ReferenceModelInputIdentity,
    admission_policy: ReferenceAdmissionPolicy | None = None,
    provisional_support: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> ReferenceBankReadinessResult:
    """Compile a frozen support bank and its fail-closed readiness decision."""

    if not isinstance(policy, ReferenceBankReadinessPolicy):
        raise TypeError("policy must be a ReferenceBankReadinessPolicy")
    if not isinstance(model_identity, ReferenceModelInputIdentity):
        raise TypeError("model_identity must be a ReferenceModelInputIdentity")
    admission_policy = admission_policy or strict_reference_admission_policy()
    if not isinstance(admission_policy, ReferenceAdmissionPolicy):
        raise TypeError("admission_policy must be a ReferenceAdmissionPolicy")
    if admission_policy.mode == DEFAULT_REFERENCE_ADMISSION_MODE:
        if provisional_support is None:
            raise ValueError(
                "adaptive admission requires explicit provisional support evidence"
            )
        validate_reference_provisional_support(provisional_support)
    elif provisional_support is not None and not provisional_support.is_empty():
        raise ValueError(
            "provisional support evidence is permitted only in adaptive mode"
        )
    registry = _required_text(registry_version, field="registry_version")
    bank_version = _required_text(
        reference_bank_version,
        field="reference_bank_version",
    )
    timestamp = _utc_datetime(
        created_at if created_at is not None else datetime.now(UTC),
        field="created_at",
    )

    validate_regional_candidate_species(candidate_species)
    validate_reference_acquisition_plan(acquisition_plan)
    validate_reference_acquisition_selections(acquisition_selections)
    validate_reference_observations(observations)
    validate_reference_media_candidates(media_candidates)
    validate_reference_media_objects(media_objects)
    validate_reference_media_duplicate_relationships(duplicate_relationships)
    validate_reference_review_queue(review_queue)
    validate_reference_review_decisions(review_decisions)
    validate_reference_bank_split_assignments(split_assignments)
    validate_reference_media_deduplication_artifacts(
        media_objects=media_objects,
        relationships=duplicate_relationships,
        media_candidates=media_candidates,
        observations=observations,
        report=deduplication_report,
    )
    candidate_context = _validate_candidate_policy_bindings(
        candidate_species,
        acquisition_plan=acquisition_plan,
        acquisition_selections=acquisition_selections,
        policy=policy,
        registry_version=registry,
        reference_bank_version=bank_version,
        review_queue=review_queue,
    )
    indexes = _inventory_indexes(
        observations=observations,
        media_candidates=media_candidates,
        media_objects=media_objects,
        split_assignments=split_assignments,
    )
    _validate_reference_identity_bindings(
        candidate_species=candidate_species,
        acquisition_plan=acquisition_plan,
        acquisition_selections=acquisition_selections,
        review_queue=review_queue,
        indexes=indexes,
        registry_version=registry,
    )
    validate_reference_review_queue_source_bindings(
        review_queue,
        queue_provenance,
        acquisition_selections,
        media_objects,
        media_candidates,
        observations,
        duplicate_relationships,
        deduplication_report=deduplication_report,
        reference_bank_version=bank_version,
    )
    review = resolve_reference_review_statuses(
        review_queue,
        review_decisions,
        queue_provenance=queue_provenance,
        resolved_at=timestamp,
    )

    input_fingerprints = {
        "candidate_species": _semantic_frame_fingerprint(
            candidate_species,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["candidate_species"],
        ),
        "acquisition_plan": _semantic_frame_fingerprint(
            acquisition_plan,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["acquisition_plan"],
        ),
        "acquisition_selections": _semantic_frame_fingerprint(
            acquisition_selections,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["acquisition_selections"],
        ),
        "observations": _semantic_frame_fingerprint(
            observations,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["observations"],
        ),
        "media_candidates": _semantic_frame_fingerprint(
            media_candidates,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["media_candidates"],
        ),
        "media_objects": _semantic_frame_fingerprint(
            media_objects,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["media_objects"],
        ),
        "duplicate_relationships": _semantic_frame_fingerprint(
            duplicate_relationships,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["duplicate_relationships"],
        ),
        "deduplication_report": _deduplication_report_fingerprint(
            deduplication_report,
            media_objects=media_objects,
            media_candidates=media_candidates,
            observations=observations,
            duplicate_relationships=duplicate_relationships,
        ),
        "review_queue": _semantic_frame_fingerprint(
            review_queue,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["review_queue"],
        ),
        "queue_provenance": _semantic_frame_fingerprint(
            queue_provenance,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["queue_provenance"],
        ),
        "review_decisions": _semantic_frame_fingerprint(
            review_decisions,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["review_decisions"],
        ),
        "split_assignments": _semantic_frame_fingerprint(
            split_assignments,
            fields=_READINESS_INPUT_SEMANTIC_FIELDS["split_assignments"],
        ),
    }
    if provisional_support is not None:
        input_fingerprints["provisional_support"] = _frame_fingerprint(
            provisional_support
        )
    split_fingerprint = input_fingerprints["split_assignments"]
    bank_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_BANK_READINESS_SCHEMA_VERSION,
            "reference_bank_version": bank_version,
            "registry_version": registry,
            "target_accepted_taxon_key": policy.target_accepted_taxon_key,
            "policy_fingerprint": policy.fingerprint,
            "reference_admission_mode": admission_policy.mode,
            "admission_policy_fingerprint": admission_policy.fingerprint,
            "model_input_fingerprint": model_identity.fingerprint,
            "candidate_set_ids": candidate_context["candidate_set_ids"],
            "candidate_set_fingerprints": candidate_context[
                "candidate_set_fingerprints"
            ],
            "inputs": input_fingerprints,
        }
    )

    support_rows, structural_issues = _build_support_rows(
        review=review,
        indexes=indexes,
        policy=policy,
        admission_policy=admission_policy,
        registry_version=registry,
        reference_bank_version=bank_version,
        bank_fingerprint=bank_fingerprint,
    )
    if admission_policy.mode == DEFAULT_REFERENCE_ADMISSION_MODE:
        assert provisional_support is not None
        provisional_rows, provisional_issues = _build_provisional_support_rows(
            provisional_support=provisional_support,
            review=review,
            indexes=indexes,
            policy=policy,
            admission_policy=admission_policy,
            registry_version=registry,
            reference_bank_version=bank_version,
            bank_fingerprint=bank_fingerprint,
            human_support_media_ids={
                str(row["reference_media_id"]) for row in support_rows
            },
        )
        support_rows.extend(provisional_rows)
        structural_issues.extend(provisional_issues)
    support_manifest = _strict_frame(
        support_rows,
        schema=reference_support_manifest_schema(),
        sort_by=_SUPPORT_SORT,
    )
    validate_reference_support_manifest(support_manifest)
    support_fingerprint = _support_manifest_fingerprint(support_manifest)

    summary = _build_reference_bank_summary(
        candidate_context=candidate_context,
        policy=policy,
        acquisition_plan=acquisition_plan,
        observations=observations,
        media_candidates=media_candidates,
        media_objects=media_objects,
        review=review,
        support_manifest=support_manifest,
        registry_version=registry,
        reference_bank_version=bank_version,
        bank_fingerprint=bank_fingerprint,
        support_fingerprint=support_fingerprint,
        split_fingerprint=split_fingerprint,
    )
    validate_reference_bank_summary(summary)
    summary_fingerprint = _frame_fingerprint(summary)

    checks, counts, documented_shortfalls = _readiness_checks(
        candidate_context=candidate_context,
        policy=policy,
        admission_policy=admission_policy,
        acquisition_plan=acquisition_plan,
        acquisition_selections=acquisition_selections,
        review=review,
        support_manifest=support_manifest,
        duplicate_relationships=duplicate_relationships,
        split_assignments=split_assignments,
        indexes=indexes,
        structural_issues=structural_issues,
        model_identity=model_identity,
        bank_fingerprint=bank_fingerprint,
        support_fingerprint=support_fingerprint,
        summary_fingerprint=summary_fingerprint,
        split_fingerprint=split_fingerprint,
    )
    status = _readiness_status(
        checks=checks,
        counts=counts,
        documented_shortfalls=documented_shortfalls,
    )
    capabilities = _readiness_capabilities(status)
    readiness: dict[str, Any] = {
        "schema_version": REFERENCE_BANK_READINESS_SCHEMA_VERSION,
        "status": status,
        "permits_vision": capabilities["permits_reference_embedding"],
        **capabilities,
        "reference_admission_mode": admission_policy.mode,
        "admission_policy_fingerprint": admission_policy.fingerprint,
        "provisional_support_count": counts["provisional_support_count"],
        "human_verified_support_count": counts["human_verified_support_count"],
        "statistical_audit_required": admission_policy.require_statistical_audit,
        "registry_version": registry,
        "reference_bank_version": bank_version,
        "policy_version": policy.policy_version,
        "policy_fingerprint": policy.fingerprint,
        "policy": policy.to_dict(),
        "target_accepted_taxon_key": policy.target_accepted_taxon_key,
        "candidate_set_ids": candidate_context["candidate_set_ids"],
        "candidate_set_fingerprints": candidate_context[
            "candidate_set_fingerprints"
        ],
        "created_at": timestamp.isoformat(),
        "git_sha": current_git_sha(),
        "bank_fingerprint": bank_fingerprint,
        "support_manifest_fingerprint": support_fingerprint,
        "summary_fingerprint": summary_fingerprint,
        "split_assignments_fingerprint": split_fingerprint,
        "model_input_identity": model_identity.to_dict(),
        "model_input_fingerprint": model_identity.fingerprint,
        "checks": checks,
        "counts": counts,
        "documented_shortfalls": documented_shortfalls,
        "inputs": input_fingerprints,
        "artifacts": {
            "support_manifest": {
                "file": REFERENCE_SUPPORT_MANIFEST_FILE,
                "sha256": None,
                "semantic_fingerprint": support_fingerprint,
            },
            "summary": {
                "file": REFERENCE_BANK_SUMMARY_FILE,
                "sha256": None,
                "semantic_fingerprint": summary_fingerprint,
            },
        },
    }
    result = ReferenceBankReadinessResult(
        support_manifest=support_manifest,
        summary=summary,
        readiness=readiness,
    )
    validate_reference_bank_readiness(result)
    return result


def validate_reference_bank_readiness(result: ReferenceBankReadinessResult) -> None:
    if not isinstance(result, ReferenceBankReadinessResult):
        raise TypeError("result must be a ReferenceBankReadinessResult")
    validate_reference_support_manifest(result.support_manifest)
    validate_reference_bank_summary(result.summary)
    payload = result.readiness
    _validate_readiness_payload(payload, published=False)
    if payload["support_manifest_fingerprint"] != _support_manifest_fingerprint(
        result.support_manifest
    ):
        raise ValueError("readiness support manifest fingerprint mismatch")
    if payload["summary_fingerprint"] != _frame_fingerprint(result.summary):
        raise ValueError("readiness summary fingerprint mismatch")
    if not result.support_manifest.is_empty() and set(
        result.support_manifest["reference_bank_fingerprint"]
    ) != {payload["bank_fingerprint"]}:
        raise ValueError("support manifest bank fingerprint mismatch")
    if not result.summary.is_empty() and set(
        result.summary["reference_bank_fingerprint"]
    ) != {payload["bank_fingerprint"]}:
        raise ValueError("summary bank fingerprint mismatch")
    eligible_all = result.support_manifest.filter(pl.col("support_eligible"))
    if not result.support_manifest.is_empty() and (
        set(result.support_manifest["reference_admission_mode"])
        != {payload["reference_admission_mode"]}
        or set(result.support_manifest["reference_admission_policy_fingerprint"])
        != {payload["admission_policy_fingerprint"]}
    ):
        raise ValueError("readiness admission identity conflicts with support manifest")
    if eligible_all.filter(pl.col("provisional_support")).height != payload[
        "provisional_support_count"
    ]:
        raise ValueError("readiness provisional support count mismatch")
    if eligible_all.filter(pl.col("human_verified_identity")).height != payload[
        "human_verified_support_count"
    ]:
        raise ValueError("readiness human-verified support count mismatch")


def publish_reference_bank_readiness(
    result: ReferenceBankReadinessResult,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    validate_reference_bank_readiness(result)
    directory = Path(output_dir)
    if directory.suffix:
        raise ValueError("reference bank readiness output must be a directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{uuid4().hex}.tmp"
    started_at = datetime.now(UTC)
    try:
        if directory.exists():
            raise FileExistsError(directory)
        staging.mkdir(parents=False, exist_ok=False)
        support_path = write_parquet(
            result.support_manifest,
            staging / REFERENCE_SUPPORT_MANIFEST_FILE,
            overwrite=False,
        )
        summary_path = write_parquet(
            result.summary,
            staging / REFERENCE_BANK_SUMMARY_FILE,
            overwrite=False,
        )
        payload = _json_roundtrip(result.readiness)
        artifacts = payload["artifacts"]
        artifacts["support_manifest"]["sha256"] = _sha256_file(support_path)
        artifacts["summary"]["sha256"] = _sha256_file(summary_path)
        payload["publication"] = {
            "command": "references.validate_readiness",
            "run_id": run_id or "not_instrumented",
            "pid": os.getpid(),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "network_requests": 0,
        }
        _validate_readiness_payload(payload, published=True)
        readiness_path = staging / REFERENCE_BANK_READINESS_FILE
        _write_json_create(readiness_path, payload)
        _rename_directory_no_replace(staging, directory)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        _write_failed_audit(
            directory,
            command="references.validate_readiness",
            run_id=run_id,
            started_at=started_at,
            error=exc,
        )
        raise
    return {
        "support_manifest": directory / REFERENCE_SUPPORT_MANIFEST_FILE,
        "summary": directory / REFERENCE_BANK_SUMMARY_FILE,
        "readiness": directory / REFERENCE_BANK_READINESS_FILE,
    }


def load_reference_bank_readiness(
    output_dir: str | Path,
    *,
    expected_registry_version: str | None = None,
    expected_target_accepted_taxon_key: str | None = None,
    expected_model_name: str | None = None,
    expected_preprocessing_version: str | None = None,
    expected_model_input_fingerprint: str | None = None,
    expected_readiness_sha256: str | None = None,
) -> ReferenceBankReadinessPermit:
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ValueError(f"reference bank readiness directory does not exist: {directory}")
    readiness_path = directory / REFERENCE_BANK_READINESS_FILE
    support_path = directory / REFERENCE_SUPPORT_MANIFEST_FILE
    summary_path = directory / REFERENCE_BANK_SUMMARY_FILE
    for path in (readiness_path, support_path, summary_path):
        if not path.is_file():
            raise ValueError(f"reference bank readiness artifact is missing: {path}")
    readiness_sha = _sha256_file(readiness_path)
    if expected_readiness_sha256 is not None:
        expected_sha = _fingerprint(
            expected_readiness_sha256,
            field="expected_readiness_sha256",
        )
        if readiness_sha != expected_sha:
            raise ValueError("reference bank readiness checksum does not match its pin")
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference bank readiness JSON must contain an object")
    _validate_readiness_payload(payload, published=True)
    artifacts = payload["artifacts"]
    support_sha = _sha256_file(support_path)
    summary_sha = _sha256_file(summary_path)
    if artifacts["support_manifest"]["sha256"] != support_sha:
        raise ValueError("reference support manifest checksum mismatch")
    if artifacts["summary"]["sha256"] != summary_sha:
        raise ValueError("reference bank summary checksum mismatch")
    support = pl.read_parquet(support_path)
    summary = pl.read_parquet(summary_path)
    validate_reference_support_manifest(support)
    validate_reference_bank_summary(summary)
    if payload["support_manifest_fingerprint"] != _support_manifest_fingerprint(
        support
    ):
        raise ValueError("reference support manifest semantic fingerprint mismatch")
    if payload["summary_fingerprint"] != _frame_fingerprint(summary):
        raise ValueError("reference bank summary semantic fingerprint mismatch")
    _validate_cross_artifact_readiness(payload, support=support, summary=summary)
    _expect_identity(
        payload,
        field="registry_version",
        expected=expected_registry_version,
    )
    _expect_identity(
        payload,
        field="target_accepted_taxon_key",
        expected=expected_target_accepted_taxon_key,
    )
    model = payload["model_input_identity"]
    _expect_model_name(model, expected=expected_model_name)
    _expect_identity(
        model,
        field="preprocessing_version",
        expected=expected_preprocessing_version,
    )
    _expect_identity(
        payload,
        field="model_input_fingerprint",
        expected=expected_model_input_fingerprint,
    )
    if not reference_readiness_allows_vision(payload):
        raise ValueError(
            "reference bank readiness does not permit vision: "
            f"status={payload['status']}"
        )
    return ReferenceBankReadinessPermit(
        status=str(payload["status"]),
        registry_version=str(payload["registry_version"]),
        reference_bank_version=str(payload["reference_bank_version"]),
        target_accepted_taxon_key=str(payload["target_accepted_taxon_key"]),
        policy_fingerprint=str(payload["policy_fingerprint"]),
        bank_fingerprint=str(payload["bank_fingerprint"]),
        support_manifest_fingerprint=str(payload["support_manifest_fingerprint"]),
        summary_fingerprint=str(payload["summary_fingerprint"]),
        split_assignments_fingerprint=str(payload["split_assignments_fingerprint"]),
        model_input_schema_version=str(model["schema_version"]),
        model_name=str(model["model_name"]),
        model_version=str(model["model_version"]),
        model_revision=str(model["model_revision"]),
        checkpoint_uri=str(model["checkpoint_uri"]),
        checkpoint_sha256=str(model["checkpoint_sha256"]),
        open_clip_version=str(model["open_clip_version"]),
        open_clip_config_sha256=str(model["open_clip_config_sha256"]),
        preprocessing_version=str(model["preprocessing_version"]),
        preprocessing_contract_fingerprint=str(
            model["preprocessing_contract_fingerprint"]
        ),
        preprocessing_attestation_fingerprint=str(
            model["preprocessing_attestation_fingerprint"]
        ),
        input_contract_version=str(model["input_contract_version"]),
        model_input_fingerprint=str(payload["model_input_fingerprint"]),
        readiness_sha256=readiness_sha,
        support_manifest_sha256=support_sha,
        summary_sha256=summary_sha,
        permits_reference_embedding=bool(payload["permits_reference_embedding"]),
        permits_provisional_scoring=bool(payload["permits_provisional_scoring"]),
        permits_calibrated_scoring=bool(payload["permits_calibrated_scoring"]),
        permits_scientific_release=bool(payload["permits_scientific_release"]),
        reference_admission_mode=str(payload["reference_admission_mode"]),
        admission_policy_fingerprint=str(payload["admission_policy_fingerprint"]),
        provisional_support_count=int(payload["provisional_support_count"]),
        human_verified_support_count=int(payload["human_verified_support_count"]),
        statistical_audit_required=bool(payload["statistical_audit_required"]),
        permits_prototype_creation=bool(payload["permits_prototype_creation"]),
        requires_downstream_flickr_review=bool(
            payload["requires_downstream_flickr_review"]
        ),
        candidate_set_fingerprints=tuple(payload["candidate_set_fingerprints"]),
        target_adult_requirements=_target_adult_requirement_statuses(payload),
    )


def reference_readiness_allows_vision(status_or_mapping: object) -> bool:
    if isinstance(status_or_mapping, Mapping):
        status = status_or_mapping.get("status")
        permits = status_or_mapping.get("permits_reference_embedding")
        return (
            status in PERMITTING_READINESS_STATUSES
            and permits is True
        )
    return status_or_mapping in PERMITTING_READINESS_STATUSES


def _target_adult_requirement_statuses(
    payload: Mapping[str, object],
) -> tuple[ReferenceBankRequirementStatus, ...]:
    checks = payload["checks"]
    assert isinstance(checks, list)
    target_check = next(
        item
        for item in checks
        if isinstance(item, Mapping) and item.get("check_id") == "target_adult_minimum"
    )
    observed = target_check["observed"]
    assert isinstance(observed, list)
    return tuple(
        ReferenceBankRequirementStatus(
            accepted_taxon_key=str(item["accepted_taxon_key"]),
            route=str(item["route"]),
            minimum_count=int(item["minimum_count"]),
            observed_count=int(item["observed_count"]),
            geo_cluster_id=(
                None
                if item.get("geo_cluster_id") is None
                else str(item["geo_cluster_id"])
            ),
        )
        for item in observed
        if isinstance(item, Mapping)
    )


def _validate_candidate_policy_bindings(
    candidate_species: pl.DataFrame,
    *,
    acquisition_plan: pl.DataFrame,
    acquisition_selections: pl.DataFrame,
    policy: ReferenceBankReadinessPolicy,
    registry_version: str,
    reference_bank_version: str,
    review_queue: pl.DataFrame,
) -> dict[str, object]:
    target_keys = set(candidate_species["target_accepted_taxon_key"])
    if target_keys != {policy.target_accepted_taxon_key}:
        raise ValueError("readiness policy target conflicts with candidate sets")
    candidate_keys = set(candidate_species["candidate_accepted_taxon_key"])
    geo_clusters = sorted(
        set(str(value) for value in candidate_species["geo_cluster_id"])
    )
    missing_requirements = sorted(
        {
            item.accepted_taxon_key
            for item in policy.requirements
            if item.accepted_taxon_key not in candidate_keys
        }
    )
    if missing_requirements:
        raise ValueError(
            "readiness requirements reference species outside the candidate union: "
            + ", ".join(missing_requirements)
        )
    requirement_clusters = {
        item.geo_cluster_id
        for item in policy.requirements
        if item.geo_cluster_id is not None
    }
    unknown_requirement_clusters = sorted(requirement_clusters - set(geo_clusters))
    if unknown_requirement_clusters:
        raise ValueError(
            "readiness requirements reference unknown geographic clusters: "
            + ", ".join(unknown_requirement_clusters)
        )
    uncovered_clusters = sorted(set(geo_clusters) - requirement_clusters)
    if uncovered_clusters:
        raise ValueError(
            "readiness policy lacks cluster-scoped geographic requirements: "
            + ", ".join(uncovered_clusters)
        )
    if set(acquisition_plan["target_accepted_taxon_key"]) != {
        policy.target_accepted_taxon_key
    }:
        raise ValueError("acquisition plan target conflicts with readiness policy")
    if not acquisition_selections.is_empty() and set(
        acquisition_selections["target_accepted_taxon_key"]
    ) != {policy.target_accepted_taxon_key}:
        raise ValueError("acquisition selections target conflicts with readiness policy")
    if not review_queue.is_empty() and set(review_queue["reference_bank_version"]) != {
        reference_bank_version
    }:
        raise ValueError("review queue reference bank version mismatch")
    plan_candidate_ids = set(acquisition_plan["candidate_set_id"])
    if not acquisition_selections.is_empty() and not set(
        acquisition_selections["candidate_set_id"]
    ) <= plan_candidate_ids:
        raise ValueError("acquisition selections reference an unknown candidate set")
    scientific_names: dict[str, str] = {}
    target_by_key: dict[str, bool] = {}
    for row in candidate_species.iter_rows(named=True):
        key = str(row["candidate_accepted_taxon_key"])
        name = str(row["scientific_name"])
        previous = scientific_names.setdefault(key, name)
        if previous != name:
            raise ValueError(f"candidate species {key} has conflicting names")
        target_by_key[key] = bool(row["target_candidate"])
    candidate_fingerprints = sorted(
        set(str(value) for value in candidate_species["candidate_set_fingerprint"])
    )
    if any(_SHA256_PATTERN.fullmatch(value) is None for value in candidate_fingerprints):
        raise ValueError("candidate set contains an invalid fingerprint")
    return {
        "target_key": policy.target_accepted_taxon_key,
        "candidate_keys": sorted(candidate_keys),
        "scientific_names": scientific_names,
        "target_by_key": target_by_key,
        "candidate_set_ids": sorted(
            set(str(value) for value in candidate_species["candidate_set_id"])
        ),
        "candidate_set_fingerprints": candidate_fingerprints,
        "geo_clusters": geo_clusters,
        "registry_version": registry_version,
    }


def _inventory_indexes(
    *,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    split_assignments: pl.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    indexes = {
        "observations": _unique_index(
            observations,
            key="reference_observation_id",
            artifact="reference observations",
        ),
        "media_candidates": _unique_index(
            media_candidates,
            key="reference_media_id",
            artifact="reference media candidates",
        ),
        "media_objects": _unique_index(
            media_objects,
            key="reference_media_id",
            artifact="reference media objects",
        ),
        "split_assignments": _unique_index(
            split_assignments,
            key="reference_media_id",
            artifact="split assignments",
        ),
    }
    candidate_observation_ids = {
        str(row["reference_observation_id"])
        for row in indexes["media_candidates"].values()
    }
    if not candidate_observation_ids <= set(indexes["observations"]):
        raise ValueError("media candidates reference unknown observations")
    if not set(indexes["media_objects"]) <= set(indexes["media_candidates"]):
        raise ValueError("media objects reference unknown candidates")
    if not set(indexes["split_assignments"]) <= set(indexes["media_objects"]):
        raise ValueError("split assignments reference unknown media objects")
    return indexes


def _validate_reference_identity_bindings(
    *,
    candidate_species: pl.DataFrame,
    acquisition_plan: pl.DataFrame,
    acquisition_selections: pl.DataFrame,
    review_queue: pl.DataFrame,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    registry_version: str,
) -> None:
    candidate_names: dict[str, str] = {}
    candidate_scopes: dict[tuple[str, str, str], str] = {}
    candidate_clusters: dict[str, set[str]] = defaultdict(set)
    candidate_union_id = make_reference_candidate_union_id(candidate_species)
    for row in candidate_species.iter_rows(named=True):
        taxon_key = str(row["candidate_accepted_taxon_key"])
        scientific_name = str(row["scientific_name"])
        candidate_names.setdefault(taxon_key, scientific_name)
        cluster_id = str(row["geo_cluster_id"])
        candidate_clusters[taxon_key].add(cluster_id)
        candidate_scopes[
            (str(row["candidate_set_id"]), cluster_id, taxon_key)
        ] = scientific_name

    for observation in indexes["observations"].values():
        observation_id = str(observation["reference_observation_id"])
        if observation["registry_version"] != registry_version:
            raise ValueError(
                "reference observation registry version mismatch: " + observation_id
            )
        taxon_key = observation["accepted_taxon_key"]
        scientific_name = observation["reconciled_scientific_name"]
        if taxon_key is None or scientific_name is None:
            continue
        expected_name = candidate_names.get(str(taxon_key))
        if expected_name is None:
            raise ValueError(
                "reference observation taxon is outside the candidate union: "
                + observation_id
            )
        if scientific_name != expected_name:
            raise ValueError(
                "reference observation scientific name conflicts with candidate union: "
                + observation_id
            )
        cluster_id = str(observation["geo_cluster_id"] or "no_geo")
        if cluster_id not in candidate_clusters[str(taxon_key)]:
            raise ValueError(
                "reference observation geographic cluster conflicts with candidate union: "
                + observation_id
            )

    plan_bindings: set[tuple[object, ...]] = set()
    for row in acquisition_plan.iter_rows(named=True):
        if row["candidate_set_id"] != candidate_union_id:
            raise ValueError("acquisition plan candidate union identity is stale")
        taxon_key = str(row["candidate_accepted_taxon_key"])
        cluster_id = str(row["geo_cluster_id"])
        expected_name = candidate_names.get(taxon_key)
        if expected_name is None:
            raise ValueError("acquisition plan scope is outside the candidate union")
        if cluster_id not in candidate_clusters[taxon_key]:
            raise ValueError("acquisition plan cluster is outside the candidate union")
        if row["scientific_name"] != expected_name:
            raise ValueError(
                "acquisition plan scientific name conflicts with candidate union"
            )
        plan_bindings.add(_plan_selection_binding(row))

    selected_group_ids: set[str] = set()
    for selection in acquisition_selections.iter_rows(named=True):
        if _plan_selection_binding(selection) not in plan_bindings:
            raise ValueError("acquisition selection does not match its plan row")
        if selection["candidate_set_id"] != candidate_union_id:
            raise ValueError("acquisition selection candidate union identity is stale")
        scope = (
            str(selection["source_candidate_set_id"]),
            str(selection["geo_cluster_id"]),
            str(selection["candidate_accepted_taxon_key"]),
        )
        if candidate_scopes.get(scope) != selection["scientific_name"]:
            raise ValueError(
                "acquisition selection identity conflicts with candidate union"
            )
        media_id = str(selection["reference_media_id"])
        candidate = indexes["media_candidates"].get(media_id)
        if candidate is None:
            raise ValueError(f"selected media has no media candidate: {media_id}")
        observation_id = str(selection["reference_observation_id"])
        observation = indexes["observations"].get(observation_id)
        if observation is None:
            raise ValueError(f"selected media has no source observation: {media_id}")
        if candidate["reference_observation_id"] != observation_id:
            raise ValueError("selected media observation provenance is inconsistent")
        if not (
            selection["source"] == candidate["source"] == observation["source"]
        ):
            raise ValueError("selected media source provenance is inconsistent")
        if (
            selection["candidate_accepted_taxon_key"]
            != observation["accepted_taxon_key"]
            or selection["scientific_name"]
            != observation["reconciled_scientific_name"]
        ):
            raise ValueError("selected media taxonomy provenance is inconsistent")
        if str(observation["geo_cluster_id"] or "no_geo") != str(
            selection["geo_cluster_id"]
        ):
            raise ValueError("selected media geographic provenance is inconsistent")
        if selection["source_snapshot_version"] != candidate[
            "source_snapshot_version"
        ] or selection["licence"] != candidate["licence"]:
            raise ValueError("selected media source snapshot or licence is stale")
        if candidate["source_snapshot_version"] != observation[
            "source_snapshot_version"
        ]:
            raise ValueError("selected media source snapshot provenance is inconsistent")
        object_row = indexes["media_objects"].get(media_id)
        if object_row is not None:
            selected_group_ids.add(str(object_row["duplicate_group_id"]))

    for queue_row in review_queue.iter_rows(named=True):
        media_id = str(queue_row["reference_media_id"])
        candidate = indexes["media_candidates"].get(media_id)
        object_row = indexes["media_objects"].get(media_id)
        if candidate is None or object_row is None:
            raise ValueError(f"review queue media inventory is incomplete: {media_id}")
        observation_id = str(candidate["reference_observation_id"])
        observation = indexes["observations"].get(observation_id)
        if observation is None:
            raise ValueError(f"review queue media has no source observation: {media_id}")
        if str(object_row["duplicate_group_id"]) not in selected_group_ids:
            raise ValueError("review queue media was not produced by a selected group")
        expected_fields = {
            "reference_observation_id": observation_id,
            "canonical_reference_media_id": object_row[
                "canonical_reference_media_id"
            ],
            "durable_preview_uri": object_row["source_object_uri"],
            "media_object_fingerprint": object_row["object_fingerprint"],
            "duplicate_group_id": object_row["duplicate_group_id"],
            "source": candidate["source"],
            "provider_media_id": candidate["provider_media_id"],
            "provider_verification_status": candidate["verification_status"],
            "accepted_taxon_key": observation["accepted_taxon_key"],
            "scientific_name": observation["reconciled_scientific_name"],
            "creator": candidate["creator"],
            "rights_holder": candidate["rights_holder"],
            "licence": candidate["licence"],
            "licence_uri": candidate["licence_uri"],
            "licence_policy_status": object_row["licence_policy_status"],
            "attribution": candidate["attribution"],
        }
        mismatched = sorted(
            field
            for field, expected in expected_fields.items()
            if queue_row[field] != expected
        )
        if mismatched:
            raise ValueError(
                "review queue provenance conflicts with current inventory: "
                + ", ".join(mismatched)
            )


def _plan_selection_binding(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["acquisition_plan_id"],
        row["target_accepted_taxon_key"],
        row["candidate_set_id"],
        row["candidate_accepted_taxon_key"],
        row["scientific_name"],
        row["geo_cluster_id"],
        row["life_stage"],
        row["visual_domain"],
        row["source"],
        row["fallback_level"],
        row["selection_strategy"],
        row["selection_seed"],
        row["plan_configuration_fingerprint"],
    )


def _build_support_rows(
    *,
    review: Any,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    policy: ReferenceBankReadinessPolicy,
    admission_policy: ReferenceAdmissionPolicy,
    registry_version: str,
    reference_bank_version: str,
    bank_fingerprint: str,
) -> tuple[list[dict[str, object]], list[str]]:
    outcomes = {
        str(row["review_request_id"]): row
        for row in review.outcomes.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    issues: list[str] = []
    seen_media: set[str] = set()
    for resolved in review.verified.iter_rows(named=True):
        media_id = str(resolved["reference_media_id"])
        if media_id in seen_media:
            issues.append(f"duplicate resolved review media: {media_id}")
            continue
        seen_media.add(media_id)
        candidate = indexes["media_candidates"].get(media_id)
        object_row = indexes["media_objects"].get(media_id)
        if candidate is None or object_row is None:
            issues.append(f"resolved review media has no inventory object: {media_id}")
            continue
        observation_id = str(candidate["reference_observation_id"])
        observation = indexes["observations"].get(observation_id)
        if observation is None:
            issues.append(f"resolved review media has no observation: {media_id}")
            continue
        canonical_id = str(object_row["canonical_reference_media_id"] or "")
        if canonical_id != media_id:
            issues.append(f"resolved review media is not canonical: {media_id}")
            continue
        request_id = str(resolved["review_request_id"])
        outcome = outcomes.get(request_id)
        if outcome is None:
            issues.append(f"resolved review has no outcome: {request_id}")
            continue
        assignment = indexes["split_assignments"].get(media_id)
        included = bool(assignment and assignment["included"])
        route = _reference_route(
            life_stage=str(resolved["resolved_life_stage"]),
            visual_domain=str(resolved["resolved_visual_domain"]),
        )
        if route is None:
            # A verified but prohibited domain remains visible in review counts,
            # but it cannot enter the frozen support projection.
            if included:
                issues.append(
                    f"included verified media has no supported reference route: {media_id}"
                )
            continue
        accepted_licence = str(candidate["licence_policy_status"]) in (
            policy.accepted_licence_policy_statuses
        ) and str(object_row["licence_policy_status"]) in (
            policy.accepted_licence_policy_statuses
        )
        blockers = set(str(value) for value in outcome["blocker_reasons"])
        if not included:
            blockers.add("not_included_in_split")
        if not accepted_licence:
            blockers.add("licence_not_accepted")
        if object_row["decode_status"] != "valid":
            blockers.add("media_object_not_decodable")
        if not _complete_attribution(candidate, observation):
            blockers.add("source_attribution_incomplete")
        human_verified_identity = (
            outcome["review_status"] == "completed"
            and outcome["resolved_verification_status"] == "verified"
            and bool(outcome["target_identity_verified"])
        )
        provider_asserted_identity = bool(
            _present(observation["source_taxon_id"])
            and _present(observation["supplied_scientific_name"])
        )
        admission = evaluate_support_admission(
            SupportAdmissionEvidence(
                source=str(candidate["source"]),
                review_status=str(outcome["review_status"]),
                verification_status=str(
                    outcome["resolved_verification_status"]
                ),
                target_identity_verified=bool(
                    outcome["target_identity_verified"]
                ),
                human_rejected=False,
                human_rejection_reasons=(),
                provider_assertion_passed=provider_asserted_identity,
                automated_admission_gates_passed=False,
                route_compatible=route in admission_policy.allowed_unreviewed_routes,
                canonical_media=canonical_id == media_id,
                deduplication_completed=True,
                provisional_support_declared=False,
                policy_permits_human_verified_support=True,
                policy_permits_provisional_support=False,
                exclusion_reasons=tuple(blockers),
            ),
            admission_policy,
        )
        support_eligible = admission.eligible
        base: dict[str, object] = {
            "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
            "reference_bank_version": reference_bank_version,
            "registry_version": registry_version,
            "reference_media_id": media_id,
            "canonical_reference_media_id": canonical_id,
            "reference_observation_id": observation_id,
            "reference_admission_mode": admission_policy.mode,
            "reference_admission_policy_version": admission_policy.policy_version,
            "reference_admission_policy_fingerprint": admission_policy.fingerprint,
            "identity_evidence_basis": (
                "human_verified"
                if human_verified_identity
                else admission.evidence_path
            ),
            "provider_asserted_identity": provider_asserted_identity,
            "provider_asserted_taxon_key": observation["source_taxon_id"],
            "provider_asserted_scientific_name": observation[
                "supplied_scientific_name"
            ],
            "provider_dataset_key": observation["source_dataset_key"],
            "provider_quality_status": observation["identification_quality"],
            "human_review_status": outcome["review_status"],
            "human_verified_identity": human_verified_identity,
            "provisional_support": admission.provisional,
            "statistical_audit_required": (
                admission.provisional and admission_policy.require_statistical_audit
            ),
            "admission_status": admission.admission_status,
            "admission_reasons": list(admission.reasons),
            "reference_quality_flags": [],
            "route_evidence_basis": "human_verified_review",
            "geographic_prototype_eligible": bool(
                support_eligible and _present(observation["geo_cluster_id"])
            ),
            "review_request_id": request_id,
            "review_decision_ids": sorted(
                str(value) for value in resolved["effective_decision_ids"]
            ),
            "reviewer_ids": sorted(
                str(value) for value in resolved["effective_reviewer_ids"]
            ),
            "source": candidate["source"],
            "source_observation_id": observation["source_observation_id"],
            "provider_media_id": candidate["provider_media_id"],
            "source_record_url": observation["source_record_url"],
            "source_snapshot_version": observation["source_snapshot_version"],
            "source_dataset_key": observation["source_dataset_key"],
            "accepted_taxon_key": observation["accepted_taxon_key"],
            "scientific_name": observation["reconciled_scientific_name"],
            "target_candidate": observation["accepted_taxon_key"]
            == policy.target_accepted_taxon_key,
            "geo_cluster_id": observation["geo_cluster_id"],
            "observer_id": observation["observer_id"],
            "observed_at": observation["observed_at"],
            "latitude": observation["latitude"],
            "longitude": observation["longitude"],
            "source_object_uri": object_row["source_object_uri"],
            "image_sha256": object_row["sha256"],
            "perceptual_hash": object_row["perceptual_hash"],
            "object_fingerprint": object_row["object_fingerprint"],
            "duplicate_group_id": object_row["duplicate_group_id"],
            "duplicate_type": object_row["duplicate_type"],
            "creator": candidate["creator"],
            "rights_holder": candidate["rights_holder"],
            "licence": candidate["licence"],
            "licence_uri": candidate["licence_uri"],
            "licence_policy_status": candidate["licence_policy_status"],
            "attribution": candidate["attribution"],
            "review_status": outcome["review_status"],
            "verification_status": outcome["resolved_verification_status"],
            "target_identity_verified": outcome["target_identity_verified"],
            "life_stage": resolved["resolved_life_stage"],
            "visual_domain": resolved["resolved_visual_domain"],
            "view": resolved["resolved_view"],
            "route": route,
            "support_split": assignment["support_split"] if included else None,
            "support_eligible": support_eligible,
            "exclusion_reasons": sorted(blockers),
            "split_assignment_fingerprint": assignment[
                "assignment_fingerprint"
            ]
            if assignment is not None
            else None,
            "reference_bank_fingerprint": bank_fingerprint,
        }
        base["support_row_fingerprint"] = _support_row_fingerprint(base)
        rows.append(base)
    return rows, sorted(set(issues))


def _build_provisional_support_rows(
    *,
    provisional_support: pl.DataFrame,
    review: Any,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    policy: ReferenceBankReadinessPolicy,
    admission_policy: ReferenceAdmissionPolicy,
    registry_version: str,
    reference_bank_version: str,
    bank_fingerprint: str,
    human_support_media_ids: set[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """Project compiler support into readiness while honoring later review."""

    outcomes = {
        str(row["reference_media_id"]): row
        for row in review.outcomes.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    issues: list[str] = []
    for source_row in provisional_support.iter_rows(named=True):
        media_id = str(source_row["reference_media_id"])
        if media_id in human_support_media_ids:
            continue
        candidate = indexes["media_candidates"].get(media_id)
        object_row = indexes["media_objects"].get(media_id)
        assignment = indexes["split_assignments"].get(media_id)
        if candidate is None or object_row is None:
            issues.append(f"provisional support lacks inventory object: {media_id}")
            continue
        observation_id = str(candidate["reference_observation_id"])
        observation = indexes["observations"].get(observation_id)
        if observation is None:
            issues.append(f"provisional support lacks observation: {media_id}")
            continue
        _validate_provisional_inventory_binding(
            source_row=source_row,
            candidate=candidate,
            object_row=object_row,
            observation=observation,
            admission_policy=admission_policy,
        )
        outcome = outcomes.get(media_id)
        human_verified = bool(
            outcome
            and outcome["review_status"] == "completed"
            and outcome["resolved_verification_status"] == "verified"
            and outcome["target_identity_verified"]
        )
        if human_verified:
            issues.append(
                f"human-verified provisional media lacks strict support row: {media_id}"
            )
            continue
        human_rejected = bool(
            outcome
            and outcome["review_status"] == "completed"
            and outcome["resolved_verification_status"] != "verified"
        )
        rejection_reasons = (
            tuple(str(value) for value in outcome["blocker_reasons"])
            if human_rejected and outcome and outcome["blocker_reasons"]
            else (("human_identity_rejected",) if human_rejected else ())
        )
        included = bool(assignment and assignment["included"])
        automated_passed = bool(
            source_row["admission_decision"] == "admitted"
            and source_row["automated_gate_ids"]
            and set(source_row["automated_gate_dispositions"]) == {"passed"}
        )
        route = str(source_row["route"])
        blockers: set[str] = set()
        if not included:
            blockers.add("not_included_in_split")
        admission = evaluate_support_admission(
            SupportAdmissionEvidence(
                source=str(source_row["source"]),
                review_status=(
                    str(outcome["review_status"])
                    if outcome is not None
                    else "not_requested"
                ),
                verification_status=(
                    str(outcome["resolved_verification_status"] or "unreviewed")
                    if outcome is not None
                    else "unreviewed"
                ),
                target_identity_verified=False,
                human_rejected=human_rejected,
                human_rejection_reasons=rejection_reasons,
                provider_assertion_passed=(
                    source_row["provider_assertion_status"]
                    == "provider_asserted_unreviewed"
                    and source_row["provider_assertion_identity_basis"]
                    == "gbif_provider_asserted"
                ),
                automated_admission_gates_passed=automated_passed,
                route_compatible=route
                in admission_policy.allowed_unreviewed_routes,
                canonical_media=(
                    source_row["canonical_reference_media_id"] == media_id
                ),
                deduplication_completed=(
                    source_row["duplicate_group_id"] is not None
                    and source_row["duplicate_resolution_status"] == "resolved"
                ),
                provisional_support_declared=bool(
                    source_row["provisional_support"]
                ),
                policy_permits_human_verified_support=True,
                policy_permits_provisional_support=(
                    admission_policy.mode == DEFAULT_REFERENCE_ADMISSION_MODE
                ),
                exclusion_reasons=tuple(blockers),
            ),
            admission_policy,
        )
        review_status = _support_human_review_status(outcome)
        provider_asserted = bool(
            _present(observation["source_taxon_id"])
            and _present(observation["supplied_scientific_name"])
        )
        eligible = admission.eligible
        base: dict[str, object] = {
            "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
            "reference_bank_version": reference_bank_version,
            "registry_version": registry_version,
            "reference_media_id": media_id,
            "canonical_reference_media_id": media_id,
            "reference_observation_id": observation_id,
            "reference_admission_mode": admission_policy.mode,
            "reference_admission_policy_version": admission_policy.policy_version,
            "reference_admission_policy_fingerprint": admission_policy.fingerprint,
            "identity_evidence_basis": admission.evidence_path,
            "provider_asserted_identity": provider_asserted,
            "provider_asserted_taxon_key": observation["source_taxon_id"],
            "provider_asserted_scientific_name": observation[
                "supplied_scientific_name"
            ],
            "provider_dataset_key": observation["source_dataset_key"],
            "provider_quality_status": observation["identification_quality"],
            "human_review_status": review_status,
            "human_verified_identity": False,
            "provisional_support": admission.provisional,
            "statistical_audit_required": admission.provisional,
            "admission_status": admission.admission_status,
            "admission_reasons": list(admission.reasons),
            "reference_quality_flags": [],
            "route_evidence_basis": "yoloe",
            "geographic_prototype_eligible": bool(
                eligible and _present(observation["geo_cluster_id"])
            ),
            "review_request_id": (
                outcome["review_request_id"] if outcome is not None else None
            ),
            "review_decision_ids": (
                list(outcome["effective_decision_ids"])
                if outcome is not None
                else []
            ),
            "reviewer_ids": (
                list(outcome["effective_reviewer_ids"])
                if outcome is not None
                else []
            ),
            "source": candidate["source"],
            "source_observation_id": observation["source_observation_id"],
            "provider_media_id": candidate["provider_media_id"],
            "source_record_url": observation["source_record_url"],
            "source_snapshot_version": observation["source_snapshot_version"],
            "source_dataset_key": observation["source_dataset_key"],
            "accepted_taxon_key": observation["accepted_taxon_key"],
            "scientific_name": observation["reconciled_scientific_name"],
            "target_candidate": observation["accepted_taxon_key"]
            == policy.target_accepted_taxon_key,
            "geo_cluster_id": observation["geo_cluster_id"],
            "observer_id": observation["observer_id"],
            "observed_at": observation["observed_at"],
            "latitude": observation["latitude"],
            "longitude": observation["longitude"],
            "source_object_uri": object_row["source_object_uri"],
            "image_sha256": object_row["sha256"],
            "perceptual_hash": object_row["perceptual_hash"],
            "object_fingerprint": object_row["object_fingerprint"],
            "duplicate_group_id": object_row["duplicate_group_id"],
            "duplicate_type": object_row["duplicate_type"],
            "creator": candidate["creator"],
            "rights_holder": candidate["rights_holder"],
            "licence": candidate["licence"],
            "licence_uri": candidate["licence_uri"],
            "licence_policy_status": candidate["licence_policy_status"],
            "attribution": candidate["attribution"],
            "review_status": review_status,
            "verification_status": (
                str(outcome["resolved_verification_status"] or "unreviewed")
                if outcome is not None
                else "unreviewed"
            ),
            "target_identity_verified": False,
            "life_stage": source_row["provisional_life_stage"],
            "visual_domain": source_row["provisional_visual_domain"],
            "view": "unknown",
            "route": route,
            "support_split": assignment["support_split"] if included else None,
            "support_eligible": eligible,
            "exclusion_reasons": [] if eligible else list(admission.reasons),
            "split_assignment_fingerprint": (
                assignment["assignment_fingerprint"]
                if assignment is not None
                else None
            ),
            "reference_bank_fingerprint": bank_fingerprint,
        }
        base["support_row_fingerprint"] = _support_row_fingerprint(base)
        rows.append(base)
    return rows, sorted(set(issues))


def _validate_provisional_inventory_binding(
    *,
    source_row: Mapping[str, object],
    candidate: Mapping[str, object],
    object_row: Mapping[str, object],
    observation: Mapping[str, object],
    admission_policy: ReferenceAdmissionPolicy,
) -> None:
    expected = {
        "reference_observation_id": candidate["reference_observation_id"],
        "source": candidate["source"],
        "provider_media_id": candidate["provider_media_id"],
        "accepted_taxon_key": observation["accepted_taxon_key"],
        "scientific_name": observation["reconciled_scientific_name"],
        "canonical_reference_media_id": object_row[
            "canonical_reference_media_id"
        ],
        "image_sha256": object_row["sha256"],
        "reference_admission_mode": admission_policy.mode,
        "reference_admission_policy_version": admission_policy.policy_version,
        "reference_admission_policy_fingerprint": admission_policy.fingerprint,
    }
    mismatches = sorted(
        field for field, value in expected.items() if source_row[field] != value
    )
    if mismatches:
        raise ValueError(
            "provisional support conflicts with readiness inputs: "
            + ", ".join(mismatches)
        )


def _support_human_review_status(
    outcome: Mapping[str, object] | None,
) -> str:
    if outcome is None:
        return "not_requested"
    status = str(outcome["review_status"])
    if status == "second_review_required":
        return "pending"
    if status in REFERENCE_HUMAN_REVIEW_STATUSES:
        return status
    return "pending"


def _build_reference_bank_summary(
    *,
    candidate_context: Mapping[str, object],
    policy: ReferenceBankReadinessPolicy,
    acquisition_plan: pl.DataFrame,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    media_objects: pl.DataFrame,
    review: Any,
    support_manifest: pl.DataFrame,
    registry_version: str,
    reference_bank_version: str,
    bank_fingerprint: str,
    support_fingerprint: str,
    split_fingerprint: str,
) -> pl.DataFrame:
    scientific_names = candidate_context["scientific_names"]
    target_by_key = candidate_context["target_by_key"]
    assert isinstance(scientific_names, Mapping)
    assert isinstance(target_by_key, Mapping)
    specs: dict[tuple[str, str, str, str, str, str], int] = {}
    for requirement in policy.requirements:
        life_stage, visual_domain = reference_route_dimensions(requirement.route)
        key = (
            requirement.accepted_taxon_key,
            requirement.geo_cluster_id or "all",
            requirement.route,
            life_stage,
            visual_domain,
            "support_train",
        )
        specs[key] = requirement.minimum_count
    for row in support_manifest.iter_rows(named=True):
        key = (
            str(row["accepted_taxon_key"]),
            str(row["geo_cluster_id"] or "no_geo"),
            str(row["route"]),
            str(row["life_stage"]),
            str(row["visual_domain"]),
            str(row["support_split"] or "unassigned"),
        )
        specs.setdefault(key, 0)

    observation_rows = observations.to_dicts()
    candidates_by_observation: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in media_candidates.iter_rows(named=True):
        candidates_by_observation[str(row["reference_observation_id"])].append(row)
    objects_by_media = {
        str(row["reference_media_id"]): row
        for row in media_objects.iter_rows(named=True)
    }
    queue_by_request = {
        str(row["review_request_id"]): row for row in review.queue.iter_rows(named=True)
    }
    outcome_by_request = {
        str(row["review_request_id"]): row
        for row in review.outcomes.iter_rows(named=True)
    }
    resolved_verified = review.verified.to_dicts()
    resolved_excluded = review.excluded.to_dicts()
    approvals = {_shortfall_key(item): item for item in policy.documented_shortfalls}
    plan_fingerprints = set(str(value) for value in acquisition_plan["plan_configuration_fingerprint"])

    rows: list[dict[str, object]] = []
    for spec, required_count in sorted(specs.items()):
        taxon_key, cluster_id, route, life_stage, visual_domain, support_split = spec
        matching_observations = [
            row
            for row in observation_rows
            if _inventory_row_matches(
                row,
                taxon_key=taxon_key,
                cluster_id=cluster_id,
                route=route,
            )
        ]
        matching_observation_ids = {
            str(row["reference_observation_id"]) for row in matching_observations
        }
        candidate_rows = [
            candidate
            for observation_id in matching_observation_ids
            for candidate in candidates_by_observation.get(observation_id, [])
        ]
        candidate_media_ids = {
            str(row["reference_media_id"]) for row in candidate_rows
        }
        downloaded = [
            objects_by_media[media_id]
            for media_id in candidate_media_ids
            if media_id in objects_by_media
            and objects_by_media[media_id]["decode_status"] == "valid"
        ]
        deduplicated_ids = {
            str(row["canonical_reference_media_id"])
            for row in downloaded
            if row["canonical_reference_media_id"]
        }
        relevant_requests = []
        for request_id, queue_row in queue_by_request.items():
            if str(queue_row["accepted_taxon_key"]) != taxon_key:
                continue
            candidate = objects_by_media.get(str(queue_row["reference_media_id"]))
            if candidate is None:
                continue
            media_candidate = next(
                (
                    row
                    for row in candidate_rows
                    if row["reference_media_id"] == queue_row["reference_media_id"]
                ),
                None,
            )
            if media_candidate is None:
                continue
            if not _cluster_matches_observation_id(
                str(media_candidate["reference_observation_id"]),
                matching_observation_ids,
                cluster_id=cluster_id,
            ):
                continue
            queue_route = _reference_route(
                life_stage=str(queue_row["life_stage"]),
                visual_domain=str(queue_row["visual_domain"]),
            )
            if queue_route == route:
                relevant_requests.append(request_id)
        completed_request_ids = {
            request_id
            for request_id in relevant_requests
            if outcome_by_request[request_id]["review_status"] == "completed"
        }
        pending_request_ids = set(relevant_requests) - completed_request_ids
        verified_rows = [
            row
            for row in resolved_verified
            if str(row["review_request_id"]) in completed_request_ids
            and _reference_route(
                life_stage=str(row["resolved_life_stage"]),
                visual_domain=str(row["resolved_visual_domain"]),
            )
            == route
        ]
        excluded_rows = [
            row
            for row in resolved_excluded
            if str(row["review_request_id"]) in completed_request_ids
        ]
        support_rows = [
            row
            for row in support_manifest.iter_rows(named=True)
            if str(row["accepted_taxon_key"]) == taxon_key
            and (cluster_id == "all" or str(row["geo_cluster_id"] or "no_geo") == cluster_id)
            and str(row["route"]) == route
            and str(row["support_split"] or "unassigned") == support_split
        ]
        eligible_rows = [row for row in support_rows if row["support_eligible"]]
        provider_asserted_rows = [
            row for row in support_rows if row["provider_asserted_identity"]
        ]
        human_verified_rows = [
            row for row in support_rows if row["human_verified_identity"]
        ]
        provisional_rows = [
            row for row in eligible_rows if row["provisional_support"]
        ]
        flagged_rows = [
            row
            for row in support_rows
            if row["admission_status"] == "review_required"
            or row["human_review_status"] in {"pending", "in_review", "conflict"}
        ]
        automated_exclusions = [
            row
            for row in support_rows
            if row["admission_status"] == "excluded"
            and row["human_review_status"] != "completed"
        ]
        shortfall = max(0, required_count - len(eligible_rows))
        approval = approvals.get((taxon_key, cluster_id if cluster_id != "all" else None, route))
        documented_count = 0
        if (
            shortfall
            and approval is not None
            and approval.plan_configuration_fingerprint in plan_fingerprints
            and len(eligible_rows) >= approval.approved_minimum_count
        ):
            documented_count = shortfall
        base: dict[str, object] = {
            "schema_version": REFERENCE_BANK_SUMMARY_SCHEMA_VERSION,
            "reference_bank_version": reference_bank_version,
            "registry_version": registry_version,
            "accepted_taxon_key": taxon_key,
            "scientific_name": str(scientific_names.get(taxon_key) or ""),
            "target_candidate": bool(target_by_key.get(taxon_key, False)),
            "geo_cluster_id": cluster_id,
            "route": route,
            "life_stage": life_stage,
            "visual_domain": visual_domain,
            "support_split": support_split,
            "required_count": required_count,
            "candidate_count": len(candidate_rows),
            "downloaded_count": len(downloaded),
            "deduplicated_count": len(deduplicated_ids),
            "reviewed_count": len(completed_request_ids),
            "verified_count": len(verified_rows),
            "eligible_count": len(eligible_rows),
            "excluded_count": len(excluded_rows),
            "pending_review_count": len(pending_request_ids),
            "provider_asserted_count": len(provider_asserted_rows),
            "provider_asserted_eligible_count": sum(
                bool(row["support_eligible"]) for row in provider_asserted_rows
            ),
            "human_verified_count": len(human_verified_rows),
            "human_verified_eligible_count": sum(
                bool(row["support_eligible"]) for row in human_verified_rows
            ),
            "provisional_support_count": len(provisional_rows),
            "strict_support_count": sum(
                bool(row["support_eligible"]) for row in human_verified_rows
            ),
            "flagged_for_review_count": len(flagged_rows),
            "excluded_by_automated_qa_count": len(automated_exclusions),
            "excluded_by_human_review_count": len(excluded_rows),
            "shortfall_count": shortfall,
            "documented_shortfall_count": documented_count,
            "source_count": len({str(row["source"]) for row in support_rows}),
            "licence_count": len({str(row["licence"]) for row in support_rows}),
            "creator_count": len({str(row["creator"]) for row in support_rows}),
            "observer_count": len({str(row["observer_id"]) for row in support_rows}),
            "observation_count": len(
                {str(row["reference_observation_id"]) for row in support_rows}
            ),
            "geographic_cluster_count": len(
                {str(row["geo_cluster_id"] or "no_geo") for row in support_rows}
            ),
            "reference_bank_fingerprint": bank_fingerprint,
            "support_manifest_fingerprint": support_fingerprint,
            "split_assignments_fingerprint": split_fingerprint,
        }
        base["summary_row_fingerprint"] = _summary_row_fingerprint(base)
        rows.append(base)
    return _strict_frame(
        rows,
        schema=reference_bank_summary_schema(),
        sort_by=_SUMMARY_SORT,
    )


def _readiness_checks(
    *,
    candidate_context: Mapping[str, object],
    policy: ReferenceBankReadinessPolicy,
    admission_policy: ReferenceAdmissionPolicy,
    acquisition_plan: pl.DataFrame,
    acquisition_selections: pl.DataFrame,
    review: Any,
    support_manifest: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
    split_assignments: pl.DataFrame,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    structural_issues: Sequence[str],
    model_identity: ReferenceModelInputIdentity,
    bank_fingerprint: str,
    support_fingerprint: str,
    summary_fingerprint: str,
    split_fingerprint: str,
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    eligible_all = [
        row
        for row in support_manifest.iter_rows(named=True)
        if row["support_eligible"]
    ]
    eligible = [
        row
        for row in eligible_all
        if row["support_eligible"] and row["support_split"] == "support_train"
    ]
    requirement_results: list[dict[str, object]] = []
    for requirement in policy.requirements:
        observed = sum(
            1 for row in eligible if _support_matches_requirement(row, requirement)
        )
        requirement_results.append(
            {
                "requirement": requirement,
                "observed": observed,
                "shortfall": max(0, requirement.minimum_count - observed),
            }
        )
    accepted_shortfalls, approval_issues = _validate_documented_shortfalls(
        policy=policy,
        requirement_results=requirement_results,
        acquisition_plan=acquisition_plan,
    )
    all_structural_issues = sorted(set(structural_issues) | set(approval_issues))
    target_results = [
        item
        for item in requirement_results
        if item["requirement"].accepted_taxon_key
        == policy.target_accepted_taxon_key
        and item["requirement"].route == "adult_field"
    ]
    competitor_results = [
        item
        for item in requirement_results
        if item["requirement"].accepted_taxon_key
        != policy.target_accepted_taxon_key
    ]
    geographic_results = [
        item
        for item in requirement_results
        if item["requirement"].geo_cluster_id is not None
    ]

    unresolved_relationships = [
        row
        for row in duplicate_relationships.iter_rows(named=True)
        if row["resolution_status"] != "resolved"
    ]
    included_assignments = [
        row
        for row in split_assignments.iter_rows(named=True)
        if row["included"]
    ]
    included_media_ids = {
        str(row["reference_media_id"]) for row in included_assignments
    }
    review_by_media = {
        str(row["reference_media_id"]): row
        for row in review.outcomes.iter_rows(named=True)
    }
    pending_review = [
        row
        for row in review.outcomes.iter_rows(named=True)
        if row["review_status"] != "completed"
        and str(row["reference_media_id"]) in included_media_ids
    ]
    pending_review_media = {
        str(row["reference_media_id"]) for row in pending_review
    }
    pending_target_review = [
        row
        for row in pending_review
        if _pending_outcome_could_supply_target(
            row,
            target_accepted_taxon_key=policy.target_accepted_taxon_key,
            indexes=indexes,
        )
    ]
    unverified_media = sorted(
        str(row["reference_media_id"])
        for row in included_assignments
        if not _outcome_is_human_verified(
            review_by_media.get(str(row["reference_media_id"]))
        )
    )
    provisional_rows = [
        row for row in eligible_all if bool(row["provisional_support"])
    ]
    admission_failures = sorted(
        str(row["reference_media_id"])
        for row in support_manifest.iter_rows(named=True)
        if bool(row["support_eligible"])
        != (row["admission_status"] == "admitted")
        or (
            row["support_eligible"]
            and row["identity_evidence_basis"]
            not in {"human_verified", "gbif_provider_asserted"}
        )
    )
    provider_integrity_failures = sorted(
        str(row["reference_media_id"])
        for row in provisional_rows
        if row["source"] != "gbif"
        or not row["provider_asserted_identity"]
        or not _present(row["provider_asserted_taxon_key"])
        or not _present(row["provider_asserted_scientific_name"])
    )
    automated_qa_failures = sorted(
        str(row["reference_media_id"])
        for row in provisional_rows
        if "automated_gbif_quality_gates_passed"
        not in row["admission_reasons"]
        or row["route_evidence_basis"] != "yoloe"
    )
    undeclared_provisional = sorted(
        str(row["reference_media_id"])
        for row in eligible_all
        if row["identity_evidence_basis"] == "gbif_provider_asserted"
        and not row["provisional_support"]
    )
    rejected_media_ids = {
        str(row["reference_media_id"])
        for row in review.outcomes.iter_rows(named=True)
        if row["review_status"] == "completed"
        and (
            row["resolved_verification_status"] != "verified"
            or not row["target_identity_verified"]
        )
    }
    ignored_human_rejections = sorted(
        str(row["reference_media_id"])
        for row in eligible_all
        if str(row["reference_media_id"]) in rejected_media_ids
    )
    licence_blockers: list[str] = []
    attribution_blockers: list[str] = []
    for assignment in included_assignments:
        media_id = str(assignment["reference_media_id"])
        candidate = indexes["media_candidates"].get(media_id)
        object_row = indexes["media_objects"].get(media_id)
        if candidate is None or object_row is None:
            continue
        observation = indexes["observations"].get(
            str(candidate["reference_observation_id"])
        )
        if (
            candidate["licence_policy_status"]
            not in policy.accepted_licence_policy_statuses
            or object_row["licence_policy_status"]
            not in policy.accepted_licence_policy_statuses
        ):
            licence_blockers.append(media_id)
        if observation is None or not _complete_attribution(candidate, observation):
            attribution_blockers.append(media_id)

    larval_route_conflicts = _route_group_conflicts(
        support_manifest,
        route="larval",
    )
    pinned_route_conflicts = _route_group_conflicts(
        support_manifest,
        route="pinned_specimen",
    )
    leakage = _split_leakage(
        included_assignments,
        acquisition_selections=acquisition_selections,
        indexes=indexes,
    )
    competitor_shortfalls = [item for item in competitor_results if item["shortfall"]]
    geographic_shortfalls = [item for item in geographic_results if item["shortfall"]]
    target_shortfalls = [item for item in target_results if item["shortfall"]]
    undocumented_competitor = [
        item
        for item in competitor_shortfalls
        if _requirement_key(item["requirement"]) not in accepted_shortfalls
    ]
    undocumented_geographic = [
        item
        for item in geographic_shortfalls
        if _requirement_key(item["requirement"]) not in accepted_shortfalls
    ]
    candidate_keys = candidate_context["candidate_keys"]
    assert isinstance(candidate_keys, list)
    competitor_required = any(
        item.accepted_taxon_key != policy.target_accepted_taxon_key
        for item in policy.requirements
    )
    eligible_species = {str(row["accepted_taxon_key"]) for row in eligible}
    model_inputs_available = bool(eligible) and (
        not competitor_required
        or bool(eligible_species - {policy.target_accepted_taxon_key})
    )

    evidence = {
        "reference_bank_fingerprint": bank_fingerprint,
        "support_manifest_fingerprint": support_fingerprint,
        "summary_fingerprint": summary_fingerprint,
        "split_assignments_fingerprint": split_fingerprint,
    }
    checks = [
        _check(
            "artifact_integrity",
            passed=not all_structural_issues,
            observed=len(all_structural_issues),
            required=0,
            evidence={**evidence, "issues": all_structural_issues},
        ),
        _requirement_check(
            "target_adult_minimum",
            target_results,
            accepted_shortfalls=set(),
            evidence=evidence,
        ),
        _requirement_check(
            "competitor_minima",
            competitor_results,
            accepted_shortfalls=accepted_shortfalls,
            evidence=evidence,
        ),
        _requirement_check(
            "geographic_cluster_coverage",
            geographic_results,
            accepted_shortfalls=accepted_shortfalls,
            evidence=evidence,
        ),
        _check(
            "larval_route_separation",
            passed=not larval_route_conflicts,
            observed=len(larval_route_conflicts),
            required=0,
            affected_routes=["larval"],
            evidence={**evidence, "conflicting_groups": larval_route_conflicts},
        ),
        _check(
            "pinned_specimen_separation",
            passed=not pinned_route_conflicts,
            observed=len(pinned_route_conflicts),
            required=0,
            affected_routes=["pinned_specimen"],
            evidence={**evidence, "conflicting_groups": pinned_route_conflicts},
        ),
        _check(
            "support_admission_policy_satisfied",
            passed=not admission_failures,
            observed=len(admission_failures),
            required=0,
            evidence={**evidence, "media_ids": admission_failures},
        ),
        _check(
            "provider_assertion_integrity",
            passed=not provider_integrity_failures,
            observed=len(provider_integrity_failures),
            required=0,
            evidence={**evidence, "media_ids": provider_integrity_failures},
        ),
        _check(
            "automated_reference_qa_passed",
            passed=not automated_qa_failures,
            observed=len(automated_qa_failures),
            required=0,
            evidence={**evidence, "media_ids": automated_qa_failures},
        ),
        _check(
            "reference_routes_separated",
            passed=not larval_route_conflicts and not pinned_route_conflicts,
            observed=len(
                set(larval_route_conflicts) | set(pinned_route_conflicts)
            ),
            required=0,
            evidence={
                **evidence,
                "conflicting_groups": sorted(
                    set(larval_route_conflicts) | set(pinned_route_conflicts)
                ),
            },
        ),
        _check(
            "provisional_support_declared",
            passed=not undeclared_provisional,
            observed=len(undeclared_provisional),
            required=0,
            evidence={**evidence, "media_ids": undeclared_provisional},
        ),
        _check(
            "statistical_audit_plan_available",
            passed=(
                admission_policy.mode != DEFAULT_REFERENCE_ADMISSION_MODE
                or (
                    admission_policy.require_statistical_audit
                    and bool(admission_policy.audit_policy_version)
                    and all(
                        bool(row["statistical_audit_required"])
                        for row in provisional_rows
                    )
                )
            ),
            observed=sum(
                bool(row["statistical_audit_required"])
                for row in provisional_rows
            ),
            required=len(provisional_rows),
            evidence={
                **evidence,
                "audit_policy_version": admission_policy.audit_policy_version,
            },
        ),
        _check(
            "human_rejections_respected",
            passed=not ignored_human_rejections,
            observed=len(ignored_human_rejections),
            required=0,
            evidence={**evidence, "media_ids": ignored_human_rejections},
        ),
        *(
            [
                _check(
                    "strict_support_only",
                    passed=not provisional_rows and not unverified_media,
                    observed=len(provisional_rows) + len(unverified_media),
                    required=0,
                    evidence={
                        **evidence,
                        "provisional_media_ids": sorted(
                            str(row["reference_media_id"])
                            for row in provisional_rows
                        ),
                        "unverified_media_ids": unverified_media,
                    },
                    pending=bool(unverified_media)
                    and set(unverified_media) <= pending_review_media,
                )
            ]
            if admission_policy.mode == "human_verified_strict"
            else []
        ),
        _check(
            "duplicate_groups_resolved",
            passed=not unresolved_relationships,
            observed=len(unresolved_relationships),
            required=0,
            evidence={
                **evidence,
                "relationship_ids": sorted(
                    str(row["duplicate_relationship_id"])
                    for row in unresolved_relationships
                ),
            },
            pending=bool(unresolved_relationships),
        ),
        _check(
            "licences_accepted",
            passed=not licence_blockers,
            observed=len(set(licence_blockers)),
            required=0,
            evidence={**evidence, "media_ids": sorted(set(licence_blockers))},
        ),
        _check(
            "source_attribution_complete",
            passed=not attribution_blockers,
            observed=len(set(attribution_blockers)),
            required=0,
            evidence={**evidence, "media_ids": sorted(set(attribution_blockers))},
        ),
        _check(
            "split_group_separation",
            passed=not leakage,
            observed=len(leakage),
            required=0,
            evidence={**evidence, "leakage": leakage},
        ),
        _check(
            "model_building_inputs_available",
            passed=model_inputs_available,
            observed=len(eligible),
            required=1,
            affected_species=sorted(eligible_species),
            evidence={
                **evidence,
                "model_input_fingerprint": model_identity.fingerprint,
                "candidate_species_count": len(candidate_keys),
                "eligible_species_count": len(eligible_species),
            },
        ),
    ]
    counts = {
        "support_manifest_rows": support_manifest.height,
        "eligible_support_rows": len(eligible),
        "provisional_support_count": sum(
            bool(row["provisional_support"]) for row in eligible_all
        ),
        "human_verified_support_count": sum(
            bool(row["human_verified_identity"]) for row in eligible_all
        ),
        "target_minimum_shortfall_count": len(target_shortfalls),
        "competitor_minimum_shortfall_count": len(competitor_shortfalls),
        "geographic_coverage_shortfall_count": len(geographic_shortfalls),
        "undocumented_competitor_shortfall_count": len(undocumented_competitor),
        "undocumented_geographic_shortfall_count": len(undocumented_geographic),
        "documented_shortfall_count": len(accepted_shortfalls),
        "pending_review_count": len(pending_review),
        "pending_target_review_count": len(pending_target_review),
        "unresolved_duplicate_count": len(unresolved_relationships),
        "licence_blocker_count": len(set(licence_blockers)),
        "attribution_blocker_count": len(set(attribution_blockers)),
        "unverified_support_count": len(unverified_media),
        "route_separation_conflict_count": len(
            set(larval_route_conflicts) | set(pinned_route_conflicts)
        ),
        "split_leakage_count": len(leakage),
        "structural_issue_count": len(all_structural_issues),
    }
    documented_payload = [
        {
            **_dataclass_payload(shortfall),
            "requirement_minimum_count": requirement.minimum_count,
            "observed_count": next(
                int(item["observed"])
                for item in requirement_results
                if item["requirement"] == requirement
            ),
        }
        for key, (shortfall, requirement) in sorted(
            accepted_shortfalls.items(),
            key=lambda item: item[0],
        )
    ]
    return checks, counts, documented_payload


def _readiness_status(
    *,
    checks: Sequence[Mapping[str, object]],
    counts: Mapping[str, int],
    documented_shortfalls: Sequence[Mapping[str, object]],
) -> str:
    by_id = {str(item["check_id"]): item for item in checks}
    invalid_checks = (
        "artifact_integrity",
        "larval_route_separation",
        "pinned_specimen_separation",
        "support_admission_policy_satisfied",
        "provider_assertion_integrity",
        "automated_reference_qa_passed",
        "reference_routes_separated",
        "provisional_support_declared",
        "statistical_audit_plan_available",
        "human_rejections_respected",
        "source_attribution_complete",
        "split_group_separation",
    )
    if any(by_id[check_id]["status"] == "failed" for check_id in invalid_checks):
        return "invalid"
    if by_id["licences_accepted"]["status"] == "failed":
        return "blocked_licence"
    target_failed = by_id["target_adult_minimum"]["status"] == "failed"
    if target_failed:
        if "strict_support_only" in by_id and (
            counts["pending_target_review_count"]
            or counts["unresolved_duplicate_count"]
        ):
            return "awaiting_manual_review"
        return "blocked_missing_target_support"
    if "strict_support_only" in by_id and (
        counts["pending_review_count"]
        or counts["unresolved_duplicate_count"]
        or by_id["strict_support_only"]["status"] == "pending"
    ):
        return "awaiting_manual_review"
    if (
        counts["undocumented_competitor_shortfall_count"]
        or counts["undocumented_geographic_shortfall_count"]
    ):
        return "invalid"
    if by_id["model_building_inputs_available"]["status"] == "failed":
        return "invalid"
    if counts["provisional_support_count"] and all(
        item["status"] in {"passed", "warning"} for item in checks
    ):
        return "ready_provisional"
    if documented_shortfalls:
        return "ready_with_documented_shortfalls"
    if any(item["status"] != "passed" for item in checks):
        return "invalid"
    return "ready"


def _readiness_capabilities(status: str) -> dict[str, bool]:
    if status == "ready_provisional":
        return {
            "permits_reference_embedding": True,
            "permits_prototype_creation": True,
            "permits_provisional_scoring": True,
            "permits_calibrated_scoring": False,
            "permits_scientific_release": False,
            "requires_downstream_flickr_review": True,
        }
    if status in STRICT_PERMITTING_READINESS_STATUSES:
        return {
            "permits_reference_embedding": True,
            "permits_prototype_creation": True,
            "permits_provisional_scoring": True,
            "permits_calibrated_scoring": True,
            "permits_scientific_release": True,
            "requires_downstream_flickr_review": True,
        }
    return {
        "permits_reference_embedding": False,
        "permits_prototype_creation": False,
        "permits_provisional_scoring": False,
        "permits_calibrated_scoring": False,
        "permits_scientific_release": False,
        "requires_downstream_flickr_review": True,
    }


def _validate_documented_shortfalls(
    *,
    policy: ReferenceBankReadinessPolicy,
    requirement_results: Sequence[Mapping[str, object]],
    acquisition_plan: pl.DataFrame,
) -> tuple[
    dict[
        tuple[str, str | None, str],
        tuple[DocumentedReferenceShortfall, ReferenceBankRequirement],
    ],
    list[str],
]:
    results = {
        _requirement_key(item["requirement"]): item
        for item in requirement_results
    }
    accepted: dict[
        tuple[str, str | None, str],
        tuple[DocumentedReferenceShortfall, ReferenceBankRequirement],
    ] = {}
    issues: list[str] = []
    for shortfall in policy.documented_shortfalls:
        key = _shortfall_key(shortfall)
        result = results.get(key)
        if result is None:
            issues.append(f"documented shortfall has no requirement: {shortfall.shortfall_id}")
            continue
        requirement = result["requirement"]
        assert isinstance(requirement, ReferenceBankRequirement)
        observed = int(result["observed"])
        if int(result["shortfall"]) == 0:
            issues.append(f"documented shortfall is stale: {shortfall.shortfall_id}")
            continue
        if observed < shortfall.approved_minimum_count:
            issues.append(
                f"documented shortfall approved minimum is not met: {shortfall.shortfall_id}"
            )
            continue
        matching_plan_fingerprints = _requirement_plan_fingerprints(
            requirement,
            acquisition_plan,
        )
        if shortfall.plan_configuration_fingerprint not in matching_plan_fingerprints:
            issues.append(
                f"documented shortfall plan fingerprint is stale: {shortfall.shortfall_id}"
            )
            continue
        accepted[key] = (shortfall, requirement)
    return accepted, issues


def _requirement_plan_fingerprints(
    requirement: ReferenceBankRequirement,
    acquisition_plan: pl.DataFrame,
) -> set[str]:
    life_stage, visual_domain = reference_route_dimensions(requirement.route)
    rows = acquisition_plan.filter(
        (pl.col("candidate_accepted_taxon_key") == requirement.accepted_taxon_key)
        & (pl.col("life_stage") == life_stage)
        & (pl.col("visual_domain") == visual_domain)
    )
    if requirement.geo_cluster_id is not None:
        rows = rows.filter(pl.col("geo_cluster_id") == requirement.geo_cluster_id)
    return set(str(value) for value in rows["plan_configuration_fingerprint"])


def _check(
    check_id: str,
    *,
    passed: bool,
    observed: object,
    required: object,
    evidence: Mapping[str, object],
    affected_species: Sequence[str] = (),
    affected_clusters: Sequence[str] = (),
    affected_routes: Sequence[str] = (),
    pending: bool = False,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else ("pending" if pending else "failed"),
        "observed": _canonical(observed),
        "required": _canonical(required),
        "affected_species": sorted(set(str(value) for value in affected_species)),
        "affected_clusters": sorted(set(str(value) for value in affected_clusters)),
        "affected_routes": sorted(set(str(value) for value in affected_routes)),
        "evidence": _canonical(dict(evidence)),
    }


def _requirement_check(
    check_id: str,
    results: Sequence[Mapping[str, object]],
    *,
    accepted_shortfalls: Mapping[tuple[str, str | None, str], object] | set[object],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    failures = [item for item in results if int(item["shortfall"]) > 0]
    unapproved = [
        item
        for item in failures
        if _requirement_key(item["requirement"]) not in accepted_shortfalls
    ]
    requirement_payload = []
    for item in results:
        requirement = item["requirement"]
        assert isinstance(requirement, ReferenceBankRequirement)
        requirement_payload.append(
            {
                **_dataclass_payload(requirement),
                "observed_count": int(item["observed"]),
                "shortfall_count": int(item["shortfall"]),
                "documented": _requirement_key(requirement) in accepted_shortfalls,
            }
        )
    if not failures:
        status = "passed"
    elif not unapproved:
        status = "warning"
    else:
        status = "failed"
    return {
        "check_id": check_id,
        "status": status,
        "observed": requirement_payload,
        "required": [
            _dataclass_payload(item["requirement"]) for item in results
        ],
        "affected_species": sorted(
            {
                item["requirement"].accepted_taxon_key
                for item in failures
            }
        ),
        "affected_clusters": sorted(
            {
                item["requirement"].geo_cluster_id
                for item in failures
                if item["requirement"].geo_cluster_id is not None
            }
        ),
        "affected_routes": sorted(
            {item["requirement"].route for item in failures}
        ),
        "evidence": _canonical(dict(evidence)),
    }


def _route_group_conflicts(
    support_manifest: pl.DataFrame,
    *,
    route: str,
) -> list[str]:
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in support_manifest.iter_rows(named=True):
        if not row["support_split"]:
            continue
        groups[str(row["duplicate_group_id"])].add(str(row["route"]))
        hashes[str(row["image_sha256"])].add(str(row["route"]))
    conflicts = {
        f"duplicate_group:{key}"
        for key, routes in groups.items()
        if route in routes and len(routes) > 1
    }
    conflicts.update(
        f"image_sha256:{key}"
        for key, routes in hashes.items()
        if route in routes and len(routes) > 1
    )
    return sorted(conflicts)


def _split_leakage(
    included_assignments: Sequence[Mapping[str, object]],
    *,
    acquisition_selections: pl.DataFrame,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    background_groups_by_media: dict[str, set[str]] = defaultdict(set)
    for selection in acquisition_selections.iter_rows(named=True):
        background_group_id = selection["background_group_id"]
        if background_group_id is not None:
            background_groups_by_media[str(selection["reference_media_id"])].add(
                str(background_group_id)
            )
    for assignment in included_assignments:
        media_id = str(assignment["reference_media_id"])
        split = str(assignment["support_split"])
        object_row = indexes["media_objects"].get(media_id)
        candidate = indexes["media_candidates"].get(media_id)
        if object_row is None or candidate is None:
            continue
        observation_id = str(candidate["reference_observation_id"])
        observation = indexes["observations"].get(observation_id)
        group_values = {
            "reference_media_id": media_id,
            "image_sha256": str(object_row["sha256"]),
            "duplicate_group_id": str(object_row["duplicate_group_id"]),
            "reference_observation_id": observation_id,
            "provider_media_id": str(candidate["provider_media_id"]),
        }
        if observation is not None:
            group_values.update(
                {
                    "source_observation": (
                        f"{observation['source']}:{observation['source_observation_id']}"
                    ),
                    "observer_id": str(observation["observer_id"] or "unknown"),
                }
            )
        mirror_ids = sorted(
            {media_id, *(str(value) for value in object_row["provider_mirror_ids"])}
        )
        if len(mirror_ids) > 1:
            group_values["provider_mirror_group"] = "|".join(mirror_ids)
        for group_type, group_value in group_values.items():
            if group_value and group_value != "unknown":
                grouped[(group_type, group_value)].add(split)
        for background_group_id in background_groups_by_media.get(media_id, set()):
            grouped[("background_group_id", background_group_id)].add(split)
    return [
        {
            "group_type": group_type,
            "group_value": group_value,
            "splits": sorted(splits),
        }
        for (group_type, group_value), splits in sorted(grouped.items())
        if len(splits) > 1
    ]


def _support_split_leakage(
    assigned_support: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in assigned_support:
        split = str(row["support_split"])
        group_values = {
            "reference_media_id": str(row["reference_media_id"]),
            "image_sha256": str(row["image_sha256"]),
            "duplicate_group_id": str(row["duplicate_group_id"]),
            "reference_observation_id": str(row["reference_observation_id"]),
            "provider_media_id": str(row["provider_media_id"]),
            "source_observation": (
                f"{row['source']}:{row['source_observation_id']}"
            ),
            "observer_id": str(row["observer_id"] or "unknown"),
        }
        for group_type, group_value in group_values.items():
            if group_value and group_value != "unknown":
                grouped[(group_type, group_value)].add(split)
    return [
        {
            "group_type": group_type,
            "group_value": group_value,
            "splits": sorted(splits),
        }
        for (group_type, group_value), splits in sorted(grouped.items())
        if len(splits) > 1
    ]


def _validate_readiness_payload(
    payload: Mapping[str, object],
    *,
    published: bool,
) -> None:
    required_keys = {
        "schema_version",
        "status",
        "permits_vision",
        "permits_reference_embedding",
        "permits_prototype_creation",
        "permits_provisional_scoring",
        "permits_calibrated_scoring",
        "permits_scientific_release",
        "requires_downstream_flickr_review",
        "reference_admission_mode",
        "admission_policy_fingerprint",
        "provisional_support_count",
        "human_verified_support_count",
        "statistical_audit_required",
        "registry_version",
        "reference_bank_version",
        "policy_version",
        "policy_fingerprint",
        "policy",
        "target_accepted_taxon_key",
        "candidate_set_ids",
        "candidate_set_fingerprints",
        "created_at",
        "git_sha",
        "bank_fingerprint",
        "support_manifest_fingerprint",
        "summary_fingerprint",
        "split_assignments_fingerprint",
        "model_input_identity",
        "model_input_fingerprint",
        "checks",
        "counts",
        "documented_shortfalls",
        "inputs",
        "artifacts",
    }
    optional_keys = {"publication"}
    _exact_mapping_keys(
        payload,
        required=required_keys,
        optional=optional_keys,
        artifact="reference bank readiness",
    )
    if payload["schema_version"] != REFERENCE_BANK_READINESS_SCHEMA_VERSION:
        raise ValueError("unsupported reference bank readiness schema version")
    status = payload["status"]
    if status not in READINESS_STATUSES:
        raise ValueError("reference bank readiness status is invalid")
    expected_capabilities = _readiness_capabilities(str(status))
    if any(
        payload[field] is not expected
        for field, expected in expected_capabilities.items()
    ):
        raise ValueError(
            "reference bank readiness permit flag or capabilities are inconsistent"
        )
    if payload["permits_vision"] is not payload["permits_reference_embedding"]:
        raise ValueError("reference bank readiness permit flag is inconsistent")
    admission_mode = str(payload["reference_admission_mode"])
    if admission_mode not in REFERENCE_ADMISSION_MODES:
        raise ValueError("reference bank readiness admission mode is invalid")
    _fingerprint(
        payload["admission_policy_fingerprint"],
        field="admission_policy_fingerprint",
    )
    provisional_support_count = _nonnegative_int(
        payload["provisional_support_count"],
        field="provisional_support_count",
    )
    human_verified_support_count = _nonnegative_int(
        payload["human_verified_support_count"],
        field="human_verified_support_count",
    )
    if not isinstance(payload["statistical_audit_required"], bool):
        raise ValueError("statistical_audit_required must be Boolean")
    if status == "ready_provisional" and (
        admission_mode != "adaptive_gbif_fast_start"
        or provisional_support_count == 0
        or payload["statistical_audit_required"] is not True
    ):
        raise ValueError("provisional readiness lacks adaptive support or audit policy")
    if payload["permits_scientific_release"] and (
        provisional_support_count != 0 or human_verified_support_count == 0
    ):
        raise ValueError("scientific release cannot rely on provisional support")
    for field_name in (
        "registry_version",
        "reference_bank_version",
        "policy_version",
        "target_accepted_taxon_key",
    ):
        _required_text(payload[field_name], field=field_name)
    for field_name in (
        "policy_fingerprint",
        "bank_fingerprint",
        "support_manifest_fingerprint",
        "summary_fingerprint",
        "split_assignments_fingerprint",
        "model_input_fingerprint",
    ):
        _fingerprint(payload[field_name], field=field_name)
    policy_mapping = payload["policy"]
    if not isinstance(policy_mapping, Mapping):
        raise ValueError("readiness policy must be an object")
    policy = ReferenceBankReadinessPolicy.from_mapping(policy_mapping)
    if (
        payload["policy_fingerprint"] != policy.fingerprint
        or payload["policy_version"] != policy.policy_version
        or payload["target_accepted_taxon_key"]
        != policy.target_accepted_taxon_key
    ):
        raise ValueError(
            "reference bank fingerprint or policy identity is inconsistent"
        )
    created_at = payload["created_at"]
    if not isinstance(created_at, str):
        raise ValueError("readiness created_at must be an ISO timestamp")
    _parse_datetime(created_at, field="created_at")
    git_sha = payload["git_sha"]
    if git_sha is not None and (
        not isinstance(git_sha, str) or not re.fullmatch(r"[0-9a-f]{7,64}", git_sha)
    ):
        raise ValueError("readiness git SHA is invalid")
    _sorted_unique_string_list(payload["candidate_set_ids"], field="candidate_set_ids")
    candidate_fingerprints = _sorted_unique_string_list(
        payload["candidate_set_fingerprints"],
        field="candidate_set_fingerprints",
    )
    if not candidate_fingerprints or any(
        _SHA256_PATTERN.fullmatch(value) is None for value in candidate_fingerprints
    ):
        raise ValueError("readiness candidate set fingerprints are invalid")
    model_mapping = payload["model_input_identity"]
    if not isinstance(model_mapping, Mapping):
        raise ValueError("readiness model identity must be an object")
    model_identity = ReferenceModelInputIdentity.from_mapping(model_mapping)
    if payload["model_input_fingerprint"] != model_identity.fingerprint:
        raise ValueError("readiness model input fingerprint is invalid")
    checks = payload["checks"]
    if not isinstance(checks, list) or [
        item.get("check_id") if isinstance(item, Mapping) else None for item in checks
    ] != list(_required_check_ids(admission_mode)):
        raise ValueError("readiness checks are missing, duplicated, or out of order")
    for check in checks:
        assert isinstance(check, Mapping)
        _validate_readiness_check(check, payload=payload)
    counts = payload["counts"]
    required_count_fields = {
        "support_manifest_rows",
        "eligible_support_rows",
        "provisional_support_count",
        "human_verified_support_count",
        "target_minimum_shortfall_count",
        "competitor_minimum_shortfall_count",
        "geographic_coverage_shortfall_count",
        "undocumented_competitor_shortfall_count",
        "undocumented_geographic_shortfall_count",
        "documented_shortfall_count",
        "pending_review_count",
        "pending_target_review_count",
        "unresolved_duplicate_count",
        "licence_blocker_count",
        "attribution_blocker_count",
        "unverified_support_count",
        "route_separation_conflict_count",
        "split_leakage_count",
        "structural_issue_count",
    }
    if not isinstance(counts, Mapping) or set(counts) != required_count_fields:
        raise ValueError("readiness counts shape is invalid")
    for field_name, value in counts.items():
        _nonnegative_int(value, field=str(field_name))
    if (
        counts["provisional_support_count"] != provisional_support_count
        or counts["human_verified_support_count"] != human_verified_support_count
    ):
        raise ValueError("readiness support evidence counts are inconsistent")
    documented = payload["documented_shortfalls"]
    if not isinstance(documented, list):
        raise ValueError("documented_shortfalls must be a list")
    documented_ids: list[str] = []
    policy_shortfalls = {
        item.shortfall_id: _dataclass_payload(item)
        for item in policy.documented_shortfalls
    }
    for item in documented:
        if not isinstance(item, Mapping):
            raise ValueError("documented shortfall entry must be an object")
        documented_ids.append(_required_text(item.get("shortfall_id"), field="shortfall_id"))
        policy_shortfall = policy_shortfalls.get(str(item["shortfall_id"]))
        if policy_shortfall is None or any(
            item.get(field_name) != value
            for field_name, value in policy_shortfall.items()
        ):
            raise ValueError("documented shortfall conflicts with the versioned policy")
        for field_name in (
            "requirement_minimum_count",
            "approved_minimum_count",
            "observed_count",
        ):
            _nonnegative_int(item.get(field_name), field=field_name)
        if int(item["observed_count"]) < int(item["approved_minimum_count"]):
            raise ValueError("documented shortfall approved minimum is not met")
        if int(item["approved_minimum_count"]) >= int(
            item["requirement_minimum_count"]
        ):
            raise ValueError("documented shortfall does not lower a requirement")
        if item.get("accepted_taxon_key") == payload["target_accepted_taxon_key"]:
            raise ValueError("target support cannot use a documented shortfall")
        _fingerprint(
            item.get("plan_configuration_fingerprint"),
            field="plan_configuration_fingerprint",
        )
    if documented_ids != sorted(set(documented_ids)):
        raise ValueError("documented shortfalls must be sorted and unique")
    if int(counts["documented_shortfall_count"]) != len(documented):
        raise ValueError("documented shortfall count is inconsistent")
    if status in PERMITTING_READINESS_STATUSES and set(documented_ids) != set(
        policy_shortfalls
    ):
        raise ValueError("permitting readiness omits a policy shortfall approval")
    inputs = payload["inputs"]
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("readiness input fingerprints are missing")
    for field_name, value in inputs.items():
        _fingerprint(value, field=f"inputs.{field_name}")
    expected_bank_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_BANK_READINESS_SCHEMA_VERSION,
            "reference_bank_version": payload["reference_bank_version"],
            "registry_version": payload["registry_version"],
            "target_accepted_taxon_key": payload["target_accepted_taxon_key"],
            "policy_fingerprint": payload["policy_fingerprint"],
            "reference_admission_mode": payload["reference_admission_mode"],
            "admission_policy_fingerprint": payload[
                "admission_policy_fingerprint"
            ],
            "model_input_fingerprint": payload["model_input_fingerprint"],
            "candidate_set_ids": payload["candidate_set_ids"],
            "candidate_set_fingerprints": payload["candidate_set_fingerprints"],
            "inputs": dict(inputs),
        }
    )
    if payload["bank_fingerprint"] != expected_bank_fingerprint:
        raise ValueError("reference bank fingerprint is invalid")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "support_manifest",
        "summary",
    }:
        raise ValueError("readiness artifact map is invalid")
    expected_artifacts = {
        "support_manifest": (
            REFERENCE_SUPPORT_MANIFEST_FILE,
            payload["support_manifest_fingerprint"],
        ),
        "summary": (REFERENCE_BANK_SUMMARY_FILE, payload["summary_fingerprint"]),
    }
    for name, (filename, fingerprint) in expected_artifacts.items():
        artifact = artifacts[name]
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "file",
            "sha256",
            "semantic_fingerprint",
        }:
            raise ValueError(f"readiness {name} artifact record is invalid")
        if artifact["file"] != filename or artifact["semantic_fingerprint"] != fingerprint:
            raise ValueError(f"readiness {name} artifact identity is invalid")
        sha = artifact["sha256"]
        if published:
            _fingerprint(sha, field=f"artifacts.{name}.sha256")
        elif sha is not None:
            _fingerprint(sha, field=f"artifacts.{name}.sha256")
    if published and "publication" not in payload:
        raise ValueError("published readiness manifest lacks publication audit")
    if "publication" in payload:
        _validate_publication(payload["publication"])
    _validate_status_consistency(
        status=str(status),
        checks=checks,
        counts=counts,
        documented=documented,
    )


def _validate_readiness_check(
    check: Mapping[str, object],
    *,
    payload: Mapping[str, object],
) -> None:
    if set(check) != {
        "check_id",
        "status",
        "observed",
        "required",
        "affected_species",
        "affected_clusters",
        "affected_routes",
        "evidence",
    }:
        raise ValueError("readiness check shape is invalid")
    if check["status"] not in READINESS_CHECK_STATUSES:
        raise ValueError("readiness check status is invalid")
    _sorted_unique_string_list(check["affected_species"], field="affected_species")
    _sorted_unique_string_list(check["affected_clusters"], field="affected_clusters")
    affected_routes = _sorted_unique_string_list(
        check["affected_routes"],
        field="affected_routes",
    )
    if set(affected_routes) - REFERENCE_ROUTES:
        raise ValueError("readiness check contains an unsupported route")
    evidence = check["evidence"]
    if not isinstance(evidence, Mapping):
        raise ValueError("readiness check evidence must be an object")
    expected = {
        "reference_bank_fingerprint": payload["bank_fingerprint"],
        "support_manifest_fingerprint": payload["support_manifest_fingerprint"],
        "summary_fingerprint": payload["summary_fingerprint"],
        "split_assignments_fingerprint": payload["split_assignments_fingerprint"],
    }
    if any(evidence.get(field_name) != value for field_name, value in expected.items()):
        raise ValueError("readiness check evidence fingerprint mismatch")


def _validate_status_consistency(
    *,
    status: str,
    checks: Sequence[Mapping[str, object]],
    counts: Mapping[str, object],
    documented: Sequence[Mapping[str, object]],
) -> None:
    check_statuses = {str(item["check_id"]): str(item["status"]) for item in checks}
    if status == "ready":
        if documented or any(value != "passed" for value in check_statuses.values()):
            raise ValueError("ready status contains a nonpassing readiness check")
    elif status == "ready_provisional":
        nonpassing = {
            check_id: value
            for check_id, value in check_statuses.items()
            if value != "passed"
        }
        if any(value in {"failed", "pending"} for value in nonpassing.values()) or set(
            nonpassing
        ) - {"competitor_minima", "geographic_cluster_coverage"}:
            raise ValueError("provisional readiness contains an unsafe check")
        if int(counts["provisional_support_count"]) == 0:
            raise ValueError("provisional readiness has no provisional support")
    elif status == "ready_with_documented_shortfalls":
        if not documented:
            raise ValueError("documented-shortfall readiness has no approvals")
        warning_ids = {
            check_id for check_id, value in check_statuses.items() if value == "warning"
        }
        if not warning_ids or not warning_ids <= {
            "competitor_minima",
            "geographic_cluster_coverage",
        }:
            raise ValueError("documented shortfalls apply to an unsupported check")
        if any(value in {"failed", "pending"} for value in check_statuses.values()):
            raise ValueError("permitting readiness contains a failed or pending check")
    elif status == "blocked_licence" and check_statuses["licences_accepted"] != "failed":
        raise ValueError("blocked_licence status lacks a licence failure")
    elif status == "blocked_missing_target_support" and (
        check_statuses["target_adult_minimum"] != "failed"
        or int(counts["pending_target_review_count"]) != 0
        or int(counts["unresolved_duplicate_count"]) != 0
    ):
        raise ValueError("blocked target status is inconsistent")
    elif status == "awaiting_manual_review":
        strict_status = check_statuses.get("strict_support_only")
        if not (
            int(counts["pending_review_count"])
            or int(counts["unresolved_duplicate_count"])
            or strict_status == "pending"
        ):
            raise ValueError("manual-review status has no pending work")
    elif status == "invalid" and not any(
        value == "failed" for value in check_statuses.values()
    ):
        raise ValueError("invalid status has no failed readiness check")


def _validate_publication(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "command",
        "run_id",
        "pid",
        "started_at",
        "ended_at",
        "network_requests",
    }:
        raise ValueError("readiness publication audit is invalid")
    if value["command"] != "references.validate_readiness":
        raise ValueError("readiness publication command is invalid")
    _required_text(value["run_id"], field="run_id")
    _positive_int(value["pid"], field="pid")
    _parse_datetime(str(value["started_at"]), field="started_at")
    _parse_datetime(str(value["ended_at"]), field="ended_at")
    if value["network_requests"] != 0:
        raise ValueError("readiness publication must not use the network")


def _validate_cross_artifact_readiness(
    payload: Mapping[str, object],
    *,
    support: pl.DataFrame,
    summary: pl.DataFrame,
) -> None:
    bank_fingerprint = payload["bank_fingerprint"]
    split_fingerprint = payload["split_assignments_fingerprint"]
    support_fingerprint = payload["support_manifest_fingerprint"]
    registry_version = payload["registry_version"]
    reference_bank_version = payload["reference_bank_version"]
    if not support.is_empty():
        expected_support = {
            "reference_bank_fingerprint": bank_fingerprint,
            "registry_version": registry_version,
            "reference_bank_version": reference_bank_version,
            "reference_admission_mode": payload["reference_admission_mode"],
            "reference_admission_policy_fingerprint": payload[
                "admission_policy_fingerprint"
            ],
        }
        for field_name, expected in expected_support.items():
            if set(support[field_name]) != {expected}:
                raise ValueError(f"support manifest {field_name} mismatch")
    if not summary.is_empty():
        expected_summary = {
            "reference_bank_fingerprint": bank_fingerprint,
            "support_manifest_fingerprint": support_fingerprint,
            "split_assignments_fingerprint": split_fingerprint,
            "registry_version": registry_version,
            "reference_bank_version": reference_bank_version,
        }
        for field_name, expected in expected_summary.items():
            if set(summary[field_name]) != {expected}:
                raise ValueError(f"reference bank summary {field_name} mismatch")
    counts = payload["counts"]
    assert isinstance(counts, Mapping)
    eligible_all = support.filter(pl.col("support_eligible"))
    eligible = support.filter(
        pl.col("support_eligible") & (pl.col("support_split") == "support_train")
    )
    if counts["support_manifest_rows"] != support.height:
        raise ValueError("readiness support manifest row count mismatch")
    if counts["eligible_support_rows"] != eligible.height:
        raise ValueError("readiness eligible support row count mismatch")
    provisional_count = eligible_all.filter(pl.col("provisional_support")).height
    human_verified_count = eligible_all.filter(
        pl.col("human_verified_identity")
    ).height
    if (
        counts["provisional_support_count"] != provisional_count
        or payload["provisional_support_count"] != provisional_count
    ):
        raise ValueError("readiness provisional support count mismatch")
    if (
        counts["human_verified_support_count"] != human_verified_count
        or payload["human_verified_support_count"] != human_verified_count
    ):
        raise ValueError("readiness human-verified support count mismatch")
    checks = payload["checks"]
    assert isinstance(checks, list)
    check_by_id = {str(item["check_id"]): item for item in checks}
    policy_mapping = payload["policy"]
    assert isinstance(policy_mapping, Mapping)
    policy = ReferenceBankReadinessPolicy.from_mapping(policy_mapping)
    assigned_support = [
        row
        for row in support.iter_rows(named=True)
        if row["support_split"] is not None
    ]
    licence_blockers = sorted(
        str(row["reference_media_id"])
        for row in assigned_support
        if row["licence_policy_status"]
        not in policy.accepted_licence_policy_statuses
    )
    if any(
        row["support_eligible"]
        and row["licence_policy_status"]
        not in policy.accepted_licence_policy_statuses
        for row in assigned_support
    ):
        raise ValueError("eligible support manifest row has an unaccepted licence")
    if counts["licence_blocker_count"] != len(licence_blockers):
        raise ValueError("readiness licence blocker count mismatch")
    licence_check = check_by_id["licences_accepted"]
    if licence_check["status"] != (
        "passed" if not licence_blockers else "failed"
    ) or licence_check["evidence"].get("media_ids") != licence_blockers:
        raise ValueError("readiness licence check conflicts with support manifest")

    attribution_blockers = sorted(
        str(row["reference_media_id"])
        for row in assigned_support
        if not _support_attribution_complete(row)
    )
    if counts["attribution_blocker_count"] != len(attribution_blockers):
        raise ValueError("readiness attribution blocker count mismatch")
    attribution_check = check_by_id["source_attribution_complete"]
    if attribution_check["status"] != (
        "passed" if not attribution_blockers else "failed"
    ) or attribution_check["evidence"].get("media_ids") != attribution_blockers:
        raise ValueError("readiness attribution check conflicts with support manifest")

    larval_conflicts = _route_group_conflicts(support, route="larval")
    pinned_conflicts = _route_group_conflicts(support, route="pinned_specimen")
    route_conflicts = sorted(set(larval_conflicts) | set(pinned_conflicts))
    if counts["route_separation_conflict_count"] != len(route_conflicts):
        raise ValueError("readiness route separation count mismatch")
    for check_id, conflicts in (
        ("larval_route_separation", larval_conflicts),
        ("pinned_specimen_separation", pinned_conflicts),
    ):
        check = check_by_id[check_id]
        if check["status"] != ("passed" if not conflicts else "failed") or (
            check["evidence"].get("conflicting_groups") != conflicts
        ):
            raise ValueError(f"{check_id} conflicts with support manifest")

    derived_leakage = _support_split_leakage(assigned_support)
    leakage_check = check_by_id["split_group_separation"]
    declared_leakage = leakage_check["evidence"].get("leakage")
    if not isinstance(declared_leakage, list) or counts[
        "split_leakage_count"
    ] != len(declared_leakage):
        raise ValueError("readiness split leakage evidence is inconsistent")
    if any(item not in declared_leakage for item in derived_leakage):
        raise ValueError("readiness split leakage omits support-manifest leakage")
    if leakage_check["status"] == "passed" and declared_leakage:
        raise ValueError("readiness split leakage check has concealed evidence")

    eligible_species = set(str(value) for value in eligible["accepted_taxon_key"])
    competitor_required = any(
        item.accepted_taxon_key != policy.target_accepted_taxon_key
        for item in policy.requirements
    )
    model_inputs_available = bool(eligible_species) and (
        not competitor_required
        or bool(eligible_species - {policy.target_accepted_taxon_key})
    )
    model_check = check_by_id["model_building_inputs_available"]
    if model_check["status"] != (
        "passed" if model_inputs_available else "failed"
    ):
        raise ValueError("model-building input check conflicts with support manifest")
    documented_keys = {
        (
            str(item["accepted_taxon_key"]),
            _optional_text(item.get("geo_cluster_id"), field="geo_cluster_id"),
            str(item["route"]),
        )
        for item in payload["documented_shortfalls"]
    }
    for check_id in (
        "target_adult_minimum",
        "competitor_minima",
        "geographic_cluster_coverage",
    ):
        check = check_by_id[check_id]
        observed_rows = check["observed"]
        if not isinstance(observed_rows, list):
            raise ValueError(f"{check_id} observed requirements must be a list")
        for item in observed_rows:
            if not isinstance(item, Mapping):
                raise ValueError(f"{check_id} contains an invalid requirement")
            requirement = ReferenceBankRequirement.from_mapping(
                {
                    key: item[key]
                    for key in (
                        "accepted_taxon_key",
                        "route",
                        "minimum_count",
                        "geo_cluster_id",
                    )
                    if key in item
                }
            )
            actual = sum(
                1
                for row in eligible.iter_rows(named=True)
                if _support_matches_requirement(row, requirement)
            )
            if item.get("observed_count") != actual:
                raise ValueError(f"{check_id} observed support count mismatch")
            if item.get("shortfall_count") != max(
                0,
                requirement.minimum_count - actual,
            ):
                raise ValueError(f"{check_id} shortfall count mismatch")
            documented = _requirement_key(requirement) in documented_keys
            if item.get("documented") is not documented:
                raise ValueError(f"{check_id} documented-shortfall binding mismatch")


def make_reference_split_assignment_fingerprint(
    *,
    reference_media_id: str,
    split_version: str,
    support_split: str,
    included: bool,
    exclusion_reason: str | None,
    assigned_by: str,
    assigned_at: datetime,
) -> str:
    return _assignment_fingerprint(
        reference_media_id=reference_media_id,
        split_version=split_version,
        support_split=support_split,
        included=included,
        exclusion_reason=exclusion_reason,
        assigned_by=assigned_by,
        assigned_at=assigned_at,
    )


def _assignment_fingerprint(
    *,
    reference_media_id: str,
    split_version: str,
    support_split: str,
    included: bool,
    exclusion_reason: str | None,
    assigned_by: str,
    assigned_at: datetime,
) -> str:
    return _sha256_json(
        {
            "schema_version": REFERENCE_BANK_SPLIT_ASSIGNMENTS_SCHEMA_VERSION,
            "reference_media_id": reference_media_id,
            "split_version": split_version,
            "support_split": support_split,
            "included": included,
            "exclusion_reason": exclusion_reason,
            "assigned_by": assigned_by,
            "assigned_at": assigned_at,
        }
    )


def _support_row_fingerprint(row: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        {
            "projection_version": "reference-support-row-semantic-v3",
            "row": _semantic_mapping_projection(
                row,
                fields=_SUPPORT_ROW_SEMANTIC_FIELDS,
                artifact="reference support row",
            ),
        }
    )


def _legacy_support_row_fingerprint_v2(row: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        {
            "projection_version": "reference-support-row-semantic-v2",
            "row": _semantic_mapping_projection(
                row,
                fields=_LEGACY_SUPPORT_ROW_SEMANTIC_FIELDS,
                artifact="legacy reference support row v2",
            ),
        }
    )


def _summary_row_fingerprint(row: Mapping[str, object]) -> str:
    return _sha256_json(
        {
            key: value
            for key, value in row.items()
            if key != "summary_row_fingerprint"
        }
    )


def _reference_route(*, life_stage: str, visual_domain: str) -> str | None:
    stage = str(life_stage or "").casefold()
    domain = str(visual_domain or "").casefold()
    if domain == "pinned_specimen":
        return "pinned_specimen"
    if domain != "live_field":
        return None
    return {
        "adult": "adult_field",
        "larva": "larval",
        "pupa": "pupal",
        "egg": "egg",
    }.get(stage)


def reference_route_dimensions(route: str) -> tuple[str, str]:
    try:
        return {
            "adult_field": ("adult", "live_field"),
            "larval": ("larva", "live_field"),
            "pupal": ("pupa", "live_field"),
            "egg": ("egg", "live_field"),
            "pinned_specimen": ("unknown", "pinned_specimen"),
        }[route]
    except KeyError as exc:
        raise ValueError(f"unsupported reference route: {route}") from exc


def _support_matches_requirement(
    row: Mapping[str, object],
    requirement: ReferenceBankRequirement,
) -> bool:
    return (
        str(row["accepted_taxon_key"]) == requirement.accepted_taxon_key
        and str(row["route"]) == requirement.route
        and (
            requirement.geo_cluster_id is None
            or str(row["geo_cluster_id"]) == requirement.geo_cluster_id
        )
    )


def _outcome_is_human_verified(outcome: Mapping[str, object] | None) -> bool:
    return bool(
        outcome is not None
        and outcome["review_status"] == "completed"
        and outcome["resolved_verification_status"] == "verified"
        and outcome["target_identity_verified"] is True
    )


def _pending_outcome_could_supply_target(
    outcome: Mapping[str, object],
    *,
    target_accepted_taxon_key: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> bool:
    media_id = str(outcome["reference_media_id"])
    candidate = indexes["media_candidates"].get(media_id)
    if candidate is None:
        return False
    observation = indexes["observations"].get(
        str(candidate["reference_observation_id"])
    )
    if observation is None or observation["accepted_taxon_key"] != (
        target_accepted_taxon_key
    ):
        return False
    life_stage = outcome.get("life_stage")
    visual_domain = outcome.get("visual_domain")
    if life_stage is None or visual_domain is None:
        return True
    return _reference_route(
        life_stage=str(life_stage),
        visual_domain=str(visual_domain),
    ) == "adult_field"


def _inventory_row_matches(
    row: Mapping[str, object],
    *,
    taxon_key: str,
    cluster_id: str,
    route: str,
) -> bool:
    if str(row["accepted_taxon_key"]) != taxon_key:
        return False
    if cluster_id != "all" and str(row["geo_cluster_id"] or "no_geo") != cluster_id:
        return False
    inferred_route = (
        "pinned_specimen"
        if bool(row["preserved_specimen"])
        else _reference_route(
            life_stage=str(row["life_stage"]),
            visual_domain="live_field",
        )
    )
    return inferred_route == route


def _cluster_matches_observation_id(
    observation_id: str,
    matching_observation_ids: set[str],
    *,
    cluster_id: str,
) -> bool:
    del cluster_id
    return observation_id in matching_observation_ids


def _complete_attribution(
    candidate: Mapping[str, object],
    observation: Mapping[str, object],
) -> bool:
    return all(
        _present(value)
        for value in (
            candidate.get("source"),
            candidate.get("provider_media_id"),
            candidate.get("creator"),
            candidate.get("rights_holder"),
            candidate.get("licence"),
            candidate.get("licence_uri"),
            candidate.get("attribution"),
            observation.get("source_observation_id"),
            observation.get("source_record_url"),
            observation.get("source_snapshot_version"),
        )
    )


def _support_attribution_complete(row: Mapping[str, object]) -> bool:
    return all(
        _present(row.get(field_name))
        for field_name in (
            "source",
            "provider_media_id",
            "creator",
            "rights_holder",
            "licence",
            "licence_uri",
            "attribution",
            "source_observation_id",
            "source_record_url",
            "source_snapshot_version",
        )
    )


def _requirement_key(
    requirement: ReferenceBankRequirement,
) -> tuple[str, str | None, str]:
    return (
        requirement.accepted_taxon_key,
        requirement.geo_cluster_id,
        requirement.route,
    )


def _shortfall_key(
    shortfall: DocumentedReferenceShortfall,
) -> tuple[str, str | None, str]:
    return (
        shortfall.accepted_taxon_key,
        shortfall.geo_cluster_id,
        shortfall.route,
    )


def _strict_frame(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: Mapping[str, pl.DataType],
    sort_by: Sequence[str],
) -> pl.DataFrame:
    try:
        frame = pl.DataFrame(list(rows), schema=dict(schema), strict=True)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"rows do not match the physical schema: {exc}") from exc
    return frame.sort(list(sort_by)) if frame.height else frame


def _validate_exact_frame(
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    artifact: str,
    sort_by: Sequence[str],
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{artifact} must be a Polars DataFrame")
    if frame.schema != dict(schema):
        raise ValueError(f"{artifact} does not match the physical schema")
    if not frame.equals(frame.sort(list(sort_by))):
        raise ValueError(f"{artifact} is not in deterministic sort order")


def _unique_index(
    frame: pl.DataFrame,
    *,
    key: str,
    artifact: str,
) -> dict[str, dict[str, object]]:
    if frame[key].n_unique() != frame.height:
        raise ValueError(f"{artifact} contains duplicate {key} values")
    return {str(row[key]): row for row in frame.iter_rows(named=True)}


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return _sha256_json(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": frame.to_dicts(),
        }
    )


def _semantic_frame_fingerprint(
    frame: pl.DataFrame,
    *,
    fields: Sequence[str],
) -> str:
    selected = tuple(fields)
    if len(set(selected)) != len(selected):
        raise ValueError("semantic projection fields must be unique")
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise ValueError(
            "semantic projection fields are missing: " + ", ".join(missing)
        )
    rows = [{field: row[field] for field in selected} for row in frame.to_dicts()]
    rows.sort(key=canonical_semantic_bytes)
    return canonical_semantic_fingerprint(
        {
            "projection_version": "reference-readiness-input-semantic-v1",
            "schema": [(name, str(frame.schema[name])) for name in selected],
            "rows": rows,
        }
    )


def _deduplication_report_fingerprint(
    report: Mapping[str, object],
    *,
    media_objects: pl.DataFrame,
    media_candidates: pl.DataFrame,
    observations: pl.DataFrame,
    duplicate_relationships: pl.DataFrame,
) -> str:
    settings = report.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("reference media deduplication report settings are invalid")
    return canonical_semantic_fingerprint(
        {
            "projection_version": "reference-deduplication-semantic-v1",
            "schema_version": report.get("schema_version"),
            "command": report.get("command"),
            "status": report.get("status"),
            "settings": dict(settings),
            "artifacts": {
                "observations": _semantic_frame_fingerprint(
                    observations,
                    fields=_READINESS_INPUT_SEMANTIC_FIELDS["observations"],
                ),
                "media_candidates": _semantic_frame_fingerprint(
                    media_candidates,
                    fields=_READINESS_INPUT_SEMANTIC_FIELDS["media_candidates"],
                ),
                "media_objects": _semantic_frame_fingerprint(
                    media_objects,
                    fields=_READINESS_INPUT_SEMANTIC_FIELDS["media_objects"],
                ),
                "duplicate_relationships": _semantic_frame_fingerprint(
                    duplicate_relationships,
                    fields=_READINESS_INPUT_SEMANTIC_FIELDS[
                        "duplicate_relationships"
                    ],
                ),
            },
        }
    )


def _support_manifest_fingerprint(frame: pl.DataFrame) -> str:
    return _semantic_frame_fingerprint(
        frame,
        fields=_SUPPORT_ROW_SEMANTIC_FIELDS,
    )


def _semantic_mapping_projection(
    row: Mapping[str, object],
    *,
    fields: Sequence[str],
    artifact: str,
) -> dict[str, object]:
    missing = sorted(set(fields) - set(row))
    if missing:
        raise ValueError(f"{artifact} semantic fields are missing: " + ", ".join(missing))
    return {field: row[field] for field in fields}


def _sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            _canonical(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical(value: object) -> object:
    if isinstance(value, datetime):
        return _utc_datetime(value, field="datetime").isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_canonical_sort_key)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprinted values must be finite")
        return 0.0 if value == 0.0 else value
    return value


def _canonical_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _dataclass_payload(value: object) -> dict[str, object]:
    payload = asdict(value)
    canonical = _canonical(payload)
    assert isinstance(canonical, dict)
    return canonical


def _json_roundtrip(value: Mapping[str, object]) -> dict[str, Any]:
    payload = json.loads(
        json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    assert isinstance(payload, dict)
    return payload


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    if value != value.strip():
        raise ValueError(f"{field} must be canonical text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _utc_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    return _utc_datetime(parsed, field=field)


def _fingerprint(
    value: object,
    *,
    field: str,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _absolute_uri(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    parsed = urlparse(text)
    if not parsed.scheme or not (parsed.netloc or parsed.path):
        raise ValueError(f"{field} must be an absolute URI")
    return text


def _mapping_sequence(value: object, *, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field} must be a list of objects")
    return value


def _exact_mapping_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    artifact: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError(f"{artifact} fields are invalid ({'; '.join(details)})")


def _sorted_unique_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    if value != sorted(set(value)):
        raise ValueError(f"{field} must be sorted and unique")
    return value


def _expect_identity(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: str | None,
) -> None:
    if expected is not None and payload.get(field) != expected:
        raise ValueError(
            f"reference bank readiness {field} mismatch: "
            f"expected {expected!r}, found {payload.get(field)!r}"
        )


def _expect_model_name(
    payload: Mapping[str, object],
    *,
    expected: str | None,
) -> None:
    if expected is None:
        return
    found = payload.get("model_name")
    canonical_expected = str(expected).strip().removeprefix("hf-hub:")
    canonical_found = str(found or "").strip().removeprefix("hf-hub:")
    if not canonical_expected or canonical_found != canonical_expected:
        raise ValueError(
            "reference bank readiness model_name mismatch: "
            f"expected {expected!r}, found {found!r}"
        )


def _write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(path) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if destination.exists():
            raise FileExistsError(destination)
        try:
            source.rename(destination)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(destination) from exc
            raise
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _write_failed_audit(
    directory: Path,
    *,
    command: str,
    run_id: str | None,
    started_at: datetime,
    error: Exception,
) -> None:
    ended_at = datetime.now(UTC)
    audit = directory.parent / f".{directory.name}.{uuid4().hex}.failed.json"
    try:
        _write_json_create(
            audit,
            {
                "schema_version": REFERENCE_BANK_READINESS_SCHEMA_VERSION,
                "command": command,
                "run_id": run_id or "not_instrumented",
                "pid": os.getpid(),
                "git_sha": current_git_sha(),
                "status": "failed",
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
                "network_requests": 0,
                "output_dir": str(directory),
                "error_type": type(error).__name__,
                "error": str(error),
                "artifact": "not_committed",
            },
        )
    except OSError:
        return
