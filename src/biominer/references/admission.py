"""Versioned policy contract for reference-evidence admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.schemas import (
    LICENCE_POLICY_STATUSES,
    REFERENCE_ROUTES,
    TAXON_RECONCILIATION_STATUSES,
)


REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION = "reference-admission-policy-v1.0.0"
DEFAULT_REFERENCE_ADMISSION_MODE = "adaptive_gbif_fast_start"
REFERENCE_ADMISSION_MODES = frozenset(
    {
        DEFAULT_REFERENCE_ADMISSION_MODE,
        "human_verified_strict",
        "human_verified_flagged_only",
    }
)

_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "mode",
        "allowed_provider_sources",
        "allowed_unreviewed_routes",
        "accepted_taxon_reconciliation_statuses",
        "accepted_licence_policy_statuses",
        "minimum_decoded_width",
        "minimum_decoded_height",
        "minimum_subject_area_ratio",
        "require_yoloe_route",
        "require_canonical_media",
        "maximum_images_per_observation",
        "maximum_images_per_observer_before_reuse",
        "permit_research_only_licence",
        "require_statistical_audit",
        "audit_policy_version",
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceAdmissionPolicy:
    """Immutable semantic identity for strict and provisional admission rules."""

    schema_version: str
    policy_version: str
    mode: str
    allowed_provider_sources: tuple[str, ...]
    allowed_unreviewed_routes: tuple[str, ...]
    accepted_taxon_reconciliation_statuses: tuple[str, ...]
    accepted_licence_policy_statuses: tuple[str, ...]
    minimum_decoded_width: int
    minimum_decoded_height: int
    minimum_subject_area_ratio: float
    require_yoloe_route: bool
    require_canonical_media: bool
    maximum_images_per_observation: int
    maximum_images_per_observer_before_reuse: int
    permit_research_only_licence: bool
    require_statistical_audit: bool
    audit_policy_version: str

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported reference admission policy schema version")
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, field="policy_version"),
        )
        mode = _required_text(self.mode, field="mode")
        if mode not in REFERENCE_ADMISSION_MODES:
            raise ValueError(f"unsupported reference admission mode: {mode}")
        object.__setattr__(self, "mode", mode)

        sources = _normalized_values(
            self.allowed_provider_sources,
            field="allowed_provider_sources",
        )
        if not sources:
            raise ValueError("allowed_provider_sources must not be empty")
        object.__setattr__(self, "allowed_provider_sources", sources)

        routes = _normalized_values(
            self.allowed_unreviewed_routes,
            field="allowed_unreviewed_routes",
            allow_empty=True,
        )
        unsupported_routes = set(routes) - REFERENCE_ROUTES
        if unsupported_routes:
            raise ValueError(
                f"unsupported unreviewed reference routes: {sorted(unsupported_routes)}"
            )
        object.__setattr__(self, "allowed_unreviewed_routes", routes)

        taxon_statuses = _normalized_values(
            self.accepted_taxon_reconciliation_statuses,
            field="accepted_taxon_reconciliation_statuses",
        )
        unsupported_taxon_statuses = set(taxon_statuses) - TAXON_RECONCILIATION_STATUSES
        if unsupported_taxon_statuses:
            raise ValueError(
                "unsupported accepted taxon reconciliation statuses: "
                f"{sorted(unsupported_taxon_statuses)}"
            )
        object.__setattr__(
            self,
            "accepted_taxon_reconciliation_statuses",
            taxon_statuses,
        )

        licence_statuses = _normalized_values(
            self.accepted_licence_policy_statuses,
            field="accepted_licence_policy_statuses",
        )
        unsupported_licence_statuses = set(licence_statuses) - LICENCE_POLICY_STATUSES
        if unsupported_licence_statuses:
            raise ValueError(
                "unsupported accepted licence policy statuses: "
                f"{sorted(unsupported_licence_statuses)}"
            )
        object.__setattr__(
            self,
            "accepted_licence_policy_statuses",
            licence_statuses,
        )

        for field in (
            "minimum_decoded_width",
            "minimum_decoded_height",
            "maximum_images_per_observation",
            "maximum_images_per_observer_before_reuse",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")

        ratio = self.minimum_subject_area_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
        ):
            raise ValueError("minimum_subject_area_ratio must be finite and in [0, 1]")
        object.__setattr__(self, "minimum_subject_area_ratio", float(ratio))

        for field in (
            "require_yoloe_route",
            "require_canonical_media",
            "permit_research_only_licence",
            "require_statistical_audit",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be Boolean")

        object.__setattr__(
            self,
            "audit_policy_version",
            _required_text(self.audit_policy_version, field="audit_policy_version"),
        )
        self._validate_mode_invariants()

    def _validate_mode_invariants(self) -> None:
        if self.mode == DEFAULT_REFERENCE_ADMISSION_MODE:
            if self.allowed_provider_sources != ("gbif",):
                raise ValueError(
                    "adaptive GBIF fast-start permits only the GBIF provider source"
                )
            if not self.allowed_unreviewed_routes:
                raise ValueError(
                    "adaptive GBIF fast-start requires an explicit unreviewed route"
                )
            if self.minimum_subject_area_ratio <= 0:
                raise ValueError(
                    "adaptive GBIF fast-start requires a positive subject-area threshold"
                )
            if not self.require_yoloe_route or not self.require_canonical_media:
                raise ValueError(
                    "adaptive GBIF fast-start requires YOLOE routing and canonical media"
                )
            if not self.require_statistical_audit:
                raise ValueError(
                    "adaptive GBIF fast-start requires a statistical audit policy"
                )
        if self.mode == "human_verified_strict" and self.allowed_unreviewed_routes:
            raise ValueError(
                "strict admission cannot allow unreviewed reference routes"
            )
        research_only_accepted = (
            "research_only" in self.accepted_licence_policy_statuses
        )
        if research_only_accepted != self.permit_research_only_licence:
            raise ValueError(
                "research-only licence status and permission must agree explicitly"
            )

    @property
    def fingerprint(self) -> str:
        """Return the immutable semantic identity of every policy field."""

        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "mode": self.mode,
            "allowed_provider_sources": list(self.allowed_provider_sources),
            "allowed_unreviewed_routes": list(self.allowed_unreviewed_routes),
            "accepted_taxon_reconciliation_statuses": list(
                self.accepted_taxon_reconciliation_statuses
            ),
            "accepted_licence_policy_statuses": list(
                self.accepted_licence_policy_statuses
            ),
            "minimum_decoded_width": self.minimum_decoded_width,
            "minimum_decoded_height": self.minimum_decoded_height,
            "minimum_subject_area_ratio": self.minimum_subject_area_ratio,
            "require_yoloe_route": self.require_yoloe_route,
            "require_canonical_media": self.require_canonical_media,
            "maximum_images_per_observation": self.maximum_images_per_observation,
            "maximum_images_per_observer_before_reuse": (
                self.maximum_images_per_observer_before_reuse
            ),
            "permit_research_only_licence": self.permit_research_only_licence,
            "require_statistical_audit": self.require_statistical_audit,
            "audit_policy_version": self.audit_policy_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "policy_fingerprint": self.fingerprint}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ReferenceAdmissionPolicy:
        keys = set(value)
        expected = _POLICY_FIELDS | {"policy_fingerprint"}
        if keys != expected:
            missing = sorted(expected - keys)
            unknown = sorted(keys - expected)
            raise ValueError(
                "reference admission policy fields mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        policy = cls(
            schema_version=_mapping_text(value, "schema_version"),
            policy_version=_mapping_text(value, "policy_version"),
            mode=_mapping_text(value, "mode"),
            allowed_provider_sources=_mapping_text_tuple(
                value,
                "allowed_provider_sources",
            ),
            allowed_unreviewed_routes=_mapping_text_tuple(
                value,
                "allowed_unreviewed_routes",
            ),
            accepted_taxon_reconciliation_statuses=_mapping_text_tuple(
                value,
                "accepted_taxon_reconciliation_statuses",
            ),
            accepted_licence_policy_statuses=_mapping_text_tuple(
                value,
                "accepted_licence_policy_statuses",
            ),
            minimum_decoded_width=_mapping_int(value, "minimum_decoded_width"),
            minimum_decoded_height=_mapping_int(value, "minimum_decoded_height"),
            minimum_subject_area_ratio=_mapping_number(
                value,
                "minimum_subject_area_ratio",
            ),
            require_yoloe_route=_mapping_bool(value, "require_yoloe_route"),
            require_canonical_media=_mapping_bool(value, "require_canonical_media"),
            maximum_images_per_observation=_mapping_int(
                value,
                "maximum_images_per_observation",
            ),
            maximum_images_per_observer_before_reuse=_mapping_int(
                value,
                "maximum_images_per_observer_before_reuse",
            ),
            permit_research_only_licence=_mapping_bool(
                value,
                "permit_research_only_licence",
            ),
            require_statistical_audit=_mapping_bool(
                value,
                "require_statistical_audit",
            ),
            audit_policy_version=_mapping_text(value, "audit_policy_version"),
        )
        fingerprint = _mapping_text(value, "policy_fingerprint")
        if fingerprint != policy.fingerprint:
            raise ValueError("reference admission policy fingerprint mismatch")
        return policy


def default_reference_admission_policy() -> ReferenceAdmissionPolicy:
    """Return the explicit production-default adaptive GBIF policy."""

    return ReferenceAdmissionPolicy(
        schema_version=REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION,
        policy_version="adaptive-gbif-fast-start-v1",
        mode=DEFAULT_REFERENCE_ADMISSION_MODE,
        allowed_provider_sources=("gbif",),
        allowed_unreviewed_routes=("adult_field",),
        accepted_taxon_reconciliation_statuses=(
            "accepted_key_exact",
            "accepted_name_synonym",
        ),
        accepted_licence_policy_statuses=("allowed",),
        minimum_decoded_width=512,
        minimum_decoded_height=512,
        minimum_subject_area_ratio=0.05,
        require_yoloe_route=True,
        require_canonical_media=True,
        maximum_images_per_observation=1,
        maximum_images_per_observer_before_reuse=1,
        permit_research_only_licence=False,
        require_statistical_audit=True,
        audit_policy_version="reference-statistical-audit-v1",
    )


def _normalized_values(
    values: Sequence[str],
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence of strings")
    normalized = tuple(
        sorted({_required_text(value, field=field).casefold() for value in values})
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _mapping_text(value: Mapping[str, object], field: str) -> str:
    return _required_text(value[field], field=field)


def _mapping_text_tuple(value: Mapping[str, object], field: str) -> tuple[str, ...]:
    raw = value[field]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{field} must be a string list")
    return tuple(raw)


def _mapping_int(value: Mapping[str, object], field: str) -> int:
    raw = value[field]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer")
    return raw


def _mapping_number(value: Mapping[str, object], field: str) -> int | float:
    raw = value[field]
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field} must be numeric")
    return raw


def _mapping_bool(value: Mapping[str, object], field: str) -> bool:
    raw = value[field]
    if not isinstance(raw, bool):
        raise ValueError(f"{field} must be Boolean")
    return raw


__all__ = [
    "DEFAULT_REFERENCE_ADMISSION_MODE",
    "REFERENCE_ADMISSION_MODES",
    "REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION",
    "ReferenceAdmissionPolicy",
    "default_reference_admission_policy",
]
