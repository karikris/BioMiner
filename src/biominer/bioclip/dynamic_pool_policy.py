"""Immutable policy for geography-conditioned dynamic reference pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION = (
    "dynamic-reference-pool-policy-v1.0.0"
)
DYNAMIC_REFERENCE_POOL_POLICY_VERSION = "geography-conditioned-safe-v1"
DYNAMIC_REFERENCE_POOL_CLASS_BALANCE_MODES = frozenset(
    {"round_robin_equal_quota"}
)
DYNAMIC_REFERENCE_POOL_STAGE_ORDER = (
    "initial",
    "uncertainty_expansion",
    "selective_rescore",
)
DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER = (
    "exact_local_cell",
    "neighbouring_local_cell",
    "regional_cell",
    "bioregion",
    "admin1",
    "country",
    "nearest_geodesic",
    "global",
)

_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "stage_member_limits",
        "minimum_global_per_candidate",
        "maximum_global_per_candidate",
        "minimum_local_per_candidate",
        "maximum_local_per_candidate",
        "maximum_safety_per_candidate",
        "maximum_total_reference_members",
        "class_balance_mode",
        "maximum_class_count_difference",
        "geographic_fallback_order",
        "allow_no_geo_global_fallback",
        "maximum_nearest_geodesic_km",
        "minimum_independent_observation_groups_per_candidate",
        "maximum_members_per_observer",
        "maximum_members_per_locality",
        "maximum_members_per_burst_group",
        "minimum_global_countries_per_candidate",
        "uncertainty_margin_threshold",
        "uncertainty_candidate_rank_limit",
        "uncertainty_expansion_increment",
        "maximum_expansion_rounds",
        "always_include_target_candidate",
        "always_include_safety_union_candidates",
        "family_hard_pruning_allowed",
        "geography_hard_pruning_allowed",
        "selection_seed",
    }
)


@dataclass(frozen=True, slots=True)
class DynamicReferencePoolPolicy:
    """Validation-bound quotas, diversity, fallback and safety rules."""

    schema_version: str
    policy_version: str
    stage_member_limits: tuple[tuple[str, int], ...]
    minimum_global_per_candidate: int
    maximum_global_per_candidate: int
    minimum_local_per_candidate: int
    maximum_local_per_candidate: int
    maximum_safety_per_candidate: int
    maximum_total_reference_members: int
    class_balance_mode: str
    maximum_class_count_difference: int
    geographic_fallback_order: tuple[str, ...]
    allow_no_geo_global_fallback: bool
    maximum_nearest_geodesic_km: float
    minimum_independent_observation_groups_per_candidate: int
    maximum_members_per_observer: int
    maximum_members_per_locality: int
    maximum_members_per_burst_group: int
    minimum_global_countries_per_candidate: int
    uncertainty_margin_threshold: float
    uncertainty_candidate_rank_limit: int
    uncertainty_expansion_increment: int
    maximum_expansion_rounds: int
    always_include_target_candidate: bool
    always_include_safety_union_candidates: bool
    family_hard_pruning_allowed: bool
    geography_hard_pruning_allowed: bool
    selection_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, field="schema_version"),
        )
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, field="policy_version"),
        )
        if self.schema_version != DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic reference-pool policy schema")
        stage_limits = _normalize_stage_limits(self.stage_member_limits)
        object.__setattr__(self, "stage_member_limits", stage_limits)
        fallback = _text_tuple(
            self.geographic_fallback_order,
            field="geographic_fallback_order",
        )
        if fallback != DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER:
            raise ValueError(
                "geographic fallback order must retain the canonical safe sequence"
            )
        object.__setattr__(self, "geographic_fallback_order", fallback)
        balance = _required_text(self.class_balance_mode, field="class_balance_mode")
        if balance not in DYNAMIC_REFERENCE_POOL_CLASS_BALANCE_MODES:
            raise ValueError("unsupported dynamic reference-pool class-balance mode")
        object.__setattr__(self, "class_balance_mode", balance)
        self._validate_counts()
        self._validate_numbers()
        self._validate_safety()

    def _validate_counts(self) -> None:
        positive_fields = (
            "minimum_global_per_candidate",
            "maximum_global_per_candidate",
            "minimum_local_per_candidate",
            "maximum_local_per_candidate",
            "maximum_safety_per_candidate",
            "maximum_total_reference_members",
            "minimum_independent_observation_groups_per_candidate",
            "maximum_members_per_observer",
            "maximum_members_per_locality",
            "maximum_members_per_burst_group",
            "minimum_global_countries_per_candidate",
            "uncertainty_candidate_rank_limit",
            "uncertainty_expansion_increment",
            "maximum_expansion_rounds",
        )
        for field in positive_fields:
            _positive_int(getattr(self, field), field=field)
        _positive_int(
            self.maximum_class_count_difference,
            field="maximum_class_count_difference",
        )
        _uint64(self.selection_seed, field="selection_seed")
        if self.minimum_global_per_candidate > self.maximum_global_per_candidate:
            raise ValueError("global minimum cannot exceed global maximum")
        if self.minimum_local_per_candidate > self.maximum_local_per_candidate:
            raise ValueError("local minimum cannot exceed local maximum")
        minimum_one_candidate_budget = (
            self.maximum_global_per_candidate
            + self.maximum_local_per_candidate
            + self.maximum_safety_per_candidate
        )
        if self.maximum_total_reference_members < minimum_one_candidate_budget:
            raise ValueError(
                "total pool budget cannot cover one candidate's maximum quotas"
            )
        if self.stage_member_limits[-1][1] > self.maximum_total_reference_members:
            raise ValueError("stage member limit exceeds total pool budget")
        if (
            self.minimum_independent_observation_groups_per_candidate
            > minimum_one_candidate_budget
        ):
            raise ValueError(
                "independent-observation minimum exceeds per-candidate capacity"
            )
        if (
            self.minimum_global_countries_per_candidate
            > self.maximum_global_per_candidate
        ):
            raise ValueError("global country minimum exceeds global quota")
        if self.maximum_members_per_burst_group != 1:
            raise ValueError("burst groups may fill at most one dynamic-pool slot")

    def _validate_numbers(self) -> None:
        distance = _positive_float(
            self.maximum_nearest_geodesic_km,
            field="maximum_nearest_geodesic_km",
        )
        object.__setattr__(self, "maximum_nearest_geodesic_km", distance)
        margin = _nonnegative_float(
            self.uncertainty_margin_threshold,
            field="uncertainty_margin_threshold",
        )
        if margin > 2.0:
            raise ValueError("uncertainty margin threshold must be within [0, 2]")
        object.__setattr__(self, "uncertainty_margin_threshold", margin)

    def _validate_safety(self) -> None:
        boolean_fields = (
            "allow_no_geo_global_fallback",
            "always_include_target_candidate",
            "always_include_safety_union_candidates",
            "family_hard_pruning_allowed",
            "geography_hard_pruning_allowed",
        )
        for field in boolean_fields:
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be Boolean")
        if not self.allow_no_geo_global_fallback:
            raise ValueError("policy must retain a no-geo global fallback")
        if not self.always_include_target_candidate:
            raise ValueError("policy must always include the configured target")
        if not self.always_include_safety_union_candidates:
            raise ValueError("policy must retain the complete safety union")
        if self.family_hard_pruning_allowed:
            raise ValueError("family evidence cannot hard-prune candidates")
        if self.geography_hard_pruning_allowed:
            raise ValueError("geography evidence cannot hard-prune candidates")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "stage_member_limits": [
                {"stage": stage, "maximum_members": limit}
                for stage, limit in self.stage_member_limits
            ],
            "minimum_global_per_candidate": self.minimum_global_per_candidate,
            "maximum_global_per_candidate": self.maximum_global_per_candidate,
            "minimum_local_per_candidate": self.minimum_local_per_candidate,
            "maximum_local_per_candidate": self.maximum_local_per_candidate,
            "maximum_safety_per_candidate": self.maximum_safety_per_candidate,
            "maximum_total_reference_members": self.maximum_total_reference_members,
            "class_balance_mode": self.class_balance_mode,
            "maximum_class_count_difference": self.maximum_class_count_difference,
            "geographic_fallback_order": list(self.geographic_fallback_order),
            "allow_no_geo_global_fallback": self.allow_no_geo_global_fallback,
            "maximum_nearest_geodesic_km": self.maximum_nearest_geodesic_km,
            "minimum_independent_observation_groups_per_candidate": (
                self.minimum_independent_observation_groups_per_candidate
            ),
            "maximum_members_per_observer": self.maximum_members_per_observer,
            "maximum_members_per_locality": self.maximum_members_per_locality,
            "maximum_members_per_burst_group": self.maximum_members_per_burst_group,
            "minimum_global_countries_per_candidate": (
                self.minimum_global_countries_per_candidate
            ),
            "uncertainty_margin_threshold": self.uncertainty_margin_threshold,
            "uncertainty_candidate_rank_limit": self.uncertainty_candidate_rank_limit,
            "uncertainty_expansion_increment": self.uncertainty_expansion_increment,
            "maximum_expansion_rounds": self.maximum_expansion_rounds,
            "always_include_target_candidate": self.always_include_target_candidate,
            "always_include_safety_union_candidates": (
                self.always_include_safety_union_candidates
            ),
            "family_hard_pruning_allowed": self.family_hard_pruning_allowed,
            "geography_hard_pruning_allowed": self.geography_hard_pruning_allowed,
            "selection_seed": self.selection_seed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "policy_fingerprint": self.fingerprint}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> DynamicReferencePoolPolicy:
        if not isinstance(value, Mapping):
            raise TypeError("dynamic reference-pool policy must be a mapping")
        expected = _IDENTITY_FIELDS | {"policy_fingerprint"}
        if set(value) != expected:
            raise ValueError("dynamic reference-pool policy fields do not match")
        stages = value["stage_member_limits"]
        if isinstance(stages, str | bytes) or not isinstance(stages, Sequence):
            raise ValueError("stage_member_limits must be a sequence")
        parsed_stages: list[tuple[str, int]] = []
        for item in stages:
            if not isinstance(item, Mapping) or set(item) != {
                "stage",
                "maximum_members",
            }:
                raise ValueError("stage member-limit fields do not match")
            parsed_stages.append(
                (
                    _required_text(item["stage"], field="stage"),
                    _positive_int(
                        item["maximum_members"], field="maximum_members"
                    ),
                )
            )
        policy = cls(
            schema_version=_mapping_text(value, "schema_version"),
            policy_version=_mapping_text(value, "policy_version"),
            stage_member_limits=tuple(parsed_stages),
            minimum_global_per_candidate=_mapping_int(
                value, "minimum_global_per_candidate"
            ),
            maximum_global_per_candidate=_mapping_int(
                value, "maximum_global_per_candidate"
            ),
            minimum_local_per_candidate=_mapping_int(
                value, "minimum_local_per_candidate"
            ),
            maximum_local_per_candidate=_mapping_int(
                value, "maximum_local_per_candidate"
            ),
            maximum_safety_per_candidate=_mapping_int(
                value, "maximum_safety_per_candidate"
            ),
            maximum_total_reference_members=_mapping_int(
                value, "maximum_total_reference_members"
            ),
            class_balance_mode=_mapping_text(value, "class_balance_mode"),
            maximum_class_count_difference=_mapping_int(
                value, "maximum_class_count_difference"
            ),
            geographic_fallback_order=_mapping_text_tuple(
                value, "geographic_fallback_order"
            ),
            allow_no_geo_global_fallback=_mapping_bool(
                value, "allow_no_geo_global_fallback"
            ),
            maximum_nearest_geodesic_km=_mapping_float(
                value, "maximum_nearest_geodesic_km"
            ),
            minimum_independent_observation_groups_per_candidate=_mapping_int(
                value, "minimum_independent_observation_groups_per_candidate"
            ),
            maximum_members_per_observer=_mapping_int(
                value, "maximum_members_per_observer"
            ),
            maximum_members_per_locality=_mapping_int(
                value, "maximum_members_per_locality"
            ),
            maximum_members_per_burst_group=_mapping_int(
                value, "maximum_members_per_burst_group"
            ),
            minimum_global_countries_per_candidate=_mapping_int(
                value, "minimum_global_countries_per_candidate"
            ),
            uncertainty_margin_threshold=_mapping_float(
                value, "uncertainty_margin_threshold"
            ),
            uncertainty_candidate_rank_limit=_mapping_int(
                value, "uncertainty_candidate_rank_limit"
            ),
            uncertainty_expansion_increment=_mapping_int(
                value, "uncertainty_expansion_increment"
            ),
            maximum_expansion_rounds=_mapping_int(
                value, "maximum_expansion_rounds"
            ),
            always_include_target_candidate=_mapping_bool(
                value, "always_include_target_candidate"
            ),
            always_include_safety_union_candidates=_mapping_bool(
                value, "always_include_safety_union_candidates"
            ),
            family_hard_pruning_allowed=_mapping_bool(
                value, "family_hard_pruning_allowed"
            ),
            geography_hard_pruning_allowed=_mapping_bool(
                value, "geography_hard_pruning_allowed"
            ),
            selection_seed=_mapping_int(value, "selection_seed"),
        )
        if _mapping_text(value, "policy_fingerprint") != policy.fingerprint:
            raise ValueError("dynamic reference-pool policy fingerprint mismatch")
        return policy


def default_dynamic_reference_pool_policy() -> DynamicReferencePoolPolicy:
    """Return the explicit safe policy for deterministic pool planning."""

    return DynamicReferencePoolPolicy(
        schema_version=DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION,
        policy_version=DYNAMIC_REFERENCE_POOL_POLICY_VERSION,
        stage_member_limits=(
            ("initial", 96),
            ("uncertainty_expansion", 144),
            ("selective_rescore", 192),
        ),
        minimum_global_per_candidate=4,
        maximum_global_per_candidate=8,
        minimum_local_per_candidate=2,
        maximum_local_per_candidate=6,
        maximum_safety_per_candidate=4,
        maximum_total_reference_members=192,
        class_balance_mode="round_robin_equal_quota",
        maximum_class_count_difference=1,
        geographic_fallback_order=DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER,
        allow_no_geo_global_fallback=True,
        maximum_nearest_geodesic_km=2500.0,
        minimum_independent_observation_groups_per_candidate=4,
        maximum_members_per_observer=2,
        maximum_members_per_locality=2,
        maximum_members_per_burst_group=1,
        minimum_global_countries_per_candidate=2,
        uncertainty_margin_threshold=0.05,
        uncertainty_candidate_rank_limit=5,
        uncertainty_expansion_increment=2,
        maximum_expansion_rounds=3,
        always_include_target_candidate=True,
        always_include_safety_union_candidates=True,
        family_hard_pruning_allowed=False,
        geography_hard_pruning_allowed=False,
        selection_seed=20260718,
    )


def _normalize_stage_limits(
    value: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("stage_member_limits must be a sequence")
    normalized: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("stage member limits must contain two-item tuples")
        stage = _required_text(item[0], field="stage")
        limit = _positive_int(item[1], field=f"stage_member_limits[{stage}]")
        normalized.append((stage, limit))
    stages = tuple(stage for stage, _limit in normalized)
    if stages != DYNAMIC_REFERENCE_POOL_STAGE_ORDER:
        raise ValueError("stage member limits must use the canonical stage order")
    limits = [limit for _stage, limit in normalized]
    if limits != sorted(limits):
        raise ValueError("stage member limits must be nondecreasing")
    return tuple(normalized)


def _text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    result = tuple(_required_text(item, field=field) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _uint64(value: object, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result > 2**64 - 1:
        raise ValueError(f"{field} must fit UInt64")
    return result


def _nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _positive_float(value: object, *, field: str) -> float:
    result = _nonnegative_float(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _mapping_text(value: Mapping[str, object], field: str) -> str:
    return _required_text(value[field], field=field)


def _mapping_int(value: Mapping[str, object], field: str) -> int:
    return _nonnegative_int(value[field], field=field)


def _mapping_float(value: Mapping[str, object], field: str) -> float:
    return _nonnegative_float(value[field], field=field)


def _mapping_bool(value: Mapping[str, object], field: str) -> bool:
    result = value[field]
    if not isinstance(result, bool):
        raise ValueError(f"{field} must be Boolean")
    return result


def _mapping_text_tuple(
    value: Mapping[str, object], field: str
) -> tuple[str, ...]:
    return _text_tuple(value[field], field=field)


__all__ = [
    "DYNAMIC_REFERENCE_POOL_CLASS_BALANCE_MODES",
    "DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER",
    "DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION",
    "DYNAMIC_REFERENCE_POOL_POLICY_VERSION",
    "DYNAMIC_REFERENCE_POOL_STAGE_ORDER",
    "DynamicReferencePoolPolicy",
    "default_dynamic_reference_pool_policy",
]
