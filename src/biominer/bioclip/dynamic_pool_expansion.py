"""Deterministic evidence and policy for dynamic reference-pool expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import re

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    build_dynamic_reference_pool_members,
    build_dynamic_reference_pool_plans,
    build_dynamic_reference_pool_summaries,
    dynamic_reference_pool_member_schema,
    dynamic_reference_pool_plan_schema,
    dynamic_reference_pool_summary_schema,
    validate_dynamic_reference_pool_artifacts,
)
from biominer.bioclip.dynamic_pool_planner import (
    DYNAMIC_POOL_PLANNING_REQUEST_FIELDS,
    plan_dynamic_reference_pools,
)
from biominer.bioclip.dynamic_pool_policy import DynamicReferencePoolPolicy
from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.bioclip.global_reference_anchors import (
    validate_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    validate_reference_geography_index,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION = (
    "dynamic-pool-expansion-evidence-v1.0.0"
)
DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_SCHEMA_VERSION = (
    "dynamic-pool-expansion-signal-policy-v1.0.0"
)
DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_VERSION = "raw-evidence-safe-v1"
DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION = (
    "dynamic-pool-expansion-cache-reuse-v1.0.0"
)

DYNAMIC_POOL_EXPANSION_SIGNALS = (
    "small_family_margin",
    "small_species_margin",
    "global_local_disagreement",
    "prototype_method_disagreement",
    "visual_input_disagreement",
    "insufficient_local_support",
    "low_subject_area",
    "strong_known_competitor",
    "no_geo_global_fallback",
    "out_of_distribution",
    "route_domain_incompatible",
)

_SIGNAL_VALUE_FIELDS = {
    "small_family_margin": "family_margin",
    "small_species_margin": "species_margin",
    "global_local_disagreement": "global_local_disagreement",
    "prototype_method_disagreement": "prototype_method_disagreement",
    "visual_input_disagreement": "visual_input_disagreement",
    "insufficient_local_support": "local_support_ratio",
    "low_subject_area": "subject_area_ratio",
    "strong_known_competitor": "known_competitor_margin",
    "no_geo_global_fallback": "no_geo_global_fallback",
    "out_of_distribution": "out_of_distribution_score",
    "route_domain_incompatible": "route_domain_compatible",
}
_FLOAT_VALUE_FIELDS = frozenset(
    field
    for field in _SIGNAL_VALUE_FIELDS.values()
    if field not in {"no_geo_global_fallback", "route_domain_compatible"}
)
_RATIO_FIELDS = frozenset(
    {"local_support_ratio", "subject_area_ratio", "out_of_distribution_score"}
)
_INPUT_FIELDS = frozenset(
    {
        "run_id",
        "plan_id",
        "plan_fingerprint",
        "candidate_scores_fingerprint",
        "selection_policy_fingerprint",
        "model_fingerprint",
        "expansion_round",
        *_SIGNAL_VALUE_FIELDS.values(),
        "unavailable_signal_reasons",
    }
)
_SORT = ("run_id", "plan_id", "expansion_round")
_CACHE_REUSE_SORT = ("run_id", "prior_plan_id", "expansion_round")

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLAN_ID_PATTERN = re.compile(r"dynamic-pool-plan:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DynamicPoolExpansionSignalPolicy:
    """Versioned raw-evidence thresholds; never calibrated probabilities."""

    schema_version: str
    policy_version: str
    family_margin_threshold: float
    species_margin_threshold: float
    global_local_disagreement_threshold: float
    prototype_method_disagreement_threshold: float
    visual_input_disagreement_threshold: float
    minimum_local_support_ratio: float
    minimum_subject_area_ratio: float
    known_competitor_margin_threshold: float
    out_of_distribution_score_threshold: float
    expand_on_no_geo_global_fallback: bool
    expand_on_route_domain_incompatibility: bool

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported expansion signal policy schema")
        _required_text(self.policy_version, field="policy_version")
        bounded_two = (
            "family_margin_threshold",
            "species_margin_threshold",
            "global_local_disagreement_threshold",
            "prototype_method_disagreement_threshold",
            "visual_input_disagreement_threshold",
            "known_competitor_margin_threshold",
        )
        for field in bounded_two:
            value = _bounded_float(getattr(self, field), field=field, maximum=2.0)
            object.__setattr__(self, field, value)
        bounded_one = (
            "minimum_local_support_ratio",
            "minimum_subject_area_ratio",
            "out_of_distribution_score_threshold",
        )
        for field in bounded_one:
            value = _bounded_float(getattr(self, field), field=field, maximum=1.0)
            object.__setattr__(self, field, value)
        for field in (
            "expand_on_no_geo_global_fallback",
            "expand_on_route_domain_incompatibility",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be Boolean")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "family_margin_threshold": self.family_margin_threshold,
            "species_margin_threshold": self.species_margin_threshold,
            "global_local_disagreement_threshold": (
                self.global_local_disagreement_threshold
            ),
            "prototype_method_disagreement_threshold": (
                self.prototype_method_disagreement_threshold
            ),
            "visual_input_disagreement_threshold": (
                self.visual_input_disagreement_threshold
            ),
            "minimum_local_support_ratio": self.minimum_local_support_ratio,
            "minimum_subject_area_ratio": self.minimum_subject_area_ratio,
            "known_competitor_margin_threshold": (
                self.known_competitor_margin_threshold
            ),
            "out_of_distribution_score_threshold": (
                self.out_of_distribution_score_threshold
            ),
            "expand_on_no_geo_global_fallback": (
                self.expand_on_no_geo_global_fallback
            ),
            "expand_on_route_domain_incompatibility": (
                self.expand_on_route_domain_incompatibility
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "policy_fingerprint": self.fingerprint}

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object]
    ) -> DynamicPoolExpansionSignalPolicy:
        if not isinstance(values, Mapping):
            raise TypeError("expansion signal policy must be a mapping")
        expected = {
            "schema_version",
            "policy_version",
            "family_margin_threshold",
            "species_margin_threshold",
            "global_local_disagreement_threshold",
            "prototype_method_disagreement_threshold",
            "visual_input_disagreement_threshold",
            "minimum_local_support_ratio",
            "minimum_subject_area_ratio",
            "known_competitor_margin_threshold",
            "out_of_distribution_score_threshold",
            "expand_on_no_geo_global_fallback",
            "expand_on_route_domain_incompatibility",
            "policy_fingerprint",
        }
        _require_exact_fields(values, expected, label="expansion signal policy")
        policy = cls(
            schema_version=_required_text(
                values["schema_version"], field="schema_version"
            ),
            policy_version=_required_text(
                values["policy_version"], field="policy_version"
            ),
            family_margin_threshold=_number(
                values["family_margin_threshold"], field="family_margin_threshold"
            ),
            species_margin_threshold=_number(
                values["species_margin_threshold"], field="species_margin_threshold"
            ),
            global_local_disagreement_threshold=_number(
                values["global_local_disagreement_threshold"],
                field="global_local_disagreement_threshold",
            ),
            prototype_method_disagreement_threshold=_number(
                values["prototype_method_disagreement_threshold"],
                field="prototype_method_disagreement_threshold",
            ),
            visual_input_disagreement_threshold=_number(
                values["visual_input_disagreement_threshold"],
                field="visual_input_disagreement_threshold",
            ),
            minimum_local_support_ratio=_number(
                values["minimum_local_support_ratio"],
                field="minimum_local_support_ratio",
            ),
            minimum_subject_area_ratio=_number(
                values["minimum_subject_area_ratio"],
                field="minimum_subject_area_ratio",
            ),
            known_competitor_margin_threshold=_number(
                values["known_competitor_margin_threshold"],
                field="known_competitor_margin_threshold",
            ),
            out_of_distribution_score_threshold=_number(
                values["out_of_distribution_score_threshold"],
                field="out_of_distribution_score_threshold",
            ),
            expand_on_no_geo_global_fallback=_boolean(
                values["expand_on_no_geo_global_fallback"],
                field="expand_on_no_geo_global_fallback",
            ),
            expand_on_route_domain_incompatibility=_boolean(
                values["expand_on_route_domain_incompatibility"],
                field="expand_on_route_domain_incompatibility",
            ),
        )
        if values["policy_fingerprint"] != policy.fingerprint:
            raise ValueError("expansion signal policy fingerprint mismatch")
        return policy


def default_dynamic_pool_expansion_signal_policy(
) -> DynamicPoolExpansionSignalPolicy:
    return DynamicPoolExpansionSignalPolicy(
        schema_version=DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_SCHEMA_VERSION,
        policy_version=DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_VERSION,
        family_margin_threshold=0.05,
        species_margin_threshold=0.05,
        global_local_disagreement_threshold=0.20,
        prototype_method_disagreement_threshold=0.15,
        visual_input_disagreement_threshold=0.15,
        minimum_local_support_ratio=0.50,
        minimum_subject_area_ratio=0.10,
        known_competitor_margin_threshold=0.05,
        out_of_distribution_score_threshold=0.80,
        expand_on_no_geo_global_fallback=True,
        expand_on_route_domain_incompatibility=True,
    )


def dynamic_pool_expansion_evidence_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "schema_version": pl.String,
        "run_id": pl.String,
        "plan_id": pl.String,
        "plan_fingerprint": pl.String,
        "candidate_scores_fingerprint": pl.String,
        "selection_policy_fingerprint": pl.String,
        "signal_policy_version": pl.String,
        "signal_policy_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "expansion_round": pl.UInt16,
    }
    schema.update({field: pl.Float64 for field in sorted(_FLOAT_VALUE_FIELDS)})
    schema.update(
        {
            "no_geo_global_fallback": pl.Boolean,
            "route_domain_compatible": pl.Boolean,
            "observed_signals": pl.List(pl.String),
            "unavailable_signals": pl.List(pl.String),
            "unavailable_signal_reasons": pl.List(pl.String),
            "triggered_signals": pl.List(pl.String),
            "expansion_required": pl.Boolean,
            "evidence_fingerprint": pl.String,
        }
    )
    return schema


def build_dynamic_pool_expansion_evidence(
    rows: Sequence[Mapping[str, object]],
    *,
    policy: DynamicPoolExpansionSignalPolicy | None = None,
) -> pl.DataFrame:
    """Evaluate raw uncertainty signals without planning or scoring a new pool."""

    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("dynamic pool expansion evidence rows must be a sequence")
    active_policy = policy or default_dynamic_pool_expansion_signal_policy()
    if not isinstance(active_policy, DynamicPoolExpansionSignalPolicy):
        raise TypeError("policy must be a DynamicPoolExpansionSignalPolicy")
    output: list[dict[str, object]] = []
    for source in rows:
        if not isinstance(source, Mapping):
            raise TypeError("expansion evidence rows must contain mappings")
        _require_exact_fields(source, set(_INPUT_FIELDS), label="expansion evidence")
        normalized = _normalized_evidence(source)
        observed, unavailable, reasons = _signal_availability(normalized)
        triggered = _triggered_signals(normalized, policy=active_policy)
        materialized = {
            field: value
            for field, value in normalized.items()
            if field != "_reason_map"
        }
        complete: dict[str, object] = {
            "schema_version": DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION,
            **materialized,
            "signal_policy_version": active_policy.policy_version,
            "signal_policy_fingerprint": active_policy.fingerprint,
            "observed_signals": observed,
            "unavailable_signals": unavailable,
            "unavailable_signal_reasons": reasons,
            "triggered_signals": triggered,
            "expansion_required": bool(triggered),
        }
        complete["evidence_fingerprint"] = canonical_semantic_fingerprint(complete)
        output.append(complete)
    frame = (
        pl.DataFrame(
            output,
            schema=dynamic_pool_expansion_evidence_schema(),
            orient="row",
            strict=True,
        ).sort(*_SORT)
        if output
        else pl.DataFrame(schema=dynamic_pool_expansion_evidence_schema())
    )
    validate_dynamic_pool_expansion_evidence(frame, policy=active_policy)
    return frame


def validate_dynamic_pool_expansion_evidence(
    frame: pl.DataFrame,
    *,
    policy: DynamicPoolExpansionSignalPolicy | None = None,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("dynamic pool expansion evidence must be a Polars DataFrame")
    if frame.schema != dynamic_pool_expansion_evidence_schema():
        raise ValueError("dynamic pool expansion evidence schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("dynamic pool expansion evidence is not canonically sorted")
    if frame.select("plan_id", "expansion_round").n_unique() != frame.height:
        raise ValueError("dynamic pool expansion evidence grain is not unique")
    for row in frame.iter_rows(named=True):
        _validate_materialized_evidence(row, policy=policy)


def dynamic_pool_expansion_cache_reuse_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "prior_plan_id": pl.String,
        "prior_plan_fingerprint": pl.String,
        "expanded_plan_id": pl.String,
        "expanded_plan_fingerprint": pl.String,
        "expansion_evidence_fingerprint": pl.String,
        "selection_policy_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "query_embedding_fingerprint": pl.String,
        "expansion_round": pl.UInt16,
        "retained_reference_count": pl.UInt32,
        "added_reference_count": pl.UInt32,
        "dropped_reference_count": pl.UInt32,
        "prior_reference_embedding_fingerprints": pl.List(pl.String),
        "added_reference_embedding_fingerprints": pl.List(pl.String),
        "expanded_reference_embedding_fingerprints": pl.List(pl.String),
        "query_embedding_reused": pl.Boolean,
        "reference_embeddings_reused": pl.Boolean,
        "encoder_invocations": pl.UInt32,
        "embedding_vectors_materialized": pl.Boolean,
        "expanded_membership_fingerprint": pl.String,
        "reuse_fingerprint": pl.String,
    }


def expand_dynamic_reference_pools_from_cache(
    prior_plans: pl.DataFrame,
    prior_members: pl.DataFrame,
    prior_summaries: pl.DataFrame,
    expansion_evidence: pl.DataFrame,
    candidate_sets: pl.DataFrame,
    reference_geography_index: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    *,
    policy: DynamicReferencePoolPolicy,
    burst_group_by_observation: Mapping[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Materialize triggered plans using cached embedding identities only.

    The function deliberately accepts no encoder and no embedding-vector column.
    It replaces an immutable plan with a larger plan over the same query and
    reference cache identities, while retaining every prior member.
    """

    if not isinstance(policy, DynamicReferencePoolPolicy):
        raise TypeError("policy must be a DynamicReferencePoolPolicy")
    validate_dynamic_reference_pool_artifacts(
        prior_plans, prior_members, prior_summaries
    )
    validate_dynamic_pool_expansion_evidence(expansion_evidence)
    validate_family_geo_candidate_sets(candidate_sets)
    validate_reference_geography_index(reference_geography_index)
    validate_global_reference_anchors(global_reference_anchors)
    _validate_expansion_inputs(
        prior_plans,
        prior_members=prior_members,
        evidence=expansion_evidence,
        reference_index=reference_geography_index,
        policy=policy,
    )
    triggered = expansion_evidence.filter(pl.col("expansion_required"))
    if triggered.is_empty():
        return (
            pl.DataFrame(schema=dynamic_reference_pool_plan_schema()),
            pl.DataFrame(schema=dynamic_reference_pool_member_schema()),
            pl.DataFrame(schema=dynamic_reference_pool_summary_schema()),
            pl.DataFrame(schema=dynamic_pool_expansion_cache_reuse_schema()),
        )

    prior_lookup = {
        str(row["plan_id"]): row for row in prior_plans.iter_rows(named=True)
    }
    evidence_lookup = {
        str(row["plan_id"]): row for row in triggered.iter_rows(named=True)
    }
    selected_prior = [prior_lookup[plan_id] for plan_id in sorted(evidence_lookup)]
    requests = [_expansion_request(row) for row in selected_prior]
    planned_plans, planned_members, _planned_summaries = plan_dynamic_reference_pools(
        requests,
        candidate_sets,
        reference_geography_index,
        global_reference_anchors,
        policy=policy,
        burst_group_by_observation=burst_group_by_observation,
    )
    planned_lookup = {
        _plan_match_key(row): row for row in planned_plans.iter_rows(named=True)
    }
    member_inputs: list[dict[str, object]] = []
    plan_inputs: list[dict[str, object]] = []
    reuse_contexts: list[dict[str, object]] = []
    for prior in selected_prior:
        evidence = evidence_lookup[str(prior["plan_id"])]
        expanded = planned_lookup.get(_plan_match_key(prior))
        if expanded is None:
            raise ValueError("expanded plan is missing its prior plan context")
        prior_group = prior_members.filter(pl.col("plan_id") == prior["plan_id"])
        expanded_group = planned_members.filter(
            pl.col("plan_id") == expanded["plan_id"]
        )
        prior_by_identity = {
            _member_cache_identity(row): row
            for row in prior_group.iter_rows(named=True)
        }
        expanded_by_identity = {
            _member_cache_identity(row): row
            for row in expanded_group.iter_rows(named=True)
        }
        if not set(prior_by_identity) <= set(expanded_by_identity):
            raise ValueError("expanded plan dropped a prior reference membership")
        next_round = int(evidence["expansion_round"]) + 1
        group_inputs: list[dict[str, object]] = []
        for identity, row in expanded_by_identity.items():
            item = _member_input(row)
            item["expansion_round"] = (
                int(prior_by_identity[identity]["expansion_round"])
                if identity in prior_by_identity
                else next_round
            )
            group_inputs.append(item)
        _reset_pool_selection_ranks(group_inputs)
        member_inputs.extend(group_inputs)
        plan_inputs.append(_plan_input(expanded))
        reuse_contexts.append(
            {
                "prior": prior,
                "expanded_plan_id": expanded["plan_id"],
                "evidence": evidence,
                "prior_identities": set(prior_by_identity),
                "expanded_identities": set(expanded_by_identity),
                "next_round": next_round,
            }
        )

    members = build_dynamic_reference_pool_members(member_inputs)
    plans = build_dynamic_reference_pool_plans(plan_inputs, members)
    summaries = build_dynamic_reference_pool_summaries(plans, members)
    validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    plan_by_id = {str(row["plan_id"]): row for row in plans.iter_rows(named=True)}
    reuse_rows = [
        _cache_reuse_row(
            context,
            expanded_plan=plan_by_id[str(context["expanded_plan_id"])],
            expanded_members=members.filter(
                pl.col("plan_id") == context["expanded_plan_id"]
            ),
        )
        for context in reuse_contexts
    ]
    reuse = pl.DataFrame(
        reuse_rows,
        schema=dynamic_pool_expansion_cache_reuse_schema(),
        orient="row",
        strict=True,
    ).sort(*_CACHE_REUSE_SORT)
    validate_dynamic_pool_expansion_cache_reuse(reuse)
    return plans, members, summaries, reuse


def validate_dynamic_pool_expansion_cache_reuse(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("dynamic pool cache reuse must be a Polars DataFrame")
    if frame.schema != dynamic_pool_expansion_cache_reuse_schema():
        raise ValueError("dynamic pool cache reuse schema mismatch")
    if not frame.equals(frame.sort(*_CACHE_REUSE_SORT)):
        raise ValueError("dynamic pool cache reuse is not canonically sorted")
    if frame.select("prior_plan_id", "expansion_round").n_unique() != frame.height:
        raise ValueError("dynamic pool cache reuse grain is not unique")
    for row in frame.iter_rows(named=True):
        for field in (
            "prior_plan_id",
            "expanded_plan_id",
        ):
            if not _PLAN_ID_PATTERN.fullmatch(str(row[field])):
                raise ValueError(f"{field} is invalid")
        for field in (
            "prior_plan_fingerprint",
            "expanded_plan_fingerprint",
            "expansion_evidence_fingerprint",
            "selection_policy_fingerprint",
            "model_fingerprint",
            "query_embedding_fingerprint",
            "expanded_membership_fingerprint",
            "reuse_fingerprint",
        ):
            _sha256(row[field], field=field)
        for field in (
            "prior_reference_embedding_fingerprints",
            "added_reference_embedding_fingerprints",
            "expanded_reference_embedding_fingerprints",
        ):
            values = list(row[field])
            if values != sorted(set(values)):
                raise ValueError(f"{field} must be unique and sorted")
            for value in values:
                _sha256(value, field=field)
        prior = set(row["prior_reference_embedding_fingerprints"])
        added = set(row["added_reference_embedding_fingerprints"])
        expanded = set(row["expanded_reference_embedding_fingerprints"])
        if prior & added or prior | added != expanded:
            raise ValueError("cache reuse embedding identity sets are inconsistent")
        if row["retained_reference_count"] != len(prior):
            raise ValueError("retained cache identity count is inconsistent")
        if row["added_reference_count"] != len(added):
            raise ValueError("added cache identity count is inconsistent")
        if row["dropped_reference_count"] != 0:
            raise ValueError("cached expansion cannot drop prior references")
        if (
            row["query_embedding_reused"] is not True
            or row["reference_embeddings_reused"] is not True
            or row["encoder_invocations"] != 0
            or row["embedding_vectors_materialized"] is not False
        ):
            raise ValueError("cache reuse evidence permits embedding recomputation")
        identity = dict(row)
        fingerprint = identity.pop("reuse_fingerprint")
        if canonical_semantic_fingerprint(identity) != fingerprint:
            raise ValueError("dynamic pool cache reuse fingerprint mismatch")


def _validate_expansion_inputs(
    plans: pl.DataFrame,
    *,
    prior_members: pl.DataFrame,
    evidence: pl.DataFrame,
    reference_index: pl.DataFrame,
    policy: DynamicReferencePoolPolicy,
) -> None:
    plan_lookup = {
        str(row["plan_id"]): row for row in plans.iter_rows(named=True)
    }
    unknown = set(evidence["plan_id"].to_list()) - set(plan_lookup)
    if unknown:
        raise ValueError("expansion evidence references an unknown prior plan")
    for row in evidence.iter_rows(named=True):
        plan = plan_lookup[str(row["plan_id"])]
        if row["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("expansion evidence prior plan fingerprint mismatch")
        if row["selection_policy_fingerprint"] != policy.fingerprint:
            raise ValueError("expansion evidence selection policy mismatch")
        if plan["selection_policy_fingerprint"] != policy.fingerprint:
            raise ValueError("prior plan selection policy mismatch")
        if row["model_fingerprint"] != plan["model_fingerprint"]:
            raise ValueError("expansion evidence model fingerprint mismatch")
        if int(row["expansion_round"]) != int(
            prior_members.filter(pl.col("plan_id") == plan["plan_id"])[
                "expansion_round"
            ].max()
        ):
            raise ValueError("expansion evidence round does not match prior members")
        if int(row["expansion_round"]) >= policy.maximum_expansion_rounds:
            raise ValueError("expansion evidence already reached the maximum rounds")
    index_identities = {
        (str(row["reference_media_id"]), str(row["embedding_fingerprint"]))
        for row in reference_index.iter_rows(named=True)
    }
    member_identities = {
        (str(row["reference_media_id"]), str(row["reference_embedding_fingerprint"]))
        for row in prior_members.iter_rows(named=True)
    }
    if not member_identities <= index_identities:
        raise ValueError("prior member embedding identity is absent from reference index")


def _expansion_request(plan: Mapping[str, object]) -> dict[str, object]:
    request = {field: plan[field] for field in DYNAMIC_POOL_PLANNING_REQUEST_FIELDS}
    request["scoring_stage"] = "uncertainty_expansion"
    return request


def _plan_match_key(plan: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        plan[field]
        for field in (
            "run_id",
            "flickr_query_id",
            "flickr_photo_id",
            "organism_unit_id",
            "visual_input_id",
            "candidate_set_id",
        )
    )


def _member_cache_identity(member: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(member[field])
        for field in (
            "candidate_accepted_taxon_key",
            "reference_media_id",
            "reference_observation_id",
            "reference_embedding_fingerprint",
        )
    )


def _member_input(member: Mapping[str, object]) -> dict[str, object]:
    excluded = {"schema_version", "pool_id", "member_fingerprint"}
    return {field: value for field, value in member.items() if field not in excluded}


def _plan_input(plan: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "schema_version",
        "global_pool_ids",
        "local_pool_ids",
        "safety_pool_ids",
        "plan_fingerprint",
    }
    return {field: value for field, value in plan.items() if field not in excluded}


def _reset_pool_selection_ranks(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "plan_id",
                "pool_scope",
                "pool_role",
                "candidate_accepted_taxon_key",
                "expansion_round",
            )
        )
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        group.sort(
            key=lambda row: (
                int(row["selection_rank"]),
                str(row["reference_observation_id"]),
                str(row["reference_media_id"]),
            )
        )
        for rank, row in enumerate(group, start=1):
            row["selection_rank"] = rank


def _cache_reuse_row(
    context: Mapping[str, object],
    *,
    expanded_plan: Mapping[str, object],
    expanded_members: pl.DataFrame,
) -> dict[str, object]:
    prior = context["prior"]
    evidence = context["evidence"]
    if not isinstance(prior, Mapping) or not isinstance(evidence, Mapping):
        raise TypeError("cache reuse context is invalid")
    prior_identities = context["prior_identities"]
    expanded_identities = context["expanded_identities"]
    if not isinstance(prior_identities, set) or not isinstance(
        expanded_identities, set
    ):
        raise TypeError("cache reuse member identity context is invalid")
    added_identities = expanded_identities - prior_identities
    prior_fingerprints = sorted({str(identity[3]) for identity in prior_identities})
    added_fingerprints = sorted({str(identity[3]) for identity in added_identities})
    expanded_fingerprints = sorted(
        {str(identity[3]) for identity in expanded_identities}
    )
    if set(prior_fingerprints) & set(added_fingerprints):
        raise ValueError(
            "one reference embedding identity cannot be both retained and added"
        )
    membership_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": "expanded-reference-membership-set-v1",
            "expanded_plan_id": expanded_plan["plan_id"],
            "member_fingerprints": expanded_members["member_fingerprint"].to_list(),
        }
    )
    row: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION,
        "run_id": prior["run_id"],
        "prior_plan_id": prior["plan_id"],
        "prior_plan_fingerprint": prior["plan_fingerprint"],
        "expanded_plan_id": expanded_plan["plan_id"],
        "expanded_plan_fingerprint": expanded_plan["plan_fingerprint"],
        "expansion_evidence_fingerprint": evidence["evidence_fingerprint"],
        "selection_policy_fingerprint": prior["selection_policy_fingerprint"],
        "model_fingerprint": prior["model_fingerprint"],
        "query_embedding_fingerprint": prior["query_embedding_fingerprint"],
        "expansion_round": int(context["next_round"]),
        "retained_reference_count": len(prior_fingerprints),
        "added_reference_count": len(added_fingerprints),
        "dropped_reference_count": 0,
        "prior_reference_embedding_fingerprints": prior_fingerprints,
        "added_reference_embedding_fingerprints": added_fingerprints,
        "expanded_reference_embedding_fingerprints": expanded_fingerprints,
        "query_embedding_reused": True,
        "reference_embeddings_reused": True,
        "encoder_invocations": 0,
        "embedding_vectors_materialized": False,
        "expanded_membership_fingerprint": membership_fingerprint,
    }
    row["reuse_fingerprint"] = canonical_semantic_fingerprint(row)
    return row


def _normalized_evidence(values: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": _required_text(values["run_id"], field="run_id"),
        "plan_id": _required_text(values["plan_id"], field="plan_id"),
        "plan_fingerprint": _sha256(
            values["plan_fingerprint"], field="plan_fingerprint"
        ),
        "candidate_scores_fingerprint": _sha256(
            values["candidate_scores_fingerprint"],
            field="candidate_scores_fingerprint",
        ),
        "selection_policy_fingerprint": _sha256(
            values["selection_policy_fingerprint"],
            field="selection_policy_fingerprint",
        ),
        "model_fingerprint": _sha256(
            values["model_fingerprint"], field="model_fingerprint"
        ),
        "expansion_round": _nonnegative_int(
            values["expansion_round"], field="expansion_round", maximum=2**16 - 1
        ),
    }
    if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
        raise ValueError("dynamic pool expansion plan_id is invalid")
    for field in _FLOAT_VALUE_FIELDS:
        maximum = 1.0 if field in _RATIO_FIELDS else 2.0
        row[field] = _optional_bounded_float(
            values[field], field=field, maximum=maximum
        )
    for field in ("no_geo_global_fallback", "route_domain_compatible"):
        row[field] = _optional_boolean(values[field], field=field)
    raw_reasons = values["unavailable_signal_reasons"]
    if not isinstance(raw_reasons, Mapping):
        raise TypeError("unavailable_signal_reasons must be a mapping")
    reasons = {
        _required_text(signal, field="unavailable signal"): _required_text(
            reason, field=f"unavailable reason for {signal}"
        )
        for signal, reason in raw_reasons.items()
    }
    unavailable = {
        signal
        for signal, value_field in _SIGNAL_VALUE_FIELDS.items()
        if row[value_field] is None
    }
    if set(reasons) != unavailable:
        raise ValueError(
            "unavailable signal reasons must exactly match unavailable signals"
        )
    row["_reason_map"] = reasons
    return row


def _signal_availability(
    row: Mapping[str, object],
) -> tuple[list[str], list[str], list[str]]:
    reasons = row["_reason_map"]
    if not isinstance(reasons, Mapping):
        raise TypeError("normalized unavailable reasons must be a mapping")
    unavailable = sorted(str(signal) for signal in reasons)
    observed = sorted(set(DYNAMIC_POOL_EXPANSION_SIGNALS) - set(unavailable))
    return observed, unavailable, [str(reasons[signal]) for signal in unavailable]


def _triggered_signals(
    row: Mapping[str, object], *, policy: DynamicPoolExpansionSignalPolicy
) -> list[str]:
    checks = {
        "small_family_margin": _at_most(
            row["family_margin"], policy.family_margin_threshold
        ),
        "small_species_margin": _at_most(
            row["species_margin"], policy.species_margin_threshold
        ),
        "global_local_disagreement": _at_least(
            row["global_local_disagreement"],
            policy.global_local_disagreement_threshold,
        ),
        "prototype_method_disagreement": _at_least(
            row["prototype_method_disagreement"],
            policy.prototype_method_disagreement_threshold,
        ),
        "visual_input_disagreement": _at_least(
            row["visual_input_disagreement"],
            policy.visual_input_disagreement_threshold,
        ),
        "insufficient_local_support": _below(
            row["local_support_ratio"], policy.minimum_local_support_ratio
        ),
        "low_subject_area": _below(
            row["subject_area_ratio"], policy.minimum_subject_area_ratio
        ),
        "strong_known_competitor": _at_most(
            row["known_competitor_margin"],
            policy.known_competitor_margin_threshold,
        ),
        "no_geo_global_fallback": (
            row["no_geo_global_fallback"] is True
            and policy.expand_on_no_geo_global_fallback
        ),
        "out_of_distribution": _at_least(
            row["out_of_distribution_score"],
            policy.out_of_distribution_score_threshold,
        ),
        "route_domain_incompatible": (
            row["route_domain_compatible"] is False
            and policy.expand_on_route_domain_incompatibility
        ),
    }
    return sorted(signal for signal, triggered in checks.items() if triggered)


def _validate_materialized_evidence(
    row: Mapping[str, object],
    *,
    policy: DynamicPoolExpansionSignalPolicy | None,
) -> None:
    if row["schema_version"] != DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic pool expansion evidence schema")
    if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
        raise ValueError("dynamic pool expansion plan_id is invalid")
    for field in (
        "plan_fingerprint",
        "candidate_scores_fingerprint",
        "selection_policy_fingerprint",
        "signal_policy_fingerprint",
        "model_fingerprint",
        "evidence_fingerprint",
    ):
        _sha256(row[field], field=field)
    observed = list(row["observed_signals"])
    unavailable = list(row["unavailable_signals"])
    reasons = list(row["unavailable_signal_reasons"])
    triggered = list(row["triggered_signals"])
    if observed != sorted(observed) or unavailable != sorted(unavailable):
        raise ValueError("expansion signal availability is not canonical")
    if len(unavailable) != len(reasons):
        raise ValueError("unavailable expansion signals lack exact reasons")
    if set(observed).intersection(unavailable):
        raise ValueError("expansion signal cannot be observed and unavailable")
    if set(observed).union(unavailable) != set(DYNAMIC_POOL_EXPANSION_SIGNALS):
        raise ValueError("expansion signal coverage is incomplete")
    if not set(triggered) <= set(observed) or triggered != sorted(triggered):
        raise ValueError("triggered expansion signals are inconsistent")
    if bool(triggered) != row["expansion_required"]:
        raise ValueError("expansion requirement does not match triggers")
    identity = dict(row)
    fingerprint = identity.pop("evidence_fingerprint")
    if canonical_semantic_fingerprint(identity) != fingerprint:
        raise ValueError("dynamic pool expansion evidence fingerprint mismatch")
    if policy is not None:
        if row["signal_policy_version"] != policy.policy_version:
            raise ValueError("expansion signal policy version mismatch")
        if row["signal_policy_fingerprint"] != policy.fingerprint:
            raise ValueError("expansion signal policy fingerprint mismatch")
        source = {
            **{
                field: row[field]
                for field in (
                    "run_id",
                    "plan_id",
                    "plan_fingerprint",
                    "candidate_scores_fingerprint",
                    "selection_policy_fingerprint",
                    "model_fingerprint",
                    "expansion_round",
                    *_SIGNAL_VALUE_FIELDS.values(),
                )
            },
            "_reason_map": dict(zip(unavailable, reasons, strict=True)),
        }
        if _triggered_signals(source, policy=policy) != triggered:
            raise ValueError("expansion triggers do not match the supplied policy")


def _at_most(value: object, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def _at_least(value: object, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def _below(value: object, threshold: float) -> bool:
    return value is not None and float(value) < threshold


def _require_exact_fields(
    values: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    missing = expected - set(values)
    unexpected = set(values) - expected
    if missing or unexpected:
        raise ValueError(
            f"{label} fields mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded_float(value: object, *, field: str, maximum: float) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= maximum:
        raise ValueError(f"{field} must be within [0, {maximum:g}]")
    return result


def _optional_bounded_float(
    value: object, *, field: str, maximum: float
) -> float | None:
    if value is None:
        return None
    return _bounded_float(value, field=field, maximum=maximum)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be Boolean")
    return value


def _optional_boolean(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field=field)


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{field} must be within [0, {maximum}]")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


__all__ = [
    "DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_SIGNALS",
    "DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_VERSION",
    "DynamicPoolExpansionSignalPolicy",
    "build_dynamic_pool_expansion_evidence",
    "default_dynamic_pool_expansion_signal_policy",
    "dynamic_pool_expansion_cache_reuse_schema",
    "dynamic_pool_expansion_evidence_schema",
    "expand_dynamic_reference_pools_from_cache",
    "validate_dynamic_pool_expansion_cache_reuse",
    "validate_dynamic_pool_expansion_evidence",
]
