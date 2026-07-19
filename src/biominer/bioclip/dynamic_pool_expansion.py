"""Deterministic evidence and policy for dynamic reference-pool expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    build_dynamic_reference_pool_members,
    build_dynamic_reference_pool_plans,
    build_dynamic_reference_pool_summaries,
    dynamic_reference_pool_plan_id,
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
from biominer.storage.parquet import write_parquet


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
DYNAMIC_POOL_EXPANSION_DECISION_SCHEMA_VERSION = (
    "dynamic-pool-expansion-decision-v1.0.0"
)
DYNAMIC_POOL_EXPANSION_DECISIONS_FILE = "dynamic_pool_expansion_decisions.parquet"
DYNAMIC_POOL_EXPANSION_EVIDENCE_FILE = "dynamic_pool_expansion_evidence.parquet"
DYNAMIC_POOL_EXPANSION_CACHE_REUSE_FILE = (
    "dynamic_pool_expansion_cache_reuse.parquet"
)
DYNAMIC_POOL_EXPANSION_ACTIONS = frozenset({"expand", "stop"})
DYNAMIC_POOL_EXPANSION_STOP_REASONS = frozenset(
    {
        "round_complete_rescore_required",
        "signals_clear",
        "maximum_rounds_reached",
        "stage_budget_exhausted",
        "total_budget_exhausted",
        "no_cached_reference_additions",
    }
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
_DECISION_SORT = ("run_id", "prior_plan_id", "current_expansion_round")

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


def dynamic_pool_expansion_decision_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "prior_plan_id": pl.String,
        "prior_plan_fingerprint": pl.String,
        "expanded_plan_id": pl.String,
        "expanded_plan_fingerprint": pl.String,
        "expansion_evidence_fingerprint": pl.String,
        "selection_policy_version": pl.String,
        "selection_policy_fingerprint": pl.String,
        "signal_policy_version": pl.String,
        "signal_policy_fingerprint": pl.String,
        "current_expansion_round": pl.UInt16,
        "next_expansion_round": pl.UInt16,
        "maximum_expansion_rounds": pl.UInt16,
        "stage_member_limit": pl.UInt32,
        "maximum_total_reference_members": pl.UInt32,
        "prior_member_count": pl.UInt32,
        "remaining_stage_budget": pl.UInt32,
        "remaining_total_budget": pl.UInt32,
        "candidate_rank_limit": pl.UInt32,
        "per_candidate_increment": pl.UInt32,
        "triggered_signals": pl.List(pl.String),
        "eligible_candidate_accepted_taxon_keys": pl.List(pl.String),
        "added_candidate_accepted_taxon_keys": pl.List(pl.String),
        "added_candidate_reference_counts": pl.List(pl.UInt32),
        "added_reference_media_ids": pl.List(pl.String),
        "added_reference_observation_ids": pl.List(pl.String),
        "added_reference_embedding_fingerprints": pl.List(pl.String),
        "added_reference_count": pl.UInt32,
        "action": pl.String,
        "stop_reason": pl.String,
        "rescore_required": pl.Boolean,
        "production_release_authorized": pl.Boolean,
        "decision_fingerprint": pl.String,
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
    signal_policy: DynamicPoolExpansionSignalPolicy | None = None,
    burst_group_by_observation: Mapping[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Materialize triggered plans using cached embedding identities only.

    The function deliberately accepts no encoder and no embedding-vector column.
    It replaces an immutable plan with a larger plan over the same query and
    reference cache identities, while retaining every prior member.
    """

    if not isinstance(policy, DynamicReferencePoolPolicy):
        raise TypeError("policy must be a DynamicReferencePoolPolicy")
    active_signal_policy = (
        signal_policy or default_dynamic_pool_expansion_signal_policy()
    )
    if not isinstance(active_signal_policy, DynamicPoolExpansionSignalPolicy):
        raise TypeError("signal_policy must be a DynamicPoolExpansionSignalPolicy")
    if (
        active_signal_policy.species_margin_threshold
        != policy.uncertainty_margin_threshold
    ):
        raise ValueError(
            "expansion signal and selection policy species-margin thresholds differ"
        )
    validate_dynamic_reference_pool_artifacts(
        prior_plans, prior_members, prior_summaries
    )
    validate_dynamic_pool_expansion_evidence(
        expansion_evidence, policy=active_signal_policy
    )
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
    prior_lookup = {
        str(row["plan_id"]): row for row in prior_plans.iter_rows(named=True)
    }
    evidence_lookup = {
        str(row["plan_id"]): row for row in expansion_evidence.iter_rows(named=True)
    }
    preflight_by_id = {
        plan_id: _expansion_preflight(
            prior_lookup[plan_id],
            evidence=evidence,
            prior_members=prior_members.filter(pl.col("plan_id") == plan_id),
            candidate_sets=candidate_sets,
            policy=policy,
        )
        for plan_id, evidence in evidence_lookup.items()
    }
    selected_prior = [
        prior_lookup[plan_id]
        for plan_id in sorted(preflight_by_id)
        if preflight_by_id[plan_id]["actionable"]
    ]
    if selected_prior:
        requests = [_expansion_request(row) for row in selected_prior]
        planned_plans, planned_members, _planned_summaries = (
            plan_dynamic_reference_pools(
                requests,
                candidate_sets,
                reference_geography_index,
                global_reference_anchors,
                policy=policy,
                burst_group_by_observation=burst_group_by_observation,
            )
        )
    else:
        planned_plans = pl.DataFrame(schema=dynamic_reference_pool_plan_schema())
        planned_members = pl.DataFrame(schema=dynamic_reference_pool_member_schema())
    planned_lookup = {
        _plan_match_key(row): row for row in planned_plans.iter_rows(named=True)
    }
    member_inputs: list[dict[str, object]] = []
    plan_inputs: list[dict[str, object]] = []
    reuse_contexts: list[dict[str, object]] = []
    for prior in selected_prior:
        evidence = evidence_lookup[str(prior["plan_id"])]
        preflight = preflight_by_id[str(prior["plan_id"])]
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
        selected_additions = _bounded_addition_identities(
            expanded_group,
            prior_identities=set(prior_by_identity),
            eligible_candidates=set(preflight["eligible_candidates"]),
            addition_budget=min(
                int(preflight["remaining_stage_budget"]),
                int(preflight["remaining_total_budget"]),
            ),
            per_candidate_increment=policy.uncertainty_expansion_increment,
        )
        if not selected_additions:
            preflight["stop_reason"] = "no_cached_reference_additions"
            continue
        retained_and_added = set(prior_by_identity) | selected_additions
        expanded_by_identity = {
            identity: expanded_by_identity[identity]
            for identity in retained_and_added
        }
        next_round = int(evidence["expansion_round"]) + 1
        plan_input = _plan_input(expanded)
        local_available = any(
            row["pool_scope"] == "local" for row in expanded_by_identity.values()
        )
        plan_input["local_pool_status"] = (
            "available" if local_available else "unavailable"
        )
        plan_input["local_pool_unavailable_reason"] = (
            None
            if local_available
            else (
                "no_geo_global_fallback"
                if prior["local_pool_unavailable_reason"]
                == "no_geo_global_fallback"
                else "local_pool_not_selected_within_expansion_budget"
            )
        )
        plan_context = {
            field: value for field, value in plan_input.items() if field != "plan_id"
        }
        expanded_plan_id = dynamic_reference_pool_plan_id(plan_context)
        plan_input["plan_id"] = expanded_plan_id
        group_inputs: list[dict[str, object]] = []
        for identity, row in expanded_by_identity.items():
            item = _member_input(row)
            item["plan_id"] = expanded_plan_id
            item["expansion_round"] = (
                int(prior_by_identity[identity]["expansion_round"])
                if identity in prior_by_identity
                else next_round
            )
            group_inputs.append(item)
        _reset_pool_selection_ranks(group_inputs)
        member_inputs.extend(group_inputs)
        plan_inputs.append(plan_input)
        reuse_contexts.append(
            {
                "prior": prior,
                "expanded_plan_id": expanded_plan_id,
                "evidence": evidence,
                "prior_identities": set(prior_by_identity),
                "expanded_identities": set(expanded_by_identity),
                "selected_additions": selected_additions,
                "next_round": next_round,
            }
        )
        preflight["actual_context"] = reuse_contexts[-1]

    if member_inputs:
        members = build_dynamic_reference_pool_members(member_inputs)
        plans = build_dynamic_reference_pool_plans(plan_inputs, members)
        summaries = build_dynamic_reference_pool_summaries(plans, members)
        validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    else:
        plans = pl.DataFrame(schema=dynamic_reference_pool_plan_schema())
        members = pl.DataFrame(schema=dynamic_reference_pool_member_schema())
        summaries = pl.DataFrame(schema=dynamic_reference_pool_summary_schema())
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
    reuse = (
        pl.DataFrame(
            reuse_rows,
            schema=dynamic_pool_expansion_cache_reuse_schema(),
            orient="row",
            strict=True,
        ).sort(*_CACHE_REUSE_SORT)
        if reuse_rows
        else pl.DataFrame(schema=dynamic_pool_expansion_cache_reuse_schema())
    )
    validate_dynamic_pool_expansion_cache_reuse(reuse)
    decision_rows = [
        _expansion_decision_row(
            preflight_by_id[plan_id],
            expanded_plan=(
                plan_by_id[
                    str(preflight_by_id[plan_id]["actual_context"]["expanded_plan_id"])
                ]
                if "actual_context" in preflight_by_id[plan_id]
                else None
            ),
        )
        for plan_id in sorted(preflight_by_id)
    ]
    decisions = pl.DataFrame(
        decision_rows,
        schema=dynamic_pool_expansion_decision_schema(),
        orient="row",
        strict=True,
    ).sort(*_DECISION_SORT)
    validate_dynamic_pool_expansion_decisions(decisions)
    validate_dynamic_pool_expansion_execution(
        expansion_evidence, reuse, decisions
    )
    return plans, members, summaries, reuse, decisions


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


def validate_dynamic_pool_expansion_decisions(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("dynamic pool expansion decisions must be a Polars DataFrame")
    if frame.schema != dynamic_pool_expansion_decision_schema():
        raise ValueError("dynamic pool expansion decision schema mismatch")
    if not frame.equals(frame.sort(*_DECISION_SORT)):
        raise ValueError("dynamic pool expansion decisions are not canonically sorted")
    if frame.select("prior_plan_id", "current_expansion_round").n_unique() != frame.height:
        raise ValueError("dynamic pool expansion decision grain is not unique")
    for row in frame.iter_rows(named=True):
        if not _PLAN_ID_PATTERN.fullmatch(str(row["prior_plan_id"])):
            raise ValueError("expansion decision prior_plan_id is invalid")
        for field in (
            "prior_plan_fingerprint",
            "expansion_evidence_fingerprint",
            "selection_policy_fingerprint",
            "signal_policy_fingerprint",
            "decision_fingerprint",
        ):
            _sha256(row[field], field=field)
        action = row["action"]
        reason = row["stop_reason"]
        if action not in DYNAMIC_POOL_EXPANSION_ACTIONS:
            raise ValueError("unsupported dynamic pool expansion action")
        if reason not in DYNAMIC_POOL_EXPANSION_STOP_REASONS:
            raise ValueError("unsupported dynamic pool expansion stop reason")
        eligible = list(row["eligible_candidate_accepted_taxon_keys"])
        added_candidates = list(row["added_candidate_accepted_taxon_keys"])
        candidate_counts = list(row["added_candidate_reference_counts"])
        if eligible != sorted(set(eligible)):
            raise ValueError("eligible expansion candidates are not canonical")
        if added_candidates != sorted(set(added_candidates)):
            raise ValueError("added expansion candidates are not canonical")
        if not set(added_candidates) <= set(eligible):
            raise ValueError("added expansion candidate was not eligible")
        if len(added_candidates) != len(candidate_counts):
            raise ValueError("added candidate counts are incomplete")
        if any(
            count <= 0 or count > row["per_candidate_increment"]
            for count in candidate_counts
        ):
            raise ValueError("per-candidate expansion increment was exceeded")
        added_count = int(row["added_reference_count"])
        if sum(candidate_counts) != added_count:
            raise ValueError("added candidate counts do not match references")
        for field in (
            "added_reference_media_ids",
            "added_reference_observation_ids",
            "added_reference_embedding_fingerprints",
        ):
            values = list(row[field])
            if len(values) != added_count:
                raise ValueError(f"{field} count does not match expansion decision")
            if values != sorted(values):
                raise ValueError(f"{field} is not canonically sorted")
        for fingerprint in row["added_reference_embedding_fingerprints"]:
            _sha256(fingerprint, field="added_reference_embedding_fingerprints")
        if added_count > min(
            row["remaining_stage_budget"], row["remaining_total_budget"]
        ):
            raise ValueError("expansion exceeded its remaining pool budget")
        current_round = int(row["current_expansion_round"])
        next_round = int(row["next_expansion_round"])
        maximum_rounds = int(row["maximum_expansion_rounds"])
        if current_round > maximum_rounds:
            raise ValueError("expansion decision exceeds maximum round")
        if (
            int(row["prior_member_count"])
            + int(row["remaining_total_budget"])
            != int(row["maximum_total_reference_members"])
        ):
            raise ValueError("remaining total expansion budget is inconsistent")
        if action == "expand":
            for field in ("expanded_plan_id", "expanded_plan_fingerprint"):
                if row[field] is None:
                    raise ValueError("expanded decision lacks replacement plan identity")
            if not _PLAN_ID_PATTERN.fullmatch(str(row["expanded_plan_id"])):
                raise ValueError("expansion decision expanded_plan_id is invalid")
            _sha256(
                row["expanded_plan_fingerprint"],
                field="expanded_plan_fingerprint",
            )
            if (
                reason != "round_complete_rescore_required"
                or not row["rescore_required"]
                or added_count == 0
                or next_round != current_round + 1
                or current_round >= maximum_rounds
            ):
                raise ValueError("expanded decision rescore state is inconsistent")
        elif (
            row["expanded_plan_id"] is not None
            or row["expanded_plan_fingerprint"] is not None
            or row["rescore_required"]
            or added_count
            or next_round != current_round
            or reason == "round_complete_rescore_required"
        ):
            raise ValueError("stopped expansion decision is inconsistent")
        if action == "stop":
            triggered = list(row["triggered_signals"])
            expected_stop = {
                "signals_clear": not triggered,
                "maximum_rounds_reached": (
                    bool(triggered) and current_round >= maximum_rounds
                ),
                "stage_budget_exhausted": (
                    bool(triggered)
                    and current_round < maximum_rounds
                    and row["remaining_total_budget"] > 0
                    and row["remaining_stage_budget"] == 0
                ),
                "total_budget_exhausted": (
                    bool(triggered)
                    and current_round < maximum_rounds
                    and row["remaining_total_budget"] == 0
                ),
                "no_cached_reference_additions": (
                    bool(triggered)
                    and current_round < maximum_rounds
                    and row["remaining_stage_budget"] > 0
                    and row["remaining_total_budget"] > 0
                ),
            }
            if not expected_stop.get(str(reason), False):
                raise ValueError("expansion stop reason does not match evidence and budget")
        if row["production_release_authorized"]:
            raise ValueError("expansion decision cannot authorize production release")
        identity = dict(row)
        fingerprint = identity.pop("decision_fingerprint")
        if canonical_semantic_fingerprint(identity) != fingerprint:
            raise ValueError("dynamic pool expansion decision fingerprint mismatch")


def validate_dynamic_pool_expansion_execution(
    evidence: pl.DataFrame,
    cache_reuse: pl.DataFrame,
    decisions: pl.DataFrame,
) -> None:
    validate_dynamic_pool_expansion_evidence(evidence)
    validate_dynamic_pool_expansion_cache_reuse(cache_reuse)
    validate_dynamic_pool_expansion_decisions(decisions)
    evidence_by_fingerprint = {
        str(row["evidence_fingerprint"]): row
        for row in evidence.iter_rows(named=True)
    }
    decision_by_key = {
        (
            str(row["prior_plan_id"]),
            int(row["current_expansion_round"]),
        ): row
        for row in decisions.iter_rows(named=True)
    }
    if set(decisions["expansion_evidence_fingerprint"].to_list()) - set(
        evidence_by_fingerprint
    ):
        raise ValueError("expansion decision references unknown evidence")
    reuse_keys: set[tuple[str, int]] = set()
    for row in cache_reuse.iter_rows(named=True):
        key = (str(row["prior_plan_id"]), int(row["expansion_round"]) - 1)
        decision = decision_by_key.get(key)
        if decision is None or decision["action"] != "expand":
            raise ValueError("cache reuse lacks a matching expansion decision")
        if (
            row["expanded_plan_id"] != decision["expanded_plan_id"]
            or row["expanded_plan_fingerprint"]
            != decision["expanded_plan_fingerprint"]
            or row["expansion_evidence_fingerprint"]
            != decision["expansion_evidence_fingerprint"]
            or row["added_reference_count"] != decision["added_reference_count"]
        ):
            raise ValueError("cache reuse and expansion decision conflict")
        reuse_keys.add(key)
    expanded_keys = {
        key for key, row in decision_by_key.items() if row["action"] == "expand"
    }
    if reuse_keys != expanded_keys:
        raise ValueError("expanded decisions and cache reuse identities differ")
    for decision in decisions.iter_rows(named=True):
        evidence_row = evidence_by_fingerprint[
            str(decision["expansion_evidence_fingerprint"])
        ]
        if (
            decision["prior_plan_id"] != evidence_row["plan_id"]
            or decision["prior_plan_fingerprint"]
            != evidence_row["plan_fingerprint"]
            or decision["current_expansion_round"]
            != evidence_row["expansion_round"]
            or decision["triggered_signals"] != evidence_row["triggered_signals"]
        ):
            raise ValueError("expansion decision does not match its evidence")


def write_dynamic_pool_expansion_artifacts(
    evidence: pl.DataFrame,
    cache_reuse: pl.DataFrame,
    decisions: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_dynamic_pool_expansion_execution(evidence, cache_reuse, decisions)
    destination = Path(output_dir)
    return {
        "evidence": write_parquet(
            evidence, destination / DYNAMIC_POOL_EXPANSION_EVIDENCE_FILE
        ),
        "cache_reuse": write_parquet(
            cache_reuse, destination / DYNAMIC_POOL_EXPANSION_CACHE_REUSE_FILE
        ),
        "decisions": write_parquet(
            decisions, destination / DYNAMIC_POOL_EXPANSION_DECISIONS_FILE
        ),
    }


def _expansion_preflight(
    prior: Mapping[str, object],
    *,
    evidence: Mapping[str, object],
    prior_members: pl.DataFrame,
    candidate_sets: pl.DataFrame,
    policy: DynamicReferencePoolPolicy,
) -> dict[str, object]:
    stage_limit = min(
        dict(policy.stage_member_limits)["uncertainty_expansion"],
        policy.maximum_total_reference_members,
    )
    prior_count = prior_members.height
    remaining_stage = max(stage_limit - prior_count, 0)
    remaining_total = max(policy.maximum_total_reference_members - prior_count, 0)
    candidates = candidate_sets.filter(
        pl.col("candidate_set_id") == prior["candidate_set_id"]
    ).sort("candidate_priority", "candidate_accepted_taxon_key")
    ranked = candidates.head(policy.uncertainty_candidate_rank_limit)
    retained = candidates.filter(
        pl.col("target_candidate") | pl.col("safety_union_membership")
    )
    eligible = sorted(
        set(ranked["candidate_accepted_taxon_key"].to_list())
        | set(retained["candidate_accepted_taxon_key"].to_list())
    )
    current_round = int(evidence["expansion_round"])
    if not evidence["expansion_required"]:
        stop_reason = "signals_clear"
    elif current_round >= policy.maximum_expansion_rounds:
        stop_reason = "maximum_rounds_reached"
    elif remaining_total == 0:
        stop_reason = "total_budget_exhausted"
    elif remaining_stage == 0:
        stop_reason = "stage_budget_exhausted"
    else:
        stop_reason = None
    return {
        "prior": prior,
        "evidence": evidence,
        "actionable": stop_reason is None,
        "stop_reason": stop_reason,
        "eligible_candidates": eligible,
        "stage_limit": stage_limit,
        "prior_member_count": prior_count,
        "remaining_stage_budget": remaining_stage,
        "remaining_total_budget": remaining_total,
        "maximum_total_reference_members": policy.maximum_total_reference_members,
        "candidate_rank_limit": policy.uncertainty_candidate_rank_limit,
        "per_candidate_increment": policy.uncertainty_expansion_increment,
    }


def _bounded_addition_identities(
    expanded_members: pl.DataFrame,
    *,
    prior_identities: set[tuple[str, ...]],
    eligible_candidates: set[str],
    addition_budget: int,
    per_candidate_increment: int,
) -> set[tuple[str, ...]]:
    queues: dict[str, list[tuple[str, ...]]] = {
        candidate: [] for candidate in sorted(eligible_candidates)
    }
    for row in expanded_members.iter_rows(named=True):
        identity = _member_cache_identity(row)
        candidate = identity[0]
        if identity not in prior_identities and candidate in queues:
            queues[candidate].append(identity)
    selected: set[tuple[str, ...]] = set()
    counts = {candidate: 0 for candidate in queues}
    positions = {candidate: 0 for candidate in queues}
    while len(selected) < addition_budget:
        progress = False
        for candidate in sorted(queues):
            if len(selected) >= addition_budget:
                break
            if counts[candidate] >= per_candidate_increment:
                continue
            position = positions[candidate]
            if position >= len(queues[candidate]):
                continue
            selected.add(queues[candidate][position])
            positions[candidate] += 1
            counts[candidate] += 1
            progress = True
        if not progress:
            break
    return selected


def _expansion_decision_row(
    preflight: Mapping[str, object],
    *,
    expanded_plan: Mapping[str, object] | None,
) -> dict[str, object]:
    prior = preflight["prior"]
    evidence = preflight["evidence"]
    if not isinstance(prior, Mapping) or not isinstance(evidence, Mapping):
        raise TypeError("expansion decision context is invalid")
    actual = preflight.get("actual_context")
    selected: set[tuple[str, ...]] = set()
    if isinstance(actual, Mapping):
        raw_selected = actual["selected_additions"]
        if not isinstance(raw_selected, set):
            raise TypeError("selected expansion identities must be a set")
        selected = raw_selected
    candidate_counts: dict[str, int] = {}
    for identity in selected:
        candidate_counts[identity[0]] = candidate_counts.get(identity[0], 0) + 1
    added_candidates = sorted(candidate_counts)
    expanded = expanded_plan is not None
    current_round = int(evidence["expansion_round"])
    row: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_EXPANSION_DECISION_SCHEMA_VERSION,
        "run_id": prior["run_id"],
        "prior_plan_id": prior["plan_id"],
        "prior_plan_fingerprint": prior["plan_fingerprint"],
        "expanded_plan_id": expanded_plan["plan_id"] if expanded else None,
        "expanded_plan_fingerprint": (
            expanded_plan["plan_fingerprint"] if expanded else None
        ),
        "expansion_evidence_fingerprint": evidence["evidence_fingerprint"],
        "selection_policy_version": prior["selection_policy_version"],
        "selection_policy_fingerprint": prior["selection_policy_fingerprint"],
        "signal_policy_version": evidence["signal_policy_version"],
        "signal_policy_fingerprint": evidence["signal_policy_fingerprint"],
        "current_expansion_round": current_round,
        "next_expansion_round": current_round + 1 if expanded else current_round,
        "maximum_expansion_rounds": prior["maximum_expansion_rounds"],
        "stage_member_limit": preflight["stage_limit"],
        "maximum_total_reference_members": preflight[
            "maximum_total_reference_members"
        ],
        "prior_member_count": preflight["prior_member_count"],
        "remaining_stage_budget": preflight["remaining_stage_budget"],
        "remaining_total_budget": preflight["remaining_total_budget"],
        "candidate_rank_limit": preflight["candidate_rank_limit"],
        "per_candidate_increment": preflight["per_candidate_increment"],
        "triggered_signals": list(evidence["triggered_signals"]),
        "eligible_candidate_accepted_taxon_keys": list(
            preflight["eligible_candidates"]
        ),
        "added_candidate_accepted_taxon_keys": added_candidates,
        "added_candidate_reference_counts": [
            candidate_counts[candidate] for candidate in added_candidates
        ],
        "added_reference_media_ids": sorted(identity[1] for identity in selected),
        "added_reference_observation_ids": sorted(
            identity[2] for identity in selected
        ),
        "added_reference_embedding_fingerprints": sorted(
            identity[3] for identity in selected
        ),
        "added_reference_count": len(selected),
        "action": "expand" if expanded else "stop",
        "stop_reason": (
            "round_complete_rescore_required"
            if expanded
            else preflight["stop_reason"]
        ),
        "rescore_required": expanded,
        "production_release_authorized": False,
    }
    row["decision_fingerprint"] = canonical_semantic_fingerprint(row)
    return row


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
    "DYNAMIC_POOL_EXPANSION_CACHE_REUSE_FILE",
    "DYNAMIC_POOL_EXPANSION_DECISIONS_FILE",
    "DYNAMIC_POOL_EXPANSION_DECISION_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_EVIDENCE_FILE",
    "DYNAMIC_POOL_EXPANSION_SIGNALS",
    "DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_SCHEMA_VERSION",
    "DYNAMIC_POOL_EXPANSION_SIGNAL_POLICY_VERSION",
    "DynamicPoolExpansionSignalPolicy",
    "build_dynamic_pool_expansion_evidence",
    "default_dynamic_pool_expansion_signal_policy",
    "dynamic_pool_expansion_cache_reuse_schema",
    "dynamic_pool_expansion_decision_schema",
    "dynamic_pool_expansion_evidence_schema",
    "expand_dynamic_reference_pools_from_cache",
    "validate_dynamic_pool_expansion_cache_reuse",
    "validate_dynamic_pool_expansion_decisions",
    "validate_dynamic_pool_expansion_evidence",
    "validate_dynamic_pool_expansion_execution",
    "write_dynamic_pool_expansion_artifacts",
]
