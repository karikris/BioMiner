"""Sampling and review contracts for dynamic-pool Flickr candidates.

The audit frame describes provisional model evidence.  Its taxonomic fields
are candidate strata, never reviewed identity, and its raw scores are never
probabilities.  Representative sampling, targeted failure discovery and
release review are separate downstream contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION = "dynamic-pool-audit-frame-v1.0.0"
DYNAMIC_POOL_AUDIT_FRAME_FILE = "dynamic_pool_audit_frame.parquet"
DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION = (
    "dynamic-pool-probability-register-v1.0.0"
)
DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA_VERSION = (
    "dynamic-pool-probability-sample-v1.0.0"
)
DYNAMIC_POOL_PROBABILITY_REGISTER_FILE = (
    "dynamic_pool_probability_audit_register.parquet"
)
DYNAMIC_POOL_PROBABILITY_SAMPLE_FILE = "dynamic_pool_probability_audit_sample.parquet"
DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA_VERSION = "dynamic-pool-failure-queue-v1.0.0"
DYNAMIC_POOL_FAILURE_QUEUE_FILE = "dynamic_pool_failure_discovery_queue.parquet"

QUERY_TIERS = frozenset({"T1", "T2", "T3", "T4", "T5"})
RAW_SCORE_SEMANTICS = "raw_model_evidence_not_probability"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "sampling_unit_id",
        "source_record_hash",
        "source_artifact_fingerprint",
        "flickr_photo_id",
        "organism_unit_id",
        "candidate_family_accepted_taxon_key",
        "candidate_family_scientific_name",
        "candidate_genus_accepted_taxon_key",
        "candidate_genus_scientific_name",
        "candidate_species_accepted_taxon_key",
        "candidate_species_scientific_name",
        "geographic_cluster_id",
        "no_geo",
        "primary_query_tier",
        "raw_fusion_score",
        "raw_competitor_margin",
        "pool_disagreement",
        "route",
        "visual_domain",
        "subject_area_ratio",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
        "final_release_candidate",
    }
)

DYNAMIC_POOL_AUDIT_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "strata_policy_fingerprint": pl.String,
    "frame_fingerprint": pl.String,
    "audit_unit_fingerprint": pl.String,
    "sampling_unit_id": pl.String,
    "source_record_hash": pl.String,
    "source_artifact_fingerprint": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "candidate_family_accepted_taxon_key": pl.String,
    "candidate_family_scientific_name": pl.String,
    "candidate_genus_accepted_taxon_key": pl.String,
    "candidate_genus_scientific_name": pl.String,
    "candidate_species_accepted_taxon_key": pl.String,
    "candidate_species_scientific_name": pl.String,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "geography_stratum": pl.String,
    "primary_query_tier": pl.String,
    "raw_fusion_score": pl.Float64,
    "raw_score_band": pl.String,
    "raw_competitor_margin": pl.Float64,
    "raw_margin_band": pl.String,
    "pool_disagreement": pl.Float64,
    "pool_disagreement_band": pl.String,
    "route": pl.String,
    "visual_domain": pl.String,
    "route_domain_stratum": pl.String,
    "subject_area_ratio": pl.Float64,
    "subject_size_band": pl.String,
    "owner_group_id": pl.String,
    "duplicate_group_id": pl.String,
    "observation_group_id": pl.String,
    "independence_group_fingerprint": pl.String,
    "analysis_stratum_id": pl.String,
    "analysis_stratum_fingerprint": pl.String,
    "score_semantics": pl.String,
    "probability_available": pl.Boolean,
    "final_release_candidate": pl.Boolean,
}

_PROBABILITY_DESIGN_FIELDS: dict[str, pl.DataType] = {
    "sample_policy_fingerprint": pl.String,
    "sampling_register_fingerprint": pl.String,
    "sampling_population_unit_id": pl.String,
    "sampling_population_member_count": pl.UInt32,
    "sampling_population_member_unit_ids": pl.List(pl.String),
    "sampling_population_owner_group_ids": pl.List(pl.String),
    "sampling_population_duplicate_group_ids": pl.List(pl.String),
    "sampling_population_observation_group_ids": pl.List(pl.String),
    "member_analysis_stratum_ids": pl.List(pl.String),
    "component_crosses_analysis_strata": pl.Boolean,
    "selection_hash": pl.String,
    "selection_rank": pl.UInt32,
    "stratum_population_count": pl.UInt32,
    "stratum_sample_count": pl.UInt32,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "sampling_design": pl.String,
    "representative_estimation_eligible": pl.Boolean,
    "variance_cluster_id": pl.String,
}

DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    **_PROBABILITY_DESIGN_FIELDS,
    "representative_audit_unit_fingerprint": pl.String,
    "representative_sampling_unit_id": pl.String,
    "analysis_stratum_id": pl.String,
    "analysis_stratum_fingerprint": pl.String,
    "selected": pl.Boolean,
}

DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    **DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
    "probability_sample_schema_version": pl.String,
    **_PROBABILITY_DESIGN_FIELDS,
}

DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    **DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
    "failure_queue_schema_version": pl.String,
    "failure_policy_fingerprint": pl.String,
    "failure_queue_fingerprint": pl.String,
    "queue_kind": pl.String,
    "priority_rank": pl.UInt32,
    "priority_score": pl.Float64,
    "priority_score_semantics": pl.String,
    "priority_reasons": pl.List(pl.String),
    "targeted_component_member_count": pl.UInt32,
    "targeted_component_member_unit_ids": pl.List(pl.String),
    "targeted_component_owner_group_ids": pl.List(pl.String),
    "targeted_component_duplicate_group_ids": pl.List(pl.String),
    "targeted_component_observation_group_ids": pl.List(pl.String),
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "representative_estimation_eligible": pl.Boolean,
    "review_status": pl.String,
    "review_required": pl.Boolean,
    "release_authorized": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolAuditStrataPolicy:
    """Immutable cut points for audit analysis, not decision thresholds."""

    schema_version: str = "dynamic-pool-audit-strata-policy-v1.0.0"
    score_cutpoints: tuple[float, ...] = (0.25, 0.50, 0.75)
    margin_cutpoints: tuple[float, ...] = (0.0, 0.05, 0.15)
    pool_disagreement_cutpoints: tuple[float, ...] = (0.05, 0.15)
    subject_area_cutpoints: tuple[float, ...] = (0.02, 0.10, 0.30)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, field="schema_version"),
        )
        for field in (
            "score_cutpoints",
            "margin_cutpoints",
            "pool_disagreement_cutpoints",
            "subject_area_cutpoints",
        ):
            values = _cutpoints(getattr(self, field), field=field)
            if field in {"pool_disagreement_cutpoints", "subject_area_cutpoints"} and (
                values and values[0] < 0.0
            ):
                raise ValueError(f"{field} cannot contain negative values")
            if field == "subject_area_cutpoints" and values and values[-1] > 1.0:
                raise ValueError("subject_area_cutpoints cannot exceed one")
            object.__setattr__(self, field, values)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "score_cutpoints": self.score_cutpoints,
                "margin_cutpoints": self.margin_cutpoints,
                "pool_disagreement_cutpoints": self.pool_disagreement_cutpoints,
                "subject_area_cutpoints": self.subject_area_cutpoints,
            }
        )


@dataclass(frozen=True, slots=True)
class ProbabilityAuditSamplingPolicy:
    """Deterministic allocation for a stratified probability audit."""

    review_budget: int
    schema_version: str = "probability-audit-sampling-policy-v1.0.0"
    minimum_per_nonempty_stratum: int = 1
    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, field="schema_version"),
        )
        for field in ("review_budget", "minimum_per_nonempty_stratum"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or not 0 <= self.random_seed <= 2**64 - 1
        ):
            raise ValueError("random_seed must be an unsigned 64-bit integer")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "review_budget": self.review_budget,
                "minimum_per_nonempty_stratum": self.minimum_per_nonempty_stratum,
                "random_seed": self.random_seed,
                "target_population": (
                    "one_deterministic_representative_per_connected_"
                    "duplicate_observation_component"
                ),
                "allocation": (
                    "minimum_then_proportional_largest_remainder_by_analysis_stratum"
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class FailureDiscoveryPolicy:
    """Versioned heuristic for targeted review, never a risk model."""

    schema_version: str = "failure-discovery-policy-v1.0.0"
    near_margin_cutoff: float = 0.05
    high_disagreement_cutoff: float = 0.15
    low_score_cutoff: float = 0.50
    small_subject_cutoff: float = 0.10
    priority_route_domains: tuple[str, ...] = ()
    max_queue_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, field="schema_version"),
        )
        for field in (
            "near_margin_cutoff",
            "high_disagreement_cutoff",
            "low_score_cutoff",
            "small_subject_cutoff",
        ):
            value = _finite_float(getattr(self, field), field=field)
            if field != "low_score_cutoff" and value < 0.0:
                raise ValueError(f"{field} cannot be negative")
            object.__setattr__(self, field, value)
        if not 0.0 <= self.small_subject_cutoff <= 1.0:
            raise ValueError("small_subject_cutoff must be within [0, 1]")
        routes = tuple(
            sorted(
                {
                    _required_text(value, field="priority_route_domains")
                    for value in self.priority_route_domains
                }
            )
        )
        object.__setattr__(self, "priority_route_domains", routes)
        if self.max_queue_size is not None and (
            not isinstance(self.max_queue_size, int)
            or isinstance(self.max_queue_size, bool)
            or self.max_queue_size < 1
        ):
            raise ValueError("max_queue_size must be a positive integer or null")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "near_margin_cutoff": self.near_margin_cutoff,
                "high_disagreement_cutoff": self.high_disagreement_cutoff,
                "low_score_cutoff": self.low_score_cutoff,
                "small_subject_cutoff": self.small_subject_cutoff,
                "priority_route_domains": self.priority_route_domains,
                "max_queue_size": self.max_queue_size,
                "priority_score_semantics": "heuristic_not_probability",
            }
        )


@dataclass(frozen=True, slots=True)
class ProbabilityAuditSelection:
    """All-unit design register plus the selected review rows."""

    register_fingerprint: str
    population_count: int
    selected_count: int
    register: pl.DataFrame
    sample: pl.DataFrame


def empty_dynamic_pool_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DYNAMIC_POOL_AUDIT_FRAME_SCHEMA)


def build_dynamic_pool_audit_frame(
    candidates: Sequence[Mapping[str, object]],
    *,
    policy: DynamicPoolAuditStrataPolicy | None = None,
) -> pl.DataFrame:
    """Build deterministic audit strata without creating reviewed labels."""

    selected_policy = policy or DynamicPoolAuditStrataPolicy()
    if not isinstance(selected_policy, DynamicPoolAuditStrataPolicy):
        raise TypeError("policy must be a DynamicPoolAuditStrataPolicy")
    normalized = [
        _normalize_candidate(candidate, policy=selected_policy)
        for candidate in candidates
    ]
    unit_ids = [row["sampling_unit_id"] for row in normalized]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("sampling_unit_id must be unique")
    normalized.sort(key=lambda row: str(row["sampling_unit_id"]))
    frame_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "strata_policy_fingerprint": selected_policy.fingerprint,
            "audit_unit_fingerprints": [
                row["audit_unit_fingerprint"] for row in normalized
            ],
        }
    )
    for row in normalized:
        row["frame_fingerprint"] = frame_fingerprint
    frame = pl.DataFrame(
        normalized,
        schema=DYNAMIC_POOL_AUDIT_FRAME_SCHEMA,
        strict=True,
    )
    validate_dynamic_pool_audit_frame(frame)
    return frame


def validate_dynamic_pool_audit_frame(frame: pl.DataFrame) -> None:
    """Validate schema, identities and semantic fingerprints fail closed."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if frame.schema != DYNAMIC_POOL_AUDIT_FRAME_SCHEMA:
        raise ValueError("dynamic-pool audit frame schema does not match contract")
    if not frame.height:
        return
    if set(frame["schema_version"].to_list()) != {
        DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION
    }:
        raise ValueError("unsupported dynamic-pool audit frame schema version")
    if frame["sampling_unit_id"].n_unique() != frame.height:
        raise ValueError("sampling_unit_id must be unique")
    if frame.filter(
        pl.any_horizontal(
            pl.col(field).is_null() | (pl.col(field).str.strip_chars() == "")
            for field in (
                "owner_group_id",
                "duplicate_group_id",
                "observation_group_id",
            )
        )
    ).height:
        raise ValueError("owner, duplicate and observation groups must be complete")
    if frame.filter(pl.col("score_semantics") != RAW_SCORE_SEMANTICS).height:
        raise ValueError("raw score semantics must remain explicit")
    if frame.filter(pl.col("probability_available")).height:
        raise ValueError("raw audit scores cannot be marked as probabilities")
    fingerprints = set(frame["frame_fingerprint"].to_list())
    policy_fingerprints = set(frame["strata_policy_fingerprint"].to_list())
    if len(fingerprints) != 1 or len(policy_fingerprints) != 1:
        raise ValueError("audit frame must have one frame and policy fingerprint")
    rows = frame.sort("sampling_unit_id").to_dicts()
    expected = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "strata_policy_fingerprint": next(iter(policy_fingerprints)),
            "audit_unit_fingerprints": [row["audit_unit_fingerprint"] for row in rows],
        }
    )
    if fingerprints != {expected}:
        raise ValueError("dynamic-pool audit frame fingerprint mismatch")


def build_probability_audit_sample(
    frame: pl.DataFrame,
    *,
    policy: ProbabilityAuditSamplingPolicy,
) -> ProbabilityAuditSelection:
    """Select a stratified probability sample with exact design weights.

    Duplicate and observation identities define connected components.  One
    deterministic representative per component is the audit target population;
    owner identities remain variance clusters rather than being treated as
    independent observations.
    """

    validate_dynamic_pool_audit_frame(frame)
    if not isinstance(policy, ProbabilityAuditSamplingPolicy):
        raise TypeError("policy must be a ProbabilityAuditSamplingPolicy")
    if not frame.height:
        fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION,
                "sample_policy_fingerprint": policy.fingerprint,
                "audit_frame_fingerprint": None,
                "population_units": [],
            }
        )
        return ProbabilityAuditSelection(
            register_fingerprint=fingerprint,
            population_count=0,
            selected_count=0,
            register=pl.DataFrame(schema=DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA),
            sample=pl.DataFrame(schema=DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA),
        )

    frame_rows = frame.sort("sampling_unit_id").to_dicts()
    population_units = _build_sampling_population(frame_rows, policy=policy)
    stratum_sizes: dict[str, int] = {}
    for unit in population_units:
        stratum = str(unit["analysis_stratum_id"])
        stratum_sizes[stratum] = stratum_sizes.get(stratum, 0) + 1
    allocations = _allocate_stratified_sample(
        stratum_sizes,
        review_budget=policy.review_budget,
        minimum_per_stratum=policy.minimum_per_nonempty_stratum,
    )
    units_by_stratum: dict[str, list[dict[str, object]]] = {
        stratum: [] for stratum in stratum_sizes
    }
    for unit in population_units:
        units_by_stratum[str(unit["analysis_stratum_id"])].append(unit)
    for stratum, units in units_by_stratum.items():
        units.sort(
            key=lambda unit: (
                str(unit["selection_hash"]),
                str(unit["sampling_population_unit_id"]),
            )
        )
        population_count = len(units)
        sample_count = allocations[stratum]
        inclusion_probability = sample_count / population_count
        for rank, unit in enumerate(units, start=1):
            unit.update(
                {
                    "selection_rank": rank,
                    "stratum_population_count": population_count,
                    "stratum_sample_count": sample_count,
                    "inclusion_probability": inclusion_probability,
                    "sampling_weight": 1.0 / inclusion_probability,
                    "selected": rank <= sample_count,
                }
            )
    population_units.sort(key=lambda unit: str(unit["sampling_population_unit_id"]))
    register_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION,
            "sample_policy_fingerprint": policy.fingerprint,
            "audit_frame_fingerprint": frame["frame_fingerprint"].item(0),
            "population_units": [
                {
                    key: value
                    for key, value in unit.items()
                    if key != "representative_row"
                }
                for unit in population_units
            ],
        }
    )
    register_rows = [
        _probability_register_row(
            unit,
            policy=policy,
            register_fingerprint=register_fingerprint,
        )
        for unit in population_units
    ]
    register = pl.DataFrame(
        register_rows,
        schema=DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA,
        strict=True,
    ).sort("sampling_population_unit_id")
    sample_rows: list[dict[str, object]] = []
    for unit in population_units:
        if not unit["selected"]:
            continue
        representative = dict(unit["representative_row"])
        design = _probability_design_values(
            unit,
            policy=policy,
            register_fingerprint=register_fingerprint,
        )
        sample_rows.append(
            {
                **representative,
                "probability_sample_schema_version": (
                    DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA_VERSION
                ),
                **design,
            }
        )
    sample = pl.DataFrame(
        sample_rows,
        schema=DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA,
        strict=True,
    ).sort(["analysis_stratum_id", "selection_rank", "sampling_unit_id"])
    validate_probability_audit_selection(register, sample)
    return ProbabilityAuditSelection(
        register_fingerprint=register_fingerprint,
        population_count=register.height,
        selected_count=sample.height,
        register=register,
        sample=sample,
    )


def validate_probability_audit_selection(
    register: pl.DataFrame,
    sample: pl.DataFrame,
) -> None:
    """Validate the published first-order selection design."""

    if register.schema != DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA:
        raise ValueError("probability audit register schema does not match contract")
    if sample.schema != DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA:
        raise ValueError("probability audit sample schema does not match contract")
    if not register.height:
        if sample.height:
            raise ValueError("empty probability register cannot have sample rows")
        return
    if set(register["schema_version"].to_list()) != {
        DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION
    }:
        raise ValueError("unsupported probability audit register schema version")
    if register["sampling_population_unit_id"].n_unique() != register.height:
        raise ValueError("probability population unit IDs must be unique")
    invalid = register.filter(
        ~pl.col("inclusion_probability").is_between(0.0, 1.0, closed="right")
        | (
            (pl.col("sampling_weight") * pl.col("inclusion_probability") - 1.0).abs()
            > 1e-12
        )
        | (pl.col("stratum_sample_count") > pl.col("stratum_population_count"))
    )
    if invalid.height:
        raise ValueError("probability audit inclusion probabilities are invalid")
    selected = register.filter(pl.col("selected"))
    if selected.height != sample.height:
        raise ValueError("probability register and sample selection counts differ")
    if set(selected["representative_sampling_unit_id"].to_list()) != set(
        sample["sampling_unit_id"].to_list()
    ):
        raise ValueError("probability sample is not the selected representative set")
    for stratum in register["analysis_stratum_id"].unique().to_list():
        group = register.filter(pl.col("analysis_stratum_id") == stratum)
        if group.filter(pl.col("selected")).height != group[
            "stratum_sample_count"
        ].item(0):
            raise ValueError("probability sample count does not match stratum design")


def build_failure_discovery_queue(
    frame: pl.DataFrame,
    *,
    policy: FailureDiscoveryPolicy | None = None,
) -> pl.DataFrame:
    """Prioritize explicit failure signals outside the probability sample."""

    validate_dynamic_pool_audit_frame(frame)
    selected_policy = policy or FailureDiscoveryPolicy()
    if not isinstance(selected_policy, FailureDiscoveryPolicy):
        raise TypeError("policy must be a FailureDiscoveryPolicy")
    if not frame.height:
        return pl.DataFrame(schema=DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA)
    queue_rows: list[dict[str, object]] = []
    components = _connected_identity_components(
        frame.sort("sampling_unit_id").to_dicts()
    )
    for members in components:
        prioritized = []
        for row in members:
            score, reasons = _failure_priority(row, policy=selected_policy)
            if reasons:
                prioritized.append((score, reasons, row))
        if not prioritized:
            continue
        prioritized.sort(
            key=lambda item: (
                -item[0],
                -len(item[1]),
                str(item[2]["audit_unit_fingerprint"]),
            )
        )
        score, reasons, representative = prioritized[0]
        queue_rows.append(
            {
                **representative,
                "failure_queue_schema_version": (
                    DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA_VERSION
                ),
                "failure_policy_fingerprint": selected_policy.fingerprint,
                "failure_queue_fingerprint": "",
                "queue_kind": "targeted_failure_discovery",
                "priority_rank": 0,
                "priority_score": score,
                "priority_score_semantics": "heuristic_not_probability",
                "priority_reasons": reasons,
                "targeted_component_member_count": len(members),
                "targeted_component_member_unit_ids": sorted(
                    str(row["sampling_unit_id"]) for row in members
                ),
                "targeted_component_owner_group_ids": sorted(
                    {str(row["owner_group_id"]) for row in members}
                ),
                "targeted_component_duplicate_group_ids": sorted(
                    {str(row["duplicate_group_id"]) for row in members}
                ),
                "targeted_component_observation_group_ids": sorted(
                    {str(row["observation_group_id"]) for row in members}
                ),
                "inclusion_probability": None,
                "sampling_weight": None,
                "representative_estimation_eligible": False,
                "review_status": "pending",
                "review_required": True,
                "release_authorized": False,
            }
        )
    queue_rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -len(row["priority_reasons"]),
            str(row["audit_unit_fingerprint"]),
        )
    )
    if selected_policy.max_queue_size is not None:
        queue_rows = queue_rows[: selected_policy.max_queue_size]
    queue_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA_VERSION,
            "failure_policy_fingerprint": selected_policy.fingerprint,
            "audit_frame_fingerprint": frame["frame_fingerprint"].item(0),
            "queue_rows": [
                {
                    "audit_unit_fingerprint": row["audit_unit_fingerprint"],
                    "priority_score": row["priority_score"],
                    "priority_reasons": row["priority_reasons"],
                    "member_unit_ids": row["targeted_component_member_unit_ids"],
                }
                for row in queue_rows
            ],
        }
    )
    for rank, row in enumerate(queue_rows, start=1):
        row["failure_queue_fingerprint"] = queue_fingerprint
        row["priority_rank"] = rank
    queue = pl.DataFrame(
        queue_rows,
        schema=DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA,
        strict=True,
    )
    validate_failure_discovery_queue(queue)
    return queue


def validate_failure_discovery_queue(queue: pl.DataFrame) -> None:
    """Prevent targeted review from masquerading as representative evidence."""

    if queue.schema != DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA:
        raise ValueError("failure-discovery queue schema does not match contract")
    if not queue.height:
        return
    if set(queue["failure_queue_schema_version"].to_list()) != {
        DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA_VERSION
    }:
        raise ValueError("unsupported failure-discovery queue schema version")
    if queue.filter(
        (pl.col("queue_kind") != "targeted_failure_discovery")
        | (pl.col("priority_score_semantics") != "heuristic_not_probability")
        | pl.col("inclusion_probability").is_not_null()
        | pl.col("sampling_weight").is_not_null()
        | pl.col("representative_estimation_eligible")
        | ~pl.col("review_required")
        | pl.col("release_authorized")
    ).height:
        raise ValueError("targeted failure queue crossed its evidence boundary")
    if queue["audit_unit_fingerprint"].n_unique() != queue.height:
        raise ValueError("targeted failure queue representatives must be unique")
    expected_ranks = list(range(1, queue.height + 1))
    if queue["priority_rank"].to_list() != expected_ranks:
        raise ValueError("targeted failure queue ranks must be contiguous")


def _failure_priority(
    row: Mapping[str, object],
    *,
    policy: FailureDiscoveryPolicy,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    margin = float(row["raw_competitor_margin"])
    if margin <= 0.0:
        reasons.append("nonpositive_competitor_margin")
        score += 4.0
    elif margin < policy.near_margin_cutoff:
        reasons.append("near_competitor_margin")
        score += 3.0
    disagreement = row["pool_disagreement"]
    if disagreement is None:
        reasons.append("pool_disagreement_unavailable")
        score += 1.0
    elif float(disagreement) >= policy.high_disagreement_cutoff:
        reasons.append("high_pool_disagreement")
        score += 3.0
    if float(row["raw_fusion_score"]) < policy.low_score_cutoff:
        reasons.append("low_raw_score")
        score += 2.0
    if float(row["subject_area_ratio"]) < policy.small_subject_cutoff:
        reasons.append("small_subject")
        score += 2.0
    if bool(row["no_geo"]):
        reasons.append("no_geo")
        score += 1.0
    if str(row["route_domain_stratum"]) in policy.priority_route_domains:
        reasons.append("priority_route_domain")
        score += 1.0
    return score, reasons


def _build_sampling_population(
    rows: list[dict[str, object]],
    *,
    policy: ProbabilityAuditSamplingPolicy,
) -> list[dict[str, object]]:
    components = _connected_identity_components(rows)
    population: list[dict[str, object]] = []
    for members in components:
        members.sort(key=lambda row: str(row["sampling_unit_id"]))
        representative = min(
            members,
            key=lambda row: canonical_semantic_fingerprint(
                {
                    "role": "probability_population_representative",
                    "random_seed": policy.random_seed,
                    "audit_unit_fingerprint": row["audit_unit_fingerprint"],
                }
            ),
        )
        member_unit_ids = sorted(str(row["sampling_unit_id"]) for row in members)
        population_unit_fingerprint = canonical_semantic_fingerprint(
            {
                "duplicate_group_ids": sorted(
                    {str(row["duplicate_group_id"]) for row in members}
                ),
                "observation_group_ids": sorted(
                    {str(row["observation_group_id"]) for row in members}
                ),
                "member_audit_unit_fingerprints": sorted(
                    str(row["audit_unit_fingerprint"]) for row in members
                ),
            }
        )
        population_unit_id = (
            "dynamic-pool-review-population-unit:"
            f"{population_unit_fingerprint.removeprefix('sha256:')}"
        )
        member_strata = sorted({str(row["analysis_stratum_id"]) for row in members})
        population.append(
            {
                "sampling_population_unit_id": population_unit_id,
                "sampling_population_member_count": len(members),
                "sampling_population_member_unit_ids": member_unit_ids,
                "sampling_population_owner_group_ids": sorted(
                    {str(row["owner_group_id"]) for row in members}
                ),
                "sampling_population_duplicate_group_ids": sorted(
                    {str(row["duplicate_group_id"]) for row in members}
                ),
                "sampling_population_observation_group_ids": sorted(
                    {str(row["observation_group_id"]) for row in members}
                ),
                "member_analysis_stratum_ids": member_strata,
                "component_crosses_analysis_strata": len(member_strata) > 1,
                "representative_audit_unit_fingerprint": representative[
                    "audit_unit_fingerprint"
                ],
                "representative_sampling_unit_id": representative["sampling_unit_id"],
                "representative_row": representative,
                "analysis_stratum_id": representative["analysis_stratum_id"],
                "analysis_stratum_fingerprint": representative[
                    "analysis_stratum_fingerprint"
                ],
                "selection_hash": canonical_semantic_fingerprint(
                    {
                        "role": "probability_audit_selection",
                        "random_seed": policy.random_seed,
                        "sampling_population_unit_id": population_unit_id,
                    }
                ),
                "sampling_design": (
                    "stratified_srs_without_replacement_of_connected_"
                    "duplicate_observation_components"
                ),
                "representative_estimation_eligible": True,
                "variance_cluster_id": str(representative["owner_group_id"]),
            }
        )
    return population


def _connected_identity_components(
    rows: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identities: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for field in ("duplicate_group_id", "observation_group_id"):
            identity = (field, str(row[field]))
            previous = identities.setdefault(identity, index)
            union(index, previous)
    grouped: dict[int, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    components = [
        sorted(members, key=lambda row: str(row["sampling_unit_id"]))
        for members in grouped.values()
    ]
    components.sort(key=lambda members: str(members[0]["sampling_unit_id"]))
    return components


def _allocate_stratified_sample(
    stratum_sizes: Mapping[str, int],
    *,
    review_budget: int,
    minimum_per_stratum: int,
) -> dict[str, int]:
    allocation = {
        stratum: min(minimum_per_stratum, size)
        for stratum, size in stratum_sizes.items()
    }
    population_count = sum(stratum_sizes.values())
    effective_budget = min(review_budget, population_count)
    required_minimum = sum(allocation.values())
    if effective_budget < required_minimum:
        raise ValueError(
            "review_budget cannot represent every nonempty audit stratum at the "
            "configured minimum"
        )
    remaining = effective_budget - required_minimum
    capacities = {
        stratum: stratum_sizes[stratum] - allocation[stratum]
        for stratum in stratum_sizes
    }
    capacity_total = sum(capacities.values())
    if not remaining or not capacity_total:
        return allocation
    exact = {
        stratum: remaining * capacity / capacity_total
        for stratum, capacity in capacities.items()
    }
    floors = {stratum: math.floor(value) for stratum, value in exact.items()}
    for stratum, count in floors.items():
        allocation[stratum] += count
    leftover = remaining - sum(floors.values())
    priority = sorted(
        stratum_sizes,
        key=lambda stratum: (
            -(exact[stratum] - floors[stratum]),
            stratum,
        ),
    )
    for stratum in priority:
        if not leftover:
            break
        if allocation[stratum] < stratum_sizes[stratum]:
            allocation[stratum] += 1
            leftover -= 1
    if leftover:
        raise RuntimeError("stratified allocation did not exhaust the review budget")
    return allocation


def _probability_design_values(
    unit: Mapping[str, object],
    *,
    policy: ProbabilityAuditSamplingPolicy,
    register_fingerprint: str,
) -> dict[str, object]:
    return {
        "sample_policy_fingerprint": policy.fingerprint,
        "sampling_register_fingerprint": register_fingerprint,
        **{
            field: unit[field]
            for field in _PROBABILITY_DESIGN_FIELDS
            if field
            not in {"sample_policy_fingerprint", "sampling_register_fingerprint"}
        },
    }


def _probability_register_row(
    unit: Mapping[str, object],
    *,
    policy: ProbabilityAuditSamplingPolicy,
    register_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION,
        **_probability_design_values(
            unit,
            policy=policy,
            register_fingerprint=register_fingerprint,
        ),
        "representative_audit_unit_fingerprint": unit[
            "representative_audit_unit_fingerprint"
        ],
        "representative_sampling_unit_id": unit["representative_sampling_unit_id"],
        "analysis_stratum_id": unit["analysis_stratum_id"],
        "analysis_stratum_fingerprint": unit["analysis_stratum_fingerprint"],
        "selected": unit["selected"],
    }


def _normalize_candidate(
    candidate: Mapping[str, object],
    *,
    policy: DynamicPoolAuditStrataPolicy,
) -> dict[str, object]:
    if not isinstance(candidate, Mapping):
        raise TypeError("each candidate must be a mapping")
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(candidate))
    if missing:
        raise ValueError(f"dynamic-pool audit candidate missing fields: {missing}")
    text_fields = (
        "sampling_unit_id",
        "flickr_photo_id",
        "organism_unit_id",
        "candidate_family_accepted_taxon_key",
        "candidate_family_scientific_name",
        "candidate_genus_accepted_taxon_key",
        "candidate_genus_scientific_name",
        "candidate_species_accepted_taxon_key",
        "candidate_species_scientific_name",
        "route",
        "visual_domain",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
    )
    values = {
        field: _required_text(candidate[field], field=field) for field in text_fields
    }
    source_record_hash = _sha256(
        candidate["source_record_hash"], field="source_record_hash"
    )
    source_artifact_fingerprint = _sha256(
        candidate["source_artifact_fingerprint"],
        field="source_artifact_fingerprint",
    )
    no_geo = _required_bool(candidate["no_geo"], field="no_geo")
    geographic_cluster_id = _optional_text(
        candidate["geographic_cluster_id"], field="geographic_cluster_id"
    )
    if no_geo and geographic_cluster_id is not None:
        raise ValueError("no_geo candidates cannot claim a geographic cluster")
    if not no_geo and geographic_cluster_id is None:
        raise ValueError("georeferenced candidates require geographic_cluster_id")
    geography_stratum = "no_geo" if no_geo else f"geo:{geographic_cluster_id}"
    query_tier = _required_text(
        candidate["primary_query_tier"], field="primary_query_tier"
    ).upper()
    if query_tier not in QUERY_TIERS:
        raise ValueError(f"unsupported primary_query_tier: {query_tier}")
    raw_score = _finite_float(candidate["raw_fusion_score"], field="raw_fusion_score")
    margin = _finite_float(
        candidate["raw_competitor_margin"], field="raw_competitor_margin"
    )
    disagreement = _optional_finite_float(
        candidate["pool_disagreement"], field="pool_disagreement"
    )
    if disagreement is not None and disagreement < 0.0:
        raise ValueError("pool_disagreement cannot be negative")
    subject_area = _finite_float(
        candidate["subject_area_ratio"], field="subject_area_ratio"
    )
    if not 0.0 <= subject_area <= 1.0:
        raise ValueError("subject_area_ratio must be within [0, 1]")
    final_release_candidate = _required_bool(
        candidate["final_release_candidate"], field="final_release_candidate"
    )
    route_domain = f"{values['route']}|{values['visual_domain']}"
    stratum_values = {
        "candidate_family_accepted_taxon_key": values[
            "candidate_family_accepted_taxon_key"
        ],
        "candidate_genus_accepted_taxon_key": values[
            "candidate_genus_accepted_taxon_key"
        ],
        "candidate_species_accepted_taxon_key": values[
            "candidate_species_accepted_taxon_key"
        ],
        "geography_stratum": geography_stratum,
        "primary_query_tier": query_tier,
        "raw_score_band": _band(raw_score, policy.score_cutpoints),
        "raw_margin_band": _band(margin, policy.margin_cutpoints),
        "pool_disagreement_band": (
            "unavailable"
            if disagreement is None
            else _band(disagreement, policy.pool_disagreement_cutpoints)
        ),
        "route_domain_stratum": route_domain,
        "subject_size_band": _band(subject_area, policy.subject_area_cutpoints),
    }
    analysis_stratum_fingerprint = canonical_semantic_fingerprint(stratum_values)
    independence_group_fingerprint = canonical_semantic_fingerprint(
        {
            "owner_group_id": values["owner_group_id"],
            "duplicate_group_id": values["duplicate_group_id"],
            "observation_group_id": values["observation_group_id"],
        }
    )
    semantic_values: dict[str, object] = {
        **values,
        "source_record_hash": source_record_hash,
        "source_artifact_fingerprint": source_artifact_fingerprint,
        "geographic_cluster_id": geographic_cluster_id,
        "no_geo": no_geo,
        "primary_query_tier": query_tier,
        "raw_fusion_score": raw_score,
        "raw_competitor_margin": margin,
        "pool_disagreement": disagreement,
        "subject_area_ratio": subject_area,
        "final_release_candidate": final_release_candidate,
        **stratum_values,
        "independence_group_fingerprint": independence_group_fingerprint,
        "analysis_stratum_fingerprint": analysis_stratum_fingerprint,
        "strata_policy_fingerprint": policy.fingerprint,
    }
    audit_unit_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
            "candidate": semantic_values,
        }
    )
    return {
        "schema_version": DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION,
        "strata_policy_fingerprint": policy.fingerprint,
        "frame_fingerprint": "",
        "audit_unit_fingerprint": audit_unit_fingerprint,
        **values,
        "source_record_hash": source_record_hash,
        "source_artifact_fingerprint": source_artifact_fingerprint,
        "geographic_cluster_id": geographic_cluster_id,
        "no_geo": no_geo,
        "geography_stratum": geography_stratum,
        "primary_query_tier": query_tier,
        "raw_fusion_score": raw_score,
        "raw_score_band": stratum_values["raw_score_band"],
        "raw_competitor_margin": margin,
        "raw_margin_band": stratum_values["raw_margin_band"],
        "pool_disagreement": disagreement,
        "pool_disagreement_band": stratum_values["pool_disagreement_band"],
        "route_domain_stratum": route_domain,
        "subject_area_ratio": subject_area,
        "subject_size_band": stratum_values["subject_size_band"],
        "independence_group_fingerprint": independence_group_fingerprint,
        "analysis_stratum_id": (
            f"dynamic-pool-audit-stratum:{analysis_stratum_fingerprint.removeprefix('sha256:')}"
        ),
        "analysis_stratum_fingerprint": analysis_stratum_fingerprint,
        "score_semantics": RAW_SCORE_SEMANTICS,
        "probability_available": False,
        "final_release_candidate": final_release_candidate,
    }


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


def _required_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field=field)


def _cutpoints(values: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(_finite_float(value, field=field) for value in values)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{field} must be unique and strictly increasing")
    return normalized


def _band(value: float, cutpoints: tuple[float, ...]) -> str:
    for index, upper in enumerate(cutpoints):
        if value < upper:
            return f"band_{index:02d}_lt_{upper:g}"
    return f"band_{len(cutpoints):02d}_gte_{cutpoints[-1]:g}" if cutpoints else "all"


__all__ = [
    "DYNAMIC_POOL_AUDIT_FRAME_FILE",
    "DYNAMIC_POOL_AUDIT_FRAME_SCHEMA",
    "DYNAMIC_POOL_AUDIT_FRAME_SCHEMA_VERSION",
    "DYNAMIC_POOL_FAILURE_QUEUE_FILE",
    "DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA",
    "DYNAMIC_POOL_FAILURE_QUEUE_SCHEMA_VERSION",
    "DYNAMIC_POOL_PROBABILITY_REGISTER_FILE",
    "DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA",
    "DYNAMIC_POOL_PROBABILITY_REGISTER_SCHEMA_VERSION",
    "DYNAMIC_POOL_PROBABILITY_SAMPLE_FILE",
    "DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA",
    "DYNAMIC_POOL_PROBABILITY_SAMPLE_SCHEMA_VERSION",
    "RAW_SCORE_SEMANTICS",
    "DynamicPoolAuditStrataPolicy",
    "FailureDiscoveryPolicy",
    "ProbabilityAuditSamplingPolicy",
    "ProbabilityAuditSelection",
    "build_dynamic_pool_audit_frame",
    "build_failure_discovery_queue",
    "build_probability_audit_sample",
    "empty_dynamic_pool_audit_frame",
    "validate_dynamic_pool_audit_frame",
    "validate_failure_discovery_queue",
    "validate_probability_audit_selection",
]
