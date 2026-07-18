"""Frozen evidence and execution contract for the bounded dynamic-pool pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from biominer.bioclip.dynamic_pool_fusion import RAW_FUSION_METHODS
from biominer.candidates.strategy_ablation import CANDIDATE_STRATEGIES
from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOL_PILOT_PLAN_VERSION = "dynamic-pool-pilot-plan-v1.0.0"
DYNAMIC_POOL_PILOT_PLAN_FILE = "geography_conditioned_dynamic_pool_pilot_v1.json"
PILOT_CANDIDATE_STRATEGIES = (
    "geography_first",
    "family_first_safe",
    "parallel_family_geography_union",
)
PILOT_POOL_VARIANTS = (
    "global_only_control",
    "dynamic_global_local",
)
PILOT_CASE_EVIDENCE_BASIS = "fixture_expected_taxon_not_human_review"
PILOT_REQUIRED_INVARIANTS = (
    "raw_scores_are_not_probabilities",
    "missing_geography_is_not_biological_absence",
    "representative_and_targeted_review_are_separate",
    "provider_asserted_support_is_not_human_verification",
    "fixture_metrics_cannot_select_a_production_default",
    "occurrence_release_requires_source_bound_human_review",
)

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "pilot_id",
        "title",
        "frozen_at",
        "producer_commit",
        "evidence_boundary",
        "durable_inputs",
        "taxon_catalog",
        "cases",
        "ablations",
        "execution_limits",
        "acceptance_policy",
        "scientific_invariants",
        "plan_fingerprint",
    }
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def load_dynamic_pool_pilot_plan(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate one frozen pilot plan and optional durable inputs."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dynamic-pool pilot plan JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("dynamic-pool pilot plan must contain an object")
    validate_dynamic_pool_pilot_plan(payload)
    if repository_root is not None:
        validate_dynamic_pool_pilot_inputs(payload, repository_root)
    return payload


def validate_dynamic_pool_pilot_plan(plan: Mapping[str, object]) -> None:
    """Fail closed when pilot scope, evidence maturity, or policy drifts."""

    if set(plan) != _PLAN_FIELDS:
        raise ValueError("dynamic-pool pilot plan fields do not match")
    if plan.get("schema_version") != DYNAMIC_POOL_PILOT_PLAN_VERSION:
        raise ValueError("unsupported dynamic-pool pilot plan version")
    _nonblank(plan.get("pilot_id"), field="pilot_id")
    _nonblank(plan.get("title"), field="title")
    _nonblank(plan.get("frozen_at"), field="frozen_at")
    producer_commit = plan.get("producer_commit")
    if not isinstance(producer_commit, str) or not _GIT_SHA.fullmatch(producer_commit):
        raise ValueError("pilot producer commit must be a full Git SHA")

    _validate_evidence_boundary(_mapping(plan, "evidence_boundary"))
    _validate_durable_inputs(_sequence(plan, "durable_inputs"))
    taxa = _validate_taxon_catalog(_sequence(plan, "taxon_catalog"))
    _validate_cases(_sequence(plan, "cases"), taxa=taxa)
    _validate_ablations(_mapping(plan, "ablations"))
    _validate_execution_limits(
        _mapping(plan, "execution_limits"),
        case_count=len(_sequence(plan, "cases")),
    )
    _validate_acceptance_policy(_mapping(plan, "acceptance_policy"))
    if tuple(_sequence(plan, "scientific_invariants")) != PILOT_REQUIRED_INVARIANTS:
        raise ValueError("pilot scientific invariants are incomplete or reordered")

    expected = canonical_semantic_fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    if plan.get("plan_fingerprint") != expected:
        raise ValueError("dynamic-pool pilot plan fingerprint mismatch")


def validate_dynamic_pool_pilot_inputs(
    plan: Mapping[str, object], repository_root: str | Path
) -> None:
    """Verify every committed durable-input descriptor against local bytes."""

    validate_dynamic_pool_pilot_plan(plan)
    root = Path(repository_root).resolve()
    for raw_descriptor in _sequence(plan, "durable_inputs"):
        descriptor = _as_mapping(raw_descriptor, field="durable input")
        relative_path = Path(str(descriptor["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("pilot durable input path must remain repository-relative")
        source = (root / relative_path).resolve()
        if not source.is_relative_to(root):
            raise ValueError("pilot durable input escapes the repository")
        if not source.is_file():
            raise ValueError(f"pilot durable input is unavailable: {relative_path}")
        observed = f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"
        if observed != descriptor["sha256"]:
            raise ValueError(f"pilot durable input SHA-256 differs: {relative_path}")
        if source.stat().st_size != descriptor["byte_count"]:
            raise ValueError(f"pilot durable input byte count differs: {relative_path}")


def _validate_evidence_boundary(boundary: Mapping[str, object]) -> None:
    expected = {
        "current_execution_basis": "fixture_backed",
        "historical_real_source_inventory_present": True,
        "historical_outputs_count_as_current_execution": False,
        "fixture_expected_taxa_are_human_labels": False,
        "current_human_reviewed_label_count": 0,
        "live_network_calls_planned": False,
        "live_model_execution_planned": False,
        "production_default_authorized": False,
        "occurrence_release_authorized": False,
    }
    if dict(boundary) != expected:
        raise ValueError("pilot evidence boundary was weakened or changed")


def _validate_durable_inputs(values: Sequence[object]) -> None:
    if len(values) < 5:
        raise ValueError("pilot requires a broad durable-input inventory")
    paths: set[str] = set()
    allowed_kinds = {
        "committed_registry",
        "committed_taxon_table",
        "committed_source_snapshot",
        "historical_real_execution_manifest",
    }
    for raw_descriptor in values:
        descriptor = _as_mapping(raw_descriptor, field="durable input")
        if set(descriptor) != {
            "relative_path",
            "sha256",
            "byte_count",
            "evidence_kind",
            "pilot_use",
            "scientific_authority",
        }:
            raise ValueError("pilot durable-input fields do not match")
        path = _nonblank(descriptor.get("relative_path"), field="relative_path")
        if path in paths:
            raise ValueError("pilot durable-input paths must be unique")
        paths.add(path)
        if (
            not isinstance(descriptor.get("byte_count"), int)
            or descriptor["byte_count"] <= 0
        ):
            raise ValueError("pilot durable-input byte count must be positive")
        if not _is_sha256(descriptor.get("sha256")):
            raise ValueError("pilot durable-input SHA-256 is invalid")
        if descriptor.get("evidence_kind") not in allowed_kinds:
            raise ValueError("pilot durable-input evidence kind is unsupported")
        _nonblank(descriptor.get("pilot_use"), field="pilot_use")
        if descriptor.get("scientific_authority") not in {
            "taxonomy_identity_only",
            "discovery_metadata_only",
            "historical_execution_evidence_only",
        }:
            raise ValueError("pilot durable-input authority is unsupported")


def _validate_taxon_catalog(
    values: Sequence[object],
) -> dict[str, Mapping[str, object]]:
    if len(values) < 5:
        raise ValueError("pilot taxon catalog is too narrow")
    taxa: dict[str, Mapping[str, object]] = {}
    for raw_taxon in values:
        taxon = _as_mapping(raw_taxon, field="taxon")
        if set(taxon) != {
            "accepted_taxon_key",
            "scientific_name",
            "family_key",
            "family",
            "genus_key",
            "genus",
            "pilot_roles",
            "registry_evidence",
        }:
            raise ValueError("pilot taxon fields do not match")
        key = _nonblank(taxon.get("accepted_taxon_key"), field="accepted_taxon_key")
        if key in taxa:
            raise ValueError("pilot accepted taxon keys must be unique")
        for field in ("scientific_name", "family_key", "family", "genus_key", "genus"):
            _nonblank(taxon.get(field), field=field)
        roles = tuple(_sequence(taxon, "pilot_roles"))
        if not roles or any(not isinstance(role, str) or not role for role in roles):
            raise ValueError("pilot taxon roles must be nonblank strings")
        if taxon.get("registry_evidence") != "butterflies-v2-20260712":
            raise ValueError("pilot taxon is not bound to the frozen registry")
        taxa[key] = taxon
    if not any("primary_target" in taxon["pilot_roles"] for taxon in taxa.values()):
        raise ValueError("pilot taxon catalog requires a primary target")
    if not any(
        "stress_same_genus_competitor" in taxon["pilot_roles"]
        for taxon in taxa.values()
    ):
        raise ValueError("pilot requires a same-genus stress competitor")
    return taxa


def _validate_cases(
    values: Sequence[object], *, taxa: Mapping[str, Mapping[str, object]]
) -> None:
    if len(values) < 7:
        raise ValueError("pilot case register is too narrow")
    case_ids: set[str] = set()
    australian_taxa: set[str] = set()
    located_regions: set[str] = set()
    missing_geo_count = 0
    for raw_case in values:
        case = _as_mapping(raw_case, field="case")
        if set(case) != {
            "case_id",
            "accepted_taxon_key",
            "fixture_media_id",
            "country_code",
            "region_id",
            "geographic_evidence_status",
            "expected_label_basis",
            "biological_occurrence_claim",
            "review_status",
        }:
            raise ValueError("pilot case fields do not match")
        case_id = _nonblank(case.get("case_id"), field="case_id")
        if case_id in case_ids:
            raise ValueError("pilot case IDs must be unique")
        case_ids.add(case_id)
        key = _nonblank(case.get("accepted_taxon_key"), field="accepted_taxon_key")
        if key not in taxa:
            raise ValueError("pilot case taxon is absent from the catalog")
        _nonblank(case.get("fixture_media_id"), field="fixture_media_id")
        if case.get("expected_label_basis") != PILOT_CASE_EVIDENCE_BASIS:
            raise ValueError("pilot case expected label maturity was promoted")
        if case.get("biological_occurrence_claim") is not False:
            raise ValueError("pilot fixture case cannot make an occurrence claim")
        if case.get("review_status") != "not_human_reviewed_fixture":
            raise ValueError("pilot fixture case review status is invalid")
        status = case.get("geographic_evidence_status")
        country = case.get("country_code")
        region = case.get("region_id")
        if status == "located_fixture_context":
            if not isinstance(country, str) or len(country) != 2:
                raise ValueError("located pilot case requires a country code")
            _nonblank(region, field="region_id")
            located_regions.add(str(region))
            if country == "AU":
                australian_taxa.add(key)
        elif status == "missing_source_geography":
            if country is not None or region is not None:
                raise ValueError("missing-geography case cannot contain a region")
            missing_geo_count += 1
        else:
            raise ValueError("pilot geographic evidence status is unsupported")
    if len(australian_taxa) < 4:
        raise ValueError("pilot requires several distinct Australian butterfly taxa")
    if len(located_regions) < 4:
        raise ValueError("pilot requires several distinct located regions")
    if missing_geo_count < 1:
        raise ValueError("pilot requires an explicit no-geography case")


def _validate_ablations(ablations: Mapping[str, object]) -> None:
    if set(ablations) != {
        "candidate_strategies",
        "pool_variants",
        "fusion_methods",
        "variant_count",
        "comparability",
    }:
        raise ValueError("pilot ablation fields do not match")
    candidate_strategies = tuple(_sequence(ablations, "candidate_strategies"))
    if candidate_strategies != PILOT_CANDIDATE_STRATEGIES:
        raise ValueError("pilot candidate strategies are incomplete or reordered")
    if set(candidate_strategies) != CANDIDATE_STRATEGIES:
        raise ValueError("pilot candidate strategy set differs from production")
    if tuple(_sequence(ablations, "pool_variants")) != PILOT_POOL_VARIANTS:
        raise ValueError("pilot pool variants are incomplete or reordered")
    if tuple(_sequence(ablations, "fusion_methods")) != RAW_FUSION_METHODS:
        raise ValueError("pilot fusion methods are incomplete or reordered")
    expected_count = (
        len(PILOT_CANDIDATE_STRATEGIES)
        * len(PILOT_POOL_VARIANTS)
        * len(RAW_FUSION_METHODS)
    )
    if ablations.get("variant_count") != expected_count:
        raise ValueError("pilot ablation variant count mismatch")
    comparability = _mapping(ablations, "comparability")
    if dict(comparability) != {
        "same_case_register": True,
        "same_candidate_union": True,
        "same_fixture_embeddings": True,
        "target_pruning_allowed": False,
        "raw_scores_are_probabilities": False,
        "strategy_order_is_accuracy_claim": False,
    }:
        raise ValueError("pilot ablation comparability contract was weakened")


def _validate_execution_limits(
    limits: Mapping[str, object], *, case_count: int
) -> None:
    expected = {
        "fixture_case_count": case_count,
        "maximum_unique_fixture_media": case_count,
        "encode_each_unique_media_once": True,
        "reuse_reference_embeddings": True,
        "reuse_candidate_and_pool_matrices": True,
        "live_network_calls_allowed": False,
        "source_media_bytes_in_artifacts": False,
        "maximum_matrix_cache_bytes": 536870912,
        "random_seed": 20260718,
    }
    if dict(limits) != expected:
        raise ValueError("pilot execution limits changed")


def _validate_acceptance_policy(policy: Mapping[str, object]) -> None:
    if set(policy) != {
        "policy_version",
        "eligible_evidence_basis",
        "fixture_evidence_can_select_default",
        "minimum_target_candidate_recall",
        "minimum_reviewed_precision_lower_bound",
        "minimum_effective_reviewed_records",
        "minimum_subgroup_independent_records",
        "no_target_pruning_regressions_required",
        "embedding_reuse_required",
        "matrix_reuse_required",
        "mps_memory_limit_bytes",
        "unsupported_statistical_claims_allowed",
        "allowed_decisions",
        "fixture_forced_decision",
    }:
        raise ValueError("pilot acceptance-policy fields do not match")
    expected = {
        "policy_version": "dynamic-pool-production-selection-v1.0.0",
        "eligible_evidence_basis": "real_source_bound_human_review",
        "fixture_evidence_can_select_default": False,
        "minimum_target_candidate_recall": 1.0,
        "minimum_reviewed_precision_lower_bound": 0.95,
        "minimum_effective_reviewed_records": 86,
        "minimum_subgroup_independent_records": 30,
        "no_target_pruning_regressions_required": True,
        "embedding_reuse_required": True,
        "matrix_reuse_required": True,
        "mps_memory_limit_bytes": 536870912,
        "unsupported_statistical_claims_allowed": False,
        "allowed_decisions": ["select", "reject", "insufficient_evidence"],
        "fixture_forced_decision": "insufficient_evidence",
    }
    if dict(policy) != expected:
        raise ValueError("pilot acceptance policy was weakened or changed")


def _mapping(parent: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _as_mapping(parent.get(field), field=field)


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"pilot {field} must be an object")
    return value


def _sequence(parent: Mapping[str, object], field: str) -> Sequence[object]:
    value = parent.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"pilot {field} must be an array")
    return value


def _nonblank(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"pilot {field} must be a canonical nonblank string")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "DYNAMIC_POOL_PILOT_PLAN_FILE",
    "DYNAMIC_POOL_PILOT_PLAN_VERSION",
    "PILOT_CANDIDATE_STRATEGIES",
    "PILOT_CASE_EVIDENCE_BASIS",
    "PILOT_POOL_VARIANTS",
    "PILOT_REQUIRED_INVARIANTS",
    "load_dynamic_pool_pilot_plan",
    "validate_dynamic_pool_pilot_inputs",
    "validate_dynamic_pool_pilot_plan",
]
