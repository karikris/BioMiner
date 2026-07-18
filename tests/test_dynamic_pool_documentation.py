"""Documentation contracts for geography-conditioned dynamic pooling."""

from __future__ import annotations

from pathlib import Path

from biominer.bioclip.dynamic_pool_expansion import (
    DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION,
    DYNAMIC_POOL_EXPANSION_DECISION_SCHEMA_VERSION,
    DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION,
)
from biominer.bioclip.reference_geography_qa import (
    REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION,
)
from biominer.evaluation.candidate_strategies import (
    FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION,
)
from biominer.evaluation.dynamic_pool_splits import (
    REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
)


ROOT = Path(__file__).parents[1]
ARCHITECTURE = ROOT / "docs/architecture/geography_conditioned_dynamic_pooling.md"
STATISTICAL_SUPPORT = (
    ROOT / "docs/architecture/statistical_support_and_human_verification.md"
)
SCHEMA_CATALOG = (
    ROOT / "docs/schemas/geography_conditioned_dynamic_pooling_contracts.md"
)
TARGET_AWARE_CONTRACTS = ROOT / "docs/schemas/target_aware_few_shot_contracts.md"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_dynamic_pool_architecture_reports_implemented_and_pending_boundaries() -> None:
    architecture = _normalized(ARCHITECTURE)

    required = (
        "software and fixture implementation complete through Phase 15",
        "live empirical strategy selection remains pending",
        "insufficient_evidence",
        "full 86-effective-review shortfall",
        "leaves runtime settings unchanged",
        "536,870,912-byte policy limit",
        "No candidate strategy, pool variant, or fusion method is a production default",
    )
    assert all(term in architecture for term in required)


def test_dynamic_pool_schema_catalog_names_canonical_artifacts_and_authority() -> None:
    catalog = _normalized(SCHEMA_CATALOG)

    required = (
        "reference_geography_index.parquet",
        "flickr_scoring_units.parquet",
        "dynamic_reference_pool_plans.parquet",
        "dynamic_pool_candidate_scores.parquet",
        "dynamic_pool_probability_audit_register.parquet",
        "storage-handoff-inventory-v1.0.0",
        "Candidate, model, probability, human-review, statistical-support, release, and publication maturity remain separate",
        "None of these values is a probability",
        "Targeted work has null inclusion probabilities/weights",
        "No production default or occurrence release is authorized",
    )
    assert all(term in catalog for term in required)


def test_dynamic_pool_schema_catalog_tracks_implemented_version_constants() -> None:
    catalog = SCHEMA_CATALOG.read_text(encoding="utf-8")

    schema_versions = (
        REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION,
        DYNAMIC_POOL_EXPANSION_EVIDENCE_SCHEMA_VERSION,
        DYNAMIC_POOL_EXPANSION_DECISION_SCHEMA_VERSION,
        DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION,
        FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION,
        REVIEWED_FLICKR_COMPONENT_SCHEMA_VERSION,
    )
    assert all(f"`{version}`" in catalog for version in schema_versions)


def test_pooling_contracts_pin_exact_downstream_consumers() -> None:
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    catalog = SCHEMA_CATALOG.read_text(encoding="utf-8")
    pins = (
        "e845dd98493979f37b04dbb6538e0d7b8758ca11",
        "1cea643623f2f20a2bea72afc754c7b194db3278",
    )

    assert all(pin in architecture for pin in pins)
    assert all(pin in catalog for pin in pins)


def test_related_contracts_link_to_dynamic_pool_schema_catalog() -> None:
    target_aware = TARGET_AWARE_CONTRACTS.read_text(encoding="utf-8")
    statistical_support = _normalized(STATISTICAL_SUPPORT)

    assert (
        "[geography_conditioned_dynamic_pooling_contracts.md]"
        "(geography_conditioned_dynamic_pooling_contracts.md)" in target_aware
    )
    assert "zero reviewers, assignments, completed source-bound reviews" in (
        statistical_support
    )
    assert "86-effective-review production minimum" in statistical_support
