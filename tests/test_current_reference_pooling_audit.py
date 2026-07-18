from __future__ import annotations

import json
from pathlib import Path

from biominer.bioclip.provisional_prototypes import RobustPrototypePolicy
from biominer.bioclip.provisional_ranking import provisional_reference_ranking
from biominer.bioclip.target_aware_scoring import TargetAwareCompleteSetResult
from biominer.candidates.regional_union import RegionalCandidateConfig
from biominer.flickr_fetch.geographic_clustering import (
    GLOBAL_FALLBACK_CLUSTER_IDS,
    NO_GEO_CLUSTER_ID,
    UNASSIGNED_GEO_CLUSTER_ID,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/architecture/current_reference_pooling_audit.md"
GITHITS_LEDGER = ROOT / "provenance/githits.jsonl"


def test_fixed_pool_audit_records_current_defaults_and_guardrails() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    prototype_policy = RobustPrototypePolicy()
    candidate_policy = RegionalCandidateConfig()

    assert provisional_reference_ranking.__kwdefaults__["top_k"] == 3
    assert provisional_reference_ranking.__kwdefaults__["prototype_method"] == (
        "trimmed_mean"
    )
    assert prototype_policy.maximum_observations_per_species_route == 64
    assert prototype_policy.prototype_count == 1
    assert prototype_policy.trim_fraction == 0.1
    assert candidate_policy.minimum_local_same_family_candidates == 20
    assert candidate_policy.include_registry_same_family_for_global_fallback
    assert GLOBAL_FALLBACK_CLUSTER_IDS == {
        NO_GEO_CLUSTER_ID,
        UNASSIGNED_GEO_CLUSTER_ID,
    }

    for required_statement in (
        "embedding-cache safe but not\npool-contract complete",
        "`(prototype_similarity + top_k_reference_mean) / 2`",
        "`geo_cluster_id`, country, bioregion and distance are\nabsent",
        "Missing global/local identities",
        "Candidate and reference class-size bias risks",
        "Repeated-work audit",
        "Current no-geography behaviour",
        "Current family-evidence role",
        "`hierarchy_pruning_applied` must be false",
        "No current production schema contains all of the following comparison-plan",
    ):
        assert required_statement in text


def test_target_aware_result_defaults_prohibit_family_pruning() -> None:
    fields = TargetAwareCompleteSetResult.__dataclass_fields__
    assert fields["hierarchy_pruning_applied"].default is False
    assert fields["hierarchy_rankings_diagnostic_only"].default is True


def test_fixed_pool_audit_has_separate_successful_githits_record() -> None:
    records = [
        json.loads(line)
        for line in GITHITS_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subtask = [row for row in records if row.get("task_id") == "geo-pool-0.1.2"]
    assert len(subtask) == 1
    assert subtask[0]["solution_id"] == "36171a86-56f0-4b48-936a-6bc08ec5589d"
    assert subtask[0]["githits_status"] == "used_patterns_only_no_code_copied"
    assert subtask[0]["feedback_recorded"] is True
