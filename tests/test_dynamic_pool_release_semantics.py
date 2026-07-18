"""Release-level traceability for dynamic-pooling scientific semantics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/release_scientific_semantics.json"
DECISION = ROOT / "reports/geo_dynamic_pooling/pilot/production_default_decision.json"
GITHITS = ROOT / "provenance/githits.jsonl"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scientific_release_gate_maps_all_required_invariants_to_tests() -> None:
    report = _json(REPORT)

    assert report["schema_version"] == (
        "geo-dynamic-pooling-scientific-semantics-gate-v1.0.0"
    )
    assert report["verdict"] == "all_required_scientific_software_semantics_verified"
    assert report["gate"]["passed"] == 21

    invariants = {item["id"]: item for item in report["invariants"]}
    assert set(invariants) == {
        "family-cannot-catastrophically-prune",
        "geography-cannot-certify-identity",
        "gbif-references-remain-provisional",
        "embeddings-not-recomputed-per-pool",
        "raw-evidence-is-not-probability",
        "statistical-support-is-not-human-verification",
        "unreviewed-cannot-enter-occurrence-export",
        "insufficient-strata-remain-unavailable",
        "targeted-samples-do-not-support-unweighted-quality",
        "downstream-maturity-is-preserved",
    }
    assert {item["status"] for item in invariants.values()} == {"passed"}
    assert all(item["test_nodes"] for item in invariants.values())


def test_every_semantic_test_node_resolves_to_a_current_test_function() -> None:
    report = _json(REPORT)

    for invariant in report["invariants"]:
        for node in invariant["test_nodes"]:
            relative_path, function = node.split("::", maxsplit=1)
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            assert f"def {function}(" in source


def test_semantic_gate_preserves_the_production_selection_boundary() -> None:
    report = _json(REPORT)
    decision = _json(DECISION)

    assert decision["decision"]["outcome"] == "insufficient_evidence"
    assert decision["decision"]["eligible_variant_count"] == 0
    assert decision["decision"]["runtime_settings_changed"] is False
    assert (
        decision["current_runtime_settings"] == decision["resulting_runtime_settings"]
    )
    assert report["production_selection_boundary"] == {
        "decision": "insufficient_evidence",
        "eligible_variant_count": 0,
        "runtime_settings_changed": False,
        "selected_candidate_strategy": None,
        "selected_pool_variant": None,
        "selected_fusion_method": None,
        "occurrence_release_authorized": False,
    }


def test_semantic_gate_records_user_directed_no_githits_provenance() -> None:
    report = _json(REPORT)
    records = [
        json.loads(line)
        for line in GITHITS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(item for item in records if item["task_id"] == "geo-pool-16.2.2")

    assert report["provenance"]["githits_calls_made"] == 0
    assert report["provenance"]["solution_id"] is None
    assert record["githits_status"] == "skipped_user_directive"
    assert record["solution_id"] is None
