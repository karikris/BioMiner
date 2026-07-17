from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "gbif_fast_start" / "adaptive_release_verification.json"


def test_adaptive_release_verification_is_complete_and_honest() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == "gbif-adaptive-release-verification-v1.0.0"
    assert report["verdict"] == "core_workflow_verified_with_documented_tooling_gaps"
    assert report["test_gates"][-1] == {
        "name": "full_regression",
        "status": "passed",
        "passed": 2530,
        "duration_seconds": 97.84,
    }
    assert report["static_and_supply_chain_gates"]["ruff_lint"]["status"] == "passed"
    assert report["static_and_supply_chain_gates"]["mypy_full"]["status"].startswith(
        "failed_"
    )
    assert report["static_and_supply_chain_gates"]["dependency_audit"][
        "known_vulnerabilities"
    ] == 0
    assert report["static_and_supply_chain_gates"]["secret_scan"][
        "private_key_or_common_live_token_prefix_matches"
    ] == 0
    assert report["artifact_inspection"]["parse_errors"] == 0
    assert report["live_source_smoke"]["status"] == "not_run"

    invariants = report["core_invariants"]
    assert len(invariants) == 10
    assert {item["status"] for item in invariants} == {"passed"}
    assert {item["id"] for item in invariants} == {
        "default-adaptive",
        "strict-available",
        "provider-not-human",
        "flickr-not-final",
        "gbif-holdout-isolation",
        "audit-required",
        "targeted-review",
        "embedding-reuse",
        "raw-score-not-probability",
        "admission-provenance",
    }
    assert report["limitations"]
