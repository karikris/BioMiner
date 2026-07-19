"""Contract checks for the dynamic-pooling technical release receipt."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/release_technical_verification.json"


def _report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_technical_release_receipt_records_every_required_gate() -> None:
    report = _report()

    assert report["schema_version"] == (
        "geo-dynamic-pooling-technical-release-verification-v1.0.0"
    )
    assert report["tested_workflow_sha"] == ("19fd744b1104c09dde75367bafb6b531ef4239a4")
    assert report["verdict"] == (
        "technical_workflow_verified_with_explicit_live_and_secret_heuristic_limitations"
    )

    gates = {gate["name"]: gate for gate in report["test_gates"]}
    assert set(gates) == {
        "full_regression",
        "strict_mode",
        "adaptive_mode",
        "dynamic_pooling",
        "cli",
        "configured_schema_and_parity",
        "downstream_handoffs",
    }
    assert {gate["status"] for gate in gates.values()} == {"passed"}
    assert gates["full_regression"]["passed"] == 3233


def test_static_supply_chain_and_tracked_artifact_results_are_explicit() -> None:
    report = _report()
    static = report["static_and_supply_chain_gates"]
    artifacts = report["tracked_artifact_inspection"]

    assert static["ruff_lint"]["status"] == "passed"
    assert static["dependency_audit"]["known_vulnerabilities"] == 0
    assert static["secret_scan"]["heuristic_findings"] == 453
    assert static["secret_scan"]["private_key_or_common_live_token_prefix_matches"] == 0
    assert artifacts["tracked_files_over_1_mib"] == 0
    assert artifacts["tracked_media_or_model_weight_files"] == 0
    assert artifacts["git_diff_check"] == "passed"


def test_technical_receipt_preserves_unavailable_live_and_external_evidence() -> None:
    report = _report()

    assert report["live_scientific_execution"]["status"] == "not_run"
    assert (
        "production-default eligibility"
        in report["live_scientific_execution"]["claims_not_available"]
    )
    assert report["provenance"]["githits_calls_made"] == 0
    assert report["provenance"]["solution_ids"] == [None, None]
    assert report["provenance"]["external_code_contribution"] == "none"
