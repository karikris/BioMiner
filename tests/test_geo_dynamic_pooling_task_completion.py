"""Validation for immutable dynamic-pooling task completion records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/task_0_1_completion.json"
TASK_1_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_1_1_completion.json"
TASK_1_2_REPORT = ROOT / "reports/geo_dynamic_pooling/task_1_2_completion.json"
PUSH_LEDGER = ROOT / "provenance/task_pushes.jsonl"


def _report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _task_1_1_report() -> dict[str, object]:
    return json.loads(TASK_1_1_REPORT.read_text(encoding="utf-8"))


def _task_1_2_report() -> dict[str, object]:
    return json.loads(TASK_1_2_REPORT.read_text(encoding="utf-8"))


def test_task_0_1_completion_records_exact_commits_and_green_gate() -> None:
    report = _report()

    assert report["task_id"] == "geo-pool-0.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "bc821bffd11ad32877aa4a704aa2bb7a0d636ab6",
        "00f987f6f23acba7135b0b349412a2e7248e933f",
        "27c93f2745e6e8d869c338623c5becee9323ba47",
    ]
    assert report["gate"]["baseline_suite"]["passed"] == 2541
    assert report["gate"]["baseline_suite"]["result"] == "passed"
    assert report["gate"]["report_validation"]["passed"] == 10
    assert report["gate"]["provenance"]["result"] == "passed"


def test_task_0_1_push_event_matches_report() -> None:
    report = _report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-0.1")

    assert event["schema_version"] == "biominer-task-push-event-v1.0.0"
    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_0_1_report_preserves_blocked_scientific_claims() -> None:
    blocked = _report()["claims"]["blocked"]

    assert any("Dynamic global/local pool" in claim for claim in blocked)
    assert any("occurrence release" in claim for claim in blocked)


def test_task_1_1_completion_records_contracts_commits_and_green_gates() -> None:
    report = _task_1_1_report()

    assert report["task_id"] == "geo-pool-1.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "d9e4365af0b65cd147ccc0f44fc543f9b02c96ce",
        "315e6f3b04dc2a18bb1679c26f92563f5a7f1ade",
        "387887bb86d7c276d83eca6e29f328ea73b8d676",
        "5e87aa3171655ae2f8883287a2661f4a41839aac",
    ]
    assert len(report["artifacts"]) == 7
    assert report["gate"]["schema_and_determinism_tests"]["passed"] == 98
    assert report["gate"]["full_regression"]["passed"] == 2628
    assert report["gate"]["provenance"]["jsonl_records_validated"] == 105


def test_task_1_1_push_event_matches_report() -> None:
    report = _task_1_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-1.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_1_1_report_blocks_unearned_scientific_claims() -> None:
    blocked = _task_1_1_report()["claims"]["blocked"]

    assert any("empirically superior" in claim for claim in blocked)
    assert any("human review" in claim for claim in blocked)
    assert any("publication" in claim for claim in blocked)


def test_task_1_2_completion_records_handoffs_pins_and_green_gates() -> None:
    report = _task_1_2_report()

    assert report["task_id"] == "geo-pool-1.2"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "1a85c1951d650e5db3fa3ef058b9772abe964bf4",
        "1e1bdacc8ec902c381ea568c51e04ad15bdb7636",
        "f8d52a0236357d10a302c77e115fd27c9cbfc985",
    ]
    assert report["consumer_pins"]["taxalens"]["commit"] == (
        "c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc"
    )
    assert report["consumer_pins"]["butterflylens"]["commit"] == (
        "1cea643623f2f20a2bea72afc754c7b194db3278"
    )
    assert report["gate"]["cross_repository_compatibility"]["passed"] == 33
    assert report["gate"]["full_regression"]["passed"] == 2660
    assert report["gate"]["provenance"]["jsonl_records_validated"] == 109


def test_task_1_2_push_event_matches_report() -> None:
    report = _task_1_2_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-1.2")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_1_2_report_blocks_import_and_release_claims() -> None:
    blocked = _task_1_2_report()["claims"]["blocked"]

    assert any("live TaxaLens or ButterflyLens import" in claim for claim in blocked)
    assert any("reviewer assignment" in claim for claim in blocked)
    assert any("occurrence release" in claim for claim in blocked)
    assert any("production deployment" in claim for claim in blocked)
