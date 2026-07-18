"""Validation for immutable dynamic-pooling task completion records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/task_0_1_completion.json"
PUSH_LEDGER = ROOT / "provenance/task_pushes.jsonl"


def _report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


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
