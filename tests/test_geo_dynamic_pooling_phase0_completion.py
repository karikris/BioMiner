"""Validation for dynamic-pooling Task 0.2 and Phase 0 completion."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT_ROOT = ROOT / "reports/geo_dynamic_pooling"
CURRENT_STATE = ROOT / "docs/agents/CURRENT_STATE.md"
PUSH_LEDGER = ROOT / "provenance/task_pushes.jsonl"


def _json(name: str) -> dict[str, object]:
    return json.loads((REPORT_ROOT / name).read_text(encoding="utf-8"))


def test_task_0_2_completion_records_exact_commits_and_gate() -> None:
    report = _json("task_0_2_completion.json")

    assert [item["commit"] for item in report["task_commits"]] == [
        "6646996b1af60736ae1927473aa7461592f1a3ad",
        "299914548b407b439cd36d1aa99397b41aa827f1",
    ]
    assert report["gate"]["adr_and_audit_tests"] == {
        "command": "uv run pytest -q tests/test_geography_conditioned_dynamic_pooling_adr.py tests/test_statistical_support_human_verification_adr.py tests/test_current_reference_pooling_audit.py tests/test_downstream_pooling_handoff_audit.py",
        "result": "passed",
        "passed": 17,
    }
    assert report["gate"]["human_decision"]["result"] == "passed"
    assert report["push"]["status"] == "verified"


def test_phase_0_acceptance_and_nonclaims_are_complete() -> None:
    report = _json("phase_0_completion.json")

    assert report["status"] == "complete"
    assert all(report["acceptance"].values())
    assert report["next_phase"]["next_task"] == "geo-pool-1.1"
    assert len(report["explicit_non_claims"]) == 4
    assert all(task["push_status"] == "verified" for task in report["tasks"])


def test_task_push_ledger_contains_both_verified_phase_0_tasks() -> None:
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {event["task_id"]: event for event in events}

    for task_id in ("geo-pool-0.1", "geo-pool-0.2"):
        assert by_id[task_id]["status"] == "verified"
        assert (
            by_id[task_id]["pushed_through_sha"]
            == by_id[task_id]["verified_remote_sha"]
        )


def test_current_state_points_to_active_dynamic_pooling_phase() -> None:
    state = " ".join(CURRENT_STATE.read_text(encoding="utf-8").split())

    assert "geography-conditioned dynamic global/local reference" in state
    assert "Phase 0 baseline, audit and design complete" in state
    assert "299914548b407b439cd36d1aa99397b41aa827f1" in state
    assert "Phase 1 contract alignment" in state
