"""Validation for the immutable Task 16.2 completion record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/task_16_2_completion.json"
GITHITS = ROOT / "provenance/githits.jsonl"
TASK_PUSHES = ROOT / "provenance/task_pushes.jsonl"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_task_16_2_records_commits_gates_and_fail_closed_outcome() -> None:
    report = _json(REPORT)

    assert report["schema_version"] == "geo-dynamic-pooling-task-completion-v1.0.0"
    assert report["task_id"] == "geo-pool-16.2"
    assert report["status"] == "completed"
    assert [row["commit"] for row in report["task_commits"]] == [
        "670185286ab78f4d538cfd2bc222fef1e7d8da7e",
        "98c64ec27e0aaa6aa3da333b3e4d37df3fc1c30b",
        "ade7c17741decb8866ce885396b8f0142cdf7eea",
    ]
    assert report["gate"]["technical"]["full_pytest"]["passed"] == 3233
    assert report["gate"]["scientific_semantics"]["passed"] == 21
    assert report["gate"]["scientific_semantics"]["invariants_verified"] == 10
    assert report["release_evidence"]["minimum_artifact_contracts_mapped"] == 36
    assert report["release_evidence"]["acceptance_criteria_reported"] == 70

    outcome = report["scientific_outcome"]
    assert outcome["production_selection_decision"] == "insufficient_evidence"
    assert outcome["eligible_variant_count"] == 0
    assert outcome["effective_real_review_shortfall"] == 86
    assert outcome["estimated_reviewed_precision"] is None
    assert outcome["runtime_settings_changed"] is False
    assert outcome["selected_candidate_strategy"] is None
    assert outcome["selected_pool_variant"] is None
    assert outcome["selected_fusion_method"] is None
    assert outcome["occurrence_release_authorized"] is False


def test_task_16_2_push_and_no_githits_call_are_exact() -> None:
    report = _json(REPORT)
    push = next(row for row in _jsonl(TASK_PUSHES) if row["task_id"] == "geo-pool-16.2")
    records = [
        row
        for row in _jsonl(GITHITS)
        if row["task_id"] == "geo-pool-16.2"
        or row["task_id"].startswith("geo-pool-16.2.")
    ]

    assert push["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert push["verified_remote_sha"] == "ade7c17741decb8866ce885396b8f0142cdf7eea"
    assert push["status"] == report["push"]["status"] == "verified"
    assert len(records) == 4
    assert all(row["githits_status"] == "skipped_user_directive" for row in records)
    assert all(row["solution_id"] is None for row in records)
    assert report["githits_architecture_impact"]["calls_made_for_task"] == 0
    assert (
        report["githits_architecture_impact"]["direct_external_code_contribution"]
        == "none"
    )
