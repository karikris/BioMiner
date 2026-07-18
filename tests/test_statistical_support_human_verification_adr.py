"""Vocabulary guardrails for evidence maturity and human verification."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
ADR = ROOT / "docs/architecture/statistical_support_and_human_verification.md"
AGENT_RULES = ROOT / "AGENTS.md"
GITHITS_LEDGER = ROOT / "provenance/githits.jsonl"


def _normalized_adr() -> str:
    return " ".join(ADR.read_text(encoding="utf-8").split())


def test_vocabulary_distinguishes_all_required_maturity_states() -> None:
    adr = _normalized_adr()

    for heading in (
        "### Human-reviewed",
        "### Calibrated",
        "### Statistically supported",
        "### Release-ready occurrence candidate",
        "### Published occurrence",
    ):
        assert heading in adr


def test_statistical_support_never_implies_item_review_or_release() -> None:
    adr = _normalized_adr()

    assert "It does not create a human review event for unsampled items" in adr
    assert "statistically supported means human/community/expert verified" in adr
    assert "a passing population metric authorizes occurrence release" in adr
    assert "statistically_supported_screening_candidate" in adr


def test_calibration_and_release_authorities_remain_separate() -> None:
    adr = _normalized_adr()

    assert "Raw similarities are uncalibrated" in adr
    assert "Calibration and statistical support answer different questions" in adr
    assert "release readiness and publication use different identities" in adr
    assert "Publication requires release readiness plus downstream authorization" in adr


def test_zero_and_unavailable_evidence_are_not_conflated() -> None:
    adr = _normalized_adr()

    assert "For available count fields, zero is a real observed value" in adr
    assert "For unavailable metrics, the value is null" in adr
    assert "precision, agreement and reliability remain unavailable" in adr
    assert "not_applicable" in adr


def test_root_agent_rule_matches_evidence_vocabulary() -> None:
    rules = " ".join(AGENT_RULES.read_text(encoding="utf-8").split())

    assert "Keep candidate evidence, model evidence, human review" in rules
    assert "Review alone is not occurrence release" in rules


def test_subtask_githits_timeout_is_recorded_without_invented_solution() -> None:
    records = [
        json.loads(line)
        for line in GITHITS_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(item for item in records if item["task_id"] == "geo-pool-0.2.2")

    assert record["githits_status"] == "unavailable_timeout"
    assert record["solution_id"] is None
    assert record["feedback_recorded"] is False
