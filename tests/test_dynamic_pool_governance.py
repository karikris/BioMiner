"""Governance and resumable-agent contracts for dynamic pooling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DECISION = ROOT / "docs/governance/geography_conditioned_dynamic_pooling_policy.md"
STRUCTURED_DECISION = (
    ROOT / "reports/geo_dynamic_pooling/pilot/production_default_decision.json"
)
INTEGRATED_REPORT = (
    ROOT / "reports/geo_dynamic_pooling/pilot/geography_conditioned_pooling_report.json"
)
CURRENT_STATE = ROOT / "docs/agents/CURRENT_STATE.md"
SCIENCE = ROOT / "docs/agents/SCIENCE_AND_PIPELINE.md"
RELEASE = ROOT / "docs/agents/TESTING_AND_RELEASE.md"
TOOLS = ROOT / "docs/agents/TOOLS_AND_SKILLS.md"
PACK_MANIFEST = ROOT / "docs/agents/PACK-MANIFEST.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_human_decision_binds_fail_closed_structured_outcome() -> None:
    decision = _normalized(DECISION)
    structured = _json(STRUCTURED_DECISION)
    report = _json(INTEGRATED_REPORT)

    assert structured["decision"]["outcome"] == "insufficient_evidence"
    assert structured["decision"]["eligible_variant_count"] == 0
    assert structured["decision"]["runtime_settings_changed"] is False
    assert (
        structured["current_runtime_settings"]
        == structured["resulting_runtime_settings"]
    )

    exact_fingerprints = (
        structured["decision_fingerprint"],
        structured["current_runtime_settings"]["settings_fingerprint"],
        structured["current_runtime_settings"]["reference_pool_policy_fingerprint"],
        report["source_fingerprints"]["selection_ablation_table"],
        report["report_fingerprint"],
    )
    assert all(fingerprint in decision for fingerprint in exact_fingerprints)
    assert "The fixture projection is not a human choice" in decision
    assert "Release continues to fail closed" in decision


def test_agent_topics_resume_at_release_without_scientific_overclaim() -> None:
    current = _normalized(CURRENT_STATE)
    science = _normalized(SCIENCE)
    release = _normalized(RELEASE)
    tools = _normalized(TOOLS)

    assert "final software release verification are complete" in current
    assert "exact next action is one bounded, instrumented current-policy live run" in (
        current
    )
    assert "No strategy is selected or production-defaulted" in current
    assert "zero variants are eligible" in science
    assert "fixture projection" in science
    assert "family optimization never catastrophically prunes" in release
    assert "unreviewed Flickr cannot enter an occurrence export" in release
    assert "does not fill the 86-effective-review shortfall" in release
    assert "Do not call GitHits again during this goal" in tools
    assert 'githits_status: "skipped_user_directive"' in tools


def test_agent_pack_manifest_matches_every_instruction_file() -> None:
    manifest = _json(PACK_MANIFEST)

    assert manifest["observed_main_commit"] == (
        "98c64ec27e0aaa6aa3da333b3e4d37df3fc1c30b"
    )
    assert "software/fixture goal is complete" in manifest["active_goal_note"]

    for item in manifest["files"]:
        path = ROOT / item["path"]
        content = path.read_bytes()
        assert item["bytes"] == len(content)
        assert item["lines"] == len(content.splitlines())
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
