"""Guardrails for the geography-conditioned dynamic-pooling ADR."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
ADR = ROOT / "docs/architecture/geography_conditioned_dynamic_pooling.md"
GITHITS_LEDGER = ROOT / "provenance/githits.jsonl"


def _adr() -> str:
    return ADR.read_text(encoding="utf-8")


def test_adr_defines_required_architecture_boundaries() -> None:
    adr = _adr()
    normalized = " ".join(adr.split())

    required = (
        "Family evidence is a batching and retrieval accelerator, never a hard gate",
        "Geography is candidate and reference-selection evidence, never proof",
        "Every plan has a global pool selected independently of query proximity",
        "local_pool_status` is unavailable with the exact reason",
        "Neither query geography nor pool membership enters this key",
        "Deterministic uncertainty expansion",
        "Dynamic pooling produces **candidate/model evidence**",
    )
    assert all(boundary in normalized for boundary in required)


def test_adr_forbids_pruning_reencoding_and_probability_claims() -> None:
    adr = _adr()

    assert "may not remove any required candidate" in adr
    assert "Re-encode references per geographic pool" in adr
    assert "Raw cosine similarities, weighted components, margins" in adr
    assert "are not probabilities" in adr
    assert "Positive human review alone is not occurrence release" in adr


def test_adr_preserves_no_geo_and_cache_safe_behavior() -> None:
    adr = _adr()

    assert "`no_geo` and `unassigned_geo` remain distinct" in adr
    assert "local counts, distances and local scores are null, not zero" in adr
    assert "one raw embedding per compatible image/input identity" in adr
    assert "cannot become an unbounded search" in adr


def test_task_and_subtask_githits_attempts_are_recorded() -> None:
    records = [
        json.loads(line)
        for line in GITHITS_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["task_id"]: record for record in records}

    for task_id in ("geo-pool-0.2", "geo-pool-0.2.1"):
        record = by_id[task_id]
        assert record["githits_status"] == "unavailable_timeout"
        assert record["solution_id"] is None
        assert record["feedback_recorded"] is False
