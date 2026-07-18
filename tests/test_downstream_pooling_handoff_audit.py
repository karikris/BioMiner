"""Contract checks for the TaxaLens and ButterflyLens handoff audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/downstream_pooling_contract_pins.json"
AUDIT = ROOT / "docs/architecture/taxalens_butterflylens_pooling_handoff.md"
AGENT_RULES = ROOT / "AGENTS.md"


def _pins() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_downstream_contract_pins_are_exact_and_versioned() -> None:
    pins = _pins()

    assert pins["schema_version"] == (
        "biominer-downstream-pooling-contract-pins-v1.1.0"
    )
    assert pins["taxalens"]["goal_written_commit"] == (
        "1440596cf4403af61ba8d57481feacda7c4e3044"
    )
    assert pins["taxalens"]["audited_commit"] == (
        "c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc"
    )
    assert pins["butterflylens"]["goal_written_commit"] == (
        "c8135a0cb0001245215cdc774d063ef49407fb26"
    )
    assert pins["butterflylens"]["previous_audited_commit"] == (
        "fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3"
    )
    assert pins["butterflylens"]["audited_commit"] == (
        "1cea643623f2f20a2bea72afc754c7b194db3278"
    )
    assert pins["butterflylens"]["pin_movement_decision"] == (
        "compatible_additive_with_stricter_review_controls"
    )


def test_handoff_policy_fails_closed() -> None:
    policy = _pins()["policy"]

    assert policy == {
        "consume_committed_objects_only": True,
        "import_sibling_implementation_code": False,
        "missing_evidence_is_false_or_zero": False,
        "review_is_occurrence_release": False,
        "silent_pin_movement_allowed": False,
    }


def test_audit_covers_required_downstream_boundaries() -> None:
    audit = AUDIT.read_text(encoding="utf-8")

    required_terms = {
        "Geographic review and quality",
        "Evidence maturity and release projections",
        "Artifact export expectations",
        "Discovery, model evidence, and review layers",
        "Map impact, RLS, and import boundary",
        "Australian taxonomy and ALA geography",
        "BioMiner dynamic-pooling handoff requirements",
        "reviewed-labels-v2",
        "butterflylens-classification-maturity:v1.0.0",
    }
    assert all(term in audit for term in required_terms)


def test_agent_rules_preserve_downstream_evidence_maturity() -> None:
    rules = " ".join(AGENT_RULES.read_text(encoding="utf-8").split())

    assert "Downstream handoffs are immutable, versioned artifacts" in rules
    assert "Review alone is not occurrence release" in rules
    assert "unavailable or unrun evidence is not false or" in rules
