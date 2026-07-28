import inspect

import biominer.gbif_quality.acceptance as acceptance
from biominer.gbif_quality.acceptance import (
    ACCEPTANCE_VERSION,
    CRITERIA,
    SCHEMA,
    _markdown,
)


def test_global_acceptance_registry_is_exact_and_renderable() -> None:
    assert len(CRITERIA) == 42
    assert len(set(CRITERIA)) == 42
    assert SCHEMA.names[:4] == [
        "acceptance_version",
        "criterion_number",
        "requirement",
        "status",
    ]
    rows = [
        {
            "criterion_number": number,
            "status": "PASS",
            "requirement": requirement,
            "evidence_summary": "fixture evidence",
        }
        for number, requirement in enumerate(CRITERIA, 1)
    ]
    rendered = _markdown(rows)
    assert rendered.count("| PASS |") == 42
    assert "| 42 |" in rendered
    assert ACCEPTANCE_VERSION.endswith("/v3")


def test_acceptance_criteria_are_not_unconditional_passes() -> None:
    source = inspect.getsource(acceptance.publish_acceptance_audit)

    assert ":(True" not in source
    assert "(True," not in source
    assert "phase4_pilot_execution/v1/audit/manifest.json" in source
    assert "restart_validation_v3/manifest.json" in source
    assert "provider_enrichment_v4/manifest.json" in source
