from biominer.gbif_quality.acceptance import CRITERIA, SCHEMA, _markdown


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
