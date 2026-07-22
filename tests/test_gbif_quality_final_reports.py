from biominer.gbif_quality.final_reports import REPORT_NAMES, _report


def test_final_report_contract_is_complete_and_status_explicit() -> None:
    assert len(REPORT_NAMES) == 19
    assert len(set(REPORT_NAMES)) == len(REPORT_NAMES)
    assert "ai_readiness.md" in REPORT_NAMES
    assert "performance_and_reproducibility.md" in REPORT_NAMES
    text = _report("Example", "Reachability is NOT_TESTED.")
    assert text.startswith("# Example")
    assert "UNKNOWN" in text
    assert "NOT_TESTED" in text
