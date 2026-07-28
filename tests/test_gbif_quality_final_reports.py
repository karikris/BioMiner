from pathlib import Path
import inspect

import pytest

from biominer.gbif_quality.final_reports import (
    LATEST_EVIDENCE_MANIFESTS,
    REPORT_NAMES,
    REPORT_VERSION,
    _report,
    publish_final_reports,
)


def test_final_report_contract_is_complete_and_status_explicit() -> None:
    assert len(REPORT_NAMES) == 19
    assert len(set(REPORT_NAMES)) == len(REPORT_NAMES)
    assert "ai_readiness.md" in REPORT_NAMES
    assert "performance_and_reproducibility.md" in REPORT_NAMES
    text = _report("Example", "Reachability is NOT_TESTED.")
    assert text.startswith("# Example")
    assert "UNKNOWN" in text
    assert "NOT_TESTED" in text
    assert REPORT_VERSION.endswith("/v3")


def test_final_reports_require_current_execution_evidence() -> None:
    assert {
        "quality_results/restart_validation_v3/manifest.json",
        "provider_enrichment_v4/manifest.json",
        "quality_results/provider_archive_review/v1/manifest.json",
        "derived_assertions/geography_v3/manifest.json",
        "quality_results/phase3_v3/manifest.json",
        "quality_results/phase4_pilot_execution/v1/audit/manifest.json",
    } <= set(LATEST_EVIDENCE_MANIFESTS)
    source = inspect.getsource(publish_final_reports)
    assert "data_manifest_sha256s" in source
    assert "rights_blocked_zero_attempts" in source
    assert "pilot_acceptance_manifest_sha256" in source


def test_final_reports_are_create_only(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    report_root.mkdir()
    with pytest.raises(FileExistsError):
        publish_final_reports(
            data_root=tmp_path / "missing-data",
            report_root=report_root,
            code_commit="commit",
            full_resolution_manifest=tmp_path / "missing-full-resolution-manifest.json",
        )
