from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_write_report_pack():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_report_pack.py"
    spec = importlib.util.spec_from_file_location("generate_report_pack", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.write_report_pack


write_report_pack = _load_write_report_pack()


REQUIRED_REPORTS = {
    "query_term_totals.json",
    "query_term_totals.md",
    "bbox_coverage_profile.json",
    "bbox_coverage_profile.md",
    "occurrence_bin_profile.json",
    "occurrence_bin_profile.md",
    "life_stage_profile.json",
    "no_geo_profile.json",
    "comment_expansion_profile.json",
    "api_budget_profile.json",
    "code_cleanup_report.md",
    "agents_update_recommendations.json",
}

REQUIRED_METRICS = {
    "gold_count",
    "silver_count",
    "bronze_count",
    "in_review_no_geo_count",
    "adult_butterfly_count",
    "egg_count",
    "caterpillar_count",
    "larva_count",
    "pupa_count",
    "chrysalis_count",
    "museum_specimen_count",
    "artwork_count",
    "tattoo_count",
    "ai_generated_count",
    "other_insect_count",
    "downloaded_images_deleted_count",
    "duplicate_skipped_count",
    "api_calls_used",
}


def test_report_pack_generator_writes_phase_7_required_reports(tmp_path) -> None:
    write_report_pack(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == REQUIRED_REPORTS


def test_json_reports_include_required_metrics_without_guessing(tmp_path) -> None:
    write_report_pack(tmp_path)

    for path in tmp_path.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED_METRICS.issubset(report)
        assert report["unsupported_metrics_policy"] == "Unsupported metrics are null or not_instrumented, never guessed."


def test_life_stage_report_includes_life_stage_counts(tmp_path) -> None:
    write_report_pack(tmp_path)

    profile = json.loads((tmp_path / "life_stage_profile.json").read_text(encoding="utf-8"))

    assert profile["adult_butterfly_count"] is None
    assert profile["egg_count"] is None
    assert profile["caterpillar_count"] is None
    assert profile["larva_count"] is None
    assert profile["pupa_count"] is None
    assert profile["chrysalis_count"] is None
    assert profile["life_stage_values"] == ["adult_butterfly", "egg", "caterpillar", "larva", "pupa", "chrysalis", "unknown"]


def test_reports_focus_on_triage_comments_no_geo_and_api_budget(tmp_path) -> None:
    write_report_pack(tmp_path)

    occurrence = json.loads((tmp_path / "occurrence_bin_profile.json").read_text(encoding="utf-8"))
    comments = json.loads((tmp_path / "comment_expansion_profile.json").read_text(encoding="utf-8"))
    no_geo = json.loads((tmp_path / "no_geo_profile.json").read_text(encoding="utf-8"))
    api_budget = json.loads((tmp_path / "api_budget_profile.json").read_text(encoding="utf-8"))

    assert occurrence["screening_evidence_only"] is True
    assert occurrence["darwin_core_dependency"] is False
    assert comments["comments_fetch_scope"] == "selected candidate records only"
    assert comments["comments_do_not_override_triage"] is True
    assert no_geo["no_geo_bin"] == "in_review/no_geo"
    assert api_budget["api_calls_used"] is None
    assert api_budget["api_calls_in_window"] == "not_instrumented"


def test_code_cleanup_report_marks_darwin_core_compatibility_only(tmp_path) -> None:
    write_report_pack(tmp_path)

    report = (tmp_path / "code_cleanup_report.md").read_text(encoding="utf-8")

    assert "Darwin Core compatibility code must not be used as the active image-triage" in report
    assert "Removal condition" in report
    assert "bioclip_run_summary" in report
