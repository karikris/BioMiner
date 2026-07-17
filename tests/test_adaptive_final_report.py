from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "gbif_fast_start" / "final_report.json"


def test_final_report_has_complete_ledger_and_honest_acceptance() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == "gbif-fast-start-final-report-v1.0.0"
    assert report["starting_main_sha"] == (
        "dcd494321abc0666ea692b5759f84bc4c7e08ba9"
    )
    assert report["ending_implementation_sha"] == (
        "477eaface3d1f5efa51255550f0ef8d6a7740f35"
    )
    assert report["branch"] == "main"

    ledger = report["task_ledger"]
    assert len(ledger) == 73
    assert len({item["task_id"] for item in ledger}) == 73
    assert len({item["commit"] for item in ledger}) == 73
    assert {item["push"] for item in ledger} == {
        "verified_ancestor_of_origin/main"
    }
    assert ledger[-1]["task_id"] == "gbif-fast-13.4"

    criteria = report["acceptance_criteria"]
    assert [item["id"] for item in criteria] == list(range(1, 67))
    deviations = [item for item in criteria if item["status"] != "passed"]
    assert deviations == [
        {
            "id": 60,
            "status": "not_met_user_authorized_deviation",
            "criterion": (
                "Work occurs only on codex/adaptive-gbif-reference-default."
            ),
            "evidence": (
                "User explicitly required main; criterion retained as an "
                "authorized deviation"
            ),
        }
    ]
    assert report["acceptance_summary"] == {
        "passed": 65,
        "user_authorized_deviations": 1,
        "failed": 0,
        "total": 66,
    }

    assert report["policy"]["default_reference_admission_mode"] == (
        "adaptive_gbif_fast_start"
    )
    assert report["policy"]["strict_mode"] == "human_verified_strict"
    assert report["pilot"]["unreviewed_flickr_records_exported"] == 0
    assert report["pilot"]["quality_metrics"]["status"] == "insufficient_sample"
    assert all(value is None for key, value in report["pilot"]["quality_metrics"].items() if key != "status")
    assert report["verification"]["full_suite"]["passed"] == 2531
    assert report["verification"]["full_suite_history"][-1]["status"] == "passed"
    assert report["unexecuted_live_steps"]
    assert report["human_review_requirements"]
    assert report["remaining_limitations"]
    assert report["recommended_merge_procedure"]
