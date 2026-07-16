from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORT_PATH = Path("reports/phase15/prototype_go_no_go.json")


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_phase15_prototype_entry_is_go_with_narrow_authorization() -> None:
    report = _report()
    authorization = report["authorization"]
    controls = report["controls"]

    assert report["decision"] == "GO"
    assert authorization["prototype_integration"] is True
    assert authorization["explicit_prototype_mode_only"] is True
    assert authorization["production_default_change"] is False
    assert authorization["scientific_release"] is False
    assert authorization["public_reference_image_display"] is False
    assert controls["fail_closed"] is True
    assert controls["missing_or_contradictory_evidence_decision"] == "NO_GO"
    assert controls["s3_used"] is False


def test_phase15_all_fourteen_required_gates_pass() -> None:
    report = _report()
    gates = report["gates"]

    assert len(gates) == 14
    assert [gate["gate_id"] for gate in gates] == [
        "01_support_bank_frozen",
        "02_bank_marked_prototype_only",
        "03_licence_and_attribution_complete",
        "04_exact_duplicates_resolved",
        "05_routes_separate",
        "06_frozen_embeddings_exist",
        "07_target_and_competitor_scoring",
        "08_target_never_hierarchy_pruned",
        "09_benchmark_executed",
        "10_staged_flickr_executed",
        "11_fingerprints_exist",
        "12_limitations_explicit",
        "13_no_probability_misrepresentation",
        "14_no_false_human_verification",
    ]
    assert all(gate["required"] is True for gate in gates)
    assert all(gate["passed"] is True for gate in gates)
    assert all(gate["evidence"] for gate in gates)
    assert report["gate_summary"] == {
        "required": 14,
        "passed": 14,
        "failed": 0,
        "pass_rate": 1.0,
    }


def test_phase15_go_no_go_provenance_hashes_match_tracked_inputs() -> None:
    report = _report()

    for item in report["input_provenance"].values():
        path = Path(item["uri"])
        assert path.is_file()
        assert item["sha256"] == (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_phase15_go_no_go_preserves_scientific_semantics() -> None:
    report = _report()
    gates = {gate["gate_id"]: gate for gate in report["gates"]}

    assert "research-only" in gates["03_licence_and_attribution_complete"]["limitation"]
    assert (
        "not classification accuracy"
        in gates["07_target_and_competitor_scoring"]["limitation"]
    )
    assert "uncalibrated" in gates["13_no_probability_misrepresentation"]["limitation"]
    assert "expert review" in gates["14_no_false_human_verification"]["limitation"]
    assert len(report["prototype_limitations"]) >= 10
    assert report["next_task"] == ("15.1_add_explicit_prototype_classification_default")
