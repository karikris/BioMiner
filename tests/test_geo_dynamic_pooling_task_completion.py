"""Validation for immutable dynamic-pooling task completion records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/task_0_1_completion.json"
TASK_1_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_1_1_completion.json"
TASK_1_2_REPORT = ROOT / "reports/geo_dynamic_pooling/task_1_2_completion.json"
TASK_2_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_2_1_completion.json"
TASK_3_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_3_1_completion.json"
TASK_4_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_4_1_completion.json"
TASK_4_2_REPORT = ROOT / "reports/geo_dynamic_pooling/task_4_2_completion.json"
TASK_5_1_REPORT = ROOT / "reports/geo_dynamic_pooling/task_5_1_completion.json"
TASK_5_2_REPORT = ROOT / "reports/geo_dynamic_pooling/task_5_2_completion.json"
PUSH_LEDGER = ROOT / "provenance/task_pushes.jsonl"


def _report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _task_1_1_report() -> dict[str, object]:
    return json.loads(TASK_1_1_REPORT.read_text(encoding="utf-8"))


def _task_1_2_report() -> dict[str, object]:
    return json.loads(TASK_1_2_REPORT.read_text(encoding="utf-8"))


def _task_2_1_report() -> dict[str, object]:
    return json.loads(TASK_2_1_REPORT.read_text(encoding="utf-8"))


def _task_3_1_report() -> dict[str, object]:
    return json.loads(TASK_3_1_REPORT.read_text(encoding="utf-8"))


def _task_4_1_report() -> dict[str, object]:
    return json.loads(TASK_4_1_REPORT.read_text(encoding="utf-8"))


def _task_4_2_report() -> dict[str, object]:
    return json.loads(TASK_4_2_REPORT.read_text(encoding="utf-8"))


def _task_5_1_report() -> dict[str, object]:
    return json.loads(TASK_5_1_REPORT.read_text(encoding="utf-8"))


def _task_5_2_report() -> dict[str, object]:
    return json.loads(TASK_5_2_REPORT.read_text(encoding="utf-8"))


def test_task_0_1_completion_records_exact_commits_and_green_gate() -> None:
    report = _report()

    assert report["task_id"] == "geo-pool-0.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "bc821bffd11ad32877aa4a704aa2bb7a0d636ab6",
        "00f987f6f23acba7135b0b349412a2e7248e933f",
        "27c93f2745e6e8d869c338623c5becee9323ba47",
    ]
    assert report["gate"]["baseline_suite"]["passed"] == 2541
    assert report["gate"]["baseline_suite"]["result"] == "passed"
    assert report["gate"]["report_validation"]["passed"] == 10
    assert report["gate"]["provenance"]["result"] == "passed"


def test_task_0_1_push_event_matches_report() -> None:
    report = _report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-0.1")

    assert event["schema_version"] == "biominer-task-push-event-v1.0.0"
    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_0_1_report_preserves_blocked_scientific_claims() -> None:
    blocked = _report()["claims"]["blocked"]

    assert any("Dynamic global/local pool" in claim for claim in blocked)
    assert any("occurrence release" in claim for claim in blocked)


def test_task_1_1_completion_records_contracts_commits_and_green_gates() -> None:
    report = _task_1_1_report()

    assert report["task_id"] == "geo-pool-1.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "d9e4365af0b65cd147ccc0f44fc543f9b02c96ce",
        "315e6f3b04dc2a18bb1679c26f92563f5a7f1ade",
        "387887bb86d7c276d83eca6e29f328ea73b8d676",
        "5e87aa3171655ae2f8883287a2661f4a41839aac",
    ]
    assert len(report["artifacts"]) == 7
    assert report["gate"]["schema_and_determinism_tests"]["passed"] == 98
    assert report["gate"]["full_regression"]["passed"] == 2628
    assert report["gate"]["provenance"]["jsonl_records_validated"] == 105


def test_task_1_1_push_event_matches_report() -> None:
    report = _task_1_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-1.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_1_1_report_blocks_unearned_scientific_claims() -> None:
    blocked = _task_1_1_report()["claims"]["blocked"]

    assert any("empirically superior" in claim for claim in blocked)
    assert any("human review" in claim for claim in blocked)
    assert any("publication" in claim for claim in blocked)


def test_task_1_2_completion_records_handoffs_pins_and_green_gates() -> None:
    report = _task_1_2_report()

    assert report["task_id"] == "geo-pool-1.2"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "1a85c1951d650e5db3fa3ef058b9772abe964bf4",
        "1e1bdacc8ec902c381ea568c51e04ad15bdb7636",
        "f8d52a0236357d10a302c77e115fd27c9cbfc985",
    ]
    assert report["consumer_pins"]["taxalens"]["commit"] == (
        "c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc"
    )
    assert report["consumer_pins"]["butterflylens"]["commit"] == (
        "1cea643623f2f20a2bea72afc754c7b194db3278"
    )
    assert report["gate"]["cross_repository_compatibility"]["passed"] == 33
    assert report["gate"]["full_regression"]["passed"] == 2660
    assert report["gate"]["provenance"]["jsonl_records_validated"] == 109


def test_task_1_2_push_event_matches_report() -> None:
    report = _task_1_2_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-1.2")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_1_2_report_blocks_import_and_release_claims() -> None:
    blocked = _task_1_2_report()["claims"]["blocked"]

    assert any("live TaxaLens or ButterflyLens import" in claim for claim in blocked)
    assert any("reviewer assignment" in claim for claim in blocked)
    assert any("occurrence release" in claim for claim in blocked)
    assert any("production deployment" in claim for claim in blocked)


def test_task_2_1_completion_records_artifacts_commits_and_green_gates() -> None:
    report = _task_2_1_report()

    assert report["task_id"] == "geo-pool-2.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "fd5a4d6d79f889c16adb78b2227adc05c2cf478b",
        "8165ae1d58db20796eacff718fca5173840a12b9",
        "e4245a6d652aca8ff20a198d957897d1c91c00fc",
        "cd37037a98a9239c2ff4bb5d30c661e9c950ce66",
    ]
    assert [item["file"] for item in report["artifacts"]] == [
        "normalized_reference_geography.parquet",
        "reference_geography_index.parquet",
        "global_reference_anchors.parquet",
        "geographic_reference_neighbours.parquet",
        "reference_geography_index_manifest.json",
    ]
    assert report["gate"]["reference_and_geography_suite"]["passed"] == 133
    assert report["gate"]["artifact_reproducibility"]["qa_status"] == "passed"
    assert (
        report["gate"]["artifact_reproducibility"]["physical_checksum_status"]
        == "complete"
    )
    assert report["gate"]["final_full_regression"]["passed"] == 2730
    assert report["gate"]["provenance"]["jsonl_records_validated"] == 114


def test_task_2_1_push_event_matches_report() -> None:
    report = _task_2_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-2.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_2_1_report_blocks_live_identity_and_release_claims() -> None:
    blocked = _task_2_1_report()["claims"]["blocked"]

    assert any("live production reference bank" in claim for claim in blocked)
    assert any("human verified or ground truth" in claim for claim in blocked)
    assert any(
        "geographic identity or biological absence" in claim for claim in blocked
    )
    assert any(
        "occurrence release or production deployment" in claim for claim in blocked
    )


def test_task_3_1_completion_records_artifacts_commits_and_green_gates() -> None:
    report = _task_3_1_report()

    assert report["task_id"] == "geo-pool-3.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "1d212bcbfad04f0644f5c3d525afa7ae39441651",
        "c7110954ed52478c90a44e7eae4474fecf66ff5a",
        "fe03e46bf00c9c064d1f52c5d83320730a5f86fa",
    ]
    assert [item["file"] for item in report["artifacts"]] == [
        "flickr_photo_embedding_units.parquet",
        "flickr_scoring_units.parquet",
        "flickr_scoring_unit_associations.parquet",
        "flickr_scoring_unit_candidates.parquet",
        "flickr_scoring_geography.parquet",
        "flickr_geo_taxon_partitions.parquet",
        "flickr_partition_summary.parquet",
    ]
    assert report["gate"]["canonical_grain_suite"]["passed"] == 118
    assert report["gate"]["artifact_reproducibility"][
        "model_input_reuse_count"
    ] == 1
    assert report["gate"]["full_regression"]["passed"] == 2745
    assert report["gate"]["provenance"]["jsonl_records"] == 118


def test_task_3_1_push_event_matches_report() -> None:
    report = _task_3_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-3.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_3_1_report_blocks_live_identity_and_release_claims() -> None:
    report = _task_3_1_report()
    blocked = report["claims"]["blocked"]

    assert any("live Flickr workload" in claim for claim in blocked)
    assert any("prove taxonomic identity or absence" in claim for claim in blocked)
    assert any("occurrence release or production deployment" in claim for claim in blocked)
    impact = report["githits_architecture_impact"]
    assert impact["direct_external_code_contribution"] == "none"
    assert "low direct implementation impact" in impact["assessment"]


def test_task_4_1_completion_records_strategies_commits_and_green_gates() -> None:
    report = _task_4_1_report()

    assert report["task_id"] == "geo-pool-4.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "efb0c576d085986ea4a3549d551ddc8ae960ee5e",
        "88e662861dc86ec25ff443035acf18b395f1cc26",
        "b372ce18d6be62c1b66025b700d5c4e4a884428c",
    ]
    assert [item["name"] for item in report["strategies"]] == [
        "geography_first",
        "family_first_safe",
        "parallel_family_geography_union",
    ]
    assert report["gate"]["strategy_and_target_preservation_suite"]["passed"] == 82
    assert report["gate"]["artifact_reproducibility"]["identical_membership"]
    assert report["gate"]["full_regression"]["passed"] == 2756
    assert report["gate"]["provenance"]["jsonl_records"] == 122
    assert report["selection_state"]["selected_strategy"] is None
    assert report["selection_state"]["production_default_changed"] is False


def test_task_4_1_push_event_matches_report() -> None:
    report = _task_4_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-4.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_4_1_report_blocks_unearned_strategy_and_release_claims() -> None:
    report = _task_4_1_report()
    blocked = report["claims"]["blocked"]

    assert any("empirically superior" in claim for claim in blocked)
    assert any("proves taxonomic identity or absence" in claim for claim in blocked)
    assert any("occurrence release or production deployment" in claim for claim in blocked)
    assert report["githits_architecture_impact"][
        "direct_external_code_contribution"
    ] == "none"


def test_task_4_2_completion_records_metrics_counterfactual_and_green_gates() -> None:
    report = _task_4_2_report()

    assert report["task_id"] == "geo-pool-4.2"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "79f94b5ce0da3ff740e8511ebf60378633c763df",
        "c45a2c195093ee3ea684f3d80cd40152db9f4fd7",
        "4735050bd1a8206d44a22d6032cfbbfa75b40635",
    ]
    assert len(report["metrics_contract"]["measures"]) == 9
    assert report["family_pruning_contract"][
        "production_candidate_membership_changed"
    ] is False
    fixture = report["fixture_ablation"]
    assert fixture["metric_rows"] == 18
    assert fixture["family_pruning_counterfactual"][
        "correct_species_lost_count"
    ] == 1
    assert fixture["validation_gate_passed"] is False
    assert fixture["failed_checks"] == ["non_fixture_evidence"]
    assert report["gate"]["strategy_evaluation_suite"]["passed"] == 93
    assert report["gate"]["full_regression"]["passed"] == 2766
    assert report["gate"]["provenance"]["jsonl_records"] == 126


def test_task_4_2_push_event_matches_report() -> None:
    report = _task_4_2_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-4.2")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_4_2_report_blocks_unearned_selection_and_release_claims() -> None:
    report = _task_4_2_report()
    state = report["selection_state"]
    blocked = report["claims"]["blocked"]

    assert state["intended_candidate"] == "parallel_family_geography_union"
    assert state["selected_strategy"] is None
    assert state["production_default_changed"] is False
    assert state["superiority_claimed"] is False
    assert any("empirically or universally superior" in claim for claim in blocked)
    assert any("selected or made default" in claim for claim in blocked)
    assert any("occurrence release or production deployment" in claim for claim in blocked)
    impact = report["githits_architecture_impact"]
    assert impact["calls_made_for_task"] == 0
    assert impact["direct_external_code_contribution"] == "none"


def test_task_5_1_completion_records_policy_planner_and_green_gates() -> None:
    report = _task_5_1_report()

    assert report["task_id"] == "geo-pool-5.1"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "8f5b2f9f25565e95868de5d900b45f1cdc663df7",
        "86cedcf9d01348770eca95a96f46bdf37dd84ba9",
        "746e5259922c057e0f7864643a861844c8fdf03f",
        "b1ae26d15e6b3c866ea57c5c8a972444a4860e0d",
    ]
    assert len(report["policy"]["controls"]) == 10
    assert report["fixture_round_trip"]["members"]["rows"] == 3
    assert report["fixture_round_trip"]["members"]["global"] == 2
    assert report["fixture_round_trip"]["members"]["local"] == 1
    assert report["fixture_round_trip"]["coverage"]["complete"] == 1
    assert report["gate"]["pool_planner_suite"]["passed"] == 82
    assert report["gate"]["full_regression"]["passed"] == 2796
    assert report["gate"]["provenance"]["jsonl_records"] == 131


def test_task_5_1_push_event_matches_report() -> None:
    report = _task_5_1_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-5.1")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_5_1_report_blocks_live_pool_and_release_claims() -> None:
    report = _task_5_1_report()
    state = report["selection_state"]
    blocked = report["claims"]["blocked"]

    assert state["live_dynamic_pool_planned"] is False
    assert state["production_default_changed"] is False
    assert any("live dynamic reference pool" in claim for claim in blocked)
    assert any("prove taxon absence" in claim for claim in blocked)
    assert any("occurrence release or production deployment" in claim for claim in blocked)
    impact = report["githits_architecture_impact"]
    assert impact["calls_made_for_task"] == 0
    assert impact["direct_external_code_contribution"] == "none"


def test_task_5_2_completion_records_bounded_cached_expansion() -> None:
    report = _task_5_2_report()

    assert report["task_id"] == "geo-pool-5.2"
    assert report["status"] == "completed"
    assert [item["commit"] for item in report["task_commits"]] == [
        "fb80aa3bd0aff02149e70d31e3679ff69f67848a",
        "b6d9af957d27ea0f6bb012e030be089d8435f437",
        "d6e03450bd301bba2ae3ea1e6ffcadb05059d8f9",
    ]
    assert len(report["expansion_evidence"]["signals"]) == 11
    fixture = report["fixture_round_trip"]
    assert fixture["initial"]["members"] == 2
    assert fixture["expanded"]["members"] == 5
    assert fixture["cache_reuse"]["retained_reference_embeddings"] == 2
    assert fixture["cache_reuse"]["added_reference_embeddings"] == 3
    assert fixture["cache_reuse"]["encoder_invocations"] == 0
    assert fixture["decisions"]["stop_reason"] == (
        "round_complete_rescore_required"
    )
    assert report["gate"]["expansion_cache_suite"]["passed"] == 117
    assert report["gate"]["full_regression"]["passed"] == 2816
    assert report["gate"]["provenance"]["jsonl_records"] == 135


def test_task_5_2_push_event_matches_report() -> None:
    report = _task_5_2_report()
    events = [
        json.loads(line)
        for line in PUSH_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["task_id"] == "geo-pool-5.2")

    assert event["verified_remote_sha"] == report["push"]["verified_remote_sha"]
    assert event["pushed_through_sha"] == report["push"]["pushed_through_sha"]
    assert event["status"] == report["push"]["status"] == "verified"


def test_task_5_2_report_blocks_live_science_and_release_claims() -> None:
    report = _task_5_2_report()
    state = report["selection_state"]
    blocked = report["claims"]["blocked"]

    assert state["live_expansion_executed"] is False
    assert state["calibrated_probability_available"] is False
    assert state["production_release_authorized"] is False
    assert any("live Flickr image" in claim for claim in blocked)
    assert any("calibrated confidence" in claim for claim in blocked)
    assert any("occurrence release or production deployment" in claim for claim in blocked)
    impact = report["githits_architecture_impact"]
    assert impact["calls_made_for_task"] == 0
    assert impact["direct_external_code_contribution"] == "none"
