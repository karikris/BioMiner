from __future__ import annotations

import hashlib
import json
from pathlib import Path


MATRIX_PATH = Path("config/pilot/papilio_demoleus_phase14_experiment_matrix.json")
REFERENCE_SOURCE_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_reference_source_manifest.json"
)
PROTOTYPE_ACQUISITION_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_acquisition_manifest.json"
)
PROTOTYPE_SELECTION_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_selection_manifest.json"
)
PROTOTYPE_DOWNLOAD_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_download_manifest.json"
)
PROTOTYPE_DUPLICATE_RESOLUTION_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/"
    "pilot_prototype_duplicate_resolution_manifest.json"
)
PROTOTYPE_QA_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_qa_manifest.json"
)
PROTOTYPE_SUPPORT_BANK_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_support_bank_manifest.json"
)
PROTOTYPE_VISION_SMOKE_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_vision_smoke_manifest.json"
)
PROTOTYPE_EMBEDDINGS_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_embeddings_manifest.json"
)
PROTOTYPE_B0_B16_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_b0_b16_manifest.json"
)
PROTOTYPE_POLICY_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_prototype_policy_manifest.json"
)
BUILD_WEEK_REPORT_MANIFEST_PATH = Path(
    "examples/species/papilio_demoleus/pilot_build_week_prototype_report_manifest.json"
)


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _reference_source_manifest() -> dict[str, object]:
    return json.loads(REFERENCE_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_acquisition_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_ACQUISITION_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_selection_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_download_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_DOWNLOAD_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_qa_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_QA_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_support_bank_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_SUPPORT_BANK_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_duplicate_resolution_manifest() -> dict[str, object]:
    return json.loads(
        PROTOTYPE_DUPLICATE_RESOLUTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def _prototype_vision_smoke_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_VISION_SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_embeddings_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_EMBEDDINGS_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_b0_b16_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_B0_B16_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_policy_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_POLICY_MANIFEST_PATH.read_text(encoding="utf-8"))


def _build_week_report_manifest() -> dict[str, object]:
    return json.loads(BUILD_WEEK_REPORT_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_phase14_matrix_freezes_local_vision_limit_and_external_execution() -> None:
    matrix = _matrix()
    execution = matrix["execution_contract"]

    assert execution["branch"] == "main"
    assert execution["local_build_verification_max_images"] == 5
    assert execution["large_bioclip_execution"] == "different_computer_required"
    assert execution["large_yoloe_execution"] == "different_computer_required"
    support_exception = execution["local_full_support_embedding_exception"]
    assert support_exception == {
        "authorized": True,
        "authorized_by": "explicit_user_instruction",
        "maximum_frozen_support_records": 81,
        "storage_backend": "local",
        "s3_permitted": False,
        "scope": "task_14.4.2_frozen_support_embeddings_only",
    }
    assert {
        experiment["execution_computer"] for experiment in matrix["experiments"]
    } == {"external_gpu_computer"}


def test_phase14_matrix_contains_exact_b0_through_b16_ablation_contract() -> None:
    matrix = _matrix()
    experiments = {
        experiment["experiment_id"]: experiment for experiment in matrix["experiments"]
    }

    assert list(experiments) == [f"B{index}" for index in range(17)]
    assert experiments["B0"]["higher_rank_candidate_pruning"] is True
    assert experiments["B1"]["higher_rank_candidate_pruning"] is False
    assert experiments["B1"]["target_always_scored"] is True
    assert experiments["B4"]["k"] == 5
    assert experiments["B7"]["implementation"] == "linear_svc_embedding"
    assert experiments["B8"]["implementation"] == "linear_svc_structured"
    assert experiments["B9"]["fit_partition"] == "calibration"
    assert experiments["B10"]["visual_inputs"] == ["raw_full_frame"]
    assert experiments["B12"]["visual_inputs"] == [
        "raw_full_frame",
        "focused_full_frame",
        "masked_full_frame",
    ]
    assert experiments["B13"]["reference_scope"] == "global"
    assert experiments["B14"]["reference_scope"] == ("geo_cluster_then_global_fallback")
    assert experiments["B15"]["evidence"] == ["frozen_image", "taxonomy_text"]
    assert experiments["B16"]["evidence"] == ["frozen_image"]


def test_phase14_matrix_blocks_benchmark_and_phase15_until_human_freeze() -> None:
    matrix = _matrix()
    gates = {gate["gate_id"]: gate for gate in matrix["prerequisite_gates"]}
    splits = matrix["split_policy"]
    labels = matrix["label_contract"]

    assert gates["phase14.3_human_review_and_freeze"]["status"] == (
        "blocked_awaiting_human_review"
    )
    assert gates["phase14.4_off_machine_benchmark"]["status"] == (
        "blocked_by_phase14.3"
    )
    assert splits["partition_weights"] == {
        "support_train": 55,
        "model_selection": 15,
        "calibration": 15,
        "final_test": 15,
    }
    assert splits["final_test_may_select_model_or_threshold"] is False
    assert labels["flickr_query_match_is_label"] is False
    assert labels["gbif_taxon_match_is_human_image_label"] is False
    assert matrix["phase15_default_gate"]["authorized"] is False
    assert (
        matrix["phase15_default_gate"]["current_default_must_remain_unchanged"] is True
    )


def test_phase14_reference_source_manifest_preserves_metadata_only_boundary() -> None:
    manifest = _reference_source_manifest()
    execution = manifest["execution_constraints"]
    counts = manifest["counts"]
    shortfalls = manifest["shortfalls"]

    assert manifest["status"] == (
        "metadata_complete_awaiting_human_review_and_additional_sources"
    )
    assert manifest["candidate_semantics"] == (
        "source_taxon_match_not_human_verified_image_label"
    )
    assert execution["local_build_verification_max_images"] == 5
    assert execution["images_downloaded"] == 0
    assert execution["bioclip_images_processed"] == 0
    assert execution["yoloe_images_processed"] == 0
    assert counts["observation_count"] == 91_176
    assert counts["media_candidate_count"] == 142_873
    assert counts["human_verified_source_media_count"] == 0
    assert shortfalls["unresolved_group_count"] == 2
    assert manifest["experiment_matrix"]["status"] == "blocked_by_phase14.3"


def test_phase14_prototype_acquisition_manifest_records_bounded_real_candidates() -> (
    None
):
    manifest = _prototype_acquisition_manifest()
    bounded = manifest["bounded_biological_negative_acquisition"]
    counts = manifest["counts"]

    assert manifest["task"] == "14.2.5"
    assert manifest["status"] == "complete_with_documented_shortfalls"
    assert manifest["prototype_only"] is True
    assert bounded["maximum_records_per_query"] == 3000
    assert bounded["query_count"] == 11
    assert bounded["source_images_downloaded"] == 0
    assert counts["selected_for_download_count"] == 93
    assert counts["selected_independent_observation_count"] == 93
    assert counts["human_verified_count"] == 0
    assert counts["provider_supported_selected_count"] == 93
    assert counts["shortfall_scope_count"] == 34
    assert manifest["score_semantics"]["score_is_probability"] is False
    assert (
        manifest["verification_semantics"]["provider_supported_is_human_verified"]
        is False
    )


def test_phase14_prototype_acquisition_manifest_tracks_required_artifacts() -> None:
    manifest = _prototype_acquisition_manifest()
    artifacts = manifest["output_artifacts"]

    assert {
        "reference_acquisition_plan",
        "prototype_reference_source_summary",
        "prototype_reference_shortfalls",
        "prototype_reference_selections",
        "prototype_reference_download_candidates",
    } <= set(artifacts)
    assert artifacts["reference_acquisition_plan"]["row_count"] == 34
    assert artifacts["prototype_reference_source_summary"]["row_count"] == 101
    assert artifacts["prototype_reference_shortfalls"]["row_count"] == 45
    assert artifacts["prototype_reference_selections"]["row_count"] == 93
    assert artifacts["prototype_reference_download_candidates"]["row_count"] == 93
    assert all(
        str(artifact["sha256"]).startswith("sha256:")
        and len(str(artifact["sha256"])) == 71
        for artifact in artifacts.values()
    )


def test_phase14_prototype_selection_manifest_enforces_independent_observations() -> (
    None
):
    manifest = _prototype_selection_manifest()
    counts = manifest["counts"]
    policy = manifest["selection_policy"]

    assert manifest["task"] == "14.3.1"
    assert counts["selected_reference_count"] == 93
    assert counts["selected_independent_observation_count"] == 93
    assert policy["one_media_per_independent_observation"] is True
    assert policy["excluded_trust_levels"] == ["R5"]
    assert manifest["trust_distribution"]["R5"] == 0
    assert manifest["next_task"] == "14.3.2_download_selected_media"


def test_phase14_prototype_download_manifest_proves_durable_validation() -> None:
    manifest = _prototype_download_manifest()
    counts = manifest["counts"]
    validation = manifest["validation"]
    resume = manifest["resume_verification"]

    assert manifest["task"] == "14.3.2"
    assert manifest["status"] == "complete"
    assert manifest["prototype_only"] is True
    assert counts["selected"] == counts["committed"] == 93
    assert counts["valid_decodes"] == 93
    assert counts["quarantined"] == counts["errors"] == 0
    assert counts["staging_object_count_after_cleanup"] == 0
    assert counts["local_temporary_image_count_after_run"] == 0
    assert validation["selection_identity_exact"] is True
    assert validation["all_s3_object_sizes_revalidated"] is True
    assert validation["all_s3_object_sha256_revalidated"] is True
    assert validation["local_temporary_images_deleted"] is True
    assert validation["unique_sha256_count"] == 93
    assert validation["unique_perceptual_hash_count"] == 93
    assert resume["resumed"] == 93
    assert resume["http_requests"] == 0
    assert resume["inventory_sha256_unchanged"] is True
    assert manifest["next_task"] == "14.3.3_resolve_duplicates"


def test_phase14_prototype_download_manifest_retains_prototype_semantics() -> None:
    manifest = _prototype_download_manifest()
    semantics = manifest["verification_semantics"]
    artifacts = manifest["artifacts"]

    assert semantics["provider_supported_is_human_verified"] is False
    assert semantics["human_taxonomic_verification_complete"] is False
    assert semantics["model_output_used_as_taxonomic_validation"] is False
    assert artifacts["reference_media_objects"]["row_count"] == 93
    assert all(
        str(value["uri"]).startswith("s3://")
        and str(value["sha256"]).startswith("sha256:")
        and len(str(value["sha256"])) == 71
        for value in artifacts.values()
    )


def test_phase14_duplicate_resolution_separates_retryable_failures() -> None:
    manifest = _prototype_duplicate_resolution_manifest()
    counts = manifest["counts"]
    semantics = manifest["semantics"]

    assert manifest["task"] == "14.3.3"
    assert manifest["status"] == "complete_with_retryable_operational_failures"
    assert manifest["execution"]["storage_backend"] == "local"
    assert manifest["execution"]["s3_deferred"] is True
    assert counts["selected"] == 93
    assert counts["valid_media"] == counts["eligible_canonical_media"] == 83
    assert counts["retryable_operational_failures"] == 10
    assert counts["duplicate_relationships"] == 0
    assert counts["noncanonical_duplicates"] == 0
    assert len(manifest["retryable_reference_media_ids"]) == 10
    assert semantics["operational_failures_are_biological_negatives"] is False
    assert semantics["eligible_rows_may_advance_to_automated_qa"] is True
    assert semantics["retryable_rows_may_advance_to_automated_qa"] is False
    assert manifest["next_task"] == "14.3.4_automated_prototype_qa"


def test_phase14_duplicate_resolution_artifacts_are_hashed_and_untracked() -> None:
    manifest = _prototype_duplicate_resolution_manifest()

    assert manifest["execution"]["local_storage_ignored_by_git"] is True
    assert all(
        str(value["uri"]).startswith("runs/")
        and str(value["sha256"]).startswith("sha256:")
        and len(str(value["sha256"])) == 71
        and int(value["byte_count"]) > 0
        for value in manifest["artifacts"].values()
    )


def test_phase14_prototype_qa_routes_unmeasured_and_failed_rows_honestly() -> None:
    manifest = _prototype_qa_manifest()
    counts = manifest["counts"]
    semantics = manifest["semantics"]

    assert manifest["task"] == "14.3.4"
    assert manifest["execution"]["storage_backend"] == "local"
    assert manifest["execution"]["s3_deferred"] is True
    assert counts["selected"] == 93
    assert counts["locally_available"] == 83
    assert counts["needs_review"] == 81
    assert counts["excluded"] == 2
    assert counts["retryable_operational_failures"] == 10
    assert counts["licence_complete"] == 83
    assert counts["attribution_complete"] == 83
    assert semantics["automated_qa_is_human_taxonomic_verification"] is False
    assert semantics["operational_failures_are_biological_negatives"] is False
    assert semantics["unmeasured_visual_evidence_is_guessed"] is False
    assert manifest["next_task"] == "14.3.5_freeze_prototype_support_bank"


def test_phase14_prototype_qa_artifacts_are_hashed_and_untracked() -> None:
    manifest = _prototype_qa_manifest()

    assert manifest["execution"]["local_storage_ignored_by_git"] is True
    assert all(
        str(value["uri"]).startswith("runs/")
        and str(value["sha256"]).startswith("sha256:")
        and len(str(value["sha256"])) == 71
        and int(value["byte_count"]) > 0
        for value in manifest["artifacts"].values()
    )


def test_phase14_prototype_support_freeze_authorises_only_prototype_work() -> None:
    manifest = _prototype_support_bank_manifest()
    counts = manifest["counts"]

    assert manifest["task"] == "14.3.5"
    assert manifest["status"] == "prototype_ready_with_shortfalls"
    assert manifest["bank_status"] == "prototype_only"
    assert manifest["classification_authorised"] is True
    assert manifest["human_verification_complete"] is False
    assert manifest["human_verification_required_for_scientific_release"] is True
    assert manifest["execution"]["storage_backend"] == "local"
    assert manifest["execution"]["s3_deferred"] is True
    assert counts["selected"] == 93
    assert counts["prototype_support"] == 81
    assert counts["excluded"] == 12
    assert counts["retryable_operational_failures"] == 10
    assert (
        sum(
            counts[name]
            for name in (
                "support_train",
                "model_selection",
                "calibration",
                "final_test",
            )
        )
        == counts["prototype_support"]
    )
    assert manifest["semantics"]["provider_supported_is_human_verified"] is False
    assert manifest["next_task"] == "14.4.1_five_image_bioclip_yoloe_smoke"


def test_phase14_prototype_support_artifacts_are_hashed_and_untracked() -> None:
    manifest = _prototype_support_bank_manifest()

    assert manifest["execution"]["local_storage_ignored_by_git"] is True
    assert all(
        str(value["uri"]).startswith("runs/")
        and str(value["sha256"]).startswith("sha256:")
        and len(str(value["sha256"])) == 71
        and int(value["byte_count"]) > 0
        for value in manifest["artifacts"].values()
    )


def test_phase14_five_image_smoke_records_exact_mps_runtime_evidence() -> None:
    manifest = _prototype_vision_smoke_manifest()

    assert manifest["task"] == "14.4.1"
    assert manifest["status"] == "passed"
    assert manifest["execution"]["completed_image_count"] == 5
    assert manifest["execution"]["maximum_images_per_invocation"] == 5
    assert manifest["execution"]["raw_images_committed"] is False
    assert manifest["runtime_preflight"]["bioclip"]["device_resolved"] == "mps"
    assert manifest["runtime_preflight"]["yoloe"]["device_resolved"] == "mps"
    assert manifest["bioclip"]["embedding_shape"] == [5, 1024]
    assert manifest["bioclip"]["finite_values"] is True
    assert manifest["bioclip"]["model_loads"] == 1
    assert manifest["bioclip"]["model_cache_hits"] == 1
    assert manifest["yoloe"]["worker_process_starts"] == 1
    assert manifest["yoloe"]["batch_sizes"] == [3, 2]
    assert manifest["semantics"]["smoke_pass_means_accuracy_validated"] is False
    assert manifest["next_task"] == "14.4.2_build_frozen_support_embeddings"


def test_phase14_full_prototype_embeddings_are_local_complete_and_resumable() -> None:
    manifest = _prototype_embeddings_manifest()
    counts = manifest["counts"]
    validation = manifest["validation"]

    assert manifest["task"] == "14.4.2"
    assert manifest["status"] == "complete"
    assert manifest["storage"]["backend"] == "local"
    assert manifest["storage"]["s3_used"] is False
    assert counts["frozen_support"] == counts["embedded"] == 81
    assert counts["retryable_failures"] == counts["operator_skips"] == 0
    assert counts["human_verified"] == 0
    assert counts["provider_supported"] == 81
    assert manifest["route_counts"] == {
        "adult_field": 80,
        "larval": 1,
        "pinned_specimen": 0,
    }
    assert validation["complete_support_coverage"] is True
    assert validation["finite_embeddings"] is True
    assert validation["unit_normalized_embeddings"] is True
    assert validation["adult_larval_specimen_embeddings_mixed"] is False
    assert validation["prototypes_consume_only_support_train"] is True
    assert validation["visual_neighbours_cross_routes"] is False
    assert validation["resume_without_model_recomputation"] is True
    assert manifest["semantics"]["provider_supported_is_human_verified"] is False
    assert manifest["next_task"] == "14.4.3_staged_flickr_prototype"


def test_phase14_full_prototype_embedding_artifacts_are_hashed_and_untracked() -> None:
    manifest = _prototype_embeddings_manifest()

    assert manifest["storage"]["ignored_by_git"] is True
    for artifact in manifest["artifacts"].values():
        assert str(artifact["uri"]).startswith("runs/")
        assert str(artifact["sha256"]).startswith("sha256:")
        assert len(str(artifact["sha256"])) == 71
        assert int(artifact["byte_count"]) > 0
    for artifact in (
        manifest["artifacts"]["prototype_reference_embeddings"],
        manifest["artifacts"]["prototype_reference_prototypes"],
        manifest["artifacts"]["prototype_visual_neighbour_species"],
    ):
        assert int(artifact["row_count"]) > 0
        assert len(str(artifact["semantic_fingerprint"])) == 71


def test_phase14_local_b0_b16_handoff_is_local_evidence_only() -> None:
    manifest = _prototype_b0_b16_manifest()

    assert manifest["task"] == "14.4.4"
    assert manifest["status"] == "complete_with_unavailable_visual_ablation_inputs"
    assert manifest["storage"]["backend"] == "local"
    assert manifest["storage"]["s3_used"] is False
    assert manifest["execution"]["records_scored"] == 81
    assert manifest["execution"]["records_skipped"] == 0
    assert manifest["artifacts"]["predictions"]["row_count"] == 1539
    assert manifest["artifacts"]["experiment_summary"]["row_count"] == 19
    assert manifest["semantics"]["classification_accuracy_reported"] is False
    assert manifest["semantics"]["provider_supported_metrics_are_accuracy"] is False
    assert manifest["visual_ablation"]["spatial_crops_used"] is False


def test_phase14_prototype_policy_is_local_uncalibrated_and_frozen() -> None:
    manifest = _prototype_policy_manifest()
    selected = manifest["selected_policy"]
    selection = manifest["selection_evidence"]
    calibration = manifest["calibration"]
    partitions = manifest["partition_contract"]
    frozen = manifest["frozen_identity"]

    assert manifest["task"] == "14.5"
    assert manifest["status"] == "selected"
    assert manifest["policy_status"] == "prototype_uncalibrated"
    assert manifest["storage"]["backend"] == "local"
    assert manifest["storage"]["s3_permitted"] is False
    assert manifest["storage"]["s3_used"] is False
    assert selected["experiment_id"] == "B13"
    assert selected["reference_scope"] == "global"
    assert selected["target_always_scored"] is True
    assert selection["record_count"] == 30
    assert selection["coverage_at_raw_margin_policy"] == 0.8
    assert manifest["b0_comparison"]["target_scoreability_improvement"] == 0.9
    assert calibration["human_verified_calibration_records"] == 0
    assert calibration["calibrator_fingerprint"] is None
    assert calibration["probabilities_emitted"] is False
    assert partitions["final_test_used_for_selection"] is False
    assert frozen["visual_input_version"] == "target-full-frame-visual-input-v2"
    assert frozen["calibrator_fingerprint"] is None
    assert str(frozen["classifier_fingerprint"]).startswith("sha256:")
    assert len(str(frozen["classifier_fingerprint"])) == 71
    assert manifest["next_task"] == "14.6_build_week_prototype_report"


def test_phase14_build_week_report_is_complete_but_prototype_only() -> None:
    manifest = _build_week_report_manifest()
    evidence = manifest["evidence"]
    policy = manifest["selected_policy"]
    staged = manifest["staged_flickr"]
    entry = manifest["phase15_prototype_entry"]

    assert manifest["task"] == "14.6"
    assert manifest["status"] == "complete_prototype_only"
    assert manifest["prototype_only"] is True
    assert manifest["scientific_release_authorized"] is False
    assert manifest["production_default_change_authorized"] is False
    assert manifest["storage"]["backend"] == "local"
    assert manifest["storage"]["s3_used"] is False
    assert evidence["prototype_support"] == evidence["provider_supported"] == 81
    assert evidence["human_verified"] == 0
    assert evidence["classification_accuracy_reported"] is False
    assert evidence["classification_accuracy"] is None
    assert evidence["scores_are_probabilities"] is False
    assert policy["experiment_id"] == "B13"
    assert policy["policy_status"] == "prototype_uncalibrated"
    assert policy["target_always_scored"] is True
    assert policy["raw_margin_threshold"] == 0.1
    assert staged["classified"] == staged["target_scored"] == 13_496
    assert staged["retryable_failures"] == 5
    assert entry["status"] == "ready_for_go_no_go_audit"
    assert entry["production_default_change_authorized"] is False
    assert manifest["next_task"] == "15_prototype_go_no_go_audit"


def test_phase14_build_week_report_artifacts_are_tracked_and_hashed() -> None:
    manifest = _build_week_report_manifest()

    for artifact in manifest["artifacts"].values():
        path = Path(artifact["uri"])
        assert path.is_file()
        assert str(artifact["sha256"]).startswith("sha256:")
        assert len(str(artifact["sha256"])) == 71
        assert int(artifact["byte_count"]) == path.stat().st_size
        assert artifact["sha256"] == (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )

    report = json.loads(
        Path(manifest["artifacts"]["json_report"]["uri"]).read_text(encoding="utf-8")
    )
    assert report["required_statement"].startswith(
        "This prototype demonstrates the architecture"
    )
    assert report["reference_bank"]["trust_distribution"] == {
        "R1": 0,
        "R2": 0,
        "R3": 0,
        "R4": 81,
        "R5": 0,
    }
    assert report["reference_bank"]["geographic_layer_distribution"] == {
        "A": 51,
        "B": 6,
        "C": 0,
        "D": 24,
        "E": 0,
    }
    assert {item["experiment_id"] for item in report["benchmark"]["experiments"]} == (
        {f"B{index}" for index in range(14)}
        | {"B14-regional", "B14-global", "B14-layered", "B15", "B16"}
    )
    assert report["evidence_semantics"]["classification_accuracy"] is None
    assert report["staged_flickr_inference"]["reference_routes_used"] == {
        "adult_field": 6527,
        "larval": 0,
        "pinned_specimen": 0,
        "none": 6969,
    }
    assert len(report["dashboard_ready_examples"]) == 4
    assert len(report["post_hackathon_human_review_plan"]) >= 8
