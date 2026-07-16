from __future__ import annotations

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


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _reference_source_manifest() -> dict[str, object]:
    return json.loads(REFERENCE_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_acquisition_manifest() -> dict[str, object]:
    return json.loads(
        PROTOTYPE_ACQUISITION_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def _prototype_selection_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_download_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_DOWNLOAD_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_qa_manifest() -> dict[str, object]:
    return json.loads(PROTOTYPE_QA_MANIFEST_PATH.read_text(encoding="utf-8"))


def _prototype_duplicate_resolution_manifest() -> dict[str, object]:
    return json.loads(
        PROTOTYPE_DUPLICATE_RESOLUTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def test_phase14_matrix_freezes_local_vision_limit_and_external_execution() -> None:
    matrix = _matrix()
    execution = matrix["execution_contract"]

    assert execution["branch"] == "main"
    assert execution["local_build_verification_max_images"] == 5
    assert execution["large_bioclip_execution"] == "different_computer_required"
    assert execution["large_yoloe_execution"] == "different_computer_required"
    assert {
        experiment["execution_computer"] for experiment in matrix["experiments"]
    } == {"external_gpu_computer"}


def test_phase14_matrix_contains_exact_b0_through_b16_ablation_contract() -> None:
    matrix = _matrix()
    experiments = {
        experiment["experiment_id"]: experiment
        for experiment in matrix["experiments"]
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
    assert experiments["B14"]["reference_scope"] == (
        "geo_cluster_then_global_fallback"
    )
    assert experiments["B15"]["evidence"] == ["frozen_image", "taxonomy_text"]
    assert experiments["B16"]["evidence"] == ["frozen_image"]


def test_phase14_matrix_blocks_benchmark_and_phase15_until_human_freeze() -> None:
    matrix = _matrix()
    gates = {
        gate["gate_id"]: gate for gate in matrix["prerequisite_gates"]
    }
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
    assert matrix["phase15_default_gate"]["current_default_must_remain_unchanged"] is True


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


def test_phase14_prototype_acquisition_manifest_records_bounded_real_candidates() -> None:
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
        manifest["verification_semantics"][
            "provider_supported_is_human_verified"
        ]
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


def test_phase14_prototype_selection_manifest_enforces_independent_observations() -> None:
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
