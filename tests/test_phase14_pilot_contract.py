from __future__ import annotations

import json
from pathlib import Path


MATRIX_PATH = Path("config/pilot/papilio_demoleus_phase14_experiment_matrix.json")


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


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
