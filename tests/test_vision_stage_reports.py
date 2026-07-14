from __future__ import annotations

import json

import polars as pl

from biominer.reports.vision import build_vision_stage_metrics, write_vision_stage_reports


def test_vision_stage_metrics_count_detection_bioclip_and_evidence_frames() -> None:
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "detector_label": "butterfly_like",
                "detection_status": "detected",
                "crop_hash": "sha256:crop-1",
                "detection_route": "adult_butterfly_field",
                "routing_action": "score",
                "bioclip_route": "adult_field",
                "routing_priority": "standard",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-2",
                "detection_id": "det-2",
                "detector_label": "hard_negative",
                "detection_status": "detected",
                "crop_hash": None,
                "detection_route": "artwork_logo_tattoo_or_other_artifact",
                "routing_action": "exclude",
                "bioclip_route": None,
                "routing_priority": "none",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-3",
                "detection_id": "det-3",
                "detector_label": "no_detection",
                "detection_status": "no_detection",
                "crop_hash": None,
                "detection_route": "no_relevant_organism",
                "routing_action": "exclude",
                "bioclip_route": None,
                "routing_priority": "none",
            },
        ]
    )
    scores = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "selected_family": "Nymphalidae",
                "species_top1_scientific_name": "Danaus plexippus",
                "family_top3": ["Nymphalidae", "Papilionidae", "Pieridae"],
                "species_top20": ["Danaus plexippus", "Danaus gilippus"],
                "species_top5": ["Danaus plexippus"],
                "species_candidate_count": 42,
                "species_rerank_strategy": "first_pass_top20",
            }
        ]
    )
    joined = pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-1", "occurrence_bin": "gold"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "occurrence_bin": "bin"},
        ]
    )
    summary = pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-1", "photo_occurrence_bin": "gold"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "photo_occurrence_bin": "bin"},
        ]
    )

    metrics = build_vision_stage_metrics(
        detections=detections,
        scores=scores,
        joined=joined,
        photo_summary=summary,
        stage_metrics={
            "records_seen": 3,
            "images_loaded": 3,
            "image_failures": 0,
            "detector_batch_size_initial": 16,
            "detector_batch_size_final": 8,
            "detector_batch_retries": 1,
            "crop_batch_size": 24,
            "bioclip_batch_size_final": 12,
            "bioclip_batch_retries": 1,
            "elapsed_seconds": 2.0,
        },
    )

    assert metrics["detection"]["images_seen"] == 3
    assert metrics["detection"]["detections_by_label"] == {
        "butterfly_like": 1,
        "hard_negative": 1,
        "no_detection": 1,
    }
    assert metrics["detection"]["eligible_bioclip_detections"] == 1
    assert metrics["detection"]["detections_by_route"] == {
        "adult_butterfly_field": 1,
        "artwork_logo_tattoo_or_other_artifact": 1,
        "no_relevant_organism": 1,
    }
    assert metrics["detection"]["routing_action_counts"] == {
        "exclude": 2,
        "score": 1,
    }
    assert metrics["detection"]["bioclip_route_counts"] == {
        "": 2,
        "adult_field": 1,
    }
    assert metrics["detection"]["ambiguous_review_detections"] == 0
    assert metrics["detection"]["hard_negative_detections"] == 1
    assert metrics["detection"]["no_detection_count"] == 1
    assert metrics["bioclip"]["crops_scored"] == 1
    assert metrics["bioclip"]["family_scores_computed"] == 3
    assert metrics["bioclip"]["species_first_pass_candidates_seen"] == 2
    assert metrics["bioclip"]["bioclip_batch_retries"] == 1
    assert metrics["bioclip"]["bioclip_batch_size_final"] == 12
    assert metrics["bioclip"]["selected_family_counts"] == {"Nymphalidae": 1}
    assert metrics["bioclip"]["species_top1_counts"] == {"Danaus plexippus": 1}
    assert metrics["evidence"]["object_occurrence_bin_counts"] == {"bin": 1, "gold": 1}
    assert metrics["throughput"]["images_per_second"] == 1.5
    assert metrics["warnings"] == ["hard_negative_detections_present", "no_detection_records_present"]


def test_vision_stage_metrics_use_score_inputs_for_rolling_gate_warning() -> None:
    detections = pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "detector_label": "butterfly_like", "detection_status": "detected"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "detection_id": "det-2", "detector_label": "moth_like", "detection_status": "detected"},
            {"source": "flickr", "flickr_photo_id": "photo-3", "detection_id": "det-3", "detector_label": "no_detection", "detection_status": "no_detection"},
            {"source": "flickr", "flickr_photo_id": "photo-4", "detection_id": "det-4", "detector_label": "hard_negative", "detection_status": "detected"},
        ]
    )
    scores = pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "ablation_mode": "detector_crop"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "detection_id": "det-2", "ablation_mode": "detector_crop"},
            {"source": "flickr", "flickr_photo_id": "photo-3", "detection_id": "det-3", "ablation_mode": "whole_image"},
        ]
    )

    metrics = build_vision_stage_metrics(
        detections=detections,
        scores=scores,
        stage_metrics={
            "bioclip_gate_mode": "exclude_hard_negative",
            "bioclip_score_inputs": 3,
            "objects_scored": 3,
        },
    )

    assert metrics["detection"]["eligible_bioclip_detections"] == 1
    assert metrics["bioclip"]["bioclip_gate_mode"] == "exclude_hard_negative"
    assert metrics["bioclip"]["score_inputs_seen"] == 3
    assert metrics["bioclip"]["whole_images_scored"] == 1
    assert metrics["bioclip"]["detector_crops_scored"] == 2
    assert "crops_scored_exceeds_eligible_bioclip_detections" not in metrics["warnings"]
    assert metrics["warnings"] == ["hard_negative_detections_present", "no_detection_records_present"]


def test_vision_stage_metrics_empty_frames_are_stable() -> None:
    metrics = build_vision_stage_metrics(
        detections=pl.DataFrame(),
        scores=pl.DataFrame(),
        joined=pl.DataFrame(),
        photo_summary=pl.DataFrame(),
    )

    assert metrics["detection"]["images_seen"] == 0
    assert metrics["detection"]["images_loaded"] is None
    assert metrics["detection"]["detections_by_label"] == {}
    assert metrics["bioclip"]["crops_seen"] == 0
    assert metrics["bioclip"]["crops_scored"] == 0
    assert metrics["evidence"]["photo_occurrence_bin_counts"] == {}
    assert metrics["throughput"]["images_per_second"] is None
    assert metrics["warnings"] == []


def test_write_vision_stage_reports_writes_json_and_markdown(tmp_path) -> None:
    metrics = build_vision_stage_metrics(
        detections=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detector_label": "butterfly_like",
                    "detection_status": "detected",
                }
            ]
        ),
        scores=pl.DataFrame(
            [
                {
                    "selected_family": "Nymphalidae",
                    "species_top1_scientific_name": "Danaus plexippus",
                    "family_top3": ["Nymphalidae"],
                    "species_top20": ["Danaus plexippus"],
                    "species_top5": ["Danaus plexippus"],
                }
            ]
        ),
    )

    paths = write_vision_stage_reports(metrics, tmp_path)

    assert json.loads(paths["metrics"].read_text(encoding="utf-8"))["schema_version"] == "vision_stage_metrics_v2"
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "# Vision Stage Metrics" in summary
    assert "Nymphalidae: 1" in summary
    assert "Danaus plexippus: 1" in summary
