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
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-2",
                "detection_id": "det-2",
                "detector_label": "hard_negative",
                "detection_status": "detected",
                "crop_hash": None,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-3",
                "detection_id": "det-3",
                "detector_label": "no_detection",
                "detection_status": "no_detection",
                "crop_hash": None,
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

    assert json.loads(paths["metrics"].read_text(encoding="utf-8"))["schema_version"] == "vision_stage_metrics_v1"
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "# Vision Stage Metrics" in summary
    assert "Nymphalidae: 1" in summary
    assert "Danaus plexippus: 1" in summary
