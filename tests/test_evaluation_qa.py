from __future__ import annotations

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.qa import (
    VISUAL_QA_FINDINGS_SCHEMA,
    build_visual_qa_findings,
    empty_visual_qa_findings_frame,
)


def test_empty_visual_qa_findings_frame_has_stable_schema() -> None:
    frame = empty_visual_qa_findings_frame()

    assert frame.schema == VISUAL_QA_FINDINGS_SCHEMA
    assert frame.is_empty()


def test_visual_qa_detects_each_fatal_condition() -> None:
    findings = build_visual_qa_findings(
        object_evidence=pl.DataFrame(
            [
                _row(flickr_photo_id="missing-family", selected_family="", selected_family_key=""),
                _row(
                    flickr_photo_id="outside-family",
                    species_top20_families=["Papilionidae", "Nymphalidae"],
                ),
                _row(
                    flickr_photo_id="bad-rerank-subset",
                    species_top5=["Papilio demoleus", "Danaus plexippus"],
                    species_top20=["Papilio demoleus", "Papilio machaon"],
                ),
                _row(
                    flickr_photo_id="noneligible-scored",
                    detector_label="moth_like",
                    species_top1_score=0.92,
                ),
            ]
        )
    )

    fatal_types = {
        row["finding_type"]
        for row in findings.filter(pl.col("severity") == "fatal").to_dicts()
    }
    assert fatal_types == {
        "hierarchical_missing_selected_family",
        "species_top20_outside_selected_family",
        "species_top5_not_subset_species_top20",
        "bioclip_score_for_noneligible_detection",
    }


def test_visual_qa_valid_evidence_has_no_fatal_findings() -> None:
    findings = build_visual_qa_findings(object_evidence=pl.DataFrame([_row()]))

    assert findings.schema == VISUAL_QA_FINDINGS_SCHEMA
    assert findings.filter(pl.col("severity") == "fatal").is_empty()


def test_visual_qa_allows_global_top20_across_families_and_reviewed_subtribe_skip() -> None:
    findings = build_visual_qa_findings(
        object_evidence=pl.DataFrame(
            [
                _cascade_row(
                    species_top20_families=["Papilionidae", "Nymphalidae"],
                )
            ]
        )
    )

    fatal_types = findings.filter(pl.col("severity") == "fatal")["finding_type"].to_list()
    assert "species_top20_outside_selected_family" not in fatal_types
    assert "hierarchical_missing_subtribe_or_skip" not in fatal_types
    assert not fatal_types


def test_visual_qa_rejects_path_cascade_without_subtribe_or_reviewed_skip() -> None:
    findings = build_visual_qa_findings(
        object_evidence=pl.DataFrame([_cascade_row(skipped_ranks=[])])
    )

    assert "hierarchical_missing_subtribe_or_skip" in findings["finding_type"].to_list()


def test_visual_qa_does_not_flag_bare_object_scores_as_noneligible_detections() -> None:
    row = _row()
    row.pop("detector_label")
    row.pop("detection_status")

    findings = build_visual_qa_findings(object_evidence=pl.DataFrame([row]))

    assert "bioclip_score_for_noneligible_detection" not in findings.select("finding_type").to_series().to_list()


def test_visual_qa_warning_counts_are_stable() -> None:
    findings = build_visual_qa_findings(
        object_evidence=pl.DataFrame(
            [
                _missing_score_row(),
                _row(flickr_photo_id="empty-family-top3", family_top3=[]),
                _row(flickr_photo_id="empty-species-top5", species_top5=[]),
                _row(flickr_photo_id="low-margin", species_top1_margin=0.01),
                _row(
                    flickr_photo_id="conflict-photo",
                    detection_id="det-1",
                    species_top1_scientific_name="Papilio demoleus",
                ),
                _row(
                    flickr_photo_id="conflict-photo",
                    detection_id="det-2",
                    species_top1_scientific_name="Papilio machaon",
                ),
                _row(
                    flickr_photo_id="metadata-conflict",
                    species_top1_scientific_name="Papilio demoleus",
                    flickr_text_species_candidate="Papilio machaon",
                ),
            ]
        ),
        photo_summary=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "summary-conflict",
                    "photo_multi_object_conflict": True,
                    "photo_review_reason": "multiple_species",
                }
            ]
        ),
    )

    warning_types = [row["finding_type"] for row in findings.filter(pl.col("severity") == "warning").to_dicts()]
    assert warning_types.count("butterfly_like_missing_bioclip_score") == 1
    assert warning_types.count("empty_family_top3") == 1
    assert warning_types.count("empty_species_top5") == 1
    assert warning_types.count("very_low_species_margin") == 1
    assert warning_types.count("metadata_vision_species_conflict") == 1
    assert warning_types.count("multi_object_species_conflict") == 2
    assert findings.filter(
        (pl.col("severity") == "info") & (pl.col("finding_type") == "high_priority_review_row")
    ).height >= 1


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "det-1",
        "crop_hash": "sha256:crop-1",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "detector_label": "butterfly_like",
        "detection_status": "detected",
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "family_top3": ["Papilionidae", "Pieridae", "Nymphalidae"],
        "species_top20": ["Papilio demoleus", "Papilio machaon"],
        "species_top20_families": ["Papilionidae", "Papilionidae"],
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "species_top1_score": 0.91,
        "species_top1_margin": 0.40,
        "detector_score": 0.88,
        "occurrence_bin": "in_review",
        "bin_reason": "hierarchical_open_classification_requires_review",
    }
    row.update(overrides)
    return row


def _cascade_row(**overrides: object) -> dict[str, object]:
    row = _row(
        classifier_schema_version="butterfly-cascade-output-v1.0.0",
        selected_family_node_id="fixture:family:papilionidae",
        selected_subfamily="Papilioninae",
        selected_tribe="Papilionini",
        selected_subtribe=None,
        selected_genus="Papilio",
        skipped_ranks=["SUBTRIBE"],
        family_top3=["Papilionidae"],
        family_top3_node_ids=["fixture:family:papilionidae"],
        family_top3_scores=[0.9],
        subfamily_top3=["Papilioninae"],
        subfamily_top3_node_ids=["fixture:subfamily:papilioninae"],
        subfamily_top3_scores=[0.8],
        tribe_top3=["Papilionini"],
        tribe_top3_node_ids=["fixture:tribe:papilionini"],
        tribe_top3_scores=[0.7],
        subtribe_top3=[],
        subtribe_top3_node_ids=[],
        subtribe_top3_scores=[],
        genus_top3=["Papilio"],
        genus_top3_node_ids=["fixture:genus:papilio"],
        genus_top3_scores=[0.6],
        species_top3=["Papilio demoleus"],
        species_top3_accepted_taxon_keys=["gbif:100"],
    )
    row.update(overrides)
    return row


def _missing_score_row() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "missing-score",
        "detection_id": "det-missing",
        "crop_hash": "sha256:missing",
        "detector_label": "butterfly_like",
        "detection_status": "detected",
        "detector_score": 0.94,
        "occurrence_bin": "in_review",
        "bin_reason": "missing_bioclip",
    }
