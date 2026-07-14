from __future__ import annotations

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.review_queue import HIERARCHICAL_REVIEW_QUEUE_SCHEMA, build_hierarchical_review_queue


def test_review_queue_low_margin_increases_priority() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _row(
                    flickr_photo_id="low-margin",
                    species_top1_margin=0.01,
                    family_top3_scores=[0.62, 0.61, 0.10],
                ),
                _row(flickr_photo_id="clean", species_top1_margin=0.42),
            ]
        )
    )

    rows = {row["flickr_photo_id"]: row for row in queue.to_dicts()}
    assert rows["low-margin"]["review_priority"] > rows["clean"]["review_priority"]
    assert "low_species_margin" in rows["low-margin"]["review_reason"]
    assert "low_family_margin" in rows["low-margin"]["review_reason"]


def test_review_queue_missing_score_for_butterfly_like_detection_enters_queue() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "missing",
                    "detection_id": "det-missing",
                    "crop_hash": "sha256:missing",
                    "detector_label": "butterfly_like",
                    "detector_score": 0.94,
                    "detection_status": "cropped",
                    "image_url": "https://example.test/missing.jpg",
                    "photo_page_url": "https://www.flickr.com/photos/example/missing",
                    "occurrence_bin": "in_review",
                    "bin_reason": "missing_bioclip",
                }
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert queue.schema == HIERARCHICAL_REVIEW_QUEUE_SCHEMA
    assert row["review_priority"] == 100
    assert row["review_reason"] == "missing_bioclip_score"
    assert row["classification_mode"] == ""


def test_review_queue_preserves_low_priority_ambiguous_detection_route() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "ambiguous",
                    "detection_id": "det-ambiguous",
                    "detector_label": "insect_like",
                    "detector_score": 0.41,
                    "detection_status": "detected",
                    "detection_route": "ambiguous_visual_domain",
                    "routing_action": "review",
                    "routing_priority": "low",
                    "image_url": "https://example.test/ambiguous.jpg",
                    "photo_page_url": "https://www.flickr.com/photos/example/ambiguous",
                    "occurrence_bin": "in_review",
                    "bin_reason": "ambiguous_visual_domain_review",
                }
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert row["review_priority"] == 10
    assert row["review_reason"] == (
        "detection_route_review:ambiguous_visual_domain"
    )


def test_review_queue_never_treats_missing_larval_score_as_adult() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "larva",
                    "detection_id": "det-larva",
                    "detector_label": "caterpillar",
                    "detector_score": 0.91,
                    "detection_status": "detected",
                    "detection_route": "caterpillar_field",
                    "routing_action": "score",
                    "bioclip_route": "larval",
                    "image_url": "https://example.test/larva.jpg",
                    "photo_page_url": "https://www.flickr.com/photos/example/larva",
                    "occurrence_bin": "in_review",
                    "bin_reason": "larval_route_without_bioclip_score",
                }
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert row["review_priority"] == 100
    assert row["review_reason"] == "missing_route_bioclip_score:larval"


def test_review_queue_preserves_pupa_as_separate_visual_domain() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "pupa",
                    "detection_id": "det-pupa",
                    "detector_label": "pupa_or_chrysalis",
                    "detector_score": 0.93,
                    "detection_status": "detected",
                    "detection_route": "pupa_or_chrysalis",
                    "routing_action": "exclude",
                    "image_url": "https://example.test/pupa.jpg",
                    "photo_page_url": "https://www.flickr.com/photos/example/pupa",
                    "occurrence_bin": "in_review",
                    "bin_reason": "pupa_or_chrysalis_requires_separate_classifier",
                }
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert row["review_priority"] == 10
    assert row["review_reason"] == "separate_visual_domain:pupa_or_chrysalis"


def test_review_queue_metadata_conflict_enters_queue() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _row(
                    flickr_photo_id="conflict",
                    species_top1_scientific_name="Papilio demoleus",
                    flickr_text_species_candidate="Papilio machaon",
                )
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert row["review_priority"] == 90
    assert row["review_reason"] == "metadata_species_conflict"
    assert row["species_top1_scientific_name"] == "Papilio demoleus"


def test_review_queue_confident_clean_prediction_is_lower_priority() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _row(flickr_photo_id="clean", species_top1_score=0.93, species_top1_margin=0.55, detector_score=0.88),
                _row(flickr_photo_id="weak", species_top1_score=0.22, detector_score=0.91),
            ]
        )
    )

    rows = {row["flickr_photo_id"]: row for row in queue.to_dicts()}
    assert rows["weak"]["review_priority"] == 75
    assert rows["clean"]["review_priority"] == 10
    assert rows["clean"]["review_reason"] == "clean_confident_prediction"


def test_review_queue_sort_order_is_stable() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _row(flickr_photo_id="b", detection_id="det-2", species_top1_margin=0.01),
                _row(flickr_photo_id="a", detection_id="det-2", species_top1_margin=0.01),
                _row(flickr_photo_id="a", detection_id="det-1", species_top1_margin=0.01),
                _row(flickr_photo_id="c", species_top1_margin=0.40),
            ]
        ),
        max_rows=3,
    )

    assert [(row["flickr_photo_id"], row["detection_id"]) for row in queue.to_dicts()] == [
        ("a", "det-1"),
        ("a", "det-2"),
        ("b", "det-2"),
    ]


def test_review_queue_uses_photo_summary_fallback_fields() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame([_row(image_url="", photo_page_url="", occurrence_bin="", bin_reason="")]),
        photo_summary=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "image_url": "https://example.test/photo-summary.jpg",
                    "photo_page_url": "https://www.flickr.com/photos/example/photo-1",
                    "photo_occurrence_bin": "in_review",
                    "photo_bin_reason": "ambiguous_species_margin",
                }
            ]
        ),
    )

    row = queue.to_dicts()[0]
    assert row["image_url"] == "https://example.test/photo-summary.jpg"
    assert row["photo_page_url"] == "https://www.flickr.com/photos/example/photo-1"
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "ambiguous_species_margin"


def test_review_queue_accepts_reviewed_subtribe_skip_and_routes_missing_skip() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _cascade_row(flickr_photo_id="valid-skip"),
                _cascade_row(flickr_photo_id="missing-skip", skipped_ranks=[]),
            ]
        )
    )

    rows = {row["flickr_photo_id"]: row for row in queue.to_dicts()}
    assert queue.schema == HIERARCHICAL_REVIEW_QUEUE_SCHEMA
    assert "missing_required_cascade_rank" not in rows["valid-skip"]["review_reason"]
    assert rows["valid-skip"]["selected_genus"] == "Papilio"
    assert rows["valid-skip"]["skipped_ranks"] == ["SUBTRIBE"]
    assert "missing_required_cascade_rank" in rows["missing-skip"]["review_reason"]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "det-1",
        "crop_hash": "sha256:crop-1",
        "photo_page_url": "https://www.flickr.com/photos/example/photo-1",
        "image_url": "https://example.test/photo-1.jpg",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "family_top3": ["Papilionidae", "Pieridae", "Nymphalidae"],
        "family_top3_scores": [0.90, 0.05, 0.04],
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top5_scores": [0.91, 0.40],
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "species_top1_score": 0.91,
        "species_top1_margin": 0.51,
        "detector_label": "butterfly_like",
        "detector_score": 0.82,
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
        selected_subfamily_node_id="fixture:subfamily:papilioninae",
        selected_tribe="Papilionini",
        selected_tribe_node_id="fixture:tribe:papilionini",
        selected_subtribe=None,
        selected_subtribe_node_id=None,
        selected_genus="Papilio",
        selected_genus_node_id="fixture:genus:papilio",
        subfamily_top3=["Papilioninae"],
        tribe_top3=["Papilionini"],
        subtribe_top3=[],
        genus_top3=["Papilio"],
        skipped_ranks=["SUBTRIBE"],
    )
    row.update(overrides)
    return row
