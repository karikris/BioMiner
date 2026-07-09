from __future__ import annotations

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION, TARGET_SCOPE_OBJECT_SCREENING
from biominer.evaluation.metrics import family_confusion_matrix, species_confusion_matrix


def test_family_confusion_matrix_counts_pairs_stably() -> None:
    frame = family_confusion_matrix(
        object_scores=pl.DataFrame(
            [
                _prediction(flickr_photo_id="1", detection_id="d1", selected_family="Papilionidae"),
                _prediction(
                    flickr_photo_id="2",
                    detection_id="d2",
                    selected_family="Nymphalidae",
                    selected_family_key="gbif:7017",
                ),
                _prediction(flickr_photo_id="3", detection_id="d3", selected_family="Papilionidae"),
            ]
        ),
        reviewed_labels=pl.DataFrame(
            [
                _label(flickr_photo_id="1", detection_id="d1", family="Papilionidae", family_key="gbif:9417"),
                _label(flickr_photo_id="2", detection_id="d2", family="Papilionidae", family_key="gbif:9417"),
                _label(flickr_photo_id="3", detection_id="d3", family="Papilionidae", family_key="gbif:9417"),
            ]
        ),
    )

    assert frame.to_dicts() == [
        {
            "true_key": "gbif:9417",
            "true_name": "Papilionidae",
            "predicted_key": "gbif:9417",
            "predicted_name": "Papilionidae",
            "count": 2,
            "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        },
        {
            "true_key": "gbif:9417",
            "true_name": "Papilionidae",
            "predicted_key": "gbif:7017",
            "predicted_name": "Nymphalidae",
            "count": 1,
            "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        },
    ]


def test_species_confusion_matrix_marks_missing_predictions() -> None:
    frame = species_confusion_matrix(
        object_scores=pl.DataFrame([]),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert frame.to_dicts() == [
        {
            "true_key": "gbif:100",
            "true_name": "Papilio demoleus",
            "predicted_key": "missing_prediction",
            "predicted_name": "missing_prediction",
            "count": 1,
            "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        }
    ]


def test_confusion_matrices_handle_negative_labels_as_not_butterfly() -> None:
    frame = species_confusion_matrix(
        object_scores=pl.DataFrame([_prediction(species_top1="Papilio demoleus", species_top1_key="gbif:100")]),
        reviewed_labels=pl.DataFrame([_label(is_butterfly=False, label_level="negative")]),
    )

    assert frame.to_dicts()[0] == {
        "true_key": "not_butterfly",
        "true_name": "not_butterfly",
        "predicted_key": "gbif:100",
        "predicted_name": "Papilio demoleus",
        "count": 1,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    }


def test_species_confusion_matrix_limit_applies_after_sorting() -> None:
    frame = species_confusion_matrix(
        object_scores=pl.DataFrame(
            [
                _prediction(flickr_photo_id="1", detection_id="d1"),
                _prediction(flickr_photo_id="2", detection_id="d2"),
                _prediction(
                    flickr_photo_id="3",
                    detection_id="d3",
                    species_top1="Papilio machaon",
                    species_top1_key="gbif:200",
                ),
            ]
        ),
        reviewed_labels=pl.DataFrame(
            [
                _label(flickr_photo_id="1", detection_id="d1"),
                _label(flickr_photo_id="2", detection_id="d2"),
                _label(flickr_photo_id="3", detection_id="d3"),
            ]
        ),
        limit=1,
    )

    assert frame.height == 1
    assert frame.to_dicts()[0]["count"] == 2
    assert frame.to_dicts()[0]["predicted_name"] == "Papilio demoleus"


def test_confusion_matrix_treats_target_scope_only_row_as_missing_prediction() -> None:
    frame = family_confusion_matrix(
        object_scores=pl.DataFrame([_prediction(classification_mode=TARGET_SCOPE_OBJECT_SCREENING)]),
        reviewed_labels=pl.DataFrame([_label()]),
    )

    assert frame.to_dicts()[0]["predicted_key"] == "missing_prediction"


def _prediction(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "1",
    detection_id: str = "d1",
    classification_mode: str = HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    selected_family: str = "Papilionidae",
    selected_family_key: str = "gbif:9417",
    species_top1: str = "Papilio demoleus",
    species_top1_key: str = "gbif:100",
) -> dict[str, object]:
    return {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "classification_mode": classification_mode,
        "selected_family": selected_family,
        "selected_family_key": selected_family_key,
        "species_top1_scientific_name": species_top1,
        "species_top1_accepted_taxon_key": species_top1_key,
    }


def _label(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "1",
    detection_id: str = "d1",
    label_level: str = "species",
    is_butterfly: bool = True,
    accepted_taxon_key: str = "gbif:100",
    scientific_name: str = "Papilio demoleus",
    family_key: str = "gbif:9417",
    family: str = "Papilionidae",
) -> dict[str, object]:
    return {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "crop_hash": f"sha256:{detection_id}",
        "label_level": label_level,
        "is_butterfly": is_butterfly,
        "accepted_taxon_key": accepted_taxon_key if is_butterfly else "",
        "scientific_name": scientific_name if is_butterfly else "",
        "family_key": family_key if is_butterfly else "",
        "family": family if is_butterfly else "",
        "genus_key": "gbif:90" if is_butterfly else "",
        "genus": "Papilio" if is_butterfly else "",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
    }
