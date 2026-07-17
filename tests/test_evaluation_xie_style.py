from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.xie_style import ARCHITECTURE, EVALUATION_PROFILE, evaluate_xie_style_hierarchical


def test_xie_style_profile_is_metrics_only_and_names_biominer_architecture() -> None:
    result = evaluate_xie_style_hierarchical(
        object_scores=pl.DataFrame([_prediction()]),
        reviewed_labels=pl.DataFrame([_label()]),
    )

    assert result["evaluation_profile"] == EVALUATION_PROFILE
    assert result["evaluation_profile"] == "xie_style_metrics_only"
    assert result["architecture"] == ARCHITECTURE
    assert result["architecture"] == "biominer_yoloe26_bioclip25_hierarchical"
    assert result["classification_mode"] == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert result["species_top1_accuracy"] == 1.0


def test_xie_style_adapter_does_not_mutate_classifier_outputs() -> None:
    object_scores = pl.DataFrame([_prediction(species_top1="Papilio machaon", species_top1_key="gbif:200")])
    before = object_scores.to_dicts()

    evaluate_xie_style_hierarchical(
        object_scores=object_scores,
        reviewed_labels=pl.DataFrame([_label()]),
    )

    assert object_scores.to_dicts() == before


def test_xie_style_macro_and_micro_averages_are_by_family() -> None:
    result = evaluate_xie_style_hierarchical(
        object_scores=pl.DataFrame(
            [
                _prediction(flickr_photo_id="p1", detection_id="d1"),
                _prediction(
                    flickr_photo_id="p2",
                    detection_id="d2",
                    species_top1="Papilio machaon",
                    species_top1_key="gbif:200",
                    species_top5=["Papilio machaon", "Papilio demoleus"],
                    species_top5_keys=["gbif:200", "gbif:100"],
                ),
                _prediction(
                    flickr_photo_id="n1",
                    detection_id="dn1",
                    selected_family="Nymphalidae",
                    selected_family_key="gbif:7017",
                    species_top1="Danaus gilippus",
                    species_top1_key="gbif:300",
                    species_top5=["Danaus gilippus"],
                    species_top5_keys=["gbif:300"],
                ),
            ]
        ),
        reviewed_labels=pl.DataFrame(
            [
                _label(flickr_photo_id="p1", detection_id="d1"),
                _label(flickr_photo_id="p2", detection_id="d2"),
                _label(
                    flickr_photo_id="n1",
                    detection_id="dn1",
                    accepted_taxon_key="gbif:400",
                    scientific_name="Danaus plexippus",
                    family_key="gbif:7017",
                    family="Nymphalidae",
                    genus_key="gbif:390",
                    genus="Danaus",
                ),
            ]
        ),
    )

    assert result["sample_count"] == 3
    assert result["matched_sample_count"] == 3
    assert result["species_top1_accuracy"] == pytest.approx(1 / 3)
    assert result["species_top5_recall"] == pytest.approx(2 / 3)
    assert result["micro_average"]["species_top1_accuracy"] == pytest.approx(1 / 3)
    assert result["micro_average"]["species_top5_recall"] == pytest.approx(2 / 3)
    assert result["macro_average_by_family"] == {
        "family_count": 2,
        "species_top1_accuracy": 0.25,
        "species_top5_recall": 0.5,
    }

    per_family = {row["family"]: row for row in result["per_family_species_accuracy"]}
    assert per_family["Papilionidae"]["species_top1_accuracy"] == 0.5
    assert per_family["Papilionidae"]["species_top5_recall"] == 1.0
    assert per_family["Nymphalidae"]["species_top1_accuracy"] == 0.0
    assert per_family["Nymphalidae"]["species_top5_recall"] == 0.0
    assert result["confusion_summary"]["species"]


def _prediction(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "p1",
    detection_id: str = "d1",
    selected_family: str = "Papilionidae",
    selected_family_key: str = "gbif:9417",
    species_top1: str = "Papilio demoleus",
    species_top1_key: str = "gbif:100",
    species_top5: list[str] | None = None,
    species_top5_keys: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "family_top3": [selected_family, "Pieridae", "Lycaenidae"],
        "family_top3_accepted_taxon_keys": [selected_family_key, "gbif:5481", "gbif:5473"],
        "selected_family": selected_family,
        "selected_family_key": selected_family_key,
        "species_top1_scientific_name": species_top1,
        "species_top1_accepted_taxon_key": species_top1_key,
        "species_top5": species_top5 or [species_top1],
        "species_top5_accepted_taxon_keys": species_top5_keys or [species_top1_key],
        "species_top20": species_top5 or [species_top1],
        "species_top20_accepted_taxon_keys": species_top5_keys or [species_top1_key],
    }


def _label(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "p1",
    detection_id: str = "d1",
    accepted_taxon_key: str = "gbif:100",
    scientific_name: str = "Papilio demoleus",
    family_key: str = "gbif:9417",
    family: str = "Papilionidae",
    genus_key: str = "gbif:90",
    genus: str = "Papilio",
) -> dict[str, object]:
    return {
        "schema_version": "reviewed-labels-v2",
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "crop_hash": f"sha256:{detection_id}",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
        "target_present": None,
        "label_certainty": "high",
        "life_stage": "unknown",
        "visual_domain": "ambiguous",
        "view": "unknown",
        "route": None,
        "geo_cluster_id": None,
        "source_query_tier": None,
        "source_query_term": None,
        "duplicate_group_id": None,
        "observer_owner_group_id": None,
        "dataset_split": "unassigned",
        "second_review_status": "unknown",
        "ambiguity_reason": "synthetic fixture",
        "unsuitable_for_species_identification": False,
    }
