from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from biominer.bioclip.reference_prototypes import build_reference_prototypes
from biominer.bioclip.reference_scoring import (
    ReferenceCandidate,
    ReferenceEvidenceIndex,
)
from biominer.ml.training_features import (
    FEW_SHOT_TRAINING_FEATURES_FILE,
    LABEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PROHIBITED_SOURCE_FEATURE_FIELDS,
    DetectionQualityFeatures,
    FewShotTrainingExample,
    FrozenEmbeddingFeatures,
    GeographicEvidenceFeatures,
    ReferenceEvidenceFeatures,
    TextEvidenceFeatures,
    TrainingLabel,
    TrainingProvenance,
    build_few_shot_training_features,
    feature_schema_fingerprint,
    few_shot_training_features_schema,
    load_few_shot_training_features,
    validate_few_shot_training_features,
    write_few_shot_training_features,
)
from test_reference_prototypes import _embedding_artifact, _spec
from test_reference_scoring import _query


TARGET = "gbif:1938069"
COMPETITOR = "gbif:1938070"


def test_builds_raw_and_engineered_features_without_label_leakage() -> None:
    result = build_few_shot_training_features((_example("photo-1"),))

    assert result.schema == few_shot_training_features_schema()
    row = result.row(0, named=True)
    assert row["embedding"] == [1.0, 0.0, 0.0]
    assert row["embedding_dimension"] == 3
    assert row["embedding_norm"] == pytest.approx(1.0)
    assert row["best_regional_competitor_similarity"] == pytest.approx(0.65)
    assert row["best_same_genus_competitor_similarity"] == pytest.approx(0.7)
    assert row["best_false_positive_competitor_similarity"] == pytest.approx(0.5)
    assert row["best_family_negative_similarity"] == pytest.approx(0.2)
    assert row["best_domain_negative_similarity"] == pytest.approx(0.4)
    assert row["target_minus_best_competitor_margin"] == pytest.approx(0.1)
    assert row["target_minus_domain_negative_margin"] == pytest.approx(0.4)
    assert row["target_prototype_distance"] == pytest.approx(0.1)
    assert row["nearest_target_support_distance"] == pytest.approx(0.05)
    assert row["target_minus_competitor_text_margin"] == pytest.approx(0.4)
    assert row["image_short_side_px"] == 200
    assert row["image_long_side_px"] == 300
    assert row["image_megapixels"] == pytest.approx(0.06)
    assert row["low_resolution_indicator"] is True
    assert row["visual_input_quality_flags"] == ["blurred", "small_subject"]
    assert row["feature_schema_fingerprint"] == feature_schema_fingerprint(3)
    assert row["training_example_id"].startswith("few-shot-training-example:")
    assert row["training_data_fingerprint"].startswith("sha256:")

    schema_fields = set(result.columns)
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(LABEL_COLUMNS)
    assert PROHIBITED_SOURCE_FEATURE_FIELDS.isdisjoint(schema_fields)
    assert {
        "target_present",
        "accepted_class_taxon_key",
        "reviewed_label_id",
        "source_item_id",
    }.isdisjoint(MODEL_FEATURE_COLUMNS)


def test_missing_reference_text_and_geo_evidence_stays_null() -> None:
    example = _example("photo-no-geo", missing_geo=True, target_present=False)
    example = replace(
        example,
        reference=replace(
            example.reference,
            target_centroid_similarity=None,
            target_nearest_similarity=None,
            target_top_three_mean_similarity=None,
            target_top_five_mean_similarity=None,
            target_local_prototype_similarity=None,
            target_global_prototype_similarity=None,
            regional_competitor_similarities=(),
            same_genus_competitor_similarities=(),
            false_positive_competitor_similarities=(),
            family_negative_similarities=(),
            domain_negative_similarities=(),
        ),
        text=TextEvidenceFeatures(),
    )

    row = build_few_shot_training_features((example,)).row(0, named=True)

    for field in (
        "target_reference_centroid_similarity",
        "target_nearest_reference_similarity",
        "best_regional_competitor_similarity",
        "best_domain_negative_similarity",
        "target_minus_best_competitor_margin",
        "target_minus_domain_negative_margin",
        "target_prototype_distance",
        "nearest_target_support_distance",
        "target_text_ensemble_similarity",
        "best_competitor_text_similarity",
        "target_minus_competitor_text_margin",
        "target_regional_overlap_score",
        "nearest_target_occurrence_cell_distance_km",
    ):
        assert row[field] is None
    assert row["missing_geo"] is True
    assert row["geo_cluster_id"] == "no_geo"


def test_output_is_deterministic_and_rejects_component_split_leakage() -> None:
    first_example = _example("photo-a", group_id="group-a")
    second_example = _example("photo-b", group_id="group-b")

    first = build_few_shot_training_features((first_example, second_example))
    second = build_few_shot_training_features((second_example, first_example))

    assert_frame_equal(first, second)
    assert first["training_data_fingerprint"].n_unique() == 1

    leaking_provenance = replace(
        second_example.provenance,
        dataset_split="model_selection",
        source_owner_id=first_example.provenance.source_owner_id,
    )
    leaking = replace(second_example, provenance=leaking_provenance)
    with pytest.raises(ValueError, match="source_owner_id.*crosses dataset splits"):
        build_few_shot_training_features((first_example, leaking))


def test_task7_candidate_scores_adapt_to_competitor_categories(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-oa",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, 0, 0),
            ),
            _spec(
                "target-b",
                "target-ob",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.8, 0.2, 0),
            ),
            _spec(
                "competitor-a",
                "competitor-oa",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0, 1, 0),
            ),
            _spec(
                "competitor-b",
                "competitor-ob",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.2, 0.8, 0),
            ),
        ),
    )
    scores = ReferenceEvidenceIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    ).score(
        _query(embeddings),
        (
            ReferenceCandidate(TARGET, "Papilio demoleus"),
            ReferenceCandidate(COMPETITOR, "Papilio polytes"),
        ),
    )

    adapted = ReferenceEvidenceFeatures.from_candidate_evidence(
        scores,
        target_accepted_taxon_key=TARGET,
        regional_competitor_taxon_keys=(COMPETITOR,),
        same_genus_competitor_taxon_keys=(COMPETITOR,),
        false_positive_competitor_taxon_keys=(COMPETITOR,),
        family_negative_taxon_keys=(COMPETITOR,),
        domain_negative_similarities=(0.1,),
    )

    target_score, competitor_score = scores
    assert adapted.target_centroid_similarity == target_score.centroid_similarity
    assert adapted.target_nearest_similarity == target_score.nearest_support_similarity
    assert adapted.regional_competitor_similarities == (
        competitor_score.centroid_similarity,
    )
    assert adapted.same_genus_competitor_similarities == (
        competitor_score.centroid_similarity,
    )
    assert adapted.domain_negative_similarities == (0.1,)

    with pytest.raises(ValueError, match="cannot contain the target"):
        ReferenceEvidenceFeatures.from_candidate_evidence(
            scores,
            target_accepted_taxon_key=TARGET,
            regional_competitor_taxon_keys=(TARGET,),
        )
    with pytest.raises(ValueError, match="mixes scoring contracts"):
        ReferenceEvidenceFeatures.from_candidate_evidence(
            (target_score, replace(competitor_score, query_id="different-query")),
            target_accepted_taxon_key=TARGET,
        )


def test_invalid_embedding_geo_and_label_contracts_fail_closed() -> None:
    invalid_embedding = replace(
        _example("bad-embedding").embedding,
        embedding=(2.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="unit-normalized"):
        build_few_shot_training_features(
            (replace(_example("bad-embedding"), embedding=invalid_embedding),)
        )

    with pytest.raises(ValueError, match="missing_geo"):
        build_few_shot_training_features(
            (
                replace(
                    _example("bad-geo"),
                    geography=GeographicEvidenceFeatures(missing_geo=True),
                ),
            )
        )

    invalid_label = replace(
        _example("bad-label").label,
        accepted_class_taxon_key=COMPETITOR,
    )
    with pytest.raises(ValueError, match="target-present"):
        build_few_shot_training_features(
            (replace(_example("bad-label"), label=invalid_label),)
        )

    mismatched_reference = replace(
        _example("bad-reference").reference,
        model_fingerprint=_sha("different-model"),
    )
    with pytest.raises(ValueError, match="model fingerprint conflicts"):
        build_few_shot_training_features(
            (replace(_example("bad-reference"), reference=mismatched_reference),)
        )


def test_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    frame = build_few_shot_training_features(
        (
            _example("photo-a", group_id="group-a"),
            _example("photo-b", group_id="group-b"),
        )
    )

    path = write_few_shot_training_features(frame, tmp_path / "published")
    assert path.name == FEW_SHOT_TRAINING_FEATURES_FILE
    loaded = load_few_shot_training_features(
        path,
        expected_training_data_fingerprint=str(frame["training_data_fingerprint"][0]),
        expected_feature_schema_fingerprint=str(frame["feature_schema_fingerprint"][0]),
        expected_model_fingerprint=_sha("model"),
    )
    assert_frame_equal(loaded, frame)

    tampered = frame.with_columns(
        (pl.col("target_minus_best_competitor_margin") + 0.1).alias(
            "target_minus_best_competitor_margin"
        )
    )
    with pytest.raises(ValueError, match="target_minus_best_competitor_margin"):
        validate_few_shot_training_features(tampered)


def _example(
    source_item_id: str,
    *,
    group_id: str | None = None,
    missing_geo: bool = False,
    target_present: bool = True,
) -> FewShotTrainingExample:
    group = group_id or f"group:{source_item_id}"
    geo_cluster_id = "no_geo" if missing_geo else "cluster-a"
    return FewShotTrainingExample(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        provenance=TrainingProvenance(
            source_item_id=source_item_id,
            source_observation_id=f"observation:{source_item_id}",
            source_owner_id="owner:shared",
            duplicate_group_id=f"duplicate:{source_item_id}",
            burst_group_id=f"burst:{source_item_id}",
            provider_mirror_group_id=f"mirror:{source_item_id}",
            leakage_group_id=group,
            geo_cluster_id=geo_cluster_id,
            dataset_split="support_train",
            support_manifest_fingerprint=_sha("support"),
            reference_embedding_fingerprint=_sha("reference-embeddings"),
            reference_prototype_fingerprint=_sha("reference-prototypes"),
            candidate_set_fingerprint=_sha(f"candidate-set:{source_item_id}"),
            model_fingerprint=_sha("model"),
        ),
        label=TrainingLabel(
            reviewed_label_id=f"label:{source_item_id}",
            reviewed_label_fingerprint=_sha(f"label:{source_item_id}"),
            target_present=target_present,
            accepted_class_taxon_key=TARGET if target_present else COMPETITOR,
            visual_domain_label="live_field",
            label_certainty="high",
            species_training_suitable=True,
        ),
        embedding=FrozenEmbeddingFeatures(
            visual_input_id=f"visual-input:{source_item_id}",
            visual_input_kind="raw_full_image",
            embedding=(1.0, 0.0, 0.0),
            embedding_fingerprint=_sha(f"embedding:{source_item_id}"),
            model_fingerprint=_sha("model"),
        ),
        reference=ReferenceEvidenceFeatures(
            model_fingerprint=_sha("model"),
            reference_embedding_fingerprint=_sha("reference-embeddings"),
            reference_prototype_fingerprint=_sha("reference-prototypes"),
            support_manifest_fingerprint=_sha("support"),
            route="adult_field",
            visual_input_kind="raw_full_image",
            geo_cluster_id=geo_cluster_id,
            target_centroid_similarity=0.8,
            target_nearest_similarity=0.95,
            target_top_three_mean_similarity=0.85,
            target_top_five_mean_similarity=0.75,
            target_local_prototype_similarity=0.88,
            target_global_prototype_similarity=0.9,
            regional_competitor_similarities=(0.6, 0.65),
            same_genus_competitor_similarities=(0.7,),
            false_positive_competitor_similarities=(0.5,),
            family_negative_similarities=(0.2,),
            domain_negative_similarities=(0.4,),
        ),
        text=TextEvidenceFeatures(
            target_text_ensemble_similarity=0.7,
            best_competitor_text_similarity=0.3,
        ),
        geography=(
            GeographicEvidenceFeatures(
                target_candidate_source_count=1,
                competitor_candidate_source_count=2,
                total_candidate_source_count=2,
                missing_geo=True,
            )
            if missing_geo
            else GeographicEvidenceFeatures(
                target_regional_overlap_score=0.8,
                best_competitor_regional_overlap_score=0.5,
                nearest_target_occurrence_cell_distance_km=12.5,
                nearest_target_support_observation_distance_km=30.0,
                target_candidate_source_count=2,
                competitor_candidate_source_count=3,
                total_candidate_source_count=4,
            )
        ),
        detection=DetectionQualityFeatures(
            yoloe_route="adult_butterfly_field",
            detector_confidence=0.91,
            subject_area_ratio=0.35,
            mask_coverage=0.3,
            multiple_organism_indicator=False,
            image_width_px=300,
            image_height_px=200,
            visual_input_quality_flags=("small_subject", "blurred"),
        ),
    )


def _sha(value: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
