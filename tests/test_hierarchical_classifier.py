from __future__ import annotations

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION, TARGET_SCOPE_OBJECT_SCREENING
from biominer.bioclip.hierarchical_classifier import (
    BUTTERFLY_CASCADE_RESULT_SCHEMA,
    HIERARCHICAL_CANDIDATE_SELECTION_MODE,
    HIERARCHICAL_OBJECT_SCORE_SCHEMA_EXTENSIONS,
    HIERARCHICAL_SPECIES_RERANK_STRATEGY,
    TAXON_SCORE_DTYPE,
    ButterflyCascadeResult,
    TaxonScore,
    butterfly_cascade_result_to_dict,
    butterfly_cascade_results_frame,
)
from biominer.bioclip.object_runner import OBJECT_SCORE_OUTPUT_SCHEMA, _ensure_columns, empty_object_score_frame


def test_empty_object_score_frame_includes_hierarchical_schema_extensions() -> None:
    frame = empty_object_score_frame()

    assert frame.schema == OBJECT_SCORE_OUTPUT_SCHEMA
    for column, dtype in HIERARCHICAL_OBJECT_SCORE_SCHEMA_EXTENSIONS.items():
        assert frame.schema[column] == dtype


def test_butterfly_cascade_result_converts_to_stable_polars_frame() -> None:
    family = TaxonScore(
        accepted_taxon_key="gbif:9417",
        scientific_name="Papilionidae",
        rank="FAMILY",
        family_key="gbif:9417",
        family="Papilionidae",
        genus_key=None,
        genus=None,
        score=0.91,
        best_label="a photo of a butterfly in the family Papilionidae",
        label_count=3,
    )
    species = TaxonScore(
        accepted_taxon_key="gbif:100",
        scientific_name="Papilio demoleus",
        rank="SPECIES",
        family_key="gbif:9417",
        family="Papilionidae",
        genus_key="gbif:90",
        genus="Papilio",
        score=0.88,
        best_label="a photo of Papilio demoleus",
        label_count=3,
    )
    result = ButterflyCascadeResult(
        source="flickr",
        flickr_photo_id="photo-1",
        detection_id="det-1",
        crop_hash="sha256:crop-1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        candidate_set_id="gbif-butterflies-v1",
        taxonomy_table_version="gbif-butterfly-classification-v1",
        prompt_variant_version="butterfly-hierarchical-prompts-v1",
        family_top3=(family,),
        selected_family_key="gbif:9417",
        selected_family="Papilionidae",
        species_candidate_count=1,
        species_top20=(species,),
        species_top5=(species,),
        species_top1=species,
        family_top1_score=0.91,
        species_top1_score=0.88,
        species_top1_margin=None,
        classified_at="2026-07-09T00:00:00Z",
    )

    row = butterfly_cascade_result_to_dict(result)
    frame = butterfly_cascade_results_frame([result])

    assert row["family_top3"][0]["scientific_name"] == "Papilionidae"
    assert row["species_top1"]["accepted_taxon_key"] == "gbif:100"
    assert frame.schema == BUTTERFLY_CASCADE_RESULT_SCHEMA
    assert frame.to_dicts()[0]["species_top1"]["scientific_name"] == "Papilio demoleus"


def test_hierarchical_list_fields_have_stable_types() -> None:
    assert BUTTERFLY_CASCADE_RESULT_SCHEMA["family_top3"] == pl.List(TAXON_SCORE_DTYPE)
    assert BUTTERFLY_CASCADE_RESULT_SCHEMA["species_top20"] == pl.List(TAXON_SCORE_DTYPE)
    assert BUTTERFLY_CASCADE_RESULT_SCHEMA["species_top5"] == pl.List(TAXON_SCORE_DTYPE)
    assert OBJECT_SCORE_OUTPUT_SCHEMA["family_top3_accepted_taxon_keys"] == pl.List(pl.String)
    assert OBJECT_SCORE_OUTPUT_SCHEMA["family_top3_scores"] == pl.List(pl.Float64)
    assert OBJECT_SCORE_OUTPUT_SCHEMA["species_top20_scores"] == pl.List(pl.Float64)
    assert OBJECT_SCORE_OUTPUT_SCHEMA["species_top5_scores"] == pl.List(pl.Float64)


def test_target_scope_rows_validate_after_hierarchical_schema_extension() -> None:
    frame = _ensure_columns(
        pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "detection_id": "det-1",
                    "crop_hash": "sha256:crop-1",
                    "classification_mode": TARGET_SCOPE_OBJECT_SCREENING,
                    "candidate_selection_mode": "taxon_scope_or_species_context",
                    "species_rerank_strategy": "first_pass_top20",
                    "species_top5": ["Danaus plexippus"],
                    "species_top20": ["Danaus plexippus"],
                    "target_species_score": 0.82,
                    "target_species_rank": 1,
                }
            ]
        ),
        OBJECT_SCORE_OUTPUT_SCHEMA,
    )

    assert frame.schema["taxonomy_table_version"] == pl.String
    assert frame.schema["family_top3_scores"] == pl.List(pl.Float64)
    assert frame.to_dicts()[0]["classification_mode"] == TARGET_SCOPE_OBJECT_SCREENING
    assert frame.to_dicts()[0]["taxonomy_table_version"] is None
    assert frame.to_dicts()[0]["species_top20_scores"] is None


def test_hierarchical_output_constants_are_explicit() -> None:
    assert HIERARCHICAL_CANDIDATE_SELECTION_MODE == "gbif_family_first"
    assert HIERARCHICAL_SPECIES_RERANK_STRATEGY == "rerank_all_first_pass_top20"
