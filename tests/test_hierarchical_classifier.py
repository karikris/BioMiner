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
    aggregate_taxon_prompt_scores,
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


def test_aggregate_taxon_prompt_scores_uses_mean_by_default() -> None:
    scores = aggregate_taxon_prompt_scores(
        label_scores={
            "a photo of Papilio demoleus": 0.4,
            "a close-up photo of the butterfly species Papilio demoleus": 0.7,
            "a photo of Danaus plexippus": 0.6,
        },
        label_rows=_species_label_rows(),
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
    )

    assert [score.scientific_name for score in scores[:2]] == ["Danaus plexippus", "Papilio demoleus"]
    assert scores[0].score == 0.6
    assert scores[1].score == 0.55


def test_aggregate_taxon_prompt_scores_supports_explicit_max() -> None:
    scores = aggregate_taxon_prompt_scores(
        label_scores={
            "a photo of Papilio demoleus": 0.4,
            "a close-up photo of the butterfly species Papilio demoleus": 0.7,
            "a photo of Danaus plexippus": 0.6,
        },
        label_rows=_species_label_rows(),
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
        aggregation="max",
    )

    assert [score.scientific_name for score in scores[:2]] == ["Papilio demoleus", "Danaus plexippus"]
    assert scores[0].score == 0.7
    assert scores[0].best_label == "a close-up photo of the butterfly species Papilio demoleus"
    assert scores[0].label_count == 2


def test_aggregate_taxon_prompt_scores_treats_missing_labels_as_zero() -> None:
    scores = aggregate_taxon_prompt_scores(
        label_scores={"a photo of Papilio demoleus": 0.4},
        label_rows=_species_label_rows(),
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
        aggregation="mean",
    )

    by_name = {score.scientific_name: score for score in scores}
    assert by_name["Papilio demoleus"].score == 0.2
    assert by_name["Danaus plexippus"].score == 0.0


def test_aggregate_taxon_prompt_scores_breaks_ties_deterministically() -> None:
    scores = aggregate_taxon_prompt_scores(
        label_scores={
            "a photo of Papilio demoleus": 0.5,
            "a photo of Danaus plexippus": 0.5,
        },
        label_rows=_species_label_rows(),
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
    )

    assert [score.scientific_name for score in scores[:2]] == ["Danaus plexippus", "Papilio demoleus"]


def test_aggregate_taxon_prompt_scores_ignores_disabled_rows_and_duplicate_labels() -> None:
    rows = pl.concat(
        [
            _species_label_rows(),
            pl.DataFrame(
                [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "scientific_name": "Papilio demoleus",
                        "family_key": "gbif:9417",
                        "family": "Papilionidae",
                        "genus_key": "gbif:90",
                        "genus": "Papilio",
                        "label": "a photo of Papilio demoleus",
                        "enabled": True,
                    },
                    {
                        "accepted_taxon_key": "gbif:100",
                        "scientific_name": "Papilio demoleus",
                        "family_key": "gbif:9417",
                        "family": "Papilionidae",
                        "genus_key": "gbif:90",
                        "genus": "Papilio",
                        "label": "disabled Papilio demoleus prompt",
                        "enabled": False,
                    },
                ]
            ),
        ],
        how="diagonal_relaxed",
    )

    scores = aggregate_taxon_prompt_scores(
        label_scores={
            "a photo of Papilio demoleus": 0.4,
            "a close-up photo of the butterfly species Papilio demoleus": 0.7,
            "disabled Papilio demoleus prompt": 0.99,
        },
        label_rows=rows,
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
        aggregation="max",
    )

    papilio = next(score for score in scores if score.scientific_name == "Papilio demoleus")
    assert papilio.score == 0.7
    assert papilio.label_count == 2


def test_aggregate_taxon_prompt_scores_supports_family_label_rows() -> None:
    rows = pl.DataFrame(
        [
            {
                "family_key": "gbif:9417",
                "family": "Papilionidae",
                "label": "a photo of a butterfly in the family Papilionidae",
                "enabled": True,
            }
        ]
    )

    scores = aggregate_taxon_prompt_scores(
        label_scores={"a photo of a butterfly in the family Papilionidae": 0.9},
        label_rows=rows,
        taxon_key_column="family_key",
        taxon_name_column="family",
    )

    assert scores == [
        TaxonScore(
            accepted_taxon_key="gbif:9417",
            scientific_name="Papilionidae",
            rank="FAMILY",
            family_key="gbif:9417",
            family="Papilionidae",
            genus_key=None,
            genus=None,
            score=0.9,
            best_label="a photo of a butterfly in the family Papilionidae",
            label_count=1,
        )
    ]


def test_aggregate_taxon_prompt_scores_rejects_unknown_aggregation() -> None:
    try:
        aggregate_taxon_prompt_scores(
            label_scores={},
            label_rows=_species_label_rows(),
            taxon_key_column="accepted_taxon_key",
            taxon_name_column="scientific_name",
            aggregation="median",
        )
    except ValueError as exc:
        assert "aggregation must be one of" in str(exc)
    else:
        raise AssertionError("unknown aggregation should fail")


def _species_label_rows() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "family_key": "gbif:9417",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "label": "a photo of Papilio demoleus",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "family_key": "gbif:9417",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "label": "a close-up photo of the butterfly species Papilio demoleus",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:200",
                "scientific_name": "Danaus plexippus",
                "family_key": "gbif:7017",
                "family": "Nymphalidae",
                "genus_key": "gbif:190",
                "genus": "Danaus",
                "label": "a photo of Danaus plexippus",
                "enabled": True,
            },
        ]
    )
