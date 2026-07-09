from __future__ import annotations

import polars as pl
import pytest

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
    classify_butterfly_crop_hierarchical,
    classify_butterfly_crops_hierarchical_batch,
    hierarchical_result_to_object_score_row,
)
from biominer.bioclip.object_runner import OBJECT_SCORE_OUTPUT_SCHEMA, _ensure_columns, empty_object_score_frame
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.registry.classification_table import (
    CLASSIFICATION_TABLE_VERSION,
    CLASSIFICATION_TAXA_SCHEMA,
    PROMPT_VARIANT_VERSION,
    build_family_label_frame,
    build_species_label_frame,
    ensure_classification_taxa_schema,
)


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


def test_classify_butterfly_crop_hierarchical_selects_top_family_and_reranks_top20() -> None:
    store = _taxonomy_store()
    species_first_pass = {f"Papilio species{index:02d}": 1.0 - (index / 100.0) for index in range(1, 26)}
    species_rerank = {f"Papilio species{index:02d}": 0.60 - (index / 100.0) for index in range(1, 21)}
    species_rerank["Papilio species18"] = 0.99
    scorer = _SequencedScorer(
        [
            _family_label_scores(store, {"Papilionidae": 0.90, "Nymphalidae": 0.80, "Pieridae": 0.30}),
            _species_label_scores(store, species_first_pass),
            _species_label_scores(store, species_rerank),
        ]
    )

    result = classify_butterfly_crop_hierarchical(
        item=_cascade_item(),
        scorer=scorer,
        taxonomy_store=store,
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=5,
    )

    assert result.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert result.selected_family_key == "gbif:9417"
    assert result.selected_family == "Papilionidae"
    assert [score.scientific_name for score in result.family_top3] == ["Papilionidae", "Nymphalidae", "Pieridae"]
    assert result.species_candidate_count == 25
    assert len(result.species_top20) == 20
    assert {score.family_key for score in result.species_top20} == {"gbif:9417"}
    assert "Papilio species18" in [score.scientific_name for score in result.species_top20]
    assert result.species_top1 is not None
    assert result.species_top1.scientific_name == "Papilio species18"
    assert result.species_top5[0].scientific_name == "Papilio species18"
    assert result.species_top1_margin is not None
    assert result.taxonomy_table_version == CLASSIFICATION_TABLE_VERSION
    assert result.prompt_variant_version == PROMPT_VARIANT_VERSION

    first_pass_labels = scorer.calls[1]
    rerank_labels = scorer.calls[2]
    assert any("Papilio species18" in label for label in first_pass_labels)
    assert any("Papilio species18" in label for label in rerank_labels)
    assert not any("Danaus plexippus" in label for label in first_pass_labels)
    assert not any("Papilio species21" in label for label in rerank_labels)


def test_classify_butterfly_crop_hierarchical_does_not_inject_target_species() -> None:
    store = _taxonomy_store(species_per_papilionidae=3)
    scorer = _SequencedScorer(
        [
            _family_label_scores(store, {"Papilionidae": 0.90, "Nymphalidae": 0.40, "Pieridae": 0.30}),
            _species_label_scores(store, {"Papilio species01": 0.80, "Papilio species02": 0.70, "Papilio species03": 0.60}),
            _species_label_scores(store, {"Papilio species01": 0.80, "Papilio species02": 0.70, "Papilio species03": 0.60}),
        ]
    )
    item = {**_cascade_item(), "target_scientific_name": "Injected target species"}

    result = classify_butterfly_crop_hierarchical(
        item=item,
        scorer=scorer,
        taxonomy_store=store,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )

    assert result.species_top1 is not None
    assert result.species_top1.scientific_name == "Papilio species01"
    assert all("Injected target species" not in label for labels in scorer.calls for label in labels)


def test_classify_butterfly_crop_hierarchical_fails_clearly_for_missing_family_labels() -> None:
    store = _taxonomy_store()
    broken = ButterflyTaxonomyStore(
        classification_taxa=store.classification_taxa,
        family_labels=pl.DataFrame(schema=store.family_labels.schema),
        species_labels=store.species_labels,
        manifest=store.manifest,
    )

    with pytest.raises(ValueError, match="no_family_labels"):
        classify_butterfly_crop_hierarchical(
            item=_cascade_item(),
            scorer=_SequencedScorer([{}]),
            taxonomy_store=broken,
        )


def test_classify_butterfly_crops_hierarchical_batch_matches_single_item_results() -> None:
    store = _taxonomy_store(species_per_papilionidae=3)
    scores = _combined_label_scores(
        store,
        family_scores={"Papilionidae": 0.90, "Nymphalidae": 0.40, "Pieridae": 0.30},
        species_scores={"Papilio species01": 0.80, "Papilio species02": 0.70, "Papilio species03": 0.60},
    )
    item = _cascade_item()
    single = classify_butterfly_crop_hierarchical(
        item=item,
        scorer=_StaticBatchScorer({"sha256:crop-1": scores}),
        taxonomy_store=store,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )
    batch = classify_butterfly_crops_hierarchical_batch(
        items=[item],
        scorer=_StaticBatchScorer({"sha256:crop-1": scores}),
        taxonomy_store=store,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )

    assert len(batch) == 1
    assert batch[0].selected_family_key == single.selected_family_key
    assert [score.accepted_taxon_key for score in batch[0].species_top20] == [
        score.accepted_taxon_key for score in single.species_top20
    ]
    assert [score.accepted_taxon_key for score in batch[0].species_top5] == [
        score.accepted_taxon_key for score in single.species_top5
    ]


def test_classify_butterfly_crops_hierarchical_batch_preserves_order_and_family_pools() -> None:
    store = _taxonomy_store(species_per_papilionidae=3)
    papilio_scores = _combined_label_scores(
        store,
        family_scores={"Papilionidae": 0.95, "Nymphalidae": 0.20, "Pieridae": 0.10},
        species_scores={"Papilio species01": 0.50, "Papilio species02": 0.80, "Papilio species03": 0.70},
    )
    danaus_scores = _combined_label_scores(
        store,
        family_scores={"Papilionidae": 0.10, "Nymphalidae": 0.96, "Pieridae": 0.20},
        species_scores={"Danaus plexippus": 0.88},
    )
    items = [
        {**_cascade_item(), "flickr_photo_id": "photo-nymph", "detection_id": "det-nymph", "crop_hash": "sha256:crop-nymph"},
        {**_cascade_item(), "flickr_photo_id": "photo-papilio", "detection_id": "det-papilio", "crop_hash": "sha256:crop-papilio"},
    ]
    scorer = _StaticBatchScorer(
        {
            "sha256:crop-nymph": danaus_scores,
            "sha256:crop-papilio": papilio_scores,
        }
    )

    results = classify_butterfly_crops_hierarchical_batch(
        items=items,
        scorer=scorer,
        taxonomy_store=store,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )

    assert [result.detection_id for result in results] == ["det-nymph", "det-papilio"]
    assert results[0].selected_family == "Nymphalidae"
    assert results[0].species_top1 is not None
    assert results[0].species_top1.scientific_name == "Danaus plexippus"
    assert results[1].selected_family == "Papilionidae"
    assert results[1].species_top1 is not None
    assert results[1].species_top1.scientific_name == "Papilio species02"

    species_batch_labels = [
        tuple(label for labels in label_sets.values() for label in labels)
        for _detections, label_sets in scorer.batch_calls
        if any(name.startswith("species:") for name in label_sets)
    ]
    assert len(species_batch_labels) == 2
    assert any(any("Danaus plexippus" in label for label in labels) and not any("Papilio species" in label for label in labels) for labels in species_batch_labels)
    assert any(any("Papilio species" in label for label in labels) and not any("Danaus plexippus" in label for label in labels) for labels in species_batch_labels)

    rerank_label_sets = [
        tuple(label for labels in label_sets.values() for label in labels)
        for _detections, label_sets in scorer.batch_calls
        if set(label_sets) == {"rerank"}
    ]
    assert len(rerank_label_sets) == 2
    assert any(any("Danaus plexippus" in label for label in labels) and not any("Papilio species" in label for label in labels) for labels in rerank_label_sets)
    assert any(any("Papilio species" in label for label in labels) and not any("Danaus plexippus" in label for label in labels) for labels in rerank_label_sets)


def test_hierarchical_result_to_object_score_row_is_conservative_open_classification() -> None:
    store = _taxonomy_store(species_per_papilionidae=3)
    scores = _combined_label_scores(
        store,
        family_scores={"Papilionidae": 0.90, "Nymphalidae": 0.40, "Pieridae": 0.30},
        species_scores={"Papilio species01": 0.60, "Papilio species02": 0.85, "Papilio species03": 0.70},
    )
    scorer = _StaticBatchScorer({"sha256:crop-1": scores})
    item = {**_cascade_item(), "ablation_mode": "detector_crop", "detector_score": 0.77}
    result = classify_butterfly_crop_hierarchical(
        item=item,
        scorer=scorer,
        taxonomy_store=store,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )

    row = hierarchical_result_to_object_score_row(
        item=item,
        result=result,
        scorer=scorer,
        family_top_k=3,
        species_first_pass_top_k=3,
        species_rerank_top_k=2,
    )
    frame = _ensure_columns(pl.DataFrame([row]), OBJECT_SCORE_OUTPUT_SCHEMA)

    assert row["classification_mode"] == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert row["candidate_selection_mode"] == HIERARCHICAL_CANDIDATE_SELECTION_MODE
    assert row["species_rerank_strategy"] == HIERARCHICAL_SPECIES_RERANK_STRATEGY
    assert row["selected_family_key"] == "gbif:9417"
    assert row["selected_family"] == "Papilionidae"
    assert row["species_top1_scientific_name"] == "Papilio species02"
    assert row["species_top1_accepted_taxon_key"] == row["accepted_taxon_key"]
    assert row["species_top5"][0] == "Papilio species02"
    assert row["species_top5_accepted_taxon_keys"][0] == row["accepted_taxon_key"]
    assert row["species_top5_scores"][0] == row["species_top1_score"]
    assert row["target_accepted_taxon_key"] is None
    assert row["target_species_score"] is None
    assert row["target_species_rank"] is None
    assert row["is_target_positive"] is False
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "hierarchical_open_classification_requires_review"
    assert frame.schema["target_species_rank"] == pl.Int64
    assert frame.schema["target_species_score"] == pl.Float64
    assert frame.schema["species_top20_scores"] == pl.List(pl.Float64)


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


class _SequencedScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, responses: list[dict[str, float]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, ...]] = []

    def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
        self.calls.append(labels)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index] if self._responses else {}
        return {label: float(response.get(label, 0.0)) for label in labels}


class _StaticBatchScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores_by_crop: dict[str, dict[str, float]]) -> None:
        self._scores_by_crop = scores_by_crop
        self.batch_calls: list[tuple[tuple[str, ...], dict[str, tuple[str, ...]]]] = []

    def score(self, item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
        scores = self._scores_by_crop.get(str(item.get("crop_hash") or ""), {})
        return {label: float(scores.get(label, 0.0)) for label in labels}

    def score_label_sets_batch(
        self,
        items: list[dict[str, object]],
        label_sets: dict[str, tuple[str, ...]],
    ) -> dict[str, list[dict[str, float]]]:
        self.batch_calls.append(
            (
                tuple(str(item.get("detection_id") or "") for item in items),
                {name: tuple(labels) for name, labels in label_sets.items()},
            )
        )
        return {
            name: [self.score(item, tuple(labels)) for item in items]
            for name, labels in label_sets.items()
        }


def _cascade_item() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "det-1",
        "crop_hash": "sha256:crop-1",
    }


def _taxonomy_store(species_per_papilionidae: int = 25) -> ButterflyTaxonomyStore:
    rows: list[dict[str, object]] = []
    for index in range(1, species_per_papilionidae + 1):
        rows.append(
            _classification_taxon(
                accepted_taxon_key=f"gbif:94{index:02d}",
                scientific_name=f"Papilio species{index:02d}",
                family_key="gbif:9417",
                family="Papilionidae",
                genus_key="gbif:90",
                genus="Papilio",
            )
        )
    rows.extend(
        [
            _classification_taxon(
                accepted_taxon_key="gbif:7017001",
                scientific_name="Danaus plexippus",
                family_key="gbif:7017",
                family="Nymphalidae",
                genus_key="gbif:190",
                genus="Danaus",
            ),
            _classification_taxon(
                accepted_taxon_key="gbif:5481001",
                scientific_name="Pieris rapae",
                family_key="gbif:5481",
                family="Pieridae",
                genus_key="gbif:91",
                genus="Pieris",
            ),
        ]
    )
    taxa = ensure_classification_taxa_schema(pl.DataFrame(rows, schema=CLASSIFICATION_TAXA_SCHEMA))
    return ButterflyTaxonomyStore(
        classification_taxa=taxa,
        family_labels=build_family_label_frame(taxa),
        species_labels=build_species_label_frame(taxa),
        manifest={
            "registry_version": "registry-v1",
            "classification_table_version": CLASSIFICATION_TABLE_VERSION,
            "prompt_variant_version": PROMPT_VARIANT_VERSION,
        },
    )


def _classification_taxon(
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    family_key: str,
    family: str,
    genus_key: str,
    genus: str,
) -> dict[str, object]:
    species = scientific_name
    return {
        "registry_version": "registry-v1",
        "classification_table_version": CLASSIFICATION_TABLE_VERSION,
        "source": "GBIF",
        "source_version": "",
        "retrieved_at": "",
        "scope_id": "scope",
        "accepted_taxon_key": accepted_taxon_key,
        "gbif_species_key": accepted_taxon_key.removeprefix("gbif:"),
        "scientific_name": scientific_name,
        "canonical_name": scientific_name,
        "rank": "SPECIES",
        "taxonomic_status": "accepted",
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": accepted_taxon_key,
        "species": species,
        "species_epithet": scientific_name.split()[-1],
        "in_scope": True,
        "classification_enabled": True,
        "classification_disabled_reason": "",
    }


def _family_label_scores(store: ButterflyTaxonomyStore, score_by_family: dict[str, float]) -> dict[str, float]:
    return {
        str(row["label"]): float(score_by_family.get(str(row["family"]), 0.0))
        for row in store.family_labels.to_dicts()
    }


def _species_label_scores(store: ButterflyTaxonomyStore, score_by_species: dict[str, float]) -> dict[str, float]:
    return {
        str(row["label"]): float(score_by_species.get(str(row["scientific_name"]), 0.0))
        for row in store.species_labels.to_dicts()
    }


def _combined_label_scores(
    store: ButterflyTaxonomyStore,
    *,
    family_scores: dict[str, float],
    species_scores: dict[str, float],
) -> dict[str, float]:
    return {
        **_family_label_scores(store, family_scores),
        **_species_label_scores(store, species_scores),
    }
