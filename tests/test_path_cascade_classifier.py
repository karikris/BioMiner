from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from itertools import chain
import json
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import pytest

from biominer.bioclip.path_cascade_classifier import (
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    PathCascadeClassificationError,
    RankCandidateScore,
    classify_path_cascade,
    classify_path_cascade_batch,
    score_rank_candidates,
)
from biominer.bioclip.path_cascade_output import (
    PATH_CASCADE_OUTPUT_SCHEMA,
    path_cascade_output_frame,
    path_cascade_result_to_object_score_row,
    path_cascade_result_to_output_row,
    write_path_cascade_output,
)
from biominer.bioclip.object_runner import (
    OBJECT_SCORE_OUTPUT_SCHEMA,
    PHOTO_EVIDENCE_SUMMARY_SCHEMA,
    _object_evidence_joined,
    _photo_summary,
)
from biominer.bioclip.taxonomy_embedding_cache import (
    TaxonomyTextEmbeddingIndex,
    build_taxonomy_text_embedding_cache,
)
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    PROMPT_LABEL_SCHEMA,
    QA_FINDING_SCHEMA,
    SOURCE_SCHEMA,
)


class _RawScorer:
    model_id = "fake-raw-bioclip"
    model_checkpoint = "fake-raw-checkpoint"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, ...]] = []

    def raw_similarities(self, item, labels):  # noqa: ANN001, ANN201 - protocol fake.
        self.calls.append(tuple(labels))
        return {label: self.scores[label] for label in labels}


class _EmbeddingScorer:
    model_id = "fake-embedding-bioclip"
    model_checkpoint = "fake-embedding-checkpoint"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.image_calls: list[int] = []
        self.text_calls: list[tuple[str, ...]] = []

    def embed_image_items(self, items: Sequence[dict[str, Any]]) -> list[list[float]]:
        self.image_calls.append(len(items))
        return [[float(value) for value in item["embedding"]] for item in items]

    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]:
        self.text_calls.append(tuple(labels))
        return [self.vectors[label] for label in labels]


def test_result_models_exclude_complete_path_selection_scores() -> None:
    names = {field.name for field in fields(RankCandidateScore)}
    assert {
        "node_id",
        "rank",
        "scientific_name",
        "raw_similarity",
        "best_label",
        "best_label_similarity",
        "label_count",
    } <= names
    assert "score" not in names
    assert not any("cumulative" in name for name in names)
    score = RankCandidateScore("n", "FAMILY", "Name", 0.5, "label", 0.5, 1)
    with pytest.raises(FrozenInstanceError):
        score.raw_similarity = 0.9  # type: ignore[misc]


def test_rank_scoring_means_prompt_variants_and_breaks_ties_by_stable_identity() -> None:
    store = _store()
    extra = store.prompt_labels.filter(pl.col("node_id") == "f:a").with_columns(
        pl.lit("variant::f:a").alias("label"),
        pl.lit("variant::{name}").alias("prompt_template"),
        pl.lit(2, dtype=pl.Int64).alias("sort_order"),
    )
    store = replace(store, prompt_labels=pl.concat([store.prompt_labels, extra]))
    candidates = store.rank_candidates("FAMILY")
    scorer = _scorer(
        store,
        {"f:a": 0.9, "f:b": 0.6, "f:c": -0.2, "f:d": -0.3},
        label_overrides={"variant::f:a": 0.1},
    )

    scores = score_rank_candidates(
        item={},
        scorer=scorer,
        taxonomy_store=store,
        candidates=candidates,
    )

    assert [score.node_id for score in scores[:2]] == ["f:b", "f:a"]
    alpha = next(score for score in scores if score.node_id == "f:a")
    assert alpha.raw_similarity == pytest.approx(0.5)
    assert alpha.best_label == _label("f:a")
    assert alpha.best_label_similarity == 0.9
    assert alpha.label_count == 2

    same_name = store.nodes.filter(pl.col("node_id").is_in(["g:a1b", "g:a2u"]))
    tied = score_rank_candidates(
        item={},
        scorer=_scorer(store, {"g:a1b": 0.4, "g:a2u": 0.4}),
        taxonomy_store=store,
        candidates=same_name.reverse(),
    )
    assert [score.node_id for score in tied] == ["g:a1b", "g:a2u"]


def test_global_subfamily_beam_can_retain_three_nodes_from_one_family() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.9,
            "f:b": 0.8,
            "f:c": 0.7,
            "f:d": 0.6,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "sf:a4": 0.96,
            "sf:b1": 0.95,
            "sf:b2": 0.94,
            "sf:c1": 0.93,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SUBFAMILY")

    assert step.candidate_count == 7
    assert step.retained_count == 3
    assert [score.node_id for score in step.top_candidates] == ["sf:a1", "sf:a2", "sf:a3"]
    assert step.retained_node_ids == ("sf:a1", "sf:a2", "sf:a3")
    assert set(step.candidate_node_ids) == {
        "sf:a1",
        "sf:a2",
        "sf:a3",
        "sf:a4",
        "sf:b1",
        "sf:b2",
        "sf:c1",
    }
    assert set(step.pruned_node_ids) == {"sf:a4", "sf:b1", "sf:b2", "sf:c1"}
    assert len(step.candidate_raw_similarities) == step.candidate_count
    assert result.beam_strategy == GLOBAL_RANK_TOP_K_BEAM_STRATEGY
    assert result.rank_beam_width == 3
    assert result.taxonomy_fingerprint == store.hierarchy_fingerprint


def test_current_rank_child_score_beats_higher_prior_branch_score() -> None:
    store = _store()
    result = classify_path_cascade(
        item={},
        taxonomy_store=store,
        scorer=_scorer(
            store,
            {
                "f:a": 0.10,
                "f:b": 0.99,
                "f:c": 0.80,
                "f:d": 0.01,
                "sf:a1": 0.95,
                "sf:b2": 0.80,
                "sf:b1": 0.79,
                "sf:c1": 0.78,
                "sf:a2": 0.20,
                "sf:a3": 0.19,
                "sf:a4": 0.18,
            },
        ),
    )

    assert [score.node_id for score in _step(result, "SUBFAMILY").top_candidates] == [
        "sf:a1",
        "sf:b2",
        "sf:b1",
    ]


def test_repeated_leaf_paths_score_each_ancestor_node_once() -> None:
    store = _only_species_prefix(_store(), "s:a1:")
    scorer = _scorer(store, {})

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)

    for rank in ("FAMILY", "SUBFAMILY", "TRIBE", "SUBTRIBE", "GENUS"):
        assert _step(result, rank).candidate_count == 1
    labels = list(chain.from_iterable(scorer.calls))
    for node_id in ("f:a", "sf:a1", "t:a1", "u:a1", "g:a1"):
        assert labels.count(_label(node_id)) == 1
    assert _step(result, "SPECIES").candidate_count == 25


def test_family_rank_three_preserves_true_branch_but_rank_four_cannot_reenter() -> None:
    store = _store()
    true_scores = {
        "f:a": 0.99,
        "f:b": 0.90,
        "f:c": 0.80,
        "f:d": 0.70,
        "sf:c1": 0.99,
        "sf:a1": 0.90,
        "sf:b1": 0.80,
        "t:c1": 0.99,
        "t:a1": 0.90,
        "t:b1": 0.80,
        "g:c1": 0.99,
        "g:a1": 0.90,
        "g:b1": 0.80,
        "s:c1": 0.99,
    }
    result = classify_path_cascade(
        item={},
        scorer=_scorer(store, true_scores),
        taxonomy_store=store,
    )

    assert _step(result, "FAMILY").top_candidates[0].node_id == "f:a"
    assert result.species_top1 is not None and result.species_top1.node_id == "s:c1"
    assert result.final_winning_path[0].node_id == "f:c"
    output = path_cascade_result_to_output_row(result)
    assert output["family_top1"] == "Alphaidae"
    assert output["family_top1_node_id"] == "f:a"
    assert output["selected_family"] == "Gammaidae"
    assert output["selected_family_node_id"] == "f:c"
    assert output["family_top3_node_ids"] == ["f:a", "f:b", "f:c"]
    assert output["species_top1_node_id"] == "s:c1"
    assert output["species_top1_accepted_taxon_key"] == "gbif:2201"
    overlay_node_ids: set[str] = set()
    for prefix in ("family", "subfamily", "tribe", "subtribe", "genus"):
        names = output[f"{prefix}_top3"]
        node_ids = output[f"{prefix}_top3_node_ids"]
        scores = output[f"{prefix}_top3_scores"]
        assert len(names) == len(node_ids) == len(scores) <= 3
        overlay_node_ids.update(node_ids)
    species_score_columns = {
        "species_top20": "species_top20_first_pass_scores",
        "species_top5": "species_top5_rerank_scores",
        "species_top3": "species_top3_rerank_scores",
    }
    for prefix, score_column in species_score_columns.items():
        assert len(output[prefix]) == len(output[f"{prefix}_node_ids"])
        assert len(output[prefix]) == len(output[f"{prefix}_accepted_taxon_keys"])
        assert len(output[prefix]) == len(output[score_column])
    accepted_keys = set(output["species_top20_accepted_taxon_keys"])
    assert all(key.startswith("gbif:") for key in accepted_keys)
    assert overlay_node_ids.isdisjoint(accepted_keys)
    assert output["species_top3"] == output["species_top5"][:3]
    assert set(output["species_top5_node_ids"]) <= set(output["species_top20_node_ids"])
    trace = json.loads(output["pruning_trace_json"])
    assert len(trace) == 7
    assert [entry["prompt_stage"] for entry in trace[-2:]] == [
        "species_first_pass",
        "species_rerank",
    ]
    assert "cumulative" not in output["pruning_trace_json"]
    for entry in trace:
        assert entry["candidate_count"] == len(entry["union_candidate_node_ids"])
        assert entry["candidate_count"] == len(entry["candidate_raw_similarities"])
        assert set(entry["retained_node_ids"]).isdisjoint(entry["pruned_node_ids"])
        assert set(entry["retained_node_ids"]) | set(entry["pruned_node_ids"]) == set(
            entry["union_candidate_node_ids"]
        )

    pruned_scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.90,
            "f:c": 0.80,
            "f:d": 0.70,
            "sf:d1": 1.0,
            "t:d1": 1.0,
            "g:d1": 1.0,
            "s:d1": 1.0,
        },
    )
    pruned = classify_path_cascade(item={}, scorer=pruned_scorer, taxonomy_store=store)
    called_labels = set(chain.from_iterable(pruned_scorer.calls[1:]))
    assert _label("sf:d1") not in called_labels
    assert _label("s:d1") not in called_labels
    assert all(score.node_id != "s:d1" for score in pruned.species_top20)


def test_cascade_object_audit_survives_join_and_summarizes_winning_genus() -> None:
    store = _store()
    result = classify_path_cascade(
        item={},
        taxonomy_store=store,
        scorer=_scorer(
            store,
            {
                "f:a": 0.99,
                "f:b": 0.90,
                "f:c": 0.80,
                "f:d": 0.70,
                "sf:c1": 0.99,
                "sf:a1": 0.90,
                "sf:b1": 0.80,
                "t:c1": 0.99,
                "t:a1": 0.90,
                "t:b1": 0.80,
                "g:c1": 0.99,
                "g:a1": 0.90,
                "g:b1": 0.80,
                "s:c1": 0.99,
            },
        ),
    )
    scorer = _scorer(store, {})
    item = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "detection-1",
        "crop_hash": "sha256:crop",
        "visual_input_id": "visual-1",
        "detector_score": 0.95,
    }
    row = path_cascade_result_to_object_score_row(
        item=item,
        result=result,
        scorer=scorer,
    )
    scores = pl.DataFrame([row], schema=OBJECT_SCORE_OUTPUT_SCHEMA)
    canonical = pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1"}])
    detections = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "detection-1",
            }
        ]
    )

    joined = _object_evidence_joined(
        canonical=canonical,
        detections=detections,
        scores=scores,
    )
    summary = _photo_summary(scores)

    assert joined["family_top3_node_ids"][0].to_list() == ["f:a", "f:b", "f:c"]
    assert joined["family_top3_accepted_taxon_keys"][0].to_list() == []
    assert joined["selected_family_node_id"][0] == "f:c"
    assert joined["selected_genus_node_id"][0] == "g:c1"
    assert summary.schema == PHOTO_EVIDENCE_SUMMARY_SCHEMA
    assert summary["photo_selected_family"][0] == "Gammaidae"
    assert summary["photo_selected_genus"][0] == "GammaOne"
    assert summary["photo_selected_genus_node_id"][0] == "g:c1"
    assert summary["photo_species_top1_key"][0] == "gbif:2201"

    alias = dict(row)
    alias.update(
        {
            "detection_id": "detection-2",
            "species_top1": "Gamma one alias",
            "species_top1_scientific_name": "Gamma one alias",
        }
    )
    alias_summary = _photo_summary(
        pl.DataFrame([row, alias], schema=OBJECT_SCORE_OUTPUT_SCHEMA)
    )
    assert alias_summary["photo_multi_object_conflict"][0] is False

    conflicting = dict(alias)
    conflicting["species_top1_accepted_taxon_key"] = "gbif:9999"
    conflicting["accepted_taxon_key"] = "gbif:9999"
    conflict_summary = _photo_summary(
        pl.DataFrame([row, conflicting], schema=OBJECT_SCORE_OUTPUT_SCHEMA)
    )
    assert conflict_summary["photo_multi_object_conflict"][0] is True


def test_genus_beam_is_three_and_excluded_genus_species_are_never_scored() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.20,
            "f:c": 0.19,
            "f:d": 0.18,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "t:a1": 0.99,
            "t:a2": 0.98,
            "t:a3": 0.97,
            "u:a1": 0.99,
            "u:a2": 0.98,
            "u:a3": 0.97,
            "g:a1": 0.90,
            "g:a1b": 0.80,
            "g:a2": 0.70,
            "g:a3": 0.60,
            "s:a3": 1.0,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    genus = _step(result, "GENUS")

    assert genus.candidate_count == 5
    assert genus.retained_count == 3
    assert [score.node_id for score in genus.top_candidates] == ["g:a1", "g:a1b", "g:a2"]
    assert _label("s:a3") not in set(scorer.calls[-1])
    assert all(score.node_id != "s:a3" for score in result.species_top20)


def test_reviewed_subtribe_skip_does_not_consume_a_beam_slot() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.20,
            "f:c": 0.19,
            "f:d": 0.18,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "t:a1": 0.99,
            "t:a2": 0.98,
            "t:a3": 0.97,
            "u:a1": 0.99,
            "u:a1-2": 0.98,
            "u:a1-3": 0.97,
            "u:a1-4": 0.20,
            "u:a2": 0.19,
            "u:a3": 0.18,
            "g:askip": 0.99,
            "g:a1": 0.90,
            "g:a2u": 0.80,
            "s:askip": 1.0,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    subtribe = _step(result, "SUBTRIBE")
    genus_call = next(call for call in scorer.calls if _label("g:askip") in call)

    assert subtribe.skipped is False
    assert subtribe.retained_count == 3
    assert [score.node_id for score in subtribe.top_candidates] == ["u:a1", "u:a1-2", "u:a1-3"]
    assert _label("g:askip") in genus_call
    assert _label("g:a4u") not in genus_call
    assert result.species_top1 is not None and result.species_top1.node_id == "s:askip"
    assert result.skipped_ranks == ("SUBTRIBE",)
    output = path_cascade_result_to_output_row(result)
    assert output["subtribe_top3_node_ids"] == ["u:a1", "u:a1-2", "u:a1-3"]
    assert output["selected_subtribe"] is None
    assert output["selected_subtribe_node_id"] is None
    assert output["skipped_ranks"] == ["SUBTRIBE"]


def test_fully_skipped_subtribe_records_skip_without_scoring_placeholders(
    tmp_path: Path,
) -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:b": 0.99,
            "f:c": 0.98,
            "f:d": 0.97,
            "f:a": 0.10,
            "sf:b1": 0.99,
            "sf:c1": 0.98,
            "sf:d1": 0.97,
            "sf:b2": 0.10,
            "t:b1": 0.99,
            "t:c1": 0.98,
            "t:d1": 0.97,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SUBTRIBE")

    assert step.skipped is True
    assert step.retained_count == 0
    assert step.active_path_count_before == step.active_path_count_after == 3
    assert step.skip_reason == "all_active_paths_reviewed_rank_skip"
    assert step.candidate_node_ids == step.retained_node_ids == step.pruned_node_ids == ()
    assert step.reviewed_skip_path_count == 3
    assert len(scorer.calls) == 6  # FAMILY, SUBFAMILY, TRIBE, GENUS, species first pass/rerank
    output = path_cascade_result_to_output_row(result)
    assert output["subtribe_top3"] == []
    assert output["subtribe_top3_node_ids"] == []
    assert output["subtribe_top3_scores"] == []
    assert output["subtribe_top1"] is None
    assert output["selected_subtribe"] is None
    assert output["fully_skipped_ranks"] == ["SUBTRIBE"]
    assert output["candidate_counts_by_rank"]["SUBTRIBE"] == 0
    assert (
        output["active_path_counts_before_by_rank"]["SUBTRIBE"]
        == output["active_path_counts_after_by_rank"]["SUBTRIBE"]
    )
    path = write_path_cascade_output(
        path_cascade_output_frame([output]),
        tmp_path / "skipped-subtribe.parquet",
    )
    restored = pl.read_parquet(path)
    assert dict(restored.schema) == PATH_CASCADE_OUTPUT_SCHEMA
    assert restored["subtribe_top3"][0].to_list() == []


def test_mixed_optional_rank_rejects_paths_without_node_or_reviewed_skip() -> None:
    store = _store()
    rows = store.leaf_paths.filter(
        pl.col("species_node_id").is_in(["s:a1b", "s:a1:00"])
    ).to_dicts()
    malformed = next(row for row in rows if row["species_node_id"] == "s:a1b")
    malformed["subtribe_node_id"] = ""
    malformed["subtribe"] = ""
    malformed["skipped_ranks"] = []
    malformed["path_completeness"] = "incomplete"
    malformed["rank_path"] = [rank for rank in malformed["rank_path"] if rank != "SUBTRIBE"]
    malformed["rank_path_node_ids"] = [
        node_id for node_id in malformed["rank_path_node_ids"] if node_id != "u:a1"
    ]
    malformed["hierarchy_hash"] = "sha256:malformed-optional-path"
    paths = pl.DataFrame(rows, schema=LEAF_PATH_SCHEMA)
    store = replace(
        store,
        leaf_paths=paths,
        _enabled_hierarchy_hashes=frozenset(paths["hierarchy_hash"].to_list()),
    )

    with pytest.raises(PathCascadeClassificationError) as captured:
        classify_path_cascade(item={}, scorer=_scorer(store, {}), taxonomy_store=store)

    assert captured.value.code == "incomplete_optional_rank_coverage"
    assert captured.value.rank == "SUBTRIBE"


def test_species_top_twenty_uses_species_score_only_beneath_genus_top_three() -> None:
    store = _store()
    overrides = {
        "f:a": 0.99,
        "f:b": 0.20,
        "f:c": 0.19,
        "f:d": 0.18,
        "sf:a1": 0.99,
        "sf:a2": 0.98,
        "sf:a3": 0.97,
        "t:a1": 0.99,
        "t:a2": 0.98,
        "t:a3": 0.97,
        "u:a1": 0.99,
        "u:a2": 0.98,
        "u:a3": 0.97,
        "g:a1": 0.90,
        "g:a2": 0.80,
        "g:a3": 0.70,
        "g:a1b": 0.60,
        "s:a2": 1.0,
        "s:a3": 0.10,
    }
    overrides.update({f"s:a1:{index:02d}": 0.90 - index * 0.01 for index in range(25)})
    scorer = _scorer(store, overrides)

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SPECIES")

    assert step.candidate_count == 27
    assert step.retained_count == 20
    assert result.species_top1 is not None and result.species_top1.node_id == "s:a2"
    assert result.species_top1.raw_similarity == 1.0
    assert result.species_top20[1].node_id == "s:a1:00"
    assert [score.node_id for score in result.species_top5] == [
        score.node_id for score in result.species_top20[:5]
    ]
    assert result.species_top3 == result.species_top5[:3]
    assert step.top1_margin == pytest.approx(0.10)
    assert len(scorer.calls[-2]) == 27
    assert len(scorer.calls[-1]) == 20


def test_species_rerank_scores_exactly_first_pass_top_twenty_with_distinct_prompts() -> None:
    store = _only_species_prefix(_store(), "s:a1:")
    first_pass_scores = {f"s:a1:{index:02d}": 1.0 - index * 0.01 for index in range(25)}
    rerank_scores = {
        _rerank_label(f"s:a1:{index:02d}"): (99.0 if index == 20 else index * 0.01)
        for index in range(25)
    }
    scorer = _scorer(store, first_pass_scores, label_overrides=rerank_scores)

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)

    assert [score.node_id for score in result.species_top20] == [
        f"s:a1:{index:02d}" for index in range(20)
    ]
    assert result.species_top1 is not None
    assert result.species_top1.node_id == "s:a1:19"
    assert result.species_top1.first_pass_raw_similarity == pytest.approx(0.81)
    assert result.species_top1.rerank_raw_similarity == pytest.approx(0.19)
    assert result.species_top1.raw_similarity == pytest.approx(0.19)
    assert result.species_top3 == result.species_top5[:3]
    assert result.species_top5 == result.species_reranked_top20[:5]
    assert {score.node_id for score in result.species_reranked_top20} == {
        score.node_id for score in result.species_top20
    }
    rerank_call = scorer.calls[-1]
    assert len(rerank_call) == 20
    assert _rerank_label("s:a1:19") in rerank_call
    assert _rerank_label("s:a1:20") not in rerank_call
    assert result.final_winning_path[-1] == result.species_top1
    assert result.species_rerank_step.candidate_count == 20
    assert result.species_rerank_step.retained_count == 5
    assert result.species_rerank_step.retained_node_ids == tuple(
        score.node_id for score in result.species_top5
    )


def test_direct_and_cached_embedding_batches_have_matching_raw_rankings() -> None:
    store = _store()
    labels = sorted(set(store.prompt_labels.filter(pl.col("enabled"))["label"].to_list()))
    denominator = max(1, len(labels) - 1)
    vectors = {}
    for index, label in enumerate(labels):
        score = 0.9 - (1.8 * index / denominator)
        vectors[label] = [score, sqrt(1.0 - score * score)]
    cache_builder = _EmbeddingScorer(vectors)
    cache = build_taxonomy_text_embedding_cache(
        store,
        model_id=cache_builder.model_id,
        model_checkpoint=cache_builder.model_checkpoint,
        embed_labels=cache_builder.embed_text_labels,
        batch_size=17,
    )
    index = TaxonomyTextEmbeddingIndex.from_frame(
        cache,
        taxonomy_store=store,
        model_id=cache_builder.model_id,
        model_checkpoint=cache_builder.model_checkpoint,
    )
    items = ({"embedding": [1.0, 0.0]}, {"embedding": [1.0, 0.0]})
    direct_scorer = _EmbeddingScorer(vectors)
    cached_scorer = _EmbeddingScorer(vectors)

    direct = classify_path_cascade_batch(
        items=items,
        embedding_scorer=direct_scorer,
        taxonomy_store=store,
    )
    cached = classify_path_cascade_batch(
        items=items,
        embedding_scorer=cached_scorer,
        taxonomy_store=store,
        taxonomy_text_embedding_index=index,
    )

    assert direct_scorer.image_calls == cached_scorer.image_calls == [2]
    assert cached_scorer.text_calls == []
    direct_labels = [label for call in direct_scorer.text_calls for label in call]
    assert len(direct_labels) == len(set(direct_labels))
    for direct_result, cached_result in zip(direct, cached, strict=True):
        assert cached_result.embedding_cache_fingerprint == index.cache_fingerprint
        assert direct_result.embedding_cache_fingerprint is None
        assert [
            [candidate.node_id for candidate in step.top_candidates]
            for step in direct_result.rank_steps
        ] == [
            [candidate.node_id for candidate in step.top_candidates]
            for step in cached_result.rank_steps
        ]
        for direct_step, cached_step in zip(
            direct_result.rank_steps,
            cached_result.rank_steps,
            strict=True,
        ):
            for direct_score, cached_score in zip(
                direct_step.top_candidates,
                cached_step.top_candidates,
                strict=True,
            ):
                assert direct_score.raw_similarity == pytest.approx(
                    cached_score.raw_similarity,
                    abs=1e-6,
                )
        assert [score.node_id for score in direct_result.species_top20] == [
            score.node_id for score in cached_result.species_top20
        ]
        assert [score.node_id for score in direct_result.species_reranked_top20] == [
            score.node_id for score in cached_result.species_reranked_top20
        ]
        for direct_score, cached_score in zip(
            direct_result.species_reranked_top20,
            cached_result.species_reranked_top20,
            strict=True,
        ):
            assert direct_score.raw_similarity == pytest.approx(
                cached_score.raw_similarity,
                abs=1e-6,
            )
            assert direct_score.first_pass_raw_similarity == pytest.approx(
                cached_score.first_pass_raw_similarity,
                abs=1e-6,
            )


@pytest.mark.parametrize("mode", ["missing", "duplicate", "nonaccepted", "mismatched_key"])
def test_retained_species_requires_exactly_one_accepted_gbif_mapping(mode: str) -> None:
    store = _only_species_prefix(_store(), "s:a1:00")
    species_id = "s:a1:00"
    if mode == "missing":
        mappings = store.gbif_mappings.filter(pl.col("species_node_id") != species_id)
    elif mode == "duplicate":
        row = store.gbif_mappings.filter(pl.col("species_node_id") == species_id)
        mappings = pl.concat([store.gbif_mappings, row])
    elif mode == "nonaccepted":
        mappings = store.gbif_mappings.with_columns(
            pl.when(pl.col("species_node_id") == species_id)
            .then(pl.lit("DOUBTFUL"))
            .otherwise(pl.col("taxonomic_status"))
            .alias("taxonomic_status")
        )
    else:
        mappings = store.gbif_mappings.with_columns(
            pl.when(pl.col("species_node_id") == species_id)
            .then(pl.lit("gbif:999999"))
            .otherwise(pl.col("accepted_taxon_key"))
            .alias("accepted_taxon_key")
        )
    store = replace(store, gbif_mappings=mappings)

    with pytest.raises(PathCascadeClassificationError) as captured:
        classify_path_cascade(item={}, scorer=_scorer(store, {}), taxonomy_store=store)

    assert captured.value.code == "invalid_species_mapping"
    assert captured.value.rank == "SPECIES"


def test_missing_mandatory_rank_candidates_raise_structured_failure() -> None:
    store = _store()
    nodes = store.nodes.with_columns(
        pl.when(pl.col("rank") == "SUBFAMILY")
        .then(pl.lit(False))
        .otherwise(pl.col("enabled"))
        .alias("enabled")
    )
    store = replace(store, nodes=nodes)

    with pytest.raises(PathCascadeClassificationError) as captured:
        classify_path_cascade(item={}, scorer=_scorer(store, {}), taxonomy_store=store)

    assert captured.value.as_dict() == {
        "code": "no_rank_candidates",
        "rank": "SUBFAMILY",
        "candidate_count": 0,
        "active_path_count": 36,
        "message": "mandatory rank has no candidates in the active path union",
    }


def _step(result, rank):  # noqa: ANN001, ANN202 - compact test helper.
    return next(step for step in result.rank_steps if step.rank == rank)


def _scorer(
    store: PathTaxonomyStore,
    overrides: dict[str, float],
    *,
    label_overrides: dict[str, float] | None = None,
) -> _RawScorer:
    scores = {
        str(row["label"]): float(overrides.get(str(row["node_id"]), 0.01))
        for row in store.prompt_labels.iter_rows(named=True)
    }
    scores.update(label_overrides or {})
    return _RawScorer(scores)


def _label(node_id: str) -> str:
    return f"prompt::{node_id}"


def _rerank_label(node_id: str) -> str:
    return f"rerank::{node_id}"


def _only_species_prefix(store: PathTaxonomyStore, prefix: str) -> PathTaxonomyStore:
    paths = store.leaf_paths.filter(pl.col("species_node_id").str.starts_with(prefix))
    return replace(
        store,
        leaf_paths=paths,
        _enabled_hierarchy_hashes=frozenset(paths["hierarchy_hash"].to_list()),
    )


@dataclass(frozen=True)
class _PathSpec:
    family: tuple[str, str]
    subfamily: tuple[str, str]
    tribe: tuple[str, str]
    subtribe: tuple[str, str] | None
    genus: tuple[str, str]
    species: tuple[str, str]
    gbif_key: str


def _store() -> PathTaxonomyStore:
    nodes: dict[str, dict[str, object]] = {}
    paths: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    for spec in _path_specs():
        ranked = [
            ("FAMILY", spec.family),
            ("SUBFAMILY", spec.subfamily),
            ("TRIBE", spec.tribe),
        ]
        if spec.subtribe is not None:
            ranked.append(("SUBTRIBE", spec.subtribe))
        ranked.extend([("GENUS", spec.genus), ("SPECIES", spec.species)])
        for rank, (node_id, name) in ranked:
            nodes.setdefault(
                node_id,
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "source_id": "fixture:v1",
                    "source_release": "fixture-v1",
                    "citation": "Phase 3 fixture",
                    "retrieved_at": "2026-07-11",
                    "evidence": "reviewed synthetic hierarchy",
                    "reviewed": True,
                    "review_status": "reviewed",
                    "reviewed_by": "Phase 3 test",
                    "reviewed_at": "2026-07-11",
                    "enabled": True,
                    "disabled_reason": "",
                },
            )
        by_rank = {rank: value for rank, value in ranked}
        rank_path = [rank for rank, _value in ranked]
        rank_path_node_ids = [value[0] for _rank, value in ranked]
        skipped = ["SUBTRIBE"] if spec.subtribe is None else []
        row: dict[str, object] = {
            "classification_version": CLASSIFICATION_V3_VERSION,
            "accepted_taxon_key": f"gbif:{spec.gbif_key}",
            "gbif_species_key": spec.gbif_key,
            "rank_path": rank_path,
            "rank_path_node_ids": rank_path_node_ids,
            "skipped_ranks": skipped,
            "path_completeness": "reviewed_optional_skip" if skipped else "complete",
            "hierarchy_hash": f"sha256:{spec.species[0]}",
            "source_release": "fixture-v1",
            "enabled": True,
            "disabled_reason": "",
        }
        for rank in CLASSIFICATION_RANKS:
            value = by_rank.get(rank, ("", ""))
            row[f"{rank.casefold()}_node_id"] = value[0]
            row[rank.casefold()] = value[1]
        paths.append(row)
        mappings.append(
            {
                "classification_version": CLASSIFICATION_V3_VERSION,
                "accepted_taxon_key": f"gbif:{spec.gbif_key}",
                "gbif_species_key": spec.gbif_key,
                "accepted_scientific_name": spec.species[1],
                "species_node_id": spec.species[0],
                "taxonomic_status": "ACCEPTED",
                "source_id": "fixture:v1",
                "source_release": "fixture-v1",
                "citation": "Phase 3 fixture",
                "retrieved_at": "2026-07-11",
                "evidence": "exact accepted fixture mapping",
                "reviewed": True,
                "review_status": "reviewed",
                "reviewed_by": "Phase 3 test",
                "reviewed_at": "2026-07-11",
                "enabled": True,
                "disabled_reason": "",
            }
        )
    node_frame = pl.DataFrame(list(nodes.values()), schema=NODE_SCHEMA)
    prompt_rows: list[dict[str, object]] = []
    for row in nodes.values():
        rank = str(row["rank"])
        stages_and_labels = (
            (
                ("species_first_pass", _label(str(row["node_id"]))),
                ("species_rerank", _rerank_label(str(row["node_id"]))),
            )
            if rank == "SPECIES"
            else (("rank_screen", _label(str(row["node_id"]))),)
        )
        for prompt_stage, label in stages_and_labels:
            prompt_rows.append(
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
                    "prompt_stage": prompt_stage,
                    "node_id": row["node_id"],
                    "rank": rank,
                    "scientific_name": row["scientific_name"],
                    "label": label,
                    "prompt_template": f"{prompt_stage}::{{name}}",
                    "sort_order": 1,
                    "enabled": True,
                }
            )
    path_frame = pl.DataFrame(paths, schema=LEAF_PATH_SCHEMA)
    return PathTaxonomyStore(
        sources=pl.DataFrame(schema=SOURCE_SCHEMA),
        nodes=node_frame,
        edges=pl.DataFrame(schema=EDGE_SCHEMA),
        gbif_mappings=pl.DataFrame(mappings, schema=GBIF_MAPPING_SCHEMA),
        leaf_paths=path_frame,
        prompt_labels=pl.DataFrame(prompt_rows, schema=PROMPT_LABEL_SCHEMA),
        qa_findings=pl.DataFrame(schema=QA_FINDING_SCHEMA),
        manifest={
            "classification_version": CLASSIFICATION_V3_VERSION,
            "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
            "hierarchy_fingerprint": "sha256:fixture-hierarchy",
            "classification_fingerprint": "sha256:fixture-classification",
        },
        _enabled_hierarchy_hashes=frozenset(path_frame["hierarchy_hash"].to_list()),
    )


def _path_specs() -> list[_PathSpec]:
    family_a = ("f:a", "Alphaidae")
    family_b = ("f:b", "Betaidae")
    family_c = ("f:c", "Gammaidae")
    family_d = ("f:d", "Deltaidae")
    specs = [
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1", "Alphaona", "g:a1b", "Duplicata", "s:a1b", "Trap alpha", "2001"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-2", "Alphatwoa", "g:a2u", "Duplicata", "s:a2u", "Alpha two", "2002"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-3", "Alphathreea", "g:a3u", "AlphaThree", "s:a3u", "Alpha three", "2003"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-4", "Alphafoura", "g:a4u", "AlphaFour", "s:a4u", "Alpha four", "2004"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", None, "", "g:askip", "AlphaSkip", "s:askip", "Alpha skip", "2005"),
        _spec(family_a, "a2", "AlphaTwoinae", "Alphatwoini", "u:a2", "Alphatwoa", "g:a2", "AlphaTwo", "s:a2", "Alpha branch two", "2006"),
        _spec(family_a, "a3", "AlphaThreeinae", "Alphathreeini", "u:a3", "Alphathreea", "g:a3", "AlphaThree", "s:a3", "Alpha branch three", "2007"),
        _spec(family_a, "a4", "AlphaFourinae", "Alphafourini", "u:a4", "Alphafoura", "g:a4", "AlphaFour", "s:a4", "Alpha branch four", "2008"),
        _spec(family_b, "b1", "BetaOneinae", "Betaonini", None, "", "g:b1", "BetaOne", "s:b1", "Beta one", "2101"),
        _spec(family_b, "b2", "BetaTwoinae", "Betatwoini", "u:b2", "Betatwoa", "g:b2", "BetaTwo", "s:b2", "Beta two", "2102"),
        _spec(family_c, "c1", "GammaOneinae", "Gammaonini", None, "", "g:c1", "GammaOne", "s:c1", "Gamma one", "2201"),
        _spec(family_d, "d1", "DeltaOneinae", "Deltaonini", None, "", "g:d1", "DeltaOne", "s:d1", "Delta one", "2301"),
    ]
    for index in range(25):
        specs.append(
            _spec(
                family_a,
                "a1",
                "AlphaOneinae",
                "Alphaonini",
                "u:a1",
                "Alphaona",
                "g:a1",
                "AlphaMain",
                f"s:a1:{index:02d}",
                f"Alpha species {index:02d}",
                str(3000 + index),
            )
        )
    return specs


def _spec(
    family: tuple[str, str],
    branch: str,
    subfamily_name: str,
    tribe_name: str,
    subtribe_id: str | None,
    subtribe_name: str,
    genus_id: str,
    genus_name: str,
    species_id: str,
    species_name: str,
    gbif_key: str,
) -> _PathSpec:
    return _PathSpec(
        family=family,
        subfamily=(f"sf:{branch}", subfamily_name),
        tribe=(f"t:{branch}", tribe_name),
        subtribe=(subtribe_id, subtribe_name) if subtribe_id else None,
        genus=(genus_id, genus_name),
        species=(species_id, species_name),
        gbif_key=gbif_key,
    )
