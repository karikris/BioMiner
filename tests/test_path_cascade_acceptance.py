from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

import polars as pl
import pytest

from biominer.benchmarks.path_cascade import (
    BENCHMARK_SELECTED_GENUS_NODE_IDS,
    DeterministicRawSimilarityScorer,
    SevenFamilyPathCascadeFixture,
    build_seven_family_path_cascade_fixture,
)
from biominer.bioclip.cascade_contract import DEFAULT_RANK_BEAM_WIDTH
from biominer.bioclip.path_cascade_classifier import (
    INTERMEDIATE_CLASSIFICATION_RANKS,
    PathCascadeResult,
    classify_path_cascade,
    classify_path_cascade_batch,
)
from biominer.bioclip.path_taxonomy_store import (
    RANK_SCREEN_PROMPT_STAGE,
    SPECIES_FIRST_PASS_PROMPT_STAGE,
    SPECIES_RERANK_PROMPT_STAGE,
    PathTaxonomyStore,
)
from biominer.bioclip.taxonomy_embedding_cache import (
    TaxonomyTextEmbeddingIndex,
    build_taxonomy_text_embedding_cache,
)


_DIVERGENCE_PROFILE: dict[tuple[str, str], float] = {
    (RANK_SCREEN_PROMPT_STAGE, "fixture:family:01"): 0.10,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:family:02"): 0.99,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:family:03"): 0.80,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:01:01"): 0.95,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:01:02"): 0.10,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:02:01"): 0.80,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:02:02"): 0.79,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:03:01"): 0.78,
    (RANK_SCREEN_PROMPT_STAGE, "fixture:subfamily:03:02"): 0.00,
}
_PRODUCTION_SUBFAMILY_TOP3 = (
    "fixture:subfamily:01:01",
    "fixture:subfamily:02:01",
    "fixture:subfamily:02:02",
)
_CUMULATIVE_SUBFAMILY_TOP3 = (
    "fixture:subfamily:02:01",
    "fixture:subfamily:02:02",
    "fixture:subfamily:03:01",
)
_PRUNED_GENUS_NODE_ID = "fixture:genus:01:01:02:01"
_PRUNED_GENUS_SPECIES_NODE_ID = "fixture:species:01:01:02:01:01"


class _FixedRawProfileScorer:
    model_id = "synthetic-acceptance-raw-scorer"
    model_checkpoint = "fixed-profile-v1"

    def __init__(
        self,
        taxonomy_store: PathTaxonomyStore,
        overrides: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        labels = tuple(
            sorted(
                set(
                    taxonomy_store.prompt_labels.filter(pl.col("enabled"))[
                        "label"
                    ].to_list()
                )
            )
        )
        baseline = DeterministicRawSimilarityScorer(taxonomy_store)
        self._score_by_label = dict(baseline.raw_similarities({}, labels))
        for row in taxonomy_store.prompt_labels.filter(pl.col("enabled")).iter_rows(
            named=True
        ):
            identity = (str(row["prompt_stage"]), str(row["node_id"]))
            if overrides is not None and identity in overrides:
                self._score_by_label[str(row["label"])] = float(overrides[identity])
        if any(not -1.0 <= score <= 1.0 for score in self._score_by_label.values()):
            raise ValueError("acceptance raw similarities must remain within [-1, 1]")
        self.calls: list[tuple[str, ...]] = []

    def raw_similarities(
        self,
        item: dict[str, Any],
        labels: tuple[str, ...],
    ) -> Mapping[str, float]:
        del item
        requested = tuple(str(label) for label in labels)
        self.calls.append(requested)
        return {label: self._score_by_label[label] for label in requested}


class _FixedEmbeddingScorer:
    model_id = "synthetic-acceptance-embedding-scorer"
    model_checkpoint = "fixed-profile-v1"

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self._vectors = {
            str(label): [float(value) for value in vector]
            for label, vector in vectors.items()
        }
        self.image_calls: list[int] = []
        self.text_calls: list[tuple[str, ...]] = []

    def embed_image_items(
        self,
        items: Sequence[dict[str, Any]],
    ) -> list[list[float]]:
        self.image_calls.append(len(items))
        return [[float(value) for value in item["embedding"]] for item in items]

    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]:
        requested = tuple(str(label) for label in labels)
        self.text_calls.append(requested)
        return [self._vectors[label] for label in requested]


@pytest.fixture(scope="module")
def seven_family_fixture() -> SevenFamilyPathCascadeFixture:
    fixture = build_seven_family_path_cascade_fixture()
    assert fixture.manifest["qa_status"] == "passed"
    assert fixture.manifest["fatal_finding_count"] == 0
    assert fixture.qa_findings.filter(pl.col("severity") == "fatal").is_empty()
    assert fixture.taxonomy_store.rank_candidates("FAMILY").height == 7
    return fixture


def test_acceptance_global_rank_beam_starts_with_seven_families_and_has_no_parent_quota(
    seven_family_fixture: SevenFamilyPathCascadeFixture,
) -> None:
    store = seven_family_fixture.taxonomy_store
    result = classify_path_cascade(
        item={"acceptance_case": "global-rank-beam"},
        scorer=DeterministicRawSimilarityScorer(store),
        taxonomy_store=store,
    )

    family_step = _rank_step(result, "FAMILY")
    assert family_step.candidate_count == 7
    assert set(family_step.candidate_node_ids) == set(
        store.rank_candidates("FAMILY")["node_id"].to_list()
    )
    for rank in INTERMEDIATE_CLASSIFICATION_RANKS:
        step = _rank_step(result, rank)
        assert step.retained_count == min(DEFAULT_RANK_BEAM_WIDTH, step.candidate_count)
        assert all(-1.0 <= score <= 1.0 for score in step.candidate_raw_similarities)

    genus_step = _rank_step(result, "GENUS")
    assert genus_step.retained_node_ids == BENCHMARK_SELECTED_GENUS_NODE_IDS
    selected_genus_paths = store.enabled_paths().filter(
        pl.col("genus_node_id").is_in(genus_step.retained_node_ids)
    )
    assert selected_genus_paths["genus_node_id"].n_unique() == 3
    assert selected_genus_paths["subtribe_node_id"].unique().to_list() == [
        "fixture:subtribe:01:01:01"
    ]
    candidate_genus_paths = store.enabled_paths().filter(
        pl.col("genus_node_id").is_in(genus_step.candidate_node_ids)
    )
    candidate_parent_count = (
        candidate_genus_paths.with_columns(
            pl.when(pl.col("subtribe_node_id") != "")
            .then(pl.col("subtribe_node_id"))
            .otherwise(pl.col("tribe_node_id"))
            .alias("genus_parent_node_id")
        )["genus_parent_node_id"]
        .n_unique()
    )
    assert candidate_parent_count > 1


def test_acceptance_current_rank_scores_not_cumulative_parent_scores_select_the_beam(
    seven_family_fixture: SevenFamilyPathCascadeFixture,
) -> None:
    store = seven_family_fixture.taxonomy_store
    result = classify_path_cascade(
        item={"acceptance_case": "current-rank-only"},
        scorer=_FixedRawProfileScorer(store, _DIVERGENCE_PROFILE),
        taxonomy_store=store,
    )
    family_step = _rank_step(result, "FAMILY")
    subfamily_step = _rank_step(result, "SUBFAMILY")

    assert subfamily_step.retained_node_ids == _PRODUCTION_SUBFAMILY_TOP3
    family_scores = dict(
        zip(
            family_step.candidate_node_ids,
            family_step.candidate_raw_similarities,
            strict=True,
        )
    )
    subfamily_scores = dict(
        zip(
            subfamily_step.candidate_node_ids,
            subfamily_step.candidate_raw_similarities,
            strict=True,
        )
    )
    parent_by_subfamily = {
        str(row["subfamily_node_id"]): str(row["family_node_id"])
        for row in store.enabled_paths()
        .select("family_node_id", "subfamily_node_id")
        .unique()
        .iter_rows(named=True)
    }
    cumulative_order = tuple(
        sorted(
            subfamily_step.candidate_node_ids,
            key=lambda node_id: (
                -(
                    family_scores[parent_by_subfamily[node_id]]
                    + subfamily_scores[node_id]
                )
                / 2.0,
                node_id,
            ),
        )[:DEFAULT_RANK_BEAM_WIDTH]
    )

    assert cumulative_order == _CUMULATIVE_SUBFAMILY_TOP3
    assert cumulative_order != subfamily_step.retained_node_ids
    assert all(-1.0 <= score <= 1.0 for score in family_scores.values())
    assert all(-1.0 <= score <= 1.0 for score in subfamily_scores.values())


def test_acceptance_reviewed_subtribe_skips_survive_without_consuming_beam_slots(
    seven_family_fixture: SevenFamilyPathCascadeFixture,
) -> None:
    store = seven_family_fixture.taxonomy_store
    result = classify_path_cascade(
        item={"acceptance_case": "reviewed-subtribe-skips"},
        scorer=DeterministicRawSimilarityScorer(store),
        taxonomy_store=store,
    )
    active_paths = store.enabled_paths()
    for rank in ("FAMILY", "SUBFAMILY", "TRIBE"):
        active_paths = store.filter_paths_by_rank_nodes(
            active_paths,
            rank,
            _rank_step(result, rank).retained_node_ids,
            carry_reviewed_skip_paths=True,
        )

    subtribe_step = _rank_step(result, "SUBTRIBE")
    reviewed_skips = store.reviewed_skip_paths(active_paths, "SUBTRIBE")
    after_subtribe = store.filter_paths_by_rank_nodes(
        active_paths,
        "SUBTRIBE",
        subtribe_step.retained_node_ids,
        carry_reviewed_skip_paths=True,
    )

    assert subtribe_step.candidate_count == subtribe_step.retained_count == 2
    assert subtribe_step.reviewed_skip_path_count == reviewed_skips.height == 2
    assert subtribe_step.retained_count == min(
        DEFAULT_RANK_BEAM_WIDTH,
        subtribe_step.candidate_count,
    )
    assert set(reviewed_skips["hierarchy_hash"].to_list()) <= set(
        after_subtribe["hierarchy_hash"].to_list()
    )
    assert after_subtribe.height == subtribe_step.active_path_count_after
    assert subtribe_step.active_path_count_after == subtribe_step.active_path_count_before


def test_acceptance_species_scoring_is_genus_bounded_then_top20_top5_top3(
    seven_family_fixture: SevenFamilyPathCascadeFixture,
) -> None:
    store = seven_family_fixture.taxonomy_store
    overrides = {
        (RANK_SCREEN_PROMPT_STAGE, _PRUNED_GENUS_NODE_ID): -1.0,
        (SPECIES_FIRST_PASS_PROMPT_STAGE, _PRUNED_GENUS_SPECIES_NODE_ID): 1.0,
        (SPECIES_RERANK_PROMPT_STAGE, _PRUNED_GENUS_SPECIES_NODE_ID): 1.0,
    }
    scorer = _FixedRawProfileScorer(store, overrides)
    result = classify_path_cascade(
        item={"acceptance_case": "species-candidate-boundary"},
        scorer=scorer,
        taxonomy_store=store,
    )
    genus_step = _rank_step(result, "GENUS")
    species_step = _rank_step(result, "SPECIES")
    genus_paths = store.filter_paths_by_rank_nodes(
        store.enabled_paths(),
        "GENUS",
        genus_step.retained_node_ids,
    )
    expected_species_universe = set(
        store.species_nodes_in_paths(genus_paths)["node_id"].to_list()
    )

    assert genus_step.retained_node_ids == BENCHMARK_SELECTED_GENUS_NODE_IDS
    assert _PRUNED_GENUS_NODE_ID in genus_step.pruned_node_ids
    assert set(species_step.candidate_node_ids) == expected_species_universe
    assert _PRUNED_GENUS_SPECIES_NODE_ID not in species_step.candidate_node_ids
    excluded_prompts = {
        str(label)
        for stage in (
            SPECIES_FIRST_PASS_PROMPT_STAGE,
            SPECIES_RERANK_PROMPT_STAGE,
        )
        for label in store.prompt_rows_for_nodes(
            (_PRUNED_GENUS_SPECIES_NODE_ID,),
            stage,
        )["label"].to_list()
    }
    assert excluded_prompts.isdisjoint(label for call in scorer.calls for label in call)

    expected_top20 = tuple(
        f"fixture:species:01:01:01:01:{number:02d}" for number in range(1, 21)
    )
    first_excluded_species = "fixture:species:01:01:01:01:21"
    expected_reranked = tuple(reversed(expected_top20))
    assert tuple(score.node_id for score in result.species_top20) == expected_top20
    assert tuple(score.node_id for score in result.species_reranked_top20) == (
        expected_reranked
    )
    assert tuple(score.node_id for score in result.species_top5) == expected_reranked[:5]
    assert tuple(score.node_id for score in result.species_top3) == expected_reranked[:3]
    assert result.species_top3 == result.species_top5[:3]
    assert result.species_top5 != result.species_top20[:5]

    expected_rerank_labels = set(
        store.prompt_rows_for_nodes(expected_top20, SPECIES_RERANK_PROMPT_STAGE)[
            "label"
        ].to_list()
    )
    assert set(scorer.calls[-1]) == expected_rerank_labels
    excluded_rerank_labels = tuple(
        store.prompt_rows_for_nodes(
            (first_excluded_species,),
            SPECIES_RERANK_PROMPT_STAGE,
        )["label"].to_list()
    )
    assert set(excluded_rerank_labels).isdisjoint(scorer.calls[-1])
    potential_rerank_scores = DeterministicRawSimilarityScorer(
        store
    ).raw_similarities(
        {},
        tuple(
            store.prompt_rows_for_nodes(
                (expected_top20[-1], first_excluded_species),
                SPECIES_RERANK_PROMPT_STAGE,
            )["label"].to_list()
        ),
    )
    retained_twentieth_labels = set(
        store.prompt_rows_for_nodes(
            (expected_top20[-1],),
            SPECIES_RERANK_PROMPT_STAGE,
        )["label"].to_list()
    )
    assert min(
        potential_rerank_scores[label] for label in excluded_rerank_labels
    ) > max(
        potential_rerank_scores[label] for label in retained_twentieth_labels
    )
    assert result.species_rerank_step.candidate_count == 20
    assert result.species_rerank_step.retained_count == 5
    assert all(-1.0 <= score <= 1.0 for score in species_step.candidate_raw_similarities)
    assert all(
        -1.0 <= score <= 1.0
        for score in result.species_rerank_step.candidate_raw_similarities
    )


def test_acceptance_cached_and_direct_embeddings_have_identical_rankings(
    seven_family_fixture: SevenFamilyPathCascadeFixture,
) -> None:
    store = seven_family_fixture.taxonomy_store
    labels = tuple(
        sorted(set(store.prompt_labels.filter(pl.col("enabled"))["label"].to_list()))
    )
    fixed_scores = DeterministicRawSimilarityScorer(store).raw_similarities({}, labels)
    assert all(-1.0 <= score <= 1.0 for score in fixed_scores.values())
    vectors = {
        label: [score, sqrt(max(0.0, 1.0 - score * score))]
        for label, score in fixed_scores.items()
    }
    cache_builder = _FixedEmbeddingScorer(vectors)
    cache = build_taxonomy_text_embedding_cache(
        store,
        model_id=cache_builder.model_id,
        model_checkpoint=cache_builder.model_checkpoint,
        embed_labels=cache_builder.embed_text_labels,
        batch_size=19,
    )
    index = TaxonomyTextEmbeddingIndex.from_frame(
        cache,
        taxonomy_store=store,
        model_id=cache_builder.model_id,
        model_checkpoint=cache_builder.model_checkpoint,
    )
    direct_scorer = _FixedEmbeddingScorer(vectors)
    cached_scorer = _FixedEmbeddingScorer(vectors)
    items = (
        {"embedding": [1.0, 0.0], "acceptance_item": "one"},
        {"embedding": [1.0, 0.0], "acceptance_item": "two"},
    )

    direct_results = classify_path_cascade_batch(
        items=items,
        embedding_scorer=direct_scorer,
        taxonomy_store=store,
    )
    cached_results = classify_path_cascade_batch(
        items=items,
        embedding_scorer=cached_scorer,
        taxonomy_store=store,
        taxonomy_text_embedding_index=index,
    )

    assert direct_scorer.image_calls == cached_scorer.image_calls == [2]
    assert direct_scorer.text_calls
    assert cached_scorer.text_calls == []
    for direct, cached in zip(direct_results, cached_results, strict=True):
        assert _ranking_signature(direct) == _ranking_signature(cached)
        assert direct.embedding_cache_fingerprint is None
        assert cached.embedding_cache_fingerprint == index.cache_fingerprint


def _rank_step(result: PathCascadeResult, rank: str):  # noqa: ANN202
    return next(step for step in result.rank_steps if step.rank == rank)


def _ranking_signature(result: PathCascadeResult) -> tuple[object, ...]:
    return (
        tuple(
            (
                step.rank,
                step.candidate_node_ids,
                step.retained_node_ids,
                step.pruned_node_ids,
            )
            for step in result.rank_steps
        ),
        tuple(score.node_id for score in result.species_top20),
        tuple(score.node_id for score in result.species_reranked_top20),
        tuple(score.node_id for score in result.species_top5),
        tuple(score.node_id for score in result.species_top3),
    )
