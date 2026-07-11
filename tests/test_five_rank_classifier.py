from __future__ import annotations

from dataclasses import replace
from math import exp, sqrt

import polars as pl
import pytest

from biominer.bioclip.five_rank_classifier import (
    classify_five_rank_crop,
    classify_five_rank_crops_batch,
    five_rank_result_to_object_score_row,
)
from biominer.bioclip.five_rank_embedding_cache import build_five_rank_text_embedding_cache
from biominer.bioclip.five_rank_store import FiveRankTaxonomyStore
from biominer.registry.classification_v2 import build_classification_v2_frames, build_classification_v2_manifest


class _RecordingScorer:
    model_id = "fake-bioclip"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, ...]] = []

    def score(self, item, labels):  # noqa: ANN001, ANN201 - classifier protocol fake.
        self.calls.append(tuple(labels))
        return {label: self.scores.get(label, 0.0) for label in labels}


class _EmbeddingScorer:
    model_id = "fake-embedding-bioclip"
    model_version = "test"
    model_checkpoint = "fake-embedding-checkpoint"

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.index = {label: index for index, label in enumerate(labels)}
        self.image = [1.0 / (index + 1) for index in range(len(labels))]
        self.image_embedding_batch_sizes: list[int] = []

    def embed_labels(self, labels: list[str]) -> list[list[float]]:
        vectors = []
        for label in labels:
            vector = [0.0] * len(self.labels)
            vector[self.index[label]] = 1.0
            vectors.append(vector)
        return vectors

    def embed_image_items(self, items):  # noqa: ANN001, ANN201 - classifier protocol fake.
        self.image_embedding_batch_sizes.append(len(items))
        return [self.image for _item in items]

    def score(self, item, labels):  # noqa: ANN001, ANN201 - classifier protocol fake.
        norm = sqrt(sum(value * value for value in self.image))
        logits = [100.0 * self.image[self.index[label]] / norm for label in labels]
        maximum = max(logits)
        weights = [exp(value - maximum) for value in logits]
        total = sum(weights)
        return {label: weight / total for label, weight in zip(labels, weights, strict=True)}


def test_five_rank_cascade_recovers_lower_rank_path_and_reranks_all_top20() -> None:
    store = _store(species_count=25)
    scorer = _RecordingScorer(_scores(species_count=25))

    result = classify_five_rank_crop(
        item=_item("one"),
        scorer=scorer,
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 1, "TRIBE": 1, "GENUS": 1},
    )

    assert result.rank_top_candidates["FAMILY"][0].scientific_name == "Alphaidae"
    assert result.selected_path["FAMILY"].scientific_name == "Betaidae"
    assert result.selected_path["SUBFAMILY"].scientific_name == "Betainae"
    assert result.selected_path["TRIBE"].scientific_name == "Betaini"
    assert result.selected_path["GENUS"].scientific_name == "Beta"
    assert result.candidate_counts["SPECIES"] == 25
    assert len(result.species_top20) == 20
    assert len(result.species_reranked) == 20
    assert result.rerank_mode == "rerank_all_first_pass_top20"
    assert len(scorer.calls[-1]) == 20
    assert all(score.scientific_name.startswith("Beta species") for score in result.species_top20)


def test_five_rank_cascade_records_pruning_and_skipped_levels() -> None:
    store = _store(species_count=2)
    store = replace(
        store,
        edges=store.edges.filter(~((pl.col("parent_rank") == "TRIBE") & (pl.col("child_rank") == "GENUS"))),
    )

    result = classify_five_rank_crop(
        item=_item("one"),
        scorer=_RecordingScorer(_scores(species_count=2)),
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 2, "TRIBE": 2, "GENUS": 2},
    )

    assert result.skipped_level_reasons == {
        "GENUS": "no_enabled_reviewed_children",
        "SPECIES": "no_surviving_genus_path",
    }
    assert result.species_top1 is None
    genus_decision = next(decision for decision in result.pruning_decisions if decision.rank == "GENUS")
    assert genus_decision.candidate_count == 0
    assert genus_decision.skipped_reason == "no_enabled_reviewed_candidates"


def test_five_rank_batch_and_single_results_are_equivalent() -> None:
    store = _store(species_count=3)
    scores = _scores(species_count=3)
    items = [_item("one"), _item("two")]

    batch = classify_five_rank_crops_batch(
        items=items,
        scorer=_RecordingScorer(scores),
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 1, "TRIBE": 1, "GENUS": 1},
    )
    singles = [
        classify_five_rank_crop(
            item=item,
            scorer=_RecordingScorer(scores),
            taxonomy_store=store,
            beam_widths={"FAMILY": 2, "SUBFAMILY": 1, "TRIBE": 1, "GENUS": 1},
        )
        for item in items
    ]

    assert [result.species_top1 for result in batch] == [result.species_top1 for result in singles]
    assert [result.selected_path for result in batch] == [result.selected_path for result in singles]
    assert [result.candidate_counts for result in batch] == [result.candidate_counts for result in singles]


def test_five_rank_cached_and_direct_scores_have_exact_ranking_parity() -> None:
    store = _store(species_count=3)
    labels = store.prompt_labels["label"].to_list()
    scorer = _EmbeddingScorer(labels)
    cache = build_five_rank_text_embedding_cache(
        store,
        model_id=scorer.model_id,
        model_checkpoint=scorer.model_checkpoint,
        embed_labels=scorer.embed_labels,
        batch_size=2,
    )

    direct = classify_five_rank_crop(
        item=_item("one"),
        scorer=scorer,
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 2, "TRIBE": 2, "GENUS": 2},
    )
    cached = classify_five_rank_crop(
        item=_item("one"),
        scorer=scorer,
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 2, "TRIBE": 2, "GENUS": 2},
        taxonomy_text_embedding_cache=cache,
    )

    for rank in cached.rank_top_candidates:
        assert [score.node_id for score in cached.rank_top_candidates[rank]] == [
            score.node_id for score in direct.rank_top_candidates[rank]
        ]
        assert [score.score for score in cached.rank_top_candidates[rank]] == pytest.approx(
            [score.score for score in direct.rank_top_candidates[rank]]
        )
    assert [score.node_id for score in cached.species_top20] == [score.node_id for score in direct.species_top20]
    assert [score.node_id for score in cached.species_reranked] == [score.node_id for score in direct.species_reranked]
    assert cached.embedding_cache_fingerprint == cache["embedding_cache_fingerprint"][0]


def test_five_rank_cache_rejects_taxonomy_fingerprint_mismatch() -> None:
    store = _store(species_count=2)
    scorer = _EmbeddingScorer(store.prompt_labels["label"].to_list())
    cache = build_five_rank_text_embedding_cache(
        store,
        model_id=scorer.model_id,
        model_checkpoint=scorer.model_checkpoint,
        embed_labels=scorer.embed_labels,
    ).with_columns(pl.lit("sha256:wrong").alias("taxonomy_fingerprint"))

    with pytest.raises(ValueError, match="taxonomy_fingerprint mismatch"):
        classify_five_rank_crop(
            item=_item("one"),
            scorer=scorer,
            taxonomy_store=store,
            taxonomy_text_embedding_cache=cache,
        )


def test_five_rank_cached_batch_embeds_all_images_once() -> None:
    store = _store(species_count=3)
    scorer = _EmbeddingScorer(store.prompt_labels["label"].to_list())
    cache = build_five_rank_text_embedding_cache(
        store,
        model_id=scorer.model_id,
        model_checkpoint=scorer.model_checkpoint,
        embed_labels=scorer.embed_labels,
    )

    results = classify_five_rank_crops_batch(
        items=[_item("one"), _item("two")],
        scorer=scorer,
        taxonomy_store=store,
        taxonomy_text_embedding_cache=cache,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 2, "TRIBE": 2, "GENUS": 2},
    )

    assert len(results) == 2
    assert scorer.image_embedding_batch_sizes == [2]


def test_five_rank_object_score_row_preserves_full_screening_audit() -> None:
    store = _store(species_count=25)
    scorer = _RecordingScorer(_scores(species_count=25))
    item = _item("one")
    result = classify_five_rank_crop(
        item=item,
        scorer=scorer,
        taxonomy_store=store,
        beam_widths={"FAMILY": 2, "SUBFAMILY": 1, "TRIBE": 1, "GENUS": 1},
    )

    row = five_rank_result_to_object_score_row(
        item=item,
        result=result,
        scorer=scorer,
        family_top_k=2,
        species_first_pass_top_k=20,
        species_rerank_top_k=5,
    )

    assert row["selected_subfamily"] == "Betainae"
    assert row["selected_tribe"] == "Betaini"
    assert row["selected_genus"] == "Beta"
    assert row["species_rerank_candidate_count"] == 20
    assert row["taxonomy_fingerprint"] == store.taxonomy_fingerprint
    assert row["rerank_mode"] == "rerank_all_first_pass_top20"
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "five_rank_open_classification_requires_review"
    assert '"SUBFAMILY"' in row["classification_path_json"]
    assert '"pruned_node_ids"' in row["pruning_decisions_json"]


def _store(*, species_count: int) -> FiveRankTaxonomyStore:
    source = _source(species_count=species_count)
    taxa = pl.DataFrame(
        [
            {
                "accepted_taxon_key": f"gbif:{1000 + index}",
                "species_key": f"gbif:{1000 + index}",
                "scientific_name": f"Beta species {index:02d}",
                "species": f"Beta species {index:02d}",
                "family": "Betaidae",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
            for index in range(species_count)
        ]
        + [
            {
                "accepted_taxon_key": "gbif:900",
                "species_key": "gbif:900",
                "scientific_name": "Alpha species",
                "species": "Alpha species",
                "family": "Alphaidae",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
        ]
    )
    frames = build_classification_v2_frames(taxa, source)
    manifest = build_classification_v2_manifest(frames, registry_version="test-v2")
    manifest.update(
        {
            "qa_status": "passed",
            "fatal_finding_count": 0,
            "classification_fingerprint": "fixture",
        }
    )
    return FiveRankTaxonomyStore(
        sources=frames.sources,
        nodes=frames.nodes,
        edges=frames.edges,
        gbif_mappings=frames.gbif_mappings,
        leaf_paths=frames.leaf_paths,
        prompt_labels=frames.prompt_labels,
        manifest=manifest,
    )


def _source(*, species_count: int) -> dict[str, object]:
    review = {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "test reviewer",
        "reviewed_at": "2026-07-11",
        "enabled": True,
    }
    sources = [
        {
            "source_id": "fixture",
            "authority": "Fixture authority",
            "release": "fixture-v1",
            "citation": "fixture citation",
            "retrieved_at": "2026-07-11",
            "evidence_url": "https://example.test/fixture",
            "evidence": "reviewed fixture hierarchy",
        }
    ]
    nodes = []
    edges = []
    mappings = []
    for prefix, names in (
        ("alpha", ("Alphaidae", "Alphainae", "Alphaini", "Alpha", "Alpha species")),
        ("beta", ("Betaidae", "Betainae", "Betaini", "Beta", None)),
    ):
        ranks = ("FAMILY", "SUBFAMILY", "TRIBE", "GENUS")
        ids = []
        for rank, name in zip(ranks, names[:4], strict=True):
            node_id = f"{rank.casefold()}:{prefix}"
            ids.append(node_id)
            nodes.append({"node_id": node_id, "rank": rank, "scientific_name": name, "source_id": "fixture", **review})
        for parent, child in zip(ids[:-1], ids[1:], strict=True):
            edges.append({"parent_node_id": parent, "child_node_id": child, "source_id": "fixture", **review})
    alpha_species_id = "species:alpha"
    nodes.append({"node_id": alpha_species_id, "rank": "SPECIES", "scientific_name": "Alpha species", "source_id": "fixture", **review})
    edges.append({"parent_node_id": "genus:alpha", "child_node_id": alpha_species_id, "source_id": "fixture", **review})
    mappings.append(
        {
            "gbif_species_key": "900",
            "accepted_scientific_name": "Alpha species",
            "species_node_id": alpha_species_id,
            "source_id": "fixture",
            **review,
        }
    )
    for index in range(species_count):
        node_id = f"species:beta-{index:02d}"
        name = f"Beta species {index:02d}"
        nodes.append({"node_id": node_id, "rank": "SPECIES", "scientific_name": name, "source_id": "fixture", **review})
        edges.append({"parent_node_id": "genus:beta", "child_node_id": node_id, "source_id": "fixture", **review})
        mappings.append(
            {
                "gbif_species_key": str(1000 + index),
                "accepted_scientific_name": name,
                "species_node_id": node_id,
                "source_id": "fixture",
                **review,
            }
        )
    return {"sources": sources, "nodes": nodes, "edges": edges, "species_mappings": mappings}


def _scores(*, species_count: int) -> dict[str, float]:
    scores = {
        "a photo of a butterfly in family Alphaidae": 0.90,
        "a photo of a butterfly in family Betaidae": 0.80,
        "a photo of a butterfly in subfamily Alphainae": 0.10,
        "a photo of a butterfly in subfamily Betainae": 0.99,
        "a photo of a butterfly in tribe Alphaini": 0.10,
        "a photo of a butterfly in tribe Betaini": 0.99,
        "a photo of a butterfly in genus Alpha": 0.10,
        "a photo of a butterfly in genus Beta": 0.99,
        "a photo of the butterfly species Alpha species": 0.10,
    }
    scores.update(
        {
            f"a photo of the butterfly species Beta species {index:02d}": 1.0 - index / 100
            for index in range(species_count)
        }
    )
    return scores


def _item(photo_id: str) -> dict[str, str]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "detection_id": f"det-{photo_id}",
        "crop_hash": f"sha256:{photo_id}",
    }
