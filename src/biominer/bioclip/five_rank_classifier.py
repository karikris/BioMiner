from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Any, Mapping, Protocol, Sequence

import polars as pl

from biominer.bioclip.five_rank_store import FiveRankTaxonomyStore
from biominer.bioclip.five_rank_embedding_cache import FiveRankTextEmbeddingIndex
from biominer.registry.classification_v2 import CLASSIFICATION_RANKS
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION


DEFAULT_BEAM_WIDTHS: dict[str, int] = {
    "FAMILY": 3,
    "SUBFAMILY": 3,
    "TRIBE": 3,
    "GENUS": 5,
}

FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS: dict[str, pl.DataType] = {
    "taxonomy_table_version": pl.String,
    "taxonomy_prompt_variant_version": pl.String,
    "selected_family_key": pl.String,
    "selected_family": pl.String,
    "family_top3_accepted_taxon_keys": pl.List(pl.String),
    "family_top3_scores": pl.List(pl.Float64),
    "species_candidate_family_key": pl.String,
    "species_candidate_family": pl.String,
    "species_candidate_count": pl.Int64,
    "species_top20_scores": pl.List(pl.Float64),
    "species_top5_scores": pl.List(pl.Float64),
    "selected_subfamily_key": pl.String,
    "selected_subfamily": pl.String,
    "selected_tribe_key": pl.String,
    "selected_tribe": pl.String,
    "selected_genus_key": pl.String,
    "selected_genus": pl.String,
    "taxonomy_source_release": pl.String,
    "taxonomy_fingerprint": pl.String,
    "embedding_cache_fingerprint": pl.String,
    "classification_path_json": pl.String,
    "rank_candidates_json": pl.String,
    "candidate_counts_json": pl.String,
    "pruning_decisions_json": pl.String,
    "skipped_level_reasons_json": pl.String,
    "rerank_mode": pl.String,
    "species_rerank_candidate_count": pl.Int64,
}


class FiveRankScorer(Protocol):
    model_id: str
    model_checkpoint: str

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]: ...


@dataclass(frozen=True)
class RankCandidateScore:
    node_id: str
    scientific_name: str
    rank: str
    score: float
    best_label: str
    label_count: int
    parent_node_id: str | None = None
    cumulative_path_score: float = 0.0
    accepted_taxon_key: str | None = None
    gbif_species_key: str | None = None


@dataclass(frozen=True)
class PruningDecision:
    rank: str
    parent_node_ids: tuple[str, ...]
    candidate_count: int
    beam_width: int
    selected_node_ids: tuple[str, ...]
    pruned_node_ids: tuple[str, ...]
    skipped_reason: str | None = None


@dataclass(frozen=True)
class FiveRankCascadeResult:
    source: str
    flickr_photo_id: str
    detection_id: str
    crop_hash: str
    classification_version: str
    source_release: str
    prompt_version: str
    taxonomy_fingerprint: str
    embedding_cache_fingerprint: str | None
    rank_top_candidates: dict[str, tuple[RankCandidateScore, ...]]
    candidate_counts: dict[str, int]
    selected_path: dict[str, RankCandidateScore]
    species_top20: tuple[RankCandidateScore, ...]
    species_reranked: tuple[RankCandidateScore, ...]
    species_top1: RankCandidateScore | None
    species_top1_margin: float | None
    pruning_decisions: tuple[PruningDecision, ...]
    skipped_level_reasons: dict[str, str]
    rerank_mode: str
    classified_at: str


@dataclass(frozen=True)
class _PathState:
    nodes: tuple[RankCandidateScore, ...]
    stage_scores: tuple[float, ...]

    @property
    def cumulative_score(self) -> float:
        return sum(self.stage_scores) / len(self.stage_scores)

    @property
    def leaf(self) -> RankCandidateScore:
        return self.nodes[-1]


def classify_five_rank_crop(
    *,
    item: dict[str, Any],
    scorer: FiveRankScorer,
    taxonomy_store: FiveRankTaxonomyStore,
    beam_widths: Mapping[str, int] | None = None,
    species_first_pass_top_k: int = 20,
    species_rerank_top_k: int = 5,
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
    _embedding_index: FiveRankTextEmbeddingIndex | None = None,
    _image_embedding: Sequence[float] | None = None,
) -> FiveRankCascadeResult:
    widths = _beam_widths(beam_widths)
    if species_first_pass_top_k <= 0:
        raise ValueError("species_first_pass_top_k must be positive")
    if species_rerank_top_k <= 0 or species_rerank_top_k > species_first_pass_top_k:
        raise ValueError("species_rerank_top_k must be positive and <= species_first_pass_top_k")
    embedding_index = _embedding_index
    image_embedding = _image_embedding
    if taxonomy_text_embedding_cache is not None and embedding_index is not None:
        raise ValueError("pass either taxonomy_text_embedding_cache or a validated embedding index")
    if taxonomy_text_embedding_cache is not None:
        embedding_index = FiveRankTextEmbeddingIndex.from_frame(
            taxonomy_text_embedding_cache,
            taxonomy_store=taxonomy_store,
            model_id=scorer.model_id,
            model_checkpoint=scorer.model_checkpoint,
        )
        embed_items = getattr(scorer, "embed_image_items", None)
        if not callable(embed_items):
            raise ValueError("classification-v2 text embedding cache requires scorer.embed_image_items")
        embedded = embed_items([item])
        if len(embedded) != 1:
            raise ValueError("classification-v2 image embedder must return one row per crop")
        image_embedding = embedded[0]

    rank_top: dict[str, tuple[RankCandidateScore, ...]] = {rank: () for rank in CLASSIFICATION_RANKS}
    candidate_counts: dict[str, int] = {rank: 0 for rank in CLASSIFICATION_RANKS}
    decisions: list[PruningDecision] = []
    skipped: dict[str, str] = {}

    family_candidates = taxonomy_store.candidates("FAMILY")
    family_scores = _score_candidates(item, scorer, taxonomy_store, family_candidates, embedding_index, image_embedding)
    candidate_counts["FAMILY"] = len(family_scores)
    family_selected = family_scores[: widths["FAMILY"]]
    rank_top["FAMILY"] = tuple(family_selected)
    decisions.append(_decision("FAMILY", (), family_scores, widths["FAMILY"], family_selected))
    paths = [_PathState(nodes=(score,), stage_scores=(score.score,)) for score in family_selected]

    for rank in ("SUBFAMILY", "TRIBE", "GENUS"):
        if not paths:
            skipped[rank] = "no_surviving_parent_path"
            decisions.append(PruningDecision(rank, (), 0, widths[rank], (), (), skipped[rank]))
            continue
        parent_ids = tuple(path.leaf.node_id for path in paths)
        candidates = taxonomy_store.child_candidates(parent_ids, child_rank=rank)
        scores = _score_candidates(item, scorer, taxonomy_store, candidates, embedding_index, image_embedding)
        candidate_counts[rank] = len(scores)
        score_by_node = {score.node_id: score for score in scores}
        extensions: list[_PathState] = []
        for path in paths:
            children = taxonomy_store.child_candidates((path.leaf.node_id,), child_rank=rank)
            for child_id in children["node_id"].to_list():
                score = score_by_node.get(str(child_id))
                if score is None:
                    continue
                child_score = RankCandidateScore(
                    **{
                        **asdict(score),
                        "parent_node_id": path.leaf.node_id,
                        "cumulative_path_score": (sum(path.stage_scores) + score.score) / (len(path.stage_scores) + 1),
                    }
                )
                extensions.append(
                    _PathState(
                        nodes=(*path.nodes, child_score),
                        stage_scores=(*path.stage_scores, score.score),
                    )
                )
        extensions.sort(key=lambda path: (-path.cumulative_score, path.leaf.scientific_name, path.leaf.node_id))
        selected_paths = extensions[: widths[rank]]
        selected_scores = [path.leaf for path in selected_paths]
        rank_top[rank] = tuple(selected_scores)
        decisions.append(_decision(rank, parent_ids, scores, widths[rank], selected_scores))
        if not selected_paths:
            skipped[rank] = "no_enabled_reviewed_children"
        paths = selected_paths

    species_top20: tuple[RankCandidateScore, ...] = ()
    species_reranked: tuple[RankCandidateScore, ...] = ()
    selected_path: dict[str, RankCandidateScore] = {}
    if paths:
        genus_ids = tuple(path.leaf.node_id for path in paths)
        species_candidates = taxonomy_store.species_candidates_for_genera(genus_ids)
        species_scores = _score_candidates(item, scorer, taxonomy_store, species_candidates, embedding_index, image_embedding)
        candidate_counts["SPECIES"] = len(species_scores)
        path_by_genus = {path.leaf.node_id: path for path in paths}
        genus_by_species: dict[str, str] = {}
        for genus_id in genus_ids:
            for species_id in taxonomy_store.species_candidates_for_genera((genus_id,))["node_id"].to_list():
                genus_by_species[str(species_id)] = genus_id
        scored_paths: list[tuple[float, RankCandidateScore, _PathState]] = []
        for score in species_scores:
            genus_id = genus_by_species.get(score.node_id, "")
            parent_path = path_by_genus.get(genus_id)
            if parent_path is None:
                continue
            mapping = taxonomy_store.gbif_mapping_for_species_nodes((score.node_id,))
            mapped = mapping.row(0, named=True) if not mapping.is_empty() else {}
            cumulative = (sum(parent_path.stage_scores) + score.score) / (len(parent_path.stage_scores) + 1)
            species_score = RankCandidateScore(
                **{
                    **asdict(score),
                    "parent_node_id": genus_id,
                    "cumulative_path_score": cumulative,
                    "accepted_taxon_key": str(mapped.get("accepted_taxon_key") or "") or None,
                    "gbif_species_key": str(mapped.get("gbif_species_key") or "") or None,
                }
            )
            scored_paths.append((cumulative, species_score, parent_path))
        scored_paths.sort(key=lambda entry: (-entry[0], entry[1].scientific_name, entry[1].node_id))
        first_pass = scored_paths[:species_first_pass_top_k]
        species_top20 = tuple(entry[1] for entry in first_pass)
        rank_top["SPECIES"] = species_top20
        decisions.append(
            _decision(
                "SPECIES",
                genus_ids,
                [entry[1] for entry in scored_paths],
                species_first_pass_top_k,
                list(species_top20),
            )
        )
        rerank_nodes = taxonomy_store.nodes.filter(
            pl.col("node_id").is_in([score.node_id for score in species_top20])
        )
        reranked_scores = _score_candidates(item, scorer, taxonomy_store, rerank_nodes, embedding_index, image_embedding)
        first_pass_by_id = {entry[1].node_id: entry for entry in first_pass}
        reranked: list[tuple[RankCandidateScore, _PathState]] = []
        for rerank_score in reranked_scores:
            first_pass_entry = first_pass_by_id[rerank_score.node_id]
            original = first_pass_entry[1]
            reranked.append(
                (
                    RankCandidateScore(
                        **{
                            **asdict(rerank_score),
                            "parent_node_id": original.parent_node_id,
                            "cumulative_path_score": original.cumulative_path_score,
                            "accepted_taxon_key": original.accepted_taxon_key,
                            "gbif_species_key": original.gbif_species_key,
                        }
                    ),
                    first_pass_entry[2],
                )
            )
        reranked.sort(key=lambda entry: (-entry[0].score, entry[0].scientific_name, entry[0].node_id))
        species_reranked = tuple(entry[0] for entry in reranked)
        if reranked:
            winning_species, winning_parent_path = reranked[0]
            selected_path = {score.rank: score for score in winning_parent_path.nodes}
            selected_path["SPECIES"] = winning_species
    else:
        skipped["SPECIES"] = "no_surviving_genus_path"
        decisions.append(PruningDecision("SPECIES", (), 0, species_first_pass_top_k, (), (), skipped["SPECIES"]))

    species_top1 = species_reranked[0] if species_reranked else None
    margin = (
        species_reranked[0].score - species_reranked[1].score
        if len(species_reranked) > 1
        else None
    )
    releases = sorted(
        {
            str(row["source_release"])
            for row in taxonomy_store.nodes.filter(pl.col("enabled")).select("source_release").unique().iter_rows(named=True)
            if str(row["source_release"])
        }
    )
    return FiveRankCascadeResult(
        source=str(item.get("source") or ""),
        flickr_photo_id=str(item.get("flickr_photo_id") or ""),
        detection_id=str(item.get("detection_id") or ""),
        crop_hash=str(item.get("crop_hash") or ""),
        classification_version=taxonomy_store.classification_version,
        source_release="; ".join(releases),
        prompt_version=taxonomy_store.prompt_version,
        taxonomy_fingerprint=taxonomy_store.taxonomy_fingerprint,
        embedding_cache_fingerprint=embedding_index.cache_fingerprint if embedding_index is not None else None,
        rank_top_candidates=rank_top,
        candidate_counts=candidate_counts,
        selected_path=selected_path,
        species_top20=species_top20,
        species_reranked=species_reranked,
        species_top1=species_top1,
        species_top1_margin=margin,
        pruning_decisions=tuple(decisions),
        skipped_level_reasons=skipped,
        rerank_mode="rerank_all_first_pass_top20",
        classified_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )


def classify_five_rank_crops_batch(
    *,
    items: Sequence[dict[str, Any]],
    scorer: FiveRankScorer,
    taxonomy_store: FiveRankTaxonomyStore,
    beam_widths: Mapping[str, int] | None = None,
    species_first_pass_top_k: int = 20,
    species_rerank_top_k: int = 5,
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
) -> list[FiveRankCascadeResult]:
    embedding_index = None
    image_embeddings: list[Sequence[float] | None] = [None] * len(items)
    if taxonomy_text_embedding_cache is not None:
        embedding_index = FiveRankTextEmbeddingIndex.from_frame(
            taxonomy_text_embedding_cache,
            taxonomy_store=taxonomy_store,
            model_id=scorer.model_id,
            model_checkpoint=scorer.model_checkpoint,
        )
        embed_items = getattr(scorer, "embed_image_items", None)
        if not callable(embed_items):
            raise ValueError("classification-v2 text embedding cache requires scorer.embed_image_items")
        embedded = embed_items(list(items))
        if len(embedded) != len(items):
            raise ValueError("classification-v2 image embedder must return one row per crop")
        image_embeddings = list(embedded)
    return [
        classify_five_rank_crop(
            item=item,
            scorer=scorer,
            taxonomy_store=taxonomy_store,
            beam_widths=beam_widths,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            _embedding_index=embedding_index,
            _image_embedding=image_embeddings[index],
        )
        for index, item in enumerate(items)
    ]


def five_rank_result_to_object_score_row(
    *,
    item: dict[str, Any],
    result: FiveRankCascadeResult,
    scorer: FiveRankScorer,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> dict[str, Any]:
    family_candidates = result.rank_top_candidates.get("FAMILY", ())[:family_top_k]
    selected_family = result.selected_path.get("FAMILY")
    selected_subfamily = result.selected_path.get("SUBFAMILY")
    selected_tribe = result.selected_path.get("TRIBE")
    selected_genus = result.selected_path.get("GENUS")
    species_top = result.species_reranked[:species_rerank_top_k]
    species_top1 = result.species_top1
    return {
        "source": result.source or str(item.get("source") or ""),
        "flickr_photo_id": result.flickr_photo_id or str(item.get("flickr_photo_id") or ""),
        "detection_id": result.detection_id or str(item.get("detection_id") or ""),
        "crop_hash": result.crop_hash or str(item.get("crop_hash") or ""),
        "model_id": scorer.model_id,
        "model_version": str(getattr(scorer, "model_version", "")),
        "model_checkpoint": scorer.model_checkpoint,
        "candidate_set_id": result.taxonomy_fingerprint,
        "classified_at": result.classified_at,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "candidate_selection_mode": "reviewed_five_rank_beam",
        "candidate_source": "reviewed_classification_v2",
        "taxonomy_table_version": result.classification_version,
        "taxonomy_prompt_variant_version": result.prompt_version,
        "ablation_mode": str(item.get("ablation_mode") or "detector_crop"),
        "species_first_pass_top_k": int(species_first_pass_top_k),
        "species_rerank_top_k": int(species_rerank_top_k),
        "species_rerank_strategy": result.rerank_mode,
        "triage_group_top": "butterfly_like",
        "triage_group_scores": {"butterfly_like": float(item.get("detector_score") or 0.0)},
        "family_top3": [score.scientific_name for score in family_candidates],
        "family_top3_accepted_taxon_keys": [score.node_id for score in family_candidates],
        "family_top3_scores": [score.score for score in family_candidates],
        "family_top1": selected_family.scientific_name if selected_family else None,
        "family_top1_score": selected_family.score if selected_family else 0.0,
        "family_margin": _score_margin(family_candidates),
        "selected_family_key": selected_family.node_id if selected_family else None,
        "selected_family": selected_family.scientific_name if selected_family else None,
        "genus_top8": [score.scientific_name for score in result.rank_top_candidates.get("GENUS", ())[:8]],
        "genus_top1": selected_genus.scientific_name if selected_genus else None,
        "genus_top1_score": selected_genus.score if selected_genus else None,
        "genus_margin": _score_margin(result.rank_top_candidates.get("GENUS", ())),
        "species_candidate_family_key": selected_family.node_id if selected_family else None,
        "species_candidate_family": selected_family.scientific_name if selected_family else None,
        "species_candidate_count": int(result.candidate_counts.get("SPECIES", 0)),
        "species_top20": [score.scientific_name for score in result.species_top20],
        "species_top20_accepted_taxon_keys": [score.accepted_taxon_key for score in result.species_top20],
        "species_top20_scores": [score.score for score in result.species_top20],
        "species_top5": [score.scientific_name for score in species_top],
        "species_top5_accepted_taxon_keys": [score.accepted_taxon_key for score in species_top],
        "species_top5_scores": [score.score for score in species_top],
        "species_top1": species_top1.scientific_name if species_top1 else None,
        "species_top1_scientific_name": species_top1.scientific_name if species_top1 else None,
        "species_top1_accepted_taxon_key": species_top1.accepted_taxon_key if species_top1 else None,
        "accepted_taxon_key": species_top1.accepted_taxon_key if species_top1 else None,
        "species_top1_score": species_top1.score if species_top1 else 0.0,
        "species_top1_margin": result.species_top1_margin,
        "target_accepted_taxon_key": None,
        "target_species_score": None,
        "target_species_rank": None,
        "geospatial_prior_score": 0.0,
        "geospatial_prior_reason": "not_applied_open_classification",
        "text_evidence_score": 0.0,
        "comment_evidence_score": 0.0,
        "is_target_positive": False,
        "is_negative_material": False,
        "occurrence_bin": "in_review",
        "bin_reason": "five_rank_open_classification_requires_review",
        "selected_subfamily_key": selected_subfamily.node_id if selected_subfamily else None,
        "selected_subfamily": selected_subfamily.scientific_name if selected_subfamily else None,
        "selected_tribe_key": selected_tribe.node_id if selected_tribe else None,
        "selected_tribe": selected_tribe.scientific_name if selected_tribe else None,
        "selected_genus_key": selected_genus.node_id if selected_genus else None,
        "selected_genus": selected_genus.scientific_name if selected_genus else None,
        "taxonomy_source_release": result.source_release,
        "taxonomy_fingerprint": result.taxonomy_fingerprint,
        "embedding_cache_fingerprint": result.embedding_cache_fingerprint,
        "classification_path_json": json.dumps(
            {rank: asdict(score) for rank, score in result.selected_path.items()}, sort_keys=True
        ),
        "rank_candidates_json": json.dumps(
            {
                rank: [asdict(score) for score in candidates]
                for rank, candidates in result.rank_top_candidates.items()
            },
            sort_keys=True,
        ),
        "candidate_counts_json": json.dumps(result.candidate_counts, sort_keys=True),
        "pruning_decisions_json": json.dumps([asdict(decision) for decision in result.pruning_decisions], sort_keys=True),
        "skipped_level_reasons_json": json.dumps(result.skipped_level_reasons, sort_keys=True),
        "rerank_mode": result.rerank_mode,
        "species_rerank_candidate_count": len(result.species_reranked),
    }


def _score_candidates(
    item: dict[str, Any],
    scorer: FiveRankScorer,
    store: FiveRankTaxonomyStore,
    candidates: pl.DataFrame,
    embedding_index: FiveRankTextEmbeddingIndex | None,
    image_embedding: Sequence[float] | None,
) -> list[RankCandidateScore]:
    if candidates.is_empty():
        return []
    node_ids = candidates["node_id"].to_list()
    prompts = store.prompt_rows_for_nodes(node_ids)
    if prompts.is_empty():
        return []
    labels = tuple(prompts["label"].to_list())
    if embedding_index is not None:
        if image_embedding is None:
            raise AssertionError("validated embedding index requires an image embedding")
        label_scores = embedding_index.score(image_embedding, labels)
    else:
        label_scores = scorer.score(item, labels)
    rows_by_node: dict[str, list[dict[str, Any]]] = {}
    for row in prompts.iter_rows(named=True):
        rows_by_node.setdefault(str(row["node_id"]), []).append(row)
    scores = []
    for candidate in candidates.iter_rows(named=True):
        node_id = str(candidate["node_id"])
        label_rows = rows_by_node.get(node_id, [])
        values = [(str(row["label"]), float(label_scores.get(str(row["label"]), 0.0))) for row in label_rows]
        if not values:
            continue
        best_label, _ = sorted(values, key=lambda entry: (-entry[1], entry[0]))[0]
        score = sum(value for _label, value in values) / len(values)
        scores.append(
            RankCandidateScore(
                node_id=node_id,
                scientific_name=str(candidate["scientific_name"]),
                rank=str(candidate["rank"]),
                score=score,
                best_label=best_label,
                label_count=len(values),
            )
        )
    return sorted(scores, key=lambda candidate: (-candidate.score, candidate.scientific_name, candidate.node_id))


def _decision(
    rank: str,
    parents: Sequence[str],
    candidates: Sequence[RankCandidateScore],
    beam_width: int,
    selected: Sequence[RankCandidateScore],
) -> PruningDecision:
    selected_ids = tuple(score.node_id for score in selected)
    selected_set = set(selected_ids)
    return PruningDecision(
        rank=rank,
        parent_node_ids=tuple(parents),
        candidate_count=len(candidates),
        beam_width=beam_width,
        selected_node_ids=selected_ids,
        pruned_node_ids=tuple(score.node_id for score in candidates if score.node_id not in selected_set),
        skipped_reason=None if candidates else "no_enabled_reviewed_candidates",
    )


def _beam_widths(values: Mapping[str, int] | None) -> dict[str, int]:
    widths = dict(DEFAULT_BEAM_WIDTHS)
    for rank, value in dict(values or {}).items():
        normalized = str(rank).upper()
        if normalized not in widths:
            raise ValueError(f"beam width is unsupported for rank {rank}")
        widths[normalized] = int(value)
    invalid = [rank for rank, value in widths.items() if value <= 0]
    if invalid:
        raise ValueError("beam widths must be positive: " + ", ".join(invalid))
    return widths


def _score_margin(scores: Sequence[RankCandidateScore]) -> float | None:
    return scores[0].score - scores[1].score if len(scores) > 1 else None


__all__ = [
    "DEFAULT_BEAM_WIDTHS",
    "FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS",
    "FiveRankCascadeResult",
    "PruningDecision",
    "RankCandidateScore",
    "classify_five_rank_crop",
    "classify_five_rank_crops_batch",
    "five_rank_result_to_object_score_row",
]
