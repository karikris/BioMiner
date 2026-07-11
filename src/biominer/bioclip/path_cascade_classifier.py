from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol

import polars as pl

from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore, RANK_SCREEN_PROMPT_STAGE
from biominer.registry.classification_v3 import CLASSIFICATION_RANKS, OPTIONAL_CLASSIFICATION_RANKS


GLOBAL_RANK_TOP_K_BEAM_STRATEGY = "global_rank_top_k"
DEFAULT_RANK_BEAM_WIDTH = 3
DEFAULT_SPECIES_FIRST_PASS_TOP_K = 20
INTERMEDIATE_CLASSIFICATION_RANKS = (
    "FAMILY",
    "SUBFAMILY",
    "TRIBE",
    "SUBTRIBE",
    "GENUS",
)


@dataclass(frozen=True)
class RankCandidateScore:
    node_id: str
    rank: str
    scientific_name: str
    raw_similarity: float
    best_label: str
    best_label_similarity: float
    label_count: int

    def __post_init__(self) -> None:
        if not self.node_id or not self.rank or not self.scientific_name:
            raise ValueError("rank candidate identity fields must be nonblank")
        if not isfinite(self.raw_similarity) or not isfinite(self.best_label_similarity):
            raise ValueError("rank candidate similarities must be finite")
        if self.label_count <= 0:
            raise ValueError("rank candidate label_count must be positive")


@dataclass(frozen=True)
class RankStepResult:
    rank: str
    candidate_count: int
    retained_count: int
    active_path_count_before: int
    active_path_count_after: int
    top_candidates: tuple[RankCandidateScore, ...]
    top1_margin: float | None
    parent_node_ids: tuple[str, ...]
    skipped: bool
    skip_reason: str | None

    def __post_init__(self) -> None:
        counts = (
            self.candidate_count,
            self.retained_count,
            self.active_path_count_before,
            self.active_path_count_after,
        )
        if any(value < 0 for value in counts):
            raise ValueError("rank-step counts must be nonnegative")
        if self.retained_count > self.candidate_count:
            raise ValueError("rank-step retained_count cannot exceed candidate_count")
        if len(self.top_candidates) != self.retained_count:
            raise ValueError("rank-step top_candidates must match retained_count")
        if self.skipped and self.retained_count:
            raise ValueError("a skipped rank cannot retain scored nodes")


@dataclass(frozen=True)
class PathCascadeResult:
    classification_version: str
    prompt_version: str
    taxonomy_fingerprint: str
    classification_fingerprint: str
    embedding_cache_fingerprint: str | None
    beam_strategy: str
    rank_beam_width: int
    rank_steps: tuple[RankStepResult, ...]
    species_top20: tuple[RankCandidateScore, ...]
    species_top5: tuple[RankCandidateScore, ...]
    species_top3: tuple[RankCandidateScore, ...]
    species_top1: RankCandidateScore | None
    final_winning_path: tuple[RankCandidateScore, ...]
    skipped_ranks: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.beam_strategy != GLOBAL_RANK_TOP_K_BEAM_STRATEGY:
            raise ValueError(f"unsupported cascade beam strategy: {self.beam_strategy}")
        if self.rank_beam_width <= 0:
            raise ValueError("rank_beam_width must be positive")


class PathCascadeClassificationError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        rank: str,
        candidate_count: int,
        active_path_count: int,
        message: str,
    ) -> None:
        self.code = str(code)
        self.rank = str(rank)
        self.candidate_count = int(candidate_count)
        self.active_path_count = int(active_path_count)
        self.message = str(message)
        super().__init__(
            f"{self.code}: rank={self.rank} candidates={self.candidate_count} "
            f"active_paths={self.active_path_count}: {self.message}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "rank": self.rank,
            "candidate_count": self.candidate_count,
            "active_path_count": self.active_path_count,
            "message": self.message,
        }


class PathCascadeScorer(Protocol):
    model_id: str
    model_checkpoint: str

    def raw_similarities(
        self,
        item: dict[str, Any],
        labels: tuple[str, ...],
    ) -> Mapping[str, float]: ...


def score_rank_candidates(
    *,
    item: dict[str, Any],
    scorer: PathCascadeScorer,
    taxonomy_store: PathTaxonomyStore,
    candidates: pl.DataFrame,
    prompt_stage: str = RANK_SCREEN_PROMPT_STAGE,
) -> tuple[RankCandidateScore, ...]:
    if candidates.is_empty():
        return ()
    required = {"node_id", "rank", "scientific_name"}
    missing_columns = sorted(required - set(candidates.columns))
    if missing_columns:
        raise ValueError("rank candidates are missing columns: " + ", ".join(missing_columns))
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates.iter_rows(named=True):
        node_id = str(candidate.get("node_id") or "")
        identity = {
            "node_id": node_id,
            "rank": str(candidate.get("rank") or ""),
            "scientific_name": str(candidate.get("scientific_name") or ""),
        }
        existing = candidate_by_id.setdefault(node_id, identity)
        if existing != identity:
            raise ValueError(f"conflicting rank candidate rows for node_id: {node_id}")
    node_ids = tuple(sorted(candidate_by_id))
    prompts = taxonomy_store.prompt_rows_for_nodes(node_ids, prompt_stage)
    if prompts.is_empty():
        raise ValueError(f"classification-v3 has no {prompt_stage} prompts for rank candidates")
    prompt_rows_by_node: dict[str, list[dict[str, Any]]] = {}
    for prompt in prompts.iter_rows(named=True):
        prompt_rows_by_node.setdefault(str(prompt["node_id"]), []).append(dict(prompt))
    labels = tuple(dict.fromkeys(str(label) for label in prompts["label"].to_list()))
    similarities = scorer.raw_similarities(item, labels)
    missing_labels = [label for label in labels if label not in similarities]
    if missing_labels:
        raise ValueError("raw similarity scorer omitted labels: " + ", ".join(missing_labels))
    raw_by_label: dict[str, float] = {}
    for label in labels:
        value = float(similarities[label])
        if not isfinite(value):
            raise ValueError(f"raw similarity is not finite for label: {label}")
        raw_by_label[label] = value
    scores: list[RankCandidateScore] = []
    for node_id in node_ids:
        candidate = candidate_by_id[node_id]
        prompt_rows = prompt_rows_by_node.get(node_id, [])
        if not prompt_rows:
            raise ValueError(f"classification-v3 node has no {prompt_stage} prompt: {node_id}")
        prompt_ranks = {str(row.get("rank") or "") for row in prompt_rows}
        if prompt_ranks != {candidate["rank"]}:
            raise ValueError(f"classification-v3 prompt rank mismatch for node: {node_id}")
        node_labels = tuple(dict.fromkeys(str(row["label"]) for row in prompt_rows))
        label_values = [(label, raw_by_label[label]) for label in node_labels]
        best_label, best_similarity = min(label_values, key=lambda pair: (-pair[1], pair[0]))
        scores.append(
            RankCandidateScore(
                node_id=node_id,
                rank=candidate["rank"],
                scientific_name=candidate["scientific_name"],
                raw_similarity=sum(value for _label, value in label_values) / len(label_values),
                best_label=best_label,
                best_label_similarity=best_similarity,
                label_count=len(label_values),
            )
        )
    return tuple(
        sorted(
            scores,
            key=lambda score: (-score.raw_similarity, score.scientific_name, score.node_id),
        )
    )


def raw_similarity_margin(scores: tuple[RankCandidateScore, ...]) -> float | None:
    return scores[0].raw_similarity - scores[1].raw_similarity if len(scores) > 1 else None


def classify_path_cascade(
    *,
    item: dict[str, Any],
    scorer: PathCascadeScorer,
    taxonomy_store: PathTaxonomyStore,
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH,
    embedding_cache_fingerprint: str | None = None,
) -> PathCascadeResult:
    if rank_beam_width != DEFAULT_RANK_BEAM_WIDTH:
        raise ValueError(f"rank_beam_width is fixed at {DEFAULT_RANK_BEAM_WIDTH}")
    active_paths = taxonomy_store.enabled_paths()
    if active_paths.is_empty():
        raise PathCascadeClassificationError(
            code="no_active_paths",
            rank="FAMILY",
            candidate_count=0,
            active_path_count=0,
            message="taxonomy store has no enabled classification paths",
        )
    rank_steps: list[RankStepResult] = []
    fully_skipped_ranks: list[str] = []
    for rank in INTERMEDIATE_CLASSIFICATION_RANKS:
        active_before = active_paths.height
        parent_node_ids = _parent_node_ids(active_paths, rank)
        candidates = taxonomy_store.candidate_nodes_in_paths(active_paths, rank)
        reviewed_skips = taxonomy_store.reviewed_skip_paths(active_paths, rank)
        if candidates.is_empty():
            if rank in OPTIONAL_CLASSIFICATION_RANKS:
                active_hashes = set(active_paths["hierarchy_hash"].to_list())
                skip_hashes = set(reviewed_skips["hierarchy_hash"].to_list())
                if active_hashes and active_hashes == skip_hashes:
                    fully_skipped_ranks.append(rank)
                    rank_steps.append(
                        RankStepResult(
                            rank=rank,
                            candidate_count=0,
                            retained_count=0,
                            active_path_count_before=active_before,
                            active_path_count_after=active_before,
                            top_candidates=(),
                            top1_margin=None,
                            parent_node_ids=parent_node_ids,
                            skipped=True,
                            skip_reason="all_active_paths_reviewed_rank_skip",
                        )
                    )
                    continue
                code = "incomplete_optional_rank_coverage"
                message = "optional rank has neither asserted nodes nor reviewed skips for every active path"
            else:
                code = "no_rank_candidates"
                message = "mandatory rank has no candidates in the active path union"
            raise PathCascadeClassificationError(
                code=code,
                rank=rank,
                candidate_count=0,
                active_path_count=active_before,
                message=message,
            )
        scores = score_rank_candidates(
            item=item,
            scorer=scorer,
            taxonomy_store=taxonomy_store,
            candidates=candidates,
            prompt_stage=RANK_SCREEN_PROMPT_STAGE,
        )
        if not scores:
            raise PathCascadeClassificationError(
                code="unscorable_rank_candidates",
                rank=rank,
                candidate_count=candidates.height,
                active_path_count=active_before,
                message="rank candidates produced no raw similarity scores",
            )
        selected = scores[:rank_beam_width]
        active_paths = taxonomy_store.filter_paths_by_rank_nodes(
            active_paths,
            rank,
            tuple(score.node_id for score in selected),
            carry_reviewed_skip_paths=True,
        )
        if active_paths.is_empty():
            raise PathCascadeClassificationError(
                code="no_active_paths_after_pruning",
                rank=rank,
                candidate_count=len(scores),
                active_path_count=0,
                message="global rank pruning removed every active path",
            )
        rank_steps.append(
            RankStepResult(
                rank=rank,
                candidate_count=len(scores),
                retained_count=len(selected),
                active_path_count_before=active_before,
                active_path_count_after=active_paths.height,
                top_candidates=selected,
                top1_margin=raw_similarity_margin(scores),
                parent_node_ids=parent_node_ids,
                skipped=False,
                skip_reason=None,
            )
        )
    return PathCascadeResult(
        classification_version=taxonomy_store.classification_version,
        prompt_version=taxonomy_store.prompt_version,
        taxonomy_fingerprint=taxonomy_store.hierarchy_fingerprint,
        classification_fingerprint=taxonomy_store.classification_fingerprint,
        embedding_cache_fingerprint=embedding_cache_fingerprint,
        beam_strategy=GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
        rank_beam_width=rank_beam_width,
        rank_steps=tuple(rank_steps),
        species_top20=(),
        species_top5=(),
        species_top3=(),
        species_top1=None,
        final_winning_path=(),
        skipped_ranks=tuple(fully_skipped_ranks),
    )


def _parent_node_ids(active_paths: pl.DataFrame, rank: str) -> tuple[str, ...]:
    rank_index = CLASSIFICATION_RANKS.index(rank)
    if rank_index == 0:
        return ()
    parent_ids: set[str] = set()
    for path in active_paths.iter_rows(named=True):
        for parent_rank in reversed(CLASSIFICATION_RANKS[:rank_index]):
            node_id = str(path.get(f"{parent_rank.casefold()}_node_id") or "")
            if node_id:
                parent_ids.add(node_id)
                break
    return tuple(sorted(parent_ids))


__all__ = [
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "INTERMEDIATE_CLASSIFICATION_RANKS",
    "PathCascadeClassificationError",
    "PathCascadeResult",
    "PathCascadeScorer",
    "RankCandidateScore",
    "RankStepResult",
    "classify_path_cascade",
    "raw_similarity_margin",
    "score_rank_candidates",
]
