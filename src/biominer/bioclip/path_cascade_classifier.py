from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


__all__ = [
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "INTERMEDIATE_CLASSIFICATION_RANKS",
    "PathCascadeClassificationError",
    "PathCascadeResult",
    "RankCandidateScore",
    "RankStepResult",
]
