from __future__ import annotations

from biominer.registry.classification_v3 import CLASSIFICATION_RANKS


CASCADE_CONTRACT_VERSION = "butterfly-global-rank-cascade-v1"
GLOBAL_RANK_TOP_K_BEAM_STRATEGY = "global_rank_top_k"
DEFAULT_RANK_BEAM_WIDTH = 3
DEFAULT_SPECIES_FIRST_PASS_TOP_K = 20
DEFAULT_SPECIES_RERANK_TOP_K = 5
DEFAULT_SPECIES_REPORT_TOP_K = 3
CASCADE_RANK_ORDER = CLASSIFICATION_RANKS


def validate_production_cascade_settings(
    *,
    beam_strategy: str,
    rank_beam_width: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
    species_report_top_k: int,
) -> tuple[str, int, int, int, int]:
    strategy = str(beam_strategy or "").strip()
    width = int(rank_beam_width)
    first_pass = int(species_first_pass_top_k)
    rerank = int(species_rerank_top_k)
    report = int(species_report_top_k)
    if strategy != GLOBAL_RANK_TOP_K_BEAM_STRATEGY:
        raise ValueError(
            "beam_strategy is fixed at " + GLOBAL_RANK_TOP_K_BEAM_STRATEGY
        )
    if width != DEFAULT_RANK_BEAM_WIDTH:
        raise ValueError(f"rank_beam_width is fixed at {DEFAULT_RANK_BEAM_WIDTH}")
    required_species_widths = (
        ("species_first_pass_top_k", first_pass, DEFAULT_SPECIES_FIRST_PASS_TOP_K),
        ("species_rerank_top_k", rerank, DEFAULT_SPECIES_RERANK_TOP_K),
        ("species_report_top_k", report, DEFAULT_SPECIES_REPORT_TOP_K),
    )
    for name, value, required in required_species_widths:
        if value != required:
            raise ValueError(f"{name} is fixed at {required}")
    return strategy, width, first_pass, rerank, report


__all__ = [
    "CASCADE_CONTRACT_VERSION",
    "CASCADE_RANK_ORDER",
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "DEFAULT_SPECIES_REPORT_TOP_K",
    "DEFAULT_SPECIES_RERANK_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "validate_production_cascade_settings",
]
