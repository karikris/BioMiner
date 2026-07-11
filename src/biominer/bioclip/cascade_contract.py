from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_VERSION,
    SPECIES_RERANK_PROMPT_STAGE,
)


CASCADE_CONTRACT_VERSION = "butterfly-global-rank-cascade-v1"
CASCADE_WORK_IDENTITY_VERSION = "butterfly-cascade-work-identity-v1"
GLOBAL_RANK_TOP_K_BEAM_STRATEGY = "global_rank_top_k"
DEFAULT_RANK_BEAM_WIDTH = 3
DEFAULT_SPECIES_FIRST_PASS_TOP_K = 20
DEFAULT_SPECIES_RERANK_TOP_K = 5
DEFAULT_SPECIES_REPORT_TOP_K = 3
CASCADE_RANK_ORDER = CLASSIFICATION_RANKS
_CASCADE_WORK_IDENTITY_FIELDS = (
    "contract_version",
    "beam_strategy",
    "rank_beam_width",
    "rank_order",
    "classification_version",
    "prompt_version",
    "taxonomy_fingerprint",
    "hierarchy_fingerprint",
    "embedding_cache_fingerprint",
    "species_first_pass_top_k",
    "species_rerank_top_k",
    "species_report_top_k",
    "species_rerank_prompt_version",
)


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


def species_rerank_prompt_version(prompt_version: str) -> str:
    version = str(prompt_version or "").strip()
    if not version:
        raise ValueError("prompt_version must be nonblank")
    return f"{version}:{SPECIES_RERANK_PROMPT_STAGE}"


def production_cascade_work_identity(
    *,
    classification_version: str,
    prompt_version: str,
    taxonomy_fingerprint: str,
    hierarchy_fingerprint: str,
    embedding_cache_fingerprint: str,
) -> dict[str, Any]:
    identity = {
        "contract_version": CASCADE_WORK_IDENTITY_VERSION,
        "beam_strategy": GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
        "rank_beam_width": DEFAULT_RANK_BEAM_WIDTH,
        "rank_order": list(CASCADE_RANK_ORDER),
        "classification_version": str(classification_version or "").strip(),
        "prompt_version": str(prompt_version or "").strip(),
        "taxonomy_fingerprint": str(taxonomy_fingerprint or "").strip(),
        "hierarchy_fingerprint": str(hierarchy_fingerprint or "").strip(),
        "embedding_cache_fingerprint": str(embedding_cache_fingerprint or "").strip(),
        "species_first_pass_top_k": DEFAULT_SPECIES_FIRST_PASS_TOP_K,
        "species_rerank_top_k": DEFAULT_SPECIES_RERANK_TOP_K,
        "species_report_top_k": DEFAULT_SPECIES_REPORT_TOP_K,
        "species_rerank_prompt_version": species_rerank_prompt_version(prompt_version),
    }
    return validate_cascade_work_identity(identity)


def validate_cascade_work_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(identity) - set(_CASCADE_WORK_IDENTITY_FIELDS))
    missing = sorted(set(_CASCADE_WORK_IDENTITY_FIELDS) - set(identity))
    if unknown or missing:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError("cascade work identity fields mismatch: " + "; ".join(details))
    normalized = dict(identity)
    if normalized["contract_version"] != CASCADE_WORK_IDENTITY_VERSION:
        raise ValueError("cascade work identity contract version mismatch")
    validate_production_cascade_settings(
        beam_strategy=str(normalized["beam_strategy"]),
        rank_beam_width=int(normalized["rank_beam_width"]),
        species_first_pass_top_k=int(normalized["species_first_pass_top_k"]),
        species_rerank_top_k=int(normalized["species_rerank_top_k"]),
        species_report_top_k=int(normalized["species_report_top_k"]),
    )
    rank_order = tuple(str(rank) for rank in normalized["rank_order"])
    if rank_order != CASCADE_RANK_ORDER:
        raise ValueError("cascade work identity rank order mismatch")
    normalized["rank_order"] = list(rank_order)
    if normalized["classification_version"] != CLASSIFICATION_V3_VERSION:
        raise ValueError("cascade work identity requires classification-v3")
    for field in (
        "prompt_version",
        "taxonomy_fingerprint",
        "hierarchy_fingerprint",
        "embedding_cache_fingerprint",
    ):
        value = str(normalized[field] or "").strip()
        if not value:
            raise ValueError(f"cascade work identity {field} must be nonblank")
        normalized[field] = value
    expected_rerank_version = species_rerank_prompt_version(normalized["prompt_version"])
    if normalized["species_rerank_prompt_version"] != expected_rerank_version:
        raise ValueError("cascade work identity species rerank prompt version mismatch")
    return {field: normalized[field] for field in _CASCADE_WORK_IDENTITY_FIELDS}


__all__ = [
    "CASCADE_CONTRACT_VERSION",
    "CASCADE_RANK_ORDER",
    "CASCADE_WORK_IDENTITY_VERSION",
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "DEFAULT_SPECIES_REPORT_TOP_K",
    "DEFAULT_SPECIES_RERANK_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "production_cascade_work_identity",
    "species_rerank_prompt_version",
    "validate_cascade_work_identity",
    "validate_production_cascade_settings",
]
