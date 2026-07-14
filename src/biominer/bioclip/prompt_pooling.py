"""Versioned BioCLIP prompt pooling with complete per-prompt diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import re

import numpy as np

from biominer.bioclip.prompt_templates import (
    EXPLICIT_GEOGRAPHY_PROMPT_ABLATION,
    TAXONOMIC_PROMPT_VERSION,
    PromptVariant,
    TaxonomicPromptEnsemble,
    taxonomic_prompt_ensemble_payload,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


PROMPT_POOLING_SCHEMA_VERSION = "prompt-ensemble-pooling-result-v1.2.0"
PROMPT_POOLING_VERSION = "bioclip-prompt-pooling-v1.1.0"
PROMPT_SUBSET_SCHEMA_VERSION = "prompt-subset-policy-v1.1.0"
PROMPT_SUBSET_VERSION = "stage-domain-prompt-subsets-v1.1.0"
PROMPT_WEIGHT_SCHEMA_VERSION = "learned-prompt-weights-v1.0.0"
PROMPT_WEIGHT_VERSION = "validation-learned-prompt-weights-v1.0.0"

NORMALIZED_MEAN_TEXT_EMBEDDING = "normalized_mean_text_embedding"
MAX_PROMPT_SIMILARITY = "maximum_prompt_similarity"
MEAN_BEST_TWO = "mean_best_two_prompt_similarities"
LEARNED_PROMPT_WEIGHTS = "learned_prompt_weights"
PROMPT_POOLING_STRATEGIES = frozenset(
    {
        NORMALIZED_MEAN_TEXT_EMBEDDING,
        MAX_PROMPT_SIMILARITY,
        MEAN_BEST_TWO,
        LEARNED_PROMPT_WEIGHTS,
    }
)
PROMPT_SCORE_KIND = "cosine_similarity"

_ROUTE_VISUAL_DOMAIN = {
    "adult_field": "field",
    "egg": "field",
    "larval": "field",
    "pupal": "field",
    "pinned_specimen": "specimen",
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PromptSubsetPolicy:
    schema_version: str
    subset_version: str
    subset_id: str
    prompt_version: str
    ensemble_fingerprint: str
    route: str
    life_stage: str
    visual_domain: str
    prompt_kinds: tuple[str, ...]
    geography_prompt_ablation_enabled: bool
    geography_ablation_fingerprint: str | None
    subset_fingerprint: str


@dataclass(frozen=True, slots=True)
class LearnedPromptWeights:
    schema_version: str
    weight_version: str
    prompt_version: str
    ensemble_fingerprint: str
    subset_fingerprint: str
    model_fingerprint: str
    split_fingerprint: str
    fit_partition: str
    weights: tuple[tuple[str, float], ...]
    weight_artifact_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptScoreDiagnostic:
    variant_fingerprint: str
    label: str
    prompt_kind: str
    template_id: str
    evidence_kind: str
    evidence_id: str | None
    trust_tier: str | None
    language: str | None
    geography_bearing: bool
    raw_similarity: float
    in_subset: bool
    contributed: bool
    pooling_weight: float
    selection_reason: str


@dataclass(frozen=True, slots=True)
class PromptPoolingResult:
    schema_version: str
    pooling_version: str
    strategy: str
    score_kind: str
    pooling_space: str
    prompt_version: str
    accepted_taxon_key: str
    scientific_name: str
    ensemble_fingerprint: str
    subset_fingerprint: str
    subset_id: str
    route: str
    life_stage: str
    visual_domain: str
    geography_prompt_ablation_enabled: bool
    geography_ablation_fingerprint: str | None
    model_fingerprint: str
    image_embedding_fingerprint: str
    text_embedding_set_fingerprint: str
    weight_artifact_fingerprint: str | None
    embedding_dimension: int
    pooled_score: float
    pooled_text_embedding: tuple[float, ...] | None
    best_prompt_variant_fingerprint: str
    best_prompt_label: str
    selected_prompt_count: int
    selected_geography_prompt_count: int
    contributing_prompt_count: int
    prompt_scores: tuple[PromptScoreDiagnostic, ...]
    result_fingerprint: str


def build_prompt_subset_policy(
    ensemble: TaxonomicPromptEnsemble,
    *,
    subset_id: str,
    visual_domain: str,
    prompt_kinds: Sequence[str],
    enable_geography_prompt_ablation: bool = False,
) -> PromptSubsetPolicy:
    """Create a canonical prompt-kind subset bound to one route/stage ensemble."""

    taxonomic_prompt_ensemble_payload(ensemble)
    _require_boolean(
        enable_geography_prompt_ablation,
        field="enable_geography_prompt_ablation",
    )
    identifier = _canonical_text(subset_id, field="subset_id")
    domain = _canonical_text(visual_domain, field="visual_domain").casefold()
    expected_domain = _ROUTE_VISUAL_DOMAIN[ensemble.route]
    if domain != expected_domain:
        raise ValueError(
            f"visual domain {domain} is incompatible with route {ensemble.route}"
        )
    requested = {_canonical_text(kind, field="prompt_kind") for kind in prompt_kinds}
    if not requested:
        raise ValueError("prompt subset must include at least one prompt kind")
    available = {variant.prompt_kind for variant in ensemble.variants}
    unknown = requested - available
    if unknown:
        raise ValueError(f"prompt subset contains unavailable kinds: {sorted(unknown)}")
    ordered_kinds = tuple(
        dict.fromkeys(
            variant.prompt_kind
            for variant in ensemble.variants
            if variant.prompt_kind in requested
        )
    )
    selected_geography = any(
        variant.geography_bearing and variant.prompt_kind in requested
        for variant in ensemble.variants
    )
    if selected_geography and not enable_geography_prompt_ablation:
        raise ValueError(
            "geography-bearing prompt kinds require explicit ablation enablement"
        )
    if enable_geography_prompt_ablation and not selected_geography:
        raise ValueError(
            "geography prompt ablation enablement requires a geographic prompt kind"
        )
    if enable_geography_prompt_ablation and (
        ensemble.geography_mode != EXPLICIT_GEOGRAPHY_PROMPT_ABLATION
        or ensemble.geography_ablation_fingerprint is None
    ):
        raise ValueError("prompt ensemble has no compatible geography ablation")
    semantics = {
        "schema_version": PROMPT_SUBSET_SCHEMA_VERSION,
        "subset_version": PROMPT_SUBSET_VERSION,
        "subset_id": identifier,
        "prompt_version": ensemble.prompt_version,
        "ensemble_fingerprint": ensemble.ensemble_fingerprint,
        "route": ensemble.route,
        "life_stage": ensemble.life_stage,
        "visual_domain": domain,
        "prompt_kinds": list(ordered_kinds),
        "geography_prompt_ablation_enabled": enable_geography_prompt_ablation,
        "geography_ablation_fingerprint": (ensemble.geography_ablation_fingerprint),
    }
    return PromptSubsetPolicy(
        schema_version=PROMPT_SUBSET_SCHEMA_VERSION,
        subset_version=PROMPT_SUBSET_VERSION,
        subset_id=identifier,
        prompt_version=ensemble.prompt_version,
        ensemble_fingerprint=ensemble.ensemble_fingerprint,
        route=ensemble.route,
        life_stage=ensemble.life_stage,
        visual_domain=domain,
        prompt_kinds=ordered_kinds,
        geography_prompt_ablation_enabled=enable_geography_prompt_ablation,
        geography_ablation_fingerprint=ensemble.geography_ablation_fingerprint,
        subset_fingerprint=canonical_semantic_fingerprint(semantics),
    )


def build_stage_domain_prompt_subset(
    ensemble: TaxonomicPromptEnsemble,
    *,
    include_vernacular_names: bool = False,
    include_reviewed_aliases: bool = False,
) -> PromptSubsetPolicy:
    """Select route/stage taxonomy prompts without blindly taking optional names."""

    _require_boolean(include_vernacular_names, field="include_vernacular_names")
    _require_boolean(include_reviewed_aliases, field="include_reviewed_aliases")
    taxonomic_prompt_ensemble_payload(ensemble)
    kinds: list[str] = []
    for variant in ensemble.variants:
        include = (
            not variant.geography_bearing
            and variant.evidence_kind == "accepted_taxonomy"
        )
        include = include or (
            include_vernacular_names and variant.evidence_kind == "vernacular_name"
        )
        include = include or (
            include_reviewed_aliases
            and variant.evidence_kind == "reviewed_prompt_alias"
        )
        if include and variant.prompt_kind not in kinds:
            kinds.append(variant.prompt_kind)
    suffixes = ["core"]
    if include_vernacular_names:
        suffixes.append("vernacular")
    if include_reviewed_aliases:
        suffixes.append("aliases")
    return build_prompt_subset_policy(
        ensemble,
        subset_id=(
            f"{_ROUTE_VISUAL_DOMAIN[ensemble.route]}-{ensemble.life_stage}-"
            f"{'+'.join(suffixes)}"
        ),
        visual_domain=_ROUTE_VISUAL_DOMAIN[ensemble.route],
        prompt_kinds=kinds,
        enable_geography_prompt_ablation=False,
    )


def build_learned_prompt_weights(
    *,
    ensemble: TaxonomicPromptEnsemble,
    subset: PromptSubsetPolicy,
    weights: Mapping[str, float],
    model_fingerprint: str,
    split_fingerprint: str,
    fit_partition: str,
) -> LearnedPromptWeights:
    """Bind externally learned nonnegative weights to model-selection evidence."""

    _validate_subset(ensemble, subset)
    model = _sha256(model_fingerprint, field="model_fingerprint")
    split = _sha256(split_fingerprint, field="split_fingerprint")
    partition = _canonical_text(fit_partition, field="fit_partition")
    if partition != "model_selection":
        raise ValueError("learned prompt weights must use model_selection data only")
    selected = _selected_variants(ensemble, subset)
    expected_keys = {variant.variant_fingerprint for variant in selected}
    if set(weights) != expected_keys:
        raise ValueError("learned prompt weight key set does not match prompt subset")
    raw_weights = {
        key: _nonnegative_finite(value, field=f"weight[{key}]")
        for key, value in weights.items()
    }
    total = sum(raw_weights.values())
    if total <= 0.0:
        raise ValueError("learned prompt weights must have positive total weight")
    normalized = tuple(
        (variant.variant_fingerprint, raw_weights[variant.variant_fingerprint] / total)
        for variant in selected
    )
    semantics = {
        "schema_version": PROMPT_WEIGHT_SCHEMA_VERSION,
        "weight_version": PROMPT_WEIGHT_VERSION,
        "prompt_version": ensemble.prompt_version,
        "ensemble_fingerprint": ensemble.ensemble_fingerprint,
        "subset_fingerprint": subset.subset_fingerprint,
        "model_fingerprint": model,
        "split_fingerprint": split,
        "fit_partition": partition,
        "weights": [[fingerprint, weight] for fingerprint, weight in normalized],
    }
    return LearnedPromptWeights(
        schema_version=PROMPT_WEIGHT_SCHEMA_VERSION,
        weight_version=PROMPT_WEIGHT_VERSION,
        prompt_version=ensemble.prompt_version,
        ensemble_fingerprint=ensemble.ensemble_fingerprint,
        subset_fingerprint=subset.subset_fingerprint,
        model_fingerprint=model,
        split_fingerprint=split,
        fit_partition=partition,
        weights=normalized,
        weight_artifact_fingerprint=canonical_semantic_fingerprint(semantics),
    )


def pool_prompt_ensemble(
    *,
    ensemble: TaxonomicPromptEnsemble,
    image_embedding: Sequence[float],
    text_embeddings: Mapping[str, Sequence[float]],
    model_fingerprint: str,
    subset: PromptSubsetPolicy,
    strategy: str,
    learned_weights: LearnedPromptWeights | None = None,
) -> PromptPoolingResult:
    """Apply one explicit pooling strategy and retain every prompt similarity."""

    taxonomic_prompt_ensemble_payload(ensemble)
    _validate_subset(ensemble, subset)
    strategy_value = _canonical_text(strategy, field="strategy")
    if strategy_value not in PROMPT_POOLING_STRATEGIES:
        raise ValueError(f"unsupported prompt pooling strategy: {strategy_value}")
    if strategy_value != LEARNED_PROMPT_WEIGHTS and learned_weights is not None:
        raise ValueError("learned weights are valid only for learned prompt pooling")
    model = _sha256(model_fingerprint, field="model_fingerprint")
    variants = ensemble.variants
    expected_embedding_keys = {variant.variant_fingerprint for variant in variants}
    if set(text_embeddings) != expected_embedding_keys:
        raise ValueError("text embedding key set does not match prompt ensemble")
    image = _normalized_vector(image_embedding, field="image_embedding")
    text_rows = tuple(
        _normalized_vector(
            text_embeddings[variant.variant_fingerprint],
            field=f"text_embeddings[{variant.variant_fingerprint}]",
        )
        for variant in variants
    )
    dimension = int(image.shape[0])
    if any(int(row.shape[0]) != dimension for row in text_rows):
        raise ValueError("image and text embedding dimensions do not match")
    matrix = np.stack(text_rows, axis=0)
    similarities = matrix @ image
    subset_indices = tuple(
        index
        for index, variant in enumerate(variants)
        if variant.prompt_kind in subset.prompt_kinds
    )
    if not subset_indices:
        raise ValueError("prompt subset selects no ensemble variants")
    ranked_indices = tuple(
        sorted(
            subset_indices,
            key=lambda index: (
                -float(similarities[index]),
                variants[index].variant_fingerprint,
            ),
        )
    )
    best_index = ranked_indices[0]
    contribution_weights: dict[int, float]
    pooled_embedding: np.ndarray | None = None
    weight_fingerprint: str | None = None
    if strategy_value == NORMALIZED_MEAN_TEXT_EMBEDDING:
        contribution_weights = {
            index: 1.0 / len(subset_indices) for index in subset_indices
        }
        pooled_embedding = _normalized_pool(
            matrix,
            contribution_weights,
            field="normalized mean text embedding",
        )
        pooled_score = float(pooled_embedding @ image)
        pooling_space = "text_embedding"
    elif strategy_value == MAX_PROMPT_SIMILARITY:
        contribution_weights = {best_index: 1.0}
        pooled_score = float(similarities[best_index])
        pooling_space = "prompt_similarity"
    elif strategy_value == MEAN_BEST_TWO:
        if len(ranked_indices) < 2:
            raise ValueError("mean-of-best-two pooling requires at least two prompts")
        contribution_weights = {index: 0.5 for index in ranked_indices[:2]}
        pooled_score = sum(
            float(similarities[index]) * weight
            for index, weight in contribution_weights.items()
        )
        pooling_space = "prompt_similarity"
    else:
        if learned_weights is None:
            raise ValueError("learned prompt pooling requires learned weights")
        _validate_learned_weights(
            learned_weights,
            ensemble=ensemble,
            subset=subset,
            model_fingerprint=model,
        )
        index_by_fingerprint = {
            variant.variant_fingerprint: index for index, variant in enumerate(variants)
        }
        contribution_weights = {
            index_by_fingerprint[fingerprint]: weight
            for fingerprint, weight in learned_weights.weights
        }
        pooled_embedding = _normalized_pool(
            matrix,
            contribution_weights,
            field="learned weighted text embedding",
        )
        pooled_score = float(pooled_embedding @ image)
        pooling_space = "text_embedding"
        weight_fingerprint = learned_weights.weight_artifact_fingerprint

    pooled_score = _bounded_cosine(pooled_score, field="pooled_score")
    diagnostics = tuple(
        _prompt_diagnostic(
            variant,
            raw_similarity=_bounded_cosine(
                float(similarities[index]),
                field="raw_similarity",
            ),
            in_subset=index in subset_indices,
            pooling_weight=contribution_weights.get(index, 0.0),
        )
        for index, variant in enumerate(variants)
    )
    image_fingerprint = canonical_semantic_fingerprint(
        {
            "model_fingerprint": model,
            "normalized_image_embedding": image.tolist(),
        }
    )
    text_fingerprint = canonical_semantic_fingerprint(
        {
            "model_fingerprint": model,
            "prompt_version": ensemble.prompt_version,
            "normalized_text_embeddings": [
                {
                    "variant_fingerprint": variant.variant_fingerprint,
                    "embedding": row.tolist(),
                }
                for variant, row in zip(variants, text_rows, strict=True)
            ],
        }
    )
    values: dict[str, object] = {
        "schema_version": PROMPT_POOLING_SCHEMA_VERSION,
        "pooling_version": PROMPT_POOLING_VERSION,
        "strategy": strategy_value,
        "score_kind": PROMPT_SCORE_KIND,
        "pooling_space": pooling_space,
        "prompt_version": ensemble.prompt_version,
        "accepted_taxon_key": ensemble.accepted_taxon_key,
        "scientific_name": ensemble.scientific_name,
        "ensemble_fingerprint": ensemble.ensemble_fingerprint,
        "subset_fingerprint": subset.subset_fingerprint,
        "subset_id": subset.subset_id,
        "route": ensemble.route,
        "life_stage": ensemble.life_stage,
        "visual_domain": subset.visual_domain,
        "geography_prompt_ablation_enabled": (subset.geography_prompt_ablation_enabled),
        "geography_ablation_fingerprint": subset.geography_ablation_fingerprint,
        "model_fingerprint": model,
        "image_embedding_fingerprint": image_fingerprint,
        "text_embedding_set_fingerprint": text_fingerprint,
        "weight_artifact_fingerprint": weight_fingerprint,
        "embedding_dimension": dimension,
        "pooled_score": pooled_score,
        "pooled_text_embedding": (
            tuple(float(value) for value in pooled_embedding.tolist())
            if pooled_embedding is not None
            else None
        ),
        "best_prompt_variant_fingerprint": variants[best_index].variant_fingerprint,
        "best_prompt_label": variants[best_index].label,
        "selected_prompt_count": len(subset_indices),
        "selected_geography_prompt_count": sum(
            variants[index].geography_bearing for index in subset_indices
        ),
        "contributing_prompt_count": sum(item.contributed for item in diagnostics),
        "prompt_scores": diagnostics,
    }
    fingerprint = canonical_semantic_fingerprint(_result_semantic_payload(values))
    return PromptPoolingResult(**values, result_fingerprint=fingerprint)


def prompt_pooling_result_payload(result: PromptPoolingResult) -> dict[str, object]:
    """Validate and return a JSON-like result with all prompt diagnostics."""

    if not isinstance(result, PromptPoolingResult):
        raise TypeError("result must be a PromptPoolingResult")
    values = {
        field: getattr(result, field)
        for field in PromptPoolingResult.__dataclass_fields__
        if field != "result_fingerprint"
    }
    semantics = _result_semantic_payload(values)
    fingerprint = _sha256(result.result_fingerprint, field="result_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt pooling result fingerprint is inconsistent")
    return {**semantics, "result_fingerprint": fingerprint}


def _validate_subset(
    ensemble: TaxonomicPromptEnsemble,
    subset: PromptSubsetPolicy,
) -> None:
    taxonomic_prompt_ensemble_payload(ensemble)
    if not isinstance(subset, PromptSubsetPolicy):
        raise TypeError("subset must be a PromptSubsetPolicy")
    if (
        subset.schema_version != PROMPT_SUBSET_SCHEMA_VERSION
        or subset.subset_version != PROMPT_SUBSET_VERSION
        or subset.prompt_version != TAXONOMIC_PROMPT_VERSION
    ):
        raise ValueError("prompt subset version is incompatible")
    if (
        subset.ensemble_fingerprint != ensemble.ensemble_fingerprint
        or subset.route != ensemble.route
        or subset.life_stage != ensemble.life_stage
        or subset.visual_domain != _ROUTE_VISUAL_DOMAIN[ensemble.route]
        or subset.geography_ablation_fingerprint
        != ensemble.geography_ablation_fingerprint
    ):
        raise ValueError("prompt subset identity does not match ensemble")
    available = {variant.prompt_kind for variant in ensemble.variants}
    if not subset.prompt_kinds or not set(subset.prompt_kinds) <= available:
        raise ValueError("prompt subset kinds are incompatible with ensemble")
    _require_boolean(
        subset.geography_prompt_ablation_enabled,
        field="geography_prompt_ablation_enabled",
    )
    selected_geography = any(
        variant.geography_bearing and variant.prompt_kind in subset.prompt_kinds
        for variant in ensemble.variants
    )
    if selected_geography != subset.geography_prompt_ablation_enabled:
        raise ValueError("prompt subset geography enablement is inconsistent")
    if subset.geography_prompt_ablation_enabled and (
        ensemble.geography_mode != EXPLICIT_GEOGRAPHY_PROMPT_ABLATION
        or subset.geography_ablation_fingerprint is None
    ):
        raise ValueError("prompt subset geography ablation is incompatible")
    semantics = {
        "schema_version": subset.schema_version,
        "subset_version": subset.subset_version,
        "subset_id": subset.subset_id,
        "prompt_version": subset.prompt_version,
        "ensemble_fingerprint": subset.ensemble_fingerprint,
        "route": subset.route,
        "life_stage": subset.life_stage,
        "visual_domain": subset.visual_domain,
        "prompt_kinds": list(subset.prompt_kinds),
        "geography_prompt_ablation_enabled": (subset.geography_prompt_ablation_enabled),
        "geography_ablation_fingerprint": subset.geography_ablation_fingerprint,
    }
    fingerprint = _sha256(subset.subset_fingerprint, field="subset_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt subset fingerprint is inconsistent")


def _validate_learned_weights(
    value: LearnedPromptWeights,
    *,
    ensemble: TaxonomicPromptEnsemble,
    subset: PromptSubsetPolicy,
    model_fingerprint: str,
) -> None:
    if not isinstance(value, LearnedPromptWeights):
        raise TypeError("learned_weights must be a LearnedPromptWeights artifact")
    if (
        value.schema_version != PROMPT_WEIGHT_SCHEMA_VERSION
        or value.weight_version != PROMPT_WEIGHT_VERSION
        or value.prompt_version != ensemble.prompt_version
        or value.ensemble_fingerprint != ensemble.ensemble_fingerprint
        or value.subset_fingerprint != subset.subset_fingerprint
        or value.model_fingerprint != model_fingerprint
        or value.fit_partition != "model_selection"
    ):
        raise ValueError("learned prompt weight identity is incompatible")
    _sha256(value.split_fingerprint, field="split_fingerprint")
    selected = _selected_variants(ensemble, subset)
    expected_keys = tuple(variant.variant_fingerprint for variant in selected)
    if tuple(fingerprint for fingerprint, _ in value.weights) != expected_keys:
        raise ValueError("learned prompt weight key order is incompatible")
    normalized_weights = tuple(
        (fingerprint, _nonnegative_finite(weight, field="learned weight"))
        for fingerprint, weight in value.weights
    )
    if abs(sum(weight for _, weight in normalized_weights) - 1.0) > 1e-12:
        raise ValueError("learned prompt weights must sum to one")
    semantics = {
        "schema_version": value.schema_version,
        "weight_version": value.weight_version,
        "prompt_version": value.prompt_version,
        "ensemble_fingerprint": value.ensemble_fingerprint,
        "subset_fingerprint": value.subset_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "split_fingerprint": value.split_fingerprint,
        "fit_partition": value.fit_partition,
        "weights": [[fingerprint, weight] for fingerprint, weight in value.weights],
    }
    fingerprint = _sha256(
        value.weight_artifact_fingerprint,
        field="weight_artifact_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("learned prompt weight fingerprint is inconsistent")


def _selected_variants(
    ensemble: TaxonomicPromptEnsemble,
    subset: PromptSubsetPolicy,
) -> tuple[PromptVariant, ...]:
    return tuple(
        variant
        for variant in ensemble.variants
        if variant.prompt_kind in subset.prompt_kinds
    )


def _normalized_vector(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric vector") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional vector")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{field} must contain finite values")
    norm = float(np.linalg.vector_norm(result))
    if not isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{field} must have nonzero norm")
    return np.asarray(result / norm, dtype=np.float64)


def _normalized_pool(
    matrix: np.ndarray,
    weights: Mapping[int, float],
    *,
    field: str,
) -> np.ndarray:
    pooled = np.zeros(matrix.shape[1], dtype=np.float64)
    for index, weight in weights.items():
        pooled += matrix[index] * weight
    norm = float(np.linalg.vector_norm(pooled))
    if not isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{field} has zero direction")
    return pooled / norm


def _prompt_diagnostic(
    variant: PromptVariant,
    *,
    raw_similarity: float,
    in_subset: bool,
    pooling_weight: float,
) -> PromptScoreDiagnostic:
    contributed = pooling_weight > 0.0
    if not in_subset:
        reason = (
            "geography_ablation_disabled"
            if variant.geography_bearing
            else "outside_prompt_subset"
        )
    elif contributed:
        reason = "contributed"
    else:
        reason = "not_selected_by_strategy"
    return PromptScoreDiagnostic(
        variant_fingerprint=variant.variant_fingerprint,
        label=variant.label,
        prompt_kind=variant.prompt_kind,
        template_id=variant.template_id,
        evidence_kind=variant.evidence_kind,
        evidence_id=variant.evidence_id,
        trust_tier=variant.trust_tier,
        language=variant.language,
        geography_bearing=variant.geography_bearing,
        raw_similarity=raw_similarity,
        in_subset=in_subset,
        contributed=contributed,
        pooling_weight=pooling_weight,
        selection_reason=reason,
    )


def _result_semantic_payload(values: Mapping[str, object]) -> dict[str, object]:
    prompt_scores = values["prompt_scores"]
    if not isinstance(prompt_scores, tuple) or not all(
        isinstance(item, PromptScoreDiagnostic) for item in prompt_scores
    ):
        raise ValueError("prompt_scores must be canonical diagnostics")
    pooled_embedding = values["pooled_text_embedding"]
    return {
        "schema_version": values["schema_version"],
        "pooling_version": values["pooling_version"],
        "strategy": values["strategy"],
        "score_kind": values["score_kind"],
        "pooling_space": values["pooling_space"],
        "prompt_version": values["prompt_version"],
        "accepted_taxon_key": values["accepted_taxon_key"],
        "scientific_name": values["scientific_name"],
        "ensemble_fingerprint": values["ensemble_fingerprint"],
        "subset_fingerprint": values["subset_fingerprint"],
        "subset_id": values["subset_id"],
        "route": values["route"],
        "life_stage": values["life_stage"],
        "visual_domain": values["visual_domain"],
        "geography_prompt_ablation_enabled": values[
            "geography_prompt_ablation_enabled"
        ],
        "geography_ablation_fingerprint": values["geography_ablation_fingerprint"],
        "model_fingerprint": values["model_fingerprint"],
        "image_embedding_fingerprint": values["image_embedding_fingerprint"],
        "text_embedding_set_fingerprint": values["text_embedding_set_fingerprint"],
        "weight_artifact_fingerprint": values["weight_artifact_fingerprint"],
        "embedding_dimension": values["embedding_dimension"],
        "pooled_score": values["pooled_score"],
        "pooled_text_embedding": (
            list(pooled_embedding) if pooled_embedding is not None else None
        ),
        "best_prompt_variant_fingerprint": values["best_prompt_variant_fingerprint"],
        "best_prompt_label": values["best_prompt_label"],
        "selected_prompt_count": values["selected_prompt_count"],
        "selected_geography_prompt_count": values["selected_geography_prompt_count"],
        "contributing_prompt_count": values["contributing_prompt_count"],
        "prompt_scores": [
            {
                "variant_fingerprint": item.variant_fingerprint,
                "label": item.label,
                "prompt_kind": item.prompt_kind,
                "template_id": item.template_id,
                "evidence_kind": item.evidence_kind,
                "evidence_id": item.evidence_id,
                "trust_tier": item.trust_tier,
                "language": item.language,
                "geography_bearing": item.geography_bearing,
                "raw_similarity": item.raw_similarity,
                "in_subset": item.in_subset,
                "contributed": item.contributed,
                "pooling_weight": item.pooling_weight,
                "selection_reason": item.selection_reason,
            }
            for item in prompt_scores
        ],
    }


def _bounded_cosine(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < -1.0 - 1e-12 or result > 1.0 + 1e-12:
        raise ValueError(f"{field} must be a cosine similarity")
    return min(1.0, max(-1.0, result))


def _nonnegative_finite(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _canonical_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _require_boolean(value: object, *, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")


__all__ = [
    "LEARNED_PROMPT_WEIGHTS",
    "MAX_PROMPT_SIMILARITY",
    "MEAN_BEST_TWO",
    "NORMALIZED_MEAN_TEXT_EMBEDDING",
    "PROMPT_POOLING_SCHEMA_VERSION",
    "PROMPT_POOLING_STRATEGIES",
    "PROMPT_POOLING_VERSION",
    "PROMPT_SCORE_KIND",
    "PROMPT_SUBSET_SCHEMA_VERSION",
    "PROMPT_SUBSET_VERSION",
    "PROMPT_WEIGHT_SCHEMA_VERSION",
    "PROMPT_WEIGHT_VERSION",
    "LearnedPromptWeights",
    "PromptPoolingResult",
    "PromptScoreDiagnostic",
    "PromptSubsetPolicy",
    "build_learned_prompt_weights",
    "build_prompt_subset_policy",
    "build_stage_domain_prompt_subset",
    "pool_prompt_ensemble",
    "prompt_pooling_result_payload",
]
