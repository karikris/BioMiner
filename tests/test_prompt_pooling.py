from __future__ import annotations

from dataclasses import replace
import math

import pytest

from biominer.bioclip.prompt_pooling import (
    LEARNED_PROMPT_WEIGHTS,
    MAX_PROMPT_SIMILARITY,
    MEAN_BEST_TWO,
    NORMALIZED_MEAN_TEXT_EMBEDDING,
    PROMPT_POOLING_SCHEMA_VERSION,
    build_learned_prompt_weights,
    build_prompt_subset_policy,
    build_stage_domain_prompt_subset,
    pool_prompt_ensemble,
    prompt_pooling_result_payload,
)
from biominer.bioclip.prompt_templates import build_taxonomic_prompt_ensemble
from test_nonmatch_scoring import _sha
from test_prompt_templates import _context, _vernacular


MODEL_FINGERPRINT = _sha("bioclip-text-model")
SPLIT_FINGERPRINT = _sha("prompt-validation-split")


def _ensemble():
    return build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        vernacular_names=(_vernacular("lime butterfly"),),
    )


def _subset(ensemble, *prompt_kinds: str):
    return build_prompt_subset_policy(
        ensemble,
        subset_id="test-subset",
        visual_domain="field",
        prompt_kinds=prompt_kinds,
    )


def _embedding_map(ensemble, by_kind=None):
    values = by_kind or {}
    return {
        variant.variant_fingerprint: values.get(variant.prompt_kind, (0.0, 0.0, 1.0))
        for variant in ensemble.variants
    }


def test_normalized_mean_text_embedding_is_not_mean_prompt_similarity() -> None:
    ensemble = _ensemble()
    subset = _subset(
        ensemble,
        "accepted_scientific_name",
        "species_description",
    )
    embeddings = _embedding_map(
        ensemble,
        {
            "accepted_scientific_name": (1.0, 0.0, 0.0),
            "species_description": (0.0, 1.0, 0.0),
        },
    )

    result = pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=subset,
        strategy=NORMALIZED_MEAN_TEXT_EMBEDDING,
    )

    assert result.schema_version == PROMPT_POOLING_SCHEMA_VERSION
    assert result.accepted_taxon_key == ensemble.accepted_taxon_key
    assert result.scientific_name == ensemble.scientific_name
    assert result.pooled_score == pytest.approx(1 / math.sqrt(2))
    assert result.pooled_score != pytest.approx((1.0 + 0.0) / 2)
    assert result.pooled_text_embedding == pytest.approx(
        (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0)
    )
    assert sum(item.in_subset for item in result.prompt_scores) == 2
    assert sum(item.contributed for item in result.prompt_scores) == 2
    assert len(result.prompt_scores) == len(ensemble.variants)
    with pytest.raises(ValueError, match="result fingerprint"):
        prompt_pooling_result_payload(replace(result, pooled_score=0.0))


@pytest.mark.parametrize(
    ("strategy", "expected_score", "contribution_count"),
    (
        (MAX_PROMPT_SIMILARITY, 0.80, 1),
        (MEAN_BEST_TWO, 0.70, 2),
    ),
)
def test_score_pooling_strategies_keep_every_prompt_diagnostic(
    strategy: str,
    expected_score: float,
    contribution_count: int,
) -> None:
    ensemble = _ensemble()
    subset = _subset(
        ensemble,
        "accepted_scientific_name",
        "species_description",
        "life_stage_adult",
    )
    embeddings = _embedding_map(
        ensemble,
        {
            "accepted_scientific_name": (0.80, 0.60, 0.0),
            "species_description": (0.60, 0.80, 0.0),
            "life_stage_adult": (-0.20, math.sqrt(0.96), 0.0),
        },
    )

    result = pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=subset,
        strategy=strategy,
    )
    payload = prompt_pooling_result_payload(result)

    assert result.pooled_score == pytest.approx(expected_score)
    assert sum(item.contributed for item in result.prompt_scores) == contribution_count
    assert len(payload["prompt_scores"]) == len(ensemble.variants)
    assert {item["evidence_kind"] for item in payload["prompt_scores"]} >= {
        "accepted_taxonomy",
        "vernacular_name",
    }
    assert {item["raw_similarity"] for item in payload["prompt_scores"]} >= {
        -0.20,
        0.60,
        0.80,
    }
    assert payload["score_kind"] == "cosine_similarity"
    assert payload["result_fingerprint"] == result.result_fingerprint


def test_learned_weights_are_validation_bound_and_use_normalized_text_pool() -> None:
    ensemble = _ensemble()
    subset = _subset(
        ensemble,
        "accepted_scientific_name",
        "species_description",
    )
    selected = [
        variant
        for variant in ensemble.variants
        if variant.prompt_kind in subset.prompt_kinds
    ]
    weights = build_learned_prompt_weights(
        ensemble=ensemble,
        subset=subset,
        weights={
            selected[0].variant_fingerprint: 0.8,
            selected[1].variant_fingerprint: 0.2,
        },
        model_fingerprint=MODEL_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
        fit_partition="model_selection",
    )
    embeddings = _embedding_map(
        ensemble,
        {
            "accepted_scientific_name": (1.0, 0.0, 0.0),
            "species_description": (0.0, 1.0, 0.0),
        },
    )

    result = pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=subset,
        strategy=LEARNED_PROMPT_WEIGHTS,
        learned_weights=weights,
    )

    assert result.pooled_score == pytest.approx(0.8 / math.sqrt(0.8**2 + 0.2**2))
    assert result.weight_artifact_fingerprint == weights.weight_artifact_fingerprint
    assert [
        item.pooling_weight for item in result.prompt_scores if item.in_subset
    ] == pytest.approx([0.8, 0.2])

    with pytest.raises(ValueError, match="model_selection"):
        build_learned_prompt_weights(
            ensemble=ensemble,
            subset=subset,
            weights={
                selected[0].variant_fingerprint: 0.8,
                selected[1].variant_fingerprint: 0.2,
            },
            model_fingerprint=MODEL_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
            fit_partition="final_test",
        )


def test_stage_domain_subset_does_not_blindly_include_optional_prompts() -> None:
    ensemble = _ensemble()

    core = build_stage_domain_prompt_subset(ensemble)
    with_names = build_stage_domain_prompt_subset(
        ensemble,
        include_vernacular_names=True,
    )

    assert core.visual_domain == "field"
    assert "trusted_vernacular_with_scientific_name" not in core.prompt_kinds
    assert "trusted_vernacular_with_scientific_name" in with_names.prompt_kinds
    assert core.subset_fingerprint != with_names.subset_fingerprint


def test_pooling_rejects_incomplete_embedding_and_weight_contracts() -> None:
    ensemble = _ensemble()
    subset = _subset(
        ensemble,
        "accepted_scientific_name",
        "species_description",
    )
    embeddings = _embedding_map(ensemble)
    embeddings.pop(next(iter(embeddings)))

    with pytest.raises(ValueError, match="text embedding key set"):
        pool_prompt_ensemble(
            ensemble=ensemble,
            image_embedding=(1.0, 0.0, 0.0),
            text_embeddings=embeddings,
            model_fingerprint=MODEL_FINGERPRINT,
            subset=subset,
            strategy=MAX_PROMPT_SIMILARITY,
        )

    with pytest.raises(ValueError, match="requires learned weights"):
        pool_prompt_ensemble(
            ensemble=ensemble,
            image_embedding=(1.0, 0.0, 0.0),
            text_embeddings=_embedding_map(ensemble),
            model_fingerprint=MODEL_FINGERPRINT,
            subset=subset,
            strategy=LEARNED_PROMPT_WEIGHTS,
        )
