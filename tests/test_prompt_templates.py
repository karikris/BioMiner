from __future__ import annotations

from biominer.bioclip.prompt_templates import (
    PromptVariant,
    aggregate_prompt_scores,
    build_species_prompt_variants,
)


def test_build_species_prompt_variants_includes_scientific_and_common_names() -> None:
    variants = build_species_prompt_variants(
        scientific_name="Papilio demoleus",
        common_names=("lime butterfly", "chequered swallowtail"),
    )

    labels = [variant.label for variant in variants]
    assert "a photo of Papilio demoleus" in labels
    assert "a field photo of Papilio demoleus adult butterfly" in labels
    assert "a photo of lime butterfly" in labels
    assert all(variant.taxon_key == "Papilio demoleus" for variant in variants)


def test_aggregate_prompt_scores_uses_max_and_keeps_evidence() -> None:
    variants = [
        PromptVariant(label="a photo of Papilio demoleus", taxon_key="Papilio demoleus", prompt_kind="scientific"),
        PromptVariant(label="a photo of lime butterfly", taxon_key="Papilio demoleus", prompt_kind="common"),
        PromptVariant(label="a photo of Papilio machaon", taxon_key="Papilio machaon", prompt_kind="scientific"),
    ]

    result = aggregate_prompt_scores(
        scores={
            "a photo of Papilio demoleus": 0.72,
            "a photo of lime butterfly": 0.81,
            "a photo of Papilio machaon": 0.11,
        },
        variants=variants,
        top_k=2,
    )

    assert result[0]["taxon_key"] == "Papilio demoleus"
    assert result[0]["score"] == 0.81
    assert result[0]["best_label"] == "a photo of lime butterfly"
    assert result[0]["prompt_scores"]["a photo of Papilio demoleus"] == 0.72
