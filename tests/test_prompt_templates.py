from __future__ import annotations

import hashlib

import pytest

from biominer.bioclip.prompt_templates import (
    TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION,
    TAXONOMIC_PROMPT_VERSION,
    AcceptedTaxonPromptContext,
    PromptVariant,
    PromptNameEvidence,
    ReviewedPromptAlias,
    SPECIES_PROMPT_AGGREGATION_DEFAULT,
    TaxonomicPathNode,
    aggregate_prompt_scores,
    build_taxonomic_prompt_ensemble,
)


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _context() -> AcceptedTaxonPromptContext:
    return AcceptedTaxonPromptContext(
        accepted_taxon_key="gbif:6432573",
        scientific_name="Papilio demoleus",
        genus="Papilio",
        family="Papilionidae",
        taxonomic_path=(
            TaxonomicPathNode(
                rank="SUPERFAMILY",
                scientific_name="Papilionoidea",
                accepted_taxon_key="gbif:1875",
            ),
            TaxonomicPathNode(
                rank="FAMILY",
                scientific_name="Papilionidae",
                accepted_taxon_key="gbif:9417",
            ),
            TaxonomicPathNode(
                rank="GENUS",
                scientific_name="Papilio",
                accepted_taxon_key="gbif:1935",
            ),
            TaxonomicPathNode(
                rank="SPECIES",
                scientific_name="Papilio demoleus",
                accepted_taxon_key="gbif:6432573",
            ),
        ),
        taxonomy_source="gbif",
        taxonomy_version="backbone-2026-07",
        taxonomy_fingerprint=_sha("gbif-backbone-2026-07"),
    )


def _vernacular(
    name: str,
    *,
    trust_tier: str = "T2",
    name_class: str = "vernacular",
    weak_homonym: bool = False,
) -> PromptNameEvidence:
    return PromptNameEvidence(
        display_name=name,
        name_class=name_class,
        trust_tier=trust_tier,
        source="gbif" if trust_tier != "T5" else "machine_translation",
        source_record_id=f"name:{name}",
        language="en",
        review_state="accepted",
        weak_homonym=weak_homonym,
    )


def test_taxonomic_prompt_ensemble_has_versioned_hierarchical_variants() -> None:
    ensemble = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        vernacular_names=(_vernacular("lime butterfly"),),
    )

    labels = [variant.label for variant in ensemble.variants]
    assert labels[:4] == [
        "Papilio demoleus",
        "the butterfly species Papilio demoleus",
        "a field photograph of an adult Papilio demoleus butterfly",
        "Papilio demoleus, a species of Papilio in Papilionidae",
    ]
    assert any(
        label.startswith("Papilio demoleus, accepted taxonomic path:")
        for label in labels
    )
    assert "a field photograph of the lime butterfly (Papilio demoleus)" in labels
    assert ensemble.schema_version == TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION
    assert ensemble.prompt_version == TAXONOMIC_PROMPT_VERSION
    assert ensemble.taxonomic_status == "ACCEPTED"
    assert ensemble.taxonomic_path == _context().taxonomic_path
    assert ensemble.ensemble_fingerprint.startswith("sha256:")
    assert all(
        variant.accepted_taxon_key == "gbif:6432573" for variant in ensemble.variants
    )
    assert all(
        variant.prompt_version == TAXONOMIC_PROMPT_VERSION
        for variant in ensemble.variants
    )
    assert all(
        variant.variant_fingerprint.startswith("sha256:")
        for variant in ensemble.variants
    )
    assert all(variant.geography_bearing is False for variant in ensemble.variants)


def test_prompt_ensemble_filters_untrusted_names_and_unreviewed_aliases() -> None:
    ensemble = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        vernacular_names=(
            _vernacular("lime butterfly"),
            _vernacular(
                "automatically translated butterfly",
                trust_tier="T5",
                name_class="generated_translation",
            ),
            _vernacular("common lime", weak_homonym=True),
            _vernacular("uncorroborated local label", trust_tier="T4"),
        ),
        reviewed_aliases=(
            ReviewedPromptAlias(
                alias_id="alias:approved",
                label="a dorsal field view of Papilio demoleus",
                source="manual_prompt_review",
                review_state="approved",
                reviewed_by="reviewer-1",
            ),
            ReviewedPromptAlias(
                alias_id="alias:pending",
                label="Papilio demoleus India Flickr search",
                source="flickr_query",
                review_state="pending",
                reviewed_by=None,
            ),
            ReviewedPromptAlias(
                alias_id="alias:pinned-wrong-route",
                label="a pinned specimen of Papilio demoleus",
                source="manual_prompt_review",
                review_state="approved",
                reviewed_by="reviewer-1",
            ),
        ),
    )

    labels = {variant.label for variant in ensemble.variants}
    assert "a field photograph of the lime butterfly (Papilio demoleus)" in labels
    assert "a dorsal field view of Papilio demoleus" in labels
    vernacular = next(
        variant
        for variant in ensemble.variants
        if variant.prompt_kind == "trusted_vernacular_with_scientific_name"
    )
    alias = next(
        variant
        for variant in ensemble.variants
        if variant.prompt_kind == "reviewed_prompt_alias"
    )
    assert (vernacular.evidence_source, vernacular.trust_tier) == ("gbif", "T2")
    assert (alias.evidence_source, alias.reviewed_by) == (
        "manual_prompt_review",
        "reviewer-1",
    )
    assert "automatically translated butterfly" not in " ".join(labels)
    assert "common lime" not in " ".join(labels)
    assert "uncorroborated local label" not in " ".join(labels)
    assert "Papilio demoleus India Flickr search" not in labels
    assert {(item.evidence_id, item.reason) for item in ensemble.exclusions} == {
        ("name:automatically translated butterfly", "generated_translation"),
        ("name:common lime", "weak_homonym"),
        ("name:uncorroborated local label", "untrusted_vernacular_tier"),
        ("alias:pending", "unreviewed_prompt_alias"),
        ("alias:pinned-wrong-route", "pinned_alias_requires_specimen_route"),
    }


def test_prompt_ensemble_is_order_independent_and_route_specific() -> None:
    names = (_vernacular("lime butterfly"), _vernacular("lemon butterfly"))
    aliases = (
        ReviewedPromptAlias(
            alias_id="alias:adult",
            label="an adult Papilio demoleus in natural posture",
            source="manual_prompt_review",
            review_state="approved",
            reviewed_by="reviewer-1",
            route="adult_field",
        ),
    )
    first = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        vernacular_names=names,
        reviewed_aliases=aliases,
    )
    second = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        vernacular_names=tuple(reversed(names)),
        reviewed_aliases=tuple(reversed(aliases)),
    )
    larval = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="larval",
        life_stage="larva",
    )
    pinned = build_taxonomic_prompt_ensemble(
        context=_context(),
        route="pinned_specimen",
        life_stage="adult",
    )

    assert first == second
    assert "a field photograph of a Papilio demoleus caterpillar" in {
        variant.label for variant in larval.variants
    }
    assert not any("pinned" in variant.label.casefold() for variant in first.variants)
    assert "a pinned museum specimen of Papilio demoleus" in {
        variant.label for variant in pinned.variants
    }


def test_aggregate_prompt_scores_uses_mean_by_default_and_keeps_evidence() -> None:
    assert SPECIES_PROMPT_AGGREGATION_DEFAULT == "mean"

    variants = [
        PromptVariant(
            label="a photo of Papilio demoleus",
            taxon_key="Papilio demoleus",
            prompt_kind="scientific",
        ),
        PromptVariant(
            label="a photo of lime butterfly",
            taxon_key="Papilio demoleus",
            prompt_kind="common",
        ),
        PromptVariant(
            label="a photo of Papilio machaon",
            taxon_key="Papilio machaon",
            prompt_kind="scientific",
        ),
    ]

    result = aggregate_prompt_scores(
        scores={
            "a photo of Papilio demoleus": 0.72,
            "a photo of lime butterfly": 0.08,
            "a photo of Papilio machaon": 0.55,
        },
        variants=variants,
        top_k=2,
    )

    assert result[0]["taxon_key"] == "Papilio machaon"
    assert result[0]["score"] == 0.55
    assert result[1]["taxon_key"] == "Papilio demoleus"
    assert result[1]["score"] == pytest.approx(0.40)
    assert result[1]["best_label"] == "a photo of Papilio demoleus"
    assert result[1]["prompt_scores"]["a photo of lime butterfly"] == 0.08


def test_aggregate_prompt_scores_supports_explicit_max() -> None:
    variants = [
        PromptVariant(
            label="a photo of Papilio demoleus",
            taxon_key="Papilio demoleus",
            prompt_kind="scientific",
        ),
        PromptVariant(
            label="a photo of lime butterfly",
            taxon_key="Papilio demoleus",
            prompt_kind="common",
        ),
        PromptVariant(
            label="a photo of Papilio machaon",
            taxon_key="Papilio machaon",
            prompt_kind="scientific",
        ),
    ]

    result = aggregate_prompt_scores(
        scores={
            "a photo of Papilio demoleus": 0.72,
            "a photo of lime butterfly": 0.08,
            "a photo of Papilio machaon": 0.55,
        },
        variants=variants,
        top_k=2,
        aggregation="max",
    )

    assert result[0]["taxon_key"] == "Papilio demoleus"
    assert result[0]["score"] == 0.72
    assert result[0]["best_label"] == "a photo of Papilio demoleus"
