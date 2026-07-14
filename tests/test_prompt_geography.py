from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.bioclip.prompt_pooling import (
    MAX_PROMPT_SIMILARITY,
    build_prompt_subset_policy,
    build_stage_domain_prompt_subset,
    pool_prompt_ensemble,
    prompt_pooling_result_payload,
)
from biominer.bioclip.prompt_templates import (
    EXPLICIT_GEOGRAPHY_PROMPT_ABLATION,
    GEOGRAPHIC_PROMPT_ABLATION_VERSION,
    STRUCTURED_GEOGRAPHY_ONLY,
    ReviewedPromptAlias,
    StructuredGeographicPromptEvidence,
    build_geography_prompt_ablation,
    build_taxonomic_prompt_ensemble,
    structured_geographic_prompt_evidence_payload,
    taxonomic_prompt_ensemble_payload,
)
from test_nonmatch_scoring import _sha
from test_prompt_templates import _context


MODEL_FINGERPRINT = _sha("geography-ablation-model")


def _base_ensemble(*, aliases=()):
    return build_taxonomic_prompt_ensemble(
        context=_context(),
        route="adult_field",
        life_stage="adult",
        reviewed_aliases=aliases,
    )


def _geographic_evidence(
    *,
    accepted_taxon_key: str = "gbif:6432573",
) -> StructuredGeographicPromptEvidence:
    return StructuredGeographicPromptEvidence(
        accepted_taxon_key=accepted_taxon_key,
        scope_type="country",
        scope_id="country:IN",
        display_name="India",
        display_language="en",
        country_code="IN",
        source_artifact="taxon_geographic_spread.parquet",
        source_schema_version="taxon-geographic-spread-v1.0.0",
        source_record_id="gbif:6432573:country:IN",
        source_record_fingerprint=_sha("papilio-demoleus-india-spread-row"),
    )


def _embedding_map(ensemble):
    return {
        variant.variant_fingerprint: (
            (1.0, 0.0) if variant.geography_bearing else (0.0, 1.0)
        )
        for variant in ensemble.variants
    }


def test_default_prompt_ensemble_keeps_geography_structured_and_out_of_text() -> None:
    ensemble = _base_ensemble()
    payload = taxonomic_prompt_ensemble_payload(ensemble)
    subset = build_stage_domain_prompt_subset(ensemble)

    assert ensemble.geography_mode == STRUCTURED_GEOGRAPHY_ONLY
    assert all(not variant.geography_bearing for variant in ensemble.variants)
    assert all("India" not in variant.label for variant in ensemble.variants)
    assert payload["geography_policy"] == {
        "mode": STRUCTURED_GEOGRAPHY_ONLY,
        "structured_geographic_evidence": None,
        "ablation_version": None,
        "ablation_id": None,
        "base_ensemble_fingerprint": None,
        "ablation_fingerprint": None,
    }
    assert subset.geography_prompt_ablation_enabled is False
    assert subset.geography_ablation_fingerprint is None


def test_geography_marked_reviewed_alias_is_excluded_from_normal_prompts() -> None:
    alias = ReviewedPromptAlias(
        alias_id="alias:india",
        label="a field photograph of Papilio demoleus in India",
        source="manual_prompt_review",
        review_state="approved",
        reviewed_by="reviewer-1",
        geography_bearing=True,
    )

    ensemble = _base_ensemble(aliases=(alias,))

    assert alias.label not in {variant.label for variant in ensemble.variants}
    assert {(item.evidence_id, item.reason) for item in ensemble.exclusions} == {
        ("alias:india", "geography_requires_explicit_ablation")
    }


def test_structured_geographic_evidence_is_versioned_and_fingerprinted() -> None:
    evidence = _geographic_evidence()
    payload = structured_geographic_prompt_evidence_payload(evidence)

    assert payload["scope_type"] == "country"
    assert payload["country_code"] == "IN"
    assert payload["source_artifact"] == "taxon_geographic_spread.parquet"
    assert payload["evidence_fingerprint"] == evidence.evidence_fingerprint
    with pytest.raises(ValueError, match="fingerprint is inconsistent"):
        replace(evidence, display_name="Indonesia")
    with pytest.raises(ValueError, match="unsupported.*source artifact"):
        replace(
            evidence,
            source_artifact="flickr_query_hits.parquet",
            evidence_fingerprint="",
        )


def test_geography_prompt_ablation_requires_explicit_matching_evidence() -> None:
    base = _base_ensemble()
    evidence = _geographic_evidence()

    with pytest.raises(ValueError, match="explicit ablation opt-in"):
        build_geography_prompt_ablation(
            ensemble=base,
            geographic_evidence=evidence,
            ablation_id="country-prompt-ablation",
            explicit_opt_in=False,
        )
    with pytest.raises(TypeError, match="must be boolean"):
        build_geography_prompt_ablation(
            ensemble=base,
            geographic_evidence=evidence,
            ablation_id="country-prompt-ablation",
            explicit_opt_in=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="taxon does not match"):
        build_geography_prompt_ablation(
            ensemble=base,
            geographic_evidence=_geographic_evidence(accepted_taxon_key="gbif:9999999"),
            ablation_id="country-prompt-ablation",
            explicit_opt_in=True,
        )


def test_explicit_ablation_binds_base_evidence_variant_and_fingerprints() -> None:
    base = _base_ensemble()
    evidence = _geographic_evidence()
    ensemble = build_geography_prompt_ablation(
        ensemble=base,
        geographic_evidence=evidence,
        ablation_id="country-prompt-ablation",
        explicit_opt_in=True,
    )
    payload = taxonomic_prompt_ensemble_payload(ensemble)
    geographic = [variant for variant in ensemble.variants if variant.geography_bearing]

    assert base.geography_mode == STRUCTURED_GEOGRAPHY_ONLY
    assert ensemble.geography_mode == EXPLICIT_GEOGRAPHY_PROMPT_ABLATION
    assert ensemble.base_ensemble_fingerprint == base.ensemble_fingerprint
    assert ensemble.geography_ablation_version == GEOGRAPHIC_PROMPT_ABLATION_VERSION
    assert len(geographic) == 1
    assert geographic[0].label == (
        "a field photograph of an adult Papilio demoleus butterfly in India"
    )
    assert geographic[0].evidence_id == evidence.source_record_id
    assert (
        payload["geography_policy"]["structured_geographic_evidence"][
            "evidence_fingerprint"
        ]
        == evidence.evidence_fingerprint
    )
    with pytest.raises(ValueError, match="ablation fingerprint is inconsistent"):
        taxonomic_prompt_ensemble_payload(
            replace(ensemble, geography_ablation_fingerprint=_sha("tampered"))
        )


def test_pooling_excludes_geography_by_default_and_requires_ablation_flag() -> None:
    ensemble = build_geography_prompt_ablation(
        ensemble=_base_ensemble(),
        geographic_evidence=_geographic_evidence(),
        ablation_id="country-prompt-ablation",
        explicit_opt_in=True,
    )
    geography_kind = next(
        variant.prompt_kind
        for variant in ensemble.variants
        if variant.geography_bearing
    )
    default_subset = build_stage_domain_prompt_subset(ensemble)
    all_kinds = tuple(
        dict.fromkeys(variant.prompt_kind for variant in ensemble.variants)
    )

    assert geography_kind not in default_subset.prompt_kinds
    with pytest.raises(ValueError, match="explicit ablation enablement"):
        build_prompt_subset_policy(
            ensemble,
            subset_id="all-prompts-without-opt-in",
            visual_domain="field",
            prompt_kinds=all_kinds,
        )

    embeddings = _embedding_map(ensemble)
    default_result = pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=default_subset,
        strategy=MAX_PROMPT_SIMILARITY,
    )
    geographic_default = next(
        item for item in default_result.prompt_scores if item.geography_bearing
    )
    assert default_result.selected_geography_prompt_count == 0
    assert default_result.geography_prompt_ablation_enabled is False
    assert geographic_default.raw_similarity == pytest.approx(1.0)
    assert geographic_default.selection_reason == "geography_ablation_disabled"

    ablation_subset = build_prompt_subset_policy(
        ensemble,
        subset_id="all-prompts-country-ablation",
        visual_domain="field",
        prompt_kinds=all_kinds,
        enable_geography_prompt_ablation=True,
    )
    result = pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=ablation_subset,
        strategy=MAX_PROMPT_SIMILARITY,
    )
    payload = prompt_pooling_result_payload(result)

    assert result.geography_prompt_ablation_enabled is True
    assert result.selected_geography_prompt_count == 1
    assert result.best_prompt_label.endswith(" in India")
    assert payload["geography_ablation_fingerprint"] == (
        ensemble.geography_ablation_fingerprint
    )
    assert sum(item["geography_bearing"] for item in payload["prompt_scores"]) == 1
