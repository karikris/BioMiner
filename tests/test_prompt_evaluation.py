from __future__ import annotations

from dataclasses import replace
from math import sqrt

import pytest

from biominer.bioclip.prompt_evaluation import (
    COMMON_NAME_EFFECT,
    PROMPT_EVALUATION_REPORT_SCHEMA_VERSION,
    TAXONOMIC_PATH_EFFECT,
    PromptCandidateEvaluation,
    PromptEvaluationConfiguration,
    evaluate_taxonomic_prompt_ensembles,
    prompt_evaluation_report_payload,
    prompt_version_selection_payload,
    select_prompt_version,
)
from biominer.bioclip.prompt_pooling import (
    MAX_PROMPT_SIMILARITY,
    build_prompt_subset_policy,
    pool_prompt_ensemble,
)
from biominer.bioclip.prompt_templates import (
    TAXONOMIC_PROMPT_VERSION,
    AcceptedTaxonPromptContext,
    PromptNameEvidence,
    StructuredGeographicPromptEvidence,
    TaxonomicPathNode,
    build_geography_prompt_ablation,
    build_taxonomic_prompt_ensemble,
)
from test_nonmatch_scoring import _sha


MODEL_FINGERPRINT = _sha("prompt-benchmark-model")
SPLIT_FINGERPRINT = _sha("prompt-benchmark-four-way-split")
TARGET_KEY = "gbif:6432573"
SPECIES = (
    (TARGET_KEY, "Papilio demoleus", "lime butterfly"),
    ("gbif:1936", "Papilio polytes", "common mormon"),
    ("gbif:1937", "Papilio machaon", "old world swallowtail"),
)
ITEMS = (
    ("adult-field", "adult_field", "adult", "field"),
    ("larva-field", "larval", "larva", "field"),
    ("adult-specimen", "pinned_specimen", "adult", "specimen"),
)
REFERENCE_SCORES = (0.9, 0.5, 0.1)

MODEL_SELECTION_SCORES = {
    "core": {
        "adult-field": (0.30, 0.70, 0.50),
        "larva-field": (0.45, 0.65, 0.55),
        "adult-specimen": (0.55, 0.60, 0.20),
    },
    "path": {
        "adult-field": (0.75, 0.60, 0.40),
        "larva-field": (0.60, 0.62, 0.40),
        "adult-specimen": (0.70, 0.50, 0.25),
    },
    "common": {
        "adult-field": (0.80, 0.50, 0.20),
        "larva-field": (0.80, 0.50, 0.20),
        "adult-specimen": (0.80, 0.50, 0.20),
    },
}

FINAL_TEST_SCORES = {
    "core": {item_id: (0.90, 0.50, 0.20) for item_id, *_ in ITEMS},
    "path": {item_id: (0.70, 0.60, 0.20) for item_id, *_ in ITEMS},
    "common": {item_id: (0.40, 0.99, 0.95) for item_id, *_ in ITEMS},
}


def _configurations() -> tuple[PromptEvaluationConfiguration, ...]:
    return (
        PromptEvaluationConfiguration(
            configuration_id="core",
            prompt_version=TAXONOMIC_PROMPT_VERSION,
            pooling_strategy=MAX_PROMPT_SIMILARITY,
            model_fingerprint=MODEL_FINGERPRINT,
            common_names_enabled=False,
            taxonomic_path_enabled=False,
        ),
        PromptEvaluationConfiguration(
            configuration_id="path",
            prompt_version=TAXONOMIC_PROMPT_VERSION,
            pooling_strategy=MAX_PROMPT_SIMILARITY,
            model_fingerprint=MODEL_FINGERPRINT,
            common_names_enabled=False,
            taxonomic_path_enabled=True,
        ),
        PromptEvaluationConfiguration(
            configuration_id="common",
            prompt_version=TAXONOMIC_PROMPT_VERSION,
            pooling_strategy=MAX_PROMPT_SIMILARITY,
            model_fingerprint=MODEL_FINGERPRINT,
            common_names_enabled=True,
            taxonomic_path_enabled=True,
        ),
    )


def _geography_configuration() -> PromptEvaluationConfiguration:
    return PromptEvaluationConfiguration(
        configuration_id="geography-ablation",
        prompt_version=TAXONOMIC_PROMPT_VERSION,
        pooling_strategy=MAX_PROMPT_SIMILARITY,
        model_fingerprint=MODEL_FINGERPRINT,
        common_names_enabled=True,
        taxonomic_path_enabled=True,
        geography_prompt_ablation_enabled=True,
    )


def _context(
    accepted_taxon_key: str,
    scientific_name: str,
) -> AcceptedTaxonPromptContext:
    return AcceptedTaxonPromptContext(
        accepted_taxon_key=accepted_taxon_key,
        scientific_name=scientific_name,
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
                scientific_name=scientific_name,
                accepted_taxon_key=accepted_taxon_key,
            ),
        ),
        taxonomy_source="gbif",
        taxonomy_version="backbone-2026-07",
        taxonomy_fingerprint=_sha("gbif-backbone-2026-07"),
    )


def _vernacular(
    accepted_taxon_key: str,
    common_name: str,
) -> PromptNameEvidence:
    return PromptNameEvidence(
        display_name=common_name,
        name_class="vernacular",
        trust_tier="T2",
        source="gbif",
        source_record_id=f"{accepted_taxon_key}:vernacular:en",
        language="en",
        review_state="accepted",
    )


def _text_vector(score: float) -> tuple[float, float]:
    return score, sqrt(1.0 - score**2)


def _pooling_result(
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    common_name: str,
    route: str,
    life_stage: str,
    visual_domain: str,
    configuration: PromptEvaluationConfiguration,
    family_scores,
):
    ensemble = build_taxonomic_prompt_ensemble(
        context=_context(accepted_taxon_key, scientific_name),
        route=route,
        life_stage=life_stage,
        vernacular_names=(_vernacular(accepted_taxon_key, common_name),),
    )
    if configuration.geography_prompt_ablation_enabled:
        ensemble = build_geography_prompt_ablation(
            ensemble=ensemble,
            geographic_evidence=StructuredGeographicPromptEvidence(
                accepted_taxon_key=accepted_taxon_key,
                scope_type="country",
                scope_id="country:IN",
                display_name="India",
                display_language="en",
                country_code="IN",
                source_artifact="taxon_geographic_spread.parquet",
                source_schema_version="taxon-geographic-spread-v1.0.0",
                source_record_id=f"{accepted_taxon_key}:country:IN",
                source_record_fingerprint=_sha(
                    f"spread:{accepted_taxon_key}:country:IN"
                ),
            ),
            ablation_id="country-text-ablation",
            explicit_opt_in=True,
        )
    kinds: list[str] = []
    for variant in ensemble.variants:
        include = variant.evidence_kind == "accepted_taxonomy"
        if variant.prompt_kind == "accepted_taxonomic_path":
            include = configuration.taxonomic_path_enabled
        if variant.evidence_kind == "vernacular_name":
            include = configuration.common_names_enabled
        if variant.geography_bearing:
            include = configuration.geography_prompt_ablation_enabled
        if include and variant.prompt_kind not in kinds:
            kinds.append(variant.prompt_kind)
    subset = build_prompt_subset_policy(
        ensemble,
        subset_id=configuration.configuration_id,
        visual_domain=visual_domain,
        prompt_kinds=kinds,
        enable_geography_prompt_ablation=(
            configuration.geography_prompt_ablation_enabled
        ),
    )
    embeddings = {
        variant.variant_fingerprint: _text_vector(
            family_scores[
                "geography"
                if variant.geography_bearing
                else (
                    "common"
                    if variant.evidence_kind == "vernacular_name"
                    else (
                        "path"
                        if variant.prompt_kind == "accepted_taxonomic_path"
                        else "core"
                    )
                )
            ]
        )
        for variant in ensemble.variants
    }
    return pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0),
        text_embeddings=embeddings,
        model_fingerprint=MODEL_FINGERPRINT,
        subset=subset,
        strategy=MAX_PROMPT_SIMILARITY,
    )


def _rows(
    dataset_split: str,
    scores_by_configuration,
) -> tuple[PromptCandidateEvaluation, ...]:
    rows: list[PromptCandidateEvaluation] = []
    for configuration in _configurations():
        for item_id, route, life_stage, visual_domain in ITEMS:
            for index, (accepted_key, scientific_name, common_name) in enumerate(
                SPECIES
            ):
                rows.append(
                    PromptCandidateEvaluation(
                        item_id=item_id,
                        dataset_split=dataset_split,
                        split_fingerprint=SPLIT_FINGERPRINT,
                        expected_accepted_taxon_key=TARGET_KEY,
                        candidate_accepted_taxon_key=accepted_key,
                        route=route,
                        life_stage=life_stage,
                        visual_domain=visual_domain,
                        candidate_set_fingerprint=_sha(f"candidate-set:{item_id}"),
                        configuration=configuration,
                        pooling_result=_pooling_result(
                            accepted_taxon_key=accepted_key,
                            scientific_name=scientific_name,
                            common_name=common_name,
                            route=route,
                            life_stage=life_stage,
                            visual_domain=visual_domain,
                            configuration=configuration,
                            family_scores={
                                "core": scores_by_configuration["core"][item_id][index],
                                "path": scores_by_configuration["path"][item_id][index],
                                "common": scores_by_configuration["common"][item_id][
                                    index
                                ],
                            },
                        ),
                        reference_image_score=REFERENCE_SCORES[index],
                        reference_evidence_fingerprint=_sha(
                            f"reference:{item_id}:{accepted_key}"
                        ),
                    )
                )
    return tuple(rows)


def _geography_rows(dataset_split: str) -> tuple[PromptCandidateEvaluation, ...]:
    configuration = _geography_configuration()
    rows: list[PromptCandidateEvaluation] = []
    for item_id, route, life_stage, visual_domain in ITEMS:
        for index, (accepted_key, scientific_name, common_name) in enumerate(SPECIES):
            rows.append(
                PromptCandidateEvaluation(
                    item_id=item_id,
                    dataset_split=dataset_split,
                    split_fingerprint=SPLIT_FINGERPRINT,
                    expected_accepted_taxon_key=TARGET_KEY,
                    candidate_accepted_taxon_key=accepted_key,
                    route=route,
                    life_stage=life_stage,
                    visual_domain=visual_domain,
                    candidate_set_fingerprint=_sha(f"candidate-set:{item_id}"),
                    configuration=configuration,
                    pooling_result=_pooling_result(
                        accepted_taxon_key=accepted_key,
                        scientific_name=scientific_name,
                        common_name=common_name,
                        route=route,
                        life_stage=life_stage,
                        visual_domain=visual_domain,
                        configuration=configuration,
                        family_scores={
                            "core": MODEL_SELECTION_SCORES["core"][item_id][index],
                            "path": MODEL_SELECTION_SCORES["path"][item_id][index],
                            "common": MODEL_SELECTION_SCORES["common"][item_id][index],
                            "geography": (0.99, 0.20, 0.10)[index],
                        },
                    ),
                    reference_image_score=REFERENCE_SCORES[index],
                    reference_evidence_fingerprint=_sha(
                        f"reference:{item_id}:{accepted_key}"
                    ),
                )
            )
    return tuple(rows)


def _summary(report, dataset_split: str, configuration_id: str):
    return next(
        summary
        for summary in report.summaries
        if summary.dataset_split == dataset_split
        and summary.configuration.configuration_id == configuration_id
    )


def test_benchmark_measures_rank_recall_margin_subgroups_effects_and_correlation() -> (
    None
):
    rows = _rows("model_selection", MODEL_SELECTION_SCORES)
    report = evaluate_taxonomic_prompt_ensembles(rows, recall_ks=(1, 2, 3))
    reversed_report = evaluate_taxonomic_prompt_ensembles(
        tuple(reversed(rows)),
        recall_ks=(1, 2, 3),
    )

    assert report == reversed_report
    assert report.schema_version == PROMPT_EVALUATION_REPORT_SCHEMA_VERSION
    assert len(report.item_results) == 9
    core = _summary(report, "model_selection", "core")
    path = _summary(report, "model_selection", "path")
    common = _summary(report, "model_selection", "common")
    assert dict(core.overall.species_recall_at_k) == {1: 0.0, 2: 1 / 3, 3: 1.0}
    assert dict(path.overall.species_recall_at_k)[1] == pytest.approx(2 / 3)
    assert dict(common.overall.species_recall_at_k) == {1: 1.0, 2: 1.0, 3: 1.0}
    assert common.overall.mean_target_versus_competitor_text_margin == pytest.approx(
        0.15
    )
    assert common.overall.prompt_reference_spearman > 0.95
    assert common.overall.mean_item_prompt_reference_spearman == pytest.approx(1.0)
    assert {
        (value.slice_kind, value.slice_value, value.item_count)
        for value in common.subgroups
    } >= {
        ("life_stage", "adult", 2),
        ("life_stage", "larva", 1),
        ("visual_domain", "field", 2),
        ("visual_domain", "specimen", 1),
        ("life_stage_visual_domain", "adult|field", 1),
    }
    effects = {
        (value.effect_kind, value.baseline_configuration_id): value
        for value in report.effects
    }
    assert effects[(TAXONOMIC_PATH_EFFECT, "core")].delta_species_recall_at_k[
        0
    ] == pytest.approx((1, 2 / 3))
    assert effects[(COMMON_NAME_EFFECT, "path")].delta_species_recall_at_k[
        0
    ] == pytest.approx((1, 1 / 3))
    assert effects[(COMMON_NAME_EFFECT, "path")].delta_mean_target_prompt_rank < 0
    payload = prompt_evaluation_report_payload(report)
    assert payload["report_fingerprint"] == report.report_fingerprint
    assert payload["summaries"][2]["overall"]["species_recall_at_k"]


def test_target_prompt_rank_is_distinct_from_pooled_species_rank() -> None:
    report = evaluate_taxonomic_prompt_ensembles(
        _rows("model_selection", MODEL_SELECTION_SCORES),
        recall_ks=(1, 2, 3),
    )
    adult_core = next(
        result
        for result in report.item_results
        if result.configuration_id == "core" and result.item_id == "adult-field"
    )

    assert adult_core.target_species_rank == 3
    assert adult_core.target_prompt_rank > adult_core.target_species_rank
    assert adult_core.best_competitor_accepted_taxon_key == "gbif:1936"
    assert adult_core.target_versus_competitor_text_margin == pytest.approx(-0.4)


def test_prompt_selection_uses_model_selection_only_even_with_final_test_rows() -> None:
    model_rows = _rows("model_selection", MODEL_SELECTION_SCORES)
    model_report = evaluate_taxonomic_prompt_ensembles(
        model_rows,
        recall_ks=(1, 2, 3),
    )
    combined_report = evaluate_taxonomic_prompt_ensembles(
        (*model_rows, *_rows("final_test", FINAL_TEST_SCORES)),
        recall_ks=(1, 2, 3),
    )
    model_selection = select_prompt_version(model_report, primary_recall_k=1)
    combined_selection = select_prompt_version(combined_report, primary_recall_k=1)

    assert model_selection.selected_configuration_id == "common"
    assert combined_selection.selected_configuration_id == "common"
    assert combined_selection.selected_prompt_version == TAXONOMIC_PROMPT_VERSION
    assert combined_selection.selection_input_fingerprint == (
        model_selection.selection_input_fingerprint
    )
    assert (
        dict(
            _summary(combined_report, "final_test", "core").overall.species_recall_at_k
        )[1]
        == 1.0
    )
    assert (
        dict(
            _summary(
                combined_report, "final_test", "common"
            ).overall.species_recall_at_k
        )[1]
        == 0.0
    )
    payload = prompt_version_selection_payload(combined_selection)
    assert payload["selection_partition"] == "model_selection"
    with pytest.raises(ValueError, match="model_selection only"):
        select_prompt_version(
            combined_report,
            selection_partition="final_test",
            primary_recall_k=1,
        )


def test_prompt_selection_never_promotes_geography_ablation() -> None:
    report = evaluate_taxonomic_prompt_ensembles(
        (
            *_rows("model_selection", MODEL_SELECTION_SCORES),
            *_geography_rows("model_selection"),
        ),
        recall_ks=(1, 2, 3),
    )
    geography = _summary(report, "model_selection", "geography-ablation")
    selection = select_prompt_version(report, primary_recall_k=1)

    assert dict(geography.overall.species_recall_at_k)[1] == 1.0
    assert geography.overall.mean_target_versus_competitor_text_margin == pytest.approx(
        0.34
    )
    assert selection.selected_configuration_id == "common"
    assert selection.excluded_geography_configuration_count == 1


def test_evaluation_rejects_unbalanced_candidates_and_changed_reference_evidence() -> (
    None
):
    rows = list(_rows("model_selection", MODEL_SELECTION_SCORES))
    common_adult_competitor = next(
        index
        for index, row in enumerate(rows)
        if row.configuration.configuration_id == "common"
        and row.item_id == "adult-field"
        and row.candidate_accepted_taxon_key == "gbif:1937"
    )
    unbalanced = rows[:common_adult_competitor] + rows[common_adult_competitor + 1 :]
    with pytest.raises(ValueError, match="candidate sets differ"):
        evaluate_taxonomic_prompt_ensembles(unbalanced, recall_ks=(1, 2, 3))

    changed = list(rows)
    changed[common_adult_competitor] = replace(
        changed[common_adult_competitor],
        reference_image_score=0.8,
    )
    with pytest.raises(ValueError, match="reference-image evidence differs"):
        evaluate_taxonomic_prompt_ensembles(changed, recall_ks=(1, 2, 3))


def test_evaluation_requires_both_named_prompt_ablation_effects() -> None:
    rows = tuple(
        row
        for row in _rows("model_selection", MODEL_SELECTION_SCORES)
        if row.configuration.configuration_id != "common"
    )

    with pytest.raises(ValueError, match="common_names"):
        evaluate_taxonomic_prompt_ensembles(rows, recall_ks=(1, 2, 3))


def test_result_configuration_flags_and_report_fingerprints_fail_closed() -> None:
    rows = _rows("model_selection", MODEL_SELECTION_SCORES)
    core = rows[0]
    path_configuration = _configurations()[1]
    with pytest.raises(ValueError, match="taxonomic-path prompt selection"):
        replace(core, configuration=path_configuration)
    with pytest.raises(ValueError, match="accepted taxon does not match"):
        replace(core, candidate_accepted_taxon_key="gbif:wrong")

    changed_scores = list(rows)
    common_target_index = next(
        index
        for index, row in enumerate(changed_scores)
        if row.configuration.configuration_id == "common"
        and row.item_id == "adult-field"
        and row.candidate_accepted_taxon_key == TARGET_KEY
    )
    common_target = changed_scores[common_target_index]
    changed_scores[common_target_index] = replace(
        common_target,
        pooling_result=_pooling_result(
            accepted_taxon_key=TARGET_KEY,
            scientific_name="Papilio demoleus",
            common_name="lime butterfly",
            route="adult_field",
            life_stage="adult",
            visual_domain="field",
            configuration=common_target.configuration,
            family_scores={"core": 0.31, "path": 0.75, "common": 0.80},
        ),
    )
    with pytest.raises(ValueError, match="shared prompt score changes"):
        evaluate_taxonomic_prompt_ensembles(changed_scores, recall_ks=(1, 2, 3))

    report = evaluate_taxonomic_prompt_ensembles(rows, recall_ks=(1, 2, 3))
    tampered_item = replace(report.item_results[0], target_species_rank=99)
    with pytest.raises(ValueError, match="item fingerprint is inconsistent"):
        prompt_evaluation_report_payload(
            replace(
                report,
                item_results=(tampered_item, *report.item_results[1:]),
            )
        )

    selection = select_prompt_version(report, primary_recall_k=1)
    with pytest.raises(ValueError, match="selection fingerprint is inconsistent"):
        prompt_version_selection_payload(
            replace(selection, selected_configuration_id="core")
        )
