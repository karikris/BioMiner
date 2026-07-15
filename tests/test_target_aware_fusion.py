from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from biominer.bioclip import target_aware_fusion as target_fusion_module
from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.prompt_pooling import (
    MAX_PROMPT_SIMILARITY,
    build_prompt_subset_policy,
    pool_prompt_ensemble,
)
from biominer.bioclip.prompt_templates import (
    AcceptedTaxonPromptContext,
    TaxonomicPathNode,
    build_taxonomic_prompt_ensemble,
)
from biominer.bioclip.reference_prototypes import PROTOTYPE_METHOD_NORMALIZED_MEAN
from biominer.bioclip.reference_scoring import (
    REFERENCE_EVIDENCE_SCORING_VERSION,
    CandidateReferenceEvidence,
)
from biominer.bioclip.target_aware_fusion import (
    TARGET_AWARE_EVIDENCE_FUSION_VERSION,
    CandidateStructuredEvidence,
    TargetAwareFusionQuality,
    fuse_target_aware_species_evidence,
    score_frozen_classifier,
    target_aware_fusion_result_payload,
)
from biominer.bioclip.taxonomic_evidence import (
    DERIVED_PARENT_PROBABILITY_KIND,
    DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND,
    derive_taxonomic_evidence,
    taxonomic_evidence_result_payload,
)
from biominer.bioclip.target_aware_scoring import (
    TargetAwareAuxiliaryClass,
    build_target_aware_scoring_plan,
    score_target_aware_candidate_union,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.calibration import FrozenProbabilityCalibrator
from biominer.ml.classifiers import (
    EMBEDDING_ONLY_FEATURE_SET,
    ESTIMATOR_LINEAR_SVC,
    LINEAR_SVC_EMBEDDING_MODEL,
    NON_TARGET_CLASS_LABEL,
    classifier_feature_layout,
)
from biominer.ml.decision_policy import (
    DecisionPolicyCalibrationSample,
    SelectiveDecisionPolicyConfig,
    fit_selective_decision_policy,
)
from biominer.ml.nonmatch import DomainNegativeEvidence
from biominer.ml.persistence import CLASSIFIER_VERSION, FrozenLinearClassifier


TARGET = "gbif:6432573"
COMPETITOR = "gbif:1939773"
THIRD = "gbif:5139051"
MODEL_FINGERPRINT = "sha256:" + "1" * 64
CANDIDATE_SET_FINGERPRINT = "sha256:" + "2" * 64
REFERENCE_EMBEDDING_FINGERPRINT = "sha256:" + "3" * 64
REFERENCE_PROTOTYPE_FINGERPRINT = "sha256:" + "4" * 64
SUPPORT_MANIFEST_FINGERPRINT = "sha256:" + "5" * 64


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _candidate_set(*, geo_cluster_id: str = "cluster-au-1") -> CandidateSet:
    species = (
        CandidateTaxon(
            scientific_name="Papilio demoleus",
            accepted_taxon_key=TARGET,
            family="Papilionidae",
            genus="Papilio",
            candidate_reasons=("target",),
            source_versions=("regional-candidate-species-v1.0.0",),
            target_candidate=True,
            candidate_priority=0,
        ),
        CandidateTaxon(
            scientific_name="Papilio polytes",
            accepted_taxon_key=COMPETITOR,
            family="Papilionidae",
            genus="Papilio",
            candidate_reasons=("known_mimic", "regional_same_family"),
            source_versions=("regional-candidate-species-v1.0.0",),
            candidate_priority=1,
        ),
        CandidateTaxon(
            scientific_name="Papilio machaon",
            accepted_taxon_key=THIRD,
            family="Papilionidae",
            genus="Papilio",
            candidate_reasons=("visually_nearest",),
            source_versions=("regional-candidate-species-v1.0.0",),
            candidate_priority=2,
        ),
    )
    return CandidateSet(
        candidate_set_id="regional:papilio-pilot",
        registry_version="registry-v1",
        target_accepted_taxon_key=TARGET,
        target_scientific_name="Papilio demoleus",
        family_candidates=species,
        genus_candidates=species,
        species_candidates=species,
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope=geo_cluster_id,
        source_evidence=(
            "regional_candidate_set:regional:papilio-pilot",
            "regional_candidate_source:regional-candidate-species-v1.0.0",
        ),
        candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
    )


def _plan(*, geo_cluster_id: str = "cluster-au-1"):
    return build_target_aware_scoring_plan(
        _candidate_set(geo_cluster_id=geo_cluster_id),
        known_negative_classes=(
            TargetAwareAuxiliaryClass(
                class_id="non_butterfly_insect",
                display_name="non-butterfly insect",
                source_versions=("negative-manifest-v1",),
            ),
            TargetAwareAuxiliaryClass(
                class_id="visual_artifact",
                display_name="visual artifact",
                source_versions=("negative-manifest-v1",),
            ),
        ),
        visual_domain_classes=(
            TargetAwareAuxiliaryClass(
                class_id="adult_field",
                display_name="live adult butterfly in the field",
                source_versions=("reference-domain-v1",),
            ),
            TargetAwareAuxiliaryClass(
                class_id="pinned_specimen",
                display_name="pinned butterfly specimen",
                source_versions=("reference-domain-v1",),
            ),
        ),
    )


def _hierarchy_plan():
    candidate_set = _candidate_set()
    target, competitor, third = candidate_set.species_candidates
    species = (
        target,
        replace(competitor, genus="Graphium"),
        replace(third, family="Pieridae", genus="Pieris"),
    )
    return build_target_aware_scoring_plan(
        replace(
            candidate_set,
            family_candidates=species,
            genus_candidates=species,
            species_candidates=species,
        ),
        known_negative_classes=(
            TargetAwareAuxiliaryClass(
                class_id="non_butterfly_insect",
                display_name="non-butterfly insect",
                source_versions=("negative-manifest-v1",),
            ),
            TargetAwareAuxiliaryClass(
                class_id="visual_artifact",
                display_name="visual artifact",
                source_versions=("negative-manifest-v1",),
            ),
        ),
        visual_domain_classes=(
            TargetAwareAuxiliaryClass(
                class_id="adult_field",
                display_name="live adult butterfly in the field",
                source_versions=("reference-domain-v1",),
            ),
            TargetAwareAuxiliaryClass(
                class_id="pinned_specimen",
                display_name="pinned butterfly specimen",
                source_versions=("reference-domain-v1",),
            ),
        ),
    )


def _prompt_context(key: str, name: str) -> AcceptedTaxonPromptContext:
    genus = name.split(" ", 1)[0]
    return AcceptedTaxonPromptContext(
        accepted_taxon_key=key,
        scientific_name=name,
        genus=genus,
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
                scientific_name=genus,
                accepted_taxon_key="gbif:1935",
            ),
            TaxonomicPathNode(
                rank="SPECIES",
                scientific_name=name,
                accepted_taxon_key=key,
            ),
        ),
        taxonomy_source="gbif",
        taxonomy_version="backbone-2026-07",
        taxonomy_fingerprint=_sha("gbif-backbone-2026-07"),
    )


def _prompt_result(key: str, name: str, score: float):
    ensemble = build_taxonomic_prompt_ensemble(
        context=_prompt_context(key, name),
        route="adult_field",
        life_stage="adult",
    )
    subset = build_prompt_subset_policy(
        ensemble,
        subset_id="accepted-name-only",
        visual_domain="field",
        prompt_kinds=("accepted_scientific_name",),
    )
    vector = (score, math.sqrt(1.0 - score * score))
    return pool_prompt_ensemble(
        ensemble=ensemble,
        image_embedding=(1.0, 0.0),
        text_embeddings={
            variant.variant_fingerprint: vector for variant in ensemble.variants
        },
        model_fingerprint=MODEL_FINGERPRINT,
        subset=subset,
        strategy=MAX_PROMPT_SIMILARITY,
    )


def _prompt_results():
    return (
        _prompt_result(TARGET, "Papilio demoleus", -0.60),
        _prompt_result(COMPETITOR, "Papilio polytes", 0.90),
        _prompt_result(THIRD, "Papilio machaon", 0.50),
    )


def _reference(
    key: str,
    name: str,
    *,
    centroid: float | None,
    nearest: float | None,
    top_three: float | None,
    top_five: float | None,
    local: float | None,
    global_: float | None,
    insufficient: bool = False,
) -> CandidateReferenceEvidence:
    selected_count = 0 if insufficient and centroid is None else 5
    local_count = 0 if selected_count == 0 else 3
    selected_ids = tuple(f"{key}:support:{index}" for index in range(selected_count))
    return CandidateReferenceEvidence(
        scoring_version=REFERENCE_EVIDENCE_SCORING_VERSION,
        query_id="photo:001:raw",
        accepted_taxon_key=key,
        scientific_name=name,
        route="adult_field",
        visual_input_kind="raw_full_image",
        geo_cluster_id="cluster-au-1",
        prototype_method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
        balanced_sampling_seed=42,
        fixed_reference_count=5,
        support_count=selected_count,
        usable_support_count=selected_count,
        local_support_count=local_count,
        selected_support_count=selected_count,
        selected_local_support_count=local_count,
        selected_reference_observation_ids=selected_ids,
        nearest_reference_observation_id=(
            selected_ids[0] if nearest is not None else None
        ),
        nearest_support_similarity=nearest,
        mean_top_three_similarity=top_three,
        mean_top_five_similarity=top_five,
        centroid_similarity=centroid,
        local_cluster_prototype_similarity=local,
        global_prototype_similarity=global_,
        distance_to_nearest_independent_observation=(
            1.0 - nearest if nearest is not None else None
        ),
        insufficient_support=insufficient,
        insufficient_support_reasons=(
            (
                "no_route_support",
                "fewer_than_balanced_reference_count",
                "fewer_than_three_independent_observations",
                "fewer_than_five_independent_observations",
            )
            if insufficient
            else ()
        ),
        local_support_available=local_count > 0,
        local_prototype_available=local is not None,
        global_prototype_available=global_ is not None,
        query_embedding_norm=1.0,
        centering_fingerprint=None,
        model_fingerprint=MODEL_FINGERPRINT,
        reference_embedding_fingerprint=REFERENCE_EMBEDDING_FINGERPRINT,
        reference_prototype_fingerprint=REFERENCE_PROTOTYPE_FINGERPRINT,
        support_manifest_fingerprint=SUPPORT_MANIFEST_FINGERPRINT,
    )


def _reference_results():
    return (
        _reference(
            TARGET,
            "Papilio demoleus",
            centroid=0.86,
            nearest=0.91,
            top_three=0.89,
            top_five=0.87,
            local=0.88,
            global_=0.84,
        ),
        _reference(
            COMPETITOR,
            "Papilio polytes",
            centroid=0.72,
            nearest=0.79,
            top_three=0.75,
            top_five=0.73,
            local=0.71,
            global_=0.70,
        ),
        _reference(
            THIRD,
            "Papilio machaon",
            centroid=0.40,
            nearest=0.47,
            top_three=0.44,
            top_five=0.42,
            local=0.39,
            global_=0.38,
        ),
    )


def _structured_results():
    return (
        CandidateStructuredEvidence(
            accepted_taxon_key=TARGET,
            candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
            geo_cluster_id="cluster-au-1",
            geographic_scope="regional",
            geographic_evidence_score=0.80,
            occurrence_support=12,
            route="adult_field",
            life_stage="adult",
            visual_domain="field",
            life_stage_compatible=True,
            visual_domain_compatible=True,
            source_versions=("regional-candidate-species-v1.0.0",),
        ),
        CandidateStructuredEvidence(
            accepted_taxon_key=COMPETITOR,
            candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
            geo_cluster_id="cluster-au-1",
            geographic_scope="regional",
            geographic_evidence_score=0.64,
            occurrence_support=8,
            route="adult_field",
            life_stage="adult",
            visual_domain="field",
            life_stage_compatible=True,
            visual_domain_compatible=True,
            source_versions=("regional-candidate-species-v1.0.0",),
        ),
        CandidateStructuredEvidence(
            accepted_taxon_key=THIRD,
            candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
            geo_cluster_id="cluster-au-1",
            geographic_scope="nonregional",
            geographic_evidence_score=None,
            occurrence_support=0,
            route="adult_field",
            life_stage="adult",
            visual_domain="field",
            life_stage_compatible=True,
            visual_domain_compatible=True,
            source_versions=("regional-candidate-species-v1.0.0",),
        ),
    )


def _frozen_classifier(
    *,
    task: str,
    class_labels: tuple[str, ...],
    classifier_fingerprint: str,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> FrozenLinearClassifier:
    layout = classifier_feature_layout(EMBEDDING_ONLY_FEATURE_SET, 2)
    return FrozenLinearClassifier(
        classifier_version=CLASSIFIER_VERSION,
        classifier_fingerprint=classifier_fingerprint,
        model_name=LINEAR_SVC_EMBEDDING_MODEL,
        estimator_family=ESTIMATOR_LINEAR_SVC,
        target_task=task,
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        class_labels=class_labels,
        feature_layout=layout,
        feature_schema_fingerprint=_sha("feature-schema"),
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=_sha("preprocessing"),
        reference_bank_version="reference-bank-v1",
        reference_bank_fingerprint=_sha("reference-bank"),
        training_data_fingerprint=_sha(f"training:{task}"),
        support_manifest_fingerprint=_sha("support-manifest"),
        reference_embedding_fingerprint=_sha("reference-embeddings"),
        reference_prototype_fingerprint=_sha("reference-prototypes"),
        candidate_set_fingerprint=_sha("candidate-set"),
        probability_calibrated=False,
        coefficients=coefficients,
        intercepts=intercepts,
        class_indices=np.arange(len(class_labels), dtype=np.int64),
        continuous_imputer_statistics=np.asarray([], dtype=np.float64),
        continuous_scaler_mean=np.asarray([], dtype=np.float64),
        continuous_scaler_scale=np.asarray([], dtype=np.float64),
        continuous_scaler_variance=np.asarray([], dtype=np.float64),
    )


def _classifier_inferences(*, raw_embedding: tuple[float, float] = (1.0, 0.0)):
    return _classifier_inferences_for_candidates(raw_embedding=raw_embedding)


def _classifier_inferences_for_candidates(
    *,
    raw_embedding: tuple[float, float] = (1.0, 0.0),
    regional_class_labels: tuple[str, ...] = (TARGET, COMPETITOR, THIRD),
    regional_intercepts: tuple[float, ...] = (0.20, 1.20, -0.50),
):
    target_classifier_fingerprint = _sha("target-classifier")
    target_calibration_fingerprint = _sha("target-calibrator")
    target_classifier = _frozen_classifier(
        task="binary_target_verifier",
        class_labels=(NON_TARGET_CLASS_LABEL, TARGET),
        classifier_fingerprint=target_classifier_fingerprint,
        coefficients=np.asarray([[2.0, 0.0]], dtype=np.float64),
        intercepts=np.asarray([0.0], dtype=np.float64),
    )
    target_calibrator = FrozenProbabilityCalibrator(
        calibration_fingerprint=target_calibration_fingerprint,
        classifier_fingerprint=target_classifier_fingerprint,
        split_fingerprint=_sha("target-calibration-split"),
        target_task="binary_target_verifier",
        route="adult_field",
        method="sigmoid",
        class_labels=(NON_TARGET_CLASS_LABEL, TARGET),
        positive_class_label=TARGET,
        scalar_parameters={"slope": 1.0, "intercept": 0.0},
        array_parameters={},
    )
    regional_classifier_fingerprint = _sha("regional-classifier")
    regional_classifier = _frozen_classifier(
        task="regional_multiclass",
        class_labels=regional_class_labels,
        classifier_fingerprint=regional_classifier_fingerprint,
        coefficients=np.zeros((len(regional_class_labels), 2), dtype=np.float64),
        intercepts=np.asarray(regional_intercepts, dtype=np.float64),
    )
    regional_calibrator = FrozenProbabilityCalibrator(
        calibration_fingerprint=_sha("regional-calibrator"),
        classifier_fingerprint=regional_classifier_fingerprint,
        split_fingerprint=_sha("regional-calibration-split"),
        target_task="regional_multiclass",
        route="adult_field",
        method="temperature",
        class_labels=regional_class_labels,
        positive_class_label=None,
        scalar_parameters={"inverse_temperature": 1.0},
        array_parameters={},
    )
    raw = np.asarray([raw_embedding], dtype=np.float64)
    target = score_frozen_classifier(
        classifier=target_classifier,
        calibrator=target_calibrator,
        raw_features=raw,
        query_id="photo:001:raw",
        feature_input_fingerprint=_sha("query-feature-row"),
    )
    regional = score_frozen_classifier(
        classifier=regional_classifier,
        calibrator=regional_calibrator,
        raw_features=raw,
        query_id="photo:001:raw",
        feature_input_fingerprint=_sha("query-feature-row"),
    )
    return target, regional


def _policy(target_inference):
    samples = tuple(
        DecisionPolicyCalibrationSample(
            sample_id=f"sample-{index}",
            leakage_component_id=f"component-{index}",
            dataset_split="calibration",
            true_target=is_target,
            calibrated_target_probability=probability,
            competitor_margin=margin,
        )
        for index, (is_target, probability, margin) in enumerate(
            (
                (True, 0.80, 0.12),
                (True, 0.70, 0.10),
                (False, 0.75, 0.02),
                (False, 0.30, -0.10),
            )
        )
    )
    return fit_selective_decision_policy(
        samples,
        SelectiveDecisionPolicyConfig(
            target_task="binary_target_verifier",
            route="adult_field",
            target_precision_objective=1.0,
            model_fingerprint=MODEL_FINGERPRINT,
            classifier_fingerprint=target_inference.classifier_fingerprint,
            calibration_fingerprint=target_inference.calibration_fingerprint,
            split_fingerprint=target_inference.calibration_split_fingerprint,
        ),
    )


def _domain_negatives() -> tuple[DomainNegativeEvidence, ...]:
    return (
        DomainNegativeEvidence(
            evidence_id="pinned_specimen",
            outcome="pinned_specimen",
            reason="visual-domain:pinned_specimen",
            reference_score=None,
            reference_score_kind=None,
            score_contract_fingerprint=None,
            calibrated_probability=0.05,
            probability_task="visual_domain",
            classifier_fingerprint=_sha("domain-classifier"),
            calibrator_fingerprint=_sha("domain-calibrator"),
        ),
    )


def _fuse(
    *,
    plan=None,
    references=None,
    structured=None,
    prompts=None,
    reference_top_k=3,
    classifier_inferences=None,
):
    prompt_values = prompts or _prompt_results()
    target_inference, regional_inference = (
        classifier_inferences or _classifier_inferences()
    )
    return fuse_target_aware_species_evidence(
        plan=plan or _plan(),
        prompt_results=prompt_values,
        reference_results=references or _reference_results(),
        target_classifier=target_inference,
        regional_classifier=regional_inference,
        structured_evidence=structured or _structured_results(),
        domain_negatives=_domain_negatives(),
        decision_policy=_policy(target_inference),
        quality=TargetAwareFusionQuality(
            domain_negative_detected=False,
            out_of_distribution=False,
            visual_detail_sufficient=True,
        ),
        reference_top_k=reference_top_k,
    )


def _direct_text_result(plan, *, omit: str | None = None):
    class DirectTextScorer:
        def score(self, scoring_plan):
            scores = {
                item.scoring_class_id: 0.0
                for item in scoring_plan.scoring_classes
                if item.scoring_class_id != omit
            }
            scores.update(
                {
                    "family_diagnostic:Papilionidae": 0.1,
                    "family_diagnostic:Pieridae": 2.5,
                    "genus_diagnostic:Papilio": 1.8,
                    "genus_diagnostic:Graphium": 0.4,
                    "genus_diagnostic:Pieris": 0.2,
                }
            )
            if omit is not None:
                scores.pop(omit, None)
            return scores

    return score_target_aware_candidate_union(plan, DirectTextScorer())


def test_unassigned_geotagged_plan_uses_global_fallback_policy() -> None:
    structured = tuple(
        replace(
            item,
            geo_cluster_id="unassigned_geo",
            geographic_scope="no_geo_global",
        )
        for item in _structured_results()
    )

    result = _fuse(
        plan=_plan(geo_cluster_id="unassigned_geo"),
        references=tuple(
            replace(item, geo_cluster_id="unassigned_geo")
            for item in _reference_results()
        ),
        structured=structured,
    )

    assert result.decision.classification_decision == "no_geo_global_fallback"
    assert result.decision.abstained is True
    assert result.decision.abstention_reason == "no_geo_global_fallback"


def test_frozen_classifier_keeps_raw_decisions_separate_from_calibration() -> None:
    target, regional = _classifier_inferences()

    target_by_label = {item.class_label: item for item in target.class_scores}
    assert target_by_label[TARGET].decision_score == pytest.approx(2.0)
    assert target_by_label[NON_TARGET_CLASS_LABEL].decision_score == pytest.approx(-2.0)
    assert target_by_label[TARGET].calibrated_probability == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0))
    )
    assert target_by_label[TARGET].calibrated_probability != pytest.approx(
        target_by_label[TARGET].decision_score
    )
    assert sum(item.calibrated_probability for item in target.class_scores) == (
        pytest.approx(1.0)
    )
    assert regional.class_scores[1].decision_score == pytest.approx(1.20)
    assert regional.class_scores[1].calibrated_probability < 1.0


def test_frozen_classifier_rejects_nonunit_query_embedding() -> None:
    with pytest.raises(ValueError, match="query embedding must be unit-normalized"):
        _classifier_inferences(raw_embedding=(2.0, 0.0))


def test_support_image_and_structured_fusion_preserves_every_species_channel() -> None:
    result = _fuse()
    scores = {item.accepted_taxon_key: item for item in result.species_scores}
    target = scores[TARGET]

    assert result.fusion_version == TARGET_AWARE_EVIDENCE_FUSION_VERSION
    assert len(scores) == 3
    assert target.text_ensemble_score == pytest.approx(-0.60)
    assert target.reference_centroid_score == pytest.approx(0.86)
    assert target.nearest_reference_score == pytest.approx(0.91)
    assert target.top_k_reference_score == pytest.approx(0.89)
    assert target.reference_top_k == 3
    assert target.local_prototype_score == pytest.approx(0.88)
    assert target.global_prototype_score == pytest.approx(0.84)
    assert target.classifier_decision_score == pytest.approx(2.0)
    assert target.calibrated_probability == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert target.classifier_task == "binary_target_verifier"
    assert target.geographic_scope == "regional"
    assert target.geographic_evidence_score == pytest.approx(0.80)
    assert target.occurrence_support == 12
    assert target.life_stage_compatible is True
    assert target.visual_domain_compatible is True
    assert target.best_competitor_accepted_taxon_key == COMPETITOR
    assert target.best_competitor_scientific_name == "Papilio polytes"
    assert target.competitor_margin == pytest.approx(0.14)
    assert target.nonmatch_score == pytest.approx(0.72)
    assert target.abstention_reason is None
    assert result.decision.classification_decision == "target_confirmed"
    assert result.nonmatch_score.best_known_competitor_accepted_taxon_key == COMPETITOR
    assert target.regional_rank == 2
    assert scores[COMPETITOR].classifier_task == "regional_multiclass"
    assert scores[COMPETITOR].regional_rank == 1
    assert result.fusion_fingerprint.startswith("sha256:")


def test_low_text_rank_cannot_prune_target_that_wins_reference_comparison() -> None:
    result = _fuse()
    target = next(item for item in result.species_scores if item.target_candidate)

    assert target.text_ensemble_score < min(
        item.text_ensemble_score
        for item in result.species_scores
        if not item.target_candidate
    )
    assert target.reference_centroid_score > max(
        item.reference_centroid_score
        for item in result.species_scores
        if not item.target_candidate
    )
    assert result.decision.classification_decision == "target_confirmed"


def test_regression_target_genus_rank_20_still_wins_reference_comparison() -> None:
    competitors = tuple(
        CandidateTaxon(
            scientific_name=f"Genus{index:02d} species",
            accepted_taxon_key=f"gbif:rank20-{index:02d}",
            family="Papilionidae",
            genus=f"Genus{index:02d}",
            candidate_reasons=("regional_same_family",),
            source_versions=("regional-candidate-species-v1.0.0",),
            candidate_priority=index,
        )
        for index in range(1, 20)
    )
    target = _candidate_set().species_candidates[0]
    species = (target, *competitors)
    fingerprint = _sha("rank-20-candidate-union")
    candidate_set = replace(
        _candidate_set(),
        candidate_set_id="regional:rank-20-regression",
        family_candidates=species,
        genus_candidates=species,
        species_candidates=species,
        candidate_set_fingerprint=fingerprint,
    )
    plan = build_target_aware_scoring_plan(
        candidate_set,
        known_negative_classes=(
            TargetAwareAuxiliaryClass(
                class_id="visual_artifact",
                display_name="visual artifact",
                source_versions=("negative-manifest-v1",),
            ),
        ),
        visual_domain_classes=(
            TargetAwareAuxiliaryClass(
                class_id="adult_field",
                display_name="live adult butterfly in the field",
                source_versions=("reference-domain-v1",),
            ),
        ),
    )

    class RankTwentyTextScorer:
        def score(self, scoring_plan):
            scores = {
                item.scoring_class_id: 0.0 for item in scoring_plan.scoring_classes
            }
            for item in scoring_plan.genus_diagnostic_classes:
                scores[item.scoring_class_id] = (
                    -20.0
                    if item.class_id == "Papilio"
                    else float(20 - int(item.class_id.removeprefix("Genus")))
                )
            return scores

    direct = score_target_aware_candidate_union(plan, RankTwentyTextScorer())
    target_genus = next(
        item for item in direct.genus_diagnostics if item.class_id == "Papilio"
    )
    assert target_genus.rank == 20

    prompts = []
    references = []
    structured = []
    labels = []
    regional_intercepts = []
    for index, candidate in enumerate(species):
        key = str(candidate.accepted_taxon_key)
        labels.append(key)
        regional_intercepts.append(3.0 if candidate.target_candidate else -2.0)
        prompts.append(
            _prompt_result(
                key,
                candidate.scientific_name,
                -0.8 if candidate.target_candidate else 0.5,
            )
        )
        centroid = 0.90 if candidate.target_candidate else 0.40 - index * 0.005
        references.append(
            _reference(
                key,
                candidate.scientific_name,
                centroid=centroid,
                nearest=centroid + 0.05,
                top_three=centroid + 0.03,
                top_five=centroid + 0.02,
                local=centroid + 0.01,
                global_=centroid,
            )
        )
        structured.append(
            CandidateStructuredEvidence(
                accepted_taxon_key=key,
                candidate_set_fingerprint=fingerprint,
                geo_cluster_id="cluster-au-1",
                geographic_scope="regional",
                geographic_evidence_score=0.8,
                occurrence_support=5,
                route="adult_field",
                life_stage="adult",
                visual_domain="field",
                life_stage_compatible=True,
                visual_domain_compatible=True,
                source_versions=("regional-candidate-species-v1.0.0",),
            )
        )
    inferences = _classifier_inferences_for_candidates(
        regional_class_labels=tuple(labels),
        regional_intercepts=tuple(regional_intercepts),
    )
    fusion = fuse_target_aware_species_evidence(
        plan=plan,
        prompt_results=tuple(prompts),
        reference_results=tuple(references),
        target_classifier=inferences[0],
        regional_classifier=inferences[1],
        structured_evidence=tuple(structured),
        domain_negatives=_domain_negatives(),
        decision_policy=_policy(inferences[0]),
        quality=TargetAwareFusionQuality(
            domain_negative_detected=False,
            out_of_distribution=False,
            visual_detail_sufficient=True,
        ),
        reference_top_k=3,
    )
    target_score = next(item for item in fusion.species_scores if item.target_candidate)

    assert len(fusion.species_scores) == 20
    assert target_score.reference_centroid_score == pytest.approx(0.90)
    assert target_score.reference_centroid_score > max(
        item.reference_centroid_score
        for item in fusion.species_scores
        if not item.target_candidate
    )
    assert fusion.decision.classification_decision == "target_confirmed"


def test_regression_strong_regional_competitor_beats_target() -> None:
    inferences = _classifier_inferences_for_candidates(
        regional_intercepts=(0.0, 6.0, -5.0)
    )

    result = _fuse(classifier_inferences=inferences)

    assert result.nonmatch_score.nonmatch_margin is not None
    assert result.nonmatch_score.nonmatch_margin < 0.0
    assert result.decision.classification_decision == "known_regional_competitor"
    assert result.decision.winning_nonmatch_accepted_taxon_key == COMPETITOR
    assert result.decision.abstained is False


def test_regression_strong_target_with_weak_geography_routes_to_review() -> None:
    structured = list(_structured_results())
    structured[0] = replace(
        structured[0],
        geographic_scope="nonregional",
        geographic_evidence_score=None,
        occurrence_support=0,
    )

    result = _fuse(structured=tuple(structured))

    assert len(result.species_scores) == 3
    assert result.decision.classification_decision == "target_probable_review"
    assert result.decision.abstained is True
    assert result.decision.abstention_reason == "weak_geographic_evidence"
    assert result.decision.review_priority == "high"


def test_missing_candidate_evidence_fails_closed_instead_of_shortening_union() -> None:
    with pytest.raises(ValueError, match="prompt_results.*missing.*gbif:5139051"):
        _fuse(prompts=_prompt_results()[:-1])


def test_reference_contract_mismatch_is_rejected_before_fusion() -> None:
    references = list(_reference_results())
    references[1] = replace(
        references[1],
        reference_embedding_fingerprint=_sha("another-reference-bank"),
    )

    with pytest.raises(ValueError, match="reference_results mix scoring contracts"):
        _fuse(references=tuple(references))


def test_structured_evidence_is_bound_to_candidate_set_identity() -> None:
    structured = list(_structured_results())
    structured[0] = replace(
        structured[0],
        candidate_set_fingerprint=_sha("another-candidate-set"),
    )

    with pytest.raises(ValueError, match="structured evidence candidate set"):
        _fuse(structured=tuple(structured))


def test_life_stage_incompatibility_abstains_without_deleting_species() -> None:
    structured = list(_structured_results())
    structured[1] = replace(structured[1], life_stage_compatible=False)

    result = _fuse(structured=tuple(structured))

    assert len(result.species_scores) == 3
    assert result.decision.classification_decision == "abstain"
    assert result.decision.abstention_reason == "incompatible_route"
    assert {item.abstention_reason for item in result.species_scores} == {
        "incompatible_route"
    }


def test_missing_reference_is_not_replaced_by_a_second_text_score() -> None:
    references = list(_reference_results())
    references[0] = _reference(
        TARGET,
        "Papilio demoleus",
        centroid=None,
        nearest=None,
        top_three=None,
        top_five=None,
        local=None,
        global_=None,
        insufficient=True,
    )
    prompts = list(_prompt_results())
    prompts[0] = _prompt_result(TARGET, "Papilio demoleus", 0.99)

    result = _fuse(references=tuple(references), prompts=tuple(prompts))
    target = next(item for item in result.species_scores if item.target_candidate)

    assert target.text_ensemble_score == pytest.approx(0.99)
    assert target.reference_centroid_score is None
    assert target.nearest_reference_score is None
    assert target.top_k_reference_score is None
    assert result.decision.classification_decision == "insufficient_reference_coverage"
    assert result.decision.abstention_reason == "insufficient_reference_coverage"


def test_reference_top_k_policy_selects_exact_fixed_k_without_fallback() -> None:
    result = _fuse(reference_top_k=5)
    target = next(item for item in result.species_scores if item.target_candidate)

    assert result.reference_top_k == 5
    assert target.reference_top_k == 5
    assert target.top_k_reference_score == pytest.approx(0.87)


def test_fusion_fingerprint_detects_evidence_channel_tampering() -> None:
    result = _fuse()
    changed_target = replace(
        result.species_scores[0],
        reference_centroid_score=0.99,
    )
    tampered = replace(
        result,
        species_scores=(changed_target, *result.species_scores[1:]),
    )

    with pytest.raises(ValueError, match="fusion fingerprint"):
        target_aware_fusion_result_payload(tampered)


def test_taxonomic_evidence_is_derived_after_complete_species_scoring() -> None:
    plan = _hierarchy_plan()
    fusion = _fuse(plan=plan)
    direct = _direct_text_result(plan)

    result = derive_taxonomic_evidence(
        plan=plan,
        fusion_result=fusion,
        direct_text_result=direct,
    )

    species = {item.accepted_taxon_key: item for item in fusion.species_scores}
    families = {item.taxon_name: item for item in result.family_evidence}
    genera = {item.taxon_name: item for item in result.genus_evidence}
    papilionidae = families["Papilionidae"]
    papilio = genera["Papilio"]

    assert result.species_candidate_keys == tuple(
        item.accepted_taxon_key for item in plan.species_classes
    )
    assert result.species_candidate_set_modified is False
    assert result.species_probability_sum == pytest.approx(1.0)
    assert sum(item.derived_probability for item in result.family_evidence) == (
        pytest.approx(1.0)
    )
    assert sum(item.derived_probability for item in result.genus_evidence) == (
        pytest.approx(1.0)
    )
    assert papilionidae.derived_probability == pytest.approx(
        species[TARGET].regional_calibrated_probability
        + species[COMPETITOR].regional_calibrated_probability
    )
    assert {item.accepted_taxon_key for item in papilionidae.member_species} == {
        TARGET,
        COMPETITOR,
    }
    target_member = next(
        item for item in papilio.member_species if item.accepted_taxon_key == TARGET
    )
    assert target_member.calibrated_probability == pytest.approx(
        species[TARGET].regional_calibrated_probability
    )
    assert target_member.calibrated_probability != pytest.approx(
        species[TARGET].calibrated_probability
    )
    assert papilionidae.probability_kind == DERIVED_PARENT_PROBABILITY_KIND
    assert papilionidae.direct_text_score_kind == DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND
    assert papilionidae.direct_text_decision_score == pytest.approx(0.1)
    assert families["Pieridae"].direct_text_decision_score == pytest.approx(2.5)
    assert result.derived_family_top1 == "Papilionidae"
    assert result.direct_text_family_top1 == "Pieridae"
    assert result.derived_genus_top1 == "Graphium"
    assert result.direct_text_genus_top1 == "Papilio"
    assert result.taxonomic_inconsistency is True
    assert result.inconsistency_codes == (
        "family_top1_disagreement",
        "genus_top1_disagreement",
    )
    assert taxonomic_evidence_result_payload(result)["result_fingerprint"] == (
        result.result_fingerprint
    )


def test_taxonomic_evidence_is_order_independent_and_does_not_prune_species() -> None:
    plan = _hierarchy_plan()
    fusion = _fuse(plan=plan)
    direct = _direct_text_result(plan)

    first = derive_taxonomic_evidence(
        plan=plan,
        fusion_result=fusion,
        direct_text_result=direct,
    )
    second = derive_taxonomic_evidence(
        plan=plan,
        fusion_result=fusion,
        direct_text_result=replace(
            direct,
            scored_classes=tuple(reversed(direct.scored_classes)),
        ),
    )

    assert first.result_fingerprint == second.result_fingerprint
    assert len(first.species_candidate_keys) == len(fusion.species_scores)
    assert set(first.species_candidate_keys) == {
        item.accepted_taxon_key for item in fusion.species_scores
    }


def test_taxonomic_evidence_rejects_identity_and_parent_coverage_mismatches() -> None:
    plan = _hierarchy_plan()
    fusion = _fuse(plan=plan)
    direct = _direct_text_result(plan)

    with pytest.raises(ValueError, match="candidate-set identity"):
        derive_taxonomic_evidence(
            plan=plan,
            fusion_result=fusion,
            direct_text_result=replace(
                direct,
                candidate_set_fingerprint=_sha("wrong-candidate-set"),
            ),
        )

    incomplete = replace(
        direct,
        scored_classes=tuple(
            item
            for item in direct.scored_classes
            if item.scoring_class_id != "genus_diagnostic:Graphium"
        ),
    )
    with pytest.raises(ValueError, match="genus diagnostic coverage"):
        derive_taxonomic_evidence(
            plan=plan,
            fusion_result=fusion,
            direct_text_result=incomplete,
        )


def test_taxonomic_evidence_rejects_incomplete_regional_probability_mass() -> None:
    plan = _hierarchy_plan()
    fusion = _fuse(plan=plan)
    changed_species = replace(
        fusion.species_scores[0],
        regional_calibrated_probability=(
            fusion.species_scores[0].regional_calibrated_probability - 0.05
        ),
    )
    changed = replace(
        fusion,
        species_scores=(changed_species, *fusion.species_scores[1:]),
    )
    values = {
        name: getattr(changed, name)
        for name in changed.__dataclass_fields__
        if name != "fusion_fingerprint"
    }
    forged = replace(
        changed,
        fusion_fingerprint=canonical_semantic_fingerprint(
            target_fusion_module._fusion_result_semantics(values)  # noqa: SLF001
        ),
    )

    with pytest.raises(ValueError, match="probabilities must sum to one"):
        derive_taxonomic_evidence(
            plan=plan,
            fusion_result=forged,
            direct_text_result=_direct_text_result(plan),
        )
