"""Support-image and calibrated structured-evidence fusion for target screening."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field as dataclass_field
from math import isfinite
import re

from biominer.bioclip.prompt_pooling import (
    PROMPT_SCORE_KIND,
    PromptPoolingResult,
    prompt_pooling_result_payload,
)
from biominer.bioclip.reference_scoring import (
    REFERENCE_EVIDENCE_SCORING_VERSION,
    CandidateReferenceEvidence,
)
from biominer.bioclip.target_aware_scoring import (
    TargetAwareScoringClass,
    TargetAwareScoringPlan,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.ml.calibration import FrozenProbabilityCalibrator
from biominer.ml.classifiers import NON_TARGET_CLASS_LABEL
from biominer.ml.decision_policy import (
    SelectiveDecision,
    SelectiveDecisionPolicy,
    SelectivePredictionEvidence,
    apply_selective_decision_policy,
)
from biominer.ml.nonmatch import (
    GEOGRAPHIC_SCOPES,
    REFERENCE_SCORE_KIND_COSINE_SIMILARITY,
    CalibratedNonTargetEvidence,
    CandidateNonMatchEvidence,
    DomainNegativeEvidence,
    ExplicitNonMatchScore,
    NonMatchScoringRequest,
    score_nonmatch_evidence,
)
from biominer.ml.persistence import CLASSIFIER_VERSION, FrozenLinearClassifier


TARGET_AWARE_EVIDENCE_FUSION_VERSION = (
    "target-aware-support-image-structured-fusion-v1.0.0"
)
FROZEN_CLASSIFIER_INFERENCE_VERSION = "frozen-linear-calibrated-inference-v1.0.0"
REFERENCE_SCORE_CONTRACT_VERSION = "target-aware-centroid-comparison-v1.0.0"
STRUCTURED_EVIDENCE_VERSION = "target-aware-candidate-structured-evidence-v1.0.0"
SUPPORTED_REFERENCE_TOP_K = frozenset({3, 5})

_TARGET_VERIFIER_TASKS = frozenset({"binary_target_verifier", "larval_target_verifier"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CandidateStructuredEvidence:
    """Versioned geography and route compatibility for one candidate species."""

    accepted_taxon_key: str
    candidate_set_fingerprint: str
    geo_cluster_id: str
    geographic_scope: str
    geographic_evidence_score: float | None
    occurrence_support: int
    route: str
    life_stage: str
    visual_domain: str
    life_stage_compatible: bool
    visual_domain_compatible: bool
    source_versions: tuple[str, ...]
    evidence_version: str = STRUCTURED_EVIDENCE_VERSION
    evidence_fingerprint: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_taxon_key",
            _required_text(
                self.accepted_taxon_key,
                field="accepted_taxon_key",
            ),
        )
        object.__setattr__(
            self,
            "candidate_set_fingerprint",
            _sha256(
                self.candidate_set_fingerprint,
                field="candidate_set_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "geo_cluster_id",
            _required_text(self.geo_cluster_id, field="geo_cluster_id"),
        )
        scope = _required_text(self.geographic_scope, field="geographic_scope")
        if scope not in GEOGRAPHIC_SCOPES:
            raise ValueError(f"unsupported geographic_scope: {scope}")
        object.__setattr__(self, "geographic_scope", scope)
        object.__setattr__(
            self,
            "geographic_evidence_score",
            _optional_unit_interval(
                self.geographic_evidence_score,
                field="geographic_evidence_score",
            ),
        )
        object.__setattr__(
            self,
            "occurrence_support",
            _nonnegative_integer(
                self.occurrence_support,
                field="occurrence_support",
            ),
        )
        for field_name in ("route", "life_stage", "visual_domain"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field=field_name),
            )
        _boolean(self.life_stage_compatible, field="life_stage_compatible")
        _boolean(self.visual_domain_compatible, field="visual_domain_compatible")
        source_versions = _unique_text_tuple(
            self.source_versions,
            field="source_versions",
        )
        if not source_versions:
            raise ValueError("source_versions must not be empty")
        object.__setattr__(self, "source_versions", source_versions)
        if self.evidence_version != STRUCTURED_EVIDENCE_VERSION:
            raise ValueError("structured evidence version is incompatible")
        object.__setattr__(
            self,
            "evidence_fingerprint",
            canonical_semantic_fingerprint(
                {
                    "evidence_version": self.evidence_version,
                    "accepted_taxon_key": self.accepted_taxon_key,
                    "candidate_set_fingerprint": self.candidate_set_fingerprint,
                    "geo_cluster_id": self.geo_cluster_id,
                    "geographic_scope": self.geographic_scope,
                    "geographic_evidence_score": self.geographic_evidence_score,
                    "occurrence_support": self.occurrence_support,
                    "route": self.route,
                    "life_stage": self.life_stage,
                    "visual_domain": self.visual_domain,
                    "life_stage_compatible": self.life_stage_compatible,
                    "visual_domain_compatible": self.visual_domain_compatible,
                    "source_versions": list(self.source_versions),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetAwareFusionQuality:
    """Runtime quality states not derivable from the supplied evidence artifacts."""

    domain_negative_detected: bool
    out_of_distribution: bool
    visual_detail_sufficient: bool

    def __post_init__(self) -> None:
        _boolean(
            self.domain_negative_detected,
            field="domain_negative_detected",
        )
        _boolean(self.out_of_distribution, field="out_of_distribution")
        _boolean(
            self.visual_detail_sufficient,
            field="visual_detail_sufficient",
        )


@dataclass(frozen=True, slots=True)
class ClassifierClassScore:
    """Raw conventional-classifier decision and separately calibrated probability."""

    class_label: str
    decision_score: float
    calibrated_probability: float


@dataclass(frozen=True, slots=True)
class FrozenClassifierInference:
    """One-query inference from an immutable classifier/calibrator pair."""

    inference_version: str
    target_task: str
    target_accepted_taxon_key: str
    query_id: str
    route: str
    class_scores: tuple[ClassifierClassScore, ...]
    classifier_version: str
    classifier_fingerprint: str
    calibration_fingerprint: str
    calibration_split_fingerprint: str
    model_fingerprint: str
    embedding_dimension: int
    feature_schema_fingerprint: str
    feature_layout_fingerprint: str
    feature_input_fingerprint: str
    image_embedding_fingerprint: str
    preprocessing_fingerprint: str
    reference_bank_version: str
    reference_bank_fingerprint: str
    training_data_fingerprint: str
    inference_fingerprint: str

    def class_score(self, class_label: str) -> ClassifierClassScore:
        label = _required_text(class_label, field="class_label")
        try:
            return next(item for item in self.class_scores if item.class_label == label)
        except StopIteration as exc:
            raise KeyError(
                f"classifier inference has no class label {label!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class TargetAwareSpeciesFusionScore:
    """Complete evidence channels and shared decision context for one species."""

    accepted_taxon_key: str
    scientific_name: str
    target_candidate: bool
    candidate_priority: int
    candidate_reasons: tuple[str, ...]
    text_ensemble_score: float
    reference_centroid_score: float | None
    nearest_reference_score: float | None
    top_k_reference_score: float | None
    reference_top_k: int
    local_prototype_score: float | None
    global_prototype_score: float | None
    classifier_task: str
    classifier_decision_score: float
    calibrated_probability: float
    regional_classifier_decision_score: float
    regional_calibrated_probability: float
    regional_rank: int
    geographic_scope: str
    geographic_evidence_score: float | None
    occurrence_support: int
    life_stage: str
    visual_domain: str
    life_stage_compatible: bool
    visual_domain_compatible: bool
    best_competitor_accepted_taxon_key: str | None
    best_competitor_scientific_name: str | None
    competitor_margin: float | None
    nonmatch_score: float | None
    abstention_reason: str | None
    prompt_result_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    support_manifest_fingerprint: str
    classifier_fingerprint: str
    calibration_fingerprint: str
    structured_evidence_fingerprint: str
    candidate_set_fingerprint: str


@dataclass(frozen=True, slots=True)
class TargetAwareFusionResult:
    """Complete candidate-union evidence plus non-match and selective decision."""

    fusion_version: str
    classification_mode: str
    query_id: str
    route: str
    life_stage: str
    visual_domain: str
    candidate_set_id: str
    candidate_set_fingerprint: str
    geo_cluster_id: str
    target_accepted_taxon_key: str
    reference_top_k: int
    reference_score_contract_fingerprint: str
    reference_coverage_sufficient: bool
    species_scores: tuple[TargetAwareSpeciesFusionScore, ...]
    nonmatch_score: ExplicitNonMatchScore
    decision: SelectiveDecision
    target_classifier_inference_fingerprint: str
    regional_classifier_inference_fingerprint: str
    fusion_fingerprint: str


def score_frozen_classifier(
    *,
    classifier: FrozenLinearClassifier,
    calibrator: FrozenProbabilityCalibrator,
    raw_features: object,
    query_id: str,
    feature_input_fingerprint: str,
) -> FrozenClassifierInference:
    """Run one frozen linear classifier and its matching fitted calibrator."""

    if not isinstance(classifier, FrozenLinearClassifier):
        raise TypeError("classifier must be a FrozenLinearClassifier")
    if not isinstance(calibrator, FrozenProbabilityCalibrator):
        raise TypeError("calibrator must be a FrozenProbabilityCalibrator")
    if classifier.classifier_version != CLASSIFIER_VERSION:
        raise ValueError("classifier version is incompatible with target-aware fusion")
    classifier_fingerprint = _sha256(
        classifier.classifier_fingerprint,
        field="classifier.classifier_fingerprint",
    )
    calibration_fingerprint = _sha256(
        calibrator.calibration_fingerprint,
        field="calibrator.calibration_fingerprint",
    )
    calibration_split_fingerprint = _sha256(
        calibrator.split_fingerprint,
        field="calibrator.split_fingerprint",
    )
    if calibrator.classifier_fingerprint != classifier_fingerprint:
        raise ValueError("calibrator does not belong to the supplied classifier")
    if calibrator.target_task != classifier.target_task:
        raise ValueError("classifier and calibrator target tasks differ")
    if calibrator.route != classifier.route:
        raise ValueError("classifier and calibrator routes differ")
    if tuple(calibrator.class_labels) != tuple(classifier.class_labels):
        raise ValueError("classifier and calibrator class order differs")
    if classifier.probability_calibrated:
        raise ValueError(
            "frozen classifier decision outputs must remain explicitly uncalibrated"
        )
    labels = tuple(
        _required_text(value, field="classifier.class_label")
        for value in classifier.class_labels
    )
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("classifier class labels must be unique and complete")
    if classifier.target_task in _TARGET_VERIFIER_TASKS and set(labels) != {
        NON_TARGET_CLASS_LABEL,
        classifier.target_accepted_taxon_key,
    }:
        raise ValueError(
            "target verifier must contain exactly target and __non_target__ labels"
        )
    if (
        classifier.target_task == "regional_multiclass"
        and classifier.target_accepted_taxon_key not in labels
    ):
        raise ValueError("regional classifier does not contain its target species")
    model_fingerprint = _sha256(
        classifier.model_fingerprint,
        field="classifier.model_fingerprint",
    )
    feature_schema_fingerprint = _sha256(
        classifier.feature_schema_fingerprint,
        field="classifier.feature_schema_fingerprint",
    )
    feature_layout_fingerprint = _sha256(
        classifier.feature_layout.fingerprint,
        field="classifier.feature_layout.fingerprint",
    )
    feature_fingerprint = _sha256(
        feature_input_fingerprint,
        field="feature_input_fingerprint",
    )
    preprocessing_fingerprint = _sha256(
        classifier.preprocessing_fingerprint,
        field="classifier.preprocessing_fingerprint",
    )
    reference_bank_fingerprint = _sha256(
        classifier.reference_bank_fingerprint,
        field="classifier.reference_bank_fingerprint",
    )
    training_data_fingerprint = _sha256(
        classifier.training_data_fingerprint,
        field="classifier.training_data_fingerprint",
    )

    transformed_features = classifier.transform_features(raw_features)
    if transformed_features.ndim != 2 or transformed_features.shape[0] != 1:
        raise ValueError("target-aware inference requires exactly one feature row")
    embedding = transformed_features[
        0,
        list(classifier.feature_layout.embedding_column_indices),
    ]
    embedding_norm = float(sum(float(value) ** 2 for value in embedding) ** 0.5)
    if abs(embedding_norm - 1.0) > 1e-5:
        raise ValueError("classifier query embedding must be unit-normalized")
    image_fingerprint = canonical_semantic_fingerprint(
        {
            "model_fingerprint": model_fingerprint,
            "normalized_image_embedding": [float(value) for value in embedding],
        }
    )

    raw_scores = classifier.decision_function(raw_features)
    if len(labels) == 2:
        if raw_scores.ndim != 1 or tuple(raw_scores.shape) != (1,):
            raise ValueError(
                "binary target-aware inference requires exactly one feature row"
            )
        margin = _finite_number(raw_scores[0], field="binary decision margin")
        decision_scores = (-margin, margin)
    else:
        if raw_scores.ndim != 2 or tuple(raw_scores.shape) != (1, len(labels)):
            raise ValueError(
                "multiclass target-aware inference requires exactly one feature row"
            )
        decision_scores = tuple(
            _finite_number(value, field="multiclass decision score")
            for value in raw_scores[0]
        )
    probabilities = calibrator.predict_proba(raw_scores)
    if probabilities.ndim != 2 or tuple(probabilities.shape) != (1, len(labels)):
        raise ValueError("calibrator returned an invalid probability shape")
    probability_values = tuple(
        _unit_interval(value, field="calibrated_probability")
        for value in probabilities[0]
    )
    if abs(sum(probability_values) - 1.0) > 1e-9:
        raise ValueError("calibrated class probabilities must sum to one")
    class_scores = tuple(
        ClassifierClassScore(
            class_label=label,
            decision_score=decision_score,
            calibrated_probability=probability,
        )
        for label, decision_score, probability in zip(
            labels,
            decision_scores,
            probability_values,
            strict=True,
        )
    )
    values: dict[str, object] = {
        "inference_version": FROZEN_CLASSIFIER_INFERENCE_VERSION,
        "target_task": classifier.target_task,
        "target_accepted_taxon_key": classifier.target_accepted_taxon_key,
        "query_id": _required_text(query_id, field="query_id"),
        "route": classifier.route,
        "class_scores": class_scores,
        "classifier_version": classifier.classifier_version,
        "classifier_fingerprint": classifier_fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
        "calibration_split_fingerprint": calibration_split_fingerprint,
        "model_fingerprint": model_fingerprint,
        "embedding_dimension": classifier.feature_layout.embedding_dimension,
        "feature_schema_fingerprint": feature_schema_fingerprint,
        "feature_layout_fingerprint": feature_layout_fingerprint,
        "feature_input_fingerprint": feature_fingerprint,
        "image_embedding_fingerprint": image_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "reference_bank_version": _required_text(
            classifier.reference_bank_version,
            field="classifier.reference_bank_version",
        ),
        "reference_bank_fingerprint": reference_bank_fingerprint,
        "training_data_fingerprint": training_data_fingerprint,
    }
    fingerprint = canonical_semantic_fingerprint(
        _classifier_inference_semantics(values)
    )
    return FrozenClassifierInference(
        **values,
        inference_fingerprint=fingerprint,
    )


def fuse_target_aware_species_evidence(
    *,
    plan: TargetAwareScoringPlan,
    prompt_results: Sequence[PromptPoolingResult],
    reference_results: Sequence[CandidateReferenceEvidence],
    target_classifier: FrozenClassifierInference,
    regional_classifier: FrozenClassifierInference,
    structured_evidence: Sequence[CandidateStructuredEvidence],
    domain_negatives: Sequence[DomainNegativeEvidence],
    decision_policy: SelectiveDecisionPolicy,
    quality: TargetAwareFusionQuality,
    reference_top_k: int = 3,
) -> TargetAwareFusionResult:
    """Fuse complete species evidence without hierarchy or text-based pruning."""

    if not isinstance(plan, TargetAwareScoringPlan):
        raise TypeError("plan must be a TargetAwareScoringPlan")
    if len(plan.species_classes) < 2:
        raise ValueError("target-aware fusion requires at least one competitor species")
    if not isinstance(quality, TargetAwareFusionQuality):
        raise TypeError("quality must be TargetAwareFusionQuality")
    top_k = _reference_top_k(reference_top_k)
    expected = {str(item.accepted_taxon_key): item for item in plan.species_classes}
    if None in {item.accepted_taxon_key for item in plan.species_classes}:
        raise ValueError("target-aware species classes require accepted taxon keys")

    prompts = _index_prompt_results(prompt_results, expected=expected)
    references = _index_reference_results(reference_results, expected=expected)
    structured = _index_structured_evidence(structured_evidence, expected=expected)
    prompt_contract = _prompt_contract(prompts.values())
    reference_contract = _reference_contract(references.values())
    _validate_prompt_reference_plan_contracts(
        plan=plan,
        prompt_contract=prompt_contract,
        reference_contract=reference_contract,
        structured=structured,
    )
    target_inference = _validate_classifier_inference(target_classifier)
    regional_inference = _validate_classifier_inference(regional_classifier)
    _validate_classifier_contracts(
        plan=plan,
        expected_keys=set(expected),
        target=target_inference,
        regional=regional_inference,
        route=str(reference_contract["route"]),
        model_fingerprint=str(reference_contract["model_fingerprint"]),
        embedding_dimension=int(prompt_contract["embedding_dimension"]),
        query_id=str(reference_contract["query_id"]),
        image_embedding_fingerprint=str(prompt_contract["image_embedding_fingerprint"]),
        decision_policy=decision_policy,
    )
    domain_rows = tuple(domain_negatives)
    if not domain_rows:
        raise ValueError("target-aware fusion requires domain-negative evidence")
    if any(not isinstance(item, DomainNegativeEvidence) for item in domain_rows):
        raise TypeError("domain_negatives must contain DomainNegativeEvidence values")

    reference_score_contract = canonical_semantic_fingerprint(
        {
            "contract_version": REFERENCE_SCORE_CONTRACT_VERSION,
            "comparison_field": "centroid_similarity",
            **reference_contract,
        }
    )
    target_class_score = target_inference.class_score(plan.target_accepted_taxon_key)
    non_target_class_score = target_inference.class_score(NON_TARGET_CLASS_LABEL)
    regional_by_key = {
        item.class_label: item for item in regional_inference.class_scores
    }
    regional_ranks = {
        item.class_label: rank
        for rank, item in enumerate(
            sorted(
                regional_inference.class_scores,
                key=lambda value: (
                    -value.calibrated_probability,
                    value.class_label,
                ),
            ),
            start=1,
        )
    }

    candidate_nonmatch_rows = []
    for key, candidate in expected.items():
        reference = references[key]
        if candidate.target_candidate:
            classifier_score = target_class_score
            classifier_task = target_inference.target_task
            classifier_fingerprint = target_inference.classifier_fingerprint
            calibration_fingerprint = target_inference.calibration_fingerprint
        else:
            classifier_score = regional_by_key[key]
            classifier_task = regional_inference.target_task
            classifier_fingerprint = regional_inference.classifier_fingerprint
            calibration_fingerprint = regional_inference.calibration_fingerprint
        candidate_nonmatch_rows.append(
            CandidateNonMatchEvidence(
                evidence_id=f"species:{key}",
                accepted_taxon_key=key,
                scientific_name=candidate.display_name,
                geographic_scope=structured[key].geographic_scope,
                candidate_reasons=candidate.candidate_reasons,
                reference_score=reference.centroid_similarity,
                reference_score_kind=(
                    REFERENCE_SCORE_KIND_COSINE_SIMILARITY
                    if reference.centroid_similarity is not None
                    else None
                ),
                score_contract_fingerprint=(
                    reference_score_contract
                    if reference.centroid_similarity is not None
                    else None
                ),
                calibrated_probability=classifier_score.calibrated_probability,
                probability_task=classifier_task,
                classifier_fingerprint=classifier_fingerprint,
                calibrator_fingerprint=calibration_fingerprint,
            )
        )
    nonmatch = score_nonmatch_evidence(
        NonMatchScoringRequest(
            query_id=str(reference_contract["query_id"]),
            route=str(reference_contract["route"]),
            target_accepted_taxon_key=plan.target_accepted_taxon_key,
            candidate_set_fingerprint=plan.candidate_set_fingerprint,
            candidates=tuple(candidate_nonmatch_rows),
            domain_negatives=domain_rows,
            non_target_classifier=CalibratedNonTargetEvidence(
                evidence_id="target-verifier:non-target",
                calibrated_probability=(non_target_class_score.calibrated_probability),
                probability_task=target_inference.target_task,
                classifier_fingerprint=target_inference.classifier_fingerprint,
                calibrator_fingerprint=target_inference.calibration_fingerprint,
            ),
        )
    )
    top_k_field = _top_k_field(top_k)
    reference_coverage_sufficient = all(
        not item.insufficient_support
        and item.centroid_similarity is not None
        and getattr(item, top_k_field) is not None
        for item in references.values()
    )
    route_compatible = all(
        item.life_stage_compatible and item.visual_domain_compatible
        for item in structured.values()
    )
    decision = apply_selective_decision_policy(
        decision_policy,
        SelectivePredictionEvidence(
            nonmatch_score=nonmatch,
            model_fingerprint=target_inference.model_fingerprint,
            classifier_fingerprint=target_inference.classifier_fingerprint,
            calibration_fingerprint=target_inference.calibration_fingerprint,
            route_compatible=route_compatible,
            reference_coverage_sufficient=reference_coverage_sufficient,
            domain_negative_detected=quality.domain_negative_detected,
            out_of_distribution=quality.out_of_distribution,
            visual_detail_sufficient=quality.visual_detail_sufficient,
            no_geo_global_fallback=plan.geo_cluster_id == NO_GEO_CLUSTER_ID,
        ),
    )

    scores = []
    for candidate in sorted(
        plan.species_classes,
        key=lambda item: (int(item.candidate_priority or 0), item.class_id),
    ):
        key = str(candidate.accepted_taxon_key)
        prompt = prompts[key]
        reference = references[key]
        geography = structured[key]
        regional_score = regional_by_key[key]
        if candidate.target_candidate:
            primary_score = target_class_score
            primary_task = target_inference.target_task
            primary_classifier = target_inference.classifier_fingerprint
            primary_calibration = target_inference.calibration_fingerprint
        else:
            primary_score = regional_score
            primary_task = regional_inference.target_task
            primary_classifier = regional_inference.classifier_fingerprint
            primary_calibration = regional_inference.calibration_fingerprint
        scores.append(
            TargetAwareSpeciesFusionScore(
                accepted_taxon_key=key,
                scientific_name=candidate.display_name,
                target_candidate=candidate.target_candidate,
                candidate_priority=int(candidate.candidate_priority or 0),
                candidate_reasons=candidate.candidate_reasons,
                text_ensemble_score=prompt.pooled_score,
                reference_centroid_score=reference.centroid_similarity,
                nearest_reference_score=reference.nearest_support_similarity,
                top_k_reference_score=getattr(reference, top_k_field),
                reference_top_k=top_k,
                local_prototype_score=(reference.local_cluster_prototype_similarity),
                global_prototype_score=reference.global_prototype_similarity,
                classifier_task=primary_task,
                classifier_decision_score=primary_score.decision_score,
                calibrated_probability=primary_score.calibrated_probability,
                regional_classifier_decision_score=(regional_score.decision_score),
                regional_calibrated_probability=(regional_score.calibrated_probability),
                regional_rank=regional_ranks[key],
                geographic_scope=geography.geographic_scope,
                geographic_evidence_score=geography.geographic_evidence_score,
                occurrence_support=geography.occurrence_support,
                life_stage=str(prompt_contract["life_stage"]),
                visual_domain=str(prompt_contract["visual_domain"]),
                life_stage_compatible=geography.life_stage_compatible,
                visual_domain_compatible=geography.visual_domain_compatible,
                best_competitor_accepted_taxon_key=(
                    nonmatch.best_known_competitor_accepted_taxon_key
                ),
                best_competitor_scientific_name=(
                    nonmatch.best_known_competitor_scientific_name
                ),
                competitor_margin=nonmatch.competitor_margin,
                nonmatch_score=nonmatch.best_non_target_score,
                abstention_reason=decision.abstention_reason,
                prompt_result_fingerprint=prompt.result_fingerprint,
                reference_embedding_fingerprint=(
                    reference.reference_embedding_fingerprint
                ),
                reference_prototype_fingerprint=(
                    reference.reference_prototype_fingerprint
                ),
                support_manifest_fingerprint=(reference.support_manifest_fingerprint),
                classifier_fingerprint=primary_classifier,
                calibration_fingerprint=primary_calibration,
                structured_evidence_fingerprint=geography.evidence_fingerprint,
                candidate_set_fingerprint=plan.candidate_set_fingerprint,
            )
        )

    result_values: dict[str, object] = {
        "fusion_version": TARGET_AWARE_EVIDENCE_FUSION_VERSION,
        "classification_mode": plan.classification_mode,
        "query_id": reference_contract["query_id"],
        "route": reference_contract["route"],
        "life_stage": prompt_contract["life_stage"],
        "visual_domain": prompt_contract["visual_domain"],
        "candidate_set_id": plan.candidate_set_id,
        "candidate_set_fingerprint": plan.candidate_set_fingerprint,
        "geo_cluster_id": plan.geo_cluster_id,
        "target_accepted_taxon_key": plan.target_accepted_taxon_key,
        "reference_top_k": top_k,
        "reference_score_contract_fingerprint": reference_score_contract,
        "reference_coverage_sufficient": reference_coverage_sufficient,
        "species_scores": tuple(scores),
        "nonmatch_score": nonmatch,
        "decision": decision,
        "target_classifier_inference_fingerprint": (
            target_inference.inference_fingerprint
        ),
        "regional_classifier_inference_fingerprint": (
            regional_inference.inference_fingerprint
        ),
    }
    fingerprint = canonical_semantic_fingerprint(
        _fusion_result_semantics(result_values)
    )
    return TargetAwareFusionResult(
        **result_values,
        fusion_fingerprint=fingerprint,
    )


def target_aware_fusion_result_payload(
    result: TargetAwareFusionResult,
) -> dict[str, object]:
    """Validate and return the semantic payload for one fused scoring result."""

    if not isinstance(result, TargetAwareFusionResult):
        raise TypeError("result must be a TargetAwareFusionResult")
    values = {
        field_name: getattr(result, field_name)
        for field_name in TargetAwareFusionResult.__dataclass_fields__
        if field_name != "fusion_fingerprint"
    }
    fingerprint = _sha256(
        result.fusion_fingerprint,
        field="fusion_fingerprint",
    )
    semantics = _fusion_result_semantics(values)
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("target-aware fusion fingerprint is inconsistent")
    return {**semantics, "fusion_fingerprint": fingerprint}


def _index_prompt_results(
    values: Sequence[PromptPoolingResult],
    *,
    expected: dict[str, TargetAwareScoringClass],
) -> dict[str, PromptPoolingResult]:
    items = tuple(values)
    if any(not isinstance(item, PromptPoolingResult) for item in items):
        raise TypeError("prompt_results must contain PromptPoolingResult values")
    for item in items:
        prompt_pooling_result_payload(item)
    by_key = {item.accepted_taxon_key: item for item in items}
    if len(by_key) != len(items):
        raise ValueError("prompt_results contain duplicate candidate taxa")
    _validate_coverage("prompt_results", expected=set(expected), actual=set(by_key))
    for key, item in by_key.items():
        if item.scientific_name != expected[key].display_name:
            raise ValueError("prompt result scientific name does not match plan")
        if item.score_kind != PROMPT_SCORE_KIND:
            raise ValueError("prompt result score kind must be cosine_similarity")
        if (
            item.geography_prompt_ablation_enabled
            or item.selected_geography_prompt_count != 0
        ):
            raise ValueError(
                "geographic prompt ablations cannot replace structured geography"
            )
    return by_key


def _index_reference_results(
    values: Sequence[CandidateReferenceEvidence],
    *,
    expected: dict[str, TargetAwareScoringClass],
) -> dict[str, CandidateReferenceEvidence]:
    items = tuple(values)
    if any(not isinstance(item, CandidateReferenceEvidence) for item in items):
        raise TypeError(
            "reference_results must contain CandidateReferenceEvidence values"
        )
    by_key = {item.accepted_taxon_key: item for item in items}
    if len(by_key) != len(items):
        raise ValueError("reference_results contain duplicate candidate taxa")
    _validate_coverage("reference_results", expected=set(expected), actual=set(by_key))
    for key, item in by_key.items():
        if item.scientific_name != expected[key].display_name:
            raise ValueError("reference result scientific name does not match plan")
        _validate_reference_result(item)
    return by_key


def _index_structured_evidence(
    values: Sequence[CandidateStructuredEvidence],
    *,
    expected: dict[str, TargetAwareScoringClass],
) -> dict[str, CandidateStructuredEvidence]:
    items = tuple(values)
    if any(not isinstance(item, CandidateStructuredEvidence) for item in items):
        raise TypeError(
            "structured_evidence must contain CandidateStructuredEvidence values"
        )
    by_key = {item.accepted_taxon_key: item for item in items}
    if len(by_key) != len(items):
        raise ValueError("structured_evidence contains duplicate candidate taxa")
    _validate_coverage(
        "structured_evidence",
        expected=set(expected),
        actual=set(by_key),
    )
    return by_key


def _prompt_contract(values: Sequence[PromptPoolingResult]) -> dict[str, object]:
    contracts = {
        (
            item.pooling_version,
            item.strategy,
            item.prompt_version,
            item.subset_id,
            item.route,
            item.life_stage,
            item.visual_domain,
            item.model_fingerprint,
            item.image_embedding_fingerprint,
            item.embedding_dimension,
        )
        for item in values
    }
    if len(contracts) != 1:
        raise ValueError("prompt_results mix scoring contracts")
    (
        pooling_version,
        strategy,
        prompt_version,
        subset_id,
        route,
        life_stage,
        visual_domain,
        model_fingerprint,
        image_embedding_fingerprint,
        embedding_dimension,
    ) = next(iter(contracts))
    return {
        "pooling_version": pooling_version,
        "strategy": strategy,
        "prompt_version": prompt_version,
        "subset_id": subset_id,
        "route": route,
        "life_stage": life_stage,
        "visual_domain": visual_domain,
        "model_fingerprint": model_fingerprint,
        "image_embedding_fingerprint": image_embedding_fingerprint,
        "embedding_dimension": embedding_dimension,
    }


def _reference_contract(
    values: Sequence[CandidateReferenceEvidence],
) -> dict[str, object]:
    contracts = {
        (
            item.scoring_version,
            item.query_id,
            item.route,
            item.visual_input_kind,
            item.geo_cluster_id,
            item.prototype_method,
            item.balanced_sampling_seed,
            item.fixed_reference_count,
            item.centering_fingerprint,
            item.model_fingerprint,
            item.reference_embedding_fingerprint,
            item.reference_prototype_fingerprint,
            item.support_manifest_fingerprint,
        )
        for item in values
    }
    if len(contracts) != 1:
        raise ValueError("reference_results mix scoring contracts")
    (
        scoring_version,
        query_id,
        route,
        visual_input_kind,
        geo_cluster_id,
        prototype_method,
        balanced_sampling_seed,
        fixed_reference_count,
        centering_fingerprint,
        model_fingerprint,
        reference_embedding_fingerprint,
        reference_prototype_fingerprint,
        support_manifest_fingerprint,
    ) = next(iter(contracts))
    if scoring_version != REFERENCE_EVIDENCE_SCORING_VERSION:
        raise ValueError("reference evidence scoring version is incompatible")
    return {
        "scoring_version": scoring_version,
        "query_id": query_id,
        "route": route,
        "visual_input_kind": visual_input_kind,
        "geo_cluster_id": geo_cluster_id,
        "prototype_method": prototype_method,
        "balanced_sampling_seed": balanced_sampling_seed,
        "fixed_reference_count": fixed_reference_count,
        "centering_fingerprint": centering_fingerprint,
        "model_fingerprint": model_fingerprint,
        "reference_embedding_fingerprint": reference_embedding_fingerprint,
        "reference_prototype_fingerprint": reference_prototype_fingerprint,
        "support_manifest_fingerprint": support_manifest_fingerprint,
    }


def _validate_prompt_reference_plan_contracts(
    *,
    plan: TargetAwareScoringPlan,
    prompt_contract: dict[str, object],
    reference_contract: dict[str, object],
    structured: dict[str, CandidateStructuredEvidence],
) -> None:
    if prompt_contract["route"] != reference_contract["route"]:
        raise ValueError("prompt and reference routes differ")
    if prompt_contract["model_fingerprint"] != reference_contract["model_fingerprint"]:
        raise ValueError("prompt and reference model fingerprints differ")
    if reference_contract["geo_cluster_id"] != plan.geo_cluster_id:
        raise ValueError("reference geo cluster does not match scoring plan")
    for item in structured.values():
        if item.candidate_set_fingerprint != plan.candidate_set_fingerprint:
            raise ValueError("structured evidence candidate set does not match plan")
        if item.geo_cluster_id != plan.geo_cluster_id:
            raise ValueError("structured evidence geo cluster does not match plan")
        if item.route != prompt_contract["route"]:
            raise ValueError("structured evidence route does not match prompt evidence")
        if item.life_stage != prompt_contract["life_stage"]:
            raise ValueError(
                "structured evidence life stage does not match prompt evidence"
            )
        if item.visual_domain != prompt_contract["visual_domain"]:
            raise ValueError(
                "structured evidence visual domain does not match prompt evidence"
            )
    no_geo = plan.geo_cluster_id == NO_GEO_CLUSTER_ID
    if no_geo and any(
        item.geographic_scope != "no_geo_global" for item in structured.values()
    ):
        raise ValueError("no_geo plans require no_geo_global candidate evidence")
    if not no_geo and any(
        item.geographic_scope == "no_geo_global" for item in structured.values()
    ):
        raise ValueError("geolocated plans cannot contain no_geo_global evidence")


def _validate_classifier_contracts(
    *,
    plan: TargetAwareScoringPlan,
    expected_keys: set[str],
    target: FrozenClassifierInference,
    regional: FrozenClassifierInference,
    route: str,
    model_fingerprint: str,
    embedding_dimension: int,
    query_id: str,
    image_embedding_fingerprint: str,
    decision_policy: SelectiveDecisionPolicy,
) -> None:
    if target.target_task not in _TARGET_VERIFIER_TASKS:
        raise ValueError("target classifier must be a target-verifier task")
    if regional.target_task != "regional_multiclass":
        raise ValueError("regional classifier must use regional_multiclass")
    if target.target_accepted_taxon_key != plan.target_accepted_taxon_key:
        raise ValueError("target classifier taxon does not match scoring plan")
    if regional.target_accepted_taxon_key != plan.target_accepted_taxon_key:
        raise ValueError("regional classifier target does not match scoring plan")
    if {item.class_label for item in target.class_scores} != {
        NON_TARGET_CLASS_LABEL,
        plan.target_accepted_taxon_key,
    }:
        raise ValueError("target classifier class coverage is invalid")
    regional_labels = {item.class_label for item in regional.class_scores}
    _validate_coverage(
        "regional_classifier",
        expected=expected_keys,
        actual=regional_labels,
    )
    if target.route != route or regional.route != route:
        raise ValueError("classifier route does not match reference route")
    if target.query_id != query_id or regional.query_id != query_id:
        raise ValueError("classifier query ID does not match reference query")
    if (
        target.image_embedding_fingerprint != image_embedding_fingerprint
        or regional.image_embedding_fingerprint != image_embedding_fingerprint
    ):
        raise ValueError("classifier image embedding does not match prompt evidence")
    if (
        target.model_fingerprint != model_fingerprint
        or regional.model_fingerprint != model_fingerprint
    ):
        raise ValueError("classifier model does not match reference model")
    if (
        target.embedding_dimension != embedding_dimension
        or regional.embedding_dimension != embedding_dimension
    ):
        raise ValueError(
            "classifier embedding dimension does not match prompt evidence"
        )
    if target.preprocessing_fingerprint != regional.preprocessing_fingerprint:
        raise ValueError("target and regional classifier preprocessing differs")
    if (
        target.reference_bank_version != regional.reference_bank_version
        or target.reference_bank_fingerprint != regional.reference_bank_fingerprint
    ):
        raise ValueError("target and regional classifier reference banks differ")
    if target.feature_input_fingerprint != regional.feature_input_fingerprint:
        raise ValueError(
            "target and regional classifiers used different query features"
        )
    if decision_policy.target_task != target.target_task:
        raise ValueError("decision policy task does not match target classifier")
    if decision_policy.route != route:
        raise ValueError("decision policy route does not match fusion route")
    if decision_policy.model_fingerprint != target.model_fingerprint:
        raise ValueError("decision policy model does not match target classifier")
    if decision_policy.classifier_fingerprint != target.classifier_fingerprint:
        raise ValueError("decision policy classifier does not match target classifier")
    if decision_policy.calibration_fingerprint != target.calibration_fingerprint:
        raise ValueError("decision policy calibration does not match target classifier")
    if decision_policy.split_fingerprint != target.calibration_split_fingerprint:
        raise ValueError("decision policy split does not match target calibration")


def _validate_classifier_inference(
    value: FrozenClassifierInference,
) -> FrozenClassifierInference:
    if not isinstance(value, FrozenClassifierInference):
        raise TypeError("classifier evidence must be FrozenClassifierInference")
    if value.inference_version != FROZEN_CLASSIFIER_INFERENCE_VERSION:
        raise ValueError("classifier inference version is incompatible")
    if value.classifier_version != CLASSIFIER_VERSION:
        raise ValueError("classifier artifact version is incompatible")
    labels = []
    for item in value.class_scores:
        if not isinstance(item, ClassifierClassScore):
            raise TypeError("class_scores must contain ClassifierClassScore values")
        labels.append(_required_text(item.class_label, field="class_label"))
        _finite_number(item.decision_score, field="decision_score")
        _unit_interval(
            item.calibrated_probability,
            field="calibrated_probability",
        )
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("classifier inference class labels are invalid")
    if (
        abs(sum(item.calibrated_probability for item in value.class_scores) - 1.0)
        > 1e-9
    ):
        raise ValueError("classifier inference probabilities must sum to one")
    _sha256(value.classifier_fingerprint, field="classifier_fingerprint")
    _sha256(value.calibration_fingerprint, field="calibration_fingerprint")
    _sha256(
        value.calibration_split_fingerprint,
        field="calibration_split_fingerprint",
    )
    _sha256(value.model_fingerprint, field="model_fingerprint")
    _positive_integer(value.embedding_dimension, field="embedding_dimension")
    _sha256(value.feature_schema_fingerprint, field="feature_schema_fingerprint")
    _sha256(value.feature_layout_fingerprint, field="feature_layout_fingerprint")
    _sha256(value.feature_input_fingerprint, field="feature_input_fingerprint")
    _sha256(value.image_embedding_fingerprint, field="image_embedding_fingerprint")
    _sha256(value.preprocessing_fingerprint, field="preprocessing_fingerprint")
    _required_text(value.reference_bank_version, field="reference_bank_version")
    _sha256(value.reference_bank_fingerprint, field="reference_bank_fingerprint")
    _sha256(value.training_data_fingerprint, field="training_data_fingerprint")
    fingerprint = _sha256(
        value.inference_fingerprint,
        field="inference_fingerprint",
    )
    semantics = _classifier_inference_semantics(
        {
            field: getattr(value, field)
            for field in FrozenClassifierInference.__dataclass_fields__
            if field != "inference_fingerprint"
        }
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("classifier inference fingerprint is inconsistent")
    return value


def _validate_reference_result(value: CandidateReferenceEvidence) -> None:
    if value.scoring_version != REFERENCE_EVIDENCE_SCORING_VERSION:
        raise ValueError("reference evidence scoring version is incompatible")
    for field in (
        "support_count",
        "usable_support_count",
        "local_support_count",
        "selected_support_count",
        "selected_local_support_count",
        "fixed_reference_count",
    ):
        _nonnegative_integer(getattr(value, field), field=field)
    if value.usable_support_count > value.support_count:
        raise ValueError("usable_support_count exceeds support_count")
    if value.local_support_count > value.usable_support_count:
        raise ValueError("local_support_count exceeds usable_support_count")
    if value.selected_support_count > value.usable_support_count:
        raise ValueError("selected_support_count exceeds usable_support_count")
    if value.selected_support_count > value.fixed_reference_count:
        raise ValueError("selected_support_count exceeds fixed_reference_count")
    if value.selected_local_support_count > value.selected_support_count:
        raise ValueError("selected_local_support_count exceeds selected_support_count")
    if len(value.selected_reference_observation_ids) != value.selected_support_count:
        raise ValueError("selected reference IDs do not match selected_support_count")
    if len(set(value.selected_reference_observation_ids)) != len(
        value.selected_reference_observation_ids
    ):
        raise ValueError("selected reference observation IDs must be unique")
    if any(
        not str(value).strip() for value in value.selected_reference_observation_ids
    ):
        raise ValueError("selected reference observation IDs must be non-empty")
    for field in (
        "nearest_support_similarity",
        "mean_top_three_similarity",
        "mean_top_five_similarity",
        "centroid_similarity",
        "local_cluster_prototype_similarity",
        "global_prototype_similarity",
    ):
        _optional_cosine(getattr(value, field), field=field)
    if (value.nearest_reference_observation_id is None) != (
        value.nearest_support_similarity is None
    ):
        raise ValueError(
            "nearest reference identity and similarity must be all-or-none"
        )
    if value.selected_support_count == 0:
        if (
            value.nearest_support_similarity is not None
            or value.centroid_similarity is not None
        ):
            raise ValueError(
                "empty selected support cannot produce reference similarities"
            )
    elif value.nearest_support_similarity is None or value.centroid_similarity is None:
        raise ValueError("selected support requires nearest and centroid similarities")
    if (value.mean_top_three_similarity is not None) != (
        value.selected_support_count >= 3
    ):
        raise ValueError("top-three similarity does not match selected support count")
    if (value.mean_top_five_similarity is not None) != (
        value.selected_support_count >= 5
    ):
        raise ValueError("top-five similarity does not match selected support count")
    if value.distance_to_nearest_independent_observation is not None:
        distance = _finite_number(
            value.distance_to_nearest_independent_observation,
            field="distance_to_nearest_independent_observation",
        )
        if not 0.0 <= distance <= 2.0:
            raise ValueError(
                "distance_to_nearest_independent_observation must be between 0 and 2"
            )
    if (
        abs(
            _finite_number(value.query_embedding_norm, field="query_embedding_norm")
            - 1.0
        )
        > 1e-5
    ):
        raise ValueError("reference query embedding must be unit-normalized")
    for field in (
        "model_fingerprint",
        "reference_embedding_fingerprint",
        "reference_prototype_fingerprint",
        "support_manifest_fingerprint",
    ):
        _sha256(getattr(value, field), field=field)
    if value.centering_fingerprint is not None:
        _sha256(value.centering_fingerprint, field="centering_fingerprint")
    if value.insufficient_support and not value.insufficient_support_reasons:
        raise ValueError("insufficient support requires explicit reasons")
    if not value.insufficient_support and value.insufficient_support_reasons:
        raise ValueError("sufficient reference evidence cannot retain failure reasons")
    if value.local_prototype_available != (
        value.local_cluster_prototype_similarity is not None
    ):
        raise ValueError("local prototype availability is inconsistent")
    if value.local_support_available != bool(value.local_support_count):
        raise ValueError("local support availability is inconsistent")
    if value.global_prototype_available != (
        value.global_prototype_similarity is not None
    ):
        raise ValueError("global prototype availability is inconsistent")


def _classifier_inference_semantics(values: dict[str, object]) -> dict[str, object]:
    return {
        "inference_version": values["inference_version"],
        "target_task": values["target_task"],
        "target_accepted_taxon_key": values["target_accepted_taxon_key"],
        "query_id": values["query_id"],
        "route": values["route"],
        "class_scores": [asdict(item) for item in values["class_scores"]],
        "classifier_version": values["classifier_version"],
        "classifier_fingerprint": values["classifier_fingerprint"],
        "calibration_fingerprint": values["calibration_fingerprint"],
        "calibration_split_fingerprint": values["calibration_split_fingerprint"],
        "model_fingerprint": values["model_fingerprint"],
        "embedding_dimension": values["embedding_dimension"],
        "feature_schema_fingerprint": values["feature_schema_fingerprint"],
        "feature_layout_fingerprint": values["feature_layout_fingerprint"],
        "feature_input_fingerprint": values["feature_input_fingerprint"],
        "image_embedding_fingerprint": values["image_embedding_fingerprint"],
        "preprocessing_fingerprint": values["preprocessing_fingerprint"],
        "reference_bank_version": values["reference_bank_version"],
        "reference_bank_fingerprint": values["reference_bank_fingerprint"],
        "training_data_fingerprint": values["training_data_fingerprint"],
    }


def _fusion_result_semantics(values: dict[str, object]) -> dict[str, object]:
    return {
        "fusion_version": values["fusion_version"],
        "classification_mode": values["classification_mode"],
        "query_id": values["query_id"],
        "route": values["route"],
        "life_stage": values["life_stage"],
        "visual_domain": values["visual_domain"],
        "candidate_set_id": values["candidate_set_id"],
        "candidate_set_fingerprint": values["candidate_set_fingerprint"],
        "geo_cluster_id": values["geo_cluster_id"],
        "target_accepted_taxon_key": values["target_accepted_taxon_key"],
        "reference_top_k": values["reference_top_k"],
        "reference_score_contract_fingerprint": values[
            "reference_score_contract_fingerprint"
        ],
        "reference_coverage_sufficient": values["reference_coverage_sufficient"],
        "species_scores": [asdict(item) for item in values["species_scores"]],
        "nonmatch_score": asdict(values["nonmatch_score"]),
        "decision": asdict(values["decision"]),
        "target_classifier_inference_fingerprint": values[
            "target_classifier_inference_fingerprint"
        ],
        "regional_classifier_inference_fingerprint": values[
            "regional_classifier_inference_fingerprint"
        ],
    }


def _validate_coverage(
    field: str,
    *,
    expected: set[str],
    actual: set[str],
) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"{field} are missing candidate taxa: {missing}")
    if unexpected:
        raise ValueError(f"{field} contain unexpected candidate taxa: {unexpected}")


def _reference_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("reference_top_k must be an integer")
    if value not in SUPPORTED_REFERENCE_TOP_K:
        raise ValueError("reference_top_k must be exactly 3 or 5")
    return value


def _top_k_field(value: int) -> str:
    return "mean_top_three_similarity" if value == 3 else "mean_top_five_similarity"


def _optional_cosine(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, field=field)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between -1 and 1")
    return result


def _optional_unit_interval(value: object, *, field: str) -> float | None:
    return None if value is None else _unit_interval(value, field=field)


def _unit_interval(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _unique_text_tuple(values: object, *, field: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence of strings")
    normalized = tuple(_required_text(value, field=field) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _sha256(value: object, *, field: str) -> str:
    result = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return result


__all__ = [
    "FROZEN_CLASSIFIER_INFERENCE_VERSION",
    "REFERENCE_SCORE_CONTRACT_VERSION",
    "STRUCTURED_EVIDENCE_VERSION",
    "SUPPORTED_REFERENCE_TOP_K",
    "TARGET_AWARE_EVIDENCE_FUSION_VERSION",
    "CandidateStructuredEvidence",
    "ClassifierClassScore",
    "FrozenClassifierInference",
    "TargetAwareFusionQuality",
    "TargetAwareFusionResult",
    "TargetAwareSpeciesFusionScore",
    "fuse_target_aware_species_evidence",
    "score_frozen_classifier",
    "target_aware_fusion_result_payload",
]
