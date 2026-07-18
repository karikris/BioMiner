"""Versioned, component-preserving inputs for raw dynamic-pool fusion."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
import re
from statistics import median

from biominer.bioclip.dynamic_pool_scoring import (
    RAW_DISAGREEMENT_COVERAGE_VERSION,
    RAW_FAMILY_EVIDENCE_SET_VERSION,
    RAW_FAMILY_EVIDENCE_VERSION,
    RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION,
    RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION,
    RAW_LOCAL_REFERENCE_EVIDENCE_SET_VERSION,
    RAW_LOCAL_REFERENCE_EVIDENCE_VERSION,
    RawCandidateDisagreementCoverage,
    RawDisagreementCoverageSet,
    RawFamilyEvidence,
    RawFamilyEvidenceSet,
    RawGlobalReferenceEvidence,
    RawGlobalReferenceEvidenceSet,
    RawLocalReferenceEvidence,
    RawLocalReferenceEvidenceSet,
    calculate_dynamic_pool_disagreement_coverage,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_SCORE_COMPONENT_VERSION = "dynamic-score-component-v1"
DYNAMIC_SCORE_COMPONENT_SET_VERSION = "dynamic-score-component-set-v1"
VALIDATION_LINEAR_FUSION_PARAMETERS_VERSION = "validation-linear-fusion-v1"
RAW_FUSION_CANDIDATE_SCORE_VERSION = "raw-fusion-candidate-score-v1"
RAW_FUSION_SCORE_SET_VERSION = "raw-fusion-score-set-v1"
RANKED_FUSION_CANDIDATE_VERSION = "ranked-fusion-candidate-v1"
RAW_FUSION_METHOD_RANKING_VERSION = "raw-fusion-method-ranking-v1"
RAW_FUSION_RANKING_SET_VERSION = "raw-fusion-ranking-set-v1"

GLOBAL_FUSION_COMPONENTS = (
    "global_prototype_similarity",
    "global_nearest_reference_similarity",
    "global_top_k_mean_similarity",
)
LOCAL_FUSION_COMPONENTS = (
    "local_prototype_similarity",
    "local_nearest_reference_similarity",
    "local_top_k_mean_similarity",
)
FUSION_COMPONENTS = (*GLOBAL_FUSION_COMPONENTS, *LOCAL_FUSION_COMPONENTS)

UNWEIGHTED_COMPONENT_MEAN = "unweighted_component_mean"
VALIDATION_FITTED_LINEAR = "validation_fitted_linear"
MAXIMUM_SCOPE_EVIDENCE = "maximum_scope_evidence"
ROBUST_RANK_AGGREGATION = "robust_rank_aggregation"
RAW_FUSION_METHODS = (
    UNWEIGHTED_COMPONENT_MEAN,
    VALIDATION_FITTED_LINEAR,
    MAXIMUM_SCOPE_EVIDENCE,
    ROBUST_RANK_AGGREGATION,
)

_METHOD_VERSIONS = {
    UNWEIGHTED_COMPONENT_MEAN: "unweighted-component-mean-v1",
    VALIDATION_FITTED_LINEAR: "validation-fitted-linear-v1",
    MAXIMUM_SCOPE_EVIDENCE: "maximum-scope-evidence-v1",
    ROBUST_RANK_AGGREGATION: "robust-rank-aggregation-v1",
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ValidationLinearFusionParameters:
    """Explicit full/local-missing coefficients fitted on validation evidence."""

    validation_artifact_fingerprint: str
    full_weights: tuple[float, ...]
    global_only_weights: tuple[float, ...]
    full_intercept: float = 0.0
    global_only_intercept: float = 0.0
    schema_version: str = VALIDATION_LINEAR_FUSION_PARAMETERS_VERSION
    parameters_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_LINEAR_FUSION_PARAMETERS_VERSION:
            raise ValueError("unsupported validation linear fusion parameter version")
        validation_fingerprint = _sha256(
            self.validation_artifact_fingerprint,
            field="validation_artifact_fingerprint",
        )
        full_weights = _weights(
            self.full_weights,
            expected_count=len(FUSION_COMPONENTS),
            field="full_weights",
        )
        global_only_weights = _weights(
            self.global_only_weights,
            expected_count=len(GLOBAL_FUSION_COMPONENTS),
            field="global_only_weights",
        )
        full_intercept = _finite_number(self.full_intercept, field="full_intercept")
        global_only_intercept = _finite_number(
            self.global_only_intercept,
            field="global_only_intercept",
        )
        base = {
            "schema_version": VALIDATION_LINEAR_FUSION_PARAMETERS_VERSION,
            "validation_artifact_fingerprint": validation_fingerprint,
            "fit_partition": "validation",
            "full_component_order": list(FUSION_COMPONENTS),
            "full_weights": list(full_weights),
            "full_intercept": full_intercept,
            "global_only_component_order": list(GLOBAL_FUSION_COMPONENTS),
            "global_only_weights": list(global_only_weights),
            "global_only_intercept": global_only_intercept,
            "missing_local_semantics": "use_explicit_global_only_coefficients",
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if (
            self.parameters_fingerprint is not None
            and _sha256(
                self.parameters_fingerprint,
                field="parameters_fingerprint",
            )
            != fingerprint
        ):
            raise ValueError("validation linear fusion parameter fingerprint mismatch")
        object.__setattr__(
            self,
            "validation_artifact_fingerprint",
            validation_fingerprint,
        )
        object.__setattr__(self, "full_weights", full_weights)
        object.__setattr__(self, "global_only_weights", global_only_weights)
        object.__setattr__(self, "full_intercept", full_intercept)
        object.__setattr__(self, "global_only_intercept", global_only_intercept)
        object.__setattr__(self, "parameters_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class RawFusionCandidateScore:
    """One provisional method result linked to its complete component row."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    candidate_accepted_taxon_key: str
    candidate_scientific_name: str
    component_fingerprint: str
    method: str
    method_version: str
    method_policy_fingerprint: str
    local_evidence_status: str
    component_names_used: tuple[str, ...]
    raw_component_values: tuple[float, ...]
    method_component_values: tuple[float, ...]
    raw_fusion_score: float
    score_fingerprint: str


@dataclass(frozen=True, slots=True)
class RawFusionScoreSet:
    """All provisional methods over one exact preserved component set."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    component_set: DynamicScoreComponentSet
    component_set_fingerprint: str
    linear_parameters: ValidationLinearFusionParameters
    methods: tuple[str, ...]
    method_policy_fingerprints: tuple[tuple[str, str], ...]
    scores: tuple[RawFusionCandidateScore, ...]
    score_set_fingerprint: str


@dataclass(frozen=True, slots=True)
class RankedFusionCandidate:
    """One candidate's deterministic position within one raw fusion method."""

    schema_version: str
    method: str
    method_version: str
    candidate_accepted_taxon_key: str
    candidate_scientific_name: str
    candidate_rank: int
    raw_fusion_score: float
    margin_to_next_raw: float | None
    score_tied_with_previous: bool
    score_tied_with_next: bool
    source_score_fingerprint: str
    component_fingerprint: str
    rank_fingerprint: str


@dataclass(frozen=True, slots=True)
class RawFusionMethodRanking:
    """Complete ranking and explicit alternatives for one provisional method."""

    schema_version: str
    method: str
    method_version: str
    method_policy_fingerprint: str
    candidate_count: int
    complete_candidate_set: bool
    top_candidate_accepted_taxon_key: str
    top_candidate_scientific_name: str
    top_raw_fusion_score: float
    top_margin_raw: float | None
    alternative_candidate_keys: tuple[str, ...]
    candidates: tuple[RankedFusionCandidate, ...]
    ranking_fingerprint: str


@dataclass(frozen=True, slots=True)
class RawFusionRankingSet:
    """All method rankings with agreement state and no selected method."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    fusion_scores: RawFusionScoreSet
    fusion_score_set_fingerprint: str
    method_rankings: tuple[RawFusionMethodRanking, ...]
    top_candidate_keys_by_method: tuple[tuple[str, str], ...]
    cross_method_top1_agreement: bool
    agreed_top_candidate_key: str | None
    method_selection_status: str
    ranking_set_fingerprint: str


@dataclass(frozen=True, slots=True)
class DynamicCandidateScoreComponents:
    """Exact unfused evidence rows for one candidate."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    candidate_accepted_taxon_key: str
    candidate_scientific_name: str
    global_evidence: RawGlobalReferenceEvidence
    local_evidence: RawLocalReferenceEvidence
    disagreement_coverage: RawCandidateDisagreementCoverage
    component_fingerprint: str


@dataclass(frozen=True, slots=True)
class DynamicScoreComponentSet:
    """Lossless family and candidate inputs; no fused score replaces evidence."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    candidate_matrix_signature: str
    candidate_set_fingerprint: str
    family_evidence: RawFamilyEvidenceSet
    global_evidence_set_fingerprint: str
    local_evidence_set_fingerprint: str
    disagreement_coverage: RawDisagreementCoverageSet
    disagreement_coverage_set_fingerprint: str
    candidates: tuple[DynamicCandidateScoreComponents, ...]
    component_set_fingerprint: str


def preserve_dynamic_score_components(
    family_evidence: RawFamilyEvidenceSet,
    global_evidence: RawGlobalReferenceEvidenceSet,
    local_evidence: RawLocalReferenceEvidenceSet,
    disagreement_coverage: RawDisagreementCoverageSet,
) -> DynamicScoreComponentSet:
    """Bind exact Task 7.1 evidence into one immutable, unfused schema."""

    _validate_source_evidence(
        family_evidence,
        global_evidence,
        local_evidence,
        disagreement_coverage,
    )
    candidates = tuple(
        _candidate_components(
            query_id=global_evidence.query_id,
            query_fingerprint=global_evidence.query_fingerprint,
            global_score=global_score,
            local_score=local_score,
            disagreement=disagreement,
        )
        for global_score, local_score, disagreement in zip(
            global_evidence.scores,
            local_evidence.scores,
            disagreement_coverage.scores,
            strict=True,
        )
    )
    base = _component_set_base(
        query_id=global_evidence.query_id,
        query_fingerprint=global_evidence.query_fingerprint,
        candidate_matrix_signature=global_evidence.candidate_matrix_signature,
        candidate_set_fingerprint=global_evidence.candidate_set_fingerprint,
        family_evidence=family_evidence,
        global_evidence_set_fingerprint=global_evidence.score_set_fingerprint,
        local_evidence_set_fingerprint=local_evidence.score_set_fingerprint,
        disagreement_coverage_set_fingerprint=(
            disagreement_coverage.evidence_set_fingerprint
        ),
        candidates=candidates,
    )
    result = DynamicScoreComponentSet(
        schema_version=DYNAMIC_SCORE_COMPONENT_SET_VERSION,
        query_id=global_evidence.query_id,
        query_fingerprint=global_evidence.query_fingerprint,
        candidate_matrix_signature=global_evidence.candidate_matrix_signature,
        candidate_set_fingerprint=global_evidence.candidate_set_fingerprint,
        family_evidence=family_evidence,
        global_evidence_set_fingerprint=global_evidence.score_set_fingerprint,
        local_evidence_set_fingerprint=local_evidence.score_set_fingerprint,
        disagreement_coverage=disagreement_coverage,
        disagreement_coverage_set_fingerprint=(
            disagreement_coverage.evidence_set_fingerprint
        ),
        candidates=candidates,
        component_set_fingerprint=canonical_semantic_fingerprint(base),
    )
    validate_dynamic_score_components(result)
    return result


def validate_dynamic_score_components(components: DynamicScoreComponentSet) -> None:
    """Fail closed if a preserved component set no longer matches its sources."""

    if not isinstance(components, DynamicScoreComponentSet):
        raise TypeError("components must be a DynamicScoreComponentSet")
    if components.schema_version != DYNAMIC_SCORE_COMPONENT_SET_VERSION:
        raise ValueError("unsupported dynamic score component set version")
    candidates = tuple(components.candidates)
    if not candidates:
        raise ValueError("dynamic score component set must not be empty")
    if any(
        not isinstance(item, DynamicCandidateScoreComponents) for item in candidates
    ):
        raise TypeError("dynamic score component set contains invalid candidate rows")
    candidate_keys = [item.candidate_accepted_taxon_key for item in candidates]
    if candidate_keys != sorted(candidate_keys):
        raise ValueError(
            "dynamic score component candidates are not canonically ordered"
        )
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("dynamic score component candidates repeat taxon keys")
    global_evidence = RawGlobalReferenceEvidenceSet(
        schema_version=RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION,
        query_id=components.query_id,
        query_fingerprint=components.query_fingerprint,
        candidate_matrix_signature=components.candidate_matrix_signature,
        candidate_set_fingerprint=components.candidate_set_fingerprint,
        scores=tuple(item.global_evidence for item in candidates),
        score_set_fingerprint=components.global_evidence_set_fingerprint,
    )
    local_evidence = RawLocalReferenceEvidenceSet(
        schema_version=RAW_LOCAL_REFERENCE_EVIDENCE_SET_VERSION,
        query_id=components.query_id,
        query_fingerprint=components.query_fingerprint,
        candidate_matrix_signature=components.candidate_matrix_signature,
        candidate_set_fingerprint=components.candidate_set_fingerprint,
        scores=tuple(item.local_evidence for item in candidates),
        score_set_fingerprint=components.local_evidence_set_fingerprint,
    )
    disagreement = components.disagreement_coverage
    if not isinstance(disagreement, RawDisagreementCoverageSet):
        raise TypeError("preserved disagreement coverage has an invalid type")
    if disagreement.evidence_set_fingerprint != (
        components.disagreement_coverage_set_fingerprint
    ):
        raise ValueError("preserved disagreement coverage fingerprint differs")
    if disagreement.scores != tuple(item.disagreement_coverage for item in candidates):
        raise ValueError("preserved disagreement candidate rows differ")
    _validate_source_evidence(
        components.family_evidence,
        global_evidence,
        local_evidence,
        disagreement,
    )
    expected_candidates = tuple(
        _candidate_components(
            query_id=components.query_id,
            query_fingerprint=components.query_fingerprint,
            global_score=item.global_evidence,
            local_score=item.local_evidence,
            disagreement=item.disagreement_coverage,
        )
        for item in candidates
    )
    if candidates != expected_candidates:
        raise ValueError("dynamic candidate component row or fingerprint mismatch")
    base = _component_set_base(
        query_id=components.query_id,
        query_fingerprint=components.query_fingerprint,
        candidate_matrix_signature=components.candidate_matrix_signature,
        candidate_set_fingerprint=components.candidate_set_fingerprint,
        family_evidence=components.family_evidence,
        global_evidence_set_fingerprint=components.global_evidence_set_fingerprint,
        local_evidence_set_fingerprint=components.local_evidence_set_fingerprint,
        disagreement_coverage_set_fingerprint=(
            components.disagreement_coverage_set_fingerprint
        ),
        candidates=candidates,
    )
    if components.component_set_fingerprint != canonical_semantic_fingerprint(base):
        raise ValueError("dynamic score component set fingerprint mismatch")


def evaluate_raw_fusion_methods(
    components: DynamicScoreComponentSet,
    linear_parameters: ValidationLinearFusionParameters,
) -> RawFusionScoreSet:
    """Evaluate every required provisional method without selecting one."""

    validate_dynamic_score_components(components)
    if not isinstance(linear_parameters, ValidationLinearFusionParameters):
        raise TypeError("linear_parameters must be ValidationLinearFusionParameters")
    result = _evaluate_raw_fusion_methods(components, linear_parameters)
    validate_raw_fusion_scores(result)
    return result


def validate_raw_fusion_scores(fusion_scores: RawFusionScoreSet) -> None:
    """Recompute all provisional method results and reject any drift."""

    if not isinstance(fusion_scores, RawFusionScoreSet):
        raise TypeError("fusion_scores must be a RawFusionScoreSet")
    if fusion_scores.schema_version != RAW_FUSION_SCORE_SET_VERSION:
        raise ValueError("unsupported raw fusion score set version")
    validate_dynamic_score_components(fusion_scores.component_set)
    if not isinstance(
        fusion_scores.linear_parameters,
        ValidationLinearFusionParameters,
    ):
        raise TypeError("raw fusion set has invalid linear parameters")
    expected = _evaluate_raw_fusion_methods(
        fusion_scores.component_set,
        fusion_scores.linear_parameters,
    )
    if fusion_scores != expected:
        raise ValueError("raw fusion score set does not match components and policy")


def rank_raw_fusion_candidates(
    fusion_scores: RawFusionScoreSet,
) -> RawFusionRankingSet:
    """Rank the complete candidate set independently for every fusion method."""

    validate_raw_fusion_scores(fusion_scores)
    result = _rank_raw_fusion_candidates(fusion_scores)
    validate_raw_fusion_rankings(result)
    return result


def validate_raw_fusion_rankings(rankings: RawFusionRankingSet) -> None:
    """Recompute every method ordering, margin, tie and alternative."""

    if not isinstance(rankings, RawFusionRankingSet):
        raise TypeError("rankings must be a RawFusionRankingSet")
    if rankings.schema_version != RAW_FUSION_RANKING_SET_VERSION:
        raise ValueError("unsupported raw fusion ranking set version")
    validate_raw_fusion_scores(rankings.fusion_scores)
    expected = _rank_raw_fusion_candidates(rankings.fusion_scores)
    if rankings != expected:
        raise ValueError("raw fusion rankings do not match the fusion score set")


def _rank_raw_fusion_candidates(
    fusion_scores: RawFusionScoreSet,
) -> RawFusionRankingSet:
    expected_candidate_keys = tuple(
        candidate.candidate_accepted_taxon_key
        for candidate in fusion_scores.component_set.candidates
    )
    method_rankings: list[RawFusionMethodRanking] = []
    for method in RAW_FUSION_METHODS:
        source_scores = tuple(
            score for score in fusion_scores.scores if score.method == method
        )
        if (
            tuple(sorted(score.candidate_accepted_taxon_key for score in source_scores))
            != expected_candidate_keys
        ):
            raise ValueError(
                f"{method} fusion scores do not preserve complete candidate membership"
            )
        ordered = sorted(
            source_scores,
            key=lambda score: (
                -score.raw_fusion_score,
                score.candidate_accepted_taxon_key,
            ),
        )
        ranked_candidates: list[RankedFusionCandidate] = []
        for index, score in enumerate(ordered):
            previous = ordered[index - 1] if index > 0 else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            margin = (
                score.raw_fusion_score - following.raw_fusion_score
                if following is not None
                else None
            )
            base = {
                "schema_version": RANKED_FUSION_CANDIDATE_VERSION,
                "query_id": fusion_scores.query_id,
                "query_fingerprint": fusion_scores.query_fingerprint,
                "method": method,
                "method_version": score.method_version,
                "candidate_accepted_taxon_key": (score.candidate_accepted_taxon_key),
                "candidate_scientific_name": score.candidate_scientific_name,
                "candidate_rank": index + 1,
                "raw_fusion_score": score.raw_fusion_score,
                "margin_to_next_raw": margin,
                "score_tied_with_previous": (
                    previous is not None
                    and previous.raw_fusion_score == score.raw_fusion_score
                ),
                "score_tied_with_next": (
                    following is not None
                    and following.raw_fusion_score == score.raw_fusion_score
                ),
                "source_score_fingerprint": score.score_fingerprint,
                "component_fingerprint": score.component_fingerprint,
            }
            ranked_candidates.append(
                RankedFusionCandidate(
                    schema_version=RANKED_FUSION_CANDIDATE_VERSION,
                    method=method,
                    method_version=score.method_version,
                    candidate_accepted_taxon_key=(score.candidate_accepted_taxon_key),
                    candidate_scientific_name=score.candidate_scientific_name,
                    candidate_rank=index + 1,
                    raw_fusion_score=score.raw_fusion_score,
                    margin_to_next_raw=margin,
                    score_tied_with_previous=bool(
                        previous is not None
                        and previous.raw_fusion_score == score.raw_fusion_score
                    ),
                    score_tied_with_next=bool(
                        following is not None
                        and following.raw_fusion_score == score.raw_fusion_score
                    ),
                    source_score_fingerprint=score.score_fingerprint,
                    component_fingerprint=score.component_fingerprint,
                    rank_fingerprint=canonical_semantic_fingerprint(base),
                )
            )
        top = ranked_candidates[0]
        policy_fingerprint = next(
            fingerprint
            for candidate_method, fingerprint in (
                fusion_scores.method_policy_fingerprints
            )
            if candidate_method == method
        )
        method_base = {
            "schema_version": RAW_FUSION_METHOD_RANKING_VERSION,
            "query_id": fusion_scores.query_id,
            "query_fingerprint": fusion_scores.query_fingerprint,
            "fusion_score_set_fingerprint": fusion_scores.score_set_fingerprint,
            "method": method,
            "method_version": top.method_version,
            "method_policy_fingerprint": policy_fingerprint,
            "candidate_count": len(ranked_candidates),
            "complete_candidate_set": True,
            "top_candidate_accepted_taxon_key": (top.candidate_accepted_taxon_key),
            "top_candidate_scientific_name": top.candidate_scientific_name,
            "top_raw_fusion_score": top.raw_fusion_score,
            "top_margin_raw": top.margin_to_next_raw,
            "alternative_candidate_keys": [
                candidate.candidate_accepted_taxon_key
                for candidate in ranked_candidates[1:]
            ],
            "rank_fingerprints": [
                candidate.rank_fingerprint for candidate in ranked_candidates
            ],
        }
        method_rankings.append(
            RawFusionMethodRanking(
                schema_version=RAW_FUSION_METHOD_RANKING_VERSION,
                method=method,
                method_version=top.method_version,
                method_policy_fingerprint=policy_fingerprint,
                candidate_count=len(ranked_candidates),
                complete_candidate_set=True,
                top_candidate_accepted_taxon_key=(top.candidate_accepted_taxon_key),
                top_candidate_scientific_name=top.candidate_scientific_name,
                top_raw_fusion_score=top.raw_fusion_score,
                top_margin_raw=top.margin_to_next_raw,
                alternative_candidate_keys=tuple(
                    candidate.candidate_accepted_taxon_key
                    for candidate in ranked_candidates[1:]
                ),
                candidates=tuple(ranked_candidates),
                ranking_fingerprint=canonical_semantic_fingerprint(method_base),
            )
        )
    top_candidates = tuple(
        (ranking.method, ranking.top_candidate_accepted_taxon_key)
        for ranking in method_rankings
    )
    distinct_top_candidates = {candidate_key for _, candidate_key in top_candidates}
    agreement = len(distinct_top_candidates) == 1
    agreed_top = next(iter(distinct_top_candidates)) if agreement else None
    set_base = {
        "schema_version": RAW_FUSION_RANKING_SET_VERSION,
        "query_id": fusion_scores.query_id,
        "query_fingerprint": fusion_scores.query_fingerprint,
        "fusion_score_set_fingerprint": fusion_scores.score_set_fingerprint,
        "method_ranking_fingerprints": [
            ranking.ranking_fingerprint for ranking in method_rankings
        ],
        "top_candidate_keys_by_method": [list(item) for item in top_candidates],
        "cross_method_top1_agreement": agreement,
        "agreed_top_candidate_key": agreed_top,
        "method_selection_status": "not_selected",
    }
    return RawFusionRankingSet(
        schema_version=RAW_FUSION_RANKING_SET_VERSION,
        query_id=fusion_scores.query_id,
        query_fingerprint=fusion_scores.query_fingerprint,
        fusion_scores=fusion_scores,
        fusion_score_set_fingerprint=fusion_scores.score_set_fingerprint,
        method_rankings=tuple(method_rankings),
        top_candidate_keys_by_method=top_candidates,
        cross_method_top1_agreement=agreement,
        agreed_top_candidate_key=agreed_top,
        method_selection_status="not_selected",
        ranking_set_fingerprint=canonical_semantic_fingerprint(set_base),
    )


def _evaluate_raw_fusion_methods(
    components: DynamicScoreComponentSet,
    linear_parameters: ValidationLinearFusionParameters,
) -> RawFusionScoreSet:
    values_by_candidate = {
        candidate.candidate_accepted_taxon_key: _fusion_component_values(candidate)
        for candidate in components.candidates
    }
    rank_utilities = _rank_utilities(values_by_candidate)
    method_policies = tuple(
        (
            method,
            _method_policy_fingerprint(method, linear_parameters),
        )
        for method in RAW_FUSION_METHODS
    )
    scores: list[RawFusionCandidateScore] = []
    for method, policy_fingerprint in method_policies:
        for candidate in components.candidates:
            candidate_key = candidate.candidate_accepted_taxon_key
            values = values_by_candidate[candidate_key]
            (
                component_names,
                raw_component_values,
                method_component_values,
                raw_score,
            ) = _method_score(
                method,
                values=values,
                rank_utilities=rank_utilities[candidate_key],
                linear_parameters=linear_parameters,
            )
            base = {
                "schema_version": RAW_FUSION_CANDIDATE_SCORE_VERSION,
                "query_id": components.query_id,
                "query_fingerprint": components.query_fingerprint,
                "candidate_accepted_taxon_key": candidate_key,
                "candidate_scientific_name": candidate.candidate_scientific_name,
                "component_fingerprint": candidate.component_fingerprint,
                "method": method,
                "method_version": _METHOD_VERSIONS[method],
                "method_policy_fingerprint": policy_fingerprint,
                "local_evidence_status": candidate.local_evidence.score_status,
                "component_names_used": list(component_names),
                "raw_component_values": list(raw_component_values),
                "method_component_values": list(method_component_values),
                "raw_fusion_score": raw_score,
            }
            scores.append(
                RawFusionCandidateScore(
                    schema_version=RAW_FUSION_CANDIDATE_SCORE_VERSION,
                    query_id=components.query_id,
                    query_fingerprint=components.query_fingerprint,
                    candidate_accepted_taxon_key=candidate_key,
                    candidate_scientific_name=candidate.candidate_scientific_name,
                    component_fingerprint=candidate.component_fingerprint,
                    method=method,
                    method_version=_METHOD_VERSIONS[method],
                    method_policy_fingerprint=policy_fingerprint,
                    local_evidence_status=candidate.local_evidence.score_status,
                    component_names_used=component_names,
                    raw_component_values=raw_component_values,
                    method_component_values=method_component_values,
                    raw_fusion_score=raw_score,
                    score_fingerprint=canonical_semantic_fingerprint(base),
                )
            )
    set_base = {
        "schema_version": RAW_FUSION_SCORE_SET_VERSION,
        "query_id": components.query_id,
        "query_fingerprint": components.query_fingerprint,
        "component_set_fingerprint": components.component_set_fingerprint,
        "linear_parameters_fingerprint": linear_parameters.parameters_fingerprint,
        "methods": list(RAW_FUSION_METHODS),
        "method_policy_fingerprints": [list(item) for item in method_policies],
        "score_fingerprints": [score.score_fingerprint for score in scores],
    }
    return RawFusionScoreSet(
        schema_version=RAW_FUSION_SCORE_SET_VERSION,
        query_id=components.query_id,
        query_fingerprint=components.query_fingerprint,
        component_set=components,
        component_set_fingerprint=components.component_set_fingerprint,
        linear_parameters=linear_parameters,
        methods=RAW_FUSION_METHODS,
        method_policy_fingerprints=method_policies,
        scores=tuple(scores),
        score_set_fingerprint=canonical_semantic_fingerprint(set_base),
    )


def _fusion_component_values(
    candidate: DynamicCandidateScoreComponents,
) -> dict[str, float]:
    values = {
        "global_prototype_similarity": candidate.global_evidence.prototype_similarity,
        "global_nearest_reference_similarity": (
            candidate.global_evidence.nearest_reference_similarity
        ),
        "global_top_k_mean_similarity": (
            candidate.global_evidence.top_k_mean_similarity
        ),
    }
    if candidate.local_evidence.score_status == "available":
        local_values = {
            "local_prototype_similarity": (
                candidate.local_evidence.prototype_similarity
            ),
            "local_nearest_reference_similarity": (
                candidate.local_evidence.nearest_reference_similarity
            ),
            "local_top_k_mean_similarity": (
                candidate.local_evidence.top_k_mean_similarity
            ),
        }
        if any(value is None for value in local_values.values()):
            raise ValueError("available local fusion components must not be null")
        values.update(
            {
                name: _finite_number(value, field=name)
                for name, value in local_values.items()
            }
        )
    elif candidate.local_evidence.score_status != "unavailable":
        raise ValueError("unsupported local fusion evidence status")
    return {name: _finite_number(value, field=name) for name, value in values.items()}


def _method_score(
    method: str,
    *,
    values: dict[str, float],
    rank_utilities: dict[str, float],
    linear_parameters: ValidationLinearFusionParameters,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...], float]:
    names = tuple(name for name in FUSION_COMPONENTS if name in values)
    raw_component_values = tuple(values[name] for name in names)
    method_component_values = raw_component_values
    if method == UNWEIGHTED_COMPONENT_MEAN:
        score = fsum(raw_component_values) / len(raw_component_values)
    elif method == VALIDATION_FITTED_LINEAR:
        if names == FUSION_COMPONENTS:
            weights = linear_parameters.full_weights
            intercept = linear_parameters.full_intercept
        elif names == GLOBAL_FUSION_COMPONENTS:
            weights = linear_parameters.global_only_weights
            intercept = linear_parameters.global_only_intercept
        else:
            raise ValueError("linear fusion received an unsupported component subset")
        score = intercept + fsum(
            weight * value
            for weight, value in zip(weights, raw_component_values, strict=True)
        )
    elif method == MAXIMUM_SCOPE_EVIDENCE:
        global_mean = fsum(values[name] for name in GLOBAL_FUSION_COMPONENTS) / len(
            GLOBAL_FUSION_COMPONENTS
        )
        scope_means = [global_mean]
        if all(name in values for name in LOCAL_FUSION_COMPONENTS):
            scope_means.append(
                fsum(values[name] for name in LOCAL_FUSION_COMPONENTS)
                / len(LOCAL_FUSION_COMPONENTS)
            )
        score = max(scope_means)
    elif method == ROBUST_RANK_AGGREGATION:
        score = median(rank_utilities[name] for name in names)
        method_component_values = tuple(rank_utilities[name] for name in names)
    else:
        raise ValueError(f"unsupported raw fusion method: {method}")
    return (
        names,
        raw_component_values,
        method_component_values,
        _finite_number(score, field=f"{method} score"),
    )


def _rank_utilities(
    values_by_candidate: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    result = {candidate_key: {} for candidate_key in values_by_candidate}
    for component_name in FUSION_COMPONENTS:
        ranked = sorted(
            (
                (values[component_name], candidate_key)
                for candidate_key, values in values_by_candidate.items()
                if component_name in values
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked:
            continue
        index = 0
        while index < len(ranked):
            end = index + 1
            while end < len(ranked) and ranked[end][0] == ranked[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2
            utility = (
                1.0
                if len(ranked) == 1
                else 1.0 - (average_rank - 1.0) / (len(ranked) - 1.0)
            )
            for _, candidate_key in ranked[index:end]:
                result[candidate_key][component_name] = utility
            index = end
    return result


def _method_policy_fingerprint(
    method: str,
    linear_parameters: ValidationLinearFusionParameters,
) -> str:
    base: dict[str, object] = {
        "method": method,
        "method_version": _METHOD_VERSIONS[method],
        "full_component_order": list(FUSION_COMPONENTS),
        "local_unavailable_semantics": "omit_local_without_imputation",
    }
    if method == UNWEIGHTED_COMPONENT_MEAN:
        base["semantics"] = "arithmetic_mean_of_available_raw_components"
    elif method == VALIDATION_FITTED_LINEAR:
        base["semantics"] = "explicit_validation_fitted_linear_combination"
        base["parameters_fingerprint"] = linear_parameters.parameters_fingerprint
    elif method == MAXIMUM_SCOPE_EVIDENCE:
        base["semantics"] = "maximum_of_global_and_available_local_scope_means"
    elif method == ROBUST_RANK_AGGREGATION:
        base.update(
            {
                "semantics": "median_of_tie_aware_normalized_component_rank_utilities",
                "rank_ties": "average_rank",
                "rank_utility": "one_minus_rank_minus_one_over_population_minus_one",
            }
        )
    else:
        raise ValueError(f"unsupported raw fusion method: {method}")
    return canonical_semantic_fingerprint(base)


def _validate_source_evidence(
    family_evidence: RawFamilyEvidenceSet,
    global_evidence: RawGlobalReferenceEvidenceSet,
    local_evidence: RawLocalReferenceEvidenceSet,
    disagreement_coverage: RawDisagreementCoverageSet,
) -> None:
    if not isinstance(family_evidence, RawFamilyEvidenceSet):
        raise TypeError("family_evidence must be RawFamilyEvidenceSet")
    if not isinstance(global_evidence, RawGlobalReferenceEvidenceSet):
        raise TypeError("global_evidence must be RawGlobalReferenceEvidenceSet")
    if not isinstance(local_evidence, RawLocalReferenceEvidenceSet):
        raise TypeError("local_evidence must be RawLocalReferenceEvidenceSet")
    if not isinstance(disagreement_coverage, RawDisagreementCoverageSet):
        raise TypeError("disagreement_coverage must be RawDisagreementCoverageSet")
    if family_evidence.query_id != global_evidence.query_id:
        raise ValueError("family/global evidence query IDs differ")
    if family_evidence.query_fingerprint != global_evidence.query_fingerprint:
        raise ValueError("family/global evidence query fingerprints differ")
    _validate_family_evidence(family_evidence)
    _validate_reference_evidence_set_fingerprints(global_evidence, local_evidence)
    _validate_candidate_source_rows(
        global_evidence,
        local_evidence,
        disagreement_coverage,
    )
    expected_disagreement = calculate_dynamic_pool_disagreement_coverage(
        global_evidence,
        local_evidence,
    )
    if disagreement_coverage != expected_disagreement:
        raise ValueError("disagreement coverage does not match global/local evidence")


def _validate_reference_evidence_set_fingerprints(
    global_evidence: RawGlobalReferenceEvidenceSet,
    local_evidence: RawLocalReferenceEvidenceSet,
) -> None:
    for label, evidence in (
        ("global", global_evidence),
        ("local", local_evidence),
    ):
        base = {
            "schema_version": evidence.schema_version,
            "query_id": evidence.query_id,
            "query_fingerprint": evidence.query_fingerprint,
            "candidate_matrix_signature": evidence.candidate_matrix_signature,
            "candidate_set_fingerprint": evidence.candidate_set_fingerprint,
            "score_fingerprints": [
                score.score_fingerprint for score in evidence.scores
            ],
        }
        if evidence.score_set_fingerprint != canonical_semantic_fingerprint(base):
            raise ValueError(f"{label} evidence set fingerprint mismatch")


def _validate_candidate_source_rows(
    global_evidence: RawGlobalReferenceEvidenceSet,
    local_evidence: RawLocalReferenceEvidenceSet,
    disagreement_coverage: RawDisagreementCoverageSet,
) -> None:
    if not (
        len(global_evidence.scores)
        == len(local_evidence.scores)
        == len(disagreement_coverage.scores)
    ):
        raise ValueError("source evidence row counts differ")
    for global_score, local_score, disagreement in zip(
        global_evidence.scores,
        local_evidence.scores,
        disagreement_coverage.scores,
        strict=True,
    ):
        if global_score.schema_version != RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION:
            raise ValueError("unsupported global evidence row version")
        if local_score.schema_version != RAW_LOCAL_REFERENCE_EVIDENCE_VERSION:
            raise ValueError("unsupported local evidence row version")
        if disagreement.schema_version != RAW_DISAGREEMENT_COVERAGE_VERSION:
            raise ValueError("unsupported disagreement coverage row version")
        if not (
            global_score.query_id
            == local_score.query_id
            == disagreement.query_id
            == global_evidence.query_id
        ):
            raise ValueError("candidate evidence row query identities differ")
        if not (
            global_score.candidate_matrix_signature
            == local_score.candidate_matrix_signature
            == global_evidence.candidate_matrix_signature
        ):
            raise ValueError("candidate evidence row matrix identities differ")
        if not (
            global_score.candidate_accepted_taxon_key
            == local_score.candidate_accepted_taxon_key
            == disagreement.candidate_accepted_taxon_key
        ):
            raise ValueError("candidate evidence row taxon identities differ")
        if not (
            global_score.candidate_scientific_name
            == local_score.candidate_scientific_name
            == disagreement.candidate_scientific_name
        ):
            raise ValueError("candidate evidence row scientific names differ")
        if disagreement.global_score_fingerprint != global_score.score_fingerprint:
            raise ValueError("disagreement row global source fingerprint differs")
        if disagreement.local_score_fingerprint != local_score.score_fingerprint:
            raise ValueError("disagreement row local source fingerprint differs")


def _validate_family_evidence(family_evidence: RawFamilyEvidenceSet) -> None:
    if family_evidence.schema_version != RAW_FAMILY_EVIDENCE_SET_VERSION:
        raise ValueError("unsupported family evidence set version")
    scores = tuple(family_evidence.scores)
    if not scores:
        raise ValueError("family evidence set must not be empty")
    if any(not isinstance(score, RawFamilyEvidence) for score in scores):
        raise TypeError("family evidence set contains invalid rows")
    expected_order = sorted(
        scores, key=lambda score: (-score.raw_similarity, score.family_key)
    )
    if list(scores) != expected_order:
        raise ValueError("family evidence scores are not canonically ordered")
    for index, score in enumerate(scores):
        if score.schema_version != RAW_FAMILY_EVIDENCE_VERSION:
            raise ValueError("unsupported family evidence version")
        if score.query_id != family_evidence.query_id:
            raise ValueError("family score query identity differs from its set")
        if score.family_matrix_signature != family_evidence.family_matrix_signature:
            raise ValueError("family score matrix identity differs from its set")
        if score.family_partition != family_evidence.family_partition:
            raise ValueError("family score partition differs from its set")
        if score.family_rank != index + 1:
            raise ValueError("family evidence ranks are not contiguous")
        if (
            not isfinite(score.raw_similarity)
            or not -1.0 <= score.raw_similarity <= 1.0
        ):
            raise ValueError("family raw similarity must be a finite cosine")
        expected_margin = (
            score.raw_similarity - scores[index + 1].raw_similarity
            if index + 1 < len(scores)
            else None
        )
        if score.margin_to_next_raw != expected_margin:
            raise ValueError("family evidence margin does not match raw similarities")
        base = {
            "schema_version": RAW_FAMILY_EVIDENCE_VERSION,
            "query_id": family_evidence.query_id,
            "query_fingerprint": family_evidence.query_fingerprint,
            "family_matrix_signature": family_evidence.family_matrix_signature,
            "family_partition": family_evidence.family_partition,
            "family_key": score.family_key,
            "family_name": score.family_name,
            "family_prototype_fingerprint": score.family_prototype_fingerprint,
            "raw_similarity": score.raw_similarity,
            "family_rank": score.family_rank,
            "margin_to_next_raw": score.margin_to_next_raw,
        }
        if score.score_fingerprint != canonical_semantic_fingerprint(base):
            raise ValueError("family evidence score fingerprint mismatch")
    set_base = {
        "schema_version": RAW_FAMILY_EVIDENCE_SET_VERSION,
        "query_id": family_evidence.query_id,
        "query_fingerprint": family_evidence.query_fingerprint,
        "family_matrix_signature": family_evidence.family_matrix_signature,
        "family_partition": family_evidence.family_partition,
        "score_fingerprints": [score.score_fingerprint for score in scores],
    }
    if family_evidence.score_set_fingerprint != canonical_semantic_fingerprint(
        set_base
    ):
        raise ValueError("family evidence set fingerprint mismatch")


def _candidate_components(
    *,
    query_id: str,
    query_fingerprint: str,
    global_score: RawGlobalReferenceEvidence,
    local_score: RawLocalReferenceEvidence,
    disagreement: RawCandidateDisagreementCoverage,
) -> DynamicCandidateScoreComponents:
    if not (
        global_score.candidate_accepted_taxon_key
        == local_score.candidate_accepted_taxon_key
        == disagreement.candidate_accepted_taxon_key
    ):
        raise ValueError("candidate component taxon identities differ")
    if not (
        global_score.candidate_scientific_name
        == local_score.candidate_scientific_name
        == disagreement.candidate_scientific_name
    ):
        raise ValueError("candidate component scientific names differ")
    base = {
        "schema_version": DYNAMIC_SCORE_COMPONENT_VERSION,
        "query_id": query_id,
        "query_fingerprint": query_fingerprint,
        "candidate_accepted_taxon_key": global_score.candidate_accepted_taxon_key,
        "candidate_scientific_name": global_score.candidate_scientific_name,
        "global_score_fingerprint": global_score.score_fingerprint,
        "local_score_fingerprint": local_score.score_fingerprint,
        "disagreement_coverage_fingerprint": disagreement.evidence_fingerprint,
    }
    return DynamicCandidateScoreComponents(
        schema_version=DYNAMIC_SCORE_COMPONENT_VERSION,
        query_id=query_id,
        query_fingerprint=query_fingerprint,
        candidate_accepted_taxon_key=global_score.candidate_accepted_taxon_key,
        candidate_scientific_name=global_score.candidate_scientific_name,
        global_evidence=global_score,
        local_evidence=local_score,
        disagreement_coverage=disagreement,
        component_fingerprint=canonical_semantic_fingerprint(base),
    )


def _component_set_base(
    *,
    query_id: str,
    query_fingerprint: str,
    candidate_matrix_signature: str,
    candidate_set_fingerprint: str,
    family_evidence: RawFamilyEvidenceSet,
    global_evidence_set_fingerprint: str,
    local_evidence_set_fingerprint: str,
    disagreement_coverage_set_fingerprint: str,
    candidates: tuple[DynamicCandidateScoreComponents, ...],
) -> dict[str, object]:
    return {
        "schema_version": DYNAMIC_SCORE_COMPONENT_SET_VERSION,
        "query_id": query_id,
        "query_fingerprint": query_fingerprint,
        "candidate_matrix_signature": candidate_matrix_signature,
        "candidate_set_fingerprint": candidate_set_fingerprint,
        "family_evidence_set_fingerprint": family_evidence.score_set_fingerprint,
        "global_evidence_set_fingerprint": global_evidence_set_fingerprint,
        "local_evidence_set_fingerprint": local_evidence_set_fingerprint,
        "disagreement_coverage_set_fingerprint": (
            disagreement_coverage_set_fingerprint
        ),
        "candidate_component_fingerprints": [
            candidate.component_fingerprint for candidate in candidates
        ],
    }


def _weights(
    values: object,
    *,
    expected_count: int,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    weights = tuple(
        _finite_number(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    if len(weights) != expected_count:
        raise ValueError(f"{field} must contain exactly {expected_count} values")
    if not any(weight != 0.0 for weight in weights):
        raise ValueError(f"{field} must contain at least one nonzero value")
    return weights


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


__all__ = [
    "DYNAMIC_SCORE_COMPONENT_SET_VERSION",
    "DYNAMIC_SCORE_COMPONENT_VERSION",
    "FUSION_COMPONENTS",
    "GLOBAL_FUSION_COMPONENTS",
    "LOCAL_FUSION_COMPONENTS",
    "MAXIMUM_SCOPE_EVIDENCE",
    "RAW_FUSION_CANDIDATE_SCORE_VERSION",
    "RAW_FUSION_METHODS",
    "RAW_FUSION_METHOD_RANKING_VERSION",
    "RAW_FUSION_RANKING_SET_VERSION",
    "RAW_FUSION_SCORE_SET_VERSION",
    "RANKED_FUSION_CANDIDATE_VERSION",
    "ROBUST_RANK_AGGREGATION",
    "UNWEIGHTED_COMPONENT_MEAN",
    "VALIDATION_FITTED_LINEAR",
    "VALIDATION_LINEAR_FUSION_PARAMETERS_VERSION",
    "DynamicCandidateScoreComponents",
    "DynamicScoreComponentSet",
    "RawFusionCandidateScore",
    "RawFusionMethodRanking",
    "RawFusionRankingSet",
    "RawFusionScoreSet",
    "RankedFusionCandidate",
    "ValidationLinearFusionParameters",
    "evaluate_raw_fusion_methods",
    "preserve_dynamic_score_components",
    "rank_raw_fusion_candidates",
    "validate_dynamic_score_components",
    "validate_raw_fusion_rankings",
    "validate_raw_fusion_scores",
]
