"""Versioned, component-preserving inputs for raw dynamic-pool fusion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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


__all__ = [
    "DYNAMIC_SCORE_COMPONENT_SET_VERSION",
    "DYNAMIC_SCORE_COMPONENT_VERSION",
    "DynamicCandidateScoreComponents",
    "DynamicScoreComponentSet",
    "preserve_dynamic_score_components",
    "validate_dynamic_score_components",
]
