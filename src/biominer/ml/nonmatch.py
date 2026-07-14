"""Explicit competitor and non-match evidence aggregation for target screening."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.calibration import CALIBRATED_PROBABILITY_KIND
from biominer.ml.training_features import TARGET_TASKS
from biominer.references.readiness import REFERENCE_ROUTES


NONMATCH_SCORING_VERSION = "regional-nonmatch-scoring-v1.0.0"
REFERENCE_SCORE_KIND_COSINE_SIMILARITY = "cosine_similarity"
HETEROGENEOUS_EVIDENCE_SCALE = (
    "heterogeneous_reference_similarity_and_calibrated_probability"
)
REFERENCE_EVIDENCE_SCALE = "reference_similarity"
CALIBRATED_PROBABILITY_SCALE = CALIBRATED_PROBABILITY_KIND

TARGET_CONFIRMED = "target_confirmed"
TARGET_PROBABLE_REVIEW = "target_probable_review"
KNOWN_REGIONAL_COMPETITOR = "known_regional_competitor"
KNOWN_NONREGIONAL_COMPETITOR = "known_nonregional_competitor"
OTHER_BUTTERFLY = "other_butterfly"
NON_BUTTERFLY_INSECT = "non_butterfly_insect"
PINNED_SPECIMEN = "pinned_specimen"
VISUAL_ARTIFACT = "visual_artifact"
OUT_OF_DISTRIBUTION = "out_of_distribution"
INSUFFICIENT_VISUAL_DETAIL = "insufficient_visual_detail"
INSUFFICIENT_REFERENCE_COVERAGE = "insufficient_reference_coverage"
NO_GEO_GLOBAL_FALLBACK = "no_geo_global_fallback"
ABSTAIN = "abstain"

CLASSIFICATION_OUTCOMES = frozenset(
    {
        TARGET_CONFIRMED,
        TARGET_PROBABLE_REVIEW,
        KNOWN_REGIONAL_COMPETITOR,
        KNOWN_NONREGIONAL_COMPETITOR,
        OTHER_BUTTERFLY,
        NON_BUTTERFLY_INSECT,
        PINNED_SPECIMEN,
        VISUAL_ARTIFACT,
        OUT_OF_DISTRIBUTION,
        INSUFFICIENT_VISUAL_DETAIL,
        INSUFFICIENT_REFERENCE_COVERAGE,
        NO_GEO_GLOBAL_FALLBACK,
        ABSTAIN,
    }
)
DOMAIN_NEGATIVE_OUTCOMES = frozenset(
    {OTHER_BUTTERFLY, NON_BUTTERFLY_INSECT, PINNED_SPECIMEN, VISUAL_ARTIFACT}
)
GEOGRAPHIC_SCOPES = frozenset({"regional", "nonregional", "no_geo_global"})

_TARGET_PROBABILITY_TASKS = frozenset(
    {"binary_target_verifier", "larval_target_verifier", "regional_multiclass"}
)
_GENERIC_NON_TARGET_TASKS = frozenset(
    {"binary_target_verifier", "larval_target_verifier"}
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CandidateNonMatchEvidence:
    """Reference and calibrated evidence for one target or competitor species."""

    evidence_id: str
    accepted_taxon_key: str
    scientific_name: str
    geographic_scope: str
    candidate_reasons: tuple[str, ...]
    reference_score: float | None
    reference_score_kind: str | None
    score_contract_fingerprint: str | None
    calibrated_probability: float | None = None
    probability_task: str | None = None
    classifier_fingerprint: str | None = None
    calibrator_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DomainNegativeEvidence:
    """Scored evidence for one closed visual-domain non-target category."""

    evidence_id: str
    outcome: str
    reason: str
    reference_score: float | None
    reference_score_kind: str | None
    score_contract_fingerprint: str | None
    calibrated_probability: float | None = None
    probability_task: str | None = None
    classifier_fingerprint: str | None = None
    calibrator_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class CalibratedNonTargetEvidence:
    """Generic calibrated non-target evidence without fabricated taxon identity."""

    evidence_id: str
    calibrated_probability: float
    probability_task: str
    classifier_fingerprint: str
    calibrator_fingerprint: str


@dataclass(frozen=True, slots=True)
class NonMatchScoringRequest:
    """Complete comparable evidence for one routed target-aware scoring unit."""

    query_id: str
    route: str
    target_accepted_taxon_key: str
    candidate_set_fingerprint: str
    candidates: tuple[CandidateNonMatchEvidence, ...]
    domain_negatives: tuple[DomainNegativeEvidence, ...] = ()
    non_target_classifier: CalibratedNonTargetEvidence | None = None


@dataclass(frozen=True, slots=True)
class ExplicitNonMatchScore:
    """Required maxima and margins with the winning evidence identities retained."""

    scoring_version: str
    query_id: str
    route: str
    target_accepted_taxon_key: str
    target_scientific_name: str
    candidate_set_fingerprint: str
    best_target_reference_score: float | None
    best_target_reference_evidence_id: str | None
    best_known_competitor_score: float | None
    best_known_competitor_accepted_taxon_key: str | None
    best_known_competitor_scientific_name: str | None
    best_known_competitor_evidence_id: str | None
    best_known_competitor_outcome: str | None
    best_known_competitor_reasons: tuple[str, ...]
    calibrated_best_competitor_probability: float | None
    calibrated_best_competitor_accepted_taxon_key: str | None
    calibrated_best_competitor_scientific_name: str | None
    best_competitor_calibrator_fingerprint: str | None
    best_domain_negative_score: float | None
    best_domain_negative_evidence_id: str | None
    best_domain_negative_outcome: str | None
    best_domain_negative_reason: str | None
    calibrated_best_domain_negative_probability: float | None
    best_domain_calibrator_fingerprint: str | None
    calibrated_non_target_probability: float | None
    non_target_calibrator_fingerprint: str | None
    best_non_target_score: float | None
    best_non_target_score_kind: str | None
    best_non_target_score_scale: str | None
    best_non_target_score_is_calibrated_probability: bool
    best_non_target_evidence_kind: str | None
    best_non_target_evidence_id: str | None
    raw_score_contract_fingerprint: str | None
    competitor_margin: float | None
    calibrated_target_probability: float | None
    target_calibrator_fingerprint: str | None
    calibrated_best_non_target_probability: float | None
    calibrated_best_non_target_evidence_id: str | None
    calibrated_best_non_target_outcome: str | None
    calibrated_best_non_target_accepted_taxon_key: str | None
    nonmatch_margin: float | None
    leading_nonmatch_outcome: str
    leading_nonmatch_evidence_id: str | None
    leading_nonmatch_accepted_taxon_key: str | None
    leading_nonmatch_reason: str
    candidate_evidence_count: int
    domain_negative_evidence_count: int
    scoring_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ScoredEvidence:
    score: float
    score_kind: str
    evidence_kind: str
    evidence_id: str
    outcome: str
    reason: str
    accepted_taxon_key: str | None = None


@dataclass(frozen=True, slots=True)
class _CalibratedEvidence:
    probability: float
    evidence_kind: str
    evidence_id: str
    outcome: str
    reason: str
    calibrator_fingerprint: str
    accepted_taxon_key: str | None = None


def score_nonmatch_evidence(request: NonMatchScoringRequest) -> ExplicitNonMatchScore:
    """Compute explicit raw and calibrated non-match maxima without thresholds."""

    normalized = _validate_request(request)
    candidates = normalized["candidates"]
    domains = normalized["domain_negatives"]
    generic = normalized["non_target_classifier"]
    target_key = str(normalized["target_accepted_taxon_key"])
    assert isinstance(candidates, tuple)
    assert isinstance(domains, tuple)
    assert generic is None or isinstance(generic, CalibratedNonTargetEvidence)
    target = next(item for item in candidates if item.accepted_taxon_key == target_key)
    competitors = tuple(
        item for item in candidates if item.accepted_taxon_key != target_key
    )
    raw_score_contract = next(
        (
            item.score_contract_fingerprint
            for item in (*candidates, *domains)
            if item.reference_score is not None
        ),
        None,
    )

    target_raw = (
        None
        if target.reference_score is None
        else _ScoredEvidence(
            score=target.reference_score,
            score_kind=str(target.reference_score_kind),
            evidence_kind="target_reference",
            evidence_id=target.evidence_id,
            outcome=TARGET_PROBABLE_REVIEW,
            reason="target_reference_evidence",
            accepted_taxon_key=target.accepted_taxon_key,
        )
    )
    competitor_raw = _best_raw_candidate(competitors)
    domain_raw = _best_raw_domain(domains)
    generic_raw = (
        None
        if generic is None
        else _ScoredEvidence(
            score=generic.calibrated_probability,
            score_kind=CALIBRATED_PROBABILITY_KIND,
            evidence_kind="generic_non_target_classifier",
            evidence_id=generic.evidence_id,
            outcome=ABSTAIN,
            reason="unresolved_calibrated_non_target",
        )
    )
    heterogeneous_components = tuple(
        item for item in (competitor_raw, domain_raw, generic_raw) if item is not None
    )
    best_non_target = _best_scored_evidence(heterogeneous_components)
    component_kinds = {item.score_kind for item in heterogeneous_components}
    if not heterogeneous_components:
        score_scale = None
    elif len(component_kinds) > 1:
        score_scale = HETEROGENEOUS_EVIDENCE_SCALE
    elif component_kinds == {CALIBRATED_PROBABILITY_KIND}:
        score_scale = CALIBRATED_PROBABILITY_SCALE
    else:
        score_scale = REFERENCE_EVIDENCE_SCALE

    target_calibrated = _candidate_calibrated(target, outcome=TARGET_PROBABLE_REVIEW)
    competitor_calibrated = _best_calibrated_candidate(competitors)
    domain_calibrated = _best_calibrated_domain(domains)
    generic_calibrated = (
        None
        if generic is None
        else _CalibratedEvidence(
            probability=generic.calibrated_probability,
            evidence_kind="generic_non_target_classifier",
            evidence_id=generic.evidence_id,
            outcome=ABSTAIN,
            reason="unresolved_calibrated_non_target",
            calibrator_fingerprint=generic.calibrator_fingerprint,
        )
    )
    best_calibrated_non_target = _best_calibrated_evidence(
        tuple(
            item
            for item in (
                competitor_calibrated,
                domain_calibrated,
                generic_calibrated,
            )
            if item is not None
        )
    )
    competitor_margin = _difference(
        target_raw.score if target_raw is not None else None,
        competitor_raw.score if competitor_raw is not None else None,
    )
    nonmatch_margin = _difference(
        target_calibrated.probability if target_calibrated is not None else None,
        (
            best_calibrated_non_target.probability
            if best_calibrated_non_target is not None
            else None
        ),
    )
    leading = best_calibrated_non_target or (
        None
        if best_non_target is None
        else _CalibratedEvidence(
            probability=best_non_target.score,
            evidence_kind=best_non_target.evidence_kind,
            evidence_id=best_non_target.evidence_id,
            outcome=best_non_target.outcome,
            reason=best_non_target.reason,
            calibrator_fingerprint="",
            accepted_taxon_key=best_non_target.accepted_taxon_key,
        )
    )

    result_values: dict[str, object] = {
        "scoring_version": NONMATCH_SCORING_VERSION,
        "query_id": normalized["query_id"],
        "route": normalized["route"],
        "target_accepted_taxon_key": target_key,
        "target_scientific_name": target.scientific_name,
        "candidate_set_fingerprint": normalized["candidate_set_fingerprint"],
        "best_target_reference_score": (
            target_raw.score if target_raw is not None else None
        ),
        "best_target_reference_evidence_id": (
            target_raw.evidence_id if target_raw is not None else None
        ),
        "best_known_competitor_score": (
            competitor_raw.score if competitor_raw is not None else None
        ),
        "best_known_competitor_accepted_taxon_key": (
            competitor_raw.accepted_taxon_key if competitor_raw is not None else None
        ),
        "best_known_competitor_scientific_name": (
            _candidate_name(competitors, competitor_raw) if competitor_raw else None
        ),
        "best_known_competitor_evidence_id": (
            competitor_raw.evidence_id if competitor_raw is not None else None
        ),
        "best_known_competitor_outcome": (
            competitor_raw.outcome if competitor_raw is not None else None
        ),
        "best_known_competitor_reasons": (
            _candidate_reasons(competitors, competitor_raw) if competitor_raw else ()
        ),
        "calibrated_best_competitor_probability": (
            competitor_calibrated.probability
            if competitor_calibrated is not None
            else None
        ),
        "calibrated_best_competitor_accepted_taxon_key": (
            competitor_calibrated.accepted_taxon_key
            if competitor_calibrated is not None
            else None
        ),
        "calibrated_best_competitor_scientific_name": (
            _candidate_name_by_evidence_id(
                competitors,
                competitor_calibrated.evidence_id,
            )
            if competitor_calibrated is not None
            else None
        ),
        "best_competitor_calibrator_fingerprint": (
            competitor_calibrated.calibrator_fingerprint
            if competitor_calibrated is not None
            else None
        ),
        "best_domain_negative_score": (
            domain_raw.score if domain_raw is not None else None
        ),
        "best_domain_negative_evidence_id": (
            domain_raw.evidence_id if domain_raw is not None else None
        ),
        "best_domain_negative_outcome": (
            domain_raw.outcome if domain_raw is not None else None
        ),
        "best_domain_negative_reason": (
            domain_raw.reason if domain_raw is not None else None
        ),
        "calibrated_best_domain_negative_probability": (
            domain_calibrated.probability if domain_calibrated is not None else None
        ),
        "best_domain_calibrator_fingerprint": (
            domain_calibrated.calibrator_fingerprint
            if domain_calibrated is not None
            else None
        ),
        "calibrated_non_target_probability": (
            generic.calibrated_probability if generic is not None else None
        ),
        "non_target_calibrator_fingerprint": (
            generic.calibrator_fingerprint if generic is not None else None
        ),
        "best_non_target_score": (
            best_non_target.score if best_non_target is not None else None
        ),
        "best_non_target_score_kind": (
            best_non_target.score_kind if best_non_target is not None else None
        ),
        "best_non_target_score_scale": score_scale,
        "best_non_target_score_is_calibrated_probability": bool(
            best_non_target is not None
            and best_non_target.score_kind == CALIBRATED_PROBABILITY_KIND
        ),
        "best_non_target_evidence_kind": (
            best_non_target.evidence_kind if best_non_target is not None else None
        ),
        "best_non_target_evidence_id": (
            best_non_target.evidence_id if best_non_target is not None else None
        ),
        "raw_score_contract_fingerprint": raw_score_contract,
        "competitor_margin": competitor_margin,
        "calibrated_target_probability": (
            target_calibrated.probability if target_calibrated is not None else None
        ),
        "target_calibrator_fingerprint": (
            target_calibrated.calibrator_fingerprint
            if target_calibrated is not None
            else None
        ),
        "calibrated_best_non_target_probability": (
            best_calibrated_non_target.probability
            if best_calibrated_non_target is not None
            else None
        ),
        "calibrated_best_non_target_evidence_id": (
            best_calibrated_non_target.evidence_id
            if best_calibrated_non_target is not None
            else None
        ),
        "calibrated_best_non_target_outcome": (
            best_calibrated_non_target.outcome
            if best_calibrated_non_target is not None
            else None
        ),
        "calibrated_best_non_target_accepted_taxon_key": (
            best_calibrated_non_target.accepted_taxon_key
            if best_calibrated_non_target is not None
            else None
        ),
        "nonmatch_margin": nonmatch_margin,
        "leading_nonmatch_outcome": leading.outcome if leading else ABSTAIN,
        "leading_nonmatch_evidence_id": leading.evidence_id if leading else None,
        "leading_nonmatch_accepted_taxon_key": (
            leading.accepted_taxon_key if leading else None
        ),
        "leading_nonmatch_reason": (
            leading.reason if leading else "no_non_target_evidence"
        ),
        "candidate_evidence_count": len(candidates),
        "domain_negative_evidence_count": len(domains),
    }
    fingerprint = canonical_semantic_fingerprint(
        {
            "scoring": result_values,
            "candidate_evidence": [asdict(item) for item in candidates],
            "domain_negative_evidence": [asdict(item) for item in domains],
            "generic_non_target_evidence": (
                asdict(generic) if generic is not None else None
            ),
        }
    )
    return ExplicitNonMatchScore(
        **result_values,
        scoring_fingerprint=fingerprint,
    )


def _validate_request(request: NonMatchScoringRequest) -> dict[str, object]:
    if not isinstance(request, NonMatchScoringRequest):
        raise TypeError("request must be a NonMatchScoringRequest")
    query_id = _required_text(request.query_id, field="query_id")
    route = _required_choice(request.route, field="route", allowed=REFERENCE_ROUTES)
    target_key = _required_text(
        request.target_accepted_taxon_key,
        field="target_accepted_taxon_key",
    )
    candidate_set = _sha256(
        request.candidate_set_fingerprint,
        field="candidate_set_fingerprint",
    )
    candidates = tuple(
        _validate_candidate(item, target_key=target_key) for item in request.candidates
    )
    if not candidates:
        raise ValueError("non-match scoring requires candidate evidence")
    candidate_keys = [item.accepted_taxon_key for item in candidates]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate taxon keys must be unique")
    if sum(key == target_key for key in candidate_keys) != 1:
        raise ValueError("candidate evidence must contain exactly one target")
    domains = tuple(_validate_domain(item) for item in request.domain_negatives)
    generic = (
        None
        if request.non_target_classifier is None
        else _validate_generic_non_target(request.non_target_classifier)
    )
    evidence_ids = [item.evidence_id for item in candidates]
    evidence_ids.extend(item.evidence_id for item in domains)
    if generic is not None:
        evidence_ids.append(generic.evidence_id)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("non-match evidence IDs must be globally unique")
    raw_contracts = {
        item.score_contract_fingerprint
        for item in (*candidates, *domains)
        if item.reference_score is not None
    }
    if len(raw_contracts) > 1:
        raise ValueError("all raw score contracts must match before taking maxima")
    return {
        "query_id": query_id,
        "route": route,
        "target_accepted_taxon_key": target_key,
        "candidate_set_fingerprint": candidate_set,
        "candidates": tuple(
            sorted(
                candidates,
                key=lambda item: (item.accepted_taxon_key, item.evidence_id),
            )
        ),
        "domain_negatives": tuple(
            sorted(domains, key=lambda item: (item.outcome, item.evidence_id))
        ),
        "non_target_classifier": generic,
    }


def _validate_candidate(
    value: CandidateNonMatchEvidence,
    *,
    target_key: str,
) -> CandidateNonMatchEvidence:
    if not isinstance(value, CandidateNonMatchEvidence):
        raise TypeError("candidates must contain CandidateNonMatchEvidence values")
    evidence_id = _required_text(value.evidence_id, field="candidate.evidence_id")
    key = _required_text(
        value.accepted_taxon_key,
        field="candidate.accepted_taxon_key",
    )
    name = _required_text(value.scientific_name, field="candidate.scientific_name")
    geographic_scope = _required_choice(
        value.geographic_scope,
        field="candidate.geographic_scope",
        allowed=GEOGRAPHIC_SCOPES,
    )
    reasons = _sorted_unique_text(
        value.candidate_reasons,
        field="candidate.candidate_reasons",
    )
    if not reasons:
        raise ValueError("candidate.candidate_reasons must not be empty")
    score, score_kind, score_contract = _validate_reference_score(
        value.reference_score,
        value.reference_score_kind,
        value.score_contract_fingerprint,
        field="candidate",
    )
    probability, task, classifier, calibrator = _validate_probability(
        value.calibrated_probability,
        value.probability_task,
        value.classifier_fingerprint,
        value.calibrator_fingerprint,
        field="candidate",
    )
    if probability is not None:
        if key == target_key:
            if task not in _TARGET_PROBABILITY_TASKS:
                raise ValueError("target probability task is incompatible")
        elif task != "regional_multiclass":
            raise ValueError(
                "competitor probabilities require probability_task='regional_multiclass'"
            )
    return CandidateNonMatchEvidence(
        evidence_id=evidence_id,
        accepted_taxon_key=key,
        scientific_name=name,
        geographic_scope=geographic_scope,
        candidate_reasons=reasons,
        reference_score=score,
        reference_score_kind=score_kind,
        score_contract_fingerprint=score_contract,
        calibrated_probability=probability,
        probability_task=task,
        classifier_fingerprint=classifier,
        calibrator_fingerprint=calibrator,
    )


def _validate_domain(value: DomainNegativeEvidence) -> DomainNegativeEvidence:
    if not isinstance(value, DomainNegativeEvidence):
        raise TypeError("domain_negatives must contain DomainNegativeEvidence values")
    evidence_id = _required_text(value.evidence_id, field="domain.evidence_id")
    outcome = _required_choice(
        value.outcome,
        field="domain.outcome",
        allowed=DOMAIN_NEGATIVE_OUTCOMES,
    )
    reason = _required_text(value.reason, field="domain.reason")
    score, score_kind, score_contract = _validate_reference_score(
        value.reference_score,
        value.reference_score_kind,
        value.score_contract_fingerprint,
        field="domain",
    )
    probability, task, classifier, calibrator = _validate_probability(
        value.calibrated_probability,
        value.probability_task,
        value.classifier_fingerprint,
        value.calibrator_fingerprint,
        field="domain",
    )
    if probability is not None and task != "visual_domain":
        raise ValueError(
            "domain probabilities require probability_task='visual_domain'"
        )
    if score is None and probability is None:
        raise ValueError("domain evidence requires a reference score or probability")
    return DomainNegativeEvidence(
        evidence_id=evidence_id,
        outcome=outcome,
        reason=reason,
        reference_score=score,
        reference_score_kind=score_kind,
        score_contract_fingerprint=score_contract,
        calibrated_probability=probability,
        probability_task=task,
        classifier_fingerprint=classifier,
        calibrator_fingerprint=calibrator,
    )


def _validate_generic_non_target(
    value: CalibratedNonTargetEvidence,
) -> CalibratedNonTargetEvidence:
    if not isinstance(value, CalibratedNonTargetEvidence):
        raise TypeError("non_target_classifier must be CalibratedNonTargetEvidence")
    evidence_id = _required_text(value.evidence_id, field="non_target.evidence_id")
    probability = _unit_interval(
        value.calibrated_probability,
        field="non_target.calibrated_probability",
    )
    task = _required_choice(
        value.probability_task,
        field="non_target.probability_task",
        allowed=TARGET_TASKS,
    )
    if task not in _GENERIC_NON_TARGET_TASKS:
        raise ValueError("generic non-target evidence requires a target-verifier task")
    return CalibratedNonTargetEvidence(
        evidence_id=evidence_id,
        calibrated_probability=probability,
        probability_task=task,
        classifier_fingerprint=_sha256(
            value.classifier_fingerprint,
            field="non_target.classifier_fingerprint",
        ),
        calibrator_fingerprint=_sha256(
            value.calibrator_fingerprint,
            field="non_target.calibrator_fingerprint",
        ),
    )


def _validate_reference_score(
    score: object,
    score_kind: object,
    contract_fingerprint: object,
    *,
    field: str,
) -> tuple[float | None, str | None, str | None]:
    values = (score, score_kind, contract_fingerprint)
    if all(value is None for value in values):
        return None, None, None
    if any(value is None for value in values):
        raise ValueError(f"{field} reference score provenance must be all-or-none")
    result = _finite_number(score, field=f"{field}.reference_score")
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field}.reference_score must be between -1 and 1")
    kind = _required_text(score_kind, field=f"{field}.reference_score_kind")
    if kind != REFERENCE_SCORE_KIND_COSINE_SIMILARITY:
        raise ValueError("non-match reference scores must be cosine similarities")
    contract = _sha256(
        contract_fingerprint,
        field=f"{field}.score_contract_fingerprint",
    )
    return result, kind, contract


def _validate_probability(
    probability: object,
    probability_task: object,
    classifier_fingerprint: object,
    calibrator_fingerprint: object,
    *,
    field: str,
) -> tuple[float | None, str | None, str | None, str | None]:
    values = (
        probability,
        probability_task,
        classifier_fingerprint,
        calibrator_fingerprint,
    )
    if all(value is None for value in values):
        return None, None, None, None
    if any(value is None for value in values):
        raise ValueError(f"{field} probability provenance must be all-or-none")
    return (
        _unit_interval(probability, field=f"{field}.calibrated_probability"),
        _required_choice(
            probability_task,
            field=f"{field}.probability_task",
            allowed=TARGET_TASKS,
        ),
        _sha256(
            classifier_fingerprint,
            field=f"{field}.classifier_fingerprint",
        ),
        _sha256(
            calibrator_fingerprint,
            field=f"{field}.calibrator_fingerprint",
        ),
    )


def _best_raw_candidate(
    values: Sequence[CandidateNonMatchEvidence],
) -> _ScoredEvidence | None:
    scored = tuple(
        _ScoredEvidence(
            score=float(item.reference_score),
            score_kind=str(item.reference_score_kind),
            evidence_kind="known_competitor",
            evidence_id=item.evidence_id,
            outcome=_competitor_outcome(item),
            reason=";".join(item.candidate_reasons),
            accepted_taxon_key=item.accepted_taxon_key,
        )
        for item in values
        if item.reference_score is not None
    )
    return _best_scored_evidence(scored)


def _best_raw_domain(
    values: Sequence[DomainNegativeEvidence],
) -> _ScoredEvidence | None:
    scored = tuple(
        _ScoredEvidence(
            score=float(item.reference_score),
            score_kind=str(item.reference_score_kind),
            evidence_kind="domain_negative",
            evidence_id=item.evidence_id,
            outcome=item.outcome,
            reason=item.reason,
        )
        for item in values
        if item.reference_score is not None
    )
    return _best_scored_evidence(scored)


def _best_scored_evidence(
    values: Sequence[_ScoredEvidence],
) -> _ScoredEvidence | None:
    if not values:
        return None
    return min(
        values,
        key=lambda item: (
            -item.score,
            item.accepted_taxon_key or "",
            item.outcome,
            item.evidence_kind,
            item.evidence_id,
        ),
    )


def _candidate_calibrated(
    value: CandidateNonMatchEvidence,
    *,
    outcome: str,
) -> _CalibratedEvidence | None:
    if value.calibrated_probability is None:
        return None
    assert value.calibrator_fingerprint is not None
    return _CalibratedEvidence(
        probability=value.calibrated_probability,
        evidence_kind=(
            "target_classifier"
            if outcome == TARGET_PROBABLE_REVIEW
            else "known_competitor_classifier"
        ),
        evidence_id=value.evidence_id,
        outcome=outcome,
        reason=(
            "calibrated_target_evidence"
            if outcome == TARGET_PROBABLE_REVIEW
            else ";".join(value.candidate_reasons)
        ),
        calibrator_fingerprint=value.calibrator_fingerprint,
        accepted_taxon_key=value.accepted_taxon_key,
    )


def _best_calibrated_candidate(
    values: Sequence[CandidateNonMatchEvidence],
) -> _CalibratedEvidence | None:
    calibrated = tuple(
        item
        for value in values
        if (
            item := _candidate_calibrated(
                value,
                outcome=_competitor_outcome(value),
            )
        )
        is not None
    )
    return _best_calibrated_evidence(calibrated)


def _best_calibrated_domain(
    values: Sequence[DomainNegativeEvidence],
) -> _CalibratedEvidence | None:
    calibrated = tuple(
        _CalibratedEvidence(
            probability=float(item.calibrated_probability),
            evidence_kind="domain_classifier",
            evidence_id=item.evidence_id,
            outcome=item.outcome,
            reason=item.reason,
            calibrator_fingerprint=str(item.calibrator_fingerprint),
        )
        for item in values
        if item.calibrated_probability is not None
    )
    return _best_calibrated_evidence(calibrated)


def _best_calibrated_evidence(
    values: Sequence[_CalibratedEvidence],
) -> _CalibratedEvidence | None:
    if not values:
        return None
    return min(
        values,
        key=lambda item: (
            -item.probability,
            item.accepted_taxon_key or "",
            item.outcome,
            item.evidence_kind,
            item.evidence_id,
        ),
    )


def _competitor_outcome(value: CandidateNonMatchEvidence) -> str:
    return (
        KNOWN_REGIONAL_COMPETITOR
        if value.geographic_scope == "regional"
        else KNOWN_NONREGIONAL_COMPETITOR
    )


def _candidate_name(
    candidates: Sequence[CandidateNonMatchEvidence],
    winner: _ScoredEvidence,
) -> str:
    return next(
        item.scientific_name
        for item in candidates
        if item.evidence_id == winner.evidence_id
    )


def _candidate_name_by_evidence_id(
    candidates: Sequence[CandidateNonMatchEvidence],
    evidence_id: str,
) -> str:
    return next(
        item.scientific_name for item in candidates if item.evidence_id == evidence_id
    )


def _candidate_reasons(
    candidates: Sequence[CandidateNonMatchEvidence],
    winner: _ScoredEvidence,
) -> tuple[str, ...]:
    return next(
        item.candidate_reasons
        for item in candidates
        if item.evidence_id == winner.evidence_id
    )


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _sorted_unique_text(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise TypeError(f"{field} must be a sequence of strings")
    result = tuple(sorted(_required_text(value, field=field) for value in values))
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _required_choice(
    value: object,
    *,
    field: str,
    allowed: frozenset[str] | set[str],
) -> str:
    result = _required_text(value, field=field)
    if result not in allowed:
        raise ValueError(f"{field} is not supported: {result}")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _unit_interval(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return result


__all__ = [
    "ABSTAIN",
    "CALIBRATED_PROBABILITY_SCALE",
    "CLASSIFICATION_OUTCOMES",
    "DOMAIN_NEGATIVE_OUTCOMES",
    "HETEROGENEOUS_EVIDENCE_SCALE",
    "INSUFFICIENT_REFERENCE_COVERAGE",
    "INSUFFICIENT_VISUAL_DETAIL",
    "KNOWN_NONREGIONAL_COMPETITOR",
    "KNOWN_REGIONAL_COMPETITOR",
    "NONMATCH_SCORING_VERSION",
    "NON_BUTTERFLY_INSECT",
    "NO_GEO_GLOBAL_FALLBACK",
    "OTHER_BUTTERFLY",
    "OUT_OF_DISTRIBUTION",
    "PINNED_SPECIMEN",
    "REFERENCE_EVIDENCE_SCALE",
    "REFERENCE_SCORE_KIND_COSINE_SIMILARITY",
    "TARGET_CONFIRMED",
    "TARGET_PROBABLE_REVIEW",
    "VISUAL_ARTIFACT",
    "CalibratedNonTargetEvidence",
    "CandidateNonMatchEvidence",
    "DomainNegativeEvidence",
    "ExplicitNonMatchScore",
    "NonMatchScoringRequest",
    "score_nonmatch_evidence",
]
