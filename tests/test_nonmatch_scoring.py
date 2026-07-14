from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from biominer.ml.nonmatch import (
    CLASSIFICATION_OUTCOMES,
    HETEROGENEOUS_EVIDENCE_SCALE,
    NONMATCH_SCORING_VERSION,
    CandidateNonMatchEvidence,
    CalibratedNonTargetEvidence,
    DomainNegativeEvidence,
    NonMatchScoringRequest,
    score_nonmatch_evidence,
)


TARGET = "gbif:6432573"
REGIONAL = "gbif:1939773"
NONREGIONAL = "gbif:5139051"
SCORE_CONTRACT = "sha256:" + "1" * 64
CANDIDATE_SET = "sha256:" + "2" * 64


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _candidate(
    *,
    evidence_id: str,
    key: str,
    name: str,
    scope: str,
    score: float | None,
    probability: float | None,
    probability_task: str | None,
    reasons: tuple[str, ...],
) -> CandidateNonMatchEvidence:
    return CandidateNonMatchEvidence(
        evidence_id=evidence_id,
        accepted_taxon_key=key,
        scientific_name=name,
        geographic_scope=scope,
        candidate_reasons=reasons,
        reference_score=score,
        reference_score_kind="cosine_similarity" if score is not None else None,
        score_contract_fingerprint=SCORE_CONTRACT if score is not None else None,
        calibrated_probability=probability,
        probability_task=probability_task,
        classifier_fingerprint=(
            _sha(f"classifier:{evidence_id}") if probability is not None else None
        ),
        calibrator_fingerprint=(
            _sha(f"calibrator:{evidence_id}") if probability is not None else None
        ),
    )


def _domain(
    *,
    evidence_id: str = "domain-pinned",
    outcome: str = "pinned_specimen",
    score: float | None = 0.60,
    probability: float | None = 0.40,
) -> DomainNegativeEvidence:
    return DomainNegativeEvidence(
        evidence_id=evidence_id,
        outcome=outcome,
        reason=f"visual-domain:{outcome}",
        reference_score=score,
        reference_score_kind="cosine_similarity" if score is not None else None,
        score_contract_fingerprint=SCORE_CONTRACT if score is not None else None,
        calibrated_probability=probability,
        probability_task="visual_domain" if probability is not None else None,
        classifier_fingerprint=(
            _sha(f"classifier:{evidence_id}") if probability is not None else None
        ),
        calibrator_fingerprint=(
            _sha(f"calibrator:{evidence_id}") if probability is not None else None
        ),
    )


def _request() -> NonMatchScoringRequest:
    return NonMatchScoringRequest(
        query_id="query-001",
        route="adult_field",
        target_accepted_taxon_key=TARGET,
        candidate_set_fingerprint=CANDIDATE_SET,
        candidates=(
            _candidate(
                evidence_id="target",
                key=TARGET,
                name="Papilio demoleus",
                scope="regional",
                score=0.82,
                probability=0.76,
                probability_task="binary_target_verifier",
                reasons=("target",),
            ),
            _candidate(
                evidence_id="regional",
                key=REGIONAL,
                name="Papilio polytes",
                scope="regional",
                score=0.71,
                probability=0.62,
                probability_task="regional_multiclass",
                reasons=("known_mimic", "regional_same_family"),
            ),
            _candidate(
                evidence_id="nonregional",
                key=NONREGIONAL,
                name="Papilio machaon",
                scope="nonregional",
                score=0.65,
                probability=0.55,
                probability_task="regional_multiclass",
                reasons=("close_congener",),
            ),
        ),
        domain_negatives=(_domain(),),
        non_target_classifier=CalibratedNonTargetEvidence(
            evidence_id="binary-non-target",
            calibrated_probability=0.58,
            probability_task="binary_target_verifier",
            classifier_fingerprint=_sha("binary-classifier"),
            calibrator_fingerprint=_sha("binary-calibrator"),
        ),
    )


def test_explicit_nonmatch_maxima_and_margins_preserve_winning_identity() -> None:
    result = score_nonmatch_evidence(_request())

    assert result.scoring_version == NONMATCH_SCORING_VERSION
    assert result.best_target_reference_score == pytest.approx(0.82)
    assert result.best_known_competitor_score == pytest.approx(0.71)
    assert result.best_known_competitor_accepted_taxon_key == REGIONAL
    assert result.best_known_competitor_scientific_name == "Papilio polytes"
    assert result.best_known_competitor_outcome == "known_regional_competitor"
    assert result.best_known_competitor_reasons == (
        "known_mimic",
        "regional_same_family",
    )
    assert result.best_domain_negative_score == pytest.approx(0.60)
    assert result.best_domain_negative_outcome == "pinned_specimen"
    assert result.best_non_target_score == pytest.approx(0.71)
    assert result.best_non_target_score_scale == HETEROGENEOUS_EVIDENCE_SCALE
    assert result.best_non_target_score_kind == "cosine_similarity"
    assert result.best_non_target_score_is_calibrated_probability is False
    assert result.raw_score_contract_fingerprint == SCORE_CONTRACT
    assert result.competitor_margin == pytest.approx(0.11)
    assert result.calibrated_target_probability == pytest.approx(0.76)
    assert result.calibrated_best_non_target_probability == pytest.approx(0.62)
    assert result.nonmatch_margin == pytest.approx(0.14)
    assert result.leading_nonmatch_outcome == "known_regional_competitor"
    assert result.leading_nonmatch_evidence_id == "regional"
    assert result.scoring_fingerprint.startswith("sha256:")


def test_raw_and_calibrated_competitor_winners_keep_separate_taxon_identity() -> None:
    request = _request()
    calibrated_winner = replace(
        request.candidates[2],
        calibrated_probability=0.70,
    )

    result = score_nonmatch_evidence(
        replace(request, candidates=(*request.candidates[:2], calibrated_winner))
    )

    assert result.best_known_competitor_accepted_taxon_key == REGIONAL
    assert result.calibrated_best_competitor_accepted_taxon_key == NONREGIONAL
    assert result.calibrated_best_competitor_scientific_name == "Papilio machaon"
    assert result.calibrated_best_non_target_accepted_taxon_key == NONREGIONAL
    assert result.leading_nonmatch_accepted_taxon_key == NONREGIONAL


def test_domain_negative_can_lead_calibrated_nonmatch_without_changing_raw_winner() -> (
    None
):
    request = _request()
    strong_domain = _domain(probability=0.91)

    result = score_nonmatch_evidence(
        replace(request, domain_negatives=(strong_domain,))
    )

    assert result.best_non_target_score == pytest.approx(0.71)
    assert result.best_non_target_evidence_id == "regional"
    assert result.calibrated_best_non_target_probability == pytest.approx(0.91)
    assert result.calibrated_best_non_target_evidence_id == "domain-pinned"
    assert result.leading_nonmatch_outcome == "pinned_specimen"
    assert result.nonmatch_margin == pytest.approx(-0.15)


def test_generic_non_target_probability_has_no_fabricated_biological_identity() -> None:
    request = _request()
    generic = replace(
        request.non_target_classifier,
        calibrated_probability=0.95,
    )

    result = score_nonmatch_evidence(replace(request, non_target_classifier=generic))

    assert result.best_non_target_score == pytest.approx(0.95)
    assert result.best_non_target_score_kind == "calibrated_probability"
    assert result.best_non_target_score_is_calibrated_probability is True
    assert result.best_non_target_evidence_kind == "generic_non_target_classifier"
    assert result.calibrated_best_non_target_probability == pytest.approx(0.95)
    assert result.leading_nonmatch_outcome == "abstain"
    assert result.leading_nonmatch_reason == "unresolved_calibrated_non_target"


def test_no_calibrated_evidence_leaves_probability_margin_null() -> None:
    request = _request()
    candidates = tuple(
        replace(
            item,
            calibrated_probability=None,
            probability_task=None,
            classifier_fingerprint=None,
            calibrator_fingerprint=None,
        )
        for item in request.candidates
    )
    domains = tuple(
        replace(
            item,
            calibrated_probability=None,
            probability_task=None,
            classifier_fingerprint=None,
            calibrator_fingerprint=None,
        )
        for item in request.domain_negatives
    )

    result = score_nonmatch_evidence(
        replace(
            request,
            candidates=candidates,
            domain_negatives=domains,
            non_target_classifier=None,
        )
    )

    assert result.calibrated_target_probability is None
    assert result.calibrated_best_non_target_probability is None
    assert result.nonmatch_margin is None
    assert result.competitor_margin == pytest.approx(0.11)
    assert result.leading_nonmatch_outcome == "known_regional_competitor"


def test_input_order_does_not_change_winners_or_fingerprint() -> None:
    request = _request()
    forward = score_nonmatch_evidence(request)
    reversed_result = score_nonmatch_evidence(
        replace(
            request,
            candidates=tuple(reversed(request.candidates)),
            domain_negatives=tuple(reversed(request.domain_negatives)),
        )
    )

    assert reversed_result == forward


def test_ties_use_stable_identity_order() -> None:
    request = _request()
    tied = replace(
        request.candidates[2], reference_score=0.71, calibrated_probability=0.62
    )

    result = score_nonmatch_evidence(
        replace(request, candidates=(*request.candidates[:2], tied))
    )

    assert result.best_known_competitor_accepted_taxon_key == min(REGIONAL, NONREGIONAL)


def test_mismatched_raw_score_contracts_fail_closed() -> None:
    request = _request()
    incompatible = replace(
        request.candidates[1],
        score_contract_fingerprint=_sha("different-score-contract"),
    )

    with pytest.raises(ValueError, match="raw score contracts must match"):
        score_nonmatch_evidence(
            replace(request, candidates=(request.candidates[0], incompatible))
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_target", "exactly one target"),
        ("duplicate_candidate", "candidate taxon keys must be unique"),
        ("probability_without_provenance", "probability provenance"),
        ("raw_out_of_range", "reference_score must be between -1 and 1"),
        ("wrong_competitor_task", "competitor probabilities require"),
    ),
)
def test_invalid_evidence_is_rejected(mutation: str, message: str) -> None:
    request = _request()
    if mutation == "missing_target":
        changed = replace(request, candidates=request.candidates[1:])
    elif mutation == "duplicate_candidate":
        changed = replace(
            request,
            candidates=(
                *request.candidates,
                replace(request.candidates[1], evidence_id="dup"),
            ),
        )
    elif mutation == "probability_without_provenance":
        invalid = replace(request.candidates[1], calibrator_fingerprint=None)
        changed = replace(request, candidates=(request.candidates[0], invalid))
    elif mutation == "raw_out_of_range":
        invalid = replace(request.candidates[1], reference_score=1.1)
        changed = replace(request, candidates=(request.candidates[0], invalid))
    else:
        invalid = replace(
            request.candidates[1],
            probability_task="binary_target_verifier",
        )
        changed = replace(request, candidates=(request.candidates[0], invalid))

    with pytest.raises(ValueError, match=message):
        score_nonmatch_evidence(changed)


def test_complete_outcome_vocabulary_is_closed_and_versioned() -> None:
    assert CLASSIFICATION_OUTCOMES == frozenset(
        {
            "target_confirmed",
            "target_probable_review",
            "known_regional_competitor",
            "known_nonregional_competitor",
            "other_butterfly",
            "non_butterfly_insect",
            "pinned_specimen",
            "visual_artifact",
            "out_of_distribution",
            "insufficient_visual_detail",
            "insufficient_reference_coverage",
            "no_geo_global_fallback",
            "abstain",
        }
    )
    assert NONMATCH_SCORING_VERSION == "regional-nonmatch-scoring-v1.0.0"
