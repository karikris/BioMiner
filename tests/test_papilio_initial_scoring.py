from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from biominer.evaluation.flickr_release import (
    UnreviewedFlickrCandidateEvidence,
    score_unreviewed_flickr_candidate,
)
from biominer.references.admission import default_reference_admission_policy
from biominer.references.readiness import reference_readiness_allows_vision
from biominer.references.support_admission import evaluate_support_admission
from biominer.reports.adaptive_pilot_initial import (
    load_initial_pilot_report,
    validate_initial_pilot_report,
)
from test_adaptive_readiness_matrix import _provisional_payload
from test_support_admission import _evidence


REPORT_PATH = Path(
    "reports/gbif_fast_start/papilio_demoleus_initial_scoring.json"
)


def test_fixture_backed_papilio_workflow_reaches_provisional_scoring() -> None:
    report = load_initial_pilot_report(REPORT_PATH)
    admission = evaluate_support_admission(
        _evidence(), default_reference_admission_policy()
    )
    readiness = _provisional_payload()
    score = score_unreviewed_flickr_candidate(
        UnreviewedFlickrCandidateEvidence(
            source_record_id="flickr:papilio-pilot-fixture",
            route="adult_field",
            embedding_artifact_sha256="sha256:" + "c" * 64,
            candidate_ranking=("Papilio demoleus", "Papilio polytes"),
            provisional_margin=0.17,
            review_priority=0.82,
        )
    )

    assert admission.eligible and admission.provisional
    assert reference_readiness_allows_vision(readiness)
    assert score.scoring_state == "provisional_candidate_scored"
    assert score.eligible_for_final_occurrence_dataset is False
    assert report["current_execution"]["metrics"][
        "reference_reviews_before_first_scoring"
    ] == 0


def test_initial_report_rejects_live_and_scientific_overclaiming() -> None:
    report = load_initial_pilot_report(REPORT_PATH)
    for section, field, value in (
        ("current_execution", "live_status", "complete"),
        ("historical_context", "counted_as_current_execution", True),
        ("semantics", "raw_scores_are_probabilities", True),
        ("semantics", "scientific_release_authorized", True),
    ):
        tampered = deepcopy(report)
        tampered[section][field] = value
        with pytest.raises(ValueError):
            validate_initial_pilot_report(tampered)
