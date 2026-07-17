from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.evaluation.flickr_release import (
    FlickrReleaseEvidence,
    FlickrReleaseReason,
    FlickrReleaseState,
    UnreviewedFlickrCandidateEvidence,
    decide_flickr_release,
    score_unreviewed_flickr_candidate,
)


IMAGE_SHA256 = "sha256:" + "a" * 64


def _eligible_evidence() -> FlickrReleaseEvidence:
    return FlickrReleaseEvidence(
        source_record_id="flickr:123",
        source_image_sha256=IMAGE_SHA256,
        review_decision="include",
        review_source_image_sha256=IMAGE_SHA256,
        duplicate_group_resolved=True,
        target_identity_supported=True,
        visual_domain_suitable=True,
        life_stage_suitable=True,
        coordinate_requirements_pass=True,
        date_requirements_pass=True,
        release_policy_permits=True,
    )


def test_every_mandatory_release_condition_produces_eligible_record() -> None:
    decision = decide_flickr_release(_eligible_evidence())

    assert decision.state is FlickrReleaseState.ELIGIBLE
    assert decision.eligible_for_final_occurrence_dataset is True
    assert decision.reasons == ()
    assert decision.review_source_image_sha256 == IMAGE_SHA256


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"review_decision": None, "review_source_image_sha256": None}, FlickrReleaseReason.HUMAN_REVIEW_MISSING),
        ({"review_decision": "uncertain"}, FlickrReleaseReason.REVIEW_NOT_DECISIVE),
        ({"review_decision": "exclude"}, FlickrReleaseReason.REVIEW_EXCLUDES_RECORD),
        ({"review_source_image_sha256": "sha256:" + "b" * 64}, FlickrReleaseReason.REVIEW_SOURCE_HASH_MISMATCH),
        ({"duplicate_group_resolved": False}, FlickrReleaseReason.DUPLICATE_GROUP_UNRESOLVED),
        ({"target_identity_supported": False}, FlickrReleaseReason.TARGET_IDENTITY_UNSUPPORTED),
        ({"visual_domain_suitable": False}, FlickrReleaseReason.VISUAL_DOMAIN_UNSUITABLE),
        ({"life_stage_suitable": False}, FlickrReleaseReason.LIFE_STAGE_UNSUITABLE),
        ({"coordinate_requirements_pass": False}, FlickrReleaseReason.COORDINATE_REQUIREMENTS_FAILED),
        ({"date_requirements_pass": False}, FlickrReleaseReason.DATE_REQUIREMENTS_FAILED),
        ({"second_review_required": True}, FlickrReleaseReason.SECOND_REVIEW_INCOMPLETE),
        ({"adjudication_required": True}, FlickrReleaseReason.ADJUDICATION_INCOMPLETE),
        ({"release_policy_permits": False}, FlickrReleaseReason.RELEASE_POLICY_DENIED),
    ],
)
def test_each_failed_prerequisite_excludes_with_machine_readable_reason(
    changes: dict[str, object],
    expected_reason: FlickrReleaseReason,
) -> None:
    decision = decide_flickr_release(replace(_eligible_evidence(), **changes))

    assert decision.state is FlickrReleaseState.EXCLUDED
    assert decision.eligible_for_final_occurrence_dataset is False
    assert expected_reason in decision.reasons


def test_required_second_review_and_adjudication_can_complete() -> None:
    evidence = replace(
        _eligible_evidence(),
        second_review_required=True,
        second_review_complete=True,
        adjudication_required=True,
        adjudication_complete=True,
    )

    assert decide_flickr_release(evidence).eligible_for_final_occurrence_dataset


def test_review_must_bind_to_a_valid_source_image_digest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        replace(_eligible_evidence(), review_source_image_sha256="stale")


def test_unreviewed_candidate_preserves_scoring_but_stays_excluded() -> None:
    evidence = UnreviewedFlickrCandidateEvidence(
        source_record_id="flickr:unreviewed",
        route="adult_butterfly",
        embedding_artifact_sha256="sha256:" + "c" * 64,
        candidate_ranking=("Papilio demoleus", "Papilio polytes"),
        provisional_margin=0.17,
        review_priority=0.82,
    )

    decision = score_unreviewed_flickr_candidate(evidence)

    assert decision.scoring_state == "provisional_candidate_scored"
    assert decision.route == "adult_butterfly"
    assert decision.embedding_artifact_sha256.endswith("c" * 64)
    assert decision.candidate_ranking == evidence.candidate_ranking
    assert decision.provisional_margin == 0.17
    assert decision.review_priority == 0.82
    assert decision.human_review_state == "unreviewed"
    assert decision.release_state is FlickrReleaseState.EXCLUDED
    assert decision.eligible_for_final_occurrence_dataset is False
    assert decision.release_reasons == (FlickrReleaseReason.HUMAN_REVIEW_MISSING,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_ranking", ()),
        ("provisional_margin", float("nan")),
        ("review_priority", 1.01),
    ],
)
def test_unreviewed_candidate_scoring_evidence_is_validated(
    field: str,
    value: object,
) -> None:
    values = {
        "source_record_id": "flickr:unreviewed",
        "route": "adult_butterfly",
        "embedding_artifact_sha256": "sha256:" + "c" * 64,
        "candidate_ranking": ("Papilio demoleus",),
        "provisional_margin": 0.17,
        "review_priority": 0.82,
    }
    values[field] = value

    with pytest.raises(ValueError):
        UnreviewedFlickrCandidateEvidence(**values)  # type: ignore[arg-type]
