from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.evaluation.flickr_release import (
    FlickrReleaseEvidence,
    FlickrReleaseReason,
    FlickrReleaseState,
    decide_flickr_release,
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
