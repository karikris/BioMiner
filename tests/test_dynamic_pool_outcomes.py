"""Tests for exclusive dynamic-pool release, screening and unresolved lanes."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_outcomes import (
    AUDITED_SCREENING_CANDIDATE_SCHEMA,
    HUMAN_REVIEWED_RELEASE_LABEL,
    HUMAN_REVIEWED_RELEASE_SCHEMA,
    UNRESOLVED_CANDIDATE_QUEUE_SCHEMA,
    DynamicPoolOutcomeEvidence,
    project_audited_screening_candidates,
    project_dynamic_pool_outcome_lanes,
    project_human_reviewed_release_set,
    project_unresolved_candidate_queue,
    validate_audited_screening_candidates,
    validate_human_reviewed_release_set,
    validate_unresolved_candidate_queue,
)
from biominer.evaluation.flickr_export import (
    FlickrExportValidationError,
    validate_verified_flickr_export,
)
from biominer.evaluation.flickr_release import (
    FlickrReleaseEvidence,
    decide_flickr_release,
)


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _outcome(
    index: int,
    *,
    review_decision: str | None = "include",
    release_eligible: bool = True,
    occurrence_claim_supported: bool | None = None,
    probability: float | None = 0.90,
    threshold_status: str = "selected",
) -> DynamicPoolOutcomeEvidence:
    source_hash = _sha(index)
    reviewed = review_decision is not None
    release_evidence = FlickrReleaseEvidence(
        source_record_id=f"flickr:{index}",
        source_image_sha256=source_hash,
        review_decision=review_decision,
        review_source_image_sha256=source_hash if reviewed else None,
        duplicate_group_resolved=release_eligible,
        target_identity_supported=release_eligible,
        visual_domain_suitable=release_eligible,
        life_stage_suitable=release_eligible,
        coordinate_requirements_pass=release_eligible,
        date_requirements_pass=release_eligible,
        release_policy_permits=release_eligible,
    )
    if occurrence_claim_supported is None:
        occurrence_claim_supported = review_decision == "include" and release_eligible
    return DynamicPoolOutcomeEvidence(
        item_id=f"item-{index}",
        source_record_id=f"flickr:{index}",
        source_image_sha256=source_hash,
        candidate_species_key=f"species-{index % 2}",
        route="adult_field",
        evidence_model_fingerprint=_sha(13),
        calibrator_fingerprint=_sha(14),
        split_fingerprint=_sha(15),
        release_decision=decide_flickr_release(release_evidence),
        conflict_status="not_required",
        occurrence_claim_supported=occurrence_claim_supported,
        screening_threshold_status=threshold_status,
        route_compatible=True,
        reference_coverage_sufficient=True,
        geographic_evidence_sufficient=True,
        visual_detail_sufficient=True,
        domain_negative_absent=True,
        out_of_distribution_absent=True,
        review_priority=0.4 + index * 0.05,
        human_review_decision=review_decision,
        review_decision_fingerprint=_sha(index + 1) if reviewed else None,
        review_source_image_sha256=source_hash if reviewed else None,
        calibrated_supported_probability=probability,
        screening_threshold_selection_fingerprint=(
            _sha(12) if threshold_status == "selected" else None
        ),
        screening_threshold=0.80 if threshold_status == "selected" else None,
        triage_reasons=("fixture",),
    )


def test_release_projection_contains_only_source_bound_human_eligible_rows() -> None:
    eligible = _outcome(0, probability=0.10)
    unreviewed_high_score = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
    )
    reviewed_but_blocked = _outcome(2, release_eligible=False)

    projection = project_human_reviewed_release_set(
        [reviewed_but_blocked, unreviewed_high_score, eligible]
    )

    assert projection.table.schema == HUMAN_REVIEWED_RELEASE_SCHEMA
    assert projection.source_item_count == 3
    assert projection.projected_item_count == 1
    row = projection.table.row(0, named=True)
    assert row["source_record_id"] == "flickr:0"
    assert row["outcome_label"] == HUMAN_REVIEWED_RELEASE_LABEL
    assert row["human_review_decision"] == "include"
    assert row["human_reviewed"] is True
    assert row["release_authorized"] is True
    assert row["model_evidence_authorizes_release"] is False
    assert row["calibrated_supported_probability"] == pytest.approx(0.10)
    assert validate_verified_flickr_export(projection.table) is projection.table


def test_high_calibrated_probability_cannot_release_an_unreviewed_row() -> None:
    projection = project_human_reviewed_release_set(
        [
            _outcome(
                1,
                review_decision=None,
                release_eligible=False,
                probability=1.0,
            )
        ]
    )

    assert projection.table.is_empty()
    assert projection.projected_item_count == 0
    validate_human_reviewed_release_set(projection.table)


@pytest.mark.parametrize(
    "changes",
    [
        {"conflict_status": "unresolved"},
        {"occurrence_claim_supported": False},
    ],
)
def test_projection_fails_closed_on_additional_release_lane_gates(
    changes: dict[str, object],
) -> None:
    evidence = replace(_outcome(0), **changes)

    assert project_human_reviewed_release_set([evidence]).table.is_empty()


def test_projection_excludes_a_stale_review_source_binding() -> None:
    source = _outcome(0)
    stale_hash = _sha(9)
    stale_decision = decide_flickr_release(
        FlickrReleaseEvidence(
            source_record_id=source.source_record_id,
            source_image_sha256=source.source_image_sha256,
            review_decision="include",
            review_source_image_sha256=stale_hash,
            duplicate_group_resolved=True,
            target_identity_supported=True,
            visual_domain_suitable=True,
            life_stage_suitable=True,
            coordinate_requirements_pass=True,
            date_requirements_pass=True,
            release_policy_permits=True,
        )
    )
    stale = replace(
        source,
        release_decision=stale_decision,
        review_source_image_sha256=stale_hash,
    )

    assert project_human_reviewed_release_set([stale]).table.is_empty()


def test_outcome_evidence_rejects_release_decision_from_another_source() -> None:
    source = _outcome(0)
    foreign_decision = decide_flickr_release(
        FlickrReleaseEvidence(
            source_record_id="flickr:foreign",
            source_image_sha256=_sha(0),
        )
    )

    with pytest.raises(ValueError, match="another source record"):
        replace(source, release_decision=foreign_decision)


def test_release_lane_validator_detects_authority_tampering() -> None:
    table = project_human_reviewed_release_set([_outcome(0)]).table
    tampered = table.with_columns(
        pl.lit(True).alias("model_evidence_authorizes_release")
    )

    with pytest.raises(ValueError, match="ineligible row"):
        validate_human_reviewed_release_set(tampered)


def test_release_projection_is_deterministic() -> None:
    items = [_outcome(0), _outcome(1), _outcome(2)]
    first = project_human_reviewed_release_set(items)
    second = project_human_reviewed_release_set(list(reversed(items)))

    assert first.lane_fingerprint == second.lane_fingerprint
    assert first.table.equals(second.table)


def test_unreviewed_threshold_passing_row_uses_exact_screening_label() -> None:
    unreviewed = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.91,
    )

    projection = project_audited_screening_candidates([unreviewed])

    assert projection.table.schema == AUDITED_SCREENING_CANDIDATE_SCHEMA
    assert projection.projected_item_count == 1
    row = projection.table.row(0, named=True)
    assert row["outcome_label"] == "statistically_supported_screening_candidate"
    assert row["human_review_decision"] is None
    assert row["human_reviewed"] is False
    assert row["human_review_required_before_release"] is True
    assert row["screening_supported"] is True
    assert row["screening_only"] is True


def test_reviewed_or_below_threshold_rows_never_enter_screening_lane() -> None:
    reviewed = _outcome(0, probability=0.99)
    below_threshold = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.79,
    )
    infeasible = _outcome(
        2,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
        threshold_status="infeasible",
    )

    projection = project_audited_screening_candidates(
        [reviewed, below_threshold, infeasible]
    )

    assert projection.table.is_empty()


@pytest.mark.parametrize(
    "field",
    [
        "route_compatible",
        "reference_coverage_sufficient",
        "geographic_evidence_sufficient",
        "visual_detail_sufficient",
        "domain_negative_absent",
        "out_of_distribution_absent",
    ],
)
def test_every_screening_quality_gate_fails_closed(field: str) -> None:
    source = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
    )

    assert project_audited_screening_candidates(
        [replace(source, **{field: False})]
    ).table.is_empty()


def test_screening_rows_are_structurally_rejected_by_occurrence_export() -> None:
    table = project_audited_screening_candidates(
        [
            _outcome(
                1,
                review_decision=None,
                release_eligible=False,
                probability=0.99,
            )
        ]
    ).table
    row = table.row(0, named=True)

    assert row["occurrence_claim_supported"] is False
    assert row["eligible_for_final_occurrence_dataset"] is False
    assert row["release_state"] == "excluded"
    assert row["release_authorized"] is False
    assert row["model_evidence_authorizes_release"] is False
    with pytest.raises(FlickrExportValidationError) as error:
        validate_verified_flickr_export(table)
    assert "unreviewed" in error.value.blocked_records["flickr:1"]


def test_screening_validator_rejects_language_or_release_tampering() -> None:
    table = project_audited_screening_candidates(
        [
            _outcome(
                1,
                review_decision=None,
                release_eligible=False,
                probability=0.99,
            )
        ]
    ).table
    renamed = table.with_columns(pl.lit("verified").alias("outcome_label"))
    authorized = table.with_columns(pl.lit(True).alias("release_authorized"))

    with pytest.raises(ValueError, match="evidence boundary"):
        validate_audited_screening_candidates(renamed)
    with pytest.raises(ValueError, match="evidence boundary"):
        validate_audited_screening_candidates(authorized)


def test_screening_projection_is_deterministic() -> None:
    items = [
        _outcome(
            index,
            review_decision=None,
            release_eligible=False,
            probability=0.85 + index * 0.02,
        )
        for index in (1, 2, 3)
    ]

    first = project_audited_screening_candidates(items)
    second = project_audited_screening_candidates(list(reversed(items)))

    assert first.lane_fingerprint == second.lane_fingerprint
    assert first.table.equals(second.table)


def test_unresolved_queue_retains_every_non_release_non_screening_outcome() -> None:
    below_threshold = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.79,
    )
    infeasible = _outcome(
        2,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
        threshold_status="infeasible",
    )
    insufficient_detail = replace(
        _outcome(
            3,
            review_decision=None,
            release_eligible=False,
            probability=0.99,
        ),
        visual_detail_sufficient=False,
    )
    excluded = _outcome(
        4,
        review_decision="exclude",
        release_eligible=False,
        occurrence_claim_supported=False,
    )

    projection = project_unresolved_candidate_queue(
        [below_threshold, infeasible, insufficient_detail, excluded]
    )

    assert projection.table.schema == UNRESOLVED_CANDIDATE_QUEUE_SCHEMA
    assert projection.source_item_count == 4
    assert projection.projected_item_count == 4
    assert projection.table["item_id"].to_list() == [
        "item-4",
        "item-3",
        "item-2",
        "item-1",
    ]
    assert projection.table["queue_rank"].to_list() == [1, 2, 3, 4]


def test_unresolved_queue_explains_abstention_and_human_exclusion() -> None:
    below_threshold = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.79,
    )
    infeasible = _outcome(
        2,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
        threshold_status="infeasible",
    )
    insufficient_detail = replace(
        _outcome(
            3,
            review_decision=None,
            release_eligible=False,
            probability=0.99,
        ),
        visual_detail_sufficient=False,
    )
    excluded = _outcome(
        4,
        review_decision="exclude",
        release_eligible=False,
        occurrence_claim_supported=False,
    )

    rows = {
        row["item_id"]: row
        for row in project_unresolved_candidate_queue(
            [below_threshold, infeasible, insufficient_detail, excluded]
        ).table.iter_rows(named=True)
    }

    assert "human_review_missing" in rows["item-1"]["abstention_reasons"]
    assert (
        "calibrated_probability_below_screening_threshold"
        in rows["item-1"]["abstention_reasons"]
    )
    assert "screening_threshold_infeasible" in rows["item-2"]["abstention_reasons"]
    assert "visual_detail_insufficient" in rows["item-3"]["abstention_reasons"]
    assert rows["item-4"]["outcome_label"] == "human_review_excluded"
    assert rows["item-4"]["review_required"] is False
    assert "human_review_excluded" in rows["item-4"]["abstention_reasons"]
    assert all(row["model_abstained"] for row in rows.values())
    assert not any(row["release_authorized"] for row in rows.values())


def test_outcome_bundle_is_complete_mutually_exclusive_and_fail_closed() -> None:
    release = _outcome(0)
    screening = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
    )
    unresolved = _outcome(
        2,
        review_decision=None,
        release_eligible=False,
        probability=0.50,
    )
    excluded = _outcome(
        3,
        review_decision="exclude",
        release_eligible=False,
        occurrence_claim_supported=False,
    )

    bundle = project_dynamic_pool_outcome_lanes(
        [unresolved, release, excluded, screening]
    )
    lane_sets = (
        set(bundle.human_reviewed_release.table["item_id"]),
        set(bundle.audited_screening.table["item_id"]),
        set(bundle.unresolved.table["item_id"]),
    )

    assert bundle.source_item_count == 4
    assert [len(values) for values in lane_sets] == [1, 1, 2]
    assert not lane_sets[0] & lane_sets[1]
    assert not lane_sets[0] & lane_sets[2]
    assert not lane_sets[1] & lane_sets[2]
    assert set().union(*lane_sets) == {"item-0", "item-1", "item-2", "item-3"}
    assert bundle.human_reviewed_release.table["human_reviewed"].to_list() == [True]
    assert not bundle.audited_screening.table["release_authorized"].any()
    assert not bundle.unresolved.table["release_authorized"].any()


def test_no_unreviewed_outcome_lane_can_be_exported_as_occurrence_evidence() -> None:
    screening = _outcome(
        1,
        review_decision=None,
        release_eligible=False,
        probability=0.99,
    )
    unresolved = _outcome(
        2,
        review_decision=None,
        release_eligible=False,
        probability=0.50,
    )
    bundle = project_dynamic_pool_outcome_lanes([_outcome(0), screening, unresolved])

    assert (
        validate_verified_flickr_export(bundle.human_reviewed_release.table).height == 1
    )
    for table in (bundle.audited_screening.table, bundle.unresolved.table):
        assert not table["release_authorized"].any()
        assert not table["eligible_for_final_occurrence_dataset"].any()
        with pytest.raises(FlickrExportValidationError):
            validate_verified_flickr_export(table)


def test_unresolved_validator_rejects_authority_and_rank_tampering() -> None:
    table = project_unresolved_candidate_queue(
        [
            _outcome(
                1,
                review_decision=None,
                release_eligible=False,
                probability=0.50,
            ),
            _outcome(
                2,
                review_decision=None,
                release_eligible=False,
                probability=0.60,
            ),
        ]
    ).table
    authorized = table.with_columns(pl.lit(True).alias("release_authorized"))
    misranked = table.with_columns(pl.lit(1, dtype=pl.UInt32).alias("queue_rank"))

    with pytest.raises(ValueError, match="evidence boundary"):
        validate_unresolved_candidate_queue(authorized)
    with pytest.raises(ValueError, match="ranks must be contiguous"):
        validate_unresolved_candidate_queue(misranked)


def test_unresolved_and_bundle_projections_are_deterministic() -> None:
    items = [
        _outcome(
            index,
            review_decision=None,
            release_eligible=False,
            probability=0.50 + index * 0.02,
        )
        for index in (1, 2, 3)
    ]

    first_queue = project_unresolved_candidate_queue(items)
    second_queue = project_unresolved_candidate_queue(list(reversed(items)))
    first_bundle = project_dynamic_pool_outcome_lanes(items)
    second_bundle = project_dynamic_pool_outcome_lanes(list(reversed(items)))

    assert first_queue.lane_fingerprint == second_queue.lane_fingerprint
    assert first_queue.table.equals(second_queue.table)
    assert first_bundle.bundle_fingerprint == second_bundle.bundle_fingerprint
