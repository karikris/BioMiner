from __future__ import annotations

import pytest

from biominer.run.adaptive_workflow import (
    AdaptiveWorkflowState,
    adaptive_stage_dependencies,
    validate_adaptive_stage_start,
)
from biominer.run.stages import (
    ADAPTIVE_REFERENCE_PRODUCTION_STAGES,
    MANUAL_REVIEW_STAGES,
    RunStage,
    StageStatus,
    default_stage_records,
)


def test_adaptive_reference_stage_sequence_is_complete_and_pending() -> None:
    required = {
        RunStage.REFERENCE_METADATA,
        RunStage.REFERENCE_MEDIA,
        RunStage.REFERENCE_DEDUPLICATION,
        RunStage.REFERENCE_QUALITY_ROUTING,
        RunStage.REFERENCE_ADMISSION,
        RunStage.REFERENCE_EMBEDDINGS,
        RunStage.REFERENCE_PROTOTYPES,
        RunStage.PROVISIONAL_FLICKR_SCORING,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.STATISTICAL_REFERENCE_AUDIT,
        RunStage.TARGETED_REFERENCE_REVIEW,
        RunStage.AFFECTED_REFERENCE_REBUILD,
        RunStage.AFFECTED_RECORD_RESCORE,
        RunStage.FINAL_QUALITY_GATE,
    }

    assert required <= set(ADAPTIVE_REFERENCE_PRODUCTION_STAGES)
    assert len(set(ADAPTIVE_REFERENCE_PRODUCTION_STAGES)) == len(
        ADAPTIVE_REFERENCE_PRODUCTION_STAGES
    )
    assert all(
        record.status is StageStatus.PENDING
        for record in default_stage_records(ADAPTIVE_REFERENCE_PRODUCTION_STAGES)
    )


def test_adaptive_human_stages_are_never_automatic() -> None:
    assert {
        RunStage.REFERENCE_REVIEW,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.TARGETED_REFERENCE_REVIEW,
    } <= MANUAL_REVIEW_STAGES


def test_default_dependencies_score_before_reference_review_but_gate_release() -> None:
    graph = {item.stage: item for item in adaptive_stage_dependencies()}

    assert graph[RunStage.PROVISIONAL_FLICKR_SCORING].dependencies == (
        RunStage.REFERENCE_PROTOTYPES,
    )
    assert RunStage.REFERENCE_REVIEW not in graph[
        RunStage.PROVISIONAL_FLICKR_SCORING
    ].dependencies
    assert graph[RunStage.FINAL_QUALITY_GATE].dependencies == (
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.STATISTICAL_REFERENCE_AUDIT,
    )
    assert graph[RunStage.TARGETED_REFERENCE_REVIEW].active is False
    assert graph[RunStage.AFFECTED_RECORD_RESCORE].active is False


def test_flag_and_revision_activate_only_the_required_remediation_chain() -> None:
    flagged = AdaptiveWorkflowState(flagged_species=("gbif:1",))
    flagged_graph = {item.stage: item for item in adaptive_stage_dependencies(flagged)}
    assert flagged_graph[RunStage.TARGETED_REFERENCE_REVIEW].active is True
    assert flagged_graph[RunStage.AFFECTED_REFERENCE_REBUILD].active is False

    revised = AdaptiveWorkflowState(
        flagged_species=("gbif:1",),
        reviewed_flagged_species=("gbif:1",),
        reference_bank_revision="reference-bank-v2",
    )
    revised_graph = {item.stage: item for item in adaptive_stage_dependencies(revised)}
    assert revised_graph[RunStage.AFFECTED_REFERENCE_REBUILD].active is True
    assert revised_graph[RunStage.AFFECTED_RECORD_RESCORE].active is True
    assert RunStage.AFFECTED_RECORD_RESCORE in revised_graph[
        RunStage.FINAL_QUALITY_GATE
    ].dependencies


def test_adaptive_stage_start_fails_closed_on_inactive_or_missing_dependencies() -> None:
    with pytest.raises(ValueError, match="inactive"):
        validate_adaptive_stage_start(
            RunStage.TARGETED_REFERENCE_REVIEW,
            completed_stages=(RunStage.STATISTICAL_REFERENCE_AUDIT,),
        )
    with pytest.raises(ValueError, match="REFERENCE_ADMISSION|reference_admission"):
        validate_adaptive_stage_start(
            RunStage.REFERENCE_EMBEDDINGS,
            completed_stages=(),
        )
