from __future__ import annotations

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
