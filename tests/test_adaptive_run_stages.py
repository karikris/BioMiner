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
        RunStage.REFERENCE_GEOGRAPHY_INDEX,
        RunStage.REFERENCE_PROTOTYPES,
        RunStage.FLICKR_DETECTION,
        RunStage.FLICKR_EMBEDDING,
        RunStage.FLICKR_GEO_TAXON_PARTITIONING,
        RunStage.FAMILY_ROUTING,
        RunStage.DYNAMIC_POOL_PLANNING,
        RunStage.DYNAMIC_POOL_SCORING,
        RunStage.PROVISIONAL_FLICKR_SCORING,
        RunStage.REVIEW_SAMPLE_PLANNING,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.RISK_CONTROLLED_AUDIT,
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


def test_dynamic_pooling_stage_vocabulary_is_explicit() -> None:
    assert {
        stage.value
        for stage in {
            RunStage.REFERENCE_GEOGRAPHY_INDEX,
            RunStage.FLICKR_DETECTION,
            RunStage.FLICKR_EMBEDDING,
            RunStage.FLICKR_GEO_TAXON_PARTITIONING,
            RunStage.FAMILY_ROUTING,
            RunStage.DYNAMIC_POOL_PLANNING,
            RunStage.DYNAMIC_POOL_SCORING,
            RunStage.REVIEW_SAMPLE_PLANNING,
            RunStage.RISK_CONTROLLED_AUDIT,
        }
    } == {
        "reference_geography_index",
        "flickr_detection",
        "flickr_embedding",
        "flickr_geo_taxon_partitioning",
        "family_routing",
        "dynamic_pool_planning",
        "dynamic_pool_scoring",
        "review_sample_planning",
        "risk_controlled_audit",
    }


def test_adaptive_human_stages_are_never_automatic() -> None:
    assert {
        RunStage.REFERENCE_REVIEW,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.TARGETED_REFERENCE_REVIEW,
    } <= MANUAL_REVIEW_STAGES


def test_dynamic_dependencies_join_reference_flickr_and_geography_evidence() -> None:
    graph = {item.stage: item for item in adaptive_stage_dependencies()}

    assert graph[RunStage.PROVISIONAL_FLICKR_SCORING].dependencies == (
        RunStage.DYNAMIC_POOL_SCORING,
    )
    assert (
        RunStage.REFERENCE_REVIEW
        not in graph[RunStage.PROVISIONAL_FLICKR_SCORING].dependencies
    )
    assert graph[RunStage.REFERENCE_GEOGRAPHY_INDEX].dependencies == (
        RunStage.REFERENCE_EMBEDDINGS,
    )
    assert graph[RunStage.FLICKR_GEO_TAXON_PARTITIONING].dependencies == (
        RunStage.FLICKR_GEO_CLUSTERING,
        RunStage.FLICKR_DETECTION,
        RunStage.FLICKR_EMBEDDING,
        RunStage.REGIONAL_CANDIDATE_GENERATION,
    )
    assert graph[RunStage.FAMILY_ROUTING].activation_reason == (
        "retrieval_accelerator_not_hard_gate"
    )
    assert graph[RunStage.DYNAMIC_POOL_SCORING].dependencies == (
        RunStage.REFERENCE_EMBEDDINGS,
        RunStage.FLICKR_EMBEDDING,
        RunStage.DYNAMIC_POOL_PLANNING,
    )
    assert graph[RunStage.FLICKR_HUMAN_VERIFICATION].dependencies == (
        RunStage.REVIEW_SAMPLE_PLANNING,
    )
    assert graph[RunStage.STATISTICAL_REFERENCE_AUDIT].dependencies == (
        RunStage.RISK_CONTROLLED_AUDIT,
    )
    assert graph[RunStage.FINAL_QUALITY_GATE].dependencies == (
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.STATISTICAL_REFERENCE_AUDIT,
    )
    assert graph[RunStage.TARGETED_REFERENCE_REVIEW].active is False
    assert graph[RunStage.AFFECTED_RECORD_RESCORE].active is False

    order = {
        stage: index for index, stage in enumerate(ADAPTIVE_REFERENCE_PRODUCTION_STAGES)
    }
    for item in graph.values():
        assert item.stage in order
        assert all(
            order[dependency] < order[item.stage] for dependency in item.dependencies
        )


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
    assert (
        RunStage.AFFECTED_RECORD_RESCORE
        in revised_graph[RunStage.FINAL_QUALITY_GATE].dependencies
    )


def test_adaptive_stage_start_fails_closed_on_inactive_or_missing_dependencies() -> (
    None
):
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
    with pytest.raises(ValueError, match="dynamic_pool_planning"):
        validate_adaptive_stage_start(
            RunStage.DYNAMIC_POOL_SCORING,
            completed_stages=(
                RunStage.REFERENCE_EMBEDDINGS,
                RunStage.FLICKR_EMBEDDING,
            ),
        )
