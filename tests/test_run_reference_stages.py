from __future__ import annotations

from pathlib import Path

import pytest

from biominer.run import (
    MANUAL_REVIEW_STAGES,
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunManifest,
    RunStage,
    StageExecutionResult,
    StageStatus,
    TaxonScope,
)
from biominer.run.stages import default_stage_records
from biominer.species.context import SpeciesContext


def test_manual_review_stages_are_explicit() -> None:
    assert MANUAL_REVIEW_STAGES == frozenset(
        {
            RunStage.REFERENCE_REVIEW,
            RunStage.FLICKR_HUMAN_VERIFICATION,
            RunStage.TARGETED_REFERENCE_REVIEW,
        }
    )


def test_manual_review_completion_requires_an_audited_approval() -> None:
    manifest = _manifest((RunStage.REFERENCE_REVIEW,))

    with pytest.raises(
        ValueError, match="manual-review stage cannot be completed automatically"
    ):
        manifest.with_stage_status(RunStage.REFERENCE_REVIEW, StageStatus.COMPLETE)

    waiting = manifest.with_stage_status(
        RunStage.REFERENCE_REVIEW,
        StageStatus.AWAITING_MANUAL_REVIEW,
        message="reference_review_required",
    )
    approved = waiting.with_manual_review_approval(
        RunStage.REFERENCE_REVIEW,
        reviewer="curator@example.org",
        approved_at="2026-07-14T01:02:03Z",
        message="reviewed_reference_set_accepted",
    )

    record = approved.stages[0]
    assert record.status is StageStatus.COMPLETE
    assert record.ended_at == "2026-07-14T01:02:03Z"
    assert record.message == "reviewed_reference_set_accepted"
    assert record.metrics["manual_review_approved_by"] == "curator@example.org"
    assert record.metrics["manual_review_approved_at"] == "2026-07-14T01:02:03Z"


@pytest.mark.parametrize(
    "stage",
    tuple(sorted(MANUAL_REVIEW_STAGES, key=lambda item: item.value)),
)
def test_every_manual_stage_rejects_automatic_completion(stage: RunStage) -> None:
    manifest = _manifest((stage,))

    with pytest.raises(
        ValueError,
        match="manual-review stage cannot be completed automatically",
    ):
        manifest.with_stage_status(stage, StageStatus.COMPLETE)


def test_manual_review_approval_rejects_invalid_transition() -> None:
    manifest = _manifest((RunStage.REFERENCE_REVIEW,))

    with pytest.raises(ValueError, match="must be awaiting manual review"):
        manifest.with_manual_review_approval(
            RunStage.REFERENCE_REVIEW,
            reviewer="curator@example.org",
        )
    waiting = manifest.with_stage_status(
        RunStage.REFERENCE_REVIEW,
        StageStatus.AWAITING_MANUAL_REVIEW,
    )
    with pytest.raises(ValueError, match="reviewer must be non-empty"):
        waiting.with_manual_review_approval(RunStage.REFERENCE_REVIEW, reviewer="  ")


def test_orchestrator_pauses_at_manual_review_without_completing_it(
    tmp_path: Path,
) -> None:
    called: list[RunStage] = []

    def complete(stage: RunStage) -> object:
        def handler(_plan: object) -> StageExecutionResult:
            called.append(stage)
            return StageExecutionResult(outputs={"artifact": f"{stage.value}.parquet"})

        return handler

    stages = (
        RunStage.REFERENCE_METADATA,
        RunStage.REFERENCE_REVIEW,
        RunStage.REFERENCE_EMBEDDINGS,
    )
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        output_root=tmp_path,
        run_id="manual-review-pause",
        stages=stages,
    )
    result = ProductionRunOrchestrator(
        request,
        taxon_scope=_taxon_scope(),
        stage_handlers={stage: complete(stage) for stage in stages},
    ).run()

    records = {record.stage: record for record in result.manifest.stages}
    assert called == [RunStage.REFERENCE_METADATA, RunStage.REFERENCE_REVIEW]
    assert result.manifest.status == StageStatus.AWAITING_MANUAL_REVIEW.value
    assert result.manifest.ended_at is None
    assert records[RunStage.REFERENCE_METADATA].status is StageStatus.COMPLETE
    assert (
        records[RunStage.REFERENCE_REVIEW].status is StageStatus.AWAITING_MANUAL_REVIEW
    )
    assert records[RunStage.REFERENCE_REVIEW].message == "manual_review_required"
    assert records[RunStage.REFERENCE_REVIEW].outputs == {
        "artifact": "reference_review.parquet"
    }
    assert records[RunStage.REFERENCE_REVIEW].ended_at is None
    assert records[RunStage.REFERENCE_EMBEDDINGS].status is StageStatus.PENDING


def test_dynamic_workflow_stops_before_risk_audit_for_human_review(
    tmp_path: Path,
) -> None:
    called: list[RunStage] = []

    def complete(stage: RunStage) -> object:
        def handler(_plan: object) -> StageExecutionResult:
            called.append(stage)
            return StageExecutionResult(outputs={"artifact": f"{stage.value}.parquet"})

        return handler

    stages = (
        RunStage.REVIEW_SAMPLE_PLANNING,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.RISK_CONTROLLED_AUDIT,
    )
    result = ProductionRunOrchestrator(
        ProductionRunRequest(
            taxon="Papilio demoleus",
            rank="species",
            output_root=tmp_path,
            run_id="dynamic-manual-review-pause",
            stages=stages,
        ),
        taxon_scope=_taxon_scope(),
        stage_handlers={stage: complete(stage) for stage in stages},
    ).run()

    records = {record.stage: record for record in result.manifest.stages}
    assert called == [
        RunStage.REVIEW_SAMPLE_PLANNING,
        RunStage.FLICKR_HUMAN_VERIFICATION,
    ]
    assert result.manifest.status == StageStatus.AWAITING_MANUAL_REVIEW.value
    assert records[RunStage.REVIEW_SAMPLE_PLANNING].status is StageStatus.COMPLETE
    assert (
        records[RunStage.FLICKR_HUMAN_VERIFICATION].status
        is StageStatus.AWAITING_MANUAL_REVIEW
    )
    assert records[RunStage.FLICKR_HUMAN_VERIFICATION].ended_at is None
    assert records[RunStage.RISK_CONTROLLED_AUDIT].status is StageStatus.PENDING


def _manifest(stages: tuple[RunStage, ...]) -> RunManifest:
    return RunManifest(
        run_id="reference-stage-test",
        taxon_scope=_taxon_scope(),
        stages=default_stage_records(stages),
    )


def _taxon_scope() -> TaxonScope:
    context = SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:1941315",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:9417",
        genus_key="gbif:1920490",
        species_key="gbif:1941315",
        registry_version="test-registry-v1",
    )
    return TaxonScope.from_species_context(context)
