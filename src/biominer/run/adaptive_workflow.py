"""Conditional dependencies for the adaptive reference workflow."""

from __future__ import annotations

from dataclasses import dataclass

from biominer.run.stages import RunStage


@dataclass(frozen=True, slots=True)
class AdaptiveWorkflowState:
    flagged_species: tuple[str, ...] = ()
    reviewed_flagged_species: tuple[str, ...] = ()
    reference_bank_revision: str | None = None

    def __post_init__(self) -> None:
        flagged = _canonical(self.flagged_species, field="flagged_species")
        reviewed = _canonical(
            self.reviewed_flagged_species,
            field="reviewed_flagged_species",
        )
        if set(reviewed) - set(flagged):
            raise ValueError("reviewed species must be statistically flagged")
        revision = (
            str(self.reference_bank_revision).strip()
            if self.reference_bank_revision is not None
            else None
        )
        if self.reference_bank_revision is not None and not revision:
            raise ValueError("reference_bank_revision must be nonblank")
        if revision is not None and set(reviewed) != set(flagged):
            raise ValueError("a bank revision requires every flagged species review")
        object.__setattr__(self, "flagged_species", flagged)
        object.__setattr__(self, "reviewed_flagged_species", reviewed)
        object.__setattr__(self, "reference_bank_revision", revision)


@dataclass(frozen=True, slots=True)
class AdaptiveStageDependency:
    stage: RunStage
    dependencies: tuple[RunStage, ...]
    active: bool
    activation_reason: str


def adaptive_stage_dependencies(
    state: AdaptiveWorkflowState | None = None,
) -> tuple[AdaptiveStageDependency, ...]:
    """Return the deterministic conditional DAG for one workflow state."""

    current = state or AdaptiveWorkflowState()
    flagged = bool(current.flagged_species)
    revised = current.reference_bank_revision is not None
    rows = (
        (RunStage.REFERENCE_METADATA, (), True, "always"),
        (RunStage.REFERENCE_MEDIA, (RunStage.REFERENCE_METADATA,), True, "always"),
        (RunStage.REFERENCE_DEDUPLICATION, (RunStage.REFERENCE_MEDIA,), True, "always"),
        (
            RunStage.REFERENCE_QUALITY_ROUTING,
            (RunStage.REFERENCE_DEDUPLICATION,),
            True,
            "always",
        ),
        (
            RunStage.REFERENCE_ADMISSION,
            (RunStage.REFERENCE_QUALITY_ROUTING,),
            True,
            "always",
        ),
        (
            RunStage.REFERENCE_EMBEDDINGS,
            (RunStage.REFERENCE_ADMISSION,),
            True,
            "admission_blocks_first_scoring",
        ),
        (
            RunStage.REFERENCE_GEOGRAPHY_INDEX,
            (RunStage.REFERENCE_EMBEDDINGS,),
            True,
            "embedding_bound_reference_index",
        ),
        (
            RunStage.REFERENCE_PROTOTYPES,
            (RunStage.REFERENCE_EMBEDDINGS,),
            True,
            "always",
        ),
        (
            RunStage.FLICKR_DETECTION,
            (RunStage.POLL_FLICKR,),
            True,
            "canonical_organism_routing",
        ),
        (
            RunStage.FLICKR_EMBEDDING,
            (RunStage.FLICKR_DETECTION,),
            True,
            "durable_full_frame_embeddings",
        ),
        (
            RunStage.FLICKR_GEO_TAXON_PARTITIONING,
            (
                RunStage.FLICKR_GEO_CLUSTERING,
                RunStage.FLICKR_DETECTION,
                RunStage.FLICKR_EMBEDDING,
                RunStage.REGIONAL_CANDIDATE_GENERATION,
            ),
            True,
            "canonical_scoring_grains_ready",
        ),
        (
            RunStage.FAMILY_ROUTING,
            (
                RunStage.REFERENCE_PROTOTYPES,
                RunStage.FLICKR_GEO_TAXON_PARTITIONING,
            ),
            True,
            "retrieval_accelerator_not_hard_gate",
        ),
        (
            RunStage.DYNAMIC_POOL_PLANNING,
            (
                RunStage.REFERENCE_GEOGRAPHY_INDEX,
                RunStage.FLICKR_GEO_TAXON_PARTITIONING,
                RunStage.FAMILY_ROUTING,
            ),
            True,
            "global_local_pool_inputs_ready",
        ),
        (
            RunStage.DYNAMIC_POOL_SCORING,
            (
                RunStage.REFERENCE_EMBEDDINGS,
                RunStage.FLICKR_EMBEDDING,
                RunStage.DYNAMIC_POOL_PLANNING,
            ),
            True,
            "cached_vectors_and_pool_plans_ready",
        ),
        (
            RunStage.PROVISIONAL_FLICKR_SCORING,
            (RunStage.DYNAMIC_POOL_SCORING,),
            True,
            "reference_review_not_required",
        ),
        (
            RunStage.REVIEW_SAMPLE_PLANNING,
            (RunStage.PROVISIONAL_FLICKR_SCORING,),
            True,
            "probability_sample_before_review",
        ),
        (
            RunStage.FLICKR_HUMAN_VERIFICATION,
            (RunStage.REVIEW_SAMPLE_PLANNING,),
            True,
            "blocks_final_inclusion",
        ),
        (
            RunStage.RISK_CONTROLLED_AUDIT,
            (RunStage.FLICKR_HUMAN_VERIFICATION,),
            True,
            "preregistered_quality_evaluation",
        ),
        (
            RunStage.STATISTICAL_REFERENCE_AUDIT,
            (RunStage.RISK_CONTROLLED_AUDIT,),
            True,
            "blocks_species_quality_approval",
        ),
        (
            RunStage.TARGETED_REFERENCE_REVIEW,
            (RunStage.STATISTICAL_REFERENCE_AUDIT,),
            flagged,
            "statistically_flagged_species" if flagged else "no_flagged_species",
        ),
        (
            RunStage.AFFECTED_REFERENCE_REBUILD,
            (RunStage.TARGETED_REFERENCE_REVIEW,),
            revised,
            "bank_revision_available" if revised else "no_bank_revision",
        ),
        (
            RunStage.AFFECTED_RECORD_RESCORE,
            (RunStage.AFFECTED_REFERENCE_REBUILD,),
            revised,
            "bank_revision_available" if revised else "no_bank_revision",
        ),
        (
            RunStage.FINAL_QUALITY_GATE,
            (
                RunStage.FLICKR_HUMAN_VERIFICATION,
                RunStage.STATISTICAL_REFERENCE_AUDIT,
                *(
                    (RunStage.AFFECTED_RECORD_RESCORE,)
                    if revised
                    else (RunStage.TARGETED_REFERENCE_REVIEW,)
                    if flagged
                    else ()
                ),
            ),
            True,
            "release_gate",
        ),
    )
    return tuple(
        AdaptiveStageDependency(
            stage=stage,
            dependencies=dependencies,
            active=active,
            activation_reason=reason,
        )
        for stage, dependencies, active, reason in rows
    )


def validate_adaptive_stage_start(
    stage: RunStage,
    *,
    completed_stages: tuple[RunStage, ...],
    state: AdaptiveWorkflowState | None = None,
) -> None:
    dependencies = {item.stage: item for item in adaptive_stage_dependencies(state)}
    if stage not in dependencies:
        raise ValueError(
            f"stage is outside the adaptive dependency graph: {stage.value}"
        )
    item = dependencies[stage]
    if not item.active:
        raise ValueError(
            f"adaptive stage is inactive: {stage.value}; {item.activation_reason}"
        )
    missing = sorted(
        dependency.value
        for dependency in item.dependencies
        if dependency not in completed_stages
    )
    if missing:
        raise ValueError(
            f"adaptive stage dependencies are incomplete: {stage.value}; "
            f"missing={missing}"
        )


def _canonical(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(sorted({str(value).strip() for value in values}))
    if any(not value for value in normalized):
        raise ValueError(f"{field} contains blank values")
    return normalized


__all__ = [
    "AdaptiveStageDependency",
    "AdaptiveWorkflowState",
    "adaptive_stage_dependencies",
    "validate_adaptive_stage_start",
]
