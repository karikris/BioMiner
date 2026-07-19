from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.target_aware_scoring import (
    TARGET_AWARE_CANDIDATE_POLICY_VERSION,
    TargetAwareAuxiliaryClass,
    TargetAwareScoringPlan,
    build_target_aware_scoring_plan,
    score_target_aware_candidate_union,
)


TARGET_KEY = "gbif:1938069"
TARGET_NAME = "Papilio demoleus"


def _regional_candidate_set(*, candidate_count: int = 25) -> CandidateSet:
    target = CandidateTaxon(
        scientific_name=TARGET_NAME,
        accepted_taxon_key=TARGET_KEY,
        family="Papilionidae",
        genus="Papilio",
        candidate_reasons=("target",),
        source_versions=("regional-candidate-species-v1.0.0",),
        target_candidate=True,
        candidate_priority=0,
    )
    competitors = tuple(
        CandidateTaxon(
            scientific_name=f"Genus{index:02d} species",
            accepted_taxon_key=f"gbif:competitor-{index:02d}",
            family="Nymphalidae" if index < 4 else "Papilionidae",
            genus=f"Genus{index:02d}",
            candidate_reasons=("regional_same_family",),
            source_versions=("regional-candidate-species-v1.0.0",),
            candidate_priority=index,
        )
        for index in range(1, candidate_count)
    )
    species = (target, *competitors)
    return CandidateSet(
        candidate_set_id="regional:complete-union",
        registry_version="registry-v1",
        target_accepted_taxon_key=TARGET_KEY,
        target_scientific_name=TARGET_NAME,
        family_candidates=species,
        genus_candidates=species,
        species_candidates=species,
        prompt_variant_version="object-bioclip-prompts-v1",
        geospatial_scope="cluster-au-1",
        source_evidence=(
            "regional_candidate_set:regional:complete-union",
            "regional_candidate_source:regional-candidate-species-v1.0.0",
        ),
        candidate_set_fingerprint="sha256:" + "a" * 64,
    )


def _known_negatives() -> tuple[TargetAwareAuxiliaryClass, ...]:
    return (
        TargetAwareAuxiliaryClass(
            class_id="artwork",
            display_name="visual artwork",
            source_versions=("negative-manifest-v1",),
        ),
        TargetAwareAuxiliaryClass(
            class_id="non_butterfly_insect",
            display_name="non-butterfly insect",
            source_versions=("negative-manifest-v1",),
        ),
    )


def _visual_domains() -> tuple[TargetAwareAuxiliaryClass, ...]:
    return (
        TargetAwareAuxiliaryClass(
            class_id="adult_field",
            display_name="live adult butterfly in the field",
            source_versions=("reference-domain-v1",),
        ),
        TargetAwareAuxiliaryClass(
            class_id="pinned_specimen",
            display_name="pinned butterfly specimen",
            source_versions=("reference-domain-v1",),
        ),
    )


class _RecordingScorer:
    def __init__(self) -> None:
        self.plan: TargetAwareScoringPlan | None = None

    def score(self, plan: TargetAwareScoringPlan) -> Mapping[str, float]:
        self.plan = plan
        scores = {item.scoring_class_id: 0.5 for item in plan.scoring_classes}
        species = plan.species_classes
        for rank, item in enumerate(species, start=1):
            scores[item.scoring_class_id] = float(len(species) - rank + 1)
        scores[f"species:{TARGET_KEY}"] = -1.0
        scores["family_diagnostic:Nymphalidae"] = 0.99
        scores["family_diagnostic:Papilionidae"] = 0.01
        scores["genus_diagnostic:Papilio"] = -2.0
        scores["known_negative:artwork"] = 0.8
        scores["visual_domain:pinned_specimen"] = 0.7
        return scores


def test_regression_wrong_family_text_top_one_does_not_remove_target() -> None:
    candidate_set = _regional_candidate_set()
    plan = build_target_aware_scoring_plan(
        candidate_set,
        known_negative_classes=_known_negatives(),
        visual_domain_classes=_visual_domains(),
    )
    scorer = _RecordingScorer()

    result = score_target_aware_candidate_union(plan, scorer)

    assert scorer.plan is plan
    assert plan.candidate_policy_version == TARGET_AWARE_CANDIDATE_POLICY_VERSION
    assert len(plan.species_classes) == 25
    assert {item.accepted_taxon_key for item in result.species_scores} == {
        candidate.accepted_taxon_key for candidate in candidate_set.species_candidates
    }
    assert result.target_regional_rank == 25
    assert result.target_decision_score == -1.0
    assert (
        next(
            item for item in result.species_scores if item.target_candidate
        ).candidate_priority
        == 0
    )
    assert result.family_diagnostics[0].class_id == "Nymphalidae"
    assert result.genus_diagnostics[-1].class_id == "Papilio"
    assert {item.class_id for item in result.known_negative_scores} == {
        "artwork",
        "non_butterfly_insect",
    }
    assert {item.class_id for item in result.visual_domain_scores} == {
        "adult_field",
        "pinned_specimen",
    }
    assert result.hierarchy_pruning_applied is False
    assert result.hierarchy_rankings_diagnostic_only is True


def test_target_aware_scoring_rejects_any_missing_candidate_score() -> None:
    plan = build_target_aware_scoring_plan(
        _regional_candidate_set(candidate_count=3),
        known_negative_classes=_known_negatives(),
        visual_domain_classes=_visual_domains(),
    )

    class MissingTargetScorer:
        def score(self, value: TargetAwareScoringPlan) -> Mapping[str, float]:
            return {
                item.scoring_class_id: 0.0
                for item in value.scoring_classes
                if item.scoring_class_id != f"species:{TARGET_KEY}"
            }

    with pytest.raises(ValueError, match=r"missing scores.*species:gbif:1938069"):
        score_target_aware_candidate_union(plan, MissingTargetScorer())


def test_target_aware_scoring_never_calls_the_legacy_guardrail(monkeypatch) -> None:
    from biominer.bioclip import path_cascade_classifier

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(f"legacy guardrail called with {args!r}, {kwargs!r}")

    monkeypatch.setattr(
        path_cascade_classifier,
        "_guardrail_confidence",
        fail_if_called,
    )
    plan = build_target_aware_scoring_plan(
        _regional_candidate_set(candidate_count=3),
        known_negative_classes=_known_negatives(),
        visual_domain_classes=_visual_domains(),
    )

    result = score_target_aware_candidate_union(plan, _RecordingScorer())

    assert len(result.species_scores) == 3


def test_target_aware_plan_requires_versioned_regional_union_identity() -> None:
    candidate_set = _regional_candidate_set(candidate_count=3)
    without_fingerprint = replace(candidate_set, candidate_set_fingerprint=None)

    with pytest.raises(ValueError, match="candidate_set_fingerprint"):
        build_target_aware_scoring_plan(
            without_fingerprint,
            known_negative_classes=_known_negatives(),
            visual_domain_classes=_visual_domains(),
        )

    without_regional_provenance = replace(
        candidate_set,
        source_evidence=("regional_candidate_source:test-v1",),
    )
    with pytest.raises(ValueError, match="regional candidate union provenance"):
        build_target_aware_scoring_plan(
            without_regional_provenance,
            known_negative_classes=_known_negatives(),
            visual_domain_classes=_visual_domains(),
        )


def test_target_aware_plan_requires_negative_and_domain_classes() -> None:
    candidate_set = _regional_candidate_set(candidate_count=3)

    with pytest.raises(ValueError, match="known negative"):
        build_target_aware_scoring_plan(
            candidate_set,
            known_negative_classes=(),
            visual_domain_classes=_visual_domains(),
        )
    with pytest.raises(ValueError, match="visual domain"):
        build_target_aware_scoring_plan(
            candidate_set,
            known_negative_classes=_known_negatives(),
            visual_domain_classes=(),
        )
