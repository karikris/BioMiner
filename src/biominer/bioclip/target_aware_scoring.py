"""Complete-set scoring contract for target-aware few-shot classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.classification_modes import (
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
)


TARGET_AWARE_CANDIDATE_POLICY_VERSION = "target-aware-complete-regional-union-v1.0.0"
TARGET_AWARE_COMPLETE_SET_SCORING_VERSION = "target-aware-complete-set-scoring-v1.0.0"

TargetAwareClassKind = Literal[
    "species",
    "known_negative",
    "visual_domain",
    "family_diagnostic",
    "genus_diagnostic",
]

_CLASS_KIND_ORDER: tuple[TargetAwareClassKind, ...] = (
    "species",
    "known_negative",
    "visual_domain",
    "family_diagnostic",
    "genus_diagnostic",
)


@dataclass(frozen=True, slots=True)
class TargetAwareAuxiliaryClass:
    class_id: str
    display_name: str
    source_versions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "class_id", _required_text(self.class_id, field="class_id")
        )
        object.__setattr__(
            self,
            "display_name",
            _required_text(self.display_name, field="display_name"),
        )
        versions = _unique_text(self.source_versions, field="source_versions")
        if not versions:
            raise ValueError(
                "source_versions must identify versioned auxiliary evidence"
            )
        object.__setattr__(self, "source_versions", versions)


@dataclass(frozen=True, slots=True)
class TargetAwareScoringClass:
    class_kind: TargetAwareClassKind
    class_id: str
    display_name: str
    accepted_taxon_key: str | None = None
    family: str | None = None
    genus: str | None = None
    candidate_reasons: tuple[str, ...] = ()
    source_versions: tuple[str, ...] = ()
    target_candidate: bool = False
    candidate_priority: int | None = None

    @property
    def scoring_class_id(self) -> str:
        return f"{self.class_kind}:{self.class_id}"


@dataclass(frozen=True, slots=True)
class TargetAwareScoringPlan:
    candidate_set_id: str
    candidate_set_fingerprint: str
    geo_cluster_id: str
    target_accepted_taxon_key: str
    target_scientific_name: str
    species_classes: tuple[TargetAwareScoringClass, ...]
    known_negative_classes: tuple[TargetAwareScoringClass, ...]
    visual_domain_classes: tuple[TargetAwareScoringClass, ...]
    family_diagnostic_classes: tuple[TargetAwareScoringClass, ...]
    genus_diagnostic_classes: tuple[TargetAwareScoringClass, ...]
    candidate_policy_version: str = TARGET_AWARE_CANDIDATE_POLICY_VERSION
    classification_mode: str = TARGET_AWARE_FEW_SHOT_CLASSIFICATION

    def __post_init__(self) -> None:
        _required_text(self.candidate_set_id, field="candidate_set_id")
        if not self.candidate_set_id.startswith("regional:"):
            raise ValueError(
                "target-aware scoring plan requires a regional candidate set"
            )
        _canonical_sha256(
            self.candidate_set_fingerprint,
            field="candidate_set_fingerprint",
        )
        _required_text(self.geo_cluster_id, field="geo_cluster_id")
        _required_text(
            self.target_accepted_taxon_key,
            field="target_accepted_taxon_key",
        )
        _required_text(self.target_scientific_name, field="target_scientific_name")
        if self.classification_mode != TARGET_AWARE_FEW_SHOT_CLASSIFICATION:
            raise ValueError(
                "target-aware scoring plan has an invalid classification mode"
            )
        if self.candidate_policy_version != TARGET_AWARE_CANDIDATE_POLICY_VERSION:
            raise ValueError(
                "target-aware scoring plan has an unsupported candidate policy"
            )
        grouped_classes = (
            (self.species_classes, "species"),
            (self.known_negative_classes, "known_negative"),
            (self.visual_domain_classes, "visual_domain"),
            (self.family_diagnostic_classes, "family_diagnostic"),
            (self.genus_diagnostic_classes, "genus_diagnostic"),
        )
        for values, expected_kind in grouped_classes:
            if any(item.class_kind != expected_kind for item in values):
                raise ValueError(
                    f"target-aware {expected_kind} group contains another class kind"
                )
        target_rows = [item for item in self.species_classes if item.target_candidate]
        if len(target_rows) != 1:
            raise ValueError(
                "target-aware scoring plan must contain exactly one target species"
            )
        if target_rows[0].accepted_taxon_key != self.target_accepted_taxon_key:
            raise ValueError(
                "target-aware target flag does not match the target taxon key"
            )
        if not self.known_negative_classes:
            raise ValueError(
                "target-aware scoring requires at least one known negative class"
            )
        if not self.visual_domain_classes:
            raise ValueError(
                "target-aware scoring requires at least one visual domain class"
            )
        priorities = [item.candidate_priority for item in self.species_classes]
        if any(value is None for value in priorities) or sorted(priorities) != list(
            range(len(priorities))
        ):
            raise ValueError(
                "target-aware species priorities must be complete and contiguous"
            )
        scoring_ids = [item.scoring_class_id for item in self.scoring_classes]
        if len(scoring_ids) != len(set(scoring_ids)):
            raise ValueError("target-aware scoring class IDs must be unique")

    @property
    def scoring_classes(self) -> tuple[TargetAwareScoringClass, ...]:
        return (
            *self.species_classes,
            *self.known_negative_classes,
            *self.visual_domain_classes,
            *self.family_diagnostic_classes,
            *self.genus_diagnostic_classes,
        )


class TargetAwareCompleteSetScorer(Protocol):
    def score(self, plan: TargetAwareScoringPlan) -> Mapping[str, float]:
        """Return one finite decision score for every plan scoring-class ID."""


@dataclass(frozen=True, slots=True)
class TargetAwareScoredClass:
    scoring_class_id: str
    class_kind: TargetAwareClassKind
    class_id: str
    display_name: str
    decision_score: float
    rank: int
    accepted_taxon_key: str | None
    family: str | None
    genus: str | None
    candidate_reasons: tuple[str, ...]
    source_versions: tuple[str, ...]
    target_candidate: bool
    candidate_priority: int | None


@dataclass(frozen=True, slots=True)
class TargetAwareCompleteSetResult:
    scoring_version: str
    candidate_policy_version: str
    classification_mode: str
    candidate_set_id: str
    candidate_set_fingerprint: str
    geo_cluster_id: str
    target_accepted_taxon_key: str
    scored_classes: tuple[TargetAwareScoredClass, ...]
    target_decision_score: float
    target_regional_rank: int
    hierarchy_pruning_applied: bool = False
    hierarchy_rankings_diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if self.scoring_version != TARGET_AWARE_COMPLETE_SET_SCORING_VERSION:
            raise ValueError("target-aware result has an unsupported scoring version")
        if self.classification_mode != TARGET_AWARE_FEW_SHOT_CLASSIFICATION:
            raise ValueError("target-aware result has an invalid classification mode")
        _canonical_sha256(
            self.candidate_set_fingerprint,
            field="candidate_set_fingerprint",
        )
        if self.hierarchy_pruning_applied:
            raise ValueError("target-aware results cannot apply hierarchy pruning")
        if not self.hierarchy_rankings_diagnostic_only:
            raise ValueError("target-aware hierarchy rankings must remain diagnostic")
        target_rows = [item for item in self.scored_classes if item.target_candidate]
        if len(target_rows) != 1:
            raise ValueError(
                "target-aware result must contain exactly one target score"
            )
        target = target_rows[0]
        if target.accepted_taxon_key != self.target_accepted_taxon_key:
            raise ValueError("target-aware result target identity is inconsistent")
        if (
            target.rank != self.target_regional_rank
            or target.decision_score != self.target_decision_score
        ):
            raise ValueError("target-aware result target score or rank is inconsistent")

    @property
    def species_scores(self) -> tuple[TargetAwareScoredClass, ...]:
        return self._scores_for("species")

    @property
    def known_negative_scores(self) -> tuple[TargetAwareScoredClass, ...]:
        return self._scores_for("known_negative")

    @property
    def visual_domain_scores(self) -> tuple[TargetAwareScoredClass, ...]:
        return self._scores_for("visual_domain")

    @property
    def family_diagnostics(self) -> tuple[TargetAwareScoredClass, ...]:
        return self._scores_for("family_diagnostic")

    @property
    def genus_diagnostics(self) -> tuple[TargetAwareScoredClass, ...]:
        return self._scores_for("genus_diagnostic")

    def _scores_for(
        self,
        class_kind: TargetAwareClassKind,
    ) -> tuple[TargetAwareScoredClass, ...]:
        return tuple(
            item for item in self.scored_classes if item.class_kind == class_kind
        )


def build_target_aware_scoring_plan(
    candidate_set: CandidateSet,
    *,
    known_negative_classes: Sequence[TargetAwareAuxiliaryClass],
    visual_domain_classes: Sequence[TargetAwareAuxiliaryClass],
) -> TargetAwareScoringPlan:
    if not isinstance(candidate_set, CandidateSet):
        raise TypeError("candidate_set must be a CandidateSet")
    if not candidate_set.candidate_set_id.startswith("regional:"):
        raise ValueError("target-aware scoring requires a regional candidate set ID")
    if not candidate_set.candidate_set_fingerprint:
        raise ValueError("target-aware scoring requires candidate_set_fingerprint")
    if not candidate_set.geospatial_scope:
        raise ValueError("target-aware scoring requires a geo cluster or no_geo scope")
    if not any(
        str(value).startswith("regional_candidate_set:")
        for value in candidate_set.source_evidence
    ):
        raise ValueError(
            "target-aware scoring requires regional candidate union provenance"
        )

    priorities = [item.candidate_priority for item in candidate_set.species_candidates]
    if any(value is None for value in priorities) or sorted(priorities) != list(
        range(len(priorities))
    ):
        raise ValueError(
            "regional candidate priorities must be complete and contiguous"
        )

    species_classes: list[TargetAwareScoringClass] = []
    accepted_keys: set[str] = set()
    ordered_candidates = sorted(
        candidate_set.species_candidates,
        key=lambda item: (int(item.candidate_priority or 0), item.scientific_name),
    )
    for candidate in ordered_candidates:
        accepted_key = _required_text(
            candidate.accepted_taxon_key,
            field="candidate accepted_taxon_key",
        )
        if accepted_key in accepted_keys:
            raise ValueError("target-aware regional candidate keys must be unique")
        accepted_keys.add(accepted_key)
        is_target = accepted_key == candidate_set.target_accepted_taxon_key
        if bool(candidate.target_candidate) != is_target:
            raise ValueError("regional target flag does not match the target taxon key")
        if (
            is_target
            and candidate.scientific_name != candidate_set.target_scientific_name
        ):
            raise ValueError(
                "regional target scientific name does not match candidate set"
            )
        candidate_reasons = _unique_text(
            candidate.candidate_reasons,
            field="candidate_reasons",
        )
        if not candidate_reasons:
            raise ValueError("regional candidate reasons are required")
        source_versions = _unique_text(
            candidate.source_versions,
            field="candidate source_versions",
        )
        if not source_versions:
            raise ValueError("regional candidate source_versions are required")
        species_classes.append(
            TargetAwareScoringClass(
                class_kind="species",
                class_id=accepted_key,
                display_name=_required_text(
                    candidate.scientific_name,
                    field="candidate scientific_name",
                ),
                accepted_taxon_key=accepted_key,
                family=_required_text(candidate.family, field="candidate family"),
                genus=_required_text(candidate.genus, field="candidate genus"),
                candidate_reasons=candidate_reasons,
                source_versions=source_versions,
                target_candidate=is_target,
                candidate_priority=candidate.candidate_priority,
            )
        )
    if candidate_set.target_accepted_taxon_key not in accepted_keys:
        raise ValueError("target species is absent from the regional candidate union")

    negative_items = _auxiliary_scoring_classes(
        known_negative_classes,
        class_kind="known_negative",
        group_name="known negative",
    )
    domain_items = _auxiliary_scoring_classes(
        visual_domain_classes,
        class_kind="visual_domain",
        group_name="visual domain",
    )
    family_items = _hierarchy_diagnostic_classes(
        species_classes,
        rank="family",
    )
    genus_items = _hierarchy_diagnostic_classes(
        species_classes,
        rank="genus",
    )
    return TargetAwareScoringPlan(
        candidate_set_id=candidate_set.candidate_set_id,
        candidate_set_fingerprint=candidate_set.candidate_set_fingerprint,
        geo_cluster_id=candidate_set.geospatial_scope,
        target_accepted_taxon_key=candidate_set.target_accepted_taxon_key,
        target_scientific_name=candidate_set.target_scientific_name,
        species_classes=tuple(species_classes),
        known_negative_classes=negative_items,
        visual_domain_classes=domain_items,
        family_diagnostic_classes=family_items,
        genus_diagnostic_classes=genus_items,
    )


def score_target_aware_candidate_union(
    plan: TargetAwareScoringPlan,
    scorer: TargetAwareCompleteSetScorer,
) -> TargetAwareCompleteSetResult:
    if not isinstance(plan, TargetAwareScoringPlan):
        raise TypeError("plan must be a TargetAwareScoringPlan")
    raw_scores = scorer.score(plan)
    if not isinstance(raw_scores, Mapping):
        raise TypeError("target-aware scorer must return a mapping")
    if any(not isinstance(key, str) for key in raw_scores):
        raise TypeError("target-aware scorer keys must be strings")
    scores = {
        str(key): _finite_score(value, field=f"score[{key}]")
        for key, value in raw_scores.items()
    }
    expected_ids = {item.scoring_class_id for item in plan.scoring_classes}
    actual_ids = set(scores)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        raise ValueError(
            "target-aware scorer returned missing scores: " + ", ".join(missing)
        )
    if unexpected:
        raise ValueError(
            "target-aware scorer returned unexpected scores: " + ", ".join(unexpected)
        )

    scored: list[TargetAwareScoredClass] = []
    for class_kind in _CLASS_KIND_ORDER:
        classes = [
            item for item in plan.scoring_classes if item.class_kind == class_kind
        ]
        ranked = sorted(
            classes,
            key=lambda item: (
                -scores[item.scoring_class_id],
                item.class_id.casefold(),
            ),
        )
        for rank, item in enumerate(ranked, start=1):
            scored.append(
                TargetAwareScoredClass(
                    scoring_class_id=item.scoring_class_id,
                    class_kind=item.class_kind,
                    class_id=item.class_id,
                    display_name=item.display_name,
                    decision_score=scores[item.scoring_class_id],
                    rank=rank,
                    accepted_taxon_key=item.accepted_taxon_key,
                    family=item.family,
                    genus=item.genus,
                    candidate_reasons=item.candidate_reasons,
                    source_versions=item.source_versions,
                    target_candidate=item.target_candidate,
                    candidate_priority=item.candidate_priority,
                )
            )
    target = next(item for item in scored if item.target_candidate)
    return TargetAwareCompleteSetResult(
        scoring_version=TARGET_AWARE_COMPLETE_SET_SCORING_VERSION,
        candidate_policy_version=plan.candidate_policy_version,
        classification_mode=plan.classification_mode,
        candidate_set_id=plan.candidate_set_id,
        candidate_set_fingerprint=plan.candidate_set_fingerprint,
        geo_cluster_id=plan.geo_cluster_id,
        target_accepted_taxon_key=plan.target_accepted_taxon_key,
        scored_classes=tuple(scored),
        target_decision_score=target.decision_score,
        target_regional_rank=target.rank,
    )


def _auxiliary_scoring_classes(
    values: Sequence[TargetAwareAuxiliaryClass],
    *,
    class_kind: Literal["known_negative", "visual_domain"],
    group_name: str,
) -> tuple[TargetAwareScoringClass, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(
            f"target-aware scoring requires at least one {group_name} class"
        )
    if any(not isinstance(item, TargetAwareAuxiliaryClass) for item in items):
        raise TypeError(
            f"{group_name} classes must be TargetAwareAuxiliaryClass values"
        )
    class_ids = [item.class_id for item in items]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError(f"{group_name} class IDs must be unique")
    return tuple(
        TargetAwareScoringClass(
            class_kind=class_kind,
            class_id=item.class_id,
            display_name=item.display_name,
            source_versions=item.source_versions,
        )
        for item in sorted(items, key=lambda item: item.class_id.casefold())
    )


def _hierarchy_diagnostic_classes(
    species: Sequence[TargetAwareScoringClass],
    *,
    rank: Literal["family", "genus"],
) -> tuple[TargetAwareScoringClass, ...]:
    values = sorted(
        {str(getattr(item, rank)) for item in species},
        key=str.casefold,
    )
    class_kind: TargetAwareClassKind = (
        "family_diagnostic" if rank == "family" else "genus_diagnostic"
    )
    return tuple(
        TargetAwareScoringClass(
            class_kind=class_kind,
            class_id=value,
            display_name=value,
            source_versions=_unique_text(
                version
                for item in species
                if getattr(item, rank) == value
                for version in item.source_versions
            ),
        )
        for value in values
    )


def _unique_text(values: Iterable[str], *, field: str = "value") -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _required_text(value, field=field)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return tuple(output)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _finite_score(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite number") from exc
    if not isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _canonical_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    digest = text.removeprefix("sha256:")
    if (
        not text.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} must be a canonical sha256 digest")
    return text


__all__ = [
    "TARGET_AWARE_CANDIDATE_POLICY_VERSION",
    "TARGET_AWARE_COMPLETE_SET_SCORING_VERSION",
    "TargetAwareAuxiliaryClass",
    "TargetAwareCompleteSetResult",
    "TargetAwareCompleteSetScorer",
    "TargetAwareScoredClass",
    "TargetAwareScoringClass",
    "TargetAwareScoringPlan",
    "build_target_aware_scoring_plan",
    "score_target_aware_candidate_union",
]
