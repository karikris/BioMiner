"""Post-hoc taxonomic evidence derived from complete species probabilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite
import re
from typing import Literal

from biominer.bioclip.target_aware_fusion import (
    TargetAwareFusionResult,
    TargetAwareSpeciesFusionScore,
    target_aware_fusion_result_payload,
)
from biominer.bioclip.target_aware_scoring import (
    TargetAwareCompleteSetResult,
    TargetAwareScoredClass,
    TargetAwareScoringClass,
    TargetAwareScoringPlan,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


TAXONOMIC_EVIDENCE_VERSION = "post-species-taxonomic-evidence-v1.0.0"
DERIVED_PARENT_PROBABILITY_KIND = "regional-multiclass-member-species-probability-sum"
DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND = "uncalibrated-direct-text-decision-score"
TAXONOMIC_INCONSISTENCY_VERSION = "derived-vs-direct-top1-disagreement-v1.0.0"
SPECIES_PROBABILITY_SUM_TOLERANCE = 1e-9

TaxonomicRank = Literal["family", "genus"]
_CLASS_KIND_ORDER = {
    "species": 0,
    "known_negative": 1,
    "visual_domain": 2,
    "family_diagnostic": 3,
    "genus_diagnostic": 4,
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class MemberSpeciesProbability:
    """One regional species probability contributing to a parent taxon."""

    accepted_taxon_key: str
    scientific_name: str
    calibrated_probability: float
    regional_rank: int
    target_candidate: bool
    candidate_priority: int


@dataclass(frozen=True, slots=True)
class DerivedTaxonomicEvidence:
    """Derived parent probability and separate direct-text diagnostic."""

    rank: TaxonomicRank
    taxon_name: str
    derived_probability: float
    derived_rank: int
    probability_kind: str
    member_species: tuple[MemberSpeciesProbability, ...]
    member_species_count: int
    direct_text_decision_score: float
    direct_text_rank: int
    direct_text_scoring_class_id: str
    direct_text_score_kind: str
    source_versions: tuple[str, ...]
    candidate_set_fingerprint: str
    evidence_fingerprint: str


@dataclass(frozen=True, slots=True)
class TaxonomicEvidenceResult:
    """Family/genus evidence that cannot mutate the scored species union."""

    evidence_version: str
    classification_mode: str
    query_id: str
    route: str
    candidate_set_id: str
    candidate_set_fingerprint: str
    target_accepted_taxon_key: str
    species_candidate_keys: tuple[str, ...]
    species_candidate_set_modified: bool
    species_probability_sum: float
    family_evidence: tuple[DerivedTaxonomicEvidence, ...]
    genus_evidence: tuple[DerivedTaxonomicEvidence, ...]
    derived_family_top1: str
    direct_text_family_top1: str
    derived_genus_top1: str
    direct_text_genus_top1: str
    inconsistency_version: str
    taxonomic_inconsistency: bool
    inconsistency_codes: tuple[str, ...]
    scoring_plan_fingerprint: str
    source_fusion_fingerprint: str
    direct_text_result_fingerprint: str
    result_fingerprint: str


def derive_taxonomic_evidence(
    *,
    plan: TargetAwareScoringPlan,
    fusion_result: TargetAwareFusionResult,
    direct_text_result: TargetAwareCompleteSetResult,
) -> TaxonomicEvidenceResult:
    """Aggregate complete regional species probabilities into parent evidence."""

    if not isinstance(plan, TargetAwareScoringPlan):
        raise TypeError("plan must be a TargetAwareScoringPlan")
    if not isinstance(fusion_result, TargetAwareFusionResult):
        raise TypeError("fusion_result must be a TargetAwareFusionResult")
    if not isinstance(direct_text_result, TargetAwareCompleteSetResult):
        raise TypeError("direct_text_result must be a TargetAwareCompleteSetResult")
    target_aware_fusion_result_payload(fusion_result)
    _validate_shared_identities(
        plan=plan,
        fusion_result=fusion_result,
        direct_text_result=direct_text_result,
    )

    plan_species = _index_plan_species(plan.species_classes)
    fused_species = _index_fused_species(fusion_result.species_scores)
    expected_keys = set(plan_species)
    if set(fused_species) != expected_keys:
        raise ValueError("fusion species coverage differs from the scoring plan")
    _validate_species_contracts(
        plan_species,
        fused_species,
        candidate_set_fingerprint=plan.candidate_set_fingerprint,
    )
    direct_species_keys = {
        _required_text(item.accepted_taxon_key, field="direct species taxon key")
        for item in direct_text_result.species_scores
    }
    if direct_species_keys != expected_keys:
        raise ValueError("direct-text species coverage differs from the scoring plan")

    raw_species_probability_sum = sum(
        _unit_interval(
            item.regional_calibrated_probability,
            field=f"regional probability[{key}]",
        )
        for key, item in fused_species.items()
    )
    if not isclose(
        raw_species_probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=SPECIES_PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError(
            "regional multiclass species probabilities must sum to one before "
            "taxonomic aggregation"
        )
    species_probability_sum = _bounded_probability_sum(raw_species_probability_sum)
    _validate_regional_ranks(fused_species)

    family_diagnostics = _index_diagnostics(
        direct_text_result.family_diagnostics,
        expected={item.family for item in plan_species.values()},
        rank="family",
    )
    genus_diagnostics = _index_diagnostics(
        direct_text_result.genus_diagnostics,
        expected={item.genus for item in plan_species.values()},
        rank="genus",
    )
    family_evidence = _derive_rank_evidence(
        rank="family",
        plan_species=plan_species,
        fused_species=fused_species,
        direct_diagnostics=family_diagnostics,
        candidate_set_fingerprint=plan.candidate_set_fingerprint,
    )
    genus_evidence = _derive_rank_evidence(
        rank="genus",
        plan_species=plan_species,
        fused_species=fused_species,
        direct_diagnostics=genus_diagnostics,
        candidate_set_fingerprint=plan.candidate_set_fingerprint,
    )
    for rank, evidence in (("family", family_evidence), ("genus", genus_evidence)):
        if not isclose(
            sum(item.derived_probability for item in evidence),
            species_probability_sum,
            rel_tol=0.0,
            abs_tol=SPECIES_PROBABILITY_SUM_TOLERANCE,
        ):
            raise RuntimeError(f"derived {rank} probability mass is not conserved")

    derived_family_top1 = family_evidence[0].taxon_name
    direct_text_family_top1 = min(
        family_evidence,
        key=lambda item: (item.direct_text_rank, item.taxon_name.casefold()),
    ).taxon_name
    derived_genus_top1 = genus_evidence[0].taxon_name
    direct_text_genus_top1 = min(
        genus_evidence,
        key=lambda item: (item.direct_text_rank, item.taxon_name.casefold()),
    ).taxon_name
    codes = tuple(
        code
        for condition, code in (
            (
                derived_family_top1 != direct_text_family_top1,
                "family_top1_disagreement",
            ),
            (
                derived_genus_top1 != direct_text_genus_top1,
                "genus_top1_disagreement",
            ),
        )
        if condition
    )
    values: dict[str, object] = {
        "evidence_version": TAXONOMIC_EVIDENCE_VERSION,
        "classification_mode": fusion_result.classification_mode,
        "query_id": fusion_result.query_id,
        "route": fusion_result.route,
        "candidate_set_id": fusion_result.candidate_set_id,
        "candidate_set_fingerprint": fusion_result.candidate_set_fingerprint,
        "target_accepted_taxon_key": fusion_result.target_accepted_taxon_key,
        "species_candidate_keys": tuple(
            item.accepted_taxon_key for item in plan.species_classes
        ),
        "species_candidate_set_modified": False,
        "species_probability_sum": species_probability_sum,
        "family_evidence": family_evidence,
        "genus_evidence": genus_evidence,
        "derived_family_top1": derived_family_top1,
        "direct_text_family_top1": direct_text_family_top1,
        "derived_genus_top1": derived_genus_top1,
        "direct_text_genus_top1": direct_text_genus_top1,
        "inconsistency_version": TAXONOMIC_INCONSISTENCY_VERSION,
        "taxonomic_inconsistency": bool(codes),
        "inconsistency_codes": codes,
        "scoring_plan_fingerprint": _scoring_plan_fingerprint(plan),
        "source_fusion_fingerprint": fusion_result.fusion_fingerprint,
        "direct_text_result_fingerprint": _direct_text_result_fingerprint(
            direct_text_result
        ),
    }
    fingerprint = canonical_semantic_fingerprint(_result_semantics(values))
    return TaxonomicEvidenceResult(**values, result_fingerprint=fingerprint)


def taxonomic_evidence_result_payload(
    result: TaxonomicEvidenceResult,
) -> dict[str, object]:
    """Validate and return the semantic payload for derived hierarchy evidence."""

    if not isinstance(result, TaxonomicEvidenceResult):
        raise TypeError("result must be a TaxonomicEvidenceResult")
    if result.evidence_version != TAXONOMIC_EVIDENCE_VERSION:
        raise ValueError("taxonomic evidence version is incompatible")
    if result.species_candidate_set_modified:
        raise ValueError("taxonomic evidence cannot modify the species candidate set")
    if result.inconsistency_version != TAXONOMIC_INCONSISTENCY_VERSION:
        raise ValueError("taxonomic inconsistency version is incompatible")
    for field_name in (
        "candidate_set_fingerprint",
        "scoring_plan_fingerprint",
        "source_fusion_fingerprint",
        "direct_text_result_fingerprint",
    ):
        _sha256(getattr(result, field_name), field=field_name)
    probability_sum = _unit_interval(
        result.species_probability_sum,
        field="species_probability_sum",
    )
    if not isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=SPECIES_PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("taxonomic evidence species probability sum is invalid")
    family_members = _validate_result_rank_evidence(
        result.family_evidence,
        rank="family",
        expected_probability_sum=probability_sum,
        candidate_set_fingerprint=result.candidate_set_fingerprint,
    )
    genus_members = _validate_result_rank_evidence(
        result.genus_evidence,
        rank="genus",
        expected_probability_sum=probability_sum,
        candidate_set_fingerprint=result.candidate_set_fingerprint,
    )
    species_keys = tuple(
        _required_text(value, field="species_candidate_key")
        for value in result.species_candidate_keys
    )
    if not species_keys or len(species_keys) != len(set(species_keys)):
        raise ValueError("species candidate keys must be nonempty and unique")
    if set(family_members) != set(species_keys) or set(genus_members) != set(
        species_keys
    ):
        raise ValueError("parent evidence does not partition the species candidates")
    if any(family_members[key] != genus_members[key] for key in species_keys):
        raise ValueError("family and genus evidence disagree on species probabilities")
    ordered_members = tuple(
        sorted(
            family_members.values(),
            key=lambda item: (item.candidate_priority, item.accepted_taxon_key),
        )
    )
    if species_keys != tuple(item.accepted_taxon_key for item in ordered_members):
        raise ValueError("species candidate keys are not in candidate-priority order")
    if [item.candidate_priority for item in ordered_members] != list(
        range(len(ordered_members))
    ):
        raise ValueError("species candidate priorities are not contiguous")
    targets = tuple(item for item in ordered_members if item.target_candidate)
    if (
        len(targets) != 1
        or targets[0].accepted_taxon_key != result.target_accepted_taxon_key
    ):
        raise ValueError("taxonomic evidence target species identity is inconsistent")
    expected_top1 = (
        result.family_evidence[0].taxon_name,
        min(
            result.family_evidence,
            key=lambda item: (item.direct_text_rank, item.taxon_name.casefold()),
        ).taxon_name,
        result.genus_evidence[0].taxon_name,
        min(
            result.genus_evidence,
            key=lambda item: (item.direct_text_rank, item.taxon_name.casefold()),
        ).taxon_name,
    )
    if expected_top1 != (
        result.derived_family_top1,
        result.direct_text_family_top1,
        result.derived_genus_top1,
        result.direct_text_genus_top1,
    ):
        raise ValueError("taxonomic top-one evidence is inconsistent")
    expected_codes = tuple(
        code
        for condition, code in (
            (
                result.derived_family_top1 != result.direct_text_family_top1,
                "family_top1_disagreement",
            ),
            (
                result.derived_genus_top1 != result.direct_text_genus_top1,
                "genus_top1_disagreement",
            ),
        )
        if condition
    )
    if result.inconsistency_codes != expected_codes or (
        result.taxonomic_inconsistency != bool(expected_codes)
    ):
        raise ValueError("taxonomic inconsistency fields are inconsistent")
    values = {
        name: getattr(result, name)
        for name in TaxonomicEvidenceResult.__dataclass_fields__
        if name != "result_fingerprint"
    }
    fingerprint = _sha256(result.result_fingerprint, field="result_fingerprint")
    semantics = _result_semantics(values)
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("taxonomic evidence result fingerprint is inconsistent")
    return {**semantics, "result_fingerprint": fingerprint}


def _validate_shared_identities(
    *,
    plan: TargetAwareScoringPlan,
    fusion_result: TargetAwareFusionResult,
    direct_text_result: TargetAwareCompleteSetResult,
) -> None:
    identities = (
        (
            plan.candidate_set_id,
            fusion_result.candidate_set_id,
            direct_text_result.candidate_set_id,
        ),
        (
            plan.candidate_set_fingerprint,
            fusion_result.candidate_set_fingerprint,
            direct_text_result.candidate_set_fingerprint,
        ),
        (
            plan.target_accepted_taxon_key,
            fusion_result.target_accepted_taxon_key,
            direct_text_result.target_accepted_taxon_key,
        ),
        (
            plan.classification_mode,
            fusion_result.classification_mode,
            direct_text_result.classification_mode,
        ),
    )
    if any(len(set(values)) != 1 for values in identities):
        raise ValueError("taxonomic evidence candidate-set identity is incompatible")
    if direct_text_result.hierarchy_pruning_applied:
        raise ValueError("direct hierarchy diagnostics cannot use hierarchy pruning")
    if not direct_text_result.hierarchy_rankings_diagnostic_only:
        raise ValueError("direct hierarchy scores must remain diagnostic only")


def _index_plan_species(
    values: Sequence[TargetAwareScoringClass],
) -> dict[str, TargetAwareScoringClass]:
    rows = tuple(values)
    if len(rows) < 2:
        raise ValueError("taxonomic evidence requires target and competitor species")
    result: dict[str, TargetAwareScoringClass] = {}
    priorities = []
    for item in rows:
        if item.class_kind != "species":
            raise ValueError("taxonomic evidence plan contains a non-species row")
        key = _required_text(item.accepted_taxon_key, field="accepted_taxon_key")
        if key in result:
            raise ValueError("taxonomic evidence plan contains duplicate species")
        _required_text(item.family, field=f"family[{key}]")
        _required_text(item.genus, field=f"genus[{key}]")
        priority = _nonnegative_integer(
            item.candidate_priority,
            field=f"candidate_priority[{key}]",
        )
        priorities.append(priority)
        result[key] = item
    if sorted(priorities) != list(range(len(rows))):
        raise ValueError("species candidate priorities are not complete and contiguous")
    return result


def _index_fused_species(
    values: Sequence[TargetAwareSpeciesFusionScore],
) -> dict[str, TargetAwareSpeciesFusionScore]:
    rows = tuple(values)
    if any(not isinstance(item, TargetAwareSpeciesFusionScore) for item in rows):
        raise TypeError("fusion species rows have an incompatible type")
    result = {item.accepted_taxon_key: item for item in rows}
    if len(result) != len(rows):
        raise ValueError("fusion result contains duplicate species")
    return result


def _validate_species_contracts(
    plan_species: Mapping[str, TargetAwareScoringClass],
    fused_species: Mapping[str, TargetAwareSpeciesFusionScore],
    *,
    candidate_set_fingerprint: str,
) -> None:
    for key, plan_item in plan_species.items():
        fused_item = fused_species[key]
        if fused_item.scientific_name != plan_item.display_name:
            raise ValueError("fused species scientific name differs from the plan")
        if fused_item.candidate_priority != plan_item.candidate_priority:
            raise ValueError("fused species candidate priority differs from the plan")
        if fused_item.target_candidate != plan_item.target_candidate:
            raise ValueError("fused species target flag differs from the plan")
        if fused_item.candidate_set_fingerprint != candidate_set_fingerprint:
            raise ValueError(
                "fused species candidate-set identity differs from the plan"
            )


def _validate_regional_ranks(
    values: Mapping[str, TargetAwareSpeciesFusionScore],
) -> None:
    expected = tuple(
        item.accepted_taxon_key
        for item in sorted(
            values.values(),
            key=lambda item: (
                -item.regional_calibrated_probability,
                item.accepted_taxon_key,
            ),
        )
    )
    actual = tuple(
        item.accepted_taxon_key
        for item in sorted(values.values(), key=lambda item: item.regional_rank)
    )
    ranks = sorted(item.regional_rank for item in values.values())
    if ranks != list(range(1, len(values) + 1)) or actual != expected:
        raise ValueError("regional species ranks are inconsistent with probabilities")


def _index_diagnostics(
    values: Sequence[TargetAwareScoredClass],
    *,
    expected: set[str | None],
    rank: TaxonomicRank,
) -> dict[str, TargetAwareScoredClass]:
    expected_names = {
        _required_text(value, field=f"expected {rank} name") for value in expected
    }
    rows = tuple(values)
    result = {item.class_id: item for item in rows}
    if len(result) != len(rows) or set(result) != expected_names:
        raise ValueError(f"{rank} diagnostic coverage differs from species parents")
    expected_kind = f"{rank}_diagnostic"
    if any(item.class_kind != expected_kind for item in rows):
        raise ValueError(f"{rank} diagnostics contain another class kind")
    for item in rows:
        _finite_float(item.decision_score, field=f"{rank} diagnostic score")
        _positive_integer(item.rank, field=f"{rank} diagnostic rank")
    ranked = tuple(
        item.class_id
        for item in sorted(
            rows,
            key=lambda item: (-item.decision_score, item.class_id.casefold()),
        )
    )
    actual = tuple(item.class_id for item in sorted(rows, key=lambda item: item.rank))
    if actual != ranked or sorted(item.rank for item in rows) != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError(f"{rank} diagnostic ranks are inconsistent")
    return result


def _derive_rank_evidence(
    *,
    rank: TaxonomicRank,
    plan_species: Mapping[str, TargetAwareScoringClass],
    fused_species: Mapping[str, TargetAwareSpeciesFusionScore],
    direct_diagnostics: Mapping[str, TargetAwareScoredClass],
    candidate_set_fingerprint: str,
) -> tuple[DerivedTaxonomicEvidence, ...]:
    grouped: dict[str, list[MemberSpeciesProbability]] = defaultdict(list)
    source_versions: dict[str, set[str]] = defaultdict(set)
    for key, plan_item in plan_species.items():
        parent = _required_text(getattr(plan_item, rank), field=f"{rank}[{key}]")
        fused_item = fused_species[key]
        grouped[parent].append(
            MemberSpeciesProbability(
                accepted_taxon_key=key,
                scientific_name=fused_item.scientific_name,
                calibrated_probability=fused_item.regional_calibrated_probability,
                regional_rank=fused_item.regional_rank,
                target_candidate=fused_item.target_candidate,
                candidate_priority=fused_item.candidate_priority,
            )
        )
        source_versions[parent].update(plan_item.source_versions)
    probabilities = {
        parent: _bounded_probability_sum(
            sum(item.calibrated_probability for item in members)
        )
        for parent, members in grouped.items()
    }
    ranked_parents = tuple(
        sorted(probabilities, key=lambda name: (-probabilities[name], name.casefold()))
    )
    result = []
    for derived_rank, parent in enumerate(ranked_parents, start=1):
        diagnostic = direct_diagnostics[parent]
        members = tuple(
            sorted(
                grouped[parent],
                key=lambda item: (item.candidate_priority, item.accepted_taxon_key),
            )
        )
        values: dict[str, object] = {
            "rank": rank,
            "taxon_name": parent,
            "derived_probability": probabilities[parent],
            "derived_rank": derived_rank,
            "probability_kind": DERIVED_PARENT_PROBABILITY_KIND,
            "member_species": members,
            "member_species_count": len(members),
            "direct_text_decision_score": diagnostic.decision_score,
            "direct_text_rank": diagnostic.rank,
            "direct_text_scoring_class_id": diagnostic.scoring_class_id,
            "direct_text_score_kind": DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND,
            "source_versions": tuple(sorted(source_versions[parent])),
            "candidate_set_fingerprint": candidate_set_fingerprint,
        }
        fingerprint = canonical_semantic_fingerprint(_evidence_semantics(values))
        result.append(
            DerivedTaxonomicEvidence(
                **values,
                evidence_fingerprint=fingerprint,
            )
        )
    return tuple(result)


def _validate_result_rank_evidence(
    values: Sequence[DerivedTaxonomicEvidence],
    *,
    rank: TaxonomicRank,
    expected_probability_sum: float,
    candidate_set_fingerprint: str,
) -> dict[str, MemberSpeciesProbability]:
    rows = tuple(values)
    if not rows:
        raise ValueError(f"taxonomic evidence contains no {rank} rows")
    if any(not isinstance(item, DerivedTaxonomicEvidence) for item in rows):
        raise TypeError(f"taxonomic {rank} evidence has an incompatible type")
    if [item.derived_rank for item in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"taxonomic {rank} derived ranks are not contiguous")
    if tuple(item.taxon_name for item in rows) != tuple(
        item.taxon_name
        for item in sorted(
            rows,
            key=lambda item: (-item.derived_probability, item.taxon_name.casefold()),
        )
    ):
        raise ValueError(f"taxonomic {rank} derived ordering is inconsistent")
    if len({item.taxon_name for item in rows}) != len(rows):
        raise ValueError(f"taxonomic {rank} evidence contains duplicate parents")
    direct_order = tuple(
        item.taxon_name
        for item in sorted(
            rows,
            key=lambda item: (
                -item.direct_text_decision_score,
                item.taxon_name.casefold(),
            ),
        )
    )
    recorded_direct_order = tuple(
        item.taxon_name for item in sorted(rows, key=lambda item: item.direct_text_rank)
    )
    if direct_order != recorded_direct_order or sorted(
        item.direct_text_rank for item in rows
    ) != list(range(1, len(rows) + 1)):
        raise ValueError(f"taxonomic {rank} direct-text ordering is inconsistent")
    members_by_key: dict[str, MemberSpeciesProbability] = {}
    for item in rows:
        if item.rank != rank:
            raise ValueError(f"taxonomic {rank} evidence contains another rank")
        if item.probability_kind != DERIVED_PARENT_PROBABILITY_KIND:
            raise ValueError("derived parent probability kind is incompatible")
        if item.direct_text_score_kind != DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND:
            raise ValueError("direct text diagnostic score kind is incompatible")
        if item.candidate_set_fingerprint != candidate_set_fingerprint:
            raise ValueError("parent evidence candidate-set identity differs")
        _required_text(item.taxon_name, field="taxon_name")
        if item.direct_text_scoring_class_id != f"{rank}_diagnostic:{item.taxon_name}":
            raise ValueError("direct text scoring-class identity is inconsistent")
        if not item.source_versions or item.source_versions != tuple(
            sorted(set(item.source_versions))
        ):
            raise ValueError("parent evidence source versions are not canonical")
        _unit_interval(item.derived_probability, field="derived_probability")
        _finite_float(
            item.direct_text_decision_score,
            field="direct_text_decision_score",
        )
        if item.member_species_count != len(item.member_species) or not (
            item.member_species
        ):
            raise ValueError("parent evidence member count is inconsistent")
        if any(
            not isinstance(member, MemberSpeciesProbability)
            for member in item.member_species
        ):
            raise TypeError("parent evidence members have an incompatible type")
        if tuple(member.accepted_taxon_key for member in item.member_species) != tuple(
            member.accepted_taxon_key
            for member in sorted(
                item.member_species,
                key=lambda member: (
                    member.candidate_priority,
                    member.accepted_taxon_key,
                ),
            )
        ):
            raise ValueError("parent evidence members are not canonically ordered")
        for member in item.member_species:
            key = _required_text(
                member.accepted_taxon_key,
                field="member accepted_taxon_key",
            )
            _required_text(member.scientific_name, field="member scientific_name")
            _positive_integer(member.regional_rank, field="member regional_rank")
            _nonnegative_integer(
                member.candidate_priority,
                field="member candidate_priority",
            )
            if not isinstance(member.target_candidate, bool):
                raise ValueError("member target_candidate must be boolean")
            if key in members_by_key:
                raise ValueError(f"taxonomic {rank} evidence repeats a species")
            members_by_key[key] = member
        member_probability = sum(
            _unit_interval(
                member.calibrated_probability,
                field="member calibrated_probability",
            )
            for member in item.member_species
        )
        if not isclose(
            member_probability,
            item.derived_probability,
            rel_tol=0.0,
            abs_tol=SPECIES_PROBABILITY_SUM_TOLERANCE,
        ):
            raise ValueError("parent probability differs from member-species sum")
        values_without_fingerprint = {
            name: getattr(item, name)
            for name in DerivedTaxonomicEvidence.__dataclass_fields__
            if name != "evidence_fingerprint"
        }
        fingerprint = _sha256(
            item.evidence_fingerprint,
            field="evidence_fingerprint",
        )
        if (
            canonical_semantic_fingerprint(
                _evidence_semantics(values_without_fingerprint)
            )
            != fingerprint
        ):
            raise ValueError("parent taxonomic evidence fingerprint is inconsistent")
    if not isclose(
        sum(item.derived_probability for item in rows),
        expected_probability_sum,
        rel_tol=0.0,
        abs_tol=SPECIES_PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError(f"taxonomic {rank} probability mass is not conserved")
    return members_by_key


def _scoring_plan_fingerprint(plan: TargetAwareScoringPlan) -> str:
    return canonical_semantic_fingerprint(
        {
            "candidate_set_id": plan.candidate_set_id,
            "candidate_set_fingerprint": plan.candidate_set_fingerprint,
            "geo_cluster_id": plan.geo_cluster_id,
            "target_accepted_taxon_key": plan.target_accepted_taxon_key,
            "target_scientific_name": plan.target_scientific_name,
            "candidate_policy_version": plan.candidate_policy_version,
            "classification_mode": plan.classification_mode,
            "scoring_classes": [
                asdict(item)
                for item in sorted(
                    plan.scoring_classes,
                    key=lambda item: (
                        _CLASS_KIND_ORDER[item.class_kind],
                        item.class_id,
                    ),
                )
            ],
        }
    )


def _direct_text_result_fingerprint(result: TargetAwareCompleteSetResult) -> str:
    return canonical_semantic_fingerprint(
        {
            "scoring_version": result.scoring_version,
            "candidate_policy_version": result.candidate_policy_version,
            "classification_mode": result.classification_mode,
            "candidate_set_id": result.candidate_set_id,
            "candidate_set_fingerprint": result.candidate_set_fingerprint,
            "geo_cluster_id": result.geo_cluster_id,
            "target_accepted_taxon_key": result.target_accepted_taxon_key,
            "target_decision_score": result.target_decision_score,
            "target_regional_rank": result.target_regional_rank,
            "hierarchy_pruning_applied": result.hierarchy_pruning_applied,
            "hierarchy_rankings_diagnostic_only": (
                result.hierarchy_rankings_diagnostic_only
            ),
            "scored_classes": [
                asdict(item)
                for item in sorted(
                    result.scored_classes,
                    key=lambda item: (
                        _CLASS_KIND_ORDER[item.class_kind],
                        item.class_id,
                    ),
                )
            ],
        }
    )


def _evidence_semantics(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    result["member_species"] = [
        asdict(item) for item in tuple(values["member_species"])
    ]
    result["source_versions"] = list(values["source_versions"])
    return result


def _result_semantics(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    result["species_candidate_keys"] = list(values["species_candidate_keys"])
    result["family_evidence"] = [
        _derived_evidence_payload(item) for item in tuple(values["family_evidence"])
    ]
    result["genus_evidence"] = [
        _derived_evidence_payload(item) for item in tuple(values["genus_evidence"])
    ]
    result["inconsistency_codes"] = list(values["inconsistency_codes"])
    return result


def _derived_evidence_payload(
    value: DerivedTaxonomicEvidence,
) -> dict[str, object]:
    values = {
        name: getattr(value, name)
        for name in DerivedTaxonomicEvidence.__dataclass_fields__
        if name != "evidence_fingerprint"
    }
    return {
        **_evidence_semantics(values),
        "evidence_fingerprint": value.evidence_fingerprint,
    }


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    result = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return result


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _unit_interval(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _bounded_probability_sum(value: float) -> float:
    if -SPECIES_PROBABILITY_SUM_TOLERANCE <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + SPECIES_PROBABILITY_SUM_TOLERANCE:
        return 1.0
    return _unit_interval(value, field="probability sum")


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


__all__ = [
    "DERIVED_PARENT_PROBABILITY_KIND",
    "DIRECT_TEXT_DIAGNOSTIC_SCORE_KIND",
    "TAXONOMIC_EVIDENCE_VERSION",
    "TAXONOMIC_INCONSISTENCY_VERSION",
    "DerivedTaxonomicEvidence",
    "MemberSpeciesProbability",
    "TaxonomicEvidenceResult",
    "derive_taxonomic_evidence",
    "taxonomic_evidence_result_payload",
]
