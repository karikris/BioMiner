from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES


TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION = "taxonomic-prompt-ensemble-v1.0.0"
TAXONOMIC_PROMPT_VERSION = "bioclip-taxonomic-prompts-v1.0.0"
BUTTERFLY_ROOT_SCIENTIFIC_NAME = "Papilionoidea"
BUTTERFLY_ROOT_ACCEPTED_TAXON_KEY = "gbif:1875"
TRUSTED_VERNACULAR_TIERS = frozenset({"T1", "T2", "T3"})
TRUSTED_VERNACULAR_NAME_CLASSES = frozenset(
    {"vernacular", "vernacular_alias", "common_name", "common_name_alias"}
)
REVIEWED_PROMPT_ALIAS_STATES = frozenset(
    {
        "accepted",
        "approved",
        "curator_reviewed",
        "manual_reviewed",
        "prompt_approved",
        "reviewed",
    }
)
SUPPORTED_PROMPT_LIFE_STAGES = frozenset({"adult", "egg", "larva", "pupa"})
SPECIES_PROMPT_AGGREGATION_DEFAULT = "mean_best_two"

_ROUTE_DEFAULT_LIFE_STAGE = {
    "adult_field": "adult",
    "egg": "egg",
    "larval": "larva",
    "pupal": "pupa",
    "pinned_specimen": "adult",
}
_ROUTE_ALLOWED_LIFE_STAGES = {
    route: frozenset({stage}) for route, stage in _ROUTE_DEFAULT_LIFE_STAGE.items()
}
_TAXONOMIC_RANK_ORDER = {
    "KINGDOM": 10,
    "PHYLUM": 20,
    "CLASS": 30,
    "ORDER": 40,
    "SUPERFAMILY": 50,
    "FAMILY": 60,
    "SUBFAMILY": 70,
    "TRIBE": 80,
    "SUBTRIBE": 90,
    "GENUS": 100,
    "SPECIES": 110,
}
_GENERATED_NAME_CLASSES = frozenset(
    {"generated_translation", "machine_translation", "translated_candidate"}
)
_REJECTED_REVIEW_STATES = frozenset(
    {"denied", "disabled", "excluded", "rejected", "quarantined"}
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class TaxonomicPathNode:
    """One accepted node in a root-to-species taxonomic path."""

    rank: str
    scientific_name: str
    accepted_taxon_key: str | None

    def __post_init__(self) -> None:
        rank = _canonical_text(self.rank, field="rank").upper()
        if rank not in _TAXONOMIC_RANK_ORDER:
            raise ValueError(f"unsupported taxonomic path rank: {rank}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(
            self,
            "scientific_name",
            _canonical_text(self.scientific_name, field="scientific_name"),
        )
        object.__setattr__(
            self, "accepted_taxon_key", _optional_text(self.accepted_taxon_key)
        )


@dataclass(frozen=True, slots=True)
class AcceptedTaxonPromptContext:
    """Accepted taxonomy and immutable provenance used to construct prompts."""

    accepted_taxon_key: str
    scientific_name: str
    genus: str
    family: str
    taxonomic_path: tuple[TaxonomicPathNode, ...]
    taxonomy_source: str
    taxonomy_version: str
    taxonomy_fingerprint: str
    taxonomic_status: str = "ACCEPTED"

    def __post_init__(self) -> None:
        key = _canonical_text(
            self.accepted_taxon_key,
            field="accepted_taxon_key",
        )
        name = _canonical_text(self.scientific_name, field="scientific_name")
        genus = _canonical_text(self.genus, field="genus")
        family = _canonical_text(self.family, field="family")
        source = _canonical_text(self.taxonomy_source, field="taxonomy_source")
        version = _canonical_text(self.taxonomy_version, field="taxonomy_version")
        fingerprint = _sha256(
            self.taxonomy_fingerprint,
            field="taxonomy_fingerprint",
        )
        status = _canonical_text(
            self.taxonomic_status, field="taxonomic_status"
        ).upper()
        if status != "ACCEPTED":
            raise ValueError("taxonomic prompt context must use accepted taxonomy")
        path = tuple(self.taxonomic_path)
        if not path or not all(isinstance(node, TaxonomicPathNode) for node in path):
            raise ValueError("taxonomic_path must contain accepted path nodes")
        ranks = [node.rank for node in path]
        if len(ranks) != len(set(ranks)):
            raise ValueError("taxonomic_path ranks must be unique")
        rank_orders = [_TAXONOMIC_RANK_ORDER[rank] for rank in ranks]
        if rank_orders != sorted(rank_orders):
            raise ValueError("taxonomic_path must be ordered from root to species")
        if path[-1].rank != "SPECIES":
            raise ValueError("taxonomic_path must terminate at species")
        if (
            _path_name(path, "SUPERFAMILY") != BUTTERFLY_ROOT_SCIENTIFIC_NAME
            or _path_key(path, "SUPERFAMILY") != BUTTERFLY_ROOT_ACCEPTED_TAXON_KEY
            or path[-1].scientific_name != name
            or path[-1].accepted_taxon_key != key
            or _path_name(path, "GENUS") != genus
            or _path_name(path, "FAMILY") != family
        ):
            raise ValueError("taxonomic_path does not match accepted species context")
        object.__setattr__(self, "accepted_taxon_key", key)
        object.__setattr__(self, "scientific_name", name)
        object.__setattr__(self, "genus", genus)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "taxonomic_path", path)
        object.__setattr__(self, "taxonomy_source", source)
        object.__setattr__(self, "taxonomy_version", version)
        object.__setattr__(self, "taxonomy_fingerprint", fingerprint)
        object.__setattr__(self, "taxonomic_status", status)


@dataclass(frozen=True, slots=True)
class PromptNameEvidence:
    """One registry vernacular assertion considered for visual prompting."""

    display_name: str
    name_class: str
    trust_tier: str
    source: str
    source_record_id: str
    language: str = "und"
    review_state: str = ""
    weak_homonym: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "display_name",
            _canonical_text(self.display_name, field="display_name"),
        )
        object.__setattr__(
            self,
            "name_class",
            _normalized_token(self.name_class, field="name_class"),
        )
        object.__setattr__(
            self,
            "trust_tier",
            _canonical_text(self.trust_tier, field="trust_tier").upper(),
        )
        object.__setattr__(
            self,
            "source",
            _canonical_text(self.source, field="source"),
        )
        object.__setattr__(
            self,
            "source_record_id",
            _canonical_text(self.source_record_id, field="source_record_id"),
        )
        object.__setattr__(
            self,
            "language",
            _canonical_text(self.language, field="language"),
        )
        object.__setattr__(
            self, "review_state", _normalized_optional_token(self.review_state)
        )
        _require_boolean(self.weak_homonym, field="weak_homonym")
        _require_boolean(self.enabled, field="enabled")


@dataclass(frozen=True, slots=True)
class ReviewedPromptAlias:
    """One human-reviewed visual prompt alias, never a raw search keyword."""

    alias_id: str
    label: str
    source: str
    review_state: str
    reviewed_by: str | None
    route: str | None = None
    life_stage: str | None = None
    weak_homonym: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alias_id",
            _canonical_text(self.alias_id, field="alias_id"),
        )
        object.__setattr__(self, "label", _canonical_text(self.label, field="label"))
        object.__setattr__(
            self,
            "source",
            _canonical_text(self.source, field="source"),
        )
        object.__setattr__(
            self,
            "review_state",
            _normalized_token(self.review_state, field="review_state"),
        )
        object.__setattr__(self, "reviewed_by", _optional_text(self.reviewed_by))
        route = _optional_text(self.route)
        if route is not None and route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported prompt alias route: {route}")
        stage = _optional_text(self.life_stage)
        if stage is not None:
            stage = stage.casefold()
            if stage not in SUPPORTED_PROMPT_LIFE_STAGES:
                raise ValueError(f"unsupported prompt alias life stage: {stage}")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "life_stage", stage)
        _require_boolean(self.weak_homonym, field="weak_homonym")
        _require_boolean(self.enabled, field="enabled")


@dataclass(frozen=True, slots=True)
class PromptEvidenceExclusion:
    evidence_kind: str
    evidence_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class PromptVariant:
    label: str
    taxon_key: str
    prompt_kind: str
    accepted_taxon_key: str = ""
    prompt_version: str = "legacy-unversioned"
    template_id: str = "legacy"
    route: str = "adult_field"
    life_stage: str | None = None
    evidence_kind: str = "legacy"
    evidence_id: str | None = None
    evidence_source: str | None = None
    trust_tier: str | None = None
    language: str | None = None
    review_state: str | None = None
    reviewed_by: str | None = None
    geography_bearing: bool = False
    variant_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class TaxonomicPromptEnsemble:
    schema_version: str
    prompt_version: str
    accepted_taxon_key: str
    scientific_name: str
    route: str
    life_stage: str
    taxonomy_source: str
    taxonomy_version: str
    taxonomy_fingerprint: str
    taxonomic_status: str
    taxonomic_path: tuple[TaxonomicPathNode, ...]
    variants: tuple[PromptVariant, ...]
    exclusions: tuple[PromptEvidenceExclusion, ...]
    ensemble_fingerprint: str


def build_taxonomic_prompt_ensemble(
    *,
    context: AcceptedTaxonPromptContext,
    route: str,
    life_stage: str | None = None,
    vernacular_names: Sequence[PromptNameEvidence] = (),
    reviewed_aliases: Sequence[ReviewedPromptAlias] = (),
) -> TaxonomicPromptEnsemble:
    """Build deterministic visual prompts from accepted and reviewed evidence."""

    if not isinstance(context, AcceptedTaxonPromptContext):
        raise TypeError("context must be an AcceptedTaxonPromptContext")
    route_value = _canonical_text(route, field="route")
    if route_value not in REFERENCE_ROUTES:
        raise ValueError(f"unsupported prompt route: {route_value}")
    stage = (
        _ROUTE_DEFAULT_LIFE_STAGE[route_value]
        if life_stage is None
        else _canonical_text(life_stage, field="life_stage").casefold()
    )
    if stage not in SUPPORTED_PROMPT_LIFE_STAGES:
        raise ValueError(f"unsupported prompt life stage: {stage}")
    if stage not in _ROUTE_ALLOWED_LIFE_STAGES[route_value]:
        raise ValueError(f"life stage {stage} is incompatible with route {route_value}")

    names = _validated_unique_evidence(
        vernacular_names,
        expected_type=PromptNameEvidence,
        id_field="source_record_id",
        field="vernacular_names",
    )
    aliases = _validated_unique_evidence(
        reviewed_aliases,
        expected_type=ReviewedPromptAlias,
        id_field="alias_id",
        field="reviewed_aliases",
    )
    variants: list[PromptVariant] = []
    exclusions: list[PromptEvidenceExclusion] = []
    labels_seen: set[str] = set()

    def add_variant(
        *,
        label: str,
        prompt_kind: str,
        template_id: str,
        evidence_kind: str,
        evidence_id: str | None,
        evidence_source: str | None,
        trust_tier: str | None = None,
        language: str | None = None,
        review_state: str | None = None,
        reviewed_by: str | None = None,
    ) -> bool:
        canonical_label = _canonical_text(label, field="prompt label")
        label_key = canonical_label.casefold()
        if label_key in labels_seen:
            if evidence_id is not None:
                exclusions.append(
                    PromptEvidenceExclusion(
                        evidence_kind=evidence_kind,
                        evidence_id=evidence_id,
                        reason="duplicate_prompt_label",
                    )
                )
            return False
        values: dict[str, object] = {
            "label": canonical_label,
            "taxon_key": context.scientific_name,
            "prompt_kind": prompt_kind,
            "accepted_taxon_key": context.accepted_taxon_key,
            "prompt_version": TAXONOMIC_PROMPT_VERSION,
            "template_id": template_id,
            "route": route_value,
            "life_stage": stage,
            "evidence_kind": evidence_kind,
            "evidence_id": evidence_id,
            "evidence_source": evidence_source,
            "trust_tier": trust_tier,
            "language": language,
            "review_state": review_state,
            "reviewed_by": reviewed_by,
            "geography_bearing": False,
        }
        fingerprint = canonical_semantic_fingerprint(values)
        variants.append(
            PromptVariant(
                **values,
                variant_fingerprint=fingerprint,
            )
        )
        labels_seen.add(label_key)
        return True

    accepted_evidence_id = f"accepted_taxon:{context.accepted_taxon_key}"
    add_variant(
        label=context.scientific_name,
        prompt_kind="accepted_scientific_name",
        template_id="species_scientific_name_v1",
        evidence_kind="accepted_taxonomy",
        evidence_id=accepted_evidence_id,
        evidence_source=context.taxonomy_source,
        trust_tier="T1",
    )
    add_variant(
        label=f"the butterfly species {context.scientific_name}",
        prompt_kind="species_description",
        template_id="species_description_v1",
        evidence_kind="accepted_taxonomy",
        evidence_id=accepted_evidence_id,
        evidence_source=context.taxonomy_source,
        trust_tier="T1",
    )
    add_variant(
        label=_life_stage_prompt(
            context.scientific_name, route=route_value, stage=stage
        ),
        prompt_kind=f"life_stage_{stage}",
        template_id=f"life_stage_{route_value}_{stage}_v1",
        evidence_kind="accepted_taxonomy",
        evidence_id=accepted_evidence_id,
        evidence_source=context.taxonomy_source,
        trust_tier="T1",
    )
    add_variant(
        label=(
            f"{context.scientific_name}, a species of {context.genus} "
            f"in {context.family}"
        ),
        prompt_kind="genus_family_hierarchy",
        template_id="species_genus_family_v1",
        evidence_kind="accepted_taxonomy",
        evidence_id=accepted_evidence_id,
        evidence_source=context.taxonomy_source,
        trust_tier="T1",
    )
    add_variant(
        label=_taxonomic_path_prompt(context),
        prompt_kind="accepted_taxonomic_path",
        template_id="accepted_taxonomic_path_v1",
        evidence_kind="accepted_taxonomy",
        evidence_id=accepted_evidence_id,
        evidence_source=context.taxonomy_source,
        trust_tier="T1",
    )
    if route_value == "pinned_specimen":
        add_variant(
            label=f"a pinned museum specimen of {context.scientific_name}",
            prompt_kind="pinned_specimen",
            template_id="pinned_specimen_v1",
            evidence_kind="accepted_taxonomy",
            evidence_id=accepted_evidence_id,
            evidence_source=context.taxonomy_source,
            trust_tier="T1",
        )

    for name in names:
        reason = _vernacular_exclusion_reason(name)
        if reason is not None:
            exclusions.append(
                PromptEvidenceExclusion(
                    evidence_kind="vernacular_name",
                    evidence_id=name.source_record_id,
                    reason=reason,
                )
            )
            continue
        add_variant(
            label=(
                f"a field photograph of the {name.display_name} "
                f"({context.scientific_name})"
            ),
            prompt_kind="trusted_vernacular_with_scientific_name",
            template_id="vernacular_scientific_name_v1",
            evidence_kind="vernacular_name",
            evidence_id=name.source_record_id,
            evidence_source=name.source,
            trust_tier=name.trust_tier,
            language=name.language,
            review_state=name.review_state or None,
        )

    for alias in aliases:
        reason = _alias_exclusion_reason(alias, route=route_value, life_stage=stage)
        if reason is not None:
            exclusions.append(
                PromptEvidenceExclusion(
                    evidence_kind="reviewed_prompt_alias",
                    evidence_id=alias.alias_id,
                    reason=reason,
                )
            )
            continue
        add_variant(
            label=alias.label,
            prompt_kind="reviewed_prompt_alias",
            template_id="reviewed_prompt_alias_v1",
            evidence_kind="reviewed_prompt_alias",
            evidence_id=alias.alias_id,
            evidence_source=alias.source,
            review_state=alias.review_state,
            reviewed_by=alias.reviewed_by,
        )

    exclusions_tuple = tuple(
        sorted(
            exclusions,
            key=lambda item: (item.evidence_kind, item.evidence_id, item.reason),
        )
    )
    variants_tuple = tuple(variants)
    semantics = {
        "schema_version": TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION,
        "prompt_version": TAXONOMIC_PROMPT_VERSION,
        "accepted_taxon_key": context.accepted_taxon_key,
        "scientific_name": context.scientific_name,
        "route": route_value,
        "life_stage": stage,
        "taxonomy_source": context.taxonomy_source,
        "taxonomy_version": context.taxonomy_version,
        "taxonomy_fingerprint": context.taxonomy_fingerprint,
        "taxonomic_status": context.taxonomic_status,
        "taxonomic_path": [_path_node_payload(node) for node in context.taxonomic_path],
        "variants": [_variant_payload(variant) for variant in variants_tuple],
        "exclusions": [
            {
                "evidence_kind": item.evidence_kind,
                "evidence_id": item.evidence_id,
                "reason": item.reason,
            }
            for item in exclusions_tuple
        ],
    }
    return TaxonomicPromptEnsemble(
        schema_version=TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION,
        prompt_version=TAXONOMIC_PROMPT_VERSION,
        accepted_taxon_key=context.accepted_taxon_key,
        scientific_name=context.scientific_name,
        route=route_value,
        life_stage=stage,
        taxonomy_source=context.taxonomy_source,
        taxonomy_version=context.taxonomy_version,
        taxonomy_fingerprint=context.taxonomy_fingerprint,
        taxonomic_status=context.taxonomic_status,
        taxonomic_path=context.taxonomic_path,
        variants=variants_tuple,
        exclusions=exclusions_tuple,
        ensemble_fingerprint=canonical_semantic_fingerprint(semantics),
    )


def build_species_prompt_variants(
    *,
    context: AcceptedTaxonPromptContext,
    route: str = "adult_field",
    life_stage: str | None = None,
    vernacular_names: Sequence[PromptNameEvidence] = (),
    reviewed_aliases: Sequence[ReviewedPromptAlias] = (),
) -> list[PromptVariant]:
    """Compatibility projection for callers that consume only prompt variants."""

    return list(
        build_taxonomic_prompt_ensemble(
            context=context,
            route=route,
            life_stage=life_stage,
            vernacular_names=vernacular_names,
            reviewed_aliases=reviewed_aliases,
        ).variants
    )


def taxonomic_prompt_ensemble_payload(
    ensemble: TaxonomicPromptEnsemble,
) -> dict[str, object]:
    """Validate and return the complete fingerprinted ensemble payload."""

    if not isinstance(ensemble, TaxonomicPromptEnsemble):
        raise TypeError("ensemble must be a TaxonomicPromptEnsemble")
    AcceptedTaxonPromptContext(
        accepted_taxon_key=ensemble.accepted_taxon_key,
        scientific_name=ensemble.scientific_name,
        genus=str(_path_name(ensemble.taxonomic_path, "GENUS") or ""),
        family=str(_path_name(ensemble.taxonomic_path, "FAMILY") or ""),
        taxonomic_path=ensemble.taxonomic_path,
        taxonomy_source=ensemble.taxonomy_source,
        taxonomy_version=ensemble.taxonomy_version,
        taxonomy_fingerprint=ensemble.taxonomy_fingerprint,
        taxonomic_status=ensemble.taxonomic_status,
    )
    if ensemble.schema_version != TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("taxonomic prompt ensemble schema version is incompatible")
    if ensemble.prompt_version != TAXONOMIC_PROMPT_VERSION:
        raise ValueError("taxonomic prompt version is incompatible")
    if ensemble.route not in REFERENCE_ROUTES:
        raise ValueError("taxonomic prompt ensemble route is incompatible")
    if ensemble.life_stage not in _ROUTE_ALLOWED_LIFE_STAGES[ensemble.route]:
        raise ValueError("taxonomic prompt ensemble life stage is incompatible")
    if not ensemble.variants:
        raise ValueError("taxonomic prompt ensemble must contain variants")
    labels: set[str] = set()
    fingerprints: set[str] = set()
    for variant in ensemble.variants:
        if not isinstance(variant, PromptVariant):
            raise TypeError("taxonomic prompt variants must be PromptVariant values")
        if (
            variant.prompt_version != ensemble.prompt_version
            or variant.accepted_taxon_key != ensemble.accepted_taxon_key
            or variant.taxon_key != ensemble.scientific_name
            or variant.route != ensemble.route
            or variant.life_stage != ensemble.life_stage
        ):
            raise ValueError("taxonomic prompt variant identity is inconsistent")
        label_key = variant.label.casefold()
        if label_key in labels:
            raise ValueError("taxonomic prompt variant labels must be unique")
        labels.add(label_key)
        fingerprint = _sha256(
            variant.variant_fingerprint,
            field="variant_fingerprint",
        )
        if fingerprint in fingerprints:
            raise ValueError("taxonomic prompt variant fingerprints must be unique")
        fingerprints.add(fingerprint)
        if canonical_semantic_fingerprint(_variant_semantic_payload(variant)) != (
            fingerprint
        ):
            raise ValueError("taxonomic prompt variant fingerprint is inconsistent")
    expected_exclusions = tuple(
        sorted(
            ensemble.exclusions,
            key=lambda item: (item.evidence_kind, item.evidence_id, item.reason),
        )
    )
    if expected_exclusions != ensemble.exclusions or not all(
        isinstance(item, PromptEvidenceExclusion) for item in ensemble.exclusions
    ):
        raise ValueError("taxonomic prompt exclusions are not canonical")
    semantics = {
        "schema_version": ensemble.schema_version,
        "prompt_version": ensemble.prompt_version,
        "accepted_taxon_key": ensemble.accepted_taxon_key,
        "scientific_name": ensemble.scientific_name,
        "route": ensemble.route,
        "life_stage": ensemble.life_stage,
        "taxonomy_source": ensemble.taxonomy_source,
        "taxonomy_version": ensemble.taxonomy_version,
        "taxonomy_fingerprint": ensemble.taxonomy_fingerprint,
        "taxonomic_status": ensemble.taxonomic_status,
        "taxonomic_path": [
            _path_node_payload(node) for node in ensemble.taxonomic_path
        ],
        "variants": [_variant_payload(variant) for variant in ensemble.variants],
        "exclusions": [
            {
                "evidence_kind": item.evidence_kind,
                "evidence_id": item.evidence_id,
                "reason": item.reason,
            }
            for item in ensemble.exclusions
        ],
    }
    fingerprint = _sha256(
        ensemble.ensemble_fingerprint,
        field="ensemble_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("taxonomic prompt ensemble fingerprint is inconsistent")
    return {**semantics, "ensemble_fingerprint": fingerprint}


def aggregate_prompt_scores(
    *,
    scores: Mapping[str, float],
    variants: Sequence[PromptVariant],
    top_k: int,
    aggregation: str = SPECIES_PROMPT_AGGREGATION_DEFAULT,
) -> list[dict[str, object]]:
    if aggregation not in {"max", "mean", "mean_best_two"}:
        raise ValueError("aggregation must be one of: max, mean, mean_best_two")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    expected_labels = [variant.label for variant in variants]
    if len(expected_labels) != len(set(expected_labels)):
        raise ValueError("prompt variant labels must be unique")
    if set(scores) != set(expected_labels):
        raise ValueError("prompt score key set does not match prompt variants")
    if any(not isfinite(float(score)) for score in scores.values()):
        raise ValueError("prompt scores must be finite")
    grouped: dict[str, dict[str, object]] = {}
    for variant in variants:
        score = float(scores[variant.label])
        current = grouped.setdefault(
            variant.taxon_key,
            {
                "taxon_key": variant.taxon_key,
                "score": 0.0,
                "best_label": None,
                "prompt_scores": {},
            },
        )
        prompt_scores = current["prompt_scores"]
        assert isinstance(prompt_scores, dict)
        prompt_scores[variant.label] = score
        if current["best_label"] is None or score >= float(
            scores.get(str(current["best_label"]), 0.0)
        ):
            current["best_label"] = variant.label
    rows = []
    for current in grouped.values():
        prompt_scores = current["prompt_scores"]
        assert isinstance(prompt_scores, dict)
        contributions = _score_contributions(prompt_scores, aggregation=aggregation)
        current["score"] = sum(score for _, score in contributions) / len(contributions)
        current["pooling_strategy"] = aggregation
        current["prompt_count"] = len(prompt_scores)
        current["contributing_prompt_labels"] = [label for label, _ in contributions]
        rows.append(current)
    return sorted(
        rows,
        key=lambda row: (-float(row["score"]), str(row["taxon_key"])),
    )[:top_k]


def _life_stage_prompt(scientific_name: str, *, route: str, stage: str) -> str:
    if route == "pinned_specimen":
        return f"an adult {scientific_name} butterfly specimen"
    if stage == "adult":
        return f"a field photograph of an adult {scientific_name} butterfly"
    if stage == "larva":
        return f"a field photograph of a {scientific_name} caterpillar"
    if stage == "pupa":
        return f"a field photograph of a {scientific_name} chrysalis"
    return f"a field photograph of a {scientific_name} egg"


def _taxonomic_path_prompt(context: AcceptedTaxonPromptContext) -> str:
    path = "; ".join(
        f"{node.rank.casefold()} {node.scientific_name}"
        for node in context.taxonomic_path
    )
    return f"{context.scientific_name}, accepted taxonomic path: {path}"


def _vernacular_exclusion_reason(value: PromptNameEvidence) -> str | None:
    if not value.enabled:
        return "disabled_evidence"
    if value.weak_homonym:
        return "weak_homonym"
    source = _normalized_optional_token(value.source)
    if (
        value.name_class in _GENERATED_NAME_CLASSES
        or value.trust_tier == "T5"
        or "translation" in source
    ):
        return "generated_translation"
    if value.name_class not in TRUSTED_VERNACULAR_NAME_CLASSES:
        return "unsupported_vernacular_name_class"
    if value.trust_tier not in TRUSTED_VERNACULAR_TIERS:
        return "untrusted_vernacular_tier"
    if value.review_state in _REJECTED_REVIEW_STATES:
        return "name_review_rejected"
    return None


def _alias_exclusion_reason(
    value: ReviewedPromptAlias,
    *,
    route: str,
    life_stage: str,
) -> str | None:
    if not value.enabled:
        return "disabled_evidence"
    if value.weak_homonym:
        return "weak_homonym"
    if value.review_state not in REVIEWED_PROMPT_ALIAS_STATES:
        return "unreviewed_prompt_alias"
    if value.reviewed_by is None:
        return "prompt_alias_missing_reviewer"
    if route != "pinned_specimen" and _is_pinned_specimen_prompt(value.label):
        return "pinned_alias_requires_specimen_route"
    if value.route is not None and value.route != route:
        return "prompt_alias_route_mismatch"
    if value.life_stage is not None and value.life_stage != life_stage:
        return "prompt_alias_life_stage_mismatch"
    return None


def _is_pinned_specimen_prompt(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return bool(re.search(r"\bpinned\b|\bmuseum specimen\b", normalized))


def _validated_unique_evidence(
    values: Sequence[object],
    *,
    expected_type: type,
    id_field: str,
    field: str,
) -> tuple:
    by_id: dict[str, object] = {}
    for value in values:
        if not isinstance(value, expected_type):
            raise TypeError(f"{field} must contain {expected_type.__name__} values")
        evidence_id = str(getattr(value, id_field))
        previous = by_id.get(evidence_id)
        if previous is not None and previous != value:
            raise ValueError(f"{field} evidence IDs must identify one immutable record")
        by_id[evidence_id] = value
    return tuple(sorted(by_id.values(), key=lambda item: str(getattr(item, id_field))))


def _variant_payload(value: PromptVariant) -> dict[str, object]:
    return {
        **_variant_semantic_payload(value),
        "variant_fingerprint": value.variant_fingerprint,
    }


def _variant_semantic_payload(value: PromptVariant) -> dict[str, object]:
    return {
        "label": value.label,
        "taxon_key": value.taxon_key,
        "prompt_kind": value.prompt_kind,
        "accepted_taxon_key": value.accepted_taxon_key,
        "prompt_version": value.prompt_version,
        "template_id": value.template_id,
        "route": value.route,
        "life_stage": value.life_stage,
        "evidence_kind": value.evidence_kind,
        "evidence_id": value.evidence_id,
        "evidence_source": value.evidence_source,
        "trust_tier": value.trust_tier,
        "language": value.language,
        "review_state": value.review_state,
        "reviewed_by": value.reviewed_by,
        "geography_bearing": value.geography_bearing,
    }


def _path_node_payload(value: TaxonomicPathNode) -> dict[str, str | None]:
    return {
        "rank": value.rank,
        "scientific_name": value.scientific_name,
        "accepted_taxon_key": value.accepted_taxon_key,
    }


def _path_name(path: Sequence[TaxonomicPathNode], rank: str) -> str | None:
    return next((node.scientific_name for node in path if node.rank == rank), None)


def _path_key(path: Sequence[TaxonomicPathNode], rank: str) -> str | None:
    return next((node.accepted_taxon_key for node in path if node.rank == rank), None)


def _score_contributions(
    prompt_scores: Mapping[str, object],
    *,
    aggregation: str,
) -> tuple[tuple[str, float], ...]:
    ordered = tuple(
        sorted(
            ((str(label), float(score)) for label, score in prompt_scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
    )
    if not ordered:
        return (("", 0.0),)
    if aggregation == "max":
        return ordered[:1]
    if aggregation == "mean_best_two":
        return ordered[:2]
    return tuple(sorted(ordered, key=lambda item: item[0]))


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    result = " ".join(value.split())
    if not result:
        raise ValueError(f"{field} must be non-empty text")
    return result


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _canonical_text(value, field="optional text")


def _normalized_token(value: object, *, field: str) -> str:
    return _normalized_optional_token(_canonical_text(value, field=field))


def _normalized_optional_token(value: object) -> str:
    return "_".join(str(value or "").casefold().split())


def _sha256(value: object, *, field: str) -> str:
    text = _canonical_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _require_boolean(value: object, *, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")


__all__ = [
    "BUTTERFLY_ROOT_ACCEPTED_TAXON_KEY",
    "BUTTERFLY_ROOT_SCIENTIFIC_NAME",
    "REVIEWED_PROMPT_ALIAS_STATES",
    "SPECIES_PROMPT_AGGREGATION_DEFAULT",
    "SUPPORTED_PROMPT_LIFE_STAGES",
    "TAXONOMIC_PROMPT_ENSEMBLE_SCHEMA_VERSION",
    "TAXONOMIC_PROMPT_VERSION",
    "TRUSTED_VERNACULAR_NAME_CLASSES",
    "TRUSTED_VERNACULAR_TIERS",
    "AcceptedTaxonPromptContext",
    "PromptEvidenceExclusion",
    "PromptNameEvidence",
    "PromptVariant",
    "ReviewedPromptAlias",
    "TaxonomicPathNode",
    "TaxonomicPromptEnsemble",
    "aggregate_prompt_scores",
    "build_species_prompt_variants",
    "build_taxonomic_prompt_ensemble",
    "taxonomic_prompt_ensemble_payload",
]
