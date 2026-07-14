"""Leakage-safe evaluation and validation-only selection for prompt ensembles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
import re

from biominer.bioclip.prompt_pooling import (
    PROMPT_POOLING_STRATEGIES,
    PromptPoolingResult,
    prompt_pooling_result_payload,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.splits import DATASET_SPLIT_SET
from biominer.references.readiness import REFERENCE_ROUTES


PROMPT_EVALUATION_REPORT_SCHEMA_VERSION = "taxonomic-prompt-evaluation-report-v1.0.0"
PROMPT_EVALUATION_VERSION = "taxonomic-prompt-evaluation-v1.0.0"
PROMPT_EVALUATION_CONFIGURATION_SCHEMA_VERSION = (
    "prompt-evaluation-configuration-v1.0.0"
)
PROMPT_VERSION_SELECTION_SCHEMA_VERSION = "prompt-version-selection-v1.0.0"
PROMPT_VERSION_SELECTION_VERSION = "validation-only-prompt-selection-v1.0.0"
REFERENCE_IMAGE_SCORE_KIND = "reference_image_cosine_similarity"
COMMON_NAME_EFFECT = "common_names"
TAXONOMIC_PATH_EFFECT = "taxonomic_path"
PROMPT_EFFECT_KINDS = frozenset({COMMON_NAME_EFFECT, TAXONOMIC_PATH_EFFECT})
DEFAULT_RECALL_KS = (1, 3, 5, 20)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PromptEvaluationConfiguration:
    """One comparable prompt/pooling configuration evaluated as a unit."""

    configuration_id: str
    prompt_version: str
    pooling_strategy: str
    model_fingerprint: str
    common_names_enabled: bool
    taxonomic_path_enabled: bool
    geography_prompt_ablation_enabled: bool = False
    schema_version: str = PROMPT_EVALUATION_CONFIGURATION_SCHEMA_VERSION
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        identifier = _required_text(self.configuration_id, field="configuration_id")
        prompt_version = _required_text(self.prompt_version, field="prompt_version")
        strategy = _required_text(self.pooling_strategy, field="pooling_strategy")
        if strategy not in PROMPT_POOLING_STRATEGIES:
            raise ValueError(f"unsupported prompt pooling strategy: {strategy}")
        model = _sha256(self.model_fingerprint, field="model_fingerprint")
        _require_boolean(self.common_names_enabled, field="common_names_enabled")
        _require_boolean(
            self.taxonomic_path_enabled,
            field="taxonomic_path_enabled",
        )
        _require_boolean(
            self.geography_prompt_ablation_enabled,
            field="geography_prompt_ablation_enabled",
        )
        if self.schema_version != PROMPT_EVALUATION_CONFIGURATION_SCHEMA_VERSION:
            raise ValueError("prompt evaluation configuration schema is incompatible")
        object.__setattr__(self, "configuration_id", identifier)
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(self, "pooling_strategy", strategy)
        object.__setattr__(self, "model_fingerprint", model)
        expected = canonical_semantic_fingerprint(_configuration_semantics(self))
        if self.configuration_fingerprint:
            supplied = _sha256(
                self.configuration_fingerprint,
                field="configuration_fingerprint",
            )
            if supplied != expected:
                raise ValueError("prompt configuration fingerprint is inconsistent")
        object.__setattr__(self, "configuration_fingerprint", expected)


@dataclass(frozen=True, slots=True)
class PromptCandidateEvaluation:
    """Validated text and independent reference evidence for one candidate species."""

    item_id: str
    dataset_split: str
    split_fingerprint: str
    expected_accepted_taxon_key: str
    candidate_accepted_taxon_key: str
    route: str
    life_stage: str
    visual_domain: str
    candidate_set_fingerprint: str
    configuration: PromptEvaluationConfiguration
    pooling_result: PromptPoolingResult
    reference_image_score: float | None
    reference_evidence_fingerprint: str | None
    reference_score_kind: str = REFERENCE_IMAGE_SCORE_KIND

    def __post_init__(self) -> None:
        item_id = _required_text(self.item_id, field="item_id")
        dataset_split = _required_text(self.dataset_split, field="dataset_split")
        if dataset_split not in DATASET_SPLIT_SET:
            raise ValueError(f"unsupported dataset_split: {dataset_split}")
        split = _sha256(self.split_fingerprint, field="split_fingerprint")
        expected_key = _required_text(
            self.expected_accepted_taxon_key,
            field="expected_accepted_taxon_key",
        )
        candidate_key = _required_text(
            self.candidate_accepted_taxon_key,
            field="candidate_accepted_taxon_key",
        )
        route = _required_text(self.route, field="route")
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported prompt evaluation route: {route}")
        life_stage = _required_text(self.life_stage, field="life_stage")
        visual_domain = _required_text(self.visual_domain, field="visual_domain")
        candidate_set = _sha256(
            self.candidate_set_fingerprint,
            field="candidate_set_fingerprint",
        )
        if not isinstance(self.configuration, PromptEvaluationConfiguration):
            raise TypeError("configuration must be a PromptEvaluationConfiguration")
        if not isinstance(self.pooling_result, PromptPoolingResult):
            raise TypeError("pooling_result must be a PromptPoolingResult")
        prompt_pooling_result_payload(self.pooling_result)
        result = self.pooling_result
        configuration = self.configuration
        if (
            result.prompt_version != configuration.prompt_version
            or result.strategy != configuration.pooling_strategy
            or result.model_fingerprint != configuration.model_fingerprint
            or result.geography_prompt_ablation_enabled
            != configuration.geography_prompt_ablation_enabled
        ):
            raise ValueError("prompt result does not match evaluation configuration")
        if result.accepted_taxon_key != candidate_key:
            raise ValueError("prompt result accepted taxon does not match candidate")
        if (
            result.route != route
            or result.life_stage != life_stage
            or result.visual_domain != visual_domain
        ):
            raise ValueError("prompt result route, stage, or domain is inconsistent")
        selected = tuple(item for item in result.prompt_scores if item.in_subset)
        if not selected:
            raise ValueError("prompt evaluation candidate has no selected prompts")
        selected_common_name = any(
            item.evidence_kind == "vernacular_name" for item in selected
        )
        if selected_common_name and not configuration.common_names_enabled:
            raise ValueError("common-name prompt selected by a disabled configuration")
        selected_taxonomic_path = any(
            item.prompt_kind == "accepted_taxonomic_path" for item in selected
        )
        if selected_taxonomic_path != configuration.taxonomic_path_enabled:
            raise ValueError("taxonomic-path prompt selection is inconsistent")
        selected_geography = any(item.geography_bearing for item in selected)
        if selected_geography != configuration.geography_prompt_ablation_enabled:
            raise ValueError("geographic prompt selection is inconsistent")
        if self.reference_score_kind != REFERENCE_IMAGE_SCORE_KIND:
            raise ValueError("reference image score kind is incompatible")
        reference_score = self.reference_image_score
        reference_fingerprint = self.reference_evidence_fingerprint
        if reference_score is None:
            if reference_fingerprint is not None:
                raise ValueError(
                    "missing reference image score cannot have evidence fingerprint"
                )
        else:
            reference_score = _bounded_cosine(
                reference_score,
                field="reference_image_score",
            )
            reference_fingerprint = _sha256(
                reference_fingerprint,
                field="reference_evidence_fingerprint",
            )
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "dataset_split", dataset_split)
        object.__setattr__(self, "split_fingerprint", split)
        object.__setattr__(self, "expected_accepted_taxon_key", expected_key)
        object.__setattr__(self, "candidate_accepted_taxon_key", candidate_key)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "life_stage", life_stage)
        object.__setattr__(self, "visual_domain", visual_domain)
        object.__setattr__(self, "candidate_set_fingerprint", candidate_set)
        object.__setattr__(self, "reference_image_score", reference_score)
        object.__setattr__(
            self,
            "reference_evidence_fingerprint",
            reference_fingerprint,
        )

    @property
    def candidate_fingerprint(self) -> str:
        return canonical_semantic_fingerprint(_candidate_semantics(self))


@dataclass(frozen=True, slots=True)
class PromptEvaluationItemResult:
    dataset_split: str
    configuration_id: str
    configuration_fingerprint: str
    item_id: str
    route: str
    life_stage: str
    visual_domain: str
    expected_accepted_taxon_key: str
    candidate_set_fingerprint: str
    candidate_count: int
    target_prompt_rank: int
    best_target_prompt_label: str
    best_target_prompt_variant_fingerprint: str
    best_target_prompt_score: float
    target_species_rank: int
    target_text_score: float
    best_competitor_accepted_taxon_key: str
    best_competitor_text_score: float
    target_versus_competitor_text_margin: float
    reference_pair_count: int
    prompt_reference_spearman: float | None
    candidate_evidence_fingerprint: str
    item_result_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptMetricSlice:
    slice_kind: str
    slice_value: str
    item_count: int
    candidate_count: int
    mean_target_prompt_rank: float
    mean_target_species_rank: float
    species_recall_at_k: tuple[tuple[int, float], ...]
    mean_target_versus_competitor_text_margin: float
    reference_pair_count: int
    prompt_reference_spearman: float | None
    mean_item_prompt_reference_spearman: float | None
    slice_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptConfigurationSummary:
    dataset_split: str
    split_fingerprint: str
    configuration: PromptEvaluationConfiguration
    overall: PromptMetricSlice
    subgroups: tuple[PromptMetricSlice, ...]
    summary_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptAblationEffect:
    dataset_split: str
    effect_kind: str
    baseline_configuration_id: str
    baseline_configuration_fingerprint: str
    treatment_configuration_id: str
    treatment_configuration_fingerprint: str
    paired_item_count: int
    delta_mean_target_prompt_rank: float
    delta_mean_target_species_rank: float
    delta_species_recall_at_k: tuple[tuple[int, float], ...]
    delta_mean_target_versus_competitor_text_margin: float
    delta_prompt_reference_spearman: float | None
    effect_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptEvaluationReport:
    schema_version: str
    evaluation_version: str
    recall_ks: tuple[int, ...]
    required_effects: tuple[str, ...]
    split_fingerprint: str
    configurations: tuple[PromptEvaluationConfiguration, ...]
    item_results: tuple[PromptEvaluationItemResult, ...]
    summaries: tuple[PromptConfigurationSummary, ...]
    effects: tuple[PromptAblationEffect, ...]
    input_fingerprint: str
    report_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptVersionSelection:
    schema_version: str
    selection_version: str
    selection_partition: str
    split_fingerprint: str
    primary_recall_k: int
    selected_configuration_id: str
    selected_configuration_fingerprint: str
    selected_prompt_version: str
    selected_species_recall_at_k: float
    selected_mean_target_versus_competitor_text_margin: float
    selected_mean_target_species_rank: float
    selected_mean_target_prompt_rank: float
    excluded_geography_configuration_count: int
    selection_input_fingerprint: str
    selection_fingerprint: str


@dataclass(frozen=True, slots=True)
class _EvaluatedItem:
    candidates: tuple[PromptCandidateEvaluation, ...]
    result: PromptEvaluationItemResult


def evaluate_taxonomic_prompt_ensembles(
    candidates: Sequence[PromptCandidateEvaluation],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    required_effects: Sequence[str] = (
        COMMON_NAME_EFFECT,
        TAXONOMIC_PATH_EFFECT,
    ),
) -> PromptEvaluationReport:
    """Evaluate balanced prompt configurations without crossing split identities."""

    ks = _recall_ks(recall_ks)
    effects_required = _required_effects(required_effects)
    rows = tuple(candidates)
    if not rows:
        raise ValueError("prompt evaluation candidates must not be empty")
    if not all(isinstance(row, PromptCandidateEvaluation) for row in rows):
        raise TypeError("candidates must contain PromptCandidateEvaluation values")
    split_fingerprints = {row.split_fingerprint for row in rows}
    if len(split_fingerprints) != 1:
        raise ValueError("prompt evaluation rows use multiple split fingerprints")
    split_fingerprint = next(iter(split_fingerprints))
    configurations = _configuration_index(rows)
    if len({value.model_fingerprint for value in configurations.values()}) != 1:
        raise ValueError("prompt evaluation configurations must use one model")
    grouped: dict[
        tuple[str, str, str],
        list[PromptCandidateEvaluation],
    ] = defaultdict(list)
    duplicate_keys: set[tuple[str, str, str, str]] = set()
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row.dataset_split,
            row.configuration.configuration_id,
            row.item_id,
            row.candidate_accepted_taxon_key,
        )
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
        grouped[key[:3]].append(row)
    if duplicate_keys:
        raise ValueError("prompt evaluation contains duplicate candidate rows")

    evaluated = tuple(_evaluate_item(tuple(grouped[key])) for key in sorted(grouped))
    _validate_balanced_evaluation(evaluated, configurations)
    _validate_controlled_effect_pairs(rows, configurations)
    _validate_configuration_feature_coverage(rows, configurations)
    summaries = _configuration_summaries(
        evaluated,
        configurations,
        split_fingerprint=split_fingerprint,
        recall_ks=ks,
    )
    effects = _ablation_effects(summaries)
    _validate_required_effects(effects, summaries, effects_required)
    ordered_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.dataset_split,
                row.configuration.configuration_id,
                row.item_id,
                row.candidate_accepted_taxon_key,
            ),
        )
    )
    input_fingerprint = canonical_semantic_fingerprint(
        {
            "evaluation_version": PROMPT_EVALUATION_VERSION,
            "split_fingerprint": split_fingerprint,
            "candidate_rows": [_candidate_semantics(row) for row in ordered_rows],
        }
    )
    report = PromptEvaluationReport(
        schema_version=PROMPT_EVALUATION_REPORT_SCHEMA_VERSION,
        evaluation_version=PROMPT_EVALUATION_VERSION,
        recall_ks=ks,
        required_effects=effects_required,
        split_fingerprint=split_fingerprint,
        configurations=tuple(configurations[key] for key in sorted(configurations)),
        item_results=tuple(item.result for item in evaluated),
        summaries=summaries,
        effects=effects,
        input_fingerprint=input_fingerprint,
        report_fingerprint="",
    )
    fingerprint = canonical_semantic_fingerprint(_report_semantics(report))
    return PromptEvaluationReport(
        **{
            field: getattr(report, field)
            for field in PromptEvaluationReport.__dataclass_fields__
            if field != "report_fingerprint"
        },
        report_fingerprint=fingerprint,
    )


def select_prompt_version(
    report: PromptEvaluationReport,
    *,
    selection_partition: str = "model_selection",
    primary_recall_k: int = 5,
) -> PromptVersionSelection:
    """Select one non-geographic prompt configuration from validation only."""

    prompt_evaluation_report_payload(report)
    if selection_partition != "model_selection":
        raise ValueError("prompt selection must use model_selection only")
    if isinstance(primary_recall_k, bool) or not isinstance(primary_recall_k, int):
        raise TypeError("primary_recall_k must be an integer")
    if primary_recall_k not in report.recall_ks:
        raise ValueError("primary_recall_k is absent from the evaluation report")
    partition_summaries = tuple(
        summary
        for summary in report.summaries
        if summary.dataset_split == selection_partition
    )
    if not partition_summaries:
        raise ValueError("evaluation report has no model_selection summaries")
    eligible = tuple(
        summary
        for summary in partition_summaries
        if not summary.configuration.geography_prompt_ablation_enabled
    )
    excluded_count = len(partition_summaries) - len(eligible)
    if not eligible:
        raise ValueError("no non-geographic prompt configuration is selectable")
    selected = min(
        eligible,
        key=lambda summary: (
            -_recall_value(summary.overall, primary_recall_k),
            -summary.overall.mean_target_versus_competitor_text_margin,
            summary.overall.mean_target_species_rank,
            summary.overall.mean_target_prompt_rank,
            summary.configuration.configuration_fingerprint,
        ),
    )
    selection_input_fingerprint = canonical_semantic_fingerprint(
        {
            "selection_partition": selection_partition,
            "split_fingerprint": report.split_fingerprint,
            "primary_recall_k": primary_recall_k,
            "eligible_validation_summaries": [
                _summary_payload(summary)
                for summary in sorted(
                    eligible,
                    key=lambda value: value.configuration.configuration_id,
                )
            ],
        }
    )
    values: dict[str, object] = {
        "schema_version": PROMPT_VERSION_SELECTION_SCHEMA_VERSION,
        "selection_version": PROMPT_VERSION_SELECTION_VERSION,
        "selection_partition": selection_partition,
        "split_fingerprint": report.split_fingerprint,
        "primary_recall_k": primary_recall_k,
        "selected_configuration_id": selected.configuration.configuration_id,
        "selected_configuration_fingerprint": (
            selected.configuration.configuration_fingerprint
        ),
        "selected_prompt_version": selected.configuration.prompt_version,
        "selected_species_recall_at_k": _recall_value(
            selected.overall,
            primary_recall_k,
        ),
        "selected_mean_target_versus_competitor_text_margin": (
            selected.overall.mean_target_versus_competitor_text_margin
        ),
        "selected_mean_target_species_rank": (
            selected.overall.mean_target_species_rank
        ),
        "selected_mean_target_prompt_rank": (selected.overall.mean_target_prompt_rank),
        "excluded_geography_configuration_count": excluded_count,
        "selection_input_fingerprint": selection_input_fingerprint,
    }
    fingerprint = canonical_semantic_fingerprint(values)
    return PromptVersionSelection(
        **values,
        selection_fingerprint=fingerprint,
    )


def prompt_evaluation_report_payload(
    report: PromptEvaluationReport,
) -> dict[str, object]:
    """Validate and serialize a complete prompt evaluation report."""

    if not isinstance(report, PromptEvaluationReport):
        raise TypeError("report must be a PromptEvaluationReport")
    if (
        report.schema_version != PROMPT_EVALUATION_REPORT_SCHEMA_VERSION
        or report.evaluation_version != PROMPT_EVALUATION_VERSION
    ):
        raise ValueError("prompt evaluation report version is incompatible")
    _recall_ks(report.recall_ks)
    _required_effects(report.required_effects)
    _sha256(report.split_fingerprint, field="split_fingerprint")
    _sha256(report.input_fingerprint, field="input_fingerprint")
    semantics = _report_semantics(report)
    fingerprint = _sha256(
        report.report_fingerprint,
        field="report_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt evaluation report fingerprint is inconsistent")
    return {**semantics, "report_fingerprint": fingerprint}


def prompt_version_selection_payload(
    selection: PromptVersionSelection,
) -> dict[str, object]:
    """Validate and serialize a validation-only prompt selection."""

    if not isinstance(selection, PromptVersionSelection):
        raise TypeError("selection must be a PromptVersionSelection")
    if (
        selection.schema_version != PROMPT_VERSION_SELECTION_SCHEMA_VERSION
        or selection.selection_version != PROMPT_VERSION_SELECTION_VERSION
        or selection.selection_partition != "model_selection"
    ):
        raise ValueError("prompt version selection contract is incompatible")
    values = {
        field: getattr(selection, field)
        for field in PromptVersionSelection.__dataclass_fields__
        if field != "selection_fingerprint"
    }
    _sha256(selection.split_fingerprint, field="split_fingerprint")
    _sha256(
        selection.selected_configuration_fingerprint,
        field="selected_configuration_fingerprint",
    )
    _sha256(
        selection.selection_input_fingerprint,
        field="selection_input_fingerprint",
    )
    fingerprint = _sha256(
        selection.selection_fingerprint,
        field="selection_fingerprint",
    )
    if canonical_semantic_fingerprint(values) != fingerprint:
        raise ValueError("prompt version selection fingerprint is inconsistent")
    return {**values, "selection_fingerprint": fingerprint}


def _evaluate_item(
    rows: tuple[PromptCandidateEvaluation, ...],
) -> _EvaluatedItem:
    candidates = tuple(sorted(rows, key=lambda row: row.candidate_accepted_taxon_key))
    first = candidates[0]
    if len(candidates) < 2:
        raise ValueError("prompt evaluation item requires at least two candidates")
    identity = (
        first.dataset_split,
        first.configuration,
        first.item_id,
        first.expected_accepted_taxon_key,
        first.route,
        first.life_stage,
        first.visual_domain,
        first.candidate_set_fingerprint,
    )
    if any(
        (
            row.dataset_split,
            row.configuration,
            row.item_id,
            row.expected_accepted_taxon_key,
            row.route,
            row.life_stage,
            row.visual_domain,
            row.candidate_set_fingerprint,
        )
        != identity
        for row in candidates
    ):
        raise ValueError("prompt evaluation candidate identity is inconsistent")
    candidate_keys = tuple(row.candidate_accepted_taxon_key for row in candidates)
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("prompt evaluation item contains duplicate candidate taxa")
    if candidate_keys.count(first.expected_accepted_taxon_key) != 1:
        raise ValueError("prompt evaluation target must appear exactly once")

    prompt_rows = sorted(
        (
            (
                row.candidate_accepted_taxon_key,
                diagnostic,
            )
            for row in candidates
            for diagnostic in row.pooling_result.prompt_scores
            if diagnostic.in_subset
        ),
        key=lambda value: (
            -value[1].raw_similarity,
            value[0],
            value[1].variant_fingerprint,
        ),
    )
    target_prompt_index = next(
        index
        for index, (candidate_key, _) in enumerate(prompt_rows)
        if candidate_key == first.expected_accepted_taxon_key
    )
    target_prompt = prompt_rows[target_prompt_index][1]
    species_rows = tuple(
        sorted(
            candidates,
            key=lambda row: (
                -row.pooling_result.pooled_score,
                row.candidate_accepted_taxon_key,
            ),
        )
    )
    target_species_index = next(
        index
        for index, row in enumerate(species_rows)
        if row.candidate_accepted_taxon_key == first.expected_accepted_taxon_key
    )
    target = species_rows[target_species_index]
    competitor = next(
        row
        for row in species_rows
        if row.candidate_accepted_taxon_key != first.expected_accepted_taxon_key
    )
    reference_pairs = tuple(
        (
            row.pooling_result.pooled_score,
            row.reference_image_score,
        )
        for row in candidates
        if row.reference_image_score is not None
    )
    reference_spearman = _spearman(
        tuple(pair[0] for pair in reference_pairs),
        tuple(float(pair[1]) for pair in reference_pairs),
    )
    evidence_fingerprint = canonical_semantic_fingerprint(
        {
            "candidate_set_fingerprint": first.candidate_set_fingerprint,
            "candidate_rows": [_candidate_semantics(row) for row in candidates],
        }
    )
    values: dict[str, object] = {
        "dataset_split": first.dataset_split,
        "configuration_id": first.configuration.configuration_id,
        "configuration_fingerprint": (first.configuration.configuration_fingerprint),
        "item_id": first.item_id,
        "route": first.route,
        "life_stage": first.life_stage,
        "visual_domain": first.visual_domain,
        "expected_accepted_taxon_key": first.expected_accepted_taxon_key,
        "candidate_set_fingerprint": first.candidate_set_fingerprint,
        "candidate_count": len(candidates),
        "target_prompt_rank": target_prompt_index + 1,
        "best_target_prompt_label": target_prompt.label,
        "best_target_prompt_variant_fingerprint": (target_prompt.variant_fingerprint),
        "best_target_prompt_score": target_prompt.raw_similarity,
        "target_species_rank": target_species_index + 1,
        "target_text_score": target.pooling_result.pooled_score,
        "best_competitor_accepted_taxon_key": (competitor.candidate_accepted_taxon_key),
        "best_competitor_text_score": competitor.pooling_result.pooled_score,
        "target_versus_competitor_text_margin": (
            target.pooling_result.pooled_score - competitor.pooling_result.pooled_score
        ),
        "reference_pair_count": len(reference_pairs),
        "prompt_reference_spearman": reference_spearman,
        "candidate_evidence_fingerprint": evidence_fingerprint,
    }
    result = PromptEvaluationItemResult(
        **values,
        item_result_fingerprint=canonical_semantic_fingerprint(values),
    )
    return _EvaluatedItem(candidates=candidates, result=result)


def _validate_balanced_evaluation(
    evaluated: Sequence[_EvaluatedItem],
    configurations: Mapping[str, PromptEvaluationConfiguration],
) -> None:
    by_split_config: dict[tuple[str, str], set[str]] = defaultdict(set)
    identity_by_split_item: dict[tuple[str, str], tuple[object, ...]] = {}
    references_by_split_item: dict[
        tuple[str, str],
        tuple[tuple[str, float | None, str | None], ...],
    ] = {}
    for item in evaluated:
        result = item.result
        by_split_config[(result.dataset_split, result.configuration_id)].add(
            result.item_id
        )
        key = (result.dataset_split, result.item_id)
        identity = (
            result.expected_accepted_taxon_key,
            result.route,
            result.life_stage,
            result.visual_domain,
            result.candidate_set_fingerprint,
            tuple(row.candidate_accepted_taxon_key for row in item.candidates),
        )
        previous_identity = identity_by_split_item.setdefault(key, identity)
        if previous_identity != identity:
            raise ValueError("candidate sets differ across prompt configurations")
        references = tuple(
            (
                row.candidate_accepted_taxon_key,
                row.reference_image_score,
                row.reference_evidence_fingerprint,
            )
            for row in item.candidates
        )
        previous_references = references_by_split_item.setdefault(key, references)
        if previous_references != references:
            raise ValueError(
                "reference-image evidence differs across prompt configurations"
            )
    split_names = sorted({key[0] for key in by_split_config})
    for split_name in split_names:
        item_sets = {
            configuration_id: by_split_config.get((split_name, configuration_id))
            for configuration_id in configurations
        }
        if any(value is None for value in item_sets.values()):
            raise ValueError("prompt configurations are missing from a dataset split")
        unique_item_sets = {frozenset(value or ()) for value in item_sets.values()}
        if len(unique_item_sets) != 1:
            raise ValueError("prompt configurations do not evaluate identical items")


def _validate_configuration_feature_coverage(
    rows: Sequence[PromptCandidateEvaluation],
    configurations: Mapping[str, PromptEvaluationConfiguration],
) -> None:
    selected_common: dict[str, bool] = defaultdict(bool)
    selected_path: dict[str, bool] = defaultdict(bool)
    for row in rows:
        identifier = row.configuration.configuration_id
        selected = tuple(
            item for item in row.pooling_result.prompt_scores if item.in_subset
        )
        selected_common[identifier] = selected_common[identifier] or any(
            item.evidence_kind == "vernacular_name" for item in selected
        )
        selected_path[identifier] = selected_path[identifier] or any(
            item.prompt_kind == "accepted_taxonomic_path" for item in selected
        )
    for identifier, configuration in configurations.items():
        if configuration.common_names_enabled and not selected_common[identifier]:
            raise ValueError(
                "common-name configuration selected no common-name prompts"
            )
        if configuration.taxonomic_path_enabled != selected_path[identifier]:
            raise ValueError(
                "taxonomic-path configuration has inconsistent prompt coverage"
            )


def _validate_controlled_effect_pairs(
    rows: Sequence[PromptCandidateEvaluation],
    configurations: Mapping[str, PromptEvaluationConfiguration],
) -> None:
    by_key = {
        (
            row.dataset_split,
            row.item_id,
            row.candidate_accepted_taxon_key,
            row.configuration.configuration_id,
        ): row
        for row in rows
    }
    split_items = sorted(
        {
            (
                row.dataset_split,
                row.item_id,
                row.candidate_accepted_taxon_key,
            )
            for row in rows
        }
    )
    for effect_kind in (COMMON_NAME_EFFECT, TAXONOMIC_PATH_EFFECT):
        for baseline in configurations.values():
            for treatment in configurations.values():
                if not _is_effect_pair(
                    baseline,
                    treatment,
                    effect_kind=effect_kind,
                ):
                    continue
                for dataset_split, item_id, candidate_key in split_items:
                    baseline_row = by_key[
                        (
                            dataset_split,
                            item_id,
                            candidate_key,
                            baseline.configuration_id,
                        )
                    ]
                    treatment_row = by_key[
                        (
                            dataset_split,
                            item_id,
                            candidate_key,
                            treatment.configuration_id,
                        )
                    ]
                    if (
                        baseline_row.pooling_result.ensemble_fingerprint
                        != treatment_row.pooling_result.ensemble_fingerprint
                    ):
                        raise ValueError(
                            "prompt ablation pair uses different taxonomic ensembles"
                        )
                    baseline_prompts = {
                        value.variant_fingerprint: value
                        for value in baseline_row.pooling_result.prompt_scores
                        if value.in_subset
                    }
                    treatment_prompts = {
                        value.variant_fingerprint: value
                        for value in treatment_row.pooling_result.prompt_scores
                        if value.in_subset
                    }
                    removed = set(baseline_prompts) - set(treatment_prompts)
                    if removed:
                        raise ValueError(
                            "prompt ablation treatment removes baseline prompts"
                        )
                    for fingerprint, baseline_prompt in baseline_prompts.items():
                        treatment_prompt = treatment_prompts[fingerprint]
                        if (
                            abs(
                                baseline_prompt.raw_similarity
                                - treatment_prompt.raw_similarity
                            )
                            > 1e-12
                        ):
                            raise ValueError(
                                "shared prompt score changes across ablation pair"
                            )
                    added = tuple(
                        treatment_prompts[fingerprint]
                        for fingerprint in sorted(
                            set(treatment_prompts) - set(baseline_prompts)
                        )
                    )
                    if effect_kind == TAXONOMIC_PATH_EFFECT and (
                        not added
                        or any(
                            value.prompt_kind != "accepted_taxonomic_path"
                            for value in added
                        )
                    ):
                        raise ValueError(
                            "taxonomic-path ablation changes another prompt family"
                        )
                    if effect_kind == COMMON_NAME_EFFECT and any(
                        value.evidence_kind != "vernacular_name" for value in added
                    ):
                        raise ValueError(
                            "common-name ablation changes another prompt family"
                        )


def _configuration_summaries(
    evaluated: Sequence[_EvaluatedItem],
    configurations: Mapping[str, PromptEvaluationConfiguration],
    *,
    split_fingerprint: str,
    recall_ks: tuple[int, ...],
) -> tuple[PromptConfigurationSummary, ...]:
    grouped: dict[tuple[str, str], list[_EvaluatedItem]] = defaultdict(list)
    for item in evaluated:
        grouped[(item.result.dataset_split, item.result.configuration_id)].append(item)
    summaries: list[PromptConfigurationSummary] = []
    for (dataset_split, configuration_id), items in sorted(grouped.items()):
        configuration = configurations[configuration_id]
        ordered = tuple(sorted(items, key=lambda item: item.result.item_id))
        overall = _metric_slice(
            ordered,
            slice_kind="overall",
            slice_value="all",
            recall_ks=recall_ks,
        )
        subgroup_rows: list[PromptMetricSlice] = []
        dimensions = (
            ("life_stage", lambda item: item.result.life_stage),
            ("visual_domain", lambda item: item.result.visual_domain),
            (
                "life_stage_visual_domain",
                lambda item: f"{item.result.life_stage}|{item.result.visual_domain}",
            ),
        )
        for dimension, key_fn in dimensions:
            dimension_groups: dict[str, list[_EvaluatedItem]] = defaultdict(list)
            for item in ordered:
                dimension_groups[key_fn(item)].append(item)
            for value, group in sorted(dimension_groups.items()):
                subgroup_rows.append(
                    _metric_slice(
                        tuple(group),
                        slice_kind=dimension,
                        slice_value=value,
                        recall_ks=recall_ks,
                    )
                )
        subgroups = tuple(
            sorted(
                subgroup_rows,
                key=lambda value: (value.slice_kind, value.slice_value),
            )
        )
        values = {
            "dataset_split": dataset_split,
            "split_fingerprint": split_fingerprint,
            "configuration": _configuration_payload(configuration),
            "overall": _metric_slice_payload(overall),
            "subgroups": [_metric_slice_payload(value) for value in subgroups],
        }
        summaries.append(
            PromptConfigurationSummary(
                dataset_split=dataset_split,
                split_fingerprint=split_fingerprint,
                configuration=configuration,
                overall=overall,
                subgroups=subgroups,
                summary_fingerprint=canonical_semantic_fingerprint(values),
            )
        )
    return tuple(summaries)


def _metric_slice(
    items: Sequence[_EvaluatedItem],
    *,
    slice_kind: str,
    slice_value: str,
    recall_ks: tuple[int, ...],
) -> PromptMetricSlice:
    if not items:
        raise ValueError("prompt metric slice must not be empty")
    results = tuple(item.result for item in items)
    reference_pairs = tuple(
        (row.pooling_result.pooled_score, row.reference_image_score)
        for item in items
        for row in item.candidates
        if row.reference_image_score is not None
    )
    item_correlations = tuple(
        result.prompt_reference_spearman
        for result in results
        if result.prompt_reference_spearman is not None
    )
    values: dict[str, object] = {
        "slice_kind": slice_kind,
        "slice_value": slice_value,
        "item_count": len(results),
        "candidate_count": sum(result.candidate_count for result in results),
        "mean_target_prompt_rank": _mean(
            tuple(float(result.target_prompt_rank) for result in results)
        ),
        "mean_target_species_rank": _mean(
            tuple(float(result.target_species_rank) for result in results)
        ),
        "species_recall_at_k": tuple(
            (
                k,
                sum(result.target_species_rank <= k for result in results)
                / len(results),
            )
            for k in recall_ks
        ),
        "mean_target_versus_competitor_text_margin": _mean(
            tuple(result.target_versus_competitor_text_margin for result in results)
        ),
        "reference_pair_count": len(reference_pairs),
        "prompt_reference_spearman": _spearman(
            tuple(pair[0] for pair in reference_pairs),
            tuple(float(pair[1]) for pair in reference_pairs),
        ),
        "mean_item_prompt_reference_spearman": (
            _mean(tuple(float(value) for value in item_correlations))
            if item_correlations
            else None
        ),
    }
    return PromptMetricSlice(
        **values,
        slice_fingerprint=canonical_semantic_fingerprint(
            _metric_slice_semantics(values)
        ),
    )


def _ablation_effects(
    summaries: Sequence[PromptConfigurationSummary],
) -> tuple[PromptAblationEffect, ...]:
    by_split: dict[str, list[PromptConfigurationSummary]] = defaultdict(list)
    for summary in summaries:
        by_split[summary.dataset_split].append(summary)
    effects: list[PromptAblationEffect] = []
    for dataset_split, split_summaries in sorted(by_split.items()):
        for effect_kind in (COMMON_NAME_EFFECT, TAXONOMIC_PATH_EFFECT):
            for baseline in split_summaries:
                for treatment in split_summaries:
                    if not _is_effect_pair(
                        baseline.configuration,
                        treatment.configuration,
                        effect_kind=effect_kind,
                    ):
                        continue
                    baseline_overall = baseline.overall
                    treatment_overall = treatment.overall
                    values: dict[str, object] = {
                        "dataset_split": dataset_split,
                        "effect_kind": effect_kind,
                        "baseline_configuration_id": (
                            baseline.configuration.configuration_id
                        ),
                        "baseline_configuration_fingerprint": (
                            baseline.configuration.configuration_fingerprint
                        ),
                        "treatment_configuration_id": (
                            treatment.configuration.configuration_id
                        ),
                        "treatment_configuration_fingerprint": (
                            treatment.configuration.configuration_fingerprint
                        ),
                        "paired_item_count": baseline_overall.item_count,
                        "delta_mean_target_prompt_rank": (
                            treatment_overall.mean_target_prompt_rank
                            - baseline_overall.mean_target_prompt_rank
                        ),
                        "delta_mean_target_species_rank": (
                            treatment_overall.mean_target_species_rank
                            - baseline_overall.mean_target_species_rank
                        ),
                        "delta_species_recall_at_k": tuple(
                            (
                                k,
                                _recall_value(treatment_overall, k)
                                - _recall_value(baseline_overall, k),
                            )
                            for k, _ in baseline_overall.species_recall_at_k
                        ),
                        "delta_mean_target_versus_competitor_text_margin": (
                            treatment_overall.mean_target_versus_competitor_text_margin
                            - baseline_overall.mean_target_versus_competitor_text_margin
                        ),
                        "delta_prompt_reference_spearman": _optional_delta(
                            treatment_overall.prompt_reference_spearman,
                            baseline_overall.prompt_reference_spearman,
                        ),
                    }
                    effects.append(
                        PromptAblationEffect(
                            **values,
                            effect_fingerprint=canonical_semantic_fingerprint(
                                _effect_semantics(values)
                            ),
                        )
                    )
    return tuple(
        sorted(
            effects,
            key=lambda value: (
                value.dataset_split,
                value.effect_kind,
                value.baseline_configuration_id,
                value.treatment_configuration_id,
            ),
        )
    )


def _is_effect_pair(
    baseline: PromptEvaluationConfiguration,
    treatment: PromptEvaluationConfiguration,
    *,
    effect_kind: str,
) -> bool:
    shared = (
        baseline.prompt_version == treatment.prompt_version
        and baseline.pooling_strategy == treatment.pooling_strategy
        and baseline.model_fingerprint == treatment.model_fingerprint
        and baseline.geography_prompt_ablation_enabled
        == treatment.geography_prompt_ablation_enabled
    )
    if not shared:
        return False
    if effect_kind == COMMON_NAME_EFFECT:
        return (
            not baseline.common_names_enabled
            and treatment.common_names_enabled
            and baseline.taxonomic_path_enabled == treatment.taxonomic_path_enabled
        )
    return (
        not baseline.taxonomic_path_enabled
        and treatment.taxonomic_path_enabled
        and baseline.common_names_enabled == treatment.common_names_enabled
    )


def _validate_required_effects(
    effects: Sequence[PromptAblationEffect],
    summaries: Sequence[PromptConfigurationSummary],
    required_effects: tuple[str, ...],
) -> None:
    for dataset_split in {summary.dataset_split for summary in summaries}:
        available = {
            effect.effect_kind
            for effect in effects
            if effect.dataset_split == dataset_split
        }
        missing = set(required_effects) - available
        if missing:
            raise ValueError(
                f"prompt evaluation lacks required effects: {sorted(missing)}"
            )


def _configuration_index(
    rows: Sequence[PromptCandidateEvaluation],
) -> dict[str, PromptEvaluationConfiguration]:
    result: dict[str, PromptEvaluationConfiguration] = {}
    for row in rows:
        configuration = row.configuration
        previous = result.setdefault(configuration.configuration_id, configuration)
        if previous != configuration:
            raise ValueError("configuration ID identifies conflicting prompt policies")
    return result


def _configuration_semantics(
    configuration: PromptEvaluationConfiguration,
) -> dict[str, object]:
    return {
        "schema_version": configuration.schema_version,
        "configuration_id": configuration.configuration_id,
        "prompt_version": configuration.prompt_version,
        "pooling_strategy": configuration.pooling_strategy,
        "model_fingerprint": configuration.model_fingerprint,
        "common_names_enabled": configuration.common_names_enabled,
        "taxonomic_path_enabled": configuration.taxonomic_path_enabled,
        "geography_prompt_ablation_enabled": (
            configuration.geography_prompt_ablation_enabled
        ),
    }


def _configuration_payload(
    configuration: PromptEvaluationConfiguration,
) -> dict[str, object]:
    semantics = _configuration_semantics(configuration)
    fingerprint = _sha256(
        configuration.configuration_fingerprint,
        field="configuration_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt configuration fingerprint is inconsistent")
    return {**semantics, "configuration_fingerprint": fingerprint}


def _candidate_semantics(row: PromptCandidateEvaluation) -> dict[str, object]:
    return {
        "item_id": row.item_id,
        "dataset_split": row.dataset_split,
        "split_fingerprint": row.split_fingerprint,
        "expected_accepted_taxon_key": row.expected_accepted_taxon_key,
        "candidate_accepted_taxon_key": row.candidate_accepted_taxon_key,
        "route": row.route,
        "life_stage": row.life_stage,
        "visual_domain": row.visual_domain,
        "candidate_set_fingerprint": row.candidate_set_fingerprint,
        "configuration_fingerprint": row.configuration.configuration_fingerprint,
        "pooling_result_fingerprint": row.pooling_result.result_fingerprint,
        "reference_score_kind": row.reference_score_kind,
        "reference_image_score": row.reference_image_score,
        "reference_evidence_fingerprint": row.reference_evidence_fingerprint,
    }


def _item_result_semantics(
    result: PromptEvaluationItemResult,
) -> dict[str, object]:
    return {
        field: getattr(result, field)
        for field in PromptEvaluationItemResult.__dataclass_fields__
        if field != "item_result_fingerprint"
    }


def _item_result_payload(
    result: PromptEvaluationItemResult,
) -> dict[str, object]:
    semantics = _item_result_semantics(result)
    fingerprint = _sha256(
        result.item_result_fingerprint,
        field="item_result_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt evaluation item fingerprint is inconsistent")
    return {**semantics, "item_result_fingerprint": fingerprint}


def _metric_slice_semantics(values: Mapping[str, object]) -> dict[str, object]:
    recalls = values["species_recall_at_k"]
    if not isinstance(recalls, tuple):
        raise ValueError("species_recall_at_k must be canonical")
    return {
        "slice_kind": values["slice_kind"],
        "slice_value": values["slice_value"],
        "item_count": values["item_count"],
        "candidate_count": values["candidate_count"],
        "mean_target_prompt_rank": values["mean_target_prompt_rank"],
        "mean_target_species_rank": values["mean_target_species_rank"],
        "species_recall_at_k": [{"k": k, "recall": recall} for k, recall in recalls],
        "mean_target_versus_competitor_text_margin": values[
            "mean_target_versus_competitor_text_margin"
        ],
        "reference_pair_count": values["reference_pair_count"],
        "prompt_reference_spearman": values["prompt_reference_spearman"],
        "mean_item_prompt_reference_spearman": values[
            "mean_item_prompt_reference_spearman"
        ],
    }


def _metric_slice_payload(value: PromptMetricSlice) -> dict[str, object]:
    values = {
        field: getattr(value, field)
        for field in PromptMetricSlice.__dataclass_fields__
        if field != "slice_fingerprint"
    }
    semantics = _metric_slice_semantics(values)
    fingerprint = _sha256(value.slice_fingerprint, field="slice_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt metric slice fingerprint is inconsistent")
    return {**semantics, "slice_fingerprint": fingerprint}


def _summary_payload(summary: PromptConfigurationSummary) -> dict[str, object]:
    semantics = {
        "dataset_split": summary.dataset_split,
        "split_fingerprint": summary.split_fingerprint,
        "configuration": _configuration_payload(summary.configuration),
        "overall": _metric_slice_payload(summary.overall),
        "subgroups": [_metric_slice_payload(value) for value in summary.subgroups],
    }
    fingerprint = _sha256(
        summary.summary_fingerprint,
        field="summary_fingerprint",
    )
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt configuration summary fingerprint is inconsistent")
    return {**semantics, "summary_fingerprint": fingerprint}


def _effect_semantics(values: Mapping[str, object]) -> dict[str, object]:
    recalls = values["delta_species_recall_at_k"]
    if not isinstance(recalls, tuple):
        raise ValueError("delta_species_recall_at_k must be canonical")
    return {
        "dataset_split": values["dataset_split"],
        "effect_kind": values["effect_kind"],
        "baseline_configuration_id": values["baseline_configuration_id"],
        "baseline_configuration_fingerprint": values[
            "baseline_configuration_fingerprint"
        ],
        "treatment_configuration_id": values["treatment_configuration_id"],
        "treatment_configuration_fingerprint": values[
            "treatment_configuration_fingerprint"
        ],
        "paired_item_count": values["paired_item_count"],
        "delta_mean_target_prompt_rank": values["delta_mean_target_prompt_rank"],
        "delta_mean_target_species_rank": values["delta_mean_target_species_rank"],
        "delta_species_recall_at_k": [{"k": k, "delta": delta} for k, delta in recalls],
        "delta_mean_target_versus_competitor_text_margin": values[
            "delta_mean_target_versus_competitor_text_margin"
        ],
        "delta_prompt_reference_spearman": values["delta_prompt_reference_spearman"],
    }


def _effect_payload(effect: PromptAblationEffect) -> dict[str, object]:
    values = {
        field: getattr(effect, field)
        for field in PromptAblationEffect.__dataclass_fields__
        if field != "effect_fingerprint"
    }
    semantics = _effect_semantics(values)
    fingerprint = _sha256(effect.effect_fingerprint, field="effect_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("prompt ablation effect fingerprint is inconsistent")
    return {**semantics, "effect_fingerprint": fingerprint}


def _report_semantics(report: PromptEvaluationReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "evaluation_version": report.evaluation_version,
        "recall_ks": list(report.recall_ks),
        "required_effects": list(report.required_effects),
        "split_fingerprint": report.split_fingerprint,
        "configurations": [
            _configuration_payload(value) for value in report.configurations
        ],
        "item_results": [_item_result_payload(value) for value in report.item_results],
        "summaries": [_summary_payload(value) for value in report.summaries],
        "effects": [_effect_payload(value) for value in report.effects],
        "input_fingerprint": report.input_fingerprint,
    }


def _recall_ks(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in result
        )
        or any(value <= 0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError("recall_ks must be sorted unique positive integers")
    return result


def _required_effects(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_required_text(value, field="required_effect") for value in values)
    if len(result) != len(set(result)) or not set(result) <= PROMPT_EFFECT_KINDS:
        raise ValueError("required_effects contains unsupported or duplicate values")
    return tuple(sorted(result))


def _recall_value(value: PromptMetricSlice, k: int) -> float:
    try:
        return dict(value.species_recall_at_k)[k]
    except KeyError as exc:
        raise ValueError(f"prompt metric slice has no recall@{k}") from exc


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _spearman(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    if len(first) != len(second):
        raise ValueError("correlation vectors must have equal length")
    if len(first) < 2:
        return None
    first_values = tuple(
        _finite_number(value, field="correlation value") for value in first
    )
    second_values = tuple(
        _finite_number(value, field="correlation value") for value in second
    )
    first_ranks = _average_ranks(first_values)
    second_ranks = _average_ranks(second_values)
    first_mean = _mean(first_ranks)
    second_mean = _mean(second_ranks)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_ranks, second_ranks, strict=True)
    )
    first_scale = sum((value - first_mean) ** 2 for value in first_ranks)
    second_scale = sum((value - second_mean) ** 2 for value in second_ranks)
    if first_scale <= 0.0 or second_scale <= 0.0:
        return None
    result = numerator / sqrt(first_scale * second_scale)
    return max(-1.0, min(1.0, result))


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return tuple(result)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    result = " ".join(value.split())
    if not result:
        raise ValueError(f"{field} must be non-empty text")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded_cosine(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < -1.0 or result > 1.0:
        raise ValueError(f"{field} must be in [-1, 1]")
    return result


def _require_boolean(value: object, *, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")


__all__ = [
    "COMMON_NAME_EFFECT",
    "DEFAULT_RECALL_KS",
    "PROMPT_EFFECT_KINDS",
    "PROMPT_EVALUATION_CONFIGURATION_SCHEMA_VERSION",
    "PROMPT_EVALUATION_REPORT_SCHEMA_VERSION",
    "PROMPT_EVALUATION_VERSION",
    "PROMPT_VERSION_SELECTION_SCHEMA_VERSION",
    "PROMPT_VERSION_SELECTION_VERSION",
    "REFERENCE_IMAGE_SCORE_KIND",
    "TAXONOMIC_PATH_EFFECT",
    "PromptAblationEffect",
    "PromptCandidateEvaluation",
    "PromptConfigurationSummary",
    "PromptEvaluationConfiguration",
    "PromptEvaluationItemResult",
    "PromptEvaluationReport",
    "PromptMetricSlice",
    "PromptVersionSelection",
    "evaluate_taxonomic_prompt_ensembles",
    "prompt_evaluation_report_payload",
    "prompt_version_selection_payload",
    "select_prompt_version",
]
