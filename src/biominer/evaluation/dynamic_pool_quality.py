"""Hierarchical, grouped and weighted quality audits for dynamic pooling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import NormalDist

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOL_QUALITY_POLICY_VERSION = "dynamic-pool-quality-policy-v1.0.0"
DYNAMIC_POOL_QUALITY_OBSERVATION_VERSION = "dynamic-pool-quality-observation-v1.0.0"
DYNAMIC_POOL_QUALITY_REPORT_VERSION = "dynamic-pool-quality-report-v1.0.0"
GROUPED_WEIGHTED_INTERVAL_METHOD = (
    "two_sided_wilson_min_kish_row_and_component_effective_n"
)
REPRESENTATIVE_AUDIT_PURPOSE = "representative_audit"
TARGETED_FAILURE_PURPOSE = "targeted_failure_discovery"
SAMPLING_PURPOSES = frozenset({REPRESENTATIVE_AUDIT_PURPOSE, TARGETED_FAILURE_PURPOSE})
GEOGRAPHY_QUALITY_LEVELS = (
    "availability",
    "country",
    "admin1",
    "bioregion",
    "geographic_cluster",
)

QUALITY_REPORT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "report_fingerprint": pl.String,
    "row_fingerprint": pl.String,
    "policy_fingerprint": pl.String,
    "hierarchy_level": pl.String,
    "geography_level": pl.String,
    "group_id": pl.String,
    "group_label": pl.String,
    "family_key": pl.String,
    "family_name": pl.String,
    "genus_key": pl.String,
    "genus_name": pl.String,
    "species_key": pl.String,
    "scientific_name": pl.String,
    "geography_value": pl.String,
    "no_geo": pl.Boolean,
    "source_item_count": pl.UInt32,
    "evaluated_item_count": pl.UInt32,
    "excluded_targeted_item_count": pl.UInt32,
    "independence_component_count": pl.UInt32,
    "weighted_item_count": pl.Float64,
    "group_effective_sample_size": pl.Float64,
    "group_status": pl.String,
    "group_insufficiency_reasons": pl.List(pl.String),
    "weighted_true_positive": pl.Float64,
    "weighted_true_negative": pl.Float64,
    "weighted_false_positive": pl.Float64,
    "weighted_false_negative": pl.Float64,
    "metric_name": pl.String,
    "metric_kind": pl.String,
    "metric_status": pl.String,
    "metric_insufficiency_reasons": pl.List(pl.String),
    "numerator_item_count": pl.UInt32,
    "denominator_item_count": pl.UInt32,
    "numerator_weight": pl.Float64,
    "denominator_weight": pl.Float64,
    "denominator_component_count": pl.UInt32,
    "metric_effective_sample_size": pl.Float64,
    "estimate": pl.Float64,
    "confidence_interval_lower": pl.Float64,
    "confidence_interval_upper": pl.Float64,
    "confidence_level": pl.Float64,
    "confidence_interval_method": pl.String,
    "descriptive_audit_only": pl.Boolean,
    "authorizes_occurrence_release": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolQualityPolicy:
    """Evidence floors shared by every hierarchy level."""

    schema_version: str = DYNAMIC_POOL_QUALITY_POLICY_VERSION
    confidence_level: float = 0.95
    minimum_group_items: int = 30
    minimum_group_components: int = 30
    minimum_group_effective_sample_size: float = 30.0
    minimum_metric_denominator_items: int = 10
    minimum_metric_denominator_components: int = 10
    minimum_metric_effective_sample_size: float = 10.0
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_QUALITY_POLICY_VERSION:
            raise ValueError("unsupported dynamic-pool quality policy version")
        confidence = _probability(self.confidence_level, field="confidence_level")
        if confidence in {0.0, 1.0}:
            raise ValueError("confidence_level must be in (0, 1)")
        object.__setattr__(self, "confidence_level", confidence)
        for field in (
            "minimum_group_items",
            "minimum_group_components",
            "minimum_metric_denominator_items",
            "minimum_metric_denominator_components",
            "calibration_bins",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in (
            "minimum_group_effective_sample_size",
            "minimum_metric_effective_sample_size",
        ):
            object.__setattr__(
                self,
                field,
                _positive_float(getattr(self, field), field=field),
            )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicPoolQualityObservation:
    """One decisive, source-bound review outcome for quality estimation."""

    item_id: str
    source_record_id: str
    source_image_sha256: str
    independence_component_id: str
    family_key: str
    family_name: str
    genus_key: str
    genus_name: str
    species_key: str
    scientific_name: str
    sampling_purpose: str
    representative_estimation_eligible: bool
    sampling_weight: float | None
    human_supported: bool
    screening_selected: bool
    model_abstained: bool
    family_routing_correct: bool
    global_local_disagreed: bool
    local_support_available: bool
    reference_outlier_influenced_error: bool
    out_of_distribution: bool
    calibrated_supported_probability: float | None = None
    country_code: str | None = None
    admin1: str | None = None
    bioregion: str | None = None
    geographic_cluster_id: str | None = None
    no_geo: bool = False
    schema_version: str = DYNAMIC_POOL_QUALITY_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_QUALITY_OBSERVATION_VERSION:
            raise ValueError("unsupported dynamic-pool quality observation version")
        for field in (
            "item_id",
            "source_record_id",
            "independence_component_id",
            "family_key",
            "family_name",
            "genus_key",
            "genus_name",
            "species_key",
            "scientific_name",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "source_image_sha256",
            _sha256(self.source_image_sha256, field="source_image_sha256"),
        )
        purpose = _required_text(self.sampling_purpose, field="sampling_purpose")
        if purpose not in SAMPLING_PURPOSES:
            raise ValueError(f"unsupported sampling_purpose: {purpose}")
        object.__setattr__(self, "sampling_purpose", purpose)
        for field in (
            "representative_estimation_eligible",
            "human_supported",
            "screening_selected",
            "model_abstained",
            "family_routing_correct",
            "global_local_disagreed",
            "local_support_available",
            "reference_outlier_influenced_error",
            "out_of_distribution",
            "no_geo",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")
        if self.representative_estimation_eligible != (
            purpose == REPRESENTATIVE_AUDIT_PURPOSE
        ):
            raise ValueError(
                "representative eligibility must match representative audit purpose"
            )
        weight = self.sampling_weight
        if weight is not None:
            weight = _positive_float(weight, field="sampling_weight")
            object.__setattr__(self, "sampling_weight", weight)
        if self.representative_estimation_eligible and weight is None:
            raise ValueError("representative quality evidence requires sampling_weight")
        if self.screening_selected and self.model_abstained:
            raise ValueError("an abstained model cannot select a screening candidate")
        probability = self.calibrated_supported_probability
        if probability is not None:
            object.__setattr__(
                self,
                "calibrated_supported_probability",
                _probability(probability, field="calibrated_supported_probability"),
            )
        error = not self.model_abstained and (
            self.screening_selected != self.human_supported
        )
        if self.reference_outlier_influenced_error and not error:
            raise ValueError(
                "reference_outlier_influenced_error requires an observed decision error"
            )
        for field in (
            "country_code",
            "admin1",
            "bioregion",
            "geographic_cluster_id",
        ):
            object.__setattr__(
                self,
                field,
                _optional_text(getattr(self, field), field=field),
            )
        if self.country_code is not None:
            object.__setattr__(self, "country_code", self.country_code.upper())
        if not self.no_geo and self.geographic_cluster_id is None:
            raise ValueError("geocoded evidence requires geographic_cluster_id")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


def report_overall_pooling_quality(
    observations: Sequence[DynamicPoolQualityObservation],
    *,
    policy: DynamicPoolQualityPolicy | None = None,
) -> pl.DataFrame:
    """Report representative overall quality without mixing targeted evidence."""

    items = _normalized_observations(observations)
    identity = _identity(
        hierarchy_level="overall",
        group_id="overall",
        group_label="Overall representative audit",
    )
    return _build_quality_report(
        items,
        groups=[(identity, items)],
        policy=policy or DynamicPoolQualityPolicy(),
    )


def report_family_pooling_quality(
    observations: Sequence[DynamicPoolQualityObservation],
    *,
    policy: DynamicPoolQualityPolicy | None = None,
) -> pl.DataFrame:
    """Report each canonical family under the common quality contract."""

    items = _normalized_observations(observations)
    if not items:
        raise ValueError("family quality report requires source observations")
    return _build_quality_report(
        items,
        groups=_taxonomy_groups(items, hierarchy_level="family"),
        policy=policy or DynamicPoolQualityPolicy(),
    )


def report_genus_pooling_quality(
    observations: Sequence[DynamicPoolQualityObservation],
    *,
    policy: DynamicPoolQualityPolicy | None = None,
) -> pl.DataFrame:
    """Report canonical genera with a consistent family parent."""

    items = _normalized_observations(observations)
    if not items:
        raise ValueError("genus quality report requires source observations")
    return _build_quality_report(
        items,
        groups=_taxonomy_groups(items, hierarchy_level="genus"),
        policy=policy or DynamicPoolQualityPolicy(),
    )


def report_species_pooling_quality(
    observations: Sequence[DynamicPoolQualityObservation],
    *,
    policy: DynamicPoolQualityPolicy | None = None,
) -> pl.DataFrame:
    """Report canonical species with consistent genus and family parents."""

    items = _normalized_observations(observations)
    if not items:
        raise ValueError("species quality report requires source observations")
    return _build_quality_report(
        items,
        groups=_taxonomy_groups(items, hierarchy_level="species"),
        policy=policy or DynamicPoolQualityPolicy(),
    )


def report_geographic_pooling_quality(
    observations: Sequence[DynamicPoolQualityObservation],
    *,
    policy: DynamicPoolQualityPolicy | None = None,
) -> pl.DataFrame:
    """Report complete geographic strata without inferring biological absence."""

    items = _normalized_observations(observations)
    if not items:
        raise ValueError("geographic quality report requires source observations")
    return _build_quality_report(
        items,
        groups=_geographic_groups(items),
        policy=policy or DynamicPoolQualityPolicy(),
    )


def validate_dynamic_pool_quality_report(table: pl.DataFrame) -> None:
    """Reject schema, fingerprint, interval or release-authority drift."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("dynamic-pool quality report must be a Polars DataFrame")
    if table.schema != QUALITY_REPORT_SCHEMA:
        raise ValueError("dynamic-pool quality report schema does not match contract")
    if table.is_empty():
        raise ValueError("dynamic-pool quality report must not be empty")
    expected = table.sort(["hierarchy_level", "group_id", "metric_name"])
    if not table.equals(expected):
        raise ValueError("dynamic-pool quality report is not canonically sorted")
    if table.filter(
        (pl.col("schema_version") != DYNAMIC_POOL_QUALITY_REPORT_VERSION)
        | ~pl.col("descriptive_audit_only")
        | pl.col("authorizes_occurrence_release")
        | ~pl.col("group_status").is_in(["complete", "insufficient_sample"])
        | ~pl.col("metric_status").is_in(
            ["complete", "insufficient_sample", "insufficient_metric_sample"]
        )
    ).height:
        raise ValueError("dynamic-pool quality report crossed its authority contract")
    if table.filter(
        pl.col("estimate").is_not_null()
        & ~pl.col("estimate").is_between(0.0, 1.0, closed="both")
    ).height:
        raise ValueError("dynamic-pool quality estimate is outside [0, 1]")
    if table.filter(
        pl.col("confidence_interval_lower").is_not_null()
        & (
            pl.col("confidence_interval_upper").is_null()
            | (pl.col("confidence_interval_lower") > pl.col("estimate"))
            | (pl.col("estimate") > pl.col("confidence_interval_upper"))
        )
    ).height:
        raise ValueError("dynamic-pool quality confidence interval is invalid")
    if table.filter(
        (pl.col("metric_status") != "complete") & pl.col("estimate").is_not_null()
    ).height:
        raise ValueError("insufficient quality evidence cannot emit an estimate")
    geographic = pl.col("hierarchy_level") == "geography"
    if table.filter(
        geographic
        & (
            pl.col("geography_level").is_null()
            | ~pl.col("geography_level").is_in(GEOGRAPHY_QUALITY_LEVELS)
            | pl.col("geography_value").is_null()
        )
    ).height:
        raise ValueError("geographic quality identity is incomplete")
    if table.filter(
        ~geographic
        & (
            pl.col("geography_level").is_not_null()
            | pl.col("geography_value").is_not_null()
            | pl.col("no_geo").is_not_null()
        )
    ).height:
        raise ValueError("non-geographic quality row carries geography identity")
    availability_or_cluster = pl.col("geography_level").is_in(
        ["availability", "geographic_cluster"]
    )
    if table.filter(
        geographic
        & availability_or_cluster
        & (
            pl.col("no_geo").is_null()
            | (pl.col("no_geo") != (pl.col("geography_value") == "no_geo"))
        )
    ).height:
        raise ValueError("no-geo quality identity is inconsistent")
    if table.filter(
        geographic & ~availability_or_cluster & pl.col("no_geo").is_not_null()
    ).height:
        raise ValueError("geographic field strata cannot imply no-geo status")
    for row in table.iter_rows(named=True):
        base = {
            field: row[field]
            for field in QUALITY_REPORT_SCHEMA
            if field not in {"schema_version", "report_fingerprint", "row_fingerprint"}
        }
        if row["row_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("dynamic-pool quality row fingerprint mismatch")
    if table["report_fingerprint"].n_unique() != 1:
        raise ValueError("dynamic-pool quality report has mixed fingerprints")


def _build_quality_report(
    observations: tuple[DynamicPoolQualityObservation, ...],
    *,
    groups: Sequence[
        tuple[dict[str, object], tuple[DynamicPoolQualityObservation, ...]]
    ],
    policy: DynamicPoolQualityPolicy,
) -> pl.DataFrame:
    semantic_rows: list[dict[str, object]] = []
    for identity, source_items in groups:
        eligible = tuple(
            item for item in source_items if item.representative_estimation_eligible
        )
        semantic_rows.extend(
            _measure_quality_group(
                identity,
                source_items=source_items,
                eligible=eligible,
                policy=policy,
            )
        )
    row_fingerprints = [canonical_semantic_fingerprint(row) for row in semantic_rows]
    report_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_QUALITY_REPORT_VERSION,
            "policy_fingerprint": policy.fingerprint,
            "source_observation_fingerprints": [
                item.fingerprint for item in observations
            ],
            "row_fingerprints": row_fingerprints,
        }
    )
    rows = [
        {
            "schema_version": DYNAMIC_POOL_QUALITY_REPORT_VERSION,
            "report_fingerprint": report_fingerprint,
            "row_fingerprint": row_fingerprint,
            **row,
        }
        for row, row_fingerprint in zip(semantic_rows, row_fingerprints, strict=True)
    ]
    table = pl.DataFrame(rows, schema=QUALITY_REPORT_SCHEMA, strict=True).sort(
        ["hierarchy_level", "group_id", "metric_name"]
    )
    validate_dynamic_pool_quality_report(table)
    return table


def _measure_quality_group(
    identity: dict[str, object],
    *,
    source_items: tuple[DynamicPoolQualityObservation, ...],
    eligible: tuple[DynamicPoolQualityObservation, ...],
    policy: DynamicPoolQualityPolicy,
) -> list[dict[str, object]]:
    weights = [_weight(item) for item in eligible]
    component_count = len({item.independence_component_id for item in eligible})
    weighted_count = sum(weights)
    group_effective_n = _effective_sample_size(eligible, [True] * len(eligible))
    group_reasons = _sample_reasons(
        item_count=len(eligible),
        component_count=component_count,
        effective_n=group_effective_n,
        minimum_items=policy.minimum_group_items,
        minimum_components=policy.minimum_group_components,
        minimum_effective_n=policy.minimum_group_effective_sample_size,
        prefix="group",
    )
    group_status = "insufficient_sample" if group_reasons else "complete"
    retained = [not item.model_abstained for item in eligible]
    selected = [item.screening_selected for item in eligible]
    supported = [item.human_supported for item in eligible]
    errors = [
        retained[index] and selected[index] != supported[index]
        for index in range(len(eligible))
    ]
    tp = _weighted_count(
        eligible,
        [
            retained[index] and selected[index] and supported[index]
            for index in range(len(eligible))
        ],
    )
    tn = _weighted_count(
        eligible,
        [
            retained[index] and not selected[index] and not supported[index]
            for index in range(len(eligible))
        ],
    )
    fp = _weighted_count(
        eligible,
        [
            retained[index] and selected[index] and not supported[index]
            for index in range(len(eligible))
        ],
    )
    fn = _weighted_count(
        eligible,
        [
            retained[index] and not selected[index] and supported[index]
            for index in range(len(eligible))
        ],
    )
    common = {
        "policy_fingerprint": policy.fingerprint,
        **identity,
        "source_item_count": len(source_items),
        "evaluated_item_count": len(eligible),
        "excluded_targeted_item_count": len(source_items) - len(eligible),
        "independence_component_count": component_count,
        "weighted_item_count": weighted_count if eligible else None,
        "group_effective_sample_size": group_effective_n,
        "group_status": group_status,
        "group_insufficiency_reasons": group_reasons,
        "weighted_true_positive": tp if eligible else None,
        "weighted_true_negative": tn if eligible else None,
        "weighted_false_positive": fp if eligible else None,
        "weighted_false_negative": fn if eligible else None,
    }
    specifications = (
        (
            "selection_precision",
            "grouped_weighted_proportion",
            [selected[i] and supported[i] for i in range(len(eligible))],
            selected,
        ),
        (
            "selection_recall",
            "grouped_weighted_proportion",
            [selected[i] and supported[i] for i in range(len(eligible))],
            supported,
        ),
        (
            "decision_accuracy",
            "grouped_weighted_proportion",
            [retained[i] and not errors[i] for i in range(len(eligible))],
            retained,
        ),
        (
            "model_coverage",
            "grouped_weighted_proportion",
            retained,
            [True] * len(eligible),
        ),
        (
            "abstention_rate",
            "grouped_weighted_proportion",
            [item.model_abstained for item in eligible],
            [True] * len(eligible),
        ),
        (
            "family_routing_error_rate",
            "grouped_weighted_proportion",
            [not item.family_routing_correct for item in eligible],
            [True] * len(eligible),
        ),
        (
            "global_local_disagreement_rate",
            "grouped_weighted_proportion",
            [item.global_local_disagreed for item in eligible],
            [True] * len(eligible),
        ),
        (
            "local_support_insufficiency_rate",
            "grouped_weighted_proportion",
            [not item.local_support_available for item in eligible],
            [True] * len(eligible),
        ),
        (
            "reference_outlier_error_influence_rate",
            "grouped_weighted_proportion",
            [item.reference_outlier_influenced_error for item in eligible],
            errors,
        ),
        (
            "ood_false_positive_incidence",
            "grouped_weighted_proportion",
            [
                item.out_of_distribution and selected[index] and not supported[index]
                for index, item in enumerate(eligible)
            ],
            [item.out_of_distribution for item in eligible],
        ),
        (
            "calibrated_probability_availability",
            "grouped_weighted_proportion",
            [item.calibrated_supported_probability is not None for item in eligible],
            [True] * len(eligible),
        ),
    )
    rows = [
        {
            **common,
            **_proportion_metric(
                eligible,
                metric_name=name,
                metric_kind=kind,
                numerator_mask=numerator,
                denominator_mask=denominator,
                group_reasons=group_reasons,
                policy=policy,
            ),
        }
        for name, kind, numerator, denominator in specifications
    ]
    probability_mask = [
        item.calibrated_supported_probability is not None for item in eligible
    ]
    for metric_name in ("weighted_brier_score", "weighted_ece"):
        rows.append(
            {
                **common,
                **_calibration_metric(
                    eligible,
                    metric_name=metric_name,
                    denominator_mask=probability_mask,
                    group_reasons=group_reasons,
                    policy=policy,
                ),
            }
        )
    return rows


def _proportion_metric(
    items: tuple[DynamicPoolQualityObservation, ...],
    *,
    metric_name: str,
    metric_kind: str,
    numerator_mask: Sequence[bool],
    denominator_mask: Sequence[bool],
    group_reasons: list[str],
    policy: DynamicPoolQualityPolicy,
) -> dict[str, object]:
    denominator_items = [
        item for item, selected in zip(items, denominator_mask, strict=True) if selected
    ]
    numerator_count = sum(
        bool(numerator and denominator)
        for numerator, denominator in zip(numerator_mask, denominator_mask, strict=True)
    )
    numerator_weight = sum(
        (
            _weight(item)
            for item, numerator, denominator in zip(
                items, numerator_mask, denominator_mask, strict=True
            )
            if numerator and denominator
        ),
        0.0,
    )
    denominator_weight = sum(
        (_weight(item) for item in denominator_items),
        0.0,
    )
    component_count = len(
        {item.independence_component_id for item in denominator_items}
    )
    effective_n = _effective_sample_size(items, denominator_mask)
    reasons = list(group_reasons)
    if not group_reasons:
        reasons.extend(
            _sample_reasons(
                item_count=len(denominator_items),
                component_count=component_count,
                effective_n=effective_n,
                minimum_items=policy.minimum_metric_denominator_items,
                minimum_components=policy.minimum_metric_denominator_components,
                minimum_effective_n=policy.minimum_metric_effective_sample_size,
                prefix="metric_denominator",
            )
        )
    status = (
        "insufficient_sample"
        if group_reasons
        else "insufficient_metric_sample"
        if reasons
        else "complete"
    )
    estimate = (
        numerator_weight / denominator_weight
        if status == "complete" and denominator_weight > 0.0
        else None
    )
    lower, upper = (
        _wilson_interval(
            estimate,
            effective_n=effective_n,
            confidence_level=policy.confidence_level,
        )
        if estimate is not None and effective_n is not None
        else (None, None)
    )
    return {
        "metric_name": metric_name,
        "metric_kind": metric_kind,
        "metric_status": status,
        "metric_insufficiency_reasons": reasons,
        "numerator_item_count": numerator_count,
        "denominator_item_count": len(denominator_items),
        "numerator_weight": numerator_weight if denominator_items else None,
        "denominator_weight": denominator_weight if denominator_items else None,
        "denominator_component_count": component_count,
        "metric_effective_sample_size": effective_n,
        "estimate": estimate,
        "confidence_interval_lower": lower,
        "confidence_interval_upper": upper,
        "confidence_level": policy.confidence_level,
        "confidence_interval_method": GROUPED_WEIGHTED_INTERVAL_METHOD,
        "descriptive_audit_only": True,
        "authorizes_occurrence_release": False,
    }


def _calibration_metric(
    items: tuple[DynamicPoolQualityObservation, ...],
    *,
    metric_name: str,
    denominator_mask: Sequence[bool],
    group_reasons: list[str],
    policy: DynamicPoolQualityPolicy,
) -> dict[str, object]:
    denominator_items = [
        item for item, selected in zip(items, denominator_mask, strict=True) if selected
    ]
    component_count = len(
        {item.independence_component_id for item in denominator_items}
    )
    effective_n = _effective_sample_size(items, denominator_mask)
    reasons = list(group_reasons)
    if not group_reasons:
        reasons.extend(
            _sample_reasons(
                item_count=len(denominator_items),
                component_count=component_count,
                effective_n=effective_n,
                minimum_items=policy.minimum_metric_denominator_items,
                minimum_components=policy.minimum_metric_denominator_components,
                minimum_effective_n=policy.minimum_metric_effective_sample_size,
                prefix="metric_denominator",
            )
        )
    status = (
        "insufficient_sample"
        if group_reasons
        else "insufficient_metric_sample"
        if reasons
        else "complete"
    )
    estimate = None
    if status == "complete":
        weights = [_weight(item) for item in denominator_items]
        truth = [item.human_supported for item in denominator_items]
        probability = [
            float(item.calibrated_supported_probability) for item in denominator_items
        ]
        if metric_name == "weighted_brier_score":
            estimate = sum(
                weight * (value - float(label)) ** 2
                for weight, value, label in zip(
                    weights, probability, truth, strict=True
                )
            ) / sum(weights)
        else:
            estimate = _weighted_ece(
                truth,
                probability,
                weights,
                bins=policy.calibration_bins,
            )
    denominator_weight = sum(_weight(item) for item in denominator_items)
    return {
        "metric_name": metric_name,
        "metric_kind": "weighted_calibration_error",
        "metric_status": status,
        "metric_insufficiency_reasons": reasons,
        "numerator_item_count": None,
        "denominator_item_count": len(denominator_items),
        "numerator_weight": None,
        "denominator_weight": denominator_weight if denominator_items else None,
        "denominator_component_count": component_count,
        "metric_effective_sample_size": effective_n,
        "estimate": estimate,
        "confidence_interval_lower": None,
        "confidence_interval_upper": None,
        "confidence_level": policy.confidence_level,
        "confidence_interval_method": "not_applicable_point_diagnostic",
        "descriptive_audit_only": True,
        "authorizes_occurrence_release": False,
    }


def _identity(
    *,
    hierarchy_level: str,
    group_id: str,
    group_label: str,
    geography_level: str | None = None,
    family_key: str | None = None,
    family_name: str | None = None,
    genus_key: str | None = None,
    genus_name: str | None = None,
    species_key: str | None = None,
    scientific_name: str | None = None,
    geography_value: str | None = None,
    no_geo: bool | None = None,
) -> dict[str, object]:
    return {
        "hierarchy_level": hierarchy_level,
        "geography_level": geography_level,
        "group_id": group_id,
        "group_label": group_label,
        "family_key": family_key,
        "family_name": family_name,
        "genus_key": genus_key,
        "genus_name": genus_name,
        "species_key": species_key,
        "scientific_name": scientific_name,
        "geography_value": geography_value,
        "no_geo": no_geo,
    }


def _sample_reasons(
    *,
    item_count: int,
    component_count: int,
    effective_n: float | None,
    minimum_items: int,
    minimum_components: int,
    minimum_effective_n: float,
    prefix: str,
) -> list[str]:
    reasons = []
    if item_count == 0:
        reasons.append(f"{prefix}_empty")
    elif item_count < minimum_items:
        reasons.append(f"{prefix}_items_below_minimum")
    if component_count < minimum_components:
        reasons.append(f"{prefix}_components_below_minimum")
    if effective_n is None or effective_n < minimum_effective_n:
        reasons.append(f"{prefix}_effective_sample_size_below_minimum")
    return reasons


def _effective_sample_size(
    items: Sequence[DynamicPoolQualityObservation],
    mask: Sequence[bool],
) -> float | None:
    selected = [item for item, include in zip(items, mask, strict=True) if include]
    if not selected:
        return None
    weights = [_weight(item) for item in selected]
    row_effective_n = sum(weights) ** 2 / sum(weight * weight for weight in weights)
    component_weights: dict[str, float] = defaultdict(float)
    for item, weight in zip(selected, weights, strict=True):
        component_weights[item.independence_component_id] += weight
    component_values = list(component_weights.values())
    component_effective_n = sum(component_values) ** 2 / sum(
        value * value for value in component_values
    )
    return min(row_effective_n, component_effective_n)


def _wilson_interval(
    estimate: float,
    *,
    effective_n: float,
    confidence_level: float,
) -> tuple[float, float]:
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z_score * z_score / effective_n
    center = (estimate + z_score * z_score / (2.0 * effective_n)) / denominator
    spread = (
        z_score
        * sqrt(
            estimate * (1.0 - estimate) / effective_n
            + z_score * z_score / (4.0 * effective_n * effective_n)
        )
        / denominator
    )
    return (
        min(estimate, max(0.0, center - spread)),
        max(estimate, min(1.0, center + spread)),
    )


def _weighted_ece(
    truth: Sequence[bool],
    probability: Sequence[float],
    weights: Sequence[float],
    *,
    bins: int,
) -> float:
    total_weight = sum(weights)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, value in enumerate(probability)
            if value >= lower
            and (value <= upper if index == bins - 1 else value < upper)
        ]
        if not selected:
            continue
        bin_weight = sum(weights[position] for position in selected)
        accuracy = (
            sum(weights[position] * float(truth[position]) for position in selected)
            / bin_weight
        )
        confidence = (
            sum(weights[position] * probability[position] for position in selected)
            / bin_weight
        )
        error += bin_weight / total_weight * abs(accuracy - confidence)
    return error


def _weighted_count(
    items: Sequence[DynamicPoolQualityObservation],
    mask: Sequence[bool],
) -> float:
    return sum(
        (_weight(item) for item, selected in zip(items, mask, strict=True) if selected),
        0.0,
    )


def _weight(item: DynamicPoolQualityObservation) -> float:
    assert item.sampling_weight is not None
    return item.sampling_weight


def _normalized_observations(
    observations: Sequence[DynamicPoolQualityObservation],
) -> tuple[DynamicPoolQualityObservation, ...]:
    items = tuple(observations)
    if any(not isinstance(item, DynamicPoolQualityObservation) for item in items):
        raise TypeError("quality observations must be DynamicPoolQualityObservation")
    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    if len({item.item_id for item in ordered}) != len(ordered):
        raise ValueError("quality observation item IDs must be unique")
    if len({item.source_record_id for item in ordered}) != len(ordered):
        raise ValueError("quality observation source record IDs must be unique")
    return ordered


def _unique_group_label(
    items: Sequence[DynamicPoolQualityObservation],
    *,
    field: str,
    group_id: str,
) -> str:
    values = {str(getattr(item, field)) for item in items}
    if len(values) != 1:
        raise ValueError(f"{group_id} has conflicting {field} values")
    return next(iter(values))


def _taxonomy_groups(
    items: tuple[DynamicPoolQualityObservation, ...],
    *,
    hierarchy_level: str,
) -> list[tuple[dict[str, object], tuple[DynamicPoolQualityObservation, ...]]]:
    key_field = f"{hierarchy_level}_key"
    grouped: dict[str, list[DynamicPoolQualityObservation]] = defaultdict(list)
    for item in items:
        grouped[str(getattr(item, key_field))].append(item)
    groups = []
    for taxon_key in sorted(grouped):
        group_items = tuple(grouped[taxon_key])
        family_key = _unique_group_label(
            group_items,
            field="family_key",
            group_id=taxon_key,
        )
        family_name = _unique_group_label(
            group_items,
            field="family_name",
            group_id=taxon_key,
        )
        genus_key = None
        genus_name = None
        species_key = None
        scientific_name = None
        if hierarchy_level in {"genus", "species"}:
            genus_key = _unique_group_label(
                group_items,
                field="genus_key",
                group_id=taxon_key,
            )
            genus_name = _unique_group_label(
                group_items,
                field="genus_name",
                group_id=taxon_key,
            )
        if hierarchy_level == "species":
            species_key = _unique_group_label(
                group_items,
                field="species_key",
                group_id=taxon_key,
            )
            scientific_name = _unique_group_label(
                group_items,
                field="scientific_name",
                group_id=taxon_key,
            )
        group_label = {
            "family": family_name,
            "genus": genus_name,
            "species": scientific_name,
        }[hierarchy_level]
        groups.append(
            (
                _identity(
                    hierarchy_level=hierarchy_level,
                    group_id=f"{hierarchy_level}:{taxon_key}",
                    group_label=str(group_label),
                    family_key=family_key,
                    family_name=family_name,
                    genus_key=genus_key,
                    genus_name=genus_name,
                    species_key=species_key,
                    scientific_name=scientific_name,
                ),
                group_items,
            )
        )
    return groups


def _geographic_groups(
    items: tuple[DynamicPoolQualityObservation, ...],
) -> list[tuple[dict[str, object], tuple[DynamicPoolQualityObservation, ...]]]:
    grouped: dict[tuple[str, str], list[DynamicPoolQualityObservation]] = defaultdict(
        list
    )
    for item in items:
        values = {
            "availability": "no_geo" if item.no_geo else "geocoded",
            "country": item.country_code or "unknown_country",
            "admin1": item.admin1 or "unknown_admin1",
            "bioregion": item.bioregion or "unknown_bioregion",
            "geographic_cluster": (
                "no_geo" if item.no_geo else str(item.geographic_cluster_id)
            ),
        }
        for level, value in values.items():
            grouped[(level, value)].append(item)
    groups = []
    for level, value in sorted(grouped):
        group_items = tuple(grouped[(level, value)])
        no_geo = (
            value == "no_geo"
            if level in {"availability", "geographic_cluster"}
            else None
        )
        groups.append(
            (
                _identity(
                    hierarchy_level="geography",
                    group_id=f"geography:{level}:{value}",
                    group_label=f"{level}={value}",
                    geography_level=level,
                    geography_value=value,
                    no_geo=no_geo,
                ),
                group_items,
            )
        )
    return groups


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


__all__ = [
    "DYNAMIC_POOL_QUALITY_OBSERVATION_VERSION",
    "DYNAMIC_POOL_QUALITY_POLICY_VERSION",
    "DYNAMIC_POOL_QUALITY_REPORT_VERSION",
    "GEOGRAPHY_QUALITY_LEVELS",
    "GROUPED_WEIGHTED_INTERVAL_METHOD",
    "QUALITY_REPORT_SCHEMA",
    "REPRESENTATIVE_AUDIT_PURPOSE",
    "SAMPLING_PURPOSES",
    "TARGETED_FAILURE_PURPOSE",
    "DynamicPoolQualityObservation",
    "DynamicPoolQualityPolicy",
    "report_family_pooling_quality",
    "report_geographic_pooling_quality",
    "report_genus_pooling_quality",
    "report_overall_pooling_quality",
    "report_species_pooling_quality",
    "validate_dynamic_pool_quality_report",
]
