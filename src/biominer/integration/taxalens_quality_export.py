"""TaxaLens review-sampling and quality sidecar exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_quality import (
    DYNAMIC_POOL_QUALITY_REPORT_VERSION,
    validate_dynamic_pool_quality_report,
)
from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    ProbabilityAuditSelection,
    RAW_SCORE_SEMANTICS,
    validate_probability_audit_selection,
)
from biominer.integration.product_handoff import (
    normalize_product_artifacts,
    validate_fingerprint,
)
from biominer.integration.taxalens_pool_handoff import TAXALENS_ROLE_DEFAULTS
from biominer.storage.content_address import sha256_file
from biominer.storage.parquet import write_parquet


TAXALENS_REVIEW_QUALITY_ROLES = (
    "review_sampling_frame",
    "quality_sidecar",
    "geographic_cells",
)
TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION = (
    "biominer-taxalens-review-sampling-frame-v1.0.0"
)
TAXALENS_QUALITY_SIDECAR_SCHEMA_VERSION = "biominer-taxalens-quality-sidecar-v1.0.0"
TAXALENS_GEOGRAPHIC_CELLS_SCHEMA_VERSION = "biominer-taxalens-geographic-cells-v1.0.0"
GEOGRAPHIC_CELLS_DOWNSTREAM_REASON = (
    "TaxaLens owns baseline-provider-union and geographic-impact materialization"
)

TAXALENS_REVIEW_SAMPLING_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "sampling_plan_id": pl.String,
    "sampling_plan_fingerprint": pl.String,
    "sampling_policy_fingerprint": pl.String,
    "sampling_register_fingerprint": pl.String,
    "sampling_purpose": pl.String,
    "sampling_design": pl.String,
    "representative": pl.Boolean,
    "blind_review": pl.Boolean,
    "selection_seed": pl.UInt64,
    "independent_unit": pl.String,
    "grouping_keys": pl.List(pl.String),
    "sampling_stratum_id": pl.String,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "dataset_partition": pl.String,
    "source_sampling_unit_id": pl.String,
    "source_record_hash": pl.String,
    "source_artifact_fingerprint": pl.String,
    "source_frame_fingerprint": pl.String,
    "flickr_photo_id": pl.String,
    "organism_unit_id": pl.String,
    "candidate_species_accepted_taxon_key": pl.String,
    "candidate_species_scientific_name": pl.String,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "owner_group_id": pl.String,
    "duplicate_group_id": pl.String,
    "observation_group_id": pl.String,
    "variance_cluster_id": pl.String,
    "raw_score_semantics": pl.String,
    "raw_score_is_probability": pl.Boolean,
    "no_geo_is_biological_absence": pl.Boolean,
    "occurrence_release_authorized": pl.Boolean,
    "sampling_row_fingerprint": pl.String,
}

_QUALITY_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "sidecar_fingerprint",
        "source_quality_report_schema_version",
        "source_quality_report_fingerprint",
        "quality_policy_fingerprint",
        "completed_review_count",
        "representative_evaluated_count",
        "targeted_review_excluded_count",
        "independence_component_count",
        "group_effective_sample_size",
        "quality_estimate_available",
        "quality_status",
        "quality_unavailable_reasons",
        "metrics",
        "representative_and_targeted_are_separate",
        "authorizes_occurrence_release",
        "scientific_claim_allowed",
    }
)
_QUALITY_METRIC_FIELDS = frozenset(
    {
        "metric_name",
        "metric_kind",
        "metric_status",
        "metric_insufficiency_reasons",
        "numerator_item_count",
        "denominator_item_count",
        "denominator_component_count",
        "metric_effective_sample_size",
        "estimate",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "confidence_level",
        "confidence_interval_method",
    }
)
_GROUPING_KEYS = [
    "owner_group_id",
    "duplicate_group_id",
    "observation_group_id",
]


@dataclass(frozen=True, slots=True)
class TaxaLensReviewQualityExport:
    """Sidecar descriptors and maturity inputs for the product manifest."""

    root: Path
    sidecar_directory: Path
    artifacts: tuple[dict[str, object], ...]
    completed_review_count: int
    quality_estimate_available: bool
    quality_unavailable_reason: str | None


def build_taxalens_review_sampling_frame(
    selection: ProbabilityAuditSelection,
    *,
    policy: ProbabilityAuditSamplingPolicy,
) -> pl.DataFrame:
    """Project selected representative units without changing their design."""

    if not isinstance(selection, ProbabilityAuditSelection):
        raise TypeError("selection must be a ProbabilityAuditSelection")
    if not isinstance(policy, ProbabilityAuditSamplingPolicy):
        raise TypeError("policy must be a ProbabilityAuditSamplingPolicy")
    validate_probability_audit_selection(selection.register, selection.sample)
    if (
        selection.population_count != selection.register.height
        or selection.selected_count != selection.sample.height
    ):
        raise ValueError("TaxaLens review selection counts differ from its tables")
    if selection.sample.is_empty():
        raise ValueError("TaxaLens review sampling export requires selected units")
    if set(selection.register["sample_policy_fingerprint"].to_list()) != {
        policy.fingerprint
    } or set(selection.sample["sample_policy_fingerprint"].to_list()) != {
        policy.fingerprint
    }:
        raise ValueError("TaxaLens review sample and sampling policy differ")
    register_fingerprint = selection.register_fingerprint
    if set(selection.register["sampling_register_fingerprint"].to_list()) != {
        register_fingerprint
    } or set(selection.sample["sampling_register_fingerprint"].to_list()) != {
        register_fingerprint
    }:
        raise ValueError("TaxaLens review selection register fingerprint differs")
    plan_payload = {
        "schema_version": TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION,
        "sampling_policy_fingerprint": policy.fingerprint,
        "sampling_register_fingerprint": register_fingerprint,
        "sampling_purpose": "representative_audit",
        "representative": True,
        "blind_review": True,
        "selection_seed": policy.random_seed,
        "grouping_keys": _GROUPING_KEYS,
    }
    plan_fingerprint = canonical_semantic_fingerprint(plan_payload)
    plan_id = "taxalens-review-sampling-plan:" + plan_fingerprint.removeprefix(
        "sha256:"
    )
    rows: list[dict[str, object]] = []
    for source in selection.sample.iter_rows(named=True):
        base = {
            "schema_version": TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION,
            "sampling_plan_id": plan_id,
            "sampling_plan_fingerprint": plan_fingerprint,
            "sampling_policy_fingerprint": policy.fingerprint,
            "sampling_register_fingerprint": register_fingerprint,
            "sampling_purpose": "representative_audit",
            "sampling_design": source["sampling_design"],
            "representative": True,
            "blind_review": True,
            "selection_seed": policy.random_seed,
            "independent_unit": source["sampling_population_unit_id"],
            "grouping_keys": plan_payload["grouping_keys"],
            "sampling_stratum_id": source["analysis_stratum_id"],
            "inclusion_probability": source["inclusion_probability"],
            "sampling_weight": source["sampling_weight"],
            "dataset_partition": "review_only_not_model_split",
            "source_sampling_unit_id": source["sampling_unit_id"],
            "source_record_hash": source["source_record_hash"],
            "source_artifact_fingerprint": source["source_artifact_fingerprint"],
            "source_frame_fingerprint": source["frame_fingerprint"],
            "flickr_photo_id": source["flickr_photo_id"],
            "organism_unit_id": source["organism_unit_id"],
            "candidate_species_accepted_taxon_key": source[
                "candidate_species_accepted_taxon_key"
            ],
            "candidate_species_scientific_name": source[
                "candidate_species_scientific_name"
            ],
            "geographic_cluster_id": source["geographic_cluster_id"],
            "no_geo": source["no_geo"],
            "owner_group_id": source["owner_group_id"],
            "duplicate_group_id": source["duplicate_group_id"],
            "observation_group_id": source["observation_group_id"],
            "variance_cluster_id": source["variance_cluster_id"],
            "raw_score_semantics": source["score_semantics"],
            "raw_score_is_probability": False,
            "no_geo_is_biological_absence": False,
            "occurrence_release_authorized": False,
        }
        rows.append(
            {
                **base,
                "sampling_row_fingerprint": canonical_semantic_fingerprint(base),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema=TAXALENS_REVIEW_SAMPLING_SCHEMA,
        strict=True,
    ).sort("sampling_stratum_id", "independent_unit", "source_sampling_unit_id")
    validate_taxalens_review_sampling_frame(frame)
    return frame


def validate_taxalens_review_sampling_frame(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("TaxaLens review sampling frame must be a DataFrame")
    if frame.schema != TAXALENS_REVIEW_SAMPLING_SCHEMA or frame.is_empty():
        raise ValueError("TaxaLens review sampling frame schema or rows differ")
    expected = frame.sort(
        "sampling_stratum_id", "independent_unit", "source_sampling_unit_id"
    )
    if not frame.equals(expected):
        raise ValueError("TaxaLens review sampling frame is not canonical")
    if frame["sampling_row_fingerprint"].n_unique() != frame.height:
        raise ValueError("TaxaLens review sampling rows are not unique")
    if (
        frame["independent_unit"].n_unique() != frame.height
        or frame["source_sampling_unit_id"].n_unique() != frame.height
    ):
        raise ValueError("TaxaLens review sampling units are not independent")
    if frame.filter(
        (pl.col("schema_version") != TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION)
        | (pl.col("sampling_purpose") != "representative_audit")
        | ~pl.col("representative")
        | ~pl.col("blind_review")
        | pl.col("raw_score_is_probability")
        | pl.col("no_geo_is_biological_absence")
        | pl.col("occurrence_release_authorized")
        | ~pl.col("inclusion_probability").is_between(0.0, 1.0, closed="right")
        | (
            (pl.col("inclusion_probability") * pl.col("sampling_weight") - 1.0).abs()
            > 1e-12
        )
        | (pl.col("dataset_partition") != "review_only_not_model_split")
        | (pl.col("raw_score_semantics") != RAW_SCORE_SEMANTICS)
        | (pl.col("no_geo") & pl.col("geographic_cluster_id").is_not_null())
        | (~pl.col("no_geo") & pl.col("geographic_cluster_id").is_null())
    ).height:
        raise ValueError("TaxaLens review sampling semantics differ")
    singleton_fields = (
        "sampling_plan_id",
        "sampling_plan_fingerprint",
        "sampling_policy_fingerprint",
        "sampling_register_fingerprint",
        "selection_seed",
    )
    identity: dict[str, object] = {}
    for field in singleton_fields:
        values = frame[field].unique().to_list()
        if len(values) != 1:
            raise ValueError(f"TaxaLens review sampling {field} differs")
        identity[field] = values[0]
    if any(keys != _GROUPING_KEYS for keys in frame["grouping_keys"].to_list()):
        raise ValueError("TaxaLens review sampling grouping keys differ")
    plan_payload = {
        "schema_version": TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION,
        "sampling_policy_fingerprint": identity["sampling_policy_fingerprint"],
        "sampling_register_fingerprint": identity["sampling_register_fingerprint"],
        "sampling_purpose": "representative_audit",
        "representative": True,
        "blind_review": True,
        "selection_seed": identity["selection_seed"],
        "grouping_keys": _GROUPING_KEYS,
    }
    expected_plan_fingerprint = canonical_semantic_fingerprint(plan_payload)
    if identity["sampling_plan_fingerprint"] != expected_plan_fingerprint:
        raise ValueError("TaxaLens review sampling plan fingerprint mismatch")
    if identity["sampling_plan_id"] != (
        "taxalens-review-sampling-plan:"
        + expected_plan_fingerprint.removeprefix("sha256:")
    ):
        raise ValueError("TaxaLens review sampling plan identity mismatch")
    fingerprint_fields = (
        "sampling_plan_fingerprint",
        "sampling_policy_fingerprint",
        "sampling_register_fingerprint",
        "source_record_hash",
        "source_artifact_fingerprint",
        "source_frame_fingerprint",
        "sampling_row_fingerprint",
    )
    for field in fingerprint_fields:
        for value in frame[field].unique().to_list():
            if validate_fingerprint(value, field=field) != value:
                raise ValueError(f"TaxaLens review sampling {field} is not canonical")
    required_text_fields = (
        "sampling_plan_id",
        "sampling_stratum_id",
        "independent_unit",
        "source_sampling_unit_id",
        "flickr_photo_id",
        "organism_unit_id",
        "candidate_species_accepted_taxon_key",
        "candidate_species_scientific_name",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
        "variance_cluster_id",
    )
    if any(
        not isinstance(value, str) or not value.strip()
        for field in required_text_fields
        for value in frame[field].to_list()
    ):
        raise ValueError("TaxaLens review sampling required text is blank")
    for row in frame.iter_rows(named=True):
        payload = dict(row)
        fingerprint = payload.pop("sampling_row_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("TaxaLens review sampling fingerprint mismatch")


def build_taxalens_quality_sidecar(report: pl.DataFrame) -> dict[str, object]:
    """Summarize validated overall quality without promoting unavailable metrics."""

    validate_dynamic_pool_quality_report(report)
    overall = report.filter(pl.col("hierarchy_level") == "overall")
    if overall.is_empty() or overall["group_id"].unique().to_list() != ["overall"]:
        raise ValueError("TaxaLens quality sidecar requires one overall quality group")
    identity_fields = (
        "report_fingerprint",
        "policy_fingerprint",
        "source_item_count",
        "evaluated_item_count",
        "excluded_targeted_item_count",
        "independence_component_count",
        "group_effective_sample_size",
        "group_status",
        "group_insufficiency_reasons",
    )
    identity = {}
    for field in identity_fields:
        values = overall[field].unique().to_list()
        if len(values) != 1:
            raise ValueError(f"TaxaLens overall quality {field} differs across metrics")
        identity[field] = values[0]
    metrics = [
        {
            "metric_name": row["metric_name"],
            "metric_kind": row["metric_kind"],
            "metric_status": row["metric_status"],
            "metric_insufficiency_reasons": row["metric_insufficiency_reasons"],
            "numerator_item_count": row["numerator_item_count"],
            "denominator_item_count": row["denominator_item_count"],
            "denominator_component_count": row["denominator_component_count"],
            "metric_effective_sample_size": row["metric_effective_sample_size"],
            "estimate": row["estimate"],
            "confidence_interval_lower": row["confidence_interval_lower"],
            "confidence_interval_upper": row["confidence_interval_upper"],
            "confidence_level": row["confidence_level"],
            "confidence_interval_method": row["confidence_interval_method"],
        }
        for row in overall.sort("metric_name").iter_rows(named=True)
    ]
    estimate_available = any(
        row["metric_status"] == "complete" and row["estimate"] is not None
        for row in metrics
    )
    reasons = sorted(
        {reason for row in metrics for reason in row["metric_insufficiency_reasons"]}
        | set(identity["group_insufficiency_reasons"])
    )
    body: dict[str, object] = {
        "schema_version": TAXALENS_QUALITY_SIDECAR_SCHEMA_VERSION,
        "source_quality_report_schema_version": DYNAMIC_POOL_QUALITY_REPORT_VERSION,
        "source_quality_report_fingerprint": identity["report_fingerprint"],
        "quality_policy_fingerprint": identity["policy_fingerprint"],
        "completed_review_count": identity["source_item_count"],
        "representative_evaluated_count": identity["evaluated_item_count"],
        "targeted_review_excluded_count": identity["excluded_targeted_item_count"],
        "independence_component_count": identity["independence_component_count"],
        "group_effective_sample_size": identity["group_effective_sample_size"],
        "quality_estimate_available": estimate_available,
        "quality_status": "available" if estimate_available else "insufficient_sample",
        "quality_unavailable_reasons": [] if estimate_available else reasons,
        "metrics": metrics,
        "representative_and_targeted_are_separate": True,
        "authorizes_occurrence_release": False,
        "scientific_claim_allowed": False,
    }
    sidecar = {
        **body,
        "sidecar_fingerprint": canonical_semantic_fingerprint(body),
    }
    validate_taxalens_quality_sidecar(sidecar)
    return sidecar


def validate_taxalens_quality_sidecar(sidecar: Mapping[str, object]) -> None:
    if not isinstance(sidecar, Mapping) or set(sidecar) != _QUALITY_TOP_LEVEL_FIELDS:
        raise ValueError("TaxaLens quality sidecar fields differ")
    if sidecar["schema_version"] != TAXALENS_QUALITY_SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported TaxaLens quality sidecar schema")
    if (
        sidecar["source_quality_report_schema_version"]
        != DYNAMIC_POOL_QUALITY_REPORT_VERSION
    ):
        raise ValueError("unsupported TaxaLens source quality report schema")
    for field in (
        "sidecar_fingerprint",
        "source_quality_report_fingerprint",
        "quality_policy_fingerprint",
    ):
        if validate_fingerprint(sidecar[field], field=field) != sidecar[field]:
            raise ValueError(f"TaxaLens quality sidecar {field} is not canonical")
    if (
        sidecar["representative_and_targeted_are_separate"] is not True
        or sidecar["authorizes_occurrence_release"] is not False
        or sidecar["scientific_claim_allowed"] is not False
    ):
        raise ValueError("TaxaLens quality sidecar authority differs")
    for field in (
        "completed_review_count",
        "representative_evaluated_count",
        "targeted_review_excluded_count",
        "independence_component_count",
    ):
        _nonnegative_int(sidecar[field], field=field)
    completed = int(sidecar["completed_review_count"])
    representative = int(sidecar["representative_evaluated_count"])
    targeted = int(sidecar["targeted_review_excluded_count"])
    components = int(sidecar["independence_component_count"])
    if completed < 1 or completed != representative + targeted:
        raise ValueError("TaxaLens quality sidecar review counts differ")
    if components > representative:
        raise ValueError("TaxaLens quality sidecar component count differs")
    group_effective_n = _optional_positive_float(
        sidecar["group_effective_sample_size"],
        field="group_effective_sample_size",
    )
    if group_effective_n is not None and group_effective_n > representative + 1e-12:
        raise ValueError("TaxaLens quality sidecar effective sample size differs")
    if not isinstance(sidecar["quality_estimate_available"], bool):
        raise ValueError("TaxaLens quality sidecar availability must be boolean")
    reasons = _text_array(
        sidecar["quality_unavailable_reasons"],
        field="quality_unavailable_reasons",
    )
    if reasons != sorted(reasons):
        raise ValueError("TaxaLens quality unavailable reasons are not canonical")
    metrics = sidecar["metrics"]
    if (
        not isinstance(metrics, Sequence)
        or isinstance(metrics, (str, bytes))
        or not metrics
    ):
        raise ValueError("TaxaLens quality sidecar metrics must be a nonempty array")
    metric_names: list[str] = []
    for index, row in enumerate(metrics):
        if not isinstance(row, Mapping) or set(row) != _QUALITY_METRIC_FIELDS:
            raise ValueError("TaxaLens quality metric fields differ")
        metric_names.append(_required_text(row["metric_name"], field="metric_name"))
        _required_text(row["metric_kind"], field="metric_kind")
        status = _required_text(row["metric_status"], field="metric_status")
        if status not in {
            "complete",
            "insufficient_sample",
            "insufficient_metric_sample",
        }:
            raise ValueError("TaxaLens quality metric status differs")
        metric_reasons = _text_array(
            row["metric_insufficiency_reasons"],
            field=f"metrics[{index}].metric_insufficiency_reasons",
        )
        numerator = _optional_nonnegative_int(
            row["numerator_item_count"],
            field=f"metrics[{index}].numerator_item_count",
        )
        denominator = _nonnegative_int(
            row["denominator_item_count"],
            field=f"metrics[{index}].denominator_item_count",
        )
        denominator_components = _nonnegative_int(
            row["denominator_component_count"],
            field=f"metrics[{index}].denominator_component_count",
        )
        if denominator > representative or denominator_components > denominator:
            raise ValueError("TaxaLens quality metric denominator differs")
        if numerator is not None and numerator > denominator:
            raise ValueError("TaxaLens quality metric numerator differs")
        effective_n = _optional_positive_float(
            row["metric_effective_sample_size"],
            field=f"metrics[{index}].metric_effective_sample_size",
        )
        if effective_n is not None and effective_n > denominator + 1e-12:
            raise ValueError("TaxaLens quality metric effective sample size differs")
        estimate = _optional_unit_float(
            row["estimate"], field=f"metrics[{index}].estimate"
        )
        lower = _optional_unit_float(
            row["confidence_interval_lower"],
            field=f"metrics[{index}].confidence_interval_lower",
        )
        upper = _optional_unit_float(
            row["confidence_interval_upper"],
            field=f"metrics[{index}].confidence_interval_upper",
        )
        if (lower is None) != (upper is None):
            raise ValueError("TaxaLens quality metric interval is incomplete")
        if lower is not None and (
            estimate is None or not lower <= estimate <= float(upper)
        ):
            raise ValueError("TaxaLens quality metric interval differs")
        confidence = _optional_unit_float(
            row["confidence_level"],
            field=f"metrics[{index}].confidence_level",
        )
        if confidence in {None, 0.0, 1.0}:
            raise ValueError("TaxaLens quality metric confidence level differs")
        _required_text(
            row["confidence_interval_method"],
            field=f"metrics[{index}].confidence_interval_method",
        )
        if status == "complete":
            if estimate is None or metric_reasons:
                raise ValueError("TaxaLens complete quality metric differs")
        elif estimate is not None or not metric_reasons:
            raise ValueError("TaxaLens insufficient quality metric differs")
    if metric_names != sorted(set(metric_names)):
        raise ValueError("TaxaLens quality sidecar metrics are not canonical")
    estimate_available = sidecar["quality_estimate_available"]
    if estimate_available != any(
        row.get("metric_status") == "complete" and row.get("estimate") is not None
        for row in metrics
        if isinstance(row, Mapping)
    ):
        raise ValueError("TaxaLens quality sidecar availability differs")
    if estimate_available != (sidecar["quality_status"] == "available"):
        raise ValueError("TaxaLens quality sidecar status differs")
    if (estimate_available and reasons) or (not estimate_available and not reasons):
        raise ValueError("TaxaLens quality sidecar unavailable reasons differ")
    if not estimate_available and sidecar["quality_status"] != "insufficient_sample":
        raise ValueError("TaxaLens quality sidecar insufficient status differs")
    body = dict(sidecar)
    fingerprint = body.pop("sidecar_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(body):
        raise ValueError("TaxaLens quality sidecar fingerprint mismatch")


def export_taxalens_review_quality_evidence(
    *,
    selection: ProbabilityAuditSelection,
    sampling_policy: ProbabilityAuditSamplingPolicy,
    quality_report: pl.DataFrame | None,
    output_root: str | Path,
    quality_unavailable_reason: str = "no validated reviewed quality report supplied",
) -> TaxaLensReviewQualityExport:
    """Publish review/quality sidecars and explicit downstream geography state."""

    review_frame = build_taxalens_review_sampling_frame(
        selection,
        policy=sampling_policy,
    )
    quality = (
        build_taxalens_quality_sidecar(quality_report)
        if quality_report is not None
        else None
    )
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("TaxaLens handoff root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    artifact_directory = root / "artifacts"
    if artifact_directory.is_symlink():
        raise ValueError("TaxaLens artifact directory must not be a symlink")
    artifact_directory.mkdir(exist_ok=True)
    if not artifact_directory.is_dir():
        raise ValueError("TaxaLens artifact directory must be a directory")
    sidecar_directory = artifact_directory / "review"
    if sidecar_directory.exists():
        raise FileExistsError(
            f"TaxaLens review sidecar directory is create-only: {sidecar_directory}"
        )
    staging_root = Path(mkdtemp(dir=artifact_directory, prefix=".taxalens-review-"))
    staging_sidecar_directory = staging_root / "artifacts" / "review"
    staging_sidecar_directory.mkdir(parents=True)
    try:
        review_filename = TAXALENS_ROLE_DEFAULTS["review_sampling_frame"][0]
        review_path = write_parquet(
            review_frame,
            staging_sidecar_directory / review_filename,
            overwrite=False,
        )
        review_descriptor = _available_descriptor(
            role="review_sampling_frame",
            relative_path=f"artifacts/review/{review_filename}",
            path=review_path,
            row_count=review_frame.height,
            semantic_fingerprint=canonical_semantic_fingerprint(
                {
                    "schema_version": TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION,
                    "row_fingerprints": review_frame[
                        "sampling_row_fingerprint"
                    ].to_list(),
                }
            ),
            parent_fingerprints=sorted(
                {
                    review_frame["sampling_policy_fingerprint"][0],
                    review_frame["sampling_register_fingerprint"][0],
                    *review_frame["source_artifact_fingerprint"].unique().to_list(),
                    *review_frame["source_frame_fingerprint"].unique().to_list(),
                }
            ),
            maturity=None,
        )
        if quality is None:
            quality_descriptor = _unavailable_descriptor(
                role="quality_sidecar",
                reason=quality_unavailable_reason,
            )
            completed_review_count = 0
            quality_estimate_available = False
            unavailable_reason = str(quality_descriptor["unavailable_reason"])
        else:
            quality_filename = TAXALENS_ROLE_DEFAULTS["quality_sidecar"][0]
            quality_path = staging_sidecar_directory / quality_filename
            _write_json_create_only(quality_path, quality)
            quality_descriptor = _available_descriptor(
                role="quality_sidecar",
                relative_path=f"artifacts/review/{quality_filename}",
                path=quality_path,
                row_count=len(quality["metrics"]),
                semantic_fingerprint=str(quality["sidecar_fingerprint"]),
                parent_fingerprints=[
                    str(quality["quality_policy_fingerprint"]),
                    str(quality["source_quality_report_fingerprint"]),
                ],
                maturity="human_reviewed_flickr_labels",
            )
            completed_review_count = int(quality["completed_review_count"])
            quality_estimate_available = bool(quality["quality_estimate_available"])
            unavailable_reason = (
                "validated quality report is insufficient: "
                + ", ".join(quality["quality_unavailable_reasons"])
                if not quality_estimate_available
                else None
            )
        geographic_descriptor = _unavailable_descriptor(
            role="geographic_cells",
            reason=GEOGRAPHIC_CELLS_DOWNSTREAM_REASON,
        )
        descriptors = (
            review_descriptor,
            quality_descriptor,
            geographic_descriptor,
        )
        validate_taxalens_review_quality_export(
            staging_root,
            descriptors,
            completed_review_count=completed_review_count,
            quality_estimate_available=quality_estimate_available,
        )
        staging_sidecar_directory.replace(sidecar_directory)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    result = TaxaLensReviewQualityExport(
        root=root,
        sidecar_directory=sidecar_directory,
        artifacts=descriptors,
        completed_review_count=completed_review_count,
        quality_estimate_available=quality_estimate_available,
        quality_unavailable_reason=unavailable_reason,
    )
    validate_taxalens_review_quality_export(
        result.root,
        result.artifacts,
        completed_review_count=result.completed_review_count,
        quality_estimate_available=result.quality_estimate_available,
    )
    return result


def validate_taxalens_review_quality_export(
    root: str | Path,
    descriptors: Sequence[Mapping[str, object]],
    *,
    completed_review_count: int,
    quality_estimate_available: bool,
) -> None:
    """Re-read published sidecar bytes and exact unavailable states."""

    if (
        isinstance(completed_review_count, bool)
        or not isinstance(completed_review_count, int)
        or completed_review_count < 0
    ):
        raise ValueError("completed_review_count must be a nonnegative integer")
    if not isinstance(quality_estimate_available, bool):
        raise ValueError("quality_estimate_available must be a boolean")
    normalized = normalize_product_artifacts(
        descriptors,
        required_roles=TAXALENS_REVIEW_QUALITY_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit="0" * 40,
    )
    by_role = {str(row["role"]): row for row in normalized}
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("TaxaLens handoff root must not be a symlink")
    sidecar_directory = root_path / "artifacts" / "review"
    if sidecar_directory.is_symlink() or not sidecar_directory.is_dir():
        raise ValueError("TaxaLens review sidecar directory is unavailable")
    expected_paths: set[Path] = set()
    review = by_role["review_sampling_frame"]
    if review["availability"] != "available":
        raise ValueError("TaxaLens review sampling frame must be available")
    _validate_descriptor_contract(review)
    review_path = root_path / str(review["relative_path"])
    expected_paths.add(review_path.resolve())
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("TaxaLens review sampling file is unavailable")
    if (
        review_path.stat().st_size != review["byte_count"]
        or sha256_file(review_path) != review["sha256"]
    ):
        raise ValueError("TaxaLens review sampling physical identity differs")
    review_frame = pl.read_parquet(review_path)
    validate_taxalens_review_sampling_frame(review_frame)
    expected_review_identity = canonical_semantic_fingerprint(
        {
            "schema_version": TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION,
            "row_fingerprints": review_frame["sampling_row_fingerprint"].to_list(),
        }
    )
    if (
        review_frame.height != review["row_count"]
        or expected_review_identity != review["semantic_fingerprint"]
    ):
        raise ValueError("TaxaLens review sampling semantic identity differs")
    expected_review_parents = sorted(
        {
            review_frame["sampling_policy_fingerprint"][0],
            review_frame["sampling_register_fingerprint"][0],
            *review_frame["source_artifact_fingerprint"].unique().to_list(),
            *review_frame["source_frame_fingerprint"].unique().to_list(),
        }
    )
    if review["parent_fingerprints"] != expected_review_parents:
        raise ValueError("TaxaLens review sampling lineage differs")
    if review["evidence_maturity_label"] is not None:
        raise ValueError("TaxaLens review sampling maturity differs")
    quality = by_role["quality_sidecar"]
    _validate_descriptor_contract(quality)
    if quality["availability"] == "available":
        if completed_review_count < 1:
            raise ValueError("TaxaLens quality sidecar requires completed review")
        quality_path = root_path / str(quality["relative_path"])
        expected_paths.add(quality_path.resolve())
        if quality_path.is_symlink() or not quality_path.is_file():
            raise ValueError("TaxaLens quality sidecar file is unavailable")
        if (
            quality_path.stat().st_size != quality["byte_count"]
            or sha256_file(quality_path) != quality["sha256"]
        ):
            raise ValueError("TaxaLens quality sidecar physical identity differs")
        sidecar = json.loads(quality_path.read_text(encoding="utf-8"))
        validate_taxalens_quality_sidecar(sidecar)
        if (
            sidecar["sidecar_fingerprint"] != quality["semantic_fingerprint"]
            or len(sidecar["metrics"]) != quality["row_count"]
            or bool(sidecar["quality_estimate_available"]) != quality_estimate_available
            or int(sidecar["completed_review_count"]) != completed_review_count
        ):
            raise ValueError("TaxaLens quality sidecar semantic identity differs")
        expected_quality_parents = sorted(
            {
                sidecar["quality_policy_fingerprint"],
                sidecar["source_quality_report_fingerprint"],
            }
        )
        if quality["parent_fingerprints"] != expected_quality_parents:
            raise ValueError("TaxaLens quality sidecar lineage differs")
        if quality["evidence_maturity_label"] != "human_reviewed_flickr_labels":
            raise ValueError("TaxaLens quality sidecar maturity differs")
    elif quality_estimate_available or completed_review_count:
        raise ValueError("TaxaLens completed quality evidence requires a sidecar")
    geographic = by_role["geographic_cells"]
    _validate_descriptor_contract(geographic)
    if (
        geographic["availability"] != "unavailable"
        or geographic["unavailable_reason"] != GEOGRAPHIC_CELLS_DOWNSTREAM_REASON
    ):
        raise ValueError("TaxaLens geographic cells must remain downstream-owned")
    entries = tuple(sidecar_directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("TaxaLens review sidecar directory has unsafe entries")
    if {path.resolve() for path in entries} != expected_paths:
        raise ValueError("TaxaLens review sidecar file set differs")


def _available_descriptor(
    *,
    role: str,
    relative_path: str,
    path: Path,
    row_count: int,
    semantic_fingerprint: str,
    parent_fingerprints: Sequence[str],
    maturity: str | None,
) -> dict[str, object]:
    return {
        "role": role,
        "availability": "available",
        "unavailable_reason": None,
        "relative_path": relative_path,
        "media_type": (
            "application/vnd.apache.parquet"
            if path.suffix == ".parquet"
            else "application/json"
        ),
        "schema_version": TAXALENS_ROLE_DEFAULTS[role][1],
        "semantic_fingerprint": semantic_fingerprint,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": row_count,
        "parent_fingerprints": sorted(set(parent_fingerprints)),
        "evidence_maturity_label": maturity,
    }


def _unavailable_descriptor(*, role: str, reason: str) -> dict[str, object]:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"TaxaLens unavailable {role} reason must be nonblank")
    filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
    return {
        "role": role,
        "availability": "unavailable",
        "unavailable_reason": reason.strip(),
        "relative_path": None,
        "media_type": (
            "application/vnd.apache.parquet"
            if filename.endswith(".parquet")
            else "application/json"
        ),
        "schema_version": schema_version,
        "semantic_fingerprint": None,
        "sha256": None,
        "byte_count": None,
        "row_count": None,
        "parent_fingerprints": [],
        "evidence_maturity_label": None,
    }


def _write_json_create_only(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_descriptor_contract(descriptor: Mapping[str, object]) -> None:
    role = str(descriptor["role"])
    filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
    if descriptor["schema_version"] != schema_version:
        raise ValueError(f"TaxaLens review/quality role {role!r} schema differs")
    expected_media_type = (
        "application/vnd.apache.parquet"
        if filename.endswith(".parquet")
        else "application/json"
    )
    if descriptor["media_type"] != expected_media_type:
        raise ValueError(f"TaxaLens review/quality role {role!r} media type differs")
    if (
        descriptor["availability"] == "available"
        and descriptor["relative_path"] != f"artifacts/review/{filename}"
    ):
        raise ValueError(f"TaxaLens review/quality role {role!r} path differs")
    if (
        descriptor["availability"] != "available"
        and descriptor["parent_fingerprints"] != []
    ):
        raise ValueError(
            f"TaxaLens unavailable review/quality role {role!r} has lineage"
        )
    expected_maturity = (
        "human_reviewed_flickr_labels"
        if role == "quality_sidecar" and descriptor["availability"] == "available"
        else None
    )
    if descriptor["evidence_maturity_label"] != expected_maturity:
        raise ValueError(f"TaxaLens review/quality role {role!r} maturity differs")


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    return None if value is None else _nonnegative_int(value, field=field)


def _optional_positive_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number or null")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be a positive finite number or null")
    return normalized


def _optional_unit_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number within [0, 1] or null")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be a finite number within [0, 1] or null")
    return normalized


def _text_array(value: object, *, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    normalized = [_required_text(item, field=field) for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


__all__ = [
    "GEOGRAPHIC_CELLS_DOWNSTREAM_REASON",
    "TAXALENS_GEOGRAPHIC_CELLS_SCHEMA_VERSION",
    "TAXALENS_QUALITY_SIDECAR_SCHEMA_VERSION",
    "TAXALENS_REVIEW_QUALITY_ROLES",
    "TAXALENS_REVIEW_SAMPLING_SCHEMA",
    "TAXALENS_REVIEW_SAMPLING_SCHEMA_VERSION",
    "TaxaLensReviewQualityExport",
    "build_taxalens_quality_sidecar",
    "build_taxalens_review_sampling_frame",
    "export_taxalens_review_quality_evidence",
    "validate_taxalens_quality_sidecar",
    "validate_taxalens_review_quality_export",
    "validate_taxalens_review_sampling_frame",
]
