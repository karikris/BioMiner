"""Artifact contract for species-level reference-bank quality audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import polars as pl


AUDIT_DIMENSIONS: tuple[str, ...] = (
    "target_species",
    "competitor_species",
    "region",
    "route",
    "life_stage",
    "visual_domain",
    "source_dataset",
    "admission_basis",
    "verification_basis",
)

REFERENCE_BANK_QUALITY_AUDIT_FILE = "reference_bank_quality_audit.parquet"
REFERENCE_BANK_QUALITY_SUMMARY_FILE = "reference_bank_quality_summary.parquet"
REFERENCE_BANK_QUALITY_POLICY_FILE = "reference_bank_quality_policy.json"
REFERENCE_BANK_QUALITY_REPORT_FILE = "reference_bank_quality_report.md"

REFERENCE_BANK_QUALITY_AUDIT_SCHEMA = {
    "audit_record_id": pl.String,
    **{dimension: pl.String for dimension in AUDIT_DIMENSIONS},
    "human_target_supported": pl.Boolean,
    "predicted_target": pl.Boolean,
    "provisional_margin": pl.Float64,
    "calibrated_probability": pl.Float64,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
}

REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA = {
    **{dimension: pl.String for dimension in AUDIT_DIMENSIONS},
    "reviewed_record_count": pl.UInt64,
    "metric_status": pl.String,
    "quality_approval_state": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReferenceBankQualityPolicy:
    schema_version: str = "reference-bank-quality-policy-v1.0.0"
    minimum_group_sample_size: int = 30
    confidence_level: float = 0.95
    require_sampling_weights_for_targeted_queues: bool = True
    calibrated_metrics_require_calibrated_probability: bool = True
    approval_requires_statistical_audit: bool = True

    def __post_init__(self) -> None:
        if self.minimum_group_sample_size < 1:
            raise ValueError("minimum_group_sample_size must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be within (0, 1)")


@dataclass(frozen=True, slots=True)
class ReferenceBankAuditPublication:
    audit_path: Path
    summary_path: Path
    policy_path: Path
    report_path: Path


def empty_reference_bank_quality_audit() -> pl.DataFrame:
    return pl.DataFrame(schema=REFERENCE_BANK_QUALITY_AUDIT_SCHEMA)


def empty_reference_bank_quality_summary() -> pl.DataFrame:
    return pl.DataFrame(schema=REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA)


def write_reference_bank_audit_contract(
    output_dir: str | Path,
    *,
    audit: pl.DataFrame,
    summary: pl.DataFrame,
    policy: ReferenceBankQualityPolicy | None = None,
) -> ReferenceBankAuditPublication:
    """Validate and publish the four required audit-contract artifacts."""

    _validate_schema(audit, REFERENCE_BANK_QUALITY_AUDIT_SCHEMA, artifact="audit")
    _validate_schema(
        summary,
        REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA,
        artifact="summary",
    )
    selected_policy = policy or ReferenceBankQualityPolicy()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / REFERENCE_BANK_QUALITY_AUDIT_FILE
    summary_path = root / REFERENCE_BANK_QUALITY_SUMMARY_FILE
    policy_path = root / REFERENCE_BANK_QUALITY_POLICY_FILE
    report_path = root / REFERENCE_BANK_QUALITY_REPORT_FILE
    audit.write_parquet(audit_path)
    summary.write_parquet(summary_path)
    policy_path.write_text(
        json.dumps(asdict(selected_policy), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _audit_contract_report(audit, summary, selected_policy),
        encoding="utf-8",
    )
    return ReferenceBankAuditPublication(
        audit_path=audit_path,
        summary_path=summary_path,
        policy_path=policy_path,
        report_path=report_path,
    )


def _validate_schema(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
    *,
    artifact: str,
) -> None:
    missing = sorted(set(schema) - set(frame.columns))
    if missing:
        raise ValueError(f"reference-bank {artifact} missing columns: {missing}")
    mismatched = {
        column: (frame.schema[column], dtype)
        for column, dtype in schema.items()
        if frame.schema[column] != dtype
    }
    if mismatched:
        raise ValueError(f"reference-bank {artifact} schema mismatch: {mismatched}")


def _audit_contract_report(
    audit: pl.DataFrame,
    summary: pl.DataFrame,
    policy: ReferenceBankQualityPolicy,
) -> str:
    dimensions = "\n".join(f"- `{dimension}`" for dimension in AUDIT_DIMENSIONS)
    return (
        "# Reference-bank quality audit\n\n"
        f"Audit rows: {audit.height}\n\nSummary rows: {summary.height}\n\n"
        "## Required dimensions\n\n"
        f"{dimensions}\n\n"
        "## Policy boundary\n\n"
        f"Minimum group sample: {policy.minimum_group_sample_size}. "
        "Targeted queues require sampling weights. Calibrated metrics require "
        "calibrated probabilities. Empty or underpowered groups are not quality "
        "approval evidence.\n"
    )


__all__ = [
    "AUDIT_DIMENSIONS",
    "REFERENCE_BANK_QUALITY_AUDIT_FILE",
    "REFERENCE_BANK_QUALITY_REPORT_FILE",
    "REFERENCE_BANK_QUALITY_SUMMARY_FILE",
    "REFERENCE_BANK_QUALITY_POLICY_FILE",
    "ReferenceBankAuditPublication",
    "ReferenceBankQualityPolicy",
    "empty_reference_bank_quality_audit",
    "empty_reference_bank_quality_summary",
    "write_reference_bank_audit_contract",
]
