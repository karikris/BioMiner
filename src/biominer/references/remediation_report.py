"""Evidence-bound before/after report for targeted reference remediation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.adaptive_bank_revision import (
    AdaptiveSupportBankRevision,
    validate_adaptive_support_bank_revision,
)
from biominer.references.targeted_review_decisions import (
    TargetedReferenceReviewResult,
    validate_targeted_reference_review_result,
)
from biominer.reports.evidence_maturity import (
    evidence_maturity_payload,
    validate_evidence_maturity_payload,
)


REFERENCE_REMEDIATION_REPORT_FILE = "reference_remediation_report.json"
REFERENCE_REMEDIATION_SUMMARY_FILE = "reference_remediation_report.md"
REFERENCE_REMEDIATION_REPORT_SCHEMA_VERSION = (
    "reference-remediation-report-v1.0.0"
)
REFERENCE_REMEDIATION_IMPACT_SCHEMA_VERSION = (
    "reference-remediation-impact-estimate-v1.0.0"
)
IMPACT_ESTIMATE_BASES = frozenset(
    {
        "exact_dependency_index",
        "upper_bound",
        "sample_weighted_estimate",
        "unavailable",
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


REFERENCE_REMEDIATION_IMPACT_SCHEMA = {
    "schema_version": pl.String,
    "impact_scope_id": pl.String,
    "invalidated_artifact_id": pl.String,
    "species": pl.String,
    "route": pl.String,
    "expected_impacted_record_count": pl.UInt64,
    "estimate_basis": pl.String,
    "source_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReferenceRemediationReport:
    report: dict[str, object]
    markdown: str


def reference_remediation_impact_estimates_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized = []
    for source in rows or []:
        row = dict(source)
        row.setdefault(
            "schema_version",
            REFERENCE_REMEDIATION_IMPACT_SCHEMA_VERSION,
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REFERENCE_REMEDIATION_IMPACT_SCHEMA,
        orient="row",
        strict=True,
    ).sort("impact_scope_id")
    if frame["impact_scope_id"].n_unique() != frame.height:
        raise ValueError("impact estimates repeat an impact scope")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_REMEDIATION_IMPACT_SCHEMA_VERSION:
            raise ValueError("unsupported remediation impact schema version")
        for field in (
            "impact_scope_id",
            "invalidated_artifact_id",
            "species",
            "route",
        ):
            _required_text(row[field], field=field)
        basis = _required_text(row["estimate_basis"], field="estimate_basis")
        if basis not in IMPACT_ESTIMATE_BASES:
            raise ValueError(f"unsupported impact estimate basis: {basis}")
        count = row["expected_impacted_record_count"]
        if (basis == "unavailable") != (count is None):
            raise ValueError(
                "unavailable impact estimates require a null count and measured "
                "estimates require a count"
            )
        _full_sha256(row["source_fingerprint"], field="source_fingerprint")
    return frame


def build_reference_remediation_report(
    review: TargetedReferenceReviewResult,
    revision: AdaptiveSupportBankRevision,
    impact_estimates: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> ReferenceRemediationReport:
    """Summarize measured remediation evidence without inventing impacts."""

    validate_targeted_reference_review_result(review)
    validate_adaptive_support_bank_revision(revision)
    impacts = reference_remediation_impact_estimates_frame(
        impact_estimates.to_dicts()
    )
    invalidated_artifacts = set(
        revision.invalidation_manifest.filter(pl.col("invalidated"))[
            "artifact_id"
        ]
    )
    unknown_impact_artifacts = sorted(
        set(impacts["invalidated_artifact_id"]) - invalidated_artifacts
    )
    if unknown_impact_artifacts:
        raise ValueError(
            "impact estimates reference artifacts not invalidated by the revision: "
            + ", ".join(unknown_impact_artifacts)
        )
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    changes = revision.change_manifest
    outcomes_by_id = {
        str(row["reference_media_id"]): row
        for row in review.workflow.outcomes.iter_rows(named=True)
    }
    species_summary = _species_summary(review, changes, outcomes_by_id)
    prototype_invalidations = [
        {
            "artifact_id": row["artifact_id"],
            "artifact_type": row["artifact_type"],
            "status": "invalidated_rebuild_required_not_measured",
            "affected_reference_media_ids": row[
                "affected_reference_media_ids"
            ],
            "change_types": row["change_types"],
        }
        for row in revision.invalidation_manifest.iter_rows(named=True)
        if row["invalidated"] and "prototype" in str(row["artifact_type"])
    ]
    impact_summary = _impact_summary(impacts)
    counts = {
        "species_flagged": len(species_summary),
        "references_targeted": review.targeted_queue.height,
        "references_reviewed": sum(
            bool(row["effective_decision_ids"])
            for row in outcomes_by_id.values()
        ),
        "references_verified": changes.filter(
            pl.col("change_type") == "promoted_verified"
        ).height,
        "references_excluded": changes.filter(
            pl.col("change_type") == "excluded_after_review"
        ).height,
        "unchanged_provisional_references": changes.filter(
            pl.col("change_type") == "unchanged_provisional"
        ).height,
        "flagged_review_pending": changes.filter(
            pl.col("change_type") == "flagged_review_pending"
        ).height,
        "prototype_artifacts_invalidated": len(prototype_invalidations),
    }
    report: dict[str, object] = {
        "schema_version": REFERENCE_REMEDIATION_REPORT_SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(),
        "status": "complete",
        "evidence_maturity": evidence_maturity_payload(),
        "before": {
            "reference_bank_version": revision.old_reference_bank_version,
            "reference_bank_fingerprint": (
                revision.old_reference_bank_fingerprint
            ),
            "support_manifest_fingerprint": (
                revision.old_support_manifest_fingerprint
            ),
            "support_eligible_references": int(
                changes["old_support_eligible"].sum()
            ),
            "provisional_references": int(
                changes["old_provisional_support"].sum()
            ),
        },
        "after": {
            "reference_bank_version": revision.new_reference_bank_version,
            "reference_bank_fingerprint": (
                revision.new_reference_bank_fingerprint
            ),
            "support_manifest_fingerprint": (
                revision.new_support_manifest_fingerprint
            ),
            "support_eligible_references": int(
                changes["new_support_eligible"].sum()
            ),
            "provisional_references": int(
                changes["new_provisional_support"].sum()
            ),
        },
        "counts": counts,
        "species_flagged": species_summary,
        "prototype_changes": {
            "observation_status": "not_measured_rebuild_required",
            "observed_change_count": None,
            "invalidated_artifacts": prototype_invalidations,
        },
        "expected_impacted_records": impact_summary,
        "provenance": {
            "revision_fingerprint": revision.revision_fingerprint,
            "targeting_fingerprints": sorted(
                review.targeted_queue["targeting_fingerprint"].to_list()
            ),
            "decision_binding_fingerprints": sorted(
                review.decision_bindings["binding_fingerprint"].to_list()
            ),
            "impact_estimates_fingerprint": _frame_fingerprint(impacts),
        },
        "limitations": [
            "Statistical flags prioritize human review and are not evidence of taxonomic misidentification.",
            "Prototype invalidation means a rebuild is required; this report does not claim an observed prototype change.",
            "Expected impacted-record counts are reported only from supplied fingerprinted estimates and are not measured rerun outcomes.",
            "Unreviewed targeted references remain quarantined until the strict review workflow resolves them.",
        ],
        "report_fingerprint": "",
    }
    payload = dict(report)
    payload.pop("report_fingerprint")
    report["report_fingerprint"] = canonical_semantic_fingerprint(payload)
    result = ReferenceRemediationReport(
        report=report,
        markdown=_markdown(report),
    )
    validate_reference_remediation_report(result)
    return result


def validate_reference_remediation_report(
    result: ReferenceRemediationReport,
) -> None:
    report = result.report
    if (
        report.get("schema_version")
        != REFERENCE_REMEDIATION_REPORT_SCHEMA_VERSION
        or report.get("status") != "complete"
    ):
        raise ValueError("reference remediation report identity is invalid")
    validate_evidence_maturity_payload(report.get("evidence_maturity"))
    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("reference remediation report fingerprint mismatch")
    if not result.markdown.startswith("# Targeted reference remediation"):
        raise ValueError("reference remediation Markdown identity is invalid")


def write_reference_remediation_report(
    result: ReferenceRemediationReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_reference_remediation_report(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / REFERENCE_REMEDIATION_REPORT_FILE
    markdown_path = root / REFERENCE_REMEDIATION_SUMMARY_FILE
    json_path.write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(result.markdown, encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _species_summary(
    review: TargetedReferenceReviewResult,
    changes: pl.DataFrame,
    outcomes_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    change_by_id = {
        str(row["reference_media_id"]): row
        for row in changes.iter_rows(named=True)
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in review.targeted_queue.iter_rows(named=True):
        grouped.setdefault(str(row["scientific_name"]), []).append(row)
    summaries = []
    for species, rows in sorted(grouped.items()):
        media_ids = sorted(str(row["reference_media_id"]) for row in rows)
        species_changes = [change_by_id[media_id] for media_id in media_ids]
        summaries.append(
            {
                "species": species,
                "routes": sorted(
                    {
                        str(context["route"])
                        for row in rows
                        for context in row["flagged_contexts"]
                    }
                ),
                "flag_reasons": sorted(
                    {
                        str(reason)
                        for row in rows
                        for reason in row["flag_reasons"]
                    }
                ),
                "references_targeted": len(media_ids),
                "references_reviewed": sum(
                    bool(outcomes_by_id[media_id]["effective_decision_ids"])
                    for media_id in media_ids
                ),
                "references_verified": sum(
                    row["change_type"] == "promoted_verified"
                    for row in species_changes
                ),
                "references_excluded": sum(
                    row["change_type"] == "excluded_after_review"
                    for row in species_changes
                ),
                "references_pending": sum(
                    row["change_type"] == "flagged_review_pending"
                    for row in species_changes
                ),
            }
        )
    return summaries


def _impact_summary(impacts: pl.DataFrame) -> dict[str, object]:
    available = impacts.filter(
        pl.col("expected_impacted_record_count").is_not_null()
    )
    if impacts.is_empty() or available.is_empty():
        availability = "unavailable"
    elif available.height == impacts.height:
        availability = "complete"
    else:
        availability = "partial"
    groups = []
    for key, group in impacts.group_by("species", "route", maintain_order=True):
        species, route = key
        measured = group.filter(
            pl.col("expected_impacted_record_count").is_not_null()
        )
        groups.append(
            {
                "species": species,
                "route": route,
                "estimate_scope_count": group.height,
                "availability": (
                    "complete"
                    if measured.height == group.height
                    else "unavailable"
                    if measured.is_empty()
                    else "partial"
                ),
                "expected_impacted_record_count": (
                    int(measured["expected_impacted_record_count"].sum())
                    if measured.height == group.height and not measured.is_empty()
                    else None
                ),
                "estimate_bases": sorted(set(group["estimate_basis"])),
            }
        )
    return {
        "availability": availability,
        "total_expected_impacted_records": (
            int(available["expected_impacted_record_count"].sum())
            if availability == "complete"
            else None
        ),
        "available_estimate_sum": (
            int(available["expected_impacted_record_count"].sum())
            if not available.is_empty()
            else None
        ),
        "groups": sorted(groups, key=lambda row: (row["species"], row["route"])),
    }


def _markdown(report: Mapping[str, object]) -> str:
    counts = report["counts"]
    before = report["before"]
    after = report["after"]
    impacts = report["expected_impacted_records"]
    prototypes = report["prototype_changes"]
    assert isinstance(counts, Mapping)
    assert isinstance(before, Mapping)
    assert isinstance(after, Mapping)
    assert isinstance(impacts, Mapping)
    assert isinstance(prototypes, Mapping)
    lines = [
        "# Targeted reference remediation",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Before / after",
        "",
        f"- Bank: `{before['reference_bank_version']}` → `{after['reference_bank_version']}`",
        f"- Support-eligible references: {before['support_eligible_references']} → {after['support_eligible_references']}",
        f"- Provisional references: {before['provisional_references']} → {after['provisional_references']}",
        "",
        "## Remediation counts",
        "",
        f"- Species flagged: {counts['species_flagged']}",
        f"- References targeted: {counts['references_targeted']}",
        f"- References reviewed: {counts['references_reviewed']}",
        f"- References verified: {counts['references_verified']}",
        f"- References excluded: {counts['references_excluded']}",
        f"- Unchanged provisional references: {counts['unchanged_provisional_references']}",
        f"- Flagged references still pending: {counts['flagged_review_pending']}",
        "",
        "## Prototype status",
        "",
        f"- Observation status: `{prototypes['observation_status']}`",
        f"- Prototype artifacts invalidated: {counts['prototype_artifacts_invalidated']}",
        "- No observed before/after prototype change is claimed until selective rebuild completes.",
        "",
        "## Expected impacted records",
        "",
        f"- Availability: `{impacts['availability']}`",
        f"- Total: `{impacts['total_expected_impacted_records']}`",
        "",
        "## Scientific boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Revision fingerprint: `{report['provenance']['revision_fingerprint']}`",  # type: ignore[index]
            f"- Report fingerprint: `{report['report_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": frame.to_dicts(),
        }
    )


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value.astimezone(UTC)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "IMPACT_ESTIMATE_BASES",
    "REFERENCE_REMEDIATION_IMPACT_SCHEMA",
    "REFERENCE_REMEDIATION_IMPACT_SCHEMA_VERSION",
    "REFERENCE_REMEDIATION_REPORT_FILE",
    "REFERENCE_REMEDIATION_REPORT_SCHEMA_VERSION",
    "REFERENCE_REMEDIATION_SUMMARY_FILE",
    "ReferenceRemediationReport",
    "build_reference_remediation_report",
    "reference_remediation_impact_estimates_frame",
    "validate_reference_remediation_report",
    "write_reference_remediation_report",
]
