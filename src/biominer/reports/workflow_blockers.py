"""Resumable blocker reporting for adaptive reference workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.reports.evidence_maturity import (
    evidence_maturity_payload,
    validate_evidence_maturity_payload,
)
from biominer.storage.parquet import write_parquet


ADAPTIVE_WORKFLOW_BLOCKERS_FILE = "adaptive_workflow_blockers.parquet"
ADAPTIVE_WORKFLOW_BLOCKER_REPORT_FILE = "adaptive_workflow_blocker_report.json"
ADAPTIVE_WORKFLOW_BLOCKER_SUMMARY_FILE = "adaptive_workflow_blocker_report.md"
ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA_VERSION = "adaptive-workflow-blocker-v1.0.0"
ADAPTIVE_WORKFLOW_BLOCKER_REPORT_SCHEMA_VERSION = (
    "adaptive-workflow-blocker-report-v1.0.0"
)

WORKFLOW_BLOCKER_KINDS = (
    "failed_reference_downloads",
    "retryable_media",
    "invalid_routes",
    "stale_bank_artifacts",
    "incomplete_audit_sample",
    "pending_targeted_review",
    "pending_selective_rerun",
)
BLOCKER_MEASUREMENT_STATUSES = frozenset({"measured_complete", "unavailable"})
_BLOCKER_SEMANTICS: dict[str, tuple[str, bool]] = {
    "failed_reference_downloads": (
        "retry_download_or_record_terminal_failure",
        False,
    ),
    "retryable_media": ("retry_media", False),
    "invalid_routes": ("correct_route_or_exclude", True),
    "stale_bank_artifacts": ("rebuild_from_current_bank", False),
    "incomplete_audit_sample": ("collect_human_flickr_labels", True),
    "pending_targeted_review": ("review_flagged_references", True),
    "pending_selective_rerun": ("execute_selective_rescore_plan", False),
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "blocker_kind": pl.String,
    "blocker_order": pl.UInt8,
    "blocker_ids": pl.List(pl.String),
    "blocker_count": pl.UInt64,
    "measurement_status": pl.String,
    "resume_action": pl.String,
    "human_input_required": pl.Boolean,
    "blocks_initial_scoring": pl.Boolean,
    "blocks_final_release": pl.Boolean,
    "source_artifact_id": pl.String,
    "source_artifact_fingerprint": pl.String,
    "blocker_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class AdaptiveWorkflowBlockerReport:
    blockers: pl.DataFrame
    report: dict[str, object]
    markdown: str


def adaptive_workflow_blockers_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    supplied = {str(row.get("blocker_kind")): dict(row) for row in rows}
    if len(supplied) != len(rows):
        raise ValueError("adaptive workflow blockers repeat a kind")
    if set(supplied) != set(WORKFLOW_BLOCKER_KINDS):
        raise ValueError("adaptive workflow blockers must cover every kind")
    normalized: list[dict[str, object]] = []
    for order, kind in enumerate(WORKFLOW_BLOCKER_KINDS):
        source = supplied[kind]
        status = _required_text(
            source.get("measurement_status"), field="measurement_status"
        )
        if status not in BLOCKER_MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported blocker measurement status: {status}")
        ids = _canonical_ids(source.get("blocker_ids"), kind=kind)
        if status == "unavailable" and ids:
            raise ValueError("unavailable blocker kinds cannot claim blocker IDs")
        action, human = _BLOCKER_SEMANTICS[kind]
        row: dict[str, object] = {
            "schema_version": ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA_VERSION,
            "blocker_kind": kind,
            "blocker_order": order,
            "blocker_ids": ids,
            "blocker_count": len(ids) if status == "measured_complete" else None,
            "measurement_status": status,
            "resume_action": action,
            "human_input_required": human,
            "blocks_initial_scoring": _required_bool(
                source.get("blocks_initial_scoring"),
                field="blocks_initial_scoring",
            ),
            "blocks_final_release": _required_bool(
                source.get("blocks_final_release"),
                field="blocks_final_release",
            ),
            "source_artifact_id": _required_text(
                source.get("source_artifact_id"), field="source_artifact_id"
            ),
            "source_artifact_fingerprint": _sha256(
                source.get("source_artifact_fingerprint"),
                field="source_artifact_fingerprint",
            ),
            "blocker_fingerprint": "",
        }
        row["blocker_fingerprint"] = _fingerprint_without(
            row, "blocker_fingerprint"
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA,
        orient="row",
        strict=True,
    ).sort("blocker_order")
    validate_adaptive_workflow_blockers(frame)
    return frame


def validate_adaptive_workflow_blockers(frame: pl.DataFrame) -> None:
    if frame.schema != ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA:
        raise ValueError("adaptive workflow blocker schema mismatch")
    if frame["blocker_kind"].to_list() != list(WORKFLOW_BLOCKER_KINDS):
        raise ValueError("adaptive workflow blocker kinds are incomplete")
    if frame["blocker_order"].to_list() != list(range(len(WORKFLOW_BLOCKER_KINDS))):
        raise ValueError("adaptive workflow blocker order is invalid")
    by_kind = {
        str(row["blocker_kind"]): row for row in frame.iter_rows(named=True)
    }
    for kind, row in by_kind.items():
        if row["schema_version"] != ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive workflow blocker version")
        status = row["measurement_status"]
        ids = _canonical_ids(row["blocker_ids"], kind=kind)
        expected_count = len(ids) if status == "measured_complete" else None
        if status not in BLOCKER_MEASUREMENT_STATUSES:
            raise ValueError("adaptive blocker measurement status is invalid")
        if row["blocker_count"] != expected_count:
            raise ValueError("adaptive workflow blocker count mismatch")
        if status == "unavailable" and ids:
            raise ValueError("unavailable blocker kind claims blocker IDs")
        action, human = _BLOCKER_SEMANTICS[kind]
        if row["resume_action"] != action or row["human_input_required"] != human:
            raise ValueError("adaptive workflow blocker resume semantics mismatch")
        _required_text(row["source_artifact_id"], field="source_artifact_id")
        _sha256(
            row["source_artifact_fingerprint"],
            field="source_artifact_fingerprint",
        )
        if row["blocker_fingerprint"] != _fingerprint_without(
            row, "blocker_fingerprint"
        ):
            raise ValueError("adaptive workflow blocker fingerprint mismatch")
    failed = _measured_ids(by_kind["failed_reference_downloads"])
    retryable = _measured_ids(by_kind["retryable_media"])
    if failed is not None and retryable is not None and not retryable <= failed:
        raise ValueError("retryable media must be failed reference downloads")


def build_adaptive_workflow_blocker_report(
    blockers: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> AdaptiveWorkflowBlockerReport:
    validate_adaptive_workflow_blockers(blockers)
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    summaries = [_summary(row) for row in blockers.iter_rows(named=True)]
    report: dict[str, object] = {
        "schema_version": ADAPTIVE_WORKFLOW_BLOCKER_REPORT_SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(),
        "status": "complete",
        "blockers": summaries,
        "summary": _report_summary(summaries),
        "evidence_maturity": evidence_maturity_payload(),
        "provenance": {
            "blockers_fingerprint": _frame_fingerprint(blockers),
            "resume_policy": "select_only_named_blocker_ids",
        },
        "limitations": [
            "Unavailable blocker counts are not zero.",
            "A blocker count records pending work and does not prove that a prior scientific decision was wrong.",
            "Retry is selective: unrelated completed artifacts remain reusable.",
            "Human-input blockers cannot be cleared by an automated retry.",
        ],
        "report_fingerprint": "",
    }
    report["report_fingerprint"] = _fingerprint_without(
        report, "report_fingerprint"
    )
    result = AdaptiveWorkflowBlockerReport(
        blockers=blockers,
        report=report,
        markdown=_markdown(report),
    )
    validate_adaptive_workflow_blocker_report(result)
    return result


def validate_adaptive_workflow_blocker_report(
    result: AdaptiveWorkflowBlockerReport,
) -> None:
    validate_adaptive_workflow_blockers(result.blockers)
    report = result.report
    if report.get("schema_version") != ADAPTIVE_WORKFLOW_BLOCKER_REPORT_SCHEMA_VERSION:
        raise ValueError("adaptive workflow blocker report schema mismatch")
    expected = [_summary(row) for row in result.blockers.iter_rows(named=True)]
    if report.get("blockers") != expected:
        raise ValueError("adaptive workflow blocker report summaries mismatch")
    if report.get("summary") != _report_summary(expected):
        raise ValueError("adaptive workflow blocker report total mismatch")
    validate_evidence_maturity_payload(report.get("evidence_maturity"))
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "blockers_fingerprint"
    ) != _frame_fingerprint(result.blockers):
        raise ValueError("adaptive workflow blocker provenance mismatch")
    if report.get("report_fingerprint") != _fingerprint_without(
        report, "report_fingerprint"
    ):
        raise ValueError("adaptive workflow blocker report fingerprint mismatch")
    if not result.markdown.startswith("# Adaptive workflow blockers"):
        raise ValueError("adaptive workflow blocker Markdown mismatch")


def write_adaptive_workflow_blocker_report(
    result: AdaptiveWorkflowBlockerReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_adaptive_workflow_blocker_report(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "blockers": root / ADAPTIVE_WORKFLOW_BLOCKERS_FILE,
        "json": root / ADAPTIVE_WORKFLOW_BLOCKER_REPORT_FILE,
        "markdown": root / ADAPTIVE_WORKFLOW_BLOCKER_SUMMARY_FILE,
    }
    write_parquet(result.blockers, paths["blockers"])
    paths["json"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(result.markdown, encoding="utf-8")
    return paths


def _summary(row: Mapping[str, object]) -> dict[str, object]:
    return {
        field: row[field]
        for field in (
            "blocker_kind",
            "blocker_count",
            "measurement_status",
            "resume_action",
            "human_input_required",
            "blocks_initial_scoring",
            "blocks_final_release",
            "source_artifact_id",
            "source_artifact_fingerprint",
            "blocker_fingerprint",
        )
    }


def _report_summary(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    measured = [
        row
        for row in summaries
        if row["measurement_status"] == "measured_complete"
    ]
    return {
        "known_blocker_memberships": sum(
            int(row["blocker_count"]) for row in measured
        ),
        "kinds_with_known_blockers": [
            row["blocker_kind"]
            for row in measured
            if int(row["blocker_count"]) > 0
        ],
        "unavailable_kinds": [
            row["blocker_kind"]
            for row in summaries
            if row["measurement_status"] == "unavailable"
        ],
        "membership_semantics": "category_memberships_may_overlap",
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Adaptive workflow blockers",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Blocker | Count | Status | Resume action | Human input |",
        "|---|---:|---|---|---|",
    ]
    for row in report["blockers"]:  # type: ignore[union-attr]
        count = "unavailable" if row["blocker_count"] is None else row["blocker_count"]
        lines.append(
            f"| {row['blocker_kind']} | {count} | {row['measurement_status']} | "
            f"{row['resume_action']} | {row['human_input_required']} |"
        )
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Blockers fingerprint: `{report['provenance']['blockers_fingerprint']}`",  # type: ignore[index]
            f"- Report fingerprint: `{report['report_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _measured_ids(row: Mapping[str, object]) -> set[str] | None:
    if row["measurement_status"] != "measured_complete":
        return None
    return set(row["blocker_ids"])  # type: ignore[arg-type]


def _canonical_ids(value: object, *, kind: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{kind} blocker_ids must be a list")
    values = [_required_text(item, field="blocker_ids") for item in value]
    if values != sorted(set(values)):
        raise ValueError(f"{kind} blocker_ids must be sorted and unique")
    return values


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": frame.to_dicts(),
        }
    )


def _fingerprint_without(row: Mapping[str, object], field: str) -> str:
    payload = dict(row)
    payload.pop(field)
    return canonical_semantic_fingerprint(payload)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "")
    if not text or text != text.strip():
        raise ValueError(f"{field} must be canonical nonblank text")
    return text


def _required_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA",
    "ADAPTIVE_WORKFLOW_BLOCKERS_FILE",
    "ADAPTIVE_WORKFLOW_BLOCKER_REPORT_FILE",
    "ADAPTIVE_WORKFLOW_BLOCKER_SUMMARY_FILE",
    "AdaptiveWorkflowBlockerReport",
    "WORKFLOW_BLOCKER_KINDS",
    "adaptive_workflow_blockers_frame",
    "build_adaptive_workflow_blocker_report",
    "validate_adaptive_workflow_blocker_report",
    "validate_adaptive_workflow_blockers",
    "write_adaptive_workflow_blocker_report",
]
