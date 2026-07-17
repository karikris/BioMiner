"""Measured admission-funnel reporting for adaptive reference support."""

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


ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_FILE = (
    "adaptive_reference_admission_funnel.parquet"
)
ADAPTIVE_REFERENCE_ADMISSION_REPORT_FILE = (
    "adaptive_reference_admission_report.json"
)
ADAPTIVE_REFERENCE_ADMISSION_SUMMARY_FILE = (
    "adaptive_reference_admission_report.md"
)
ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA_VERSION = (
    "adaptive-reference-admission-funnel-v1.0.0"
)
ADAPTIVE_REFERENCE_ADMISSION_REPORT_SCHEMA_VERSION = (
    "adaptive-reference-admission-report-v1.0.0"
)

REFERENCE_ADMISSION_STAGES = (
    "candidates",
    "downloaded",
    "decoded",
    "deduplicated",
    "yoloe_routed",
    "provisionally_admitted",
    "human_verified",
    "excluded",
    "flagged",
    "reviewed_later",
)
MEASUREMENT_STATUSES = frozenset({"measured_complete", "unavailable"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "stage": pl.String,
    "stage_order": pl.UInt8,
    "reference_media_ids": pl.List(pl.String),
    "record_count": pl.UInt64,
    "candidate_retention_rate": pl.Float64,
    "measurement_status": pl.String,
    "source_artifact_id": pl.String,
    "source_artifact_fingerprint": pl.String,
    "stage_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class AdaptiveReferenceAdmissionReport:
    funnel: pl.DataFrame
    report: dict[str, object]
    markdown: str


def adaptive_reference_admission_funnel_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Normalize one complete, source-bound membership snapshot per stage."""

    supplied = {str(row.get("stage")): dict(row) for row in rows}
    if len(supplied) != len(rows):
        raise ValueError("adaptive admission funnel repeats a stage")
    if set(supplied) != set(REFERENCE_ADMISSION_STAGES):
        raise ValueError("adaptive admission funnel must contain every stage exactly once")
    candidate_ids: list[str] | None = None
    normalized: list[dict[str, object]] = []
    for order, stage in enumerate(REFERENCE_ADMISSION_STAGES):
        row = supplied[stage]
        status = _required_text(
            row.get("measurement_status"), field="measurement_status"
        )
        if status not in MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported funnel measurement status: {status}")
        media_ids = _canonical_ids(row.get("reference_media_ids"), stage=stage)
        if status == "unavailable" and media_ids:
            raise ValueError("unavailable funnel stages cannot claim record identities")
        if stage == "candidates" and status == "measured_complete":
            candidate_ids = media_ids
        count = len(media_ids) if status == "measured_complete" else None
        retention = (
            None
            if count is None or candidate_ids is None or not candidate_ids
            else count / len(candidate_ids)
        )
        item: dict[str, object] = {
            "schema_version": ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA_VERSION,
            "stage": stage,
            "stage_order": order,
            "reference_media_ids": media_ids,
            "record_count": count,
            "candidate_retention_rate": retention,
            "measurement_status": status,
            "source_artifact_id": _required_text(
                row.get("source_artifact_id"), field="source_artifact_id"
            ),
            "source_artifact_fingerprint": _sha256(
                row.get("source_artifact_fingerprint"),
                field="source_artifact_fingerprint",
            ),
            "stage_fingerprint": "",
        }
        item["stage_fingerprint"] = _fingerprint_without(
            item, "stage_fingerprint"
        )
        normalized.append(item)
    frame = pl.DataFrame(
        normalized,
        schema=ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA,
        orient="row",
        strict=True,
    ).sort("stage_order")
    validate_adaptive_reference_admission_funnel(frame)
    return frame


def validate_adaptive_reference_admission_funnel(frame: pl.DataFrame) -> None:
    if frame.schema != ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA:
        raise ValueError("adaptive reference admission funnel schema mismatch")
    if frame["stage"].to_list() != list(REFERENCE_ADMISSION_STAGES):
        raise ValueError("adaptive reference admission funnel stages are incomplete")
    if frame["stage_order"].to_list() != list(range(len(REFERENCE_ADMISSION_STAGES))):
        raise ValueError("adaptive reference admission stage order is invalid")
    by_stage = {
        str(row["stage"]): row for row in frame.iter_rows(named=True)
    }
    for row in by_stage.values():
        if row["schema_version"] != ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive admission funnel version")
        status = row["measurement_status"]
        if status not in MEASUREMENT_STATUSES:
            raise ValueError("adaptive admission measurement status is invalid")
        ids = _canonical_ids(row["reference_media_ids"], stage=str(row["stage"]))
        expected_count = len(ids) if status == "measured_complete" else None
        if row["record_count"] != expected_count:
            raise ValueError("adaptive admission funnel count mismatch")
        if status == "unavailable" and ids:
            raise ValueError("unavailable funnel stages claim record identities")
        _required_text(row["source_artifact_id"], field="source_artifact_id")
        _sha256(
            row["source_artifact_fingerprint"],
            field="source_artifact_fingerprint",
        )
        if row["stage_fingerprint"] != _fingerprint_without(
            row, "stage_fingerprint"
        ):
            raise ValueError("adaptive admission stage fingerprint mismatch")
    candidates = _measured_ids(by_stage["candidates"])
    expected_retention_denominator = len(candidates) if candidates else None
    for row in by_stage.values():
        ids = _measured_ids(row)
        expected_rate = (
            None
            if ids is None or expected_retention_denominator is None
            else len(ids) / expected_retention_denominator
        )
        if row["candidate_retention_rate"] != expected_rate:
            raise ValueError("adaptive admission retention rate mismatch")
        if candidates is not None and ids is not None and not ids <= candidates:
            raise ValueError("adaptive admission stage contains a non-candidate")
    _require_subset(by_stage, "downloaded", "candidates")
    _require_subset(by_stage, "decoded", "downloaded")
    _require_subset(by_stage, "deduplicated", "decoded")
    _require_subset(by_stage, "yoloe_routed", "deduplicated")
    _require_subset(by_stage, "provisionally_admitted", "yoloe_routed")
    _require_subset(by_stage, "flagged", "provisionally_admitted")
    _require_subset(by_stage, "reviewed_later", "flagged")


def build_adaptive_reference_admission_report(
    funnel: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> AdaptiveReferenceAdmissionReport:
    validate_adaptive_reference_admission_funnel(funnel)
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    stages = _stage_summaries(funnel)
    report: dict[str, object] = {
        "schema_version": ADAPTIVE_REFERENCE_ADMISSION_REPORT_SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(),
        "status": "complete",
        "stages": stages,
        "measurement_summary": {
            "measured_stage_count": sum(
                row["measurement_status"] == "measured_complete"
                for row in stages
            ),
            "unavailable_stages": [
                row["stage"]
                for row in stages
                if row["measurement_status"] == "unavailable"
            ],
        },
        "evidence_maturity": evidence_maturity_payload(),
        "provenance": {
            "funnel_fingerprint": _frame_fingerprint(funnel),
            "count_derivation": "unique_sorted_reference_media_id_membership",
        },
        "limitations": [
            "Counts describe supplied fingerprinted stage snapshots and are not inferred from missing artifacts.",
            "Provider-asserted provisional admission is not human verification.",
            "YOLOE routing is visual-domain evidence and does not establish species identity.",
            "Flagged and reviewed-later stages are historical workflow states and may overlap later verified or excluded outcomes.",
        ],
        "report_fingerprint": "",
    }
    report["report_fingerprint"] = _fingerprint_without(
        report, "report_fingerprint"
    )
    result = AdaptiveReferenceAdmissionReport(
        funnel=funnel,
        report=report,
        markdown=_markdown(report),
    )
    validate_adaptive_reference_admission_report(result)
    return result


def validate_adaptive_reference_admission_report(
    result: AdaptiveReferenceAdmissionReport,
) -> None:
    validate_adaptive_reference_admission_funnel(result.funnel)
    report = result.report
    if report.get("schema_version") != ADAPTIVE_REFERENCE_ADMISSION_REPORT_SCHEMA_VERSION:
        raise ValueError("adaptive reference admission report schema mismatch")
    expected_stages = _stage_summaries(result.funnel)
    if report.get("stages") != expected_stages:
        raise ValueError("adaptive reference admission report stages mismatch")
    expected_measurements = {
        "measured_stage_count": sum(
            row["measurement_status"] == "measured_complete"
            for row in expected_stages
        ),
        "unavailable_stages": [
            row["stage"]
            for row in expected_stages
            if row["measurement_status"] == "unavailable"
        ],
    }
    if report.get("measurement_summary") != expected_measurements:
        raise ValueError("adaptive admission measurement summary mismatch")
    validate_evidence_maturity_payload(report.get("evidence_maturity"))
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "funnel_fingerprint"
    ) != _frame_fingerprint(result.funnel):
        raise ValueError("adaptive admission report provenance mismatch")
    if report.get("report_fingerprint") != _fingerprint_without(
        report, "report_fingerprint"
    ):
        raise ValueError("adaptive reference admission report fingerprint mismatch")
    if not result.markdown.startswith("# Adaptive reference admission"):
        raise ValueError("adaptive reference admission Markdown mismatch")


def write_adaptive_reference_admission_report(
    result: AdaptiveReferenceAdmissionReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_adaptive_reference_admission_report(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "funnel": root / ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_FILE,
        "json": root / ADAPTIVE_REFERENCE_ADMISSION_REPORT_FILE,
        "markdown": root / ADAPTIVE_REFERENCE_ADMISSION_SUMMARY_FILE,
    }
    write_parquet(result.funnel, paths["funnel"])
    paths["json"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(result.markdown, encoding="utf-8")
    return paths


def _require_subset(
    by_stage: Mapping[str, Mapping[str, object]],
    child: str,
    parent: str,
) -> None:
    child_ids = _measured_ids(by_stage[child])
    parent_ids = _measured_ids(by_stage[parent])
    if child_ids is not None and parent_ids is not None and not child_ids <= parent_ids:
        raise ValueError(f"adaptive admission {child} is not a subset of {parent}")


def _stage_summaries(funnel: pl.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "stage": row["stage"],
            "record_count": row["record_count"],
            "candidate_retention_rate": row["candidate_retention_rate"],
            "measurement_status": row["measurement_status"],
            "source_artifact_id": row["source_artifact_id"],
            "source_artifact_fingerprint": row[
                "source_artifact_fingerprint"
            ],
            "stage_fingerprint": row["stage_fingerprint"],
        }
        for row in funnel.iter_rows(named=True)
    ]


def _measured_ids(row: Mapping[str, object]) -> set[str] | None:
    if row["measurement_status"] != "measured_complete":
        return None
    return set(row["reference_media_ids"])  # type: ignore[arg-type]


def _canonical_ids(value: object, *, stage: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{stage} reference_media_ids must be a list")
    ids = [_required_text(item, field="reference_media_ids") for item in value]
    if ids != sorted(set(ids)):
        raise ValueError(f"{stage} reference_media_ids must be sorted and unique")
    return ids


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Adaptive reference admission",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Stage | Count | Candidate retention | Status |",
        "|---|---:|---:|---|",
    ]
    for row in report["stages"]:  # type: ignore[union-attr]
        count = "unavailable" if row["record_count"] is None else str(row["record_count"])
        rate = (
            "unavailable"
            if row["candidate_retention_rate"] is None
            else f"{100.0 * row['candidate_retention_rate']:.2f}%"
        )
        lines.append(
            f"| {row['stage']} | {count} | {rate} | {row['measurement_status']} |"
        )
    lines.extend(["", "## Scientific boundaries", ""])
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Funnel fingerprint: `{report['provenance']['funnel_fingerprint']}`",  # type: ignore[index]
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


def _fingerprint_without(row: Mapping[str, object], field: str) -> str:
    payload = dict(row)
    payload.pop(field)
    return canonical_semantic_fingerprint(payload)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "")
    if not text or text != text.strip():
        raise ValueError(f"{field} must be canonical nonblank text")
    return text


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
    "ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_FILE",
    "ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA",
    "ADAPTIVE_REFERENCE_ADMISSION_REPORT_FILE",
    "ADAPTIVE_REFERENCE_ADMISSION_SUMMARY_FILE",
    "AdaptiveReferenceAdmissionReport",
    "REFERENCE_ADMISSION_STAGES",
    "adaptive_reference_admission_funnel_frame",
    "build_adaptive_reference_admission_report",
    "validate_adaptive_reference_admission_funnel",
    "validate_adaptive_reference_admission_report",
    "write_adaptive_reference_admission_report",
]
