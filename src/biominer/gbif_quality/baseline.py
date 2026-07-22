from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_quality.funnel import SourceFunnel
from biominer.gbif_quality.inventory import SourceInventory
from biominer.gbif_quality.policy import FieldPolicy, field_policy_table
from biominer.gbif_quality.profile import CompletenessProfile
from biominer.gbif_quality.schema_audit import SchemaAudit


BASELINE_SCHEMA_VERSION = "biominer-gbif-media-quality-baseline/v1"
SUPPLIED_BASELINE = {
    "raw_occurrence_rows": 75_352_491,
    "raw_multimedia_rows": 18_680_565,
    "v3_media_rows": 16_612_063,
    "v3_occurrences": 11_569_412,
    "v3_columns": 114,
}


@dataclass(frozen=True, slots=True)
class BaselinePublication:
    schema_version: str
    data_root: Path
    report_root: Path
    data_manifest: dict[str, Any]
    report_manifest: dict[str, Any]


def publish_baseline(
    *,
    inventory: SourceInventory,
    funnel: SourceFunnel,
    schema_audit: SchemaAudit,
    policies: Iterable[FieldPolicy],
    completeness: CompletenessProfile,
    data_root: str | Path,
    report_root: str | Path,
    code_commit: str,
    generated_at: str | None = None,
) -> BaselinePublication:
    """Create an immutable Phase 1 publication and write both manifests last."""

    data_destination = Path(data_root).resolve()
    report_destination = Path(report_root).resolve()
    if data_destination.exists():
        raise FileExistsError(data_destination)
    if report_destination.exists():
        raise FileExistsError(report_destination)
    if not code_commit.strip():
        raise ValueError("code_commit is required")
    policy_rows = tuple(policies)
    _validate_evidence(inventory, funnel, schema_audit, policy_rows, completeness)
    timestamp = generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    data_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    data_staging = data_destination.parent / (
        f".{data_destination.name}.{uuid4().hex}.staging"
    )
    report_staging = report_destination.parent / (
        f".{report_destination.name}.{uuid4().hex}.staging"
    )
    data_staging.mkdir()
    report_staging.mkdir()
    data_published = False
    try:
        baseline = _baseline_summary(
            inventory, funnel, schema_audit, completeness, code_commit, timestamp
        )
        _write_parquet(inventory.artifact_table(), data_staging / "source_inventory.parquet")
        _write_json(inventory.to_summary(), data_staging / "source_inventory.json")
        _write_parquet(funnel.stage_table(), data_staging / "source_funnel.parquet")
        _write_json(funnel.to_summary(), data_staging / "source_funnel.json")
        _write_parquet(schema_audit.column_table(), data_staging / "schema_inventory.parquet")
        _write_parquet(schema_audit.row_group_table(), data_staging / "row_group_inventory.parquet")
        _write_parquet(field_policy_table(policy_rows), data_staging / "column_policy.parquet")
        _write_parquet(completeness.table(), data_staging / "completeness_by_applicability.parquet")
        _write_json(baseline, data_staging / "baseline.json")

        _write_text(_source_funnel_markdown(inventory, funnel), report_staging / "source_funnel.md")
        _write_text(_schema_markdown(schema_audit), report_staging / "schema_and_integrity.md")
        _write_text(
            _completeness_markdown(completeness),
            report_staging / "completeness_by_applicability.md",
        )

        data_inventory = _artifact_inventory(data_staging)
        report_inventory = _artifact_inventory(report_staging)
        data_manifest = _manifest(
            timestamp=timestamp,
            code_commit=code_commit,
            source_snapshot_id=inventory.source_snapshot_id,
            artifact_inventory=data_inventory,
            baseline=baseline,
            publication_role="phase_1_data",
        )
        report_manifest = _manifest(
            timestamp=timestamp,
            code_commit=code_commit,
            source_snapshot_id=inventory.source_snapshot_id,
            artifact_inventory=report_inventory,
            baseline=baseline,
            publication_role="phase_1_reports",
        )
        _write_json(data_manifest, data_staging / "manifest.json")
        _write_json(report_manifest, report_staging / "manifest.json")
        _verify_publication(data_staging, data_manifest)
        _verify_publication(report_staging, report_manifest)
        os.replace(data_staging, data_destination)
        data_published = True
        os.replace(report_staging, report_destination)
    except Exception:
        shutil.rmtree(data_staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        if data_published and not report_destination.exists():
            shutil.rmtree(data_destination, ignore_errors=True)
        raise
    return BaselinePublication(
        schema_version=BASELINE_SCHEMA_VERSION,
        data_root=data_destination,
        report_root=report_destination,
        data_manifest=data_manifest,
        report_manifest=report_manifest,
    )


def _validate_evidence(
    inventory: SourceInventory,
    funnel: SourceFunnel,
    schema_audit: SchemaAudit,
    policies: tuple[FieldPolicy, ...],
    completeness: CompletenessProfile,
) -> None:
    checks = {
        "inventory": all(inventory.validation.values()),
        "funnel": all(funnel.validation.values()),
        "schema_audit": all(schema_audit.validation.values()),
        "completeness": all(completeness.validation.values()),
        "policy_coverage": len(policies) == schema_audit.counts["columns"],
        "profile_column_coverage": (
            completeness.denominators["columns"] == schema_audit.counts["columns"]
        ),
        "v3_media_denominator": (
            completeness.denominators["media_rows"] == funnel.counts["v3_media_rows"]
        ),
        "v3_occurrence_denominator": (
            completeness.denominators["distinct_occurrences"]
            == funnel.counts["v3_occurrences"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"baseline evidence gate failed: {checks}")


def _baseline_summary(
    inventory: SourceInventory,
    funnel: SourceFunnel,
    schema_audit: SchemaAudit,
    completeness: CompletenessProfile,
    code_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    observed = {
        "raw_occurrence_rows": funnel.counts["raw_occurrence_rows"],
        "raw_multimedia_rows": funnel.counts["raw_multimedia_rows"],
        "v3_media_rows": completeness.denominators["media_rows"],
        "v3_occurrences": completeness.denominators["distinct_occurrences"],
        "v3_columns": completeness.denominators["columns"],
    }
    comparisons = {
        key: {
            "supplied": expected,
            "observed": observed[key],
            "difference": observed[key] - expected,
            "status": "MATCH" if observed[key] == expected else "DIFFERS",
        }
        for key, expected in SUPPLIED_BASELINE.items()
    }
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "source_snapshot_id": inventory.source_snapshot_id,
        "observed": observed,
        "supplied_baseline_comparison": comparisons,
        "schema_fingerprint": schema_audit.schema_fingerprint,
        "validation": {
            "source_inventory": inventory.validation,
            "source_funnel": funnel.validation,
            "schema_audit": schema_audit.validation,
            "completeness": completeness.validation,
        },
    }


def _manifest(
    *,
    timestamp: str,
    code_commit: str,
    source_snapshot_id: str,
    artifact_inventory: list[dict[str, Any]],
    baseline: dict[str, Any],
    publication_role: str,
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "publication_role": publication_role,
        "generated_at": timestamp,
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "semantic_fingerprint": canonical_semantic_fingerprint(
            {
                "contract": BASELINE_SCHEMA_VERSION,
                "source_snapshot_id": source_snapshot_id,
                "observed": baseline["observed"],
                "artifacts": artifact_inventory,
            }
        ),
        "artifact_inventory": artifact_inventory,
        "acceptance_gate": {
            "source_and_funnel_validated": True,
            "row_and_occurrence_denominators_recorded": True,
            "all_columns_have_policy": True,
            "physical_and_applicable_completeness_recorded": True,
            "schema_and_row_groups_validated": True,
            "all_output_checksums_recorded": True,
            "manifest_written_last": True,
            "passed": True,
        },
    }


def _source_funnel_markdown(
    inventory: SourceInventory, funnel: SourceFunnel
) -> str:
    lines = [
        "# GBIF media source funnel",
        "",
        f"Source snapshot: `{inventory.source_snapshot_id}`",
        "",
        "| Stage | Scope | Input rows | Output rows | Excluded rows | Output occurrences | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in funnel.stages:
        lines.append(
            "| {stage_id} | {scope} | {input_row_count:,} | {output_row_count:,} "
            "| {excluded_row_count:,} | {output_occurrence_count:,} | {exclusion_reason} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Unresolved multimedia rows: **{funnel.counts['unresolved_multimedia_rows']:,}**.",
            f"Unexplained residual rows: **{funnel.counts['unexplained_media_residual_rows']:,}**.",
            "",
            "All stored funnel validations passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _schema_markdown(audit: SchemaAudit) -> str:
    typed = sum(bool(row["typed_derivative_recommended"]) for row in audit.columns)
    return "\n".join(
        [
            "# Schema and physical integrity",
            "",
            f"- Rows: {audit.counts['rows']:,}",
            f"- Columns: {audit.counts['columns']:,}",
            f"- Row groups: {audit.counts['row_groups']:,}",
            f"- Schema fingerprint: `{audit.schema_fingerprint}`",
            f"- String-backed fields with a typed derivative recommendation: {typed:,}",
            "- Original strings are preserved; typed values must be separate assertions.",
            "- Every row group reconciles and every chunk is within file bounds.",
            "- The full value scan completed through the completeness profiler.",
            "",
        ]
    )


def _completeness_markdown(profile: CompletenessProfile) -> str:
    lines = [
        "# Completeness by applicability",
        "",
        f"Media-row denominator: **{profile.denominators['media_rows']:,}**  ",
        f"Distinct-occurrence denominator: **{profile.denominators['distinct_occurrences']:,}**",
        "",
        "Physical fill includes nonblank sentinel strings. Applicable fill excludes known generic semantic-null sentinels. Invalid-present and conflict checks remain `NOT_TESTED` until the local check registry phase.",
        "",
        "| Field | Rule | Physical media % | Applicable media | Applicable media % | Applicable occurrences | Applicable occurrence % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in profile.rows:
        lines.append(
            f"| {row['field_name']} | {row['applicability_rule']} | "
            f"{_format_pct(row['physical_fill_media_pct'])} | "
            f"{row['applicable_media_rows']:,} | "
            f"{_format_pct(row['applicable_fill_media_pct'])} | "
            f"{row['applicable_occurrences']:,} | "
            f"{_format_pct(row['applicable_fill_occurrence_pct'])} |"
        )
    return "\n".join(lines) + "\n"


def _format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}%"


def _write_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _write_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(value: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        item: dict[str, Any] = {
            "path": path.name,
            "physical_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            item.update(
                {
                    "row_count": parquet.metadata.num_rows,
                    "column_count": len(parquet.schema_arrow),
                    "row_group_count": parquet.metadata.num_row_groups,
                }
            )
        output.append(item)
    return output


def _verify_publication(root: Path, manifest: dict[str, Any]) -> None:
    for item in manifest["artifact_inventory"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["physical_bytes"]:
            raise ValueError(f"published artifact size mismatch: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"published artifact checksum mismatch: {path}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != item["row_count"]:
                raise ValueError(f"published artifact row count mismatch: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "SUPPLIED_BASELINE",
    "BaselinePublication",
    "publish_baseline",
]
