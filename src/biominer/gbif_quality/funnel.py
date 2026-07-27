from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


FUNNEL_SCHEMA_VERSION = "biominer-gbif-media-source-funnel/v1"
FUNNEL_SCHEMA = pa.schema(
    [
        ("stage_order", pa.int16()),
        ("stage_id", pa.string()),
        ("scope", pa.string()),
        ("input_row_count", pa.int64()),
        ("output_row_count", pa.int64()),
        ("excluded_row_count", pa.int64()),
        ("input_occurrence_count", pa.int64()),
        ("output_occurrence_count", pa.int64()),
        ("fully_excluded_occurrence_count", pa.int64()),
        ("exclusion_reason", pa.string()),
        ("evidence_path", pa.string()),
        ("evidence_type", pa.string()),
        ("status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class FunnelConfig:
    repository_root: Path
    dwca_manifest: Path
    join_manifest: Path
    join_coverage: Path
    year_manifest: Path
    completeness_manifest: Path
    verification_group_manifest: Path
    verification_normalized_manifest: Path
    cohort_manifest: Path
    v3_filter_manifest: Path

    def resolved(self) -> FunnelConfig:
        root = self.repository_root.resolve()
        values = {
            name: _resolve(root, getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "repository_root"
        }
        return FunnelConfig(repository_root=root, **values)


@dataclass(frozen=True, slots=True)
class SourceFunnel:
    schema_version: str
    stages: tuple[dict[str, Any], ...]
    validation: dict[str, bool]
    counts: dict[str, int]

    def stage_table(self) -> pa.Table:
        return pa.Table.from_pylist(list(self.stages), schema=FUNNEL_SCHEMA)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stages": list(self.stages),
            "validation": self.validation,
            "counts": self.counts,
        }


def build_source_funnel(config: FunnelConfig) -> SourceFunnel:
    """Build a reason-coded source-to-v3 funnel from pinned evidence."""

    cfg = config.resolved()
    for name in cfg.__dataclass_fields__:
        if name == "repository_root":
            continue
        path = getattr(cfg, name)
        if not path.is_file():
            raise FileNotFoundError(path)
    dwca = _read_json(cfg.dwca_manifest)
    join = _read_json(cfg.join_manifest)
    year = _read_json(cfg.year_manifest)
    completeness = _read_json(cfg.completeness_manifest)
    grouped = _read_json(cfg.verification_group_manifest)
    normalized = _read_json(cfg.verification_normalized_manifest)
    cohort = _read_json(cfg.cohort_manifest)
    v3 = _read_json(cfg.v3_filter_manifest)
    coverage = pq.read_table(cfg.join_coverage).to_pylist()
    coverage_by_category = {
        str(row["coverage_category"]): row for row in coverage
    }

    raw_occurrences = _dwca_rows(dwca, "occurrence.txt")
    raw_media = _dwca_rows(dwca, "multimedia.txt")
    zero_media = int(coverage_by_category["zero_media"]["occurrence_record_count"])
    one_media_occurrences = int(
        coverage_by_category["one_media"]["occurrence_record_count"]
    )
    multiple_media_occurrences = int(
        coverage_by_category["multiple_media"]["occurrence_record_count"]
    )
    media_occurrences = one_media_occurrences + multiple_media_occurrences
    joined_rows = int(join["counts"]["joined_rows"])
    unresolved_media = int(
        coverage_by_category["unresolved_occurrence"]["multimedia_row_count"]
    )
    year_counts = year["counts"]
    year_rows = int(year_counts["output_rows"])
    year_occurrences = int(year_counts["output_distinct_gbif_ids"])
    projected_rows = int(completeness["output"]["row_count"])
    grouped_rows = int(grouped["counts"]["total_rows"])
    normalized_rows = int(normalized["counts"]["total_rows"])
    cohort_rows = int(cohort["counts"]["retained_rows"])
    cohort_occurrences = int(cohort["counts"]["retained_distinct_occurrences"])
    v3_rows = int(v3["counts"]["retained_rows"])
    v3_occurrences = int(v3["counts"]["retained_distinct_occurrences"])

    stages = (
        _stage(
            cfg,
            1,
            "OCCURRENCE_HAS_MEDIA",
            "occurrence",
            raw_occurrences,
            media_occurrences,
            raw_occurrences,
            media_occurrences,
            "NO_MEDIA_ASSERTION",
            cfg.join_coverage,
            "computed_join_coverage",
        ),
        _stage(
            cfg,
            2,
            "MULTIMEDIA_FOREIGN_KEY",
            "media_assertion",
            raw_media,
            joined_rows,
            media_occurrences,
            media_occurrences,
            "UNRESOLVED_OCCURRENCE",
            cfg.join_manifest,
            "computed_join_manifest",
        ),
        _stage(
            cfg,
            3,
            "YEAR_COHORT",
            "media_assertion",
            joined_rows,
            year_rows,
            media_occurrences,
            year_occurrences,
            "PARSEABLE_YEAR_BEFORE_1960",
            cfg.year_manifest,
            "filter_manifest",
        ),
        _stage(
            cfg,
            4,
            "LEGACY_COLUMN_PROJECTION",
            "media_assertion",
            year_rows,
            projected_rows,
            year_occurrences,
            year_occurrences,
            "NO_ROW_EXCLUSION_COLUMN_PROJECTION_ONLY",
            cfg.completeness_manifest,
            "projection_manifest",
        ),
        _stage(
            cfg,
            5,
            "LEGACY_VERIFICATION_GROUPING",
            "media_assertion",
            projected_rows,
            grouped_rows,
            year_occurrences,
            year_occurrences,
            "NO_ROW_EXCLUSION_DERIVED_GROUP_ONLY",
            cfg.verification_group_manifest,
            "normalization_manifest",
        ),
        _stage(
            cfg,
            6,
            "LEGACY_STATUS_NORMALIZATION",
            "media_assertion",
            grouped_rows,
            normalized_rows,
            year_occurrences,
            year_occurrences,
            "NO_ROW_EXCLUSION_VALUE_TRANSFORMATION_ONLY",
            cfg.verification_normalized_manifest,
            "normalization_manifest",
        ),
        _stage(
            cfg,
            7,
            "IDENTIFIED_OR_ACCEPTED_COHORT",
            "media_assertion",
            normalized_rows,
            cohort_rows,
            year_occurrences,
            cohort_occurrences,
            "OUTSIDE_LEGACY_IDENTIFIED_OR_ACCEPTED_COHORT",
            cfg.cohort_manifest,
            "filter_manifest",
        ),
        _stage(
            cfg,
            8,
            "EXPLICIT_MEDIA_RIGHTS_FILTER",
            "media_assertion",
            cohort_rows,
            v3_rows,
            cohort_occurrences,
            v3_occurrences,
            "EXPLICIT_ALL_RIGHTS_RESERVED_OR_COPYRIGHT",
            cfg.v3_filter_manifest,
            "filter_manifest",
        ),
    )
    expected_v3_removals = int(v3["counts"]["removed_rows"])
    cohort_exclusions = normalized_rows - cohort_rows
    rights_exclusions = cohort_rows - v3_rows
    validation = {
        "occurrence_cardinality_reconciled": raw_occurrences
        == zero_media + media_occurrences,
        "multimedia_cardinality_reconciled": raw_media
        == int(coverage_by_category["one_media"]["multimedia_row_count"])
        + int(coverage_by_category["multiple_media"]["multimedia_row_count"]),
        "multimedia_join_reconciled": raw_media == joined_rows + unresolved_media,
        "year_filter_reconciled": joined_rows
        == year_rows + int(year_counts["removed_pre_cutoff_rows"]),
        "projection_preserves_rows": year_rows == projected_rows,
        "verification_grouping_preserves_rows": projected_rows == grouped_rows,
        "status_normalization_preserves_rows": grouped_rows == normalized_rows,
        "cohort_filter_reconciled": normalized_rows
        == cohort_rows + int(cohort["counts"]["removed_rows"]),
        "rights_filter_reconciled": cohort_rows
        == v3_rows + int(v3["counts"]["restricted_image_rows_excluded"]),
        "v3_total_filter_reconciled": expected_v3_removals
        == cohort_exclusions + rights_exclusions,
        "final_v3_rows_match": v3_rows == int(v3["output"]["row_count"]),
        "all_stage_exclusions_reasoned": all(
            stage["exclusion_reason"] for stage in stages
        ),
        "unexplained_media_residual_is_zero": raw_media
        - unresolved_media
        - int(year_counts["removed_pre_cutoff_rows"])
        - cohort_exclusions
        - rights_exclusions
        - v3_rows
        == 0,
    }
    if not all(validation.values()):
        raise ValueError(f"source funnel validation failed: {validation}")
    counts = {
        "raw_occurrence_rows": raw_occurrences,
        "raw_multimedia_rows": raw_media,
        "occurrences_with_media": media_occurrences,
        "occurrences_without_media": zero_media,
        "unresolved_multimedia_rows": unresolved_media,
        "pre_1960_media_rows_excluded": int(year_counts["removed_pre_cutoff_rows"]),
        "legacy_cohort_rows_excluded": cohort_exclusions,
        "explicit_rights_rows_excluded": rights_exclusions,
        "v3_media_rows": v3_rows,
        "v3_occurrences": v3_occurrences,
        "unexplained_media_residual_rows": 0,
    }
    return SourceFunnel(
        schema_version=FUNNEL_SCHEMA_VERSION,
        stages=stages,
        validation=validation,
        counts=counts,
    )


def _stage(
    cfg: FunnelConfig,
    order: int,
    stage_id: str,
    scope: str,
    input_rows: int,
    output_rows: int,
    input_occurrences: int,
    output_occurrences: int,
    reason: str,
    evidence: Path,
    evidence_type: str,
) -> dict[str, Any]:
    return {
        "stage_order": order,
        "stage_id": stage_id,
        "scope": scope,
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "excluded_row_count": input_rows - output_rows,
        "input_occurrence_count": input_occurrences,
        "output_occurrence_count": output_occurrences,
        "fully_excluded_occurrence_count": input_occurrences - output_occurrences,
        "exclusion_reason": reason,
        "evidence_path": _display_path(cfg.repository_root, evidence),
        "evidence_type": evidence_type,
        "status": "PASS",
    }


def _dwca_rows(manifest: dict[str, Any], member: str) -> int:
    for item in manifest["outputs"]:
        if item["member"] == member:
            return int(item["row_count"])
    raise ValueError(f"DWCA manifest has no {member} output")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "FUNNEL_SCHEMA",
    "FUNNEL_SCHEMA_VERSION",
    "FunnelConfig",
    "SourceFunnel",
    "build_source_funnel",
]
