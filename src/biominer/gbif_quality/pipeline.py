from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import pyarrow.parquet as pq

from biominer.gbif_quality.baseline import publish_baseline
from biominer.gbif_quality.funnel import FunnelConfig, build_source_funnel
from biominer.gbif_quality.inventory import (
    SourceInventoryConfig,
    build_source_inventory,
)
from biominer.gbif_quality.policy import build_field_policy
from biominer.gbif_quality.phase2 import publish_phase2_summary
from biominer.gbif_quality.profile import profile_completeness
from biominer.gbif_quality.schema_audit import audit_parquet_schema
from biominer.gbif_quality.source_ledger import publish_source_media_ledger
from biominer.gbif_quality.media_checks import publish_media_assertion_quality
from biominer.gbif_quality.occurrence_checks import publish_occurrence_quality


V3_PARQUET = Path(
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_identified_by_or_accepted_"
    "rights_filtered_parquet/"
    "occurrence_multimedia_year_ge_1960_completeness_ge_5pct_"
    "verification_normalized_identified_by_or_accepted_rights_filtered.parquet"
)
V3_SOURCE_MANIFEST = V3_PARQUET.parent / "identified_by_or_accepted_filter_manifest.json"
JOINED_PARQUET = Path(
    "data/reference/gbif_global_papilionoidea_occurrence_multimedia_parquet/"
    "occurrence_multimedia.parquet"
)
NORMALIZED_PARQUET = Path(
    "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_parquet/"
    "occurrence_multimedia_year_ge_1960_completeness_ge_5pct_"
    "verification_normalized.parquet"
)


@dataclass(frozen=True, slots=True)
class Phase1Config:
    repository_root: str | Path
    data_output: str | Path = Path("data/derived/gbif_media_database/v4")
    report_output: str | Path = Path("reports/gbif_media_database/v4")
    temp_directory: str | Path | None = None
    memory_limit: str = "4GB"
    occurrence_batch_size: int = 8

    def resolved(self) -> Phase1Config:
        root = Path(self.repository_root).resolve()
        return Phase1Config(
            repository_root=root,
            data_output=_resolve(root, Path(self.data_output)),
            report_output=_resolve(root, Path(self.report_output)),
            temp_directory=(
                _resolve(root, Path(self.temp_directory))
                if self.temp_directory is not None
                else None
            ),
            memory_limit=self.memory_limit,
            occurrence_batch_size=self.occurrence_batch_size,
        )


@dataclass(frozen=True, slots=True)
class Phase2Config:
    repository_root: str | Path
    data_root: str | Path = Path("data/derived/gbif_media_database/v4")
    temp_directory: str | Path | None = None
    memory_limit: str = "4GB"
    threads: int = 4
    batch_rows: int = 100_000

    def resolved(self) -> Phase2Config:
        root = Path(self.repository_root).resolve()
        return Phase2Config(
            repository_root=root,
            data_root=_resolve(root, Path(self.data_root)),
            temp_directory=(
                _resolve(root, Path(self.temp_directory))
                if self.temp_directory is not None
                else None
            ),
            memory_limit=self.memory_limit,
            threads=self.threads,
            batch_rows=self.batch_rows,
        )


def run_phase1_baseline(config: Phase1Config) -> dict[str, Any]:
    """Run the complete local Phase 1 baseline from pinned repository inputs."""

    cfg = config.resolved()
    inventory = build_source_inventory(default_inventory_config(cfg.repository_root))
    funnel = build_source_funnel(default_funnel_config(cfg.repository_root))
    v3 = _resolve(cfg.repository_root, V3_PARQUET)
    policies = build_field_policy(pq.ParquetFile(v3).schema_arrow)
    completeness = profile_completeness(
        v3,
        policies,
        occurrence_batch_size=cfg.occurrence_batch_size,
        memory_limit=cfg.memory_limit,
        temp_directory=cfg.temp_directory,
    )
    schema = audit_parquet_schema(v3, policies, full_value_scan_status="PASS")
    commit = _git_commit(cfg.repository_root)
    publication = publish_baseline(
        inventory=inventory,
        funnel=funnel,
        schema_audit=schema,
        policies=policies,
        completeness=completeness,
        data_root=cfg.data_output,
        report_root=cfg.report_output,
        code_commit=commit,
    )
    return {
        "status": "complete",
        "phase": 1,
        "code_commit": commit,
        "source_snapshot_id": inventory.source_snapshot_id,
        "data_output": str(publication.data_root),
        "report_output": str(publication.report_root),
        "denominators": completeness.denominators,
        "source_counts": funnel.counts,
        "schema_fingerprint": schema.schema_fingerprint,
        "data_manifest": str(publication.data_root / "manifest.json"),
        "report_manifest": str(publication.report_root / "manifest.json"),
    }


def run_phase2_local_checks(config: Phase2Config) -> dict[str, Any]:
    """Run or resume the four create-only Phase 2 local quality stages."""

    cfg = config.resolved()
    root = Path(cfg.repository_root)
    data_root = Path(cfg.data_root)
    baseline = _read_json(data_root / "baseline.json")
    funnel = _read_json(data_root / "source_funnel.json")
    snapshot = str(baseline["source_snapshot_id"])
    counts = funnel["counts"]
    commit = _git_commit(root)
    ledger_root = data_root / "source_lineage"
    if not _stage_ready(ledger_root, int(counts["raw_multimedia_rows"])):
        publish_source_media_ledger(
            joined_parquet=_resolve(root, JOINED_PARQUET),
            normalized_parquet=_resolve(root, NORMALIZED_PARQUET),
            output_directory=ledger_root,
            source_snapshot_id=snapshot,
            expected_counts=counts,
            code_commit=commit,
            memory_limit=cfg.memory_limit,
            temp_directory=cfg.temp_directory,
            batch_rows=cfg.batch_rows,
        )
    media_root = data_root / "media_assertion_quality"
    if not _stage_ready(media_root, int(counts["v3_media_rows"])):
        publish_media_assertion_quality(
            v3_parquet=_resolve(root, V3_PARQUET),
            source_ledger_parquet=ledger_root / "source_media_status.parquet",
            output_directory=media_root,
            source_snapshot_id=snapshot,
            expected_rows=int(counts["v3_media_rows"]),
            code_commit=commit,
            batch_rows=cfg.batch_rows,
        )
    occurrence_root = data_root / "occurrence_quality"
    if not _stage_ready(occurrence_root, int(counts["v3_occurrences"])):
        publish_occurrence_quality(
            v3_parquet=_resolve(root, V3_PARQUET),
            output_directory=occurrence_root,
            source_snapshot_id=snapshot,
            expected_media_rows=int(counts["v3_media_rows"]),
            expected_occurrences=int(counts["v3_occurrences"]),
            code_commit=commit,
            memory_limit=cfg.memory_limit,
            threads=cfg.threads,
            temp_directory=cfg.temp_directory,
            batch_rows=max(1, cfg.batch_rows // 2),
        )
    summary_root = data_root / "quality_results" / "phase2"
    if not _stage_ready(summary_root, int(counts["raw_multimedia_rows"])):
        publish_phase2_summary(
            source_ledger_parquet=ledger_root / "source_media_status.parquet",
            media_quality_parquet=media_root / "media_assertion_quality.parquet",
            occurrence_quality_parquet=occurrence_root / "occurrence_quality.parquet",
            media_manifest=media_root / "manifest.json",
            occurrence_manifest=occurrence_root / "manifest.json",
            output_directory=summary_root,
            source_snapshot_id=snapshot,
            expected_source_rows=int(counts["raw_multimedia_rows"]),
            expected_v3_rows=int(counts["v3_media_rows"]),
            expected_occurrences=int(counts["v3_occurrences"]),
            code_commit=commit,
            batch_rows=cfg.batch_rows,
        )
    manifest = _read_json(summary_root / "manifest.json")
    return {
        "status": "complete",
        "phase": 2,
        "code_commit": commit,
        "source_snapshot_id": snapshot,
        "source_rows": manifest["counts"]["source_rows"],
        "retained_rows": manifest["counts"]["retained_rows"],
        "occurrences": counts["v3_occurrences"],
        "output": str(summary_root),
        "resumable": True,
        "network_requests": 0,
    }


def default_inventory_config(repository_root: str | Path) -> SourceInventoryConfig:
    root = Path(repository_root).resolve()
    return SourceInventoryConfig(
        repository_root=root,
        archive=Path("data/reference/gbif-global-papilionoidea-download-clean.zip"),
        dwca_manifest=Path("data/reference/gbif_global_papilionoidea_parquet/dwca_parquet_manifest.json"),
        occurrence_parquet=Path("data/reference/gbif_global_papilionoidea_parquet/occurrence.parquet"),
        multimedia_parquet=Path("data/reference/gbif_global_papilionoidea_parquet/multimedia.parquet"),
        verbatim_parquet=Path("data/reference/gbif_global_papilionoidea_parquet/verbatim.parquet"),
        joined_parquet=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_parquet/occurrence_multimedia.parquet"
        ),
        joined_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_parquet/occurrence_multimedia_join_manifest.json"
        ),
        v3_parquet=V3_PARQUET,
        v3_manifest=Path("data/derived/gbif_media_database/v3/manifest.json"),
        prior_intake_manifest=Path("data/derived/gbif_media_audit/v1/intake_validation.json"),
    )


def default_funnel_config(repository_root: str | Path) -> FunnelConfig:
    root = Path(repository_root).resolve()
    return FunnelConfig(
        repository_root=root,
        dwca_manifest=Path("data/reference/gbif_global_papilionoidea_parquet/dwca_parquet_manifest.json"),
        join_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_parquet/occurrence_multimedia_join_manifest.json"
        ),
        join_coverage=Path("data/derived/gbif_media_audit/v1/join_coverage.parquet"),
        year_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_parquet/year_filter_manifest.json"
        ),
        completeness_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_parquet/completeness_filter_manifest.json"
        ),
        verification_group_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_grouped_parquet/verification_group_manifest.json"
        ),
        verification_normalized_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_parquet/verification_group_manifest.json"
        ),
        cohort_manifest=Path(
            "data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_identified_by_or_accepted_parquet/identified_by_or_accepted_filter_manifest.json"
        ),
        v3_filter_manifest=V3_SOURCE_MANIFEST,
    )


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _stage_ready(path: Path, expected_rows: int) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        if path.exists():
            raise ValueError(f"incomplete stage exists without manifest: {path}")
        return False
    manifest = _read_json(manifest_path)
    validation = manifest.get("validation", {})
    if not isinstance(validation, dict) or not all(validation.values()):
        raise ValueError(f"stage validation is incomplete: {path}")
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"stage has no artifact inventory: {path}")
    row_counts = []
    for item in artifacts:
        artifact = path / str(item["path"])
        if not artifact.is_file():
            raise ValueError(f"stage artifact is missing: {artifact}")
        if item.get("row_count") is not None:
            row_counts.append(int(item["row_count"]))
    return expected_rows in row_counts


def _read_json(path: Path) -> dict[str, Any]:
    value = __import__("json").loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "Phase1Config",
    "Phase2Config",
    "V3_PARQUET",
    "default_funnel_config",
    "default_inventory_config",
    "run_phase1_baseline",
    "run_phase2_local_checks",
]
