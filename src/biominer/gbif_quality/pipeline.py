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
from biominer.gbif_quality.biology import publish_biological_candidates
from biominer.gbif_quality.geography import publish_geographic_enrichment
from biominer.gbif_quality.phase3 import publish_phase3_summary
from biominer.gbif_quality.taxonomy import publish_species_rank_repairs
from biominer.gbif_quality.temporal import publish_temporal_quality_v2


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
BOUNDARY_MANIFEST = Path(
    "data/reference/gbif_boundaries/natural_earth/v5.1.1/manifest.json"
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


@dataclass(frozen=True, slots=True)
class Phase3Config:
    repository_root: str | Path
    data_root: str | Path = Path("data/derived/gbif_media_database/v4")
    memory_limit: str = "4GB"
    threads: int = 4
    batch_rows: int = 50_000

    def resolved(self) -> Phase3Config:
        root = Path(self.repository_root).resolve()
        return Phase3Config(
            repository_root=root,
            data_root=_resolve(root, Path(self.data_root)),
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


def run_phase3_enrichment(config: Phase3Config) -> dict[str, Any]:
    """Run or resume request-free Phase 3 deterministic enrichment."""

    cfg = config.resolved()
    root = Path(cfg.repository_root)
    data_root = Path(cfg.data_root)
    baseline = _read_json(data_root / "baseline.json")
    funnel = _read_json(data_root / "source_funnel.json")
    inventory = _read_json(data_root / "source_inventory.json")
    snapshot = str(baseline["source_snapshot_id"])
    counts = funnel["counts"]
    commit = _git_commit(root)
    derived_root = data_root / "derived_assertions"
    temporal_root = derived_root / "temporal"
    if not _validated_stage_ready(temporal_root):
        publish_temporal_quality_v2(
            v3_parquet=_resolve(root, V3_PARQUET),
            output_directory=temporal_root,
            source_snapshot_id=snapshot,
            source_publication_date=str(inventory["source_publication_date"]),
            expected_media_rows=int(counts["v3_media_rows"]),
            expected_occurrences=int(counts["v3_occurrences"]),
            expected_derived_year_media_rows=2_360,
            expected_derived_month_media_rows=4_941,
            expected_derived_day_media_rows=18_741,
            expected_ancient_media_rows=2_236,
            expected_ancient_occurrences=1_722,
            code_commit=commit,
            batch_rows=cfg.batch_rows,
        )
    geography_root = derived_root / "geography_v2"
    if not _validated_stage_ready(geography_root):
        publish_geographic_enrichment(
            v3_parquet=_resolve(root, V3_PARQUET),
            output_directory=geography_root,
            source_snapshot_id=snapshot,
            expected_coordinate_country_media_rows=7_374,
            expected_coordinate_country_occurrences=4_923,
            expected_missing_continent_media_rows=18_527,
            expected_missing_continent_occurrences=12_384,
            expected_missing_region_media_rows=679,
            expected_missing_region_occurrences=375,
            code_commit=commit,
            boundary_manifest=_resolve(root, BOUNDARY_MANIFEST),
            memory_limit=cfg.memory_limit,
            threads=cfg.threads,
        )
    taxonomy_root = derived_root / "taxonomy"
    if not _validated_stage_ready(taxonomy_root):
        publish_species_rank_repairs(
            v3_parquet=_resolve(root, V3_PARQUET),
            output_directory=taxonomy_root,
            source_snapshot_id=snapshot,
            expected_candidate_media_rows=337,
            expected_candidate_occurrences=221,
            code_commit=commit,
            memory_limit=cfg.memory_limit,
            threads=cfg.threads,
        )
    biology_root = derived_root / "biology"
    if not _validated_stage_ready(biology_root):
        publish_biological_candidates(
            v3_parquet=_resolve(root, V3_PARQUET),
            output_directory=biology_root,
            source_snapshot_id=snapshot,
            expected_media_rows=int(counts["v3_media_rows"]),
            expected_occurrences=int(counts["v3_occurrences"]),
            code_commit=commit,
            memory_limit=cfg.memory_limit,
            threads=cfg.threads,
        )
    summary_root = data_root / "quality_results" / "phase3_v2"
    if not _validated_stage_ready(summary_root):
        publish_phase3_summary(
            temporal_directory=temporal_root,
            geography_directory=geography_root,
            taxonomy_directory=taxonomy_root,
            biology_directory=biology_root,
            output_directory=summary_root,
            source_snapshot_id=snapshot,
            expected_media_rows=int(counts["v3_media_rows"]),
            expected_occurrences=int(counts["v3_occurrences"]),
            code_commit=commit,
        )
    manifest = _read_json(summary_root / "manifest.json")
    return {
        "status": "complete",
        "phase": 3,
        "code_commit": commit,
        "source_snapshot_id": snapshot,
        "source_rows": manifest["counts"]["source_media_rows"],
        "occurrences": manifest["counts"]["source_occurrences"],
        "derived_assertions": manifest["counts"]["derived_assertions"],
        "manual_review_rows": manifest["counts"]["manual_review_rows"],
        "semantic_assertion_fingerprint": manifest["semantic_assertion_fingerprint"],
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


def _validated_stage_ready(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        if path.exists():
            raise ValueError(f"incomplete stage exists without manifest: {path}")
        return False
    manifest = _read_json(manifest_path)
    validation = manifest.get("validation", {})
    if not isinstance(validation, dict) or not validation or not all(validation.values()):
        raise ValueError(f"stage validation is incomplete: {path}")
    for item in manifest.get("artifacts", []):
        if not (path / str(item["path"])).is_file():
            raise ValueError(f"stage artifact is missing: {path / str(item['path'])}")
    return True


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
    "Phase3Config",
    "V3_PARQUET",
    "default_funnel_config",
    "default_inventory_config",
    "run_phase1_baseline",
    "run_phase2_local_checks",
    "run_phase3_enrichment",
]
