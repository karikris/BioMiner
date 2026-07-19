"""Complete, immutable TaxaLens dynamic-pool handoff publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

import polars as pl

from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    ProbabilityAuditSelection,
)
from biominer.integration.taxalens_pool_export import (
    TAXALENS_SCORE_POOL_ROLES,
    export_taxalens_score_pool_evidence,
    validate_taxalens_score_pool_export,
)
from biominer.integration.taxalens_pool_handoff import (
    TAXALENS_POOL_HANDOFF_FILE,
    TAXALENS_REQUIRED_ARTIFACT_ROLES,
    build_taxalens_pool_handoff,
    validate_taxalens_pool_handoff,
    write_taxalens_pool_handoff,
)
from biominer.integration.taxalens_quality_export import (
    TAXALENS_REVIEW_QUALITY_ROLES,
    export_taxalens_review_quality_evidence,
    validate_taxalens_review_quality_export,
)
from biominer.storage.handoff import (
    HandoffBundle,
    build_handoff_bundle,
    verify_handoff_bundle,
)


TAXALENS_POOL_HANDOFF_BUNDLE_NAME = "taxalens-dynamic-pool-handoff"
_DESCRIPTOR_FIELDS = (
    "role",
    "availability",
    "unavailable_reason",
    "relative_path",
    "media_type",
    "schema_version",
    "semantic_fingerprint",
    "sha256",
    "byte_count",
    "row_count",
    "parent_fingerprints",
    "evidence_maturity_label",
)


@dataclass(frozen=True, slots=True)
class PublishedTaxaLensPoolHandoff:
    """Validated directory, manifest, and content-addressed archive."""

    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    bundle: HandoffBundle


def publish_taxalens_pool_handoff(
    *,
    score_pool_frames: Mapping[str, pl.DataFrame],
    review_selection: ProbabilityAuditSelection,
    review_sampling_policy: ProbabilityAuditSamplingPolicy,
    quality_report: pl.DataFrame | None,
    output_root: str | Path,
    archive_output_dir: str | Path,
    producer_commit: str,
    created_at: str | datetime,
    run_id: str,
    registry_version: str,
    source_snapshot_fingerprints: Sequence[str],
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    quality_unavailable_reason: str = "no validated reviewed quality report supplied",
) -> PublishedTaxaLensPoolHandoff:
    """Stage, verify, archive, and atomically publish one complete handoff."""

    if set(score_pool_frames) != set(TAXALENS_SCORE_POOL_ROLES):
        raise ValueError("TaxaLens score/pool publication frame roles differ")
    destination = Path(output_root)
    if destination.is_symlink():
        raise ValueError("TaxaLens publication root must not be a symlink")
    if destination.exists():
        raise FileExistsError(
            f"TaxaLens publication root is create-only: {destination}"
        )
    archive_directory = Path(archive_output_dir)
    if archive_directory.is_symlink():
        raise ValueError("TaxaLens archive output directory must not be a symlink")
    if archive_directory.resolve().is_relative_to(destination.resolve()):
        raise ValueError("TaxaLens archive output must be outside the publication root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-")
    )
    try:
        score_pool = export_taxalens_score_pool_evidence(
            **dict(score_pool_frames),
            output_root=staging_root,
        )
        review_quality = export_taxalens_review_quality_evidence(
            selection=review_selection,
            sampling_policy=review_sampling_policy,
            quality_report=quality_report,
            output_root=staging_root,
            quality_unavailable_reason=quality_unavailable_reason,
        )
        artifacts = (*score_pool.artifacts, *review_quality.artifacts)
        manifest = build_taxalens_pool_handoff(
            producer_commit=producer_commit,
            created_at=created_at,
            run_id=run_id,
            registry_version=registry_version,
            source_snapshot_fingerprints=source_snapshot_fingerprints,
            model_fingerprint=model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            artifacts=artifacts,
            completed_review_count=review_quality.completed_review_count,
            quality_estimate_available=(review_quality.quality_estimate_available),
            quality_unavailable_reason=(review_quality.quality_unavailable_reason),
        )
        write_taxalens_pool_handoff(manifest, staging_root)
        validate_published_taxalens_pool_handoff(staging_root, manifest)
        bundle = build_handoff_bundle(
            root=staging_root,
            sources=("artifacts", TAXALENS_POOL_HANDOFF_FILE),
            output_dir=archive_directory,
            name=TAXALENS_POOL_HANDOFF_BUNDLE_NAME,
            source_git_sha=producer_commit,
        )
        verification = verify_handoff_bundle(
            bundle.archive_path,
            expected_sha256=bundle.sha256,
        )
        expected_file_count = (
            sum(
                artifact["availability"] == "available"
                for artifact in manifest["artifacts"]
            )
            + 1
        )
        if (
            verification.file_count != expected_file_count
            or verification.source_git_sha != manifest["producer"]["commit"]
        ):
            raise ValueError("TaxaLens archive inventory differs from its manifest")
        staging_root.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    published_manifest_path = destination / TAXALENS_POOL_HANDOFF_FILE
    validate_published_taxalens_pool_handoff(destination, manifest)
    return PublishedTaxaLensPoolHandoff(
        root=destination.resolve(),
        manifest_path=published_manifest_path.resolve(),
        manifest=manifest,
        bundle=bundle,
    )


def validate_published_taxalens_pool_handoff(
    root: str | Path,
    manifest: Mapping[str, object],
) -> None:
    """Verify the stored manifest and all available evidence bytes."""

    validate_taxalens_pool_handoff(manifest)
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("TaxaLens publication root is unavailable")
    manifest_path = root_path / TAXALENS_POOL_HANDOFF_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("TaxaLens publication manifest is unavailable")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("TaxaLens publication manifest bytes differ")
    descriptors = [
        {field: artifact[field] for field in _DESCRIPTOR_FIELDS}
        for artifact in manifest["artifacts"]
    ]
    by_role = {str(value["role"]): value for value in descriptors}
    validate_taxalens_score_pool_export(
        root_path,
        [by_role[role] for role in TAXALENS_SCORE_POOL_ROLES],
    )
    maturity = manifest["evidence_maturity"]
    completed_review_count = maturity["human_review"]["completed_review_count"]
    quality_estimate_available = maturity["quality_estimate"]["status"] == "available"
    validate_taxalens_review_quality_export(
        root_path,
        [by_role[role] for role in TAXALENS_REVIEW_QUALITY_ROLES],
        completed_review_count=completed_review_count,
        quality_estimate_available=quality_estimate_available,
    )
    quality_descriptor = by_role["quality_sidecar"]
    quality_reason = maturity["quality_estimate"]["unavailable_reason"]
    if (
        quality_descriptor["availability"] != "available"
        and quality_descriptor["unavailable_reason"] != quality_reason
    ):
        raise ValueError("TaxaLens quality unavailable reasons differ")
    entries = tuple(root_path.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("TaxaLens publication root contains a symlink")
    if {path.name for path in entries} != {
        "artifacts",
        TAXALENS_POOL_HANDOFF_FILE,
    }:
        raise ValueError("TaxaLens publication root file set differs")
    if tuple(by_role) != TAXALENS_REQUIRED_ARTIFACT_ROLES:
        raise ValueError("TaxaLens publication artifact roles are not canonical")


__all__ = [
    "TAXALENS_POOL_HANDOFF_BUNDLE_NAME",
    "PublishedTaxaLensPoolHandoff",
    "publish_taxalens_pool_handoff",
    "validate_published_taxalens_pool_handoff",
]
