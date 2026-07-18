"""Complete immutable ButterflyLens evidence-handoff publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

import polars as pl

from biominer.integration.butterflylens_geographic_export import (
    export_butterflylens_geographic_impact,
    validate_butterflylens_geographic_export,
    validate_butterflylens_geographic_impact,
)
from biominer.integration.butterflylens_model_export import (
    ButterflyLensModelLayer,
    BUTTERFLYLENS_MODEL_ROLES,
    export_butterflylens_model_evidence,
    validate_butterflylens_model_export,
)
from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_POOL_HANDOFF_FILE,
    BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
    build_butterflylens_pool_handoff,
    validate_butterflylens_pool_handoff,
    write_butterflylens_pool_handoff,
)
from biominer.integration.butterflylens_review_export import (
    ButterflyLensReviewLayer,
    BUTTERFLYLENS_REVIEW_ROLES,
    export_butterflylens_review_evidence,
    validate_butterflylens_review_export,
)
from biominer.integration.product_handoff import validate_git_sha
from biominer.storage.handoff import (
    HandoffBundle,
    build_handoff_bundle,
    verify_handoff_bundle,
)


BUTTERFLYLENS_POOL_HANDOFF_BUNDLE_NAME = "butterflylens-dynamic-pool-handoff"
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
class PublishedButterflyLensPoolHandoff:
    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    bundle: HandoffBundle


def publish_butterflylens_pool_handoff(
    *,
    project: Mapping[str, object],
    run: Mapping[str, object],
    model_layer: ButterflyLensModelLayer,
    geographic_impact: pl.DataFrame,
    review_layer: ButterflyLensReviewLayer,
    output_root: str | Path,
    archive_output_dir: str | Path,
    producer_commit: str,
    created_at: str | datetime,
    registry_version: str,
) -> PublishedButterflyLensPoolHandoff:
    """Stage, validate, archive, and atomically publish all ten roles."""

    commit = validate_git_sha(producer_commit, field="producer_commit")
    validate_butterflylens_geographic_impact(geographic_impact)
    if run.get("engine", {}).get("commit") != commit:
        raise ValueError("ButterflyLens publication producer and run commit differ")
    destination = Path(output_root)
    if destination.is_symlink():
        raise ValueError("ButterflyLens publication root must not be a symlink")
    if destination.exists():
        raise FileExistsError(
            f"ButterflyLens publication root is create-only: {destination}"
        )
    archive_directory = Path(archive_output_dir)
    if archive_directory.is_symlink():
        raise ValueError("ButterflyLens archive directory must not be a symlink")
    if archive_directory.resolve().is_relative_to(destination.resolve()):
        raise ValueError("ButterflyLens archive must be outside the publication root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-")
    )
    try:
        model = export_butterflylens_model_evidence(
            project=project,
            run=run,
            layer=model_layer,
            output_root=staging_root,
        )
        geographic = export_butterflylens_geographic_impact(
            frame=geographic_impact,
            output_root=staging_root,
        )
        review = export_butterflylens_review_evidence(
            layer=review_layer,
            output_root=staging_root,
        )
        artifacts = (*model.artifacts, geographic.artifact, *review.artifacts)
        source_snapshots = sorted(
            model_layer.flickr_source_records["source_snapshot_fingerprint"].unique()
        )
        model_fingerprints = sorted(
            model_layer.model_evidence["model_fingerprint"].unique()
        )
        preprocessing_fingerprints = sorted(
            model_layer.model_evidence["preprocessing_fingerprint"].unique()
        )
        if len(model_fingerprints) != 1 or len(preprocessing_fingerprints) != 1:
            raise ValueError("ButterflyLens publication model scope differs")
        manifest = build_butterflylens_pool_handoff(
            producer_commit=commit,
            created_at=created_at,
            project_id=str(project["project_id"]),
            run_id=str(run["run_id"]),
            registry_version=registry_version,
            source_snapshot_fingerprints=source_snapshots,
            model_fingerprint=model_fingerprints[0],
            preprocessing_fingerprint=preprocessing_fingerprints[0],
            artifacts=artifacts,
        )
        write_butterflylens_pool_handoff(manifest, staging_root)
        validate_published_butterflylens_pool_handoff(staging_root, manifest)
        bundle = build_handoff_bundle(
            root=staging_root,
            sources=("artifacts", BUTTERFLYLENS_POOL_HANDOFF_FILE),
            output_dir=archive_directory,
            name=BUTTERFLYLENS_POOL_HANDOFF_BUNDLE_NAME,
            source_git_sha=commit,
        )
        verification = verify_handoff_bundle(
            bundle.archive_path, expected_sha256=bundle.sha256
        )
        if (
            verification.file_count != len(BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES) + 1
            or verification.source_git_sha != commit
        ):
            raise ValueError("ButterflyLens archive inventory differs")
        staging_root.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    validate_published_butterflylens_pool_handoff(destination, manifest)
    return PublishedButterflyLensPoolHandoff(
        root=destination.resolve(),
        manifest_path=(destination / BUTTERFLYLENS_POOL_HANDOFF_FILE).resolve(),
        manifest=manifest,
        bundle=bundle,
    )


def validate_published_butterflylens_pool_handoff(
    root: str | Path, manifest: Mapping[str, object]
) -> None:
    """Re-read the manifest and every role artifact from a published package."""

    validate_butterflylens_pool_handoff(manifest)
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("ButterflyLens publication root is unavailable")
    manifest_path = root_path / BUTTERFLYLENS_POOL_HANDOFF_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("ButterflyLens publication manifest is unavailable")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("ButterflyLens publication manifest bytes differ")
    descriptors = [
        {field: artifact[field] for field in _DESCRIPTOR_FIELDS}
        for artifact in manifest["artifacts"]
    ]
    by_role = {str(row["role"]): row for row in descriptors}
    validate_butterflylens_model_export(
        root_path, [by_role[role] for role in BUTTERFLYLENS_MODEL_ROLES]
    )
    validate_butterflylens_geographic_export(root_path, by_role["geographic_impact"])
    validate_butterflylens_review_export(
        root_path, [by_role[role] for role in BUTTERFLYLENS_REVIEW_ROLES]
    )
    if tuple(by_role) != BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES:
        raise ValueError("ButterflyLens publication roles are not canonical")
    entries = tuple(root_path.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("ButterflyLens publication root contains a symlink")
    if {path.name for path in entries} != {
        "artifacts",
        BUTTERFLYLENS_POOL_HANDOFF_FILE,
    }:
        raise ValueError("ButterflyLens publication root file set differs")
    artifact_entries = tuple((root_path / "artifacts").iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in artifact_entries):
        raise ValueError("ButterflyLens artifact directory has unsafe entries")
    if {path.name for path in artifact_entries} != {"model", "geographic", "review"}:
        raise ValueError("ButterflyLens artifact layer set differs")


__all__ = [
    "BUTTERFLYLENS_POOL_HANDOFF_BUNDLE_NAME",
    "PublishedButterflyLensPoolHandoff",
    "publish_butterflylens_pool_handoff",
    "validate_published_butterflylens_pool_handoff",
]
