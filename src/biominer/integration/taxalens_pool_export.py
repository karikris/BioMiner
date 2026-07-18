"""Create-only TaxaLens exports for canonical dynamic score and pool evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from tempfile import mkdtemp

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    validate_dynamic_reference_pool_artifacts,
)
from biominer.bioclip.dynamic_pool_scores import (
    validate_dynamic_pool_score_artifacts,
)
from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.integration.product_handoff import normalize_product_artifacts
from biominer.integration.taxalens_pool_handoff import TAXALENS_ROLE_DEFAULTS
from biominer.storage.content_address import sha256_file
from biominer.storage.parquet import write_parquet


TAXALENS_SCORE_POOL_ROLES = (
    "candidate_scores",
    "photo_summaries",
    "pool_plans",
    "pool_members",
    "pool_summaries",
    "candidate_sets",
)

_FRAME_FINGERPRINT_FIELDS = {
    "candidate_scores": "score_fingerprint",
    "photo_summaries": "summary_fingerprint",
    "pool_plans": "plan_fingerprint",
    "pool_members": "member_fingerprint",
    "pool_summaries": "summary_fingerprint",
    "candidate_sets": "candidate_row_fingerprint",
}
_PARENT_FINGERPRINT_FIELDS = {
    "candidate_scores": (
        "visual_input_contract_fingerprint",
        "plan_fingerprint",
        "candidate_set_fingerprint",
        "score_policy_fingerprint",
        "model_fingerprint",
    ),
    "photo_summaries": (
        "visual_input_contract_fingerprint",
        "plan_fingerprint",
        "candidate_set_fingerprint",
        "candidate_scores_fingerprint",
    ),
    "pool_plans": (
        "query_embedding_fingerprint",
        "reference_geography_index_fingerprint",
        "candidate_set_fingerprint",
        "selection_policy_fingerprint",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    ),
    "pool_members": (
        "query_embedding_fingerprint",
        "candidate_set_fingerprint",
        "reference_embedding_fingerprint",
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    ),
    "pool_summaries": (
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "model_fingerprint",
        "pool_membership_fingerprint",
    ),
    "candidate_sets": ("visual_neighbour_graph_fingerprint",),
}
_MATURITY = {
    "candidate_scores": "provisional_raw_score",
    "photo_summaries": "provisional_raw_score",
    "pool_plans": "provider_asserted_provisional_support",
    "pool_members": "provider_asserted_provisional_support",
    "pool_summaries": "provider_asserted_provisional_support",
    "candidate_sets": "provisional_raw_score",
}


@dataclass(frozen=True, slots=True)
class TaxaLensScorePoolExport:
    """Published paths and raw product-manifest artifact descriptors."""

    root: Path
    artifact_directory: Path
    artifacts: tuple[dict[str, object], ...]


def export_taxalens_score_pool_evidence(
    *,
    candidate_scores: pl.DataFrame,
    photo_summaries: pl.DataFrame,
    pool_plans: pl.DataFrame,
    pool_members: pl.DataFrame,
    pool_summaries: pl.DataFrame,
    candidate_sets: pl.DataFrame,
    output_root: str | Path,
) -> TaxaLensScorePoolExport:
    """Validate and atomically publish the six canonical evidence tables."""

    frames = _frames(
        candidate_scores=candidate_scores,
        photo_summaries=photo_summaries,
        pool_plans=pool_plans,
        pool_members=pool_members,
        pool_summaries=pool_summaries,
        candidate_sets=candidate_sets,
    )
    _validate_frames(frames)
    destination = Path(output_root)
    if destination.is_symlink():
        raise ValueError("TaxaLens handoff root must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    artifact_directory = destination / "artifacts"
    if artifact_directory.exists():
        raise FileExistsError(
            "TaxaLens score/pool artifact directory is create-only: "
            f"{artifact_directory}"
        )
    staging_root = Path(mkdtemp(dir=destination, prefix=".taxalens-score-pool-"))
    staging_artifacts = staging_root / "artifacts"
    staging_artifacts.mkdir()
    try:
        descriptors = tuple(
            _write_artifact(role, frames[role], staging_artifacts)
            for role in TAXALENS_SCORE_POOL_ROLES
        )
        validate_taxalens_score_pool_export(staging_root, descriptors)
        staging_artifacts.replace(artifact_directory)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return TaxaLensScorePoolExport(
        root=destination,
        artifact_directory=artifact_directory,
        artifacts=descriptors,
    )


def validate_taxalens_score_pool_export(
    root: str | Path,
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    """Re-read every byte and verify schema, lineage, identity, and maturity."""

    normalized = normalize_product_artifacts(
        artifacts,
        required_roles=TAXALENS_SCORE_POOL_ROLES,
        producer_repository="karikris/BioMiner",
        producer_commit="0" * 40,
    )
    destination = Path(root)
    frames: dict[str, pl.DataFrame] = {}
    expected_paths: set[Path] = set()
    for descriptor in normalized:
        role = str(descriptor["role"])
        if descriptor["availability"] != "available":
            raise ValueError(f"TaxaLens score/pool role {role!r} must be available")
        if descriptor["evidence_maturity_label"] != _MATURITY[role]:
            raise ValueError(f"TaxaLens score/pool role {role!r} maturity differs")
        filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
        if descriptor["schema_version"] != schema_version:
            raise ValueError(f"TaxaLens score/pool role {role!r} schema differs")
        expected_relative = f"artifacts/{filename}"
        if descriptor["relative_path"] != expected_relative:
            raise ValueError(f"TaxaLens score/pool role {role!r} path differs")
        path = destination / expected_relative
        expected_paths.add(path.resolve())
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"TaxaLens score/pool role {role!r} file is unavailable")
        if path.stat().st_size != descriptor["byte_count"]:
            raise ValueError(f"TaxaLens score/pool role {role!r} byte count differs")
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"TaxaLens score/pool role {role!r} checksum differs")
        frame = pl.read_parquet(path)
        if frame.height != descriptor["row_count"]:
            raise ValueError(f"TaxaLens score/pool role {role!r} row count differs")
        if _semantic_fingerprint(role, frame) != descriptor["semantic_fingerprint"]:
            raise ValueError(f"TaxaLens score/pool role {role!r} identity differs")
        if _parent_fingerprints(role, frame) != descriptor["parent_fingerprints"]:
            raise ValueError(f"TaxaLens score/pool role {role!r} lineage differs")
        frames[role] = frame
    artifact_directory = destination / "artifacts"
    if not artifact_directory.is_dir():
        raise ValueError("TaxaLens score/pool artifact directory is unavailable")
    entries = tuple(artifact_directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("TaxaLens score/pool artifact directory has unsafe entries")
    actual_paths = {path.resolve() for path in entries}
    if actual_paths != expected_paths:
        raise ValueError("TaxaLens score/pool artifact file set differs")
    _validate_frames(frames)


def _frames(**values: pl.DataFrame) -> dict[str, pl.DataFrame]:
    for role, frame in values.items():
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"TaxaLens score/pool role {role!r} must be a DataFrame")
    return dict(values)


def _validate_frames(frames: Mapping[str, pl.DataFrame]) -> None:
    if set(frames) != set(TAXALENS_SCORE_POOL_ROLES):
        raise ValueError("TaxaLens score/pool frame roles differ")
    for role, frame in frames.items():
        if frame.is_empty():
            raise ValueError(f"TaxaLens score/pool role {role!r} must not be empty")
    validate_dynamic_reference_pool_artifacts(
        frames["pool_plans"],
        frames["pool_members"],
        frames["pool_summaries"],
    )
    validate_family_geo_candidate_sets(frames["candidate_sets"])
    validate_dynamic_pool_score_artifacts(
        frames["candidate_scores"],
        frames["photo_summaries"],
        plans=frames["pool_plans"],
        candidate_sets=frames["candidate_sets"],
    )


def _write_artifact(
    role: str,
    frame: pl.DataFrame,
    artifact_directory: Path,
) -> dict[str, object]:
    filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
    path = write_parquet(
        frame,
        artifact_directory / filename,
        overwrite=False,
    )
    return {
        "role": role,
        "availability": "available",
        "unavailable_reason": None,
        "relative_path": f"artifacts/{filename}",
        "media_type": "application/vnd.apache.parquet",
        "schema_version": schema_version,
        "semantic_fingerprint": _semantic_fingerprint(role, frame),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": frame.height,
        "parent_fingerprints": _parent_fingerprints(role, frame),
        "evidence_maturity_label": _MATURITY[role],
    }


def _semantic_fingerprint(role: str, frame: pl.DataFrame) -> str:
    fingerprint_field = _FRAME_FINGERPRINT_FIELDS[role]
    return canonical_semantic_fingerprint(
        {
            "role": role,
            "schema_version": TAXALENS_ROLE_DEFAULTS[role][1],
            "fingerprint_field": fingerprint_field,
            "row_fingerprints": frame[fingerprint_field].to_list(),
        }
    )


def _parent_fingerprints(role: str, frame: pl.DataFrame) -> list[str]:
    values: set[str] = set()
    for field in _PARENT_FINGERPRINT_FIELDS[role]:
        if field not in frame.columns:
            continue
        for value in frame[field].to_list():
            if value is None:
                continue
            if isinstance(value, list):
                values.update(str(item) for item in value)
            else:
                values.add(str(value))
    return sorted(values)


__all__ = [
    "TAXALENS_SCORE_POOL_ROLES",
    "TaxaLensScorePoolExport",
    "export_taxalens_score_pool_evidence",
    "validate_taxalens_score_pool_export",
]
