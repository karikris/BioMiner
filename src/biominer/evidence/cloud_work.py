from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import polars as pl

from biominer.evidence.join import build_joined_object_evidence_frame, build_photo_summary_from_joined_evidence
from biominer.species.context import SpeciesContext
from biominer.storage.cloud import CloudStorage
from biominer.workstore.base import WorkStore


@dataclass(frozen=True)
class CloudObjectEvidenceJoinResult:
    frame: pl.DataFrame
    source_shards: tuple[dict[str, Any], ...]
    detection_shards: tuple[dict[str, Any], ...]
    score_shards: tuple[dict[str, Any], ...]
    source_records_seen: int
    detections_seen: int
    scores_seen: int

    @property
    def source_shards_seen(self) -> int:
        return len(self.source_shards)

    @property
    def detection_shards_seen(self) -> int:
        return len(self.detection_shards)

    @property
    def score_shards_seen(self) -> int:
        return len(self.score_shards)


@dataclass(frozen=True)
class CloudPhotoSummaryResult:
    frame: pl.DataFrame
    joined_shards: tuple[dict[str, Any], ...]
    object_evidence_rows_seen: int
    object_occurrence_bin_counts: dict[str, int]

    @property
    def joined_shards_seen(self) -> int:
        return len(self.joined_shards)


def join_object_evidence_from_cloud_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    source_stage: str,
    detection_stage: str,
    score_stage: str,
) -> CloudObjectEvidenceJoinResult:
    """Build joined object evidence from committed shard inventory."""

    source_shards = _candidate_shards(
        workstore,
        job_name=job_name,
        stage=source_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    detection_shards = _candidate_shards(
        workstore,
        job_name=job_name,
        stage=detection_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    score_shards = _candidate_shards(
        workstore,
        job_name=job_name,
        stage=score_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    missing = []
    if not source_shards:
        missing.append("source_records")
    if not detection_shards:
        missing.append("object_detections")
    if not score_shards:
        missing.append("object_scores")
    if missing:
        raise FileNotFoundError(", ".join(missing))

    canonical = _read_shards(storage, source_shards)
    detections = _read_shards(storage, detection_shards)
    scores = _read_shards(storage, score_shards)
    joined = build_joined_object_evidence_frame(
        canonical_source_records=canonical,
        object_detections=detections,
        object_scores=scores,
    )
    return CloudObjectEvidenceJoinResult(
        frame=joined,
        source_shards=tuple(source_shards),
        detection_shards=tuple(detection_shards),
        score_shards=tuple(score_shards),
        source_records_seen=canonical.height,
        detections_seen=detections.height,
        scores_seen=scores.height,
    )


def summarize_photo_evidence_from_cloud_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    joined_stage: str,
    species_context: SpeciesContext | None = None,
) -> CloudPhotoSummaryResult:
    """Build photo summaries from committed joined object-evidence shards."""

    joined_shards = _candidate_shards(
        workstore,
        job_name=job_name,
        stage=joined_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    if not joined_shards:
        raise FileNotFoundError("object_evidence")
    joined = _read_shards(storage, joined_shards)
    summary = build_photo_summary_from_joined_evidence(joined, species_context=species_context)
    return CloudPhotoSummaryResult(
        frame=summary,
        joined_shards=tuple(joined_shards),
        object_evidence_rows_seen=joined.height,
        object_occurrence_bin_counts=_value_counts(joined, "occurrence_bin"),
    )


def join_evidence_batch_id(result: CloudObjectEvidenceJoinResult) -> str:
    return _stable_hash(
        {
            "source": _shard_identities(result.source_shards),
            "detections": _shard_identities(result.detection_shards),
            "scores": _shard_identities(result.score_shards),
        }
    )


def photo_summary_batch_id(result: CloudPhotoSummaryResult) -> str:
    return _stable_hash({"joined": _shard_identities(result.joined_shards)})


def _candidate_shards(
    workstore: WorkStore,
    *,
    job_name: str,
    stage: str,
    registry_version: str | None,
    run_id: str,
) -> list[dict[str, Any]]:
    return workstore.list_candidate_shards(
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        run_id=run_id,
    )


def _read_shards(storage: CloudStorage, shards: list[dict[str, Any]]) -> pl.DataFrame:
    frames = [storage.read_parquet(str(shard["uri"])) for shard in shards]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _shard_identities(shards: tuple[dict[str, Any], ...]) -> list[str]:
    return [str(shard.get("shard_id") or shard.get("uri") or "") for shard in shards]


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CloudObjectEvidenceJoinResult",
    "CloudPhotoSummaryResult",
    "join_evidence_batch_id",
    "join_object_evidence_from_cloud_shards",
    "photo_summary_batch_id",
    "summarize_photo_evidence_from_cloud_shards",
]
