from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib

import polars as pl

from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_compacted_evidence_uri, build_report_uri
from biominer.storage.uri import is_cloud_uri, normalize_local_uri
from biominer.workstore.base import WorkStore
from biominer.workstore.keys import uri_shard_id


MB = 1024 * 1024
COMPACTED_PARQUET_TARGET_MB = 256
COMPACTED_PARQUET_MAX_MB = 512


@dataclass(frozen=True)
class CompactionCandidate:
    shard_id: str
    uri: str
    row_count: int | None
    byte_count: int | None
    stage: str
    registry_version: str | None
    run_id: str | None = None


@dataclass(frozen=True)
class CompactionGroup:
    group_id: str
    output_uri: str
    source_shards: tuple[CompactionCandidate, ...]
    estimated_input_bytes: int | None
    estimated_input_rows: int | None


@dataclass(frozen=True)
class CompactionPlan:
    compaction_run_id: str
    source_stage: str
    output_stage: str
    registry_version: str | None
    groups: tuple[CompactionGroup, ...]


@dataclass(frozen=True)
class CompactionResult:
    compaction_run_id: str
    source_stage: str
    output_stage: str
    registry_version: str | None
    groups_planned: int
    groups_written: int
    source_shards_consumed: int
    output_shards_written: int
    rows_written: int
    bytes_written: int | None
    skipped_groups: int
    failed_groups: int


def plan_compaction_groups(
    candidates: list[CompactionCandidate],
    *,
    base_prefix: str,
    source_stage: str,
    output_stage: str,
    registry_version: str | None,
    compaction_run_id: str,
    target_file_mb: int = COMPACTED_PARQUET_TARGET_MB,
    max_file_mb: int = COMPACTED_PARQUET_MAX_MB,
) -> CompactionPlan:
    target_bytes = target_file_mb * MB
    max_bytes = max_file_mb * MB
    groups: list[CompactionGroup] = []
    current: list[CompactionCandidate] = []
    current_bytes: int | None = 0
    current_rows: int | None = 0

    for candidate in sorted(candidates, key=lambda item: (item.uri, item.shard_id)):
        candidate_bytes = candidate.byte_count
        candidate_rows = candidate.row_count
        if (
            current
            and current_bytes is not None
            and candidate_bytes is not None
            and current_bytes + candidate_bytes > max_bytes
        ):
            groups.append(_build_group(groups, current, current_bytes, current_rows, base_prefix, source_stage, registry_version, compaction_run_id))
            current = []
            current_bytes = 0
            current_rows = 0

        current.append(candidate)
        current_bytes = None if current_bytes is None or candidate_bytes is None else current_bytes + candidate_bytes
        current_rows = None if current_rows is None or candidate_rows is None else current_rows + candidate_rows
        if current_bytes is not None and current_bytes >= target_bytes:
            groups.append(_build_group(groups, current, current_bytes, current_rows, base_prefix, source_stage, registry_version, compaction_run_id))
            current = []
            current_bytes = 0
            current_rows = 0

    if current:
        groups.append(_build_group(groups, current, current_bytes, current_rows, base_prefix, source_stage, registry_version, compaction_run_id))

    return CompactionPlan(
        compaction_run_id=compaction_run_id,
        source_stage=source_stage,
        output_stage=output_stage,
        registry_version=registry_version,
        groups=tuple(groups),
    )


def compact_parquet_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore | None,
    input_prefix: str | None,
    output_prefix: str,
    job_name: str,
    source_stage: str,
    output_stage: str | None = None,
    registry_version: str | None = None,
    run_id: str | None = None,
    compaction_run_id: str | None = None,
    target_file_mb: int = COMPACTED_PARQUET_TARGET_MB,
    max_file_mb: int = COMPACTED_PARQUET_MAX_MB,
    dedupe_keys: list[str] | None = None,
    schema_mode: str = "strict",
    dry_run: bool = False,
) -> CompactionResult:
    resolved_output_stage = output_stage or f"{source_stage}_compacted"
    resolved_run_id = compaction_run_id or _default_compaction_run_id()
    candidates = _candidate_shards(
        storage=storage,
        workstore=workstore,
        input_prefix=input_prefix,
        job_name=job_name,
        source_stage=source_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    plan = plan_compaction_groups(
        candidates,
        base_prefix=output_prefix,
        source_stage=source_stage,
        output_stage=resolved_output_stage,
        registry_version=registry_version,
        compaction_run_id=resolved_run_id,
        target_file_mb=target_file_mb,
        max_file_mb=max_file_mb,
    )
    groups_written = 0
    rows_written = 0
    source_shards_consumed = 0
    bytes_written_total = 0
    byte_count_known = True
    skipped_groups = 0
    failed_groups = 0

    if not dry_run:
        for group in plan.groups:
            if storage.exists(group.output_uri):
                skipped_groups += 1
                continue
            try:
                frame = _read_group(storage, group, schema_mode=schema_mode)
                if dedupe_keys:
                    _validate_dedupe_keys(frame, dedupe_keys)
                    frame = frame.unique(subset=dedupe_keys, maintain_order=True)
            except Exception:
                failed_groups += 1
                continue
            storage.write_parquet_shard(group.output_uri, frame)
            byte_count = _local_byte_count(group.output_uri)
            checksum = _local_sha256(group.output_uri)
            if byte_count is None:
                byte_count_known = False
            else:
                bytes_written_total += byte_count
            groups_written += 1
            rows_written += frame.height
            source_shards_consumed += len(group.source_shards)
            if workstore is not None:
                workstore.register_compaction_output(
                    compaction_run_id=resolved_run_id,
                    output_shard_id=uri_shard_id(group.output_uri),
                    output_uri=group.output_uri,
                    source_shards=[candidate.__dict__ for candidate in group.source_shards],
                    job_name=job_name,
                    source_stage=source_stage,
                    output_stage=resolved_output_stage,
                    registry_version=registry_version,
                    row_count=frame.height,
                    byte_count=byte_count,
                    checksum=checksum,
                    metadata={
                        "schema_mode": schema_mode,
                        "dedupe_keys": dedupe_keys or [],
                        "estimated_input_bytes": group.estimated_input_bytes,
                    },
                )

    result = CompactionResult(
        compaction_run_id=resolved_run_id,
        source_stage=source_stage,
        output_stage=resolved_output_stage,
        registry_version=registry_version,
        groups_planned=len(plan.groups),
        groups_written=groups_written,
        source_shards_consumed=source_shards_consumed,
        output_shards_written=groups_written,
        rows_written=rows_written,
        bytes_written=bytes_written_total if byte_count_known else None,
        skipped_groups=skipped_groups,
        failed_groups=failed_groups,
    )
    _write_report(
        storage=storage,
        output_prefix=output_prefix,
        result=result,
        input_prefix=input_prefix,
        target_file_mb=target_file_mb,
        max_file_mb=max_file_mb,
        dedupe_keys=dedupe_keys,
        schema_mode=schema_mode,
        dry_run=dry_run,
    )
    return result


def _build_group(
    existing_groups: list[CompactionGroup],
    source_shards: list[CompactionCandidate],
    estimated_input_bytes: int | None,
    estimated_input_rows: int | None,
    base_prefix: str,
    source_stage: str,
    registry_version: str | None,
    compaction_run_id: str,
) -> CompactionGroup:
    part_id = len(existing_groups) + 1
    output_uri = build_compacted_evidence_uri(
        base_prefix,
        source_stage=source_stage,
        registry_version=registry_version,
        compaction_run_id=compaction_run_id,
        part_id=part_id,
    )
    digest_input = "|".join(candidate.shard_id for candidate in source_shards)
    group_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    return CompactionGroup(
        group_id=f"{compaction_run_id}:{part_id:06d}:{group_digest}",
        output_uri=output_uri,
        source_shards=tuple(source_shards),
        estimated_input_bytes=estimated_input_bytes,
        estimated_input_rows=estimated_input_rows,
    )


def _candidate_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore | None,
    input_prefix: str | None,
    job_name: str,
    source_stage: str,
    registry_version: str | None,
    run_id: str | None,
) -> list[CompactionCandidate]:
    if workstore is not None:
        manifest_candidates = workstore.list_candidate_shards(
            job_name=job_name,
            stage=source_stage,
            registry_version=registry_version,
            run_id=run_id,
        )
        all_manifest_shards = workstore.list_committed_shards(
            job_name=job_name,
            stage=source_stage,
            registry_version=registry_version,
            run_id=run_id,
        )
        if manifest_candidates or all_manifest_shards:
            return [_candidate_from_manifest(row) for row in manifest_candidates]
    if input_prefix is None:
        return []
    return [
        CompactionCandidate(
            shard_id=uri_shard_id(uri),
            uri=uri,
            row_count=_cheap_row_count(storage, uri),
            byte_count=_local_byte_count(uri),
            stage=source_stage,
            registry_version=registry_version,
            run_id=run_id,
        )
        for uri in storage.list_shards(input_prefix)
        if f"stage={source_stage}_compacted" not in uri
    ]


def _candidate_from_manifest(row: dict[str, Any]) -> CompactionCandidate:
    return CompactionCandidate(
        shard_id=str(row["shard_id"]),
        uri=str(row["uri"]),
        row_count=row.get("row_count"),
        byte_count=row.get("byte_count"),
        stage=str(row["stage"]),
        registry_version=row.get("registry_version"),
        run_id=row.get("run_id"),
    )


def _read_group(storage: CloudStorage, group: CompactionGroup, *, schema_mode: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    expected_schema: dict[str, Any] | None = None
    for candidate in group.source_shards:
        frame = storage.read_parquet(candidate.uri)
        if schema_mode == "strict":
            schema = dict(frame.schema)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(f"incompatible Parquet schema in compaction group {group.group_id}: {candidate.uri}")
        elif schema_mode != "diagonal_relaxed":
            raise ValueError(f"unsupported schema_mode: {schema_mode}")
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed" if schema_mode == "diagonal_relaxed" else "vertical")


def _validate_dedupe_keys(frame: pl.DataFrame, dedupe_keys: list[str]) -> None:
    missing = [key for key in dedupe_keys if key not in frame.columns]
    if missing:
        raise ValueError(f"dedupe keys missing from compacted frame: {missing}")


def _cheap_row_count(storage: CloudStorage, uri: str) -> int | None:
    if is_cloud_uri(uri):
        return None
    try:
        return int(storage.read_parquet(uri).height)
    except Exception:
        return None


def _local_byte_count(uri: str) -> int | None:
    if is_cloud_uri(uri):
        return None
    try:
        path = normalize_local_uri(uri)
    except ValueError:
        return None
    return Path(path).stat().st_size if Path(path).exists() else None


def _local_sha256(uri: str) -> str | None:
    if is_cloud_uri(uri):
        return None
    try:
        path = normalize_local_uri(uri)
    except ValueError:
        return None
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_report(
    *,
    storage: CloudStorage,
    output_prefix: str,
    result: CompactionResult,
    input_prefix: str | None,
    target_file_mb: int,
    max_file_mb: int,
    dedupe_keys: list[str] | None,
    schema_mode: str,
    dry_run: bool,
) -> None:
    report_uri = build_report_uri(
        output_prefix,
        run_id=result.compaction_run_id,
        report_name=f"compaction_{result.source_stage}",
    )
    storage.write_json(
        report_uri,
        {
            **result.__dict__,
            "target_file_mb": target_file_mb,
            "max_file_mb": max_file_mb,
            "input_prefix": input_prefix,
            "output_prefix": output_prefix,
            "dedupe_keys": dedupe_keys or [],
            "schema_mode": schema_mode,
            "dry_run": dry_run,
        },
    )


def _default_compaction_run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
