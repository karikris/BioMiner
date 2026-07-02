from __future__ import annotations

import re

from biominer.storage.uri import join_uri


def safe_path_component(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z-]+", "_", value.casefold()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized or "blank"


def build_raw_flickr_response_uri(
    base_prefix: str,
    *,
    run_id: str,
    query_field: str,
    query_term: str,
    lane: str,
    page: int,
    work_item_id: str,
) -> str:
    return join_uri(
        base_prefix,
        "raw",
        "source=flickr",
        "method=photos_search",
        f"run_id={run_id}",
        f"field={safe_path_component(query_field)}",
        f"term={safe_path_component(query_term)}",
        f"lane={safe_path_component(lane)}",
        f"page={page:06d}",
        f"work_item_id={safe_path_component(work_item_id)}.json",
    )


def build_evidence_shard_uri(
    base_prefix: str,
    *,
    stage: str,
    run_id: str,
    worker_id: str,
    batch_id: str | int,
) -> str:
    batch = f"{batch_id:06d}" if isinstance(batch_id, int) else safe_path_component(str(batch_id))
    if not batch.endswith(".parquet"):
        batch = f"{batch}.parquet"
    return join_uri(
        base_prefix,
        "evidence",
        f"stage={safe_path_component(stage)}",
        f"run_id={run_id}",
        f"worker={safe_path_component(worker_id)}",
        f"batch={batch}",
    )


def build_compacted_evidence_uri(
    base_prefix: str,
    *,
    source_stage: str,
    registry_version: str | None,
    compaction_run_id: str,
    part_id: str | int,
) -> str:
    part = f"{part_id:06d}" if isinstance(part_id, int) else safe_path_component(str(part_id))
    if not part.endswith(".parquet"):
        part = f"{part}.parquet"
    return join_uri(
        base_prefix,
        "evidence",
        f"stage={safe_path_component(source_stage)}_compacted",
        f"registry_version={safe_path_component(registry_version or 'unknown')}",
        f"run_id={compaction_run_id}",
        f"part={part}",
    )


def build_report_uri(
    base_prefix: str,
    *,
    run_id: str,
    report_name: str,
    suffix: str = "json",
) -> str:
    clean_suffix = suffix.removeprefix(".")
    parts = (f"run_id={run_id}", f"{safe_path_component(report_name)}.{clean_suffix}")
    if str(base_prefix).rstrip("/").endswith("/reports") or str(base_prefix).rstrip("/") == "reports":
        return join_uri(base_prefix, *parts)
    return join_uri(base_prefix, "reports", *parts)


def build_registry_version_uri(base_prefix: str, *, registry_version: str, filename: str) -> str:
    return join_uri(base_prefix, "registry", f"version={safe_path_component(registry_version)}", filename.strip("/"))


def build_registry_current_uri(base_prefix: str, *, filename: str) -> str:
    return join_uri(base_prefix, "registry", "current", filename.strip("/"))


def build_registry_current_pointer(
    *,
    registry_version: str,
    registry_prefix: str,
    manifest_uri: str,
    promoted_at: str,
) -> dict[str, str]:
    return {
        "registry_version": registry_version,
        "registry_prefix": registry_prefix,
        "manifest_uri": manifest_uri,
        "promoted_at": promoted_at,
    }
