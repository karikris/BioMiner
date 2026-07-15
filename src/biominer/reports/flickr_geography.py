"""Compact workload reporting for Flickr candidate geography clusters."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.flickr_fetch.geographic_clustering import (
    GLOBAL_FALLBACK_CLUSTER_IDS,
    NO_GEO_CLUSTER_ID,
    UNASSIGNED_GEO_CLUSTER_ID,
)
from biominer.reports.flickr_fetch import current_git_sha


FLICKR_GEO_WORKLOAD_REPORT_SCHEMA_VERSION = "flickr-geographic-workload-report-v1.1.0"
FLICKR_GEO_WORKLOAD_METRICS_FILE = "flickr_geographic_workload.json"
FLICKR_GEO_WORKLOAD_SUMMARY_FILE = "flickr_geographic_workload.md"
REFERENCE_QUOTA_POLICY_VERSION = "minimum-plus-sqrt-candidates-v1.1.0"


@dataclass(frozen=True, slots=True)
class ReferenceQuotaConfig:
    total_reference_quota: int = 50
    minimum_per_populated_cluster: int = 2
    minimum_candidate_images: int = 10
    policy_version: str = REFERENCE_QUOTA_POLICY_VERSION

    def __post_init__(self) -> None:
        total = _positive_int(
            self.total_reference_quota,
            field_name="total_reference_quota",
        )
        minimum = _nonnegative_int(
            self.minimum_per_populated_cluster,
            field_name="minimum_per_populated_cluster",
        )
        populated = _positive_int(
            self.minimum_candidate_images,
            field_name="minimum_candidate_images",
        )
        policy = _required_text(self.policy_version, field_name="policy_version")
        object.__setattr__(self, "total_reference_quota", total)
        object.__setattr__(self, "minimum_per_populated_cluster", minimum)
        object.__setattr__(self, "minimum_candidate_images", populated)
        object.__setattr__(self, "policy_version", policy)


@dataclass(frozen=True, slots=True)
class FlickrGeoWorkloadReport:
    payload: dict[str, Any]
    markdown: str


def build_flickr_geo_workload_report(
    *,
    geography: pl.DataFrame,
    clusters: pl.DataFrame,
    assignments: pl.DataFrame,
    query_hits: pl.DataFrame | None,
    quota_config: ReferenceQuotaConfig | None = None,
    run_id: str,
    command: Sequence[str],
    started_at: str | datetime,
    ended_at: str | datetime,
    status: str = "completed",
    git_sha: str | None = None,
    pid: int | None = None,
) -> FlickrGeoWorkloadReport:
    context = _validated_context(
        geography=geography,
        clusters=clusters,
        assignments=assignments,
    )
    started = _utc_datetime(started_at, field_name="started_at")
    ended = _utc_datetime(ended_at, field_name="ended_at")
    if ended < started:
        raise ValueError("ended_at must not precede started_at")
    command_parts = [str(part) for part in command]
    if not command_parts or any(not part.strip() for part in command_parts):
        raise ValueError("command must contain nonblank arguments")
    effective_quota_config = quota_config or ReferenceQuotaConfig()
    if not isinstance(effective_quota_config, ReferenceQuotaConfig):
        raise TypeError("quota_config must be a ReferenceQuotaConfig")

    geography_rows = geography.to_dicts()
    assignment_rows = assignments.to_dicts()
    cluster_rows = clusters.to_dicts()
    assignment_counts = Counter(str(row["geo_cluster_id"]) for row in assignment_rows)
    outlier_counts = Counter(
        str(row["geo_cluster_id"]) for row in assignment_rows if row.get("outlier") is True
    )
    quotas = allocate_implied_reference_quotas(
        clusters,
        config=effective_quota_config,
    )
    quota_by_cluster = {
        str(row["geo_cluster_id"]): row for row in quotas["by_cluster"]
    }
    cluster_summary = [
        _cluster_summary_row(
            row,
            outlier_count=outlier_counts[str(row["geo_cluster_id"])],
            quota=quota_by_cluster[str(row["geo_cluster_id"])],
        )
        for row in sorted(cluster_rows, key=lambda item: str(item["geo_cluster_id"]))
    ]
    country_summary = _records_by_country(geography_rows)
    query_summary = _query_summaries(
        query_hits,
        assignment_rows=assignment_rows,
        assignment_counts=assignment_counts,
    )
    total_records = len(geography_rows)
    geotagged = sum(row.get("geotag_available") is True for row in geography_rows)
    missing_coordinate_count = total_records - geotagged
    no_geo_assignment_count = assignment_counts[NO_GEO_CLUSTER_ID]
    if no_geo_assignment_count != missing_coordinate_count:
        raise ValueError(
            "no_geo assignments must correspond exactly to records without usable "
            "coordinates"
        )
    unassigned_geo_count = assignment_counts[UNASSIGNED_GEO_CLUSTER_ID]
    outlier_count = sum(outlier_counts.values())
    located_clusters = [
        row
        for row in cluster_rows
        if str(row["geo_cluster_id"]) not in GLOBAL_FALLBACK_CLUSTER_IDS
    ]

    payload: dict[str, Any] = {
        "schema_version": FLICKR_GEO_WORKLOAD_REPORT_SCHEMA_VERSION,
        "run_id": _required_text(run_id, field_name="run_id"),
        "command": command_parts,
        "git_sha": git_sha or current_git_sha(),
        "pid": pid or os.getpid(),
        "status": _required_text(status, field_name="status"),
        "start_time": _timestamp(started),
        "end_time": _timestamp(ended),
        "elapsed_seconds": (ended - started).total_seconds(),
        "target_accepted_taxon_key": context["target_accepted_taxon_key"],
        "cluster_configuration_hash": context["cluster_configuration_hash"],
        "geography_config_fingerprint": context["geography_config_fingerprint"],
        "candidate_distribution_only": True,
        "inputs": {
            "geography_rows": geography.height,
            "cluster_rows": clusters.height,
            "assignment_rows": assignments.height,
            "query_hit_rows": query_hits.height if query_hits is not None else "not_instrumented",
        },
        "storage_bytes": {
            "geography_parquet_bytes": "not_instrumented",
            "cluster_parquet_bytes": "not_instrumented",
            "assignment_parquet_bytes": "not_instrumented",
            "query_hit_bytes": "not_instrumented",
            "report_bytes": "not_instrumented",
        },
        "throughput": {
            "candidate_records_per_second": (
                total_records / (ended - started).total_seconds()
                if ended > started
                else None
            ),
        },
        "summary": {
            "candidate_record_count": total_records,
            "geotagged_record_count": geotagged,
            "geotagged_percentage": _percentage(geotagged, total_records),
            "located_cluster_count": len(located_clusters),
            "missing_coordinate_record_count": missing_coordinate_count,
            "missing_coordinate_percentage": _percentage(
                missing_coordinate_count,
                total_records,
            ),
            "unassigned_geotagged_record_count": unassigned_geo_count,
            "unassigned_geotagged_percentage": _percentage(
                unassigned_geo_count,
                total_records,
            ),
            "outlier_record_count": outlier_count,
            "outlier_percentage": _percentage(outlier_count, total_records),
        },
        "clusters": cluster_summary,
        "records_by_country": country_summary,
        "records_by_query_tier_and_cluster": query_summary["by_query_tier_and_cluster"],
        "records_by_search_term_and_cluster": query_summary["by_search_term_and_cluster"],
        "query_provenance": {
            "status": query_summary["status"],
            "query_hit_link_count": query_summary["query_hit_link_count"],
        },
        "reference_quota": quotas,
        "artifacts": {
            "metrics": FLICKR_GEO_WORKLOAD_METRICS_FILE,
            "summary": FLICKR_GEO_WORKLOAD_SUMMARY_FILE,
        },
    }
    return FlickrGeoWorkloadReport(
        payload=payload,
        markdown=flickr_geo_workload_markdown(payload),
    )


def allocate_implied_reference_quotas(
    clusters: pl.DataFrame,
    *,
    config: ReferenceQuotaConfig,
) -> dict[str, Any]:
    if not isinstance(clusters, pl.DataFrame):
        raise TypeError("clusters must be a Polars DataFrame")
    if not isinstance(config, ReferenceQuotaConfig):
        raise TypeError("config must be a ReferenceQuotaConfig")
    required = {"geo_cluster_id", "member_image_count"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"cluster quota input is missing required columns: {missing}")
    rows = sorted(clusters.to_dicts(), key=lambda row: str(row["geo_cluster_id"]))
    counts: dict[str, int] = {}
    for row in rows:
        cluster_id = _required_text(row.get("geo_cluster_id"), field_name="geo_cluster_id")
        if cluster_id in counts:
            raise ValueError(f"duplicate geo_cluster_id in quota input: {cluster_id}")
        counts[cluster_id] = _nonnegative_int(
            row.get("member_image_count"),
            field_name="member_image_count",
        )
    eligible = [
        cluster_id
        for cluster_id, count in counts.items()
        if cluster_id not in GLOBAL_FALLBACK_CLUSTER_IDS
        and count >= config.minimum_candidate_images
    ]
    allocation = {cluster_id: 0 for cluster_id in counts}
    if eligible:
        base = min(
            config.minimum_per_populated_cluster,
            config.total_reference_quota // len(eligible),
        )
        for cluster_id in eligible:
            allocation[cluster_id] = base
        remaining = config.total_reference_quota - base * len(eligible)
        weights = {cluster_id: math.sqrt(counts[cluster_id]) for cluster_id in eligible}
        weight_total = sum(weights.values())
        if remaining and weight_total:
            exact = {
                cluster_id: remaining * weights[cluster_id] / weight_total
                for cluster_id in eligible
            }
            floors = {cluster_id: math.floor(value) for cluster_id, value in exact.items()}
            for cluster_id, value in floors.items():
                allocation[cluster_id] += value
            remainder = remaining - sum(floors.values())
            order = sorted(
                eligible,
                key=lambda cluster_id: (
                    -(exact[cluster_id] - floors[cluster_id]),
                    cluster_id,
                ),
            )
            for cluster_id in order[:remainder]:
                allocation[cluster_id] += 1

    by_cluster = [
        {
            "geo_cluster_id": cluster_id,
            "candidate_image_count": counts[cluster_id],
            "sufficiently_populated": cluster_id in eligible,
            "allocation_weight": (
                math.sqrt(counts[cluster_id]) if cluster_id in eligible else 0.0
            ),
            "reference_quota_implied": allocation[cluster_id],
            "configured_minimum": (
                config.minimum_per_populated_cluster if cluster_id in eligible else 0
            ),
            "minimum_shortfall": (
                max(
                    0,
                    config.minimum_per_populated_cluster - allocation[cluster_id],
                )
                if cluster_id in eligible
                else 0
            ),
        }
        for cluster_id in sorted(counts)
    ]
    allocated = sum(allocation.values())
    configuration = {
        "policy_version": config.policy_version,
        "total_reference_quota": config.total_reference_quota,
        "minimum_per_populated_cluster": config.minimum_per_populated_cluster,
        "minimum_candidate_images": config.minimum_candidate_images,
        "excluded_fallback_clusters": sorted(GLOBAL_FALLBACK_CLUSTER_IDS),
    }
    return {
        "configuration": configuration,
        "configuration_fingerprint": _sha256_json(configuration),
        "eligible_cluster_count": len(eligible),
        "allocated_reference_quota": allocated,
        "unallocated_reference_quota": config.total_reference_quota - allocated,
        "minimum_fully_satisfied": all(
            row["minimum_shortfall"] == 0 for row in by_cluster
        ),
        "by_cluster": by_cluster,
    }


def write_flickr_geo_workload_report(
    report: FlickrGeoWorkloadReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    if not isinstance(report, FlickrGeoWorkloadReport):
        raise TypeError("report must be a FlickrGeoWorkloadReport")
    output = Path(output_dir)
    metrics_path = output / FLICKR_GEO_WORKLOAD_METRICS_FILE
    summary_path = output / FLICKR_GEO_WORKLOAD_SUMMARY_FILE
    _write_text_atomic(
        json.dumps(report.payload, indent=2, sort_keys=True) + "\n",
        metrics_path,
    )
    _write_text_atomic(report.markdown, summary_path)
    return {"metrics": metrics_path, "summary": summary_path}


def flickr_geo_workload_markdown(payload: Mapping[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    clusters = _mapping_rows(payload.get("clusters"))
    countries = _mapping_rows(payload.get("records_by_country"))
    quota = _mapping(payload.get("reference_quota"))
    quota_rows = _mapping_rows(quota.get("by_cluster"))
    tier_rows = _optional_mapping_rows(payload.get("records_by_query_tier_and_cluster"))
    term_rows = _optional_mapping_rows(payload.get("records_by_search_term_and_cluster"))
    lines = [
        "# Flickr Geographic Workload",
        "",
        (
            "Flickr rows are search candidates. Cluster counts are candidate-distribution "
            "evidence, not verified occurrences."
        ),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidate records | {_display(summary.get('candidate_record_count'))} |",
        (
            f"| Geotagged | {_display(summary.get('geotagged_record_count'))} "
            f"({_display(summary.get('geotagged_percentage'))}%) |"
        ),
        f"| Located clusters | {_display(summary.get('located_cluster_count'))} |",
        (
            "| Missing coordinates | "
            f"{_display(summary.get('missing_coordinate_record_count'))} "
            f"({_display(summary.get('missing_coordinate_percentage'))}%) |"
        ),
        (
            "| Geotagged but unassigned | "
            f"{_display(summary.get('unassigned_geotagged_record_count'))} "
            f"({_display(summary.get('unassigned_geotagged_percentage'))}%) |"
        ),
        (
            f"| Outliers | {_display(summary.get('outlier_record_count'))} "
            f"({_display(summary.get('outlier_percentage'))}%) |"
        ),
        "",
        "## Clusters",
        "",
        "| Cluster | Images | Cells | Outliers | Countries | Implied references |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in clusters:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(row.get("geo_cluster_id")),
                    _display(row.get("member_image_count")),
                    _display(row.get("member_cell_count")),
                    _display(row.get("outlier_count")),
                    _markdown_text(", ".join(str(value) for value in row.get("countries") or []) or "unknown"),
                    _display(row.get("reference_quota_implied")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Countries",
            "",
            "| Country | Records | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in countries:
        lines.append(
            f"| {_markdown_text(row.get('country_code'))} | "
            f"{_display(row.get('record_count'))} | "
            f"{_display(row.get('percentage'))}% |"
        )
    lines.extend(
        [
            "",
            "## Implied Reference Quota",
            "",
            (
                "Policy: `"
                + _markdown_text(
                    _mapping(quota.get("configuration")).get("policy_version")
                )
                + "`. Allocated "
                + _display(quota.get("allocated_reference_quota"))
                + "; unallocated "
                + _display(quota.get("unallocated_reference_quota"))
                + "."
            ),
            "",
            "| Cluster | Candidates | Eligible | Weight | Quota | Minimum shortfall |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in quota_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(row.get("geo_cluster_id")),
                    _display(row.get("candidate_image_count")),
                    _display(row.get("sufficiently_populated")),
                    _display(row.get("allocation_weight")),
                    _display(row.get("reference_quota_implied")),
                    _display(row.get("minimum_shortfall")),
                ]
            )
            + " |"
        )
    lines.extend(_query_markdown("Query Tiers", tier_rows, key="query_tier"))
    lines.extend(_query_markdown("Search Terms", term_rows, key="search_term"))
    return "\n".join(lines) + "\n"


def _validated_context(
    *,
    geography: pl.DataFrame,
    clusters: pl.DataFrame,
    assignments: pl.DataFrame,
) -> dict[str, str]:
    for name, frame in (
        ("geography", geography),
        ("clusters", clusters),
        ("assignments", assignments),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a Polars DataFrame")
    _require_columns(
        geography,
        "geography",
        {
            "source",
            "flickr_photo_id",
            "source_record_hash",
            "geotag_available",
            "country_code",
            "geography_config_fingerprint",
        },
    )
    _require_columns(
        clusters,
        "clusters",
        {
            "geo_cluster_id",
            "target_accepted_taxon_key",
            "member_image_count",
            "member_cell_count",
            "countries",
            "source_resolution",
            "cluster_configuration_hash",
            "candidate_distribution_only",
        },
    )
    _require_columns(
        assignments,
        "assignments",
        {
            "source",
            "flickr_photo_id",
            "source_record_hash",
            "target_accepted_taxon_key",
            "geo_cluster_id",
            "outlier",
            "cluster_configuration_hash",
        },
    )
    geography_by_identity = _identity_rows(geography, artifact="geography")
    assignment_by_identity = _identity_rows(assignments, artifact="assignments")
    if set(geography_by_identity) != set(assignment_by_identity):
        missing_assignments = sorted(set(geography_by_identity) - set(assignment_by_identity))
        unknown_assignments = sorted(set(assignment_by_identity) - set(geography_by_identity))
        raise ValueError(
            "geography and assignment identities differ: "
            f"missing_assignments={missing_assignments[:5]}, "
            f"unknown_assignments={unknown_assignments[:5]}"
        )
    for identity, geography_row in geography_by_identity.items():
        assignment_row = assignment_by_identity[identity]
        if str(geography_row["source_record_hash"]) != str(
            assignment_row["source_record_hash"]
        ):
            raise ValueError(f"source_record_hash mismatch for {identity[0]}:{identity[1]}")

    target_key = _single_value(
        [
            *clusters["target_accepted_taxon_key"].to_list(),
            *assignments["target_accepted_taxon_key"].to_list(),
        ],
        field_name="target_accepted_taxon_key",
    )
    configuration_hash = _single_value(
        [
            *clusters["cluster_configuration_hash"].to_list(),
            *assignments["cluster_configuration_hash"].to_list(),
        ],
        field_name="cluster_configuration_hash",
    )
    geography_fingerprint = _single_value(
        geography["geography_config_fingerprint"].to_list(),
        field_name="geography_config_fingerprint",
    )
    if clusters.filter(~pl.col("candidate_distribution_only").fill_null(False)).height:
        raise ValueError("all Flickr clusters must be candidate_distribution_only")

    cluster_rows = clusters.to_dicts()
    cluster_ids = [str(row["geo_cluster_id"]) for row in cluster_rows]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster artifact contains duplicate geo_cluster_id values")
    known_clusters = set(cluster_ids)
    assignment_counts = Counter(str(value) for value in assignments["geo_cluster_id"])
    unknown_clusters = sorted(set(assignment_counts) - known_clusters)
    if unknown_clusters:
        raise ValueError(f"assignments reference unknown clusters: {unknown_clusters}")
    for row in cluster_rows:
        cluster_id = str(row["geo_cluster_id"])
        if int(row["member_image_count"]) != assignment_counts[cluster_id]:
            raise ValueError(f"member_image_count mismatch for cluster {cluster_id}")
    return {
        "target_accepted_taxon_key": target_key,
        "cluster_configuration_hash": configuration_hash,
        "geography_config_fingerprint": geography_fingerprint,
    }


def _query_summaries(
    query_hits: pl.DataFrame | None,
    *,
    assignment_rows: Sequence[Mapping[str, Any]],
    assignment_counts: Mapping[str, int],
) -> dict[str, object]:
    if query_hits is None:
        return {
            "status": "not_instrumented",
            "query_hit_link_count": "not_instrumented",
            "by_query_tier_and_cluster": "not_instrumented",
            "by_search_term_and_cluster": "not_instrumented",
        }
    if not isinstance(query_hits, pl.DataFrame):
        raise TypeError("query_hits must be a Polars DataFrame or None")
    _require_columns(
        query_hits,
        "query_hits",
        {"source", "flickr_photo_id", "query_tier", "search_term"},
    )
    cluster_by_identity = {
        (str(row["source"]), str(row["flickr_photo_id"])): str(row["geo_cluster_id"])
        for row in assignment_rows
    }
    tier_records: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    term_records: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in query_hits.iter_rows(named=True):
        identity = (str(row["source"]), str(row["flickr_photo_id"]))
        if identity not in cluster_by_identity:
            raise ValueError(
                f"query hit references unknown candidate {identity[0]}:{identity[1]}"
            )
        tier = _required_text(row.get("query_tier"), field_name="query_tier")
        term = _required_text(row.get("search_term"), field_name="search_term")
        cluster_id = cluster_by_identity[identity]
        tier_records[(tier, cluster_id)].add(identity)
        term_records[(term, cluster_id)].add(identity)
    tier_rows = [
        {
            "query_tier": tier,
            "geo_cluster_id": cluster_id,
            "record_count": len(identities),
            "percentage_of_cluster": _percentage(
                len(identities),
                assignment_counts[cluster_id],
            ),
        }
        for (tier, cluster_id), identities in sorted(tier_records.items())
    ]
    term_rows = [
        {
            "search_term": term,
            "geo_cluster_id": cluster_id,
            "record_count": len(identities),
            "percentage_of_cluster": _percentage(
                len(identities),
                assignment_counts[cluster_id],
            ),
        }
        for (term, cluster_id), identities in sorted(term_records.items())
    ]
    return {
        "status": "instrumented",
        "query_hit_link_count": query_hits.height,
        "by_query_tier_and_cluster": tier_rows,
        "by_search_term_and_cluster": term_rows,
    }


def _cluster_summary_row(
    row: Mapping[str, Any],
    *,
    outlier_count: int,
    quota: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "geo_cluster_id": str(row["geo_cluster_id"]),
        "member_image_count": int(row["member_image_count"]),
        "member_cell_count": int(row["member_cell_count"]),
        "outlier_count": outlier_count,
        "countries": sorted(str(value) for value in row.get("countries") or []),
        "source_resolution": row.get("source_resolution"),
        "candidate_distribution_only": True,
        "reference_quota_implied": int(quota["reference_quota_implied"]),
    }


def _records_by_country(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    counts = Counter(str(row.get("country_code") or "unknown") for row in rows)
    return [
        {
            "country_code": country,
            "record_count": count,
            "percentage": _percentage(count, len(rows)),
        }
        for country, count in sorted(counts.items())
    ]


def _identity_rows(
    frame: pl.DataFrame,
    *,
    artifact: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        identity = (
            _required_text(row.get("source"), field_name="source"),
            _required_text(row.get("flickr_photo_id"), field_name="flickr_photo_id"),
        )
        if identity in rows:
            raise ValueError(f"{artifact} contains duplicate identity {identity[0]}:{identity[1]}")
        rows[identity] = row
    return rows


def _require_columns(
    frame: pl.DataFrame,
    artifact: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")


def _single_value(values: Sequence[object], *, field_name: str) -> str:
    normalized = {_required_text(value, field_name=field_name) for value in values}
    if len(normalized) != 1:
        raise ValueError(f"{field_name} must have exactly one value, got {sorted(normalized)}")
    return next(iter(normalized))


def _query_markdown(
    title: str,
    rows: list[Mapping[str, Any]] | None,
    *,
    key: str,
) -> list[str]:
    lines = ["", f"## {title}", ""]
    if rows is None:
        return [*lines, "not_instrumented", ""]
    lines.extend(
        [
            f"| {title[:-1]} | Cluster | Records | Percentage of cluster |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_markdown_text(row.get(key))} | "
            f"{_markdown_text(row.get('geo_cluster_id'))} | "
            f"{_display(row.get('record_count'))} | "
            f"{_display(row.get('percentage_of_cluster'))}% |"
        )
    return lines


def _optional_mapping_rows(value: object) -> list[Mapping[str, Any]] | None:
    if value == "not_instrumented":
        return None
    return _mapping_rows(value)


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise TypeError("report table must be a list of mappings")
    return list(value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("report section must be a mapping")
    return value


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 6) if denominator else 0.0


def _display(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _markdown_text(value: object) -> str:
    return _display(value).replace("|", "\\|").replace("\n", " ")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: str | datetime, *, field_name: str) -> datetime:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be nonblank")
    return text


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return normalized


def _positive_int(value: object, *, field_name: str) -> int:
    normalized = _nonnegative_int(value, field_name=field_name)
    if normalized < 1:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _write_text_atomic(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "FLICKR_GEO_WORKLOAD_METRICS_FILE",
    "FLICKR_GEO_WORKLOAD_REPORT_SCHEMA_VERSION",
    "FLICKR_GEO_WORKLOAD_SUMMARY_FILE",
    "REFERENCE_QUOTA_POLICY_VERSION",
    "FlickrGeoWorkloadReport",
    "ReferenceQuotaConfig",
    "allocate_implied_reference_quotas",
    "build_flickr_geo_workload_report",
    "flickr_geo_workload_markdown",
    "write_flickr_geo_workload_report",
]
