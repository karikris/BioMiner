from __future__ import annotations

from datetime import UTC, datetime
import json

import polars as pl
import pytest

from biominer.flickr_fetch.geographic_clustering import (
    NO_GEO_CLUSTER_ID,
    UNASSIGNED_GEO_CLUSTER_ID,
    FlickrGeoClusterConfig,
    build_flickr_geo_clusters,
)
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.reports.flickr_geography import (
    FLICKR_GEO_WORKLOAD_METRICS_FILE,
    FLICKR_GEO_WORKLOAD_SUMMARY_FILE,
    ReferenceQuotaConfig,
    allocate_implied_reference_quotas,
    build_flickr_geo_workload_report,
    write_flickr_geo_workload_report,
)


STARTED_AT = datetime(2026, 7, 13, 1, 0, tzinfo=UTC)
ENDED_AT = datetime(2026, 7, 13, 1, 0, 2, tzinfo=UTC)
TARGET_KEY = "gbif:2734918"


def _record(
    photo_id: str,
    latitude: float | None,
    longitude: float | None,
    *,
    country_code: str | None = None,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:{photo_id}",
        "latitude": latitude,
        "longitude": longitude,
        "accuracy": 16 if latitude is not None and longitude is not None else None,
        "country_code": country_code,
    }


def _artifacts():
    geography = build_flickr_geography_frame(
        [
            _record("brisbane-1", -27.4705, 153.026, country_code="AU"),
            _record("brisbane-2", -27.471, 153.027, country_code="AU"),
            _record("sydney-1", -33.8688, 151.2093, country_code="AU"),
            _record("sydney-2", -33.869, 151.21, country_code="AU"),
            _record("missing", None, None),
        ]
    )
    clustered = build_flickr_geo_clusters(
        geography,
        target_accepted_taxon_key=TARGET_KEY,
        config=FlickrGeoClusterConfig(minimum_cluster_images=2),
        created_at=STARTED_AT,
    )
    query_hits = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "brisbane-1",
                "query_tier": "species_scientific_tags",
                "search_term": "Papilio demoleus",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "brisbane-2",
                "query_tier": "species_common_tags",
                "search_term": "lime butterfly",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "sydney-1",
                "query_tier": "species_scientific_tags",
                "search_term": "Papilio demoleus",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "sydney-2",
                "query_tier": "species_scientific_tags",
                "search_term": "Papilio demoleus",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "missing",
                "query_tier": "broad_text",
                "search_term": "butterfly",
            },
        ]
    )
    return geography, clustered.clusters, clustered.assignments, query_hits


def _report(query_hits: pl.DataFrame | None = None):
    geography, clusters, assignments, default_query_hits = _artifacts()
    return build_flickr_geo_workload_report(
        geography=geography,
        clusters=clusters,
        assignments=assignments,
        query_hits=default_query_hits if query_hits is None else query_hits,
        quota_config=ReferenceQuotaConfig(
            total_reference_quota=20,
            minimum_per_populated_cluster=5,
            minimum_candidate_images=2,
        ),
        run_id="phase2-report",
        command=["biominer", "flickr", "report-geography"],
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        git_sha="abc123",
        pid=42,
    )


def test_reports_candidate_geography_clusters_queries_and_implied_quotas() -> None:
    report = _report()
    payload = report.payload

    assert payload["candidate_distribution_only"] is True
    assert payload["target_accepted_taxon_key"] == TARGET_KEY
    assert payload["elapsed_seconds"] == 2.0
    assert payload["summary"] == {
        "candidate_record_count": 5,
        "geotagged_record_count": 4,
        "geotagged_percentage": 80.0,
        "located_cluster_count": 2,
        "missing_coordinate_record_count": 1,
        "missing_coordinate_percentage": 20.0,
        "unassigned_geotagged_record_count": 0,
        "unassigned_geotagged_percentage": 0.0,
        "outlier_record_count": 0,
        "outlier_percentage": 0.0,
    }
    assert payload["records_by_country"] == [
        {"country_code": "AU", "record_count": 4, "percentage": 80.0},
        {"country_code": "unknown", "record_count": 1, "percentage": 20.0},
    ]
    located = [
        row for row in payload["clusters"] if row["geo_cluster_id"] != NO_GEO_CLUSTER_ID
    ]
    assert len(located) == 2
    assert all(row["member_image_count"] == 2 for row in located)
    assert all(row["reference_quota_implied"] == 10 for row in located)
    no_geo = next(
        row for row in payload["clusters"] if row["geo_cluster_id"] == NO_GEO_CLUSTER_ID
    )
    assert no_geo["reference_quota_implied"] == 0
    assert payload["reference_quota"]["allocated_reference_quota"] == 20
    assert payload["reference_quota"]["minimum_fully_satisfied"] is True
    assert payload["query_provenance"] == {
        "status": "instrumented",
        "query_hit_link_count": 5,
    }
    assert sum(
        row["record_count"]
        for row in payload["records_by_query_tier_and_cluster"]
        if row["query_tier"] == "species_scientific_tags"
    ) == 3
    assert sum(
        row["record_count"]
        for row in payload["records_by_search_term_and_cluster"]
        if row["search_term"] == "Papilio demoleus"
    ) == 3
    assert "search candidates" in report.markdown
    assert "not verified occurrences" in report.markdown


def test_report_is_deterministic_for_reordered_query_hits() -> None:
    geography, clusters, assignments, query_hits = _artifacts()
    first = build_flickr_geo_workload_report(
        geography=geography,
        clusters=clusters,
        assignments=assignments,
        query_hits=query_hits,
        quota_config=ReferenceQuotaConfig(minimum_candidate_images=2),
        run_id="deterministic",
        command=["biominer", "report"],
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        git_sha="abc123",
        pid=42,
    )
    second = build_flickr_geo_workload_report(
        geography=geography.reverse(),
        clusters=clusters.reverse(),
        assignments=assignments.reverse(),
        query_hits=query_hits.reverse(),
        quota_config=ReferenceQuotaConfig(minimum_candidate_images=2),
        run_id="deterministic",
        command=["biominer", "report"],
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        git_sha="abc123",
        pid=42,
    )
    assert first.payload == second.payload
    assert first.markdown == second.markdown


def test_query_provenance_can_be_explicitly_not_instrumented() -> None:
    geography, clusters, assignments, _query_hits = _artifacts()
    report = build_flickr_geo_workload_report(
        geography=geography,
        clusters=clusters,
        assignments=assignments,
        query_hits=None,
        run_id="no-query-provenance",
        command=["biominer", "report"],
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        git_sha="abc123",
        pid=42,
    )
    assert report.payload["records_by_query_tier_and_cluster"] == "not_instrumented"
    assert report.payload["records_by_search_term_and_cluster"] == "not_instrumented"
    assert report.payload["query_provenance"]["status"] == "not_instrumented"
    assert "not_instrumented" in report.markdown


def test_reports_unassigned_geotagged_records_separately_from_missing_coordinates(
) -> None:
    geography = build_flickr_geography_frame(
        [
            _record("core-1", -27.4705, 153.026),
            _record("core-2", -27.471, 153.027),
            _record("remote", -33.8688, 151.2093),
        ]
    )
    clustered = build_flickr_geo_clusters(
        geography,
        target_accepted_taxon_key=TARGET_KEY,
        config=FlickrGeoClusterConfig(
            minimum_images_per_cell=2,
            minimum_cluster_images=2,
        ),
        created_at=STARTED_AT,
    )
    report = build_flickr_geo_workload_report(
        geography=geography,
        clusters=clustered.clusters,
        assignments=clustered.assignments,
        query_hits=None,
        run_id="unassigned-geotagged",
        command=["biominer", "report"],
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        git_sha="abc123",
        pid=42,
    )

    assert report.payload["summary"]["geotagged_record_count"] == 3
    assert report.payload["summary"]["missing_coordinate_record_count"] == 0
    assert report.payload["summary"]["unassigned_geotagged_record_count"] == 1
    assert "Geotagged but unassigned | 1" in report.markdown


def test_square_root_quota_allocation_excludes_global_fallback_clusters() -> None:
    clusters = pl.DataFrame(
        {
            "geo_cluster_id": [
                "large",
                "small",
                NO_GEO_CLUSTER_ID,
                UNASSIGNED_GEO_CLUSTER_ID,
            ],
            "member_image_count": [100, 25, 50, 25],
        }
    )
    quota = allocate_implied_reference_quotas(
        clusters,
        config=ReferenceQuotaConfig(
            total_reference_quota=30,
            minimum_per_populated_cluster=5,
            minimum_candidate_images=1,
        ),
    )
    by_cluster = {row["geo_cluster_id"]: row for row in quota["by_cluster"]}
    assert by_cluster["large"]["reference_quota_implied"] == 18
    assert by_cluster["small"]["reference_quota_implied"] == 12
    assert by_cluster[NO_GEO_CLUSTER_ID]["reference_quota_implied"] == 0
    assert by_cluster[UNASSIGNED_GEO_CLUSTER_ID]["reference_quota_implied"] == 0
    assert quota["allocated_reference_quota"] == 30
    assert quota["configuration_fingerprint"].startswith("sha256:")


def test_quota_reports_when_total_cannot_meet_every_cluster_minimum() -> None:
    clusters = pl.DataFrame(
        {
            "geo_cluster_id": ["a", "b", "c"],
            "member_image_count": [9, 4, 1],
        }
    )
    quota = allocate_implied_reference_quotas(
        clusters,
        config=ReferenceQuotaConfig(
            total_reference_quota=2,
            minimum_per_populated_cluster=5,
            minimum_candidate_images=1,
        ),
    )
    assert sum(row["reference_quota_implied"] for row in quota["by_cluster"]) == 2
    assert quota["minimum_fully_satisfied"] is False
    assert any(row["minimum_shortfall"] > 0 for row in quota["by_cluster"])


def test_rejects_stale_or_unknown_cross_artifact_records() -> None:
    geography, clusters, assignments, query_hits = _artifacts()
    stale = assignments.with_columns(
        pl.when(pl.col("flickr_photo_id") == "brisbane-1")
        .then(pl.lit("sha256:stale"))
        .otherwise(pl.col("source_record_hash"))
        .alias("source_record_hash")
    )
    with pytest.raises(ValueError, match="source_record_hash mismatch"):
        build_flickr_geo_workload_report(
            geography=geography,
            clusters=clusters,
            assignments=stale,
            query_hits=query_hits,
            run_id="stale",
            command=["biominer", "report"],
            started_at=STARTED_AT,
            ended_at=ENDED_AT,
            git_sha="abc123",
        )

    unknown_hit = query_hits.vstack(
        pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "unknown",
                    "query_tier": "broad",
                    "search_term": "butterfly",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="unknown candidate"):
        build_flickr_geo_workload_report(
            geography=geography,
            clusters=clusters,
            assignments=assignments,
            query_hits=unknown_hit,
            run_id="unknown-query-hit",
            command=["biominer", "report"],
            started_at=STARTED_AT,
            ended_at=ENDED_AT,
            git_sha="abc123",
        )


def test_writes_atomic_json_and_markdown_reports(tmp_path) -> None:
    report = _report()
    paths = write_flickr_geo_workload_report(report, tmp_path)
    assert paths == {
        "metrics": tmp_path / FLICKR_GEO_WORKLOAD_METRICS_FILE,
        "summary": tmp_path / FLICKR_GEO_WORKLOAD_SUMMARY_FILE,
    }
    assert json.loads(paths["metrics"].read_text(encoding="utf-8")) == report.payload
    assert paths["summary"].read_text(encoding="utf-8") == report.markdown
    assert not list(tmp_path.glob(".*.tmp"))
