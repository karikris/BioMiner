from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from biominer.candidates.regional_union import (
    REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
    regional_candidate_species_schema,
)
from biominer.references.gbif import GBIFReferenceAdapter
from biominer.references.inaturalist import (
    INaturalistReferenceAdapter,
    mark_inaturalist_gbif_media_duplicates,
)
from biominer.references.planner import (
    ReferencePlannerConfig,
    ReferenceStratumQuota,
    plan_geographically_balanced_support_bank,
)
from biominer.references.source_base import ReferenceSourceQuery


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "references"
NOW = datetime(2026, 7, 14, 0, 0, tzinfo=UTC)
TARGET_KEY = "gbif:1938069"
TARGET_NAME = "Papilio demoleus"
REGISTRY_VERSION = "butterflies-v2-20260712"


class RecordedHTTPGet:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = dict(payload)
        self.requests: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((path, dict(params)))
        return json.loads(json.dumps(self.payload))


def _load_json(name: str) -> dict[str, object]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _fixture_response(name: str, case: str | None = None) -> dict[str, object]:
    fixture = _load_json(name)
    if case is None:
        response = fixture["response"]
    else:
        responses = fixture["responses"]
        assert isinstance(responses, dict)
        response = responses[case]
    assert isinstance(response, dict)
    return response


def _gbif_query() -> ReferenceSourceQuery:
    return ReferenceSourceQuery(
        accepted_taxon_key=TARGET_KEY,
        scientific_name=TARGET_NAME,
        geo_cluster_id="cluster-aceh",
        fallback_level=3,
        source_taxon_id="1938069",
        cluster_medoid_latitude=4.695135,
        cluster_medoid_longitude=96.7493993,
        page_size=300,
        source_snapshot_version="gbif-recorded-2026-07-14",
    )


def _inaturalist_query(
    *,
    fallback_level: int = 3,
    geo_cluster_id: str = "cluster-aceh",
) -> ReferenceSourceQuery:
    bounding_box = None
    source_place_ids: tuple[str, ...] = ()
    medoid_latitude = 12.8915028
    medoid_longitude = 77.5923063
    if geo_cluster_id == "cluster-aceh":
        medoid_latitude = 4.695135
        medoid_longitude = 96.7493993
    if fallback_level in {0, 1}:
        bounding_box = (12.0, 77.0, 13.5, 78.0)
    elif fallback_level == 2:
        source_place_ids = ("6681",)
    return ReferenceSourceQuery(
        accepted_taxon_key=TARGET_KEY,
        scientific_name=TARGET_NAME,
        geo_cluster_id=geo_cluster_id,
        fallback_level=fallback_level,
        source_taxon_id="51583",
        source_place_ids=source_place_ids,
        bounding_box=bounding_box,
        cluster_medoid_latitude=medoid_latitude,
        cluster_medoid_longitude=medoid_longitude,
        page_size=200,
        source_snapshot_version="inaturalist-recorded-2026-07-14",
    )


def _candidate_species(geo_cluster_id: str) -> pl.DataFrame:
    fingerprint = (
        "sha256:"
        + hashlib.sha256(f"recorded:{geo_cluster_id}".encode("utf-8")).hexdigest()
    )
    return pl.DataFrame(
        [
            {
                "schema_version": REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
                "candidate_set_id": f"regional:{geo_cluster_id}",
                "target_accepted_taxon_key": TARGET_KEY,
                "geo_cluster_id": geo_cluster_id,
                "candidate_accepted_taxon_key": TARGET_KEY,
                "scientific_name": TARGET_NAME,
                "family": "Papilionidae",
                "genus": "Papilio",
                "candidate_reason": ["target"],
                "geographic_evidence_score": 1.0,
                "occurrence_support": 1,
                "same_genus": True,
                "same_family": True,
                "known_mimic": False,
                "historical_false_positive": False,
                "visually_nearest": False,
                "target_candidate": True,
                "candidate_priority": 0,
                "source_versions": ["recorded-reference-fixtures-v1"],
                "candidate_set_fingerprint": fingerprint,
            }
        ],
        schema=regional_candidate_species_schema(),
        strict=True,
    )


def _planner_config(*, quota: int) -> ReferencePlannerConfig:
    return ReferencePlannerConfig(
        strata=(
            ReferenceStratumQuota(
                life_stage="unknown",
                visual_domain="unreviewed",
                requested_per_species=quota,
            ),
        ),
        minimum_per_sufficient_cluster=0,
        sufficiently_populated_candidate_count=1,
        selection_seed=45,
    )


def test_recorded_reference_fixture_manifest_is_integrity_checked() -> None:
    manifest = _load_json("manifest.json")
    assert manifest["schema_version"] == "reference-api-fixture-manifest-v1"
    fixtures = manifest["fixtures"]
    assert isinstance(fixtures, list)
    for entry in fixtures:
        assert isinstance(entry, dict)
        path = FIXTURE_DIR / str(entry["file"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"]
        recording = json.loads(path.read_text(encoding="utf-8"))
        assert recording["_fixture"]["recorded_at"] == "2026-07-14"


def test_recorded_sources_normalize_deduplicate_and_report_real_shortfall() -> None:
    gbif_source = RecordedHTTPGet(_fixture_response("gbif_occurrence_search_v1.json"))
    gbif_page = GBIFReferenceAdapter(
        registry_version=REGISTRY_VERSION,
        http_get=gbif_source,
        retrieved_at=lambda: NOW,
    ).fetch_page(_gbif_query())
    inaturalist_source = RecordedHTTPGet(
        _fixture_response(
            "inaturalist_observation_search_v1.json",
            "duplicate_pair",
        )
    )
    inaturalist_page = INaturalistReferenceAdapter(
        registry_version=REGISTRY_VERSION,
        http_get=inaturalist_source,
        retrieved_at=lambda: NOW,
    ).fetch_page(_inaturalist_query())

    observations = pl.concat(
        [gbif_page.observations, inaturalist_page.observations],
        how="vertical",
    ).sort(["source", "source_observation_id"])
    media = pl.concat(
        [gbif_page.media_candidates, inaturalist_page.media_candidates],
        how="vertical",
    ).sort(["source", "provider_media_id", "reference_observation_id"])
    deduplicated = mark_inaturalist_gbif_media_duplicates(observations, media)

    assert observations.height == 3
    assert set(observations["taxon_reconciliation_status"].to_list()) == {
        "accepted_key_exact"
    }
    direct = observations.filter(pl.col("source") == "iNaturalist").row(
        0,
        named=True,
    )
    assert direct["source_taxon_id"] == "51583"
    assert direct["latitude"] == 4.695135
    assert direct["longitude"] == 96.7493993
    assert direct["coordinate_uncertainty"] == 229400.0
    assert direct["distance_to_cluster_medoid_km"] == pytest.approx(0.0, abs=0.001)

    licence_observation = observations.filter(
        pl.col("source_observation_id") == "6129961782"
    )["reference_observation_id"].item()
    licence_media = deduplicated.filter(
        pl.col("reference_observation_id") == licence_observation
    ).row(0, named=True)
    assert licence_media["occurrence_licence"] == (
        "http://creativecommons.org/licenses/by-nc/4.0/legalcode"
    )
    assert licence_media["licence"] == (
        "http://creativecommons.org/licenses/by-nc-nd/4.0/"
    )
    assert licence_media["occurrence_licence"] != licence_media["licence"]

    duplicate_observation_ids = observations.filter(
        pl.col("source_observation_id").is_in(["5938133297", "333386897"])
    )["reference_observation_id"].to_list()
    rows_by_source = {
        str(row["source"]): row
        for row in deduplicated.filter(
            pl.col("reference_observation_id").is_in(duplicate_observation_ids)
        ).iter_rows(named=True)
    }
    assert rows_by_source["iNaturalist"]["download_status"] == "pending"
    assert rows_by_source["GBIF"]["download_status"] == "excluded"
    assert "duplicate_inaturalist_through_gbif" in str(
        rows_by_source["GBIF"]["exclusion_reason"]
    )
    assert deduplicated["verification_status"].unique().to_list() == ["unreviewed"]

    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species("cluster-aceh"),
        observations=observations,
        media_candidates=deduplicated,
        config=_planner_config(quota=3),
        created_at=NOW,
    )
    row = result.plan.row(0, named=True)
    assert row["candidate_accepted_taxon_key"] == TARGET_KEY
    assert row["available_candidate_count"] == 1
    assert row["selected_candidate_count"] == 1
    assert row["shortfall_count"] == 2
    assert result.report["summary"]["target_selected"] == 1
    assert result.report["diversity"]["independent_observations"] == 1

    zero_eligible = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species("cluster-aceh"),
        observations=gbif_page.observations,
        media_candidates=gbif_page.media_candidates,
        config=_planner_config(quota=1),
        created_at=NOW,
    )
    assert zero_eligible.plan["candidate_accepted_taxon_key"].to_list() == [TARGET_KEY]
    assert zero_eligible.report["by_species"] == [
        {
            "candidate_accepted_taxon_key": TARGET_KEY,
            "existing_support": 0,
            "requested": 1,
            "available": 0,
            "selected": 0,
            "shortfall": 1,
        }
    ]


def test_recorded_geographic_fallback_fills_only_the_remaining_quota() -> None:
    cases = (
        "fallback_level_0_empty",
        "fallback_level_1",
        "fallback_level_2",
        "fallback_level_3",
    )
    pages = []
    sources = []
    for fallback_level, case in enumerate(cases):
        source = RecordedHTTPGet(
            _fixture_response("inaturalist_observation_search_v1.json", case)
        )
        page = INaturalistReferenceAdapter(
            registry_version=REGISTRY_VERSION,
            http_get=source,
            retrieved_at=lambda: NOW,
        ).fetch_page(
            _inaturalist_query(
                fallback_level=fallback_level,
                geo_cluster_id="cluster-bengaluru",
            )
        )
        sources.append(source)
        pages.append(page)

    assert pages[0].observations.is_empty()
    assert "swlat" in sources[0].requests[0][1]
    assert "swlat" in sources[1].requests[0][1]
    assert sources[2].requests[0][1]["place_id"] == "6681"
    assert "place_id" not in sources[3].requests[0][1]
    assert "swlat" not in sources[3].requests[0][1]

    observations = pl.concat(
        [page.observations for page in pages[1:]],
        how="vertical",
    ).sort(["source", "source_observation_id"])
    media = pl.concat(
        [page.media_candidates for page in pages[1:]],
        how="vertical",
    ).sort(["source", "provider_media_id", "reference_observation_id"])
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species("cluster-bengaluru"),
        observations=observations,
        media_candidates=media,
        config=_planner_config(quota=2),
        created_at=NOW,
    )

    assert result.selections["fallback_level"].to_list() == [1, 2]
    assert result.report["fallback_distribution"] == {"1": 1, "2": 1}
    assert result.report["summary"]["available"] == 3
    assert result.report["summary"]["selected"] == 2
    assert result.report["summary"]["shortfall"] == 0


def test_recorded_success_bodies_survive_rate_limit_retries() -> None:
    gbif_requests: list[httpx.Request] = []

    def gbif_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/occurrence/search"
        gbif_requests.append(request)
        if len(gbif_requests) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.25"},
                request=request,
            )
        if len(gbif_requests) == 2:
            return httpx.Response(
                200,
                json=_fixture_response("gbif_occurrence_search_v1.json"),
                request=request,
            )
        raise AssertionError(f"unexpected GBIF request: {request.url}")

    gbif_delays: list[float] = []
    with GBIFReferenceAdapter(
        registry_version=REGISTRY_VERSION,
        max_retries=1,
        retrieved_at=lambda: NOW,
        sleep=gbif_delays.append,
        transport=httpx.MockTransport(gbif_handler),
    ) as adapter:
        gbif_page = adapter.fetch_page(_gbif_query())

    assert len(gbif_requests) == 2
    assert gbif_requests[0].url.params["taxonKey"] == "1938069"
    assert gbif_delays == [0.25]
    assert (
        gbif_page.request_count,
        gbif_page.retry_count,
        gbif_page.rate_limit_count,
    ) == (
        2,
        1,
        1,
    )
    assert gbif_page.observations.height == 2

    inaturalist_requests: list[httpx.Request] = []
    current = [0.0]
    inaturalist_delays: list[float] = []

    def inaturalist_sleep(seconds: float) -> None:
        inaturalist_delays.append(seconds)
        current[0] += seconds

    def inaturalist_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/observations"
        inaturalist_requests.append(request)
        if len(inaturalist_requests) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                request=request,
            )
        if len(inaturalist_requests) == 2:
            return httpx.Response(
                200,
                json=_fixture_response(
                    "inaturalist_observation_search_v1.json",
                    "duplicate_pair",
                ),
                request=request,
            )
        raise AssertionError(f"unexpected iNaturalist request: {request.url}")

    with INaturalistReferenceAdapter(
        registry_version=REGISTRY_VERSION,
        max_retries=1,
        retrieved_at=lambda: NOW,
        sleep=inaturalist_sleep,
        monotonic=lambda: current[0],
        transport=httpx.MockTransport(inaturalist_handler),
    ) as adapter:
        inaturalist_page = adapter.fetch_page(_inaturalist_query())

    assert len(inaturalist_requests) == 2
    assert inaturalist_requests[0].url.params["taxon_id"] == "51583"
    assert "Authorization" not in inaturalist_requests[0].headers
    assert inaturalist_delays == [2.0]
    assert (
        inaturalist_page.request_count,
        inaturalist_page.retry_count,
        inaturalist_page.rate_limit_count,
    ) == (2, 1, 1)
    assert inaturalist_page.observations["source_taxon_id"].item() == "51583"
