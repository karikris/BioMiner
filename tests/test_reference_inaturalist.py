from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from biominer.references.gbif import GBIFReferenceAdapter
from biominer.references.inaturalist import (
    INATURALIST_API_PAGE_SIZE,
    INATURALIST_GBIF_DATASET_DOI,
    INATURALIST_GBIF_DATASET_KEY,
    INaturalistBulkAcquisitionRequired,
    INaturalistReferenceAdapter,
    build_inaturalist_bulk_acquisition_options,
    load_inaturalist_reference_checkpoint,
    load_inaturalist_reference_checkpoint_frames,
    mark_inaturalist_gbif_media_duplicates,
    write_inaturalist_reference_checkpoint,
)
from biominer.references.source_base import ReferenceSourceQuery


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
BOUNDING_BOX = (-34.0, 150.0, -33.0, 152.0)


class RecordedINaturalist:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((path, dict(params)))
        if not self.payloads:
            raise AssertionError("unexpected iNaturalist request")
        return self.payloads.pop(0)


def _query(
    *,
    fallback_level: int = 0,
    bounding_box: tuple[float, float, float, float] | None = BOUNDING_BOX,
    source_place_ids: tuple[str, ...] = (),
) -> ReferenceSourceQuery:
    return ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au-east",
        fallback_level=fallback_level,
        source_taxon_id="12345",
        source_place_ids=source_place_ids,
        bounding_box=bounding_box,
        cluster_medoid_latitude=-33.86,
        cluster_medoid_longitude=151.21,
        page_size=INATURALIST_API_PAGE_SIZE,
        source_snapshot_version="inaturalist-api-2026-07-13",
    )


def _photo(
    photo_id: int,
    *,
    licence: str | None = "cc-by",
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": photo_id,
        "url": f"https://static.inaturalist.org/photos/{photo_id}/large.jpg",
        "original_dimensions": {"width": 2048, "height": 1365},
        "attribution": f"(c) observer, {licence or 'all rights reserved'}",
        "position": 0,
    }
    if licence is not None:
        row["license_code"] = licence
    return row


def _observation(
    observation_id: int,
    *,
    taxon_id: int = 12345,
    rank: str = "species",
    quality_grade: str = "research",
    captive: bool | None = False,
    coordinates_obscured: bool | None = False,
    disagreement_count: int = 0,
    photo_licence: str | None = "cc-by",
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": observation_id,
        "uri": f"https://www.inaturalist.org/observations/{observation_id}",
        "quality_grade": quality_grade,
        "taxon": {
            "id": taxon_id,
            "name": "Papilio demoleus",
            "rank": rank,
        },
        "community_taxon": {
            "id": taxon_id,
            "name": "Papilio demoleus",
            "rank": rank,
        },
        "time_observed_at": "2025-02-03T04:05:06Z",
        "geojson": {"type": "Point", "coordinates": [151.22, -33.87]},
        "location": "-33.87,151.22",
        "positional_accuracy": 25,
        "public_positional_accuracy": 30,
        "geoprivacy": "open",
        "place_guess": "Sydney, New South Wales, Australia",
        "license_code": "cc-by-nc",
        "num_identification_disagreements": disagreement_count,
        "user": {"id": 88, "login": "observer", "name": "Example Observer"},
        "photos": [_photo(observation_id + 10_000, licence=photo_licence)],
        "annotations": [
            {
                "controlled_attribute": {"label": "Life Stage"},
                "controlled_value": {"label": "Adult"},
            },
            {
                "controlled_attribute": {"label": "Sex"},
                "controlled_value": {"label": "Female"},
            },
        ],
    }
    if captive is not None:
        row["captive"] = captive
    if coordinates_obscured is not None:
        row["obscured"] = coordinates_obscured
    return row


def _payload(
    records: list[dict[str, object]],
    *,
    total_results: int | None = None,
) -> dict[str, object]:
    return {
        "total_results": len(records) if total_results is None else total_results,
        "page": 1,
        "per_page": INATURALIST_API_PAGE_SIZE,
        "results": records,
    }


def _adapter(http_get: RecordedINaturalist) -> INaturalistReferenceAdapter:
    return INaturalistReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        http_get=http_get,
        retrieved_at=lambda: NOW,
    )


def test_inaturalist_adapter_uses_exact_filtered_regional_query() -> None:
    source = RecordedINaturalist([_payload([_observation(1001)])])
    page = _adapter(source).fetch_page(_query())

    assert source.requests == [
        (
            "/observations",
            {
                "taxon_id": "12345",
                "quality_grade": "research",
                "rank": "species",
                "captive": "false",
                "photos": "true",
                "photo_license": "cc-by,cc-by-nc,cc0",
                "taxon_is_active": "true",
                "per_page": 200,
                "page": 1,
                "order_by": "id",
                "order": "asc",
                "swlat": -34.0,
                "swlng": 150.0,
                "nelat": -33.0,
                "nelng": 152.0,
                "geo": "true",
                "mappable": "true",
                "geoprivacy": "open",
            },
        )
    ]
    assert page.complete is True
    assert page.observations.height == 1
    assert page.media_candidates.height == 1


def test_inaturalist_adapter_preserves_community_identity_coordinates_and_licences() -> None:
    page = _adapter(
        RecordedINaturalist([_payload([_observation(1001)])])
    ).fetch_page(_query())
    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)

    assert observation["source_observation_id"] == "1001"
    assert observation["source_taxon_id"] == "12345"
    assert observation["accepted_taxon_key"] == "gbif:1938069"
    assert observation["identification_quality"] == "research"
    assert observation["community_taxon_status"] == "species"
    assert observation["identification_disagreement"] is False
    assert observation["captive_or_cultivated"] is False
    assert observation["observer_id"] == "88"
    assert observation["locality"] == "Sydney, New South Wales, Australia"
    assert observation["life_stage"] == "adult"
    assert observation["sex"] == "Female"
    assert observation["latitude"] == -33.87
    assert observation["longitude"] == 151.22
    assert observation["coordinate_uncertainty"] == 30.0
    assert observation["coordinates_obscured"] is False
    assert observation["distance_to_cluster_medoid_km"] == pytest.approx(1.45, abs=0.1)
    assert observation["preserved_specimen"] is None
    assert observation["fossil"] is None
    assert observation["occurrence_absent"] is None
    assert media["provider_media_id"] == "11001"
    assert media["licence"] == "cc-by"
    assert media["occurrence_licence"] == "cc-by-nc"
    assert media["licence_policy_status"] == "allowed"
    assert media["verification_status"] == "unreviewed"
    assert media["download_status"] == "pending"


@pytest.mark.parametrize(
    ("record", "expected_reasons"),
    [
        (
            _observation(2001, taxon_id=999),
            {"uncertain_taxon_match"},
        ),
        (
            _observation(2002, quality_grade="needs_id", captive=True),
            {"quality_grade_not_research", "captive_or_cultivated"},
        ),
        (
            _observation(2003, coordinates_obscured=True, disagreement_count=1),
            {"identification_disagreement", "regional_coordinates_obscured_or_unknown"},
        ),
        (
            _observation(2004, captive=None, photo_licence=None),
            {"captive_status_unknown", "missing_photo_licence"},
        ),
        (
            _observation(2005, photo_licence="cc-by-sa"),
            {"photo_licence_not_allowed:cc-by-sa"},
        ),
    ],
)
def test_inaturalist_adapter_excludes_unreconciled_or_unsuitable_candidates(
    record: dict[str, object],
    expected_reasons: set[str],
) -> None:
    page = _adapter(RecordedINaturalist([_payload([record])])).fetch_page(_query())
    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)

    assert expected_reasons <= set(str(media["exclusion_reason"]).split(";"))
    assert media["download_status"] in {"excluded", "quarantined"}
    assert media["verification_status"] == "unreviewed"
    if "uncertain_taxon_match" in expected_reasons:
        assert observation["taxon_reconciliation_status"] == "conflict"
        assert observation["accepted_taxon_key"] is None


def test_inaturalist_adapter_uses_monotonic_id_cursor_not_page_offset() -> None:
    first_records = [_observation(value) for value in range(1001, 1201)]
    source = RecordedINaturalist(
        [
            _payload(first_records, total_results=201),
            _payload([_observation(1201)], total_results=1),
        ]
    )
    adapter = _adapter(source)
    first = adapter.fetch_page(_query())
    second = adapter.fetch_page(_query(), cursor=first.next_cursor)

    assert first.complete is False
    assert first.next_cursor == "1200"
    assert second.complete is True
    assert source.requests[1][1]["id_above"] == 1200
    assert source.requests[1][1]["page"] == 1


def test_inaturalist_adapter_requires_explicit_species_level_community_taxon() -> None:
    record = _observation(2100)
    record.pop("community_taxon")
    page = _adapter(RecordedINaturalist([_payload([record])])).fetch_page(_query())

    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)
    assert observation["taxon_reconciliation_status"] == "unresolved"
    assert observation["source_taxon_id"] is None
    assert media["download_status"] == "excluded"
    assert "uncertain_taxon_match" in str(media["exclusion_reason"])


def test_inaturalist_adapter_reconciles_compact_search_community_taxon_id() -> None:
    record = _observation(2101)
    record["community_taxon_id"] = 12345
    record.pop("community_taxon")

    page = _adapter(RecordedINaturalist([_payload([record])])).fetch_page(_query())
    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)

    assert observation["source_taxon_id"] == "12345"
    assert observation["accepted_taxon_key"] == "gbif:1938069"
    assert observation["taxon_reconciliation_status"] == "accepted_key_exact"
    assert observation["community_taxon_status"] == "species"
    assert media["download_status"] == "pending"

    conflict = _observation(2102)
    conflict["community_taxon_id"] = 999
    conflict.pop("community_taxon")
    conflict_page = _adapter(
        RecordedINaturalist([_payload([conflict])])
    ).fetch_page(_query())
    conflict_observation = conflict_page.observations.row(0, named=True)

    assert conflict_observation["source_taxon_id"] == "999"
    assert conflict_observation["accepted_taxon_key"] is None
    assert conflict_observation["taxon_reconciliation_status"] == "conflict"


def test_inaturalist_adapter_rejects_unordered_or_nonadvancing_results() -> None:
    unordered = RecordedINaturalist(
        [_payload([_observation(1002), _observation(1001)])]
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        _adapter(unordered).fetch_page(_query())

    nonadvancing = RecordedINaturalist([_payload([_observation(1001)])])
    with pytest.raises(ValueError, match="did not advance"):
        _adapter(nonadvancing).fetch_page(_query(), cursor="1001")


def test_inaturalist_geographic_fallback_and_global_scope() -> None:
    place_query = _query(
        fallback_level=2,
        bounding_box=None,
        source_place_ids=("6744", "6744"),
    )
    global_query = _query(fallback_level=3, bounding_box=None)
    source = RecordedINaturalist(
        [
            _payload([_observation(3001)]),
            _payload([_observation(3002, coordinates_obscured=True)]),
        ]
    )
    adapter = _adapter(source)
    adapter.fetch_page(place_query)
    global_page = adapter.fetch_page(global_query)

    assert source.requests[0][1]["place_id"] == "6744"
    assert source.requests[0][1]["geoprivacy"] == "open"
    assert "place_id" not in source.requests[1][1]
    assert "geoprivacy" not in source.requests[1][1]
    assert global_page.media_candidates["download_status"].item() == "pending"


def test_inaturalist_bulk_search_escalates_to_official_export_options() -> None:
    source = RecordedINaturalist([_payload([], total_results=10_001)])
    with pytest.raises(INaturalistBulkAcquisitionRequired) as captured:
        _adapter(source).fetch_page(_query())

    options = captured.value.acquisition_options
    assert captured.value.total_records == 10_001
    assert options["weekly_gbif_dataset_key"] == INATURALIST_GBIF_DATASET_KEY
    assert options["weekly_gbif_dataset_doi"] == INATURALIST_GBIF_DATASET_DOI
    assert options["authentication_required_for_api"] is False
    assert options["authentication_required_for_weekly_gbif_download"] is True
    bulk_predicates = options["weekly_gbif_download_request"]["predicate"][  # type: ignore[index]
        "predicates"
    ]
    assert {
        "type": "equals",
        "key": "DATASET_KEY",
        "value": INATURALIST_GBIF_DATASET_KEY,
    } in bulk_predicates
    assert {
        "type": "within",
        "geometry": "POLYGON((150 -34,152 -34,152 -33,150 -33,150 -34))",
    } in bulk_predicates
    assert build_inaturalist_bulk_acquisition_options(_query()) == options


def test_inaturalist_transport_paces_retries_and_never_authenticates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=_payload([]))

    current = [0.0]
    delays: list[float] = []

    def sleep(seconds: float) -> None:
        delays.append(seconds)
        current[0] += seconds

    with INaturalistReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        max_retries=1,
        retrieved_at=lambda: NOW,
        sleep=sleep,
        monotonic=lambda: current[0],
        transport=httpx.MockTransport(handler),
    ) as adapter:
        page = adapter.fetch_page(_query())

    assert len(requests) == 2
    assert requests[0].headers["User-Agent"].startswith("BioMiner/")
    assert "Authorization" not in requests[0].headers
    assert delays == [2.0]
    assert page.request_count == 2
    assert page.retry_count == 1
    assert page.rate_limit_count == 1


def test_inaturalist_transport_enforces_one_second_between_successes() -> None:
    current = [0.0]
    delays: list[float] = []

    def sleep(seconds: float) -> None:
        delays.append(seconds)
        current[0] += seconds

    adapter = INaturalistReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        retrieved_at=lambda: NOW,
        sleep=sleep,
        monotonic=lambda: current[0],
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_payload([]))
        ),
    )
    try:
        adapter.fetch_page(_query())
        adapter.fetch_page(_query())
    finally:
        adapter.close()

    assert delays == [1.0]


def test_inaturalist_checkpoints_resume_from_last_observation_id(
    tmp_path: Path,
) -> None:
    query = _query()
    first_records = [_observation(value) for value in range(4001, 4201)]
    first_page = _adapter(
        RecordedINaturalist([_payload(first_records, total_results=201)])
    ).fetch_page(query)
    checkpoint = write_inaturalist_reference_checkpoint(
        query,
        first_page,
        tmp_path,
    )

    assert checkpoint.complete is False
    assert checkpoint.next_cursor == "4200"
    source = RecordedINaturalist([_payload([_observation(4201)], total_results=1)])
    pages = list(_adapter(source).iter_pages(query, checkpoint_dir=tmp_path))
    assert source.requests[0][1]["id_above"] == 4200
    assert len(pages) == 1
    complete = load_inaturalist_reference_checkpoint(query, tmp_path)
    assert complete is not None and complete.complete is True
    observations, media = load_inaturalist_reference_checkpoint_frames(query, tmp_path)
    assert observations.height == 201
    assert media.height == 201


def test_inaturalist_through_gbif_duplicate_is_excluded_without_losing_provenance() -> None:
    inaturalist_page = _adapter(
        RecordedINaturalist([_payload([_observation(5001)])])
    ).fetch_page(_query())
    gbif_record = {
        "key": 777,
        "taxonKey": 1938069,
        "acceptedTaxonKey": 1938069,
        "speciesKey": 1938069,
        "species": "Papilio demoleus",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "occurrenceStatus": "PRESENT",
        "eventDate": "2025-02-03T04:05:06Z",
        "decimalLatitude": -33.87,
        "decimalLongitude": 151.22,
        "references": "https://www.inaturalist.org/observations/5001",
        "media": [
            {
                "type": "StillImage",
                "identifier": "https://static.inaturalist.org/photos/15001/original.jpg",
                "license": "cc-by",
            }
        ],
    }
    gbif_query = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au-east",
        fallback_level=0,
        source_taxon_id="1938069",
        geometry_wkt="POLYGON((150 -34,152 -34,152 -33,150 -33,150 -34))",
        page_size=300,
        source_snapshot_version="gbif-occurrence-2026-07-13",
    )
    gbif_page = GBIFReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        http_get=lambda path, params: {
            "offset": 0,
            "count": 1,
            "endOfRecords": True,
            "results": [gbif_record],
        },
        retrieved_at=lambda: NOW,
    ).fetch_page(gbif_query)
    observations = pl.concat(
        [gbif_page.observations, inaturalist_page.observations],
        how="vertical",
    ).sort(["source", "source_observation_id"])
    media = pl.concat(
        [gbif_page.media_candidates, inaturalist_page.media_candidates],
        how="vertical",
    ).sort(["source", "provider_media_id", "reference_observation_id"])
    deduplicated = mark_inaturalist_gbif_media_duplicates(observations, media)
    rows = {str(row["source"]): row for row in deduplicated.iter_rows(named=True)}

    assert observations.height == 2
    assert deduplicated.height == 2
    assert rows["iNaturalist"]["download_status"] == "pending"
    assert rows["GBIF"]["download_status"] == "excluded"
    assert "duplicate_inaturalist_through_gbif" in str(
        rows["GBIF"]["exclusion_reason"]
    )


def test_inaturalist_query_requires_structured_scope_and_largest_page_size() -> None:
    invalid_page = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au-east",
        fallback_level=3,
        source_taxon_id="12345",
        page_size=100,
        source_snapshot_version="inaturalist-api-2026-07-13",
    )
    source = RecordedINaturalist([_payload([])])
    with pytest.raises(ValueError, match="page_size=200"):
        _adapter(source).fetch_page(invalid_page)
    with pytest.raises(ValueError, match="require bounding_box"):
        _adapter(source).fetch_page(_query(bounding_box=None))
    with pytest.raises(ValueError, match="requires source_place_ids"):
        _adapter(source).fetch_page(_query(fallback_level=2, bounding_box=None))
