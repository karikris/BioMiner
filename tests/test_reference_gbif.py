from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from biominer.references.gbif import (
    GBIFReferenceAdapter,
    GBIFReferenceBulkDownloadRequired,
    build_gbif_reference_download_request,
    gbif_image_cache_url,
    load_gbif_reference_checkpoint,
    load_gbif_reference_checkpoint_frames,
    write_gbif_reference_checkpoint,
)
from biominer.references.source_base import ReferenceSourceQuery


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
GEOMETRY = "POLYGON((150 -34,152 -34,152 -33,150 -33,150 -34))"


class RecordedGBIF:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((path, dict(params)))
        if not self.payloads:
            raise AssertionError("unexpected GBIF request")
        return self.payloads.pop(0)


def _query(
    *,
    fallback_level: int = 0,
    geometry_wkt: str | None = GEOMETRY,
    country_codes: tuple[str, ...] = (),
    page_size: int = 2,
    maximum_records: int | None = None,
) -> ReferenceSourceQuery:
    return ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au-east",
        fallback_level=fallback_level,
        source_taxon_id="1938069",
        country_codes=country_codes,
        geometry_wkt=geometry_wkt,
        cluster_medoid_latitude=-33.86,
        cluster_medoid_longitude=151.21,
        page_size=page_size,
        maximum_records=maximum_records,
        source_snapshot_version="gbif-occurrence-2026-07-13",
    )


def _media(
    identifier: str = "https://publisher.test/images/one.jpg",
    *,
    licence: str | None = "https://creativecommons.org/licenses/by/4.0/",
    media_type: str = "StillImage",
    media_id: str | None = "media-1",
) -> dict[str, object]:
    row: dict[str, object] = {
        "type": media_type,
        "identifier": identifier,
        "creator": "A. Observer",
        "rightsHolder": "A. Observer",
        "width": 2048,
        "height": 1365,
    }
    if media_id is not None:
        row["id"] = media_id
    if licence is not None:
        row["license"] = licence
    return row


def _occurrence(
    key: int,
    *,
    species_key: int | None = 1938069,
    basis: str = "HUMAN_OBSERVATION",
    status: str = "PRESENT",
    issues: list[str] | None = None,
    media: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "key": key,
        "taxonKey": species_key,
        "acceptedTaxonKey": species_key,
        "speciesKey": species_key,
        "species": "Papilio demoleus",
        "scientificName": "Papilio demoleus Linnaeus, 1758",
        "basisOfRecord": basis,
        "occurrenceStatus": status,
        "issues": issues or [],
        "identificationVerificationStatus": "record-label-only",
        "lifeStage": "Adult",
        "sex": "FEMALE",
        "eventDate": "2025-02-03T04:05:06Z",
        "decimalLatitude": -33.87,
        "decimalLongitude": 151.22,
        "coordinateUncertaintyInMeters": 25.0,
        "country": "Australia",
        "countryCode": "AU",
        "datasetKey": "dataset-key-1",
        "datasetDOI": "10.15468/example",
        "publisher": "Example Publishing Institution",
        "references": f"https://publisher.test/occurrence/{key}",
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        "media": media if media is not None else [_media()],
    }
    if species_key is None:
        row.pop("speciesKey")
        row.pop("acceptedTaxonKey")
        row.pop("taxonKey")
    return row


def _payload(
    records: list[dict[str, object]],
    *,
    offset: int = 0,
    count: int | None = None,
    complete: bool = True,
) -> dict[str, object]:
    return {
        "offset": offset,
        "limit": 2,
        "count": len(records) if count is None else count,
        "endOfRecords": complete,
        "results": records,
    }


def _adapter(http_get: RecordedGBIF) -> GBIFReferenceAdapter:
    return GBIFReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        http_get=http_get,
        retrieved_at=lambda: NOW,
    )


def test_gbif_adapter_queries_exact_species_and_pages_by_occurrence_count() -> None:
    first_record = _occurrence(
        101,
        media=[
            _media(media_id="media-1"),
            _media(
                "https://publisher.test/images/two.jpg",
                media_id="media-2",
            ),
        ],
    )
    source = RecordedGBIF([_payload([first_record], count=2, complete=False)])
    page = _adapter(source).fetch_page(_query())

    assert source.requests == [
        (
            "/occurrence/search",
            {
                "taxonKey": "1938069",
                "mediaType": "StillImage",
                "hasCoordinate": "true",
                "limit": 2,
                "offset": 0,
                "geometry": GEOMETRY,
            },
        )
    ]
    assert page.observations.height == 1
    assert page.media_candidates.height == 2
    assert page.page_cursor == "0"
    assert page.next_cursor == "1"
    assert page.complete is False


def test_gbif_adapter_preserves_identity_provenance_and_separate_licences() -> None:
    source = RecordedGBIF([_payload([_occurrence(101)])])
    page = _adapter(source).fetch_page(_query())
    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)

    assert observation["source_observation_id"] == "101"
    assert observation["source_taxon_id"] == "1938069"
    assert observation["accepted_taxon_key"] == "gbif:1938069"
    assert observation["taxon_reconciliation_status"] == "accepted_key_exact"
    assert observation["source_dataset_key"] == "dataset-key-1"
    assert observation["source_dataset_doi"] == "10.15468/example"
    assert observation["source_record_url"] == "https://publisher.test/occurrence/101"
    assert observation["distance_to_cluster_medoid_km"] == pytest.approx(1.45, abs=0.1)
    assert observation["uncertain_taxon_match"] is False
    assert observation["basis_of_record_suitable"] is True
    assert media["provider_media_id"] == "media-1"
    assert media["media_identifier"] == "https://publisher.test/images/one.jpg"
    assert media["licence"] == "https://creativecommons.org/licenses/by/4.0/"
    assert media["occurrence_licence"] == (
        "https://creativecommons.org/publicdomain/zero/1.0/"
    )
    assert media["original_provider"] == "Example Publishing Institution"
    assert media["verification_status"] == "unreviewed"
    assert media["licence_policy_status"] == "unreviewed"
    assert media["download_status"] == "pending"


@pytest.mark.parametrize(
    ("record", "expected_reasons"),
    [
        (
            _occurrence(
                201,
                species_key=999,
                issues=["COORDINATE_INVALID"],
                media=[_media(licence=None)],
            ),
            {
                "uncertain_taxon_match",
                "geospatial_issue",
                "missing_media_licence",
            },
        ),
        (
            _occurrence(202, basis="PRESERVED_SPECIMEN"),
            {"preserved_specimen", "unsuitable_basis_of_record"},
        ),
        (
            _occurrence(203, basis="FOSSIL_SPECIMEN", status="ABSENT"),
            {"fossil", "occurrence_absent", "unsuitable_basis_of_record"},
        ),
    ],
)
def test_gbif_adapter_flags_ineligible_occurrences_and_media(
    record: dict[str, object],
    expected_reasons: set[str],
) -> None:
    page = _adapter(RecordedGBIF([_payload([record])])).fetch_page(_query())
    observation = page.observations.row(0, named=True)
    media = page.media_candidates.row(0, named=True)

    assert expected_reasons <= set(str(media["exclusion_reason"]).split(";"))
    assert media["download_status"] == "excluded"
    assert media["verification_status"] == "unreviewed"
    if "uncertain_taxon_match" in expected_reasons:
        assert observation["taxon_reconciliation_status"] == "conflict"
        assert observation["accepted_taxon_key"] is None
        assert observation["uncertain_taxon_match"] is True
        assert media["licence_policy_status"] == "quarantined"


def test_gbif_adapter_keeps_observation_but_requires_explicit_still_image_media() -> None:
    record = _occurrence(
        301,
        media=[
            _media(media_type="Sound", media_id="sound-1"),
            {"type": "StillImage", "license": "CC0"},
        ],
    )
    page = _adapter(RecordedGBIF([_payload([record])])).fetch_page(_query())

    assert page.observations.height == 1
    assert page.media_candidates.is_empty()


def test_gbif_geographic_fallback_parameters_and_bulk_download_predicates() -> None:
    country_query = _query(
        fallback_level=2,
        geometry_wkt=None,
        country_codes=("NZ", "AU"),
    )
    global_query = _query(fallback_level=3, geometry_wkt=None)
    source = RecordedGBIF([_payload([]), _payload([])])
    adapter = _adapter(source)
    adapter.fetch_page(country_query)
    adapter.fetch_page(global_query)

    assert source.requests[0][1]["country"] == ["AU", "NZ"]
    assert "geometry" not in source.requests[0][1]
    assert "country" not in source.requests[1][1]
    request = build_gbif_reference_download_request(country_query)
    assert request["format"] == "DWCA"
    predicates = request["predicate"]["predicates"]  # type: ignore[index]
    assert {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"} in predicates
    assert {
        "type": "in",
        "key": "COUNTRY_CODE",
        "values": ["AU", "NZ"],
    } in predicates


def test_gbif_adapter_requires_bulk_download_above_search_ceiling() -> None:
    source = RecordedGBIF([_payload([], count=100_001, complete=False)])
    with pytest.raises(GBIFReferenceBulkDownloadRequired) as captured:
        _adapter(source).fetch_page(_query())

    assert captured.value.total_records == 100_001
    assert captured.value.request_payload["format"] == "DWCA"
    assert captured.value.request_payload["predicate"]["predicates"][0] == {  # type: ignore[index]
        "type": "equals",
        "key": "TAXON_KEY",
        "value": "1938069",
    }


def test_gbif_adapter_stops_at_explicit_prototype_record_limit(
    tmp_path: Path,
) -> None:
    query = _query(page_size=2, maximum_records=4)
    source = RecordedGBIF(
        [
            _payload(
                [_occurrence(301), _occurrence(302)],
                count=100_001,
                complete=False,
            ),
            _payload(
                [_occurrence(303), _occurrence(304)],
                offset=2,
                count=100_001,
                complete=False,
            ),
        ]
    )

    pages = list(_adapter(source).iter_pages(query, checkpoint_dir=tmp_path))

    assert len(pages) == 2
    assert pages[-1].complete is True
    assert pages[-1].next_cursor is None
    checkpoint = load_gbif_reference_checkpoint(query, tmp_path)
    assert checkpoint is not None
    assert checkpoint.complete is True
    assert checkpoint.observation_count == 4
    assert list(
        _adapter(RecordedGBIF([])).iter_pages(query, checkpoint_dir=tmp_path)
    ) == []


def test_gbif_adapter_retries_rate_limits_with_identifying_user_agent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=_payload([]))

    delays: list[float] = []
    with GBIFReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        max_retries=1,
        retrieved_at=lambda: NOW,
        sleep=delays.append,
        transport=httpx.MockTransport(handler),
    ) as adapter:
        page = adapter.fetch_page(_query())

    assert len(requests) == 2
    assert requests[0].headers["User-Agent"].startswith("BioMiner/")
    assert delays == [0.25]
    assert page.request_count == 2
    assert page.retry_count == 1
    assert page.rate_limit_count == 1


def test_gbif_adapter_rejects_incompatible_taxon_and_scope_inputs() -> None:
    source = RecordedGBIF([_payload([])])
    mismatched = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au-east",
        fallback_level=3,
        source_taxon_id="999",
        source_snapshot_version="gbif-occurrence-2026-07-13",
    )
    with pytest.raises(ValueError, match="must equal"):
        _adapter(source).fetch_page(mismatched)
    with pytest.raises(ValueError, match="require geometry_wkt"):
        _adapter(source).fetch_page(_query(geometry_wkt=None))
    with pytest.raises(ValueError, match="requires country_codes"):
        _adapter(source).fetch_page(_query(fallback_level=2, geometry_wkt=None))


def test_gbif_page_checkpoints_are_integrity_checked_and_resumable(
    tmp_path: Path,
) -> None:
    query = _query(page_size=1)
    first_source = RecordedGBIF(
        [_payload([_occurrence(401)], count=2, complete=False)]
    )
    first_page = _adapter(first_source).fetch_page(query)
    checkpoint = write_gbif_reference_checkpoint(query, first_page, tmp_path)

    assert checkpoint.complete is False
    assert checkpoint.next_cursor == "1"
    assert checkpoint.page_count == 1
    second_source = RecordedGBIF(
        [_payload([_occurrence(402)], offset=1, count=2, complete=True)]
    )
    resumed_pages = list(
        _adapter(second_source).iter_pages(query, checkpoint_dir=tmp_path)
    )
    assert second_source.requests[0][1]["offset"] == 1
    assert len(resumed_pages) == 1
    complete = load_gbif_reference_checkpoint(query, tmp_path)
    assert complete is not None
    assert complete.complete is True
    assert complete.page_count == 2
    observations, media = load_gbif_reference_checkpoint_frames(query, tmp_path)
    assert observations["source_observation_id"].sort().to_list() == ["401", "402"]
    assert media.height == 2
    assert list(_adapter(RecordedGBIF([])).iter_pages(query, checkpoint_dir=tmp_path)) == []

    state = complete.checkpoint_directory / "state.json"
    payload = json.loads(state.read_text(encoding="utf-8"))
    media_path = complete.checkpoint_directory / payload["pages"][0]["media_file"]
    media_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_gbif_reference_checkpoint(query, tmp_path)


def test_checkpoint_load_deduplicates_identical_records_from_shifting_pages(
    tmp_path: Path,
) -> None:
    query = _query(page_size=1)
    first = GBIFReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        http_get=RecordedGBIF(
            [_payload([_occurrence(401)], count=2, complete=False)]
        ),
        retrieved_at=lambda: NOW,
    ).fetch_page(query)
    second = GBIFReferenceAdapter(
        registry_version="butterflies-v2-20260712",
        http_get=RecordedGBIF(
            [_payload([_occurrence(401)], offset=1, count=2, complete=True)]
        ),
        retrieved_at=lambda: NOW + timedelta(minutes=1),
    ).fetch_page(query, cursor="1")
    write_gbif_reference_checkpoint(query, first, tmp_path)
    write_gbif_reference_checkpoint(query, second, tmp_path)

    observations, media = load_gbif_reference_checkpoint_frames(query, tmp_path)

    assert observations.height == 1
    assert media.height == 1
    assert observations["retrieved_at"].item() == NOW
    assert media["retrieved_at"].item() == NOW


def test_checkpoint_load_rejects_conflicting_duplicate_source_records(
    tmp_path: Path,
) -> None:
    query = _query(page_size=1)
    changed = _occurrence(401)
    changed["decimalLatitude"] = -20.0
    first = _adapter(
        RecordedGBIF([_payload([_occurrence(401)], count=2, complete=False)])
    ).fetch_page(query)
    second = _adapter(
        RecordedGBIF([_payload([changed], offset=1, count=2, complete=True)])
    ).fetch_page(query, cursor="1")
    write_gbif_reference_checkpoint(query, first, tmp_path)
    write_gbif_reference_checkpoint(query, second, tmp_path)

    with pytest.raises(ValueError, match="contain conflicting rows"):
        load_gbif_reference_checkpoint_frames(query, tmp_path)


def test_gbif_cache_url_is_optional_and_does_not_replace_publisher_identifier() -> None:
    identifier = "https://publisher.test/images/one.jpg"
    cache_url = gbif_image_cache_url("101", identifier)

    assert cache_url.startswith(
        "https://api.gbif.org/v1/image/cache/occurrence/101/media/"
    )
    page = _adapter(RecordedGBIF([_payload([_occurrence(101)])])).fetch_page(_query())
    assert page.media_candidates["media_identifier"].item() == identifier


def test_reference_query_fingerprint_includes_geometry_and_medoid() -> None:
    first = _query()
    second = ReferenceSourceQuery(
        accepted_taxon_key=first.accepted_taxon_key,
        scientific_name=first.scientific_name,
        geo_cluster_id=first.geo_cluster_id,
        fallback_level=first.fallback_level,
        source_taxon_id=first.source_taxon_id,
        geometry_wkt=first.geometry_wkt,
        cluster_medoid_latitude=-33.0,
        cluster_medoid_longitude=151.0,
        page_size=first.page_size,
        source_snapshot_version=first.source_snapshot_version,
    )

    assert first.query_fingerprint != second.query_fingerprint


def test_checkpoint_frames_have_physical_contract_schemas(tmp_path: Path) -> None:
    query = _query()
    page = _adapter(RecordedGBIF([_payload([_occurrence(501)])])).fetch_page(query)
    write_gbif_reference_checkpoint(query, page, tmp_path)
    observations, media = load_gbif_reference_checkpoint_frames(query, tmp_path)

    assert isinstance(observations, pl.DataFrame)
    assert isinstance(media, pl.DataFrame)
    assert observations.height == 1
    assert media.height == 1
