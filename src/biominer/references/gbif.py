from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time

import httpx
import polars as pl

from biominer.geography import GeographicCoordinate, great_circle_distance_km
from biominer.references.checkpoints import (
    REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION,
    ReferencePageCheckpoint,
    load_reference_page_checkpoint,
    load_reference_page_checkpoint_frames,
    write_reference_page_checkpoint,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_media_candidates_frame,
    reference_observations_frame,
)
from biominer.references.source_base import (
    ReferenceMetadataPage,
    ReferenceSourceQuery,
    apply_reference_query_record_limit,
)
from biominer.registry.gbif import JSONPayload
from biominer.registry.gbif_production import GBIF_USER_AGENT, RetryingHTTPGet


GBIF_REFERENCE_SOURCE_VERSION = "gbif-occurrence-reference-v1"
GBIF_REFERENCE_CHECKPOINT_SCHEMA_VERSION = REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION
GBIF_OCCURRENCE_SEARCH_PAGE_LIMIT = 300
GBIF_OCCURRENCE_SEARCH_MAX_RECORDS = 100_000
GBIF_IMAGE_CACHE_BASE_URL = "https://api.gbif.org/v1/image/cache"

_OBSERVATION_BASIS = frozenset(
    {"HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION"}
)
_GEOSPATIAL_ISSUES = frozenset(
    {
        "CONTINENT_COUNTRY_MISMATCH",
        "CONTINENT_DERIVED_FROM_COORDINATES",
        "COORDINATE_INVALID",
        "COORDINATE_OUT_OF_RANGE",
        "COORDINATE_REPROJECTED",
        "COORDINATE_REPROJECTION_FAILED",
        "COORDINATE_ROUNDED",
        "COUNTRY_COORDINATE_MISMATCH",
        "COUNTRY_DERIVED_FROM_COORDINATES",
        "GEODETIC_DATUM_ASSUMED_WGS84",
        "GEODETIC_DATUM_INVALID",
        "PRESUMED_NEGATED_LATITUDE",
        "PRESUMED_NEGATED_LONGITUDE",
        "PRESUMED_SWAPPED_COORDINATE",
        "ZERO_COORDINATE",
    }
)

HTTPGet = Callable[[str, dict[str, object]], JSONPayload]


class GBIFReferenceBulkDownloadRequired(RuntimeError):
    def __init__(
        self,
        *,
        total_records: int,
        request_payload: dict[str, object],
    ) -> None:
        self.total_records = total_records
        self.request_payload = request_payload
        super().__init__(
            f"GBIF occurrence search returned {total_records} matching records; "
            f"the documented search ceiling is {GBIF_OCCURRENCE_SEARCH_MAX_RECORDS}, "
            "so an authenticated Darwin Core Archive occurrence download is required"
        )


@dataclass(frozen=True, slots=True)
class GBIFReferenceCheckpoint:
    query_fingerprint: str
    source_snapshot_version: str
    next_cursor: str | None
    complete: bool
    page_count: int
    observation_count: int
    media_candidate_count: int
    checkpoint_directory: Path


class GBIFReferenceAdapter:
    source = "GBIF"
    source_version = GBIF_REFERENCE_SOURCE_VERSION
    user_agent = GBIF_USER_AGENT

    def __init__(
        self,
        *,
        registry_version: str,
        http_get: HTTPGet | None = None,
        max_retries: int = 5,
        retrieved_at: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.registry_version = _required_text(
            registry_version,
            field="registry_version",
        )
        if http_get is not None and transport is not None:
            raise ValueError("http_get and transport are mutually exclusive")
        self._transport = (
            None
            if http_get is not None
            else RetryingHTTPGet(
                max_retries=max_retries,
                sleep=sleep,
                transport=transport,
            )
        )
        self._http_get = http_get or self._transport
        self._retrieved_at = retrieved_at or (lambda: datetime.now(UTC))

    def fetch_page(
        self,
        query: ReferenceSourceQuery,
        *,
        cursor: str | None = None,
    ) -> ReferenceMetadataPage:
        accepted_key = _accepted_gbif_key(query.accepted_taxon_key)
        source_taxon_key = _source_taxon_key(query.source_taxon_id, accepted_key)
        offset = _page_offset(cursor)
        if query.page_size > GBIF_OCCURRENCE_SEARCH_PAGE_LIMIT:
            raise ValueError(
                "GBIF occurrence search page_size cannot exceed "
                f"{GBIF_OCCURRENCE_SEARCH_PAGE_LIMIT}"
            )
        if (
            query.maximum_records is not None
            and query.maximum_records > GBIF_OCCURRENCE_SEARCH_MAX_RECORDS
        ):
            raise ValueError("GBIF maximum_records exceeds the search ceiling")
        if offset >= GBIF_OCCURRENCE_SEARCH_MAX_RECORDS:
            raise ValueError("GBIF occurrence search cursor reached the search ceiling")
        params = _search_params(query, source_taxon_key=source_taxon_key, offset=offset)
        counters_before = self._counter_snapshot()
        payload = self._http_get("/occurrence/search", params)
        counters_after = self._counter_snapshot()
        if not isinstance(payload, dict):
            raise ValueError("GBIF occurrence search response must be a JSON object")
        total = _nonnegative_int(payload.get("count"), field="count")
        if (
            total > GBIF_OCCURRENCE_SEARCH_MAX_RECORDS
            and query.maximum_records is None
        ):
            raise GBIFReferenceBulkDownloadRequired(
                total_records=total,
                request_payload=build_gbif_reference_download_request(query),
            )
        response_offset = _nonnegative_int(payload.get("offset", offset), field="offset")
        if response_offset != offset:
            raise ValueError(
                f"GBIF occurrence page offset {response_offset} did not match request {offset}"
            )
        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list) or not all(
            isinstance(record, dict) for record in raw_results
        ):
            raise ValueError("GBIF occurrence search results must be an array of objects")
        records = tuple(dict(record) for record in raw_results)
        next_offset = offset + len(records)
        complete = bool(payload.get("endOfRecords")) or next_offset >= total
        if not records and not complete:
            raise ValueError("GBIF occurrence search returned an empty non-terminal page")
        if not complete and next_offset >= GBIF_OCCURRENCE_SEARCH_MAX_RECORDS:
            raise GBIFReferenceBulkDownloadRequired(
                total_records=total,
                request_payload=build_gbif_reference_download_request(query),
            )

        retrieved_at = _utc_datetime(self._retrieved_at(), field="retrieved_at")
        observations: list[dict[str, object]] = []
        media_candidates: list[dict[str, object]] = []
        for record in records:
            observation, media = _normalize_occurrence(
                record,
                query=query,
                registry_version=self.registry_version,
                retrieved_at=retrieved_at,
            )
            observations.append(observation)
            media_candidates.extend(media)

        request_count, retry_count, rate_limit_count = _counter_deltas(
            counters_before,
            counters_after,
        )
        return ReferenceMetadataPage(
            source=self.source,
            source_version=self.source_version,
            query_fingerprint=query.query_fingerprint,
            page_cursor=str(offset),
            next_cursor=None if complete else str(next_offset),
            observations=reference_observations_frame(observations),
            media_candidates=reference_media_candidates_frame(media_candidates),
            request_count=request_count,
            retry_count=retry_count,
            rate_limit_count=rate_limit_count,
            complete=complete,
        )

    def iter_pages(
        self,
        query: ReferenceSourceQuery,
        *,
        checkpoint_dir: str | Path | None = None,
    ) -> Iterator[ReferenceMetadataPage]:
        checkpoint = (
            load_gbif_reference_checkpoint(query, checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        if checkpoint is not None and checkpoint.complete:
            return
        cursor = checkpoint.next_cursor if checkpoint is not None else None
        observation_count = (
            checkpoint.observation_count if checkpoint is not None else 0
        )
        while True:
            page = self.fetch_page(query, cursor=cursor)
            page = apply_reference_query_record_limit(
                query,
                page,
                prior_observation_count=observation_count,
            )
            if checkpoint_dir is not None:
                write_gbif_reference_checkpoint(query, page, checkpoint_dir)
            yield page
            observation_count += page.observations.height
            if page.complete:
                return
            cursor = page.next_cursor

    def _counter_snapshot(self) -> tuple[int, int, int] | None:
        if self._transport is None:
            return None
        return (
            self._transport.attempt_count,
            self._transport.retry_count,
            self._transport.rate_limit_count,
        )

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    def __enter__(self) -> GBIFReferenceAdapter:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def build_gbif_reference_download_request(
    query: ReferenceSourceQuery,
    *,
    notification_addresses: Sequence[str] = (),
) -> dict[str, object]:
    accepted_key = _accepted_gbif_key(query.accepted_taxon_key)
    source_taxon_key = _source_taxon_key(query.source_taxon_id, accepted_key)
    predicates: list[dict[str, object]] = [
        {"type": "equals", "key": "TAXON_KEY", "value": source_taxon_key},
        {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
    ]
    _append_geographic_download_predicate(predicates, query)
    payload: dict[str, object] = {
        # DWCA retains the multimedia extension and its per-item licence fields.
        "format": "DWCA",
        "predicate": {"type": "and", "predicates": predicates},
    }
    addresses = sorted(
        {str(address).strip() for address in notification_addresses if str(address).strip()}
    )
    if addresses:
        payload["notificationAddresses"] = addresses
        payload["sendNotification"] = True
    return payload


def gbif_image_cache_url(
    occurrence_key: str,
    media_identifier: str,
) -> str:
    key = _required_text(occurrence_key, field="occurrence_key")
    identifier = _required_text(media_identifier, field="media_identifier")
    if not key.isdigit() or int(key) <= 0:
        raise ValueError("occurrence_key must be a positive GBIF occurrence key")
    digest = hashlib.md5(identifier.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{GBIF_IMAGE_CACHE_BASE_URL}/occurrence/{int(key)}/media/{digest}"


def write_gbif_reference_checkpoint(
    query: ReferenceSourceQuery,
    page: ReferenceMetadataPage,
    output: str | Path,
) -> GBIFReferenceCheckpoint:
    if page.source != GBIFReferenceAdapter.source:
        raise ValueError("GBIF checkpoints cannot contain pages from another source")
    return _gbif_checkpoint(write_reference_page_checkpoint(query, page, output))


def load_gbif_reference_checkpoint(
    query: ReferenceSourceQuery,
    output: str | Path,
) -> GBIFReferenceCheckpoint | None:
    checkpoint = load_reference_page_checkpoint(
        query,
        source=GBIFReferenceAdapter.source,
        source_version=GBIF_REFERENCE_SOURCE_VERSION,
        output=output,
    )
    return None if checkpoint is None else _gbif_checkpoint(checkpoint)


def load_gbif_reference_checkpoint_frames(
    query: ReferenceSourceQuery,
    output: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return load_reference_page_checkpoint_frames(
        query,
        source=GBIFReferenceAdapter.source,
        source_version=GBIF_REFERENCE_SOURCE_VERSION,
        output=output,
    )


def _gbif_checkpoint(checkpoint: ReferencePageCheckpoint) -> GBIFReferenceCheckpoint:
    return GBIFReferenceCheckpoint(
        query_fingerprint=checkpoint.query_fingerprint,
        source_snapshot_version=checkpoint.source_snapshot_version,
        next_cursor=checkpoint.next_cursor,
        complete=checkpoint.complete,
        page_count=checkpoint.page_count,
        observation_count=checkpoint.observation_count,
        media_candidate_count=checkpoint.media_candidate_count,
        checkpoint_directory=checkpoint.checkpoint_directory,
    )


def _search_params(
    query: ReferenceSourceQuery,
    *,
    source_taxon_key: str,
    offset: int,
) -> dict[str, object]:
    params: dict[str, object] = {
        "taxonKey": source_taxon_key,
        "mediaType": "StillImage",
        "hasCoordinate": "true",
        "limit": query.page_size,
        "offset": offset,
    }
    if query.fallback_level in {0, 1}:
        if query.geometry_wkt is None:
            raise ValueError("local and buffered GBIF scopes require geometry_wkt")
        params["geometry"] = query.geometry_wkt
    elif query.fallback_level == 2:
        if not query.country_codes:
            raise ValueError("country GBIF fallback requires country_codes")
        params["country"] = list(query.country_codes)
    elif query.fallback_level == 3:
        pass
    else:
        raise ValueError("GBIF reference fallback_level must be between 0 and 3")
    return params


def _append_geographic_download_predicate(
    predicates: list[dict[str, object]],
    query: ReferenceSourceQuery,
) -> None:
    if query.fallback_level in {0, 1}:
        if query.geometry_wkt is None:
            raise ValueError("local and buffered GBIF scopes require geometry_wkt")
        predicates.append({"type": "within", "geometry": query.geometry_wkt})
    elif query.fallback_level == 2:
        if not query.country_codes:
            raise ValueError("country GBIF fallback requires country_codes")
        predicates.append(
            {"type": "in", "key": "COUNTRY_CODE", "values": list(query.country_codes)}
        )
    elif query.fallback_level != 3:
        raise ValueError("GBIF reference fallback_level must be between 0 and 3")


def _normalize_occurrence(
    record: Mapping[str, object],
    *,
    query: ReferenceSourceQuery,
    registry_version: str,
    retrieved_at: datetime,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    record_hash = _semantic_hash(record)
    occurrence_key = _optional_text(_value(record, "key", "gbifID"))
    if occurrence_key is None:
        raise ValueError("GBIF occurrence record is missing key/gbifID")
    reference_observation_id = make_reference_observation_id("GBIF", occurrence_key)
    target_key = _accepted_gbif_key(query.accepted_taxon_key)
    record_species_key = _optional_positive_key(_value(record, "speciesKey"))
    record_accepted_key = _optional_positive_key(_value(record, "acceptedTaxonKey"))
    record_taxon_key = _optional_positive_key(_value(record, "taxonKey"))
    reconciled = target_key in {
        value for value in (record_species_key, record_accepted_key, record_taxon_key) if value
    }
    has_any_record_key = any(
        value is not None
        for value in (record_species_key, record_accepted_key, record_taxon_key)
    )
    reconciliation_status = (
        "accepted_key_exact"
        if reconciled
        else "conflict"
        if has_any_record_key
        else "unresolved"
    )
    uncertain_taxon_match = not reconciled
    basis = _normalized_token(_value(record, "basisOfRecord"))
    occurrence_status = _normalized_token(_value(record, "occurrenceStatus"))
    issues = _issue_values(_value(record, "issues"))
    geospatial_issue = _bool_value(record.get("hasGeospatialIssues"), default=False) or bool(
        issues & _GEOSPATIAL_ISSUES
    )
    preserved_specimen = basis == "PRESERVED_SPECIMEN"
    fossil = basis == "FOSSIL_SPECIMEN"
    occurrence_absent = occurrence_status == "ABSENT"
    basis_suitable = basis in _OBSERVATION_BASIS
    latitude, longitude = _coordinate_pair(
        _value(record, "decimalLatitude"),
        _value(record, "decimalLongitude"),
    )
    distance = _distance_to_medoid(query, latitude=latitude, longitude=longitude)
    supplied_name = _optional_text(_value(record, "species", "scientificName"))
    source_taxon_id = record_species_key or record_accepted_key or record_taxon_key
    observation = {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": reference_observation_id,
        "source": "GBIF",
        "source_observation_id": occurrence_key,
        "source_taxon_id": source_taxon_id,
        "supplied_scientific_name": supplied_name,
        "accepted_taxon_key": query.accepted_taxon_key if reconciled else None,
        "reconciled_scientific_name": query.scientific_name if reconciled else None,
        "registry_version": registry_version,
        "taxon_reconciliation_status": reconciliation_status,
        "identification_quality": _optional_text(
            record.get("identificationVerificationStatus")
        ),
        "community_taxon_status": None,
        "identification_disagreement": uncertain_taxon_match,
        "captive_or_cultivated": _optional_bool(
            _value(record, "captive", "captiveOrCultivated")
        ),
        "observer_id": _optional_text(
            _value(record, "recordedByID", "recordedBy", "identifiedByID")
        ),
        "locality": _optional_text(
            _value(record, "locality", "verbatimLocality", "stateProvince")
        ),
        "life_stage": _life_stage(record.get("lifeStage")),
        "sex": _optional_text(record.get("sex")),
        "observed_at": _event_datetime(record),
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_uncertainty": _optional_nonnegative_float(
            record.get("coordinateUncertaintyInMeters")
        ),
        "coordinates_obscured": _optional_bool(record.get("coordinatesObscured")),
        "country": _optional_text(record.get("country")),
        "country_code": _country_code(record.get("countryCode")),
        "geo_cluster_id": query.geo_cluster_id,
        "distance_to_cluster_medoid_km": distance,
        "source_dataset_key": _optional_text(record.get("datasetKey")),
        "source_dataset_doi": _optional_text(
            _value(record, "datasetDOI", "datasetDoi")
        ),
        "source_record_url": _source_record_url(record, occurrence_key),
        "source_record_hash": record_hash,
        "retrieved_at": retrieved_at,
        "source_snapshot_version": query.source_snapshot_version,
        "source_query_fingerprint": query.query_fingerprint,
        "fallback_level": query.fallback_level,
        "geospatial_issue": geospatial_issue,
        "preserved_specimen": preserved_specimen,
        "fossil": fossil,
        "occurrence_absent": occurrence_absent,
        "uncertain_taxon_match": uncertain_taxon_match,
        "basis_of_record_suitable": basis_suitable,
    }
    observation_exclusions = _observation_exclusions(observation)
    raw_media = record.get("media") or []
    if not isinstance(raw_media, list):
        raise ValueError("GBIF occurrence media must be an array")
    publisher = _optional_text(
        _value(
            record,
            "publisher",
            "publishingOrganizationTitle",
            "publishingOrgKey",
            "datasetTitle",
        )
    )
    occurrence_licence = _optional_text(_value(record, "license", "licence"))
    media_rows: list[dict[str, object]] = []
    for position, media in enumerate(raw_media):
        if not isinstance(media, dict):
            continue
        media_type = _optional_text(_value(media, "type", "mediaType"))
        if media_type != "StillImage":
            continue
        identifier = _optional_text(media.get("identifier"))
        if identifier is None:
            continue
        provider_media_id = _provider_media_id(media, occurrence_key, position, identifier)
        media_licence = _optional_text(_value(media, "license", "licence"))
        exclusion_reasons = list(observation_exclusions)
        if media_licence is None:
            exclusion_reasons.append("missing_media_licence")
        exclusion_reason = ";".join(exclusion_reasons) or None
        if observation_exclusions:
            download_status = "excluded"
        elif media_licence is None:
            download_status = "quarantined"
        else:
            download_status = "pending"
        checksum, checksum_algorithm = _media_checksum(media)
        creator = _optional_text(media.get("creator"))
        rights_holder = _optional_text(_value(media, "rightsHolder", "rights_holder"))
        licence_uri = _optional_text(
            media.get("licenseUrl")
        ) or (media_licence if _is_http_url(media_licence) else None)
        attribution = _optional_text(media.get("attribution")) or _attribution(
            creator,
            rights_holder,
            media_licence,
        )
        media_rows.append(
            {
                "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                "reference_media_id": make_reference_media_id(
                    "GBIF",
                    provider_media_id,
                    reference_observation_id,
                ),
                "reference_observation_id": reference_observation_id,
                "provider_media_id": provider_media_id,
                "source": "GBIF",
                # Keep the publisher URL. Cache derivatives are optional and never replace it.
                "media_identifier": identifier,
                "media_type": media_type,
                "width": _optional_uint32(media.get("width")),
                "height": _optional_uint32(media.get("height")),
                "creator": creator,
                "rights_holder": rights_holder,
                "licence": media_licence,
                "licence_uri": licence_uri,
                "attribution": attribution,
                "occurrence_licence": occurrence_licence,
                "original_provider": _optional_text(media.get("publisher")) or publisher,
                "media_position": position,
                "source_checksum": checksum,
                "source_checksum_algorithm": checksum_algorithm,
                "download_status": download_status,
                # GBIF record labels are not evidence of manual image verification.
                "verification_status": "unreviewed",
                "exclusion_reason": exclusion_reason,
                "licence_policy_status": (
                    "quarantined" if media_licence is None else "unreviewed"
                ),
                "retrieved_at": retrieved_at,
                "source_snapshot_version": query.source_snapshot_version,
            }
        )
    return observation, media_rows


def _observation_exclusions(observation: Mapping[str, object]) -> tuple[str, ...]:
    checks = (
        ("uncertain_taxon_match", "uncertain_taxon_match"),
        ("geospatial_issue", "geospatial_issue"),
        ("preserved_specimen", "preserved_specimen"),
        ("fossil", "fossil"),
        ("occurrence_absent", "occurrence_absent"),
    )
    values = [reason for field, reason in checks if bool(observation[field])]
    if not bool(observation["basis_of_record_suitable"]):
        values.append("unsuitable_basis_of_record")
    return tuple(values)


def _accepted_gbif_key(value: object) -> str:
    text = str(value or "").strip()
    if not text.startswith("gbif:"):
        raise ValueError("accepted_taxon_key must be source-qualified as gbif:<key>")
    bare = text.removeprefix("gbif:")
    if not bare.isdigit() or int(bare) <= 0:
        raise ValueError("accepted_taxon_key must contain a positive GBIF key")
    return str(int(bare))


def _source_taxon_key(value: object, accepted_key: str) -> str:
    text = str(value or "").strip().removeprefix("gbif:") or accepted_key
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("source_taxon_id must be a positive GBIF key")
    normalized = str(int(text))
    if normalized != accepted_key:
        raise ValueError("source_taxon_id must equal the accepted GBIF species key")
    return normalized


def _page_offset(cursor: str | None) -> int:
    text = str(cursor or "0").strip()
    if not text.isdigit():
        raise ValueError("GBIF page cursor must be a nonnegative integer")
    return int(text)


def _counter_deltas(
    before: tuple[int, int, int] | None,
    after: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    if before is None or after is None:
        return 1, 0, 0
    return tuple(max(0, end - start) for start, end in zip(before, after, strict=True))


def _distance_to_medoid(
    query: ReferenceSourceQuery,
    *,
    latitude: float | None,
    longitude: float | None,
) -> float | None:
    if (
        latitude is None
        or longitude is None
        or query.cluster_medoid_latitude is None
        or query.cluster_medoid_longitude is None
    ):
        return None
    return great_circle_distance_km(
        GeographicCoordinate(latitude=latitude, longitude=longitude),
        GeographicCoordinate(
            latitude=query.cluster_medoid_latitude,
            longitude=query.cluster_medoid_longitude,
        ),
    )


def _coordinate_pair(latitude: object, longitude: object) -> tuple[float | None, float | None]:
    lat = _optional_float(latitude)
    lon = _optional_float(longitude)
    if lat is None or lon is None:
        return None, None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None, None
    return lat, lon


def _event_datetime(record: Mapping[str, object]) -> datetime | None:
    text = _optional_text(_value(record, "eventDate", "verbatimEventDate"))
    if text is not None:
        candidate = text.split("/", 1)[0].replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass
    year = _optional_int(record.get("year"))
    month = _optional_int(record.get("month")) or 1
    day = _optional_int(record.get("day")) or 1
    if year is None:
        return None
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _life_stage(value: object) -> str:
    normalized = str(value or "").strip().casefold().replace("_", " ")
    if normalized in {"imago", "adult"}:
        return "adult"
    if normalized in {"larva", "larvae", "caterpillar"}:
        return "larva"
    if normalized in {"pupa", "chrysalis"}:
        return "pupa"
    if normalized in {"egg", "eggs"}:
        return "egg"
    return normalized or "unknown"


def _provider_media_id(
    media: Mapping[str, object],
    occurrence_key: str,
    position: int,
    identifier: str,
) -> str:
    explicit = _optional_text(_value(media, "id", "mediaID", "mediaId"))
    if explicit is not None:
        return explicit
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:20]
    return f"{occurrence_key}:{position}:{digest}"


def _media_checksum(media: Mapping[str, object]) -> tuple[str | None, str | None]:
    checksum = _optional_text(media.get("checksum"))
    algorithm = _optional_text(media.get("checksumAlgorithm"))
    if checksum is not None and algorithm is not None:
        return checksum, algorithm
    for field, name in (("sha256", "SHA-256"), ("md5", "MD5")):
        value = _optional_text(media.get(field))
        if value is not None:
            return value, name
    return None, None


def _attribution(*values: str | None) -> str | None:
    unique: list[str] = []
    for value in values:
        if value is not None and value not in unique:
            unique.append(value)
    return " / ".join(unique) or None


def _source_record_url(record: Mapping[str, object], occurrence_key: str) -> str:
    for field in ("references", "occurrenceID"):
        value = _optional_text(record.get(field))
        if _is_http_url(value):
            assert value is not None
            return value
    return f"https://api.gbif.org/v1/occurrence/{occurrence_key}"


def _semantic_hash(record: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _utc_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _issue_values(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(",", ";").split(";")
    return {_normalized_token(item) for item in values if _normalized_token(item)}


def _normalized_token(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def _value(record: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in record:
            return record[name]
    return None


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _required_text(value: object, *, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_positive_key(value: object) -> str | None:
    text = str(value or "").strip().removeprefix("gbif:")
    if text.isdigit() and int(text) > 0:
        return str(int(text))
    return None


def _optional_int(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: object, *, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return parsed


def _optional_float(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_nonnegative_float(value: object) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _optional_uint32(value: object) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or not 0 <= parsed <= 4_294_967_295:
        return None
    return parsed


def _bool_value(value: object, *, default: bool) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _country_code(value: object) -> str | None:
    code = str(value or "").strip().upper()
    return code if len(code) == 2 and code.isalpha() else None


def _is_http_url(value: str | None) -> bool:
    return value is not None and value.casefold().startswith(("https://", "http://"))


__all__ = [
    "GBIF_IMAGE_CACHE_BASE_URL",
    "GBIF_OCCURRENCE_SEARCH_MAX_RECORDS",
    "GBIF_OCCURRENCE_SEARCH_PAGE_LIMIT",
    "GBIF_REFERENCE_CHECKPOINT_SCHEMA_VERSION",
    "GBIF_REFERENCE_SOURCE_VERSION",
    "GBIFReferenceAdapter",
    "GBIFReferenceBulkDownloadRequired",
    "GBIFReferenceCheckpoint",
    "build_gbif_reference_download_request",
    "gbif_image_cache_url",
    "load_gbif_reference_checkpoint",
    "load_gbif_reference_checkpoint_frames",
    "write_gbif_reference_checkpoint",
]
