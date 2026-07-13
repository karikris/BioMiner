from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

import httpx
import polars as pl

from biominer.geography import GeographicCoordinate, great_circle_distance_km
from biominer.references.checkpoints import (
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
    validate_reference_media_candidates,
    validate_reference_observations,
)
from biominer.references.source_base import ReferenceMetadataPage, ReferenceSourceQuery


logger = logging.getLogger(__name__)

INATURALIST_REFERENCE_SOURCE_VERSION = "inaturalist-v1-observations-reference-v1"
INATURALIST_API_BASE_URL = "https://api.inaturalist.org/v1"
INATURALIST_API_PAGE_SIZE = 200
INATURALIST_API_RESULT_LIMIT = 10_000
INATURALIST_MIN_REQUEST_INTERVAL_SECONDS = 1.0
INATURALIST_USER_AGENT = "BioMiner/0.1 (+https://github.com/karikris/BioMiner)"
INATURALIST_OBSERVATION_EXPORT_URL = "https://www.inaturalist.org/observations/export"
INATURALIST_GBIF_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
INATURALIST_GBIF_DATASET_DOI = "10.15468/ab3s5x"
DEFAULT_ACCEPTED_PHOTO_LICENCES = ("cc0", "cc-by", "cc-by-nc")

_INAT_OBSERVATION_PATTERN = re.compile(r"/observations/(\d+)(?:\D|$)")
_INAT_PHOTO_PATTERN = re.compile(r"/photos/(\d+)(?:/|\D|$)")
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_LICENCE_URIS = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
}

HTTPGet = Callable[[str, dict[str, object]], dict[str, object]]


class INaturalistBulkAcquisitionRequired(RuntimeError):
    def __init__(
        self,
        *,
        total_records: int,
        acquisition_options: dict[str, object],
    ) -> None:
        self.total_records = total_records
        self.acquisition_options = acquisition_options
        super().__init__(
            f"iNaturalist query returned {total_records} records; API searches above "
            f"{INATURALIST_API_RESULT_LIMIT} records must use an observation export "
            "or the weekly iNaturalist GBIF dataset"
        )


@dataclass(frozen=True, slots=True)
class INaturalistReferenceCheckpoint:
    query_fingerprint: str
    source_snapshot_version: str
    next_cursor: str | None
    complete: bool
    page_count: int
    observation_count: int
    media_candidate_count: int
    checkpoint_directory: Path


class INaturalistHTTPGet:
    def __init__(
        self,
        *,
        base_url: str = INATURALIST_API_BASE_URL,
        max_retries: int = 5,
        min_request_interval_seconds: float = INATURALIST_MIN_REQUEST_INTERVAL_SECONDS,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if (
            not math.isfinite(min_request_interval_seconds)
            or min_request_interval_seconds < INATURALIST_MIN_REQUEST_INTERVAL_SECONDS
        ):
            raise ValueError("iNaturalist requests must be limited to one per second or less")
        self.max_retries = max_retries
        self.min_request_interval_seconds = min_request_interval_seconds
        self.attempt_count = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": INATURALIST_USER_AGENT},
            transport=transport,
        )

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        last_error: BaseException | None = None
        for retry_index in range(self.max_retries + 1):
            self._pace()
            self.attempt_count += 1
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("iNaturalist response must be a JSON object")
                return payload
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 429
                ):
                    self.rate_limit_count += 1
                if not _is_retryable(exc) or retry_index >= self.max_retries:
                    raise
                self.retry_count += 1
                delay = _retry_delay(exc, retry_index)
                logger.warning(
                    "inaturalist.retry path=%s retry=%d/%d delay_seconds=%.2f error=%s",
                    path,
                    retry_index + 1,
                    self.max_retries,
                    delay,
                    type(exc).__name__,
                )
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    def _pace(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self.min_request_interval_seconds - (now - self._last_request_at)
            if remaining > 0.0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_request_at = now

    def close(self) -> None:
        self._client.close()


class INaturalistReferenceAdapter:
    source = "iNaturalist"
    source_version = INATURALIST_REFERENCE_SOURCE_VERSION
    user_agent = INATURALIST_USER_AGENT

    def __init__(
        self,
        *,
        registry_version: str,
        http_get: HTTPGet | None = None,
        accepted_photo_licences: Sequence[str] = DEFAULT_ACCEPTED_PHOTO_LICENCES,
        max_retries: int = 5,
        min_request_interval_seconds: float = INATURALIST_MIN_REQUEST_INTERVAL_SECONDS,
        retrieved_at: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.registry_version = _required_text(registry_version, field="registry_version")
        licences = tuple(
            sorted(
                {
                    _normalise_licence_code(value)
                    for value in accepted_photo_licences
                    if str(value or "").strip()
                }
            )
        )
        if not licences:
            raise ValueError("accepted_photo_licences must contain at least one licence")
        self.accepted_photo_licences = licences
        if http_get is not None and transport is not None:
            raise ValueError("http_get and transport are mutually exclusive")
        self._transport = (
            None
            if http_get is not None
            else INaturalistHTTPGet(
                max_retries=max_retries,
                min_request_interval_seconds=min_request_interval_seconds,
                sleep=sleep,
                monotonic=monotonic,
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
        source_taxon_id = _source_taxon_id(query.source_taxon_id)
        if query.page_size != INATURALIST_API_PAGE_SIZE:
            raise ValueError(
                f"iNaturalist reference queries must use page_size={INATURALIST_API_PAGE_SIZE}"
            )
        cursor_id = _cursor_id(cursor)
        params = _search_params(
            query,
            source_taxon_id=source_taxon_id,
            cursor_id=cursor_id,
            accepted_photo_licences=self.accepted_photo_licences,
        )
        counters_before = self._counter_snapshot()
        payload = self._http_get("/observations", params)
        counters_after = self._counter_snapshot()
        total_results = _nonnegative_int(
            payload.get("total_results", 0),
            field="total_results",
        )
        if total_results > INATURALIST_API_RESULT_LIMIT:
            raise INaturalistBulkAcquisitionRequired(
                total_records=total_results,
                acquisition_options=build_inaturalist_bulk_acquisition_options(query),
            )
        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list) or not all(
            isinstance(record, dict) for record in raw_results
        ):
            raise ValueError("iNaturalist observation results must be an array of objects")
        records = tuple(dict(record) for record in raw_results)
        if total_results < len(records):
            raise ValueError("iNaturalist total_results is below the returned row count")
        observation_ids = [_observation_id(record) for record in records]
        if observation_ids != sorted(observation_ids) or len(set(observation_ids)) != len(
            observation_ids
        ):
            raise ValueError("iNaturalist observations must be strictly ordered by ID")
        if cursor_id is not None and any(value <= cursor_id for value in observation_ids):
            raise ValueError("iNaturalist id_above page did not advance beyond its cursor")
        complete = len(records) < INATURALIST_API_PAGE_SIZE or total_results <= len(records)
        next_cursor = None if complete else str(observation_ids[-1])
        if not records and not complete:
            raise ValueError("iNaturalist returned an empty non-terminal page")

        retrieved_at = _utc_datetime(self._retrieved_at(), field="retrieved_at")
        observations: list[dict[str, object]] = []
        media_candidates: list[dict[str, object]] = []
        for record in records:
            observation, media = _normalise_observation(
                record,
                query=query,
                registry_version=self.registry_version,
                accepted_photo_licences=self.accepted_photo_licences,
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
            page_cursor="0" if cursor_id is None else str(cursor_id),
            next_cursor=next_cursor,
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
            load_inaturalist_reference_checkpoint(query, checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        if checkpoint is not None and checkpoint.complete:
            return
        cursor = checkpoint.next_cursor if checkpoint is not None else None
        while True:
            page = self.fetch_page(query, cursor=cursor)
            if checkpoint_dir is not None:
                write_inaturalist_reference_checkpoint(query, page, checkpoint_dir)
            yield page
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

    def __enter__(self) -> INaturalistReferenceAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def build_inaturalist_bulk_acquisition_options(
    query: ReferenceSourceQuery,
) -> dict[str, object]:
    predicates: list[dict[str, object]] = [
        {
            "type": "equals",
            "key": "DATASET_KEY",
            "value": INATURALIST_GBIF_DATASET_KEY,
        },
        {
            "type": "equals",
            "key": "TAXON_KEY",
            "value": _accepted_gbif_key(query.accepted_taxon_key),
        },
        {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
    ]
    if query.fallback_level < 3:
        predicates.append(
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"}
        )
    if query.fallback_level in {0, 1} and query.bounding_box is not None:
        predicates.append(
            {"type": "within", "geometry": _bounding_box_wkt(query.bounding_box)}
        )
    elif query.fallback_level == 2 and query.country_codes:
        predicates.append(
            {
                "type": "in",
                "key": "COUNTRY_CODE",
                "values": list(query.country_codes),
            }
        )
    return {
        "query_fingerprint": query.query_fingerprint,
        "accepted_taxon_key": query.accepted_taxon_key,
        "source_taxon_id": _source_taxon_id(query.source_taxon_id),
        "observation_export_url": INATURALIST_OBSERVATION_EXPORT_URL,
        "weekly_gbif_dataset_key": INATURALIST_GBIF_DATASET_KEY,
        "weekly_gbif_dataset_doi": INATURALIST_GBIF_DATASET_DOI,
        "weekly_gbif_download_request": {
            "format": "DWCA",
            "predicate": {"type": "and", "predicates": predicates},
        },
        "authentication_required_for_api": False,
        "authentication_required_for_weekly_gbif_download": True,
    }


def write_inaturalist_reference_checkpoint(
    query: ReferenceSourceQuery,
    page: ReferenceMetadataPage,
    output: str | Path,
) -> INaturalistReferenceCheckpoint:
    if page.source != INaturalistReferenceAdapter.source:
        raise ValueError("iNaturalist checkpoints cannot contain pages from another source")
    return _inaturalist_checkpoint(write_reference_page_checkpoint(query, page, output))


def load_inaturalist_reference_checkpoint(
    query: ReferenceSourceQuery,
    output: str | Path,
) -> INaturalistReferenceCheckpoint | None:
    checkpoint = load_reference_page_checkpoint(
        query,
        source=INaturalistReferenceAdapter.source,
        source_version=INATURALIST_REFERENCE_SOURCE_VERSION,
        output=output,
    )
    return None if checkpoint is None else _inaturalist_checkpoint(checkpoint)


def load_inaturalist_reference_checkpoint_frames(
    query: ReferenceSourceQuery,
    output: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return load_reference_page_checkpoint_frames(
        query,
        source=INaturalistReferenceAdapter.source,
        source_version=INATURALIST_REFERENCE_SOURCE_VERSION,
        output=output,
    )


def mark_inaturalist_gbif_media_duplicates(
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
) -> pl.DataFrame:
    validate_reference_observations(observations)
    validate_reference_media_candidates(media_candidates)
    observation_rows = observations.iter_rows(named=True)
    inaturalist_observation_ids: dict[str, str] = {}
    gbif_observation_ids: dict[str, str] = {}
    for row in observation_rows:
        source = str(row["source"])
        if source == "iNaturalist":
            inaturalist_observation_ids[str(row["reference_observation_id"])] = str(
                row["source_observation_id"]
            )
        elif source == "GBIF":
            observation_id = _inaturalist_observation_id(row.get("source_record_url"))
            if observation_id is not None:
                gbif_observation_ids[str(row["reference_observation_id"])] = observation_id
    direct_by_observation: dict[str, set[str]] = {}
    for row in media_candidates.iter_rows(named=True):
        reference_observation_id = str(row["reference_observation_id"])
        observation_id = inaturalist_observation_ids.get(reference_observation_id)
        if observation_id is None:
            continue
        keys = direct_by_observation.setdefault(observation_id, set())
        keys.update(_media_identity_keys(row))
    rows: list[dict[str, object]] = []
    for source_row in media_candidates.iter_rows(named=True):
        row = dict(source_row)
        if row["source"] == "GBIF":
            observation_id = gbif_observation_ids.get(str(row["reference_observation_id"]))
            direct_keys = direct_by_observation.get(str(observation_id))
            if direct_keys and direct_keys & _media_identity_keys(row):
                row["download_status"] = "excluded"
                row["exclusion_reason"] = _append_reason(
                    row.get("exclusion_reason"),
                    "duplicate_inaturalist_through_gbif",
                )
        rows.append(row)
    return reference_media_candidates_frame(rows)


def _search_params(
    query: ReferenceSourceQuery,
    *,
    source_taxon_id: str,
    cursor_id: int | None,
    accepted_photo_licences: tuple[str, ...],
) -> dict[str, object]:
    params: dict[str, object] = {
        "taxon_id": source_taxon_id,
        "quality_grade": "research",
        "rank": "species",
        "captive": "false",
        "photos": "true",
        "photo_license": ",".join(accepted_photo_licences),
        "taxon_is_active": "true",
        "per_page": INATURALIST_API_PAGE_SIZE,
        "page": 1,
        "order_by": "id",
        "order": "asc",
    }
    if cursor_id is not None:
        params["id_above"] = cursor_id
    if query.fallback_level in {0, 1}:
        if query.bounding_box is None:
            raise ValueError("local and buffered iNaturalist scopes require bounding_box")
        south, west, north, east = query.bounding_box
        params.update(
            {
                "swlat": south,
                "swlng": west,
                "nelat": north,
                "nelng": east,
                "geo": "true",
                "mappable": "true",
                "geoprivacy": "open",
            }
        )
    elif query.fallback_level == 2:
        if not query.source_place_ids:
            raise ValueError("country or bioregion iNaturalist fallback requires source_place_ids")
        params.update(
            {
                "place_id": ",".join(query.source_place_ids),
                "geo": "true",
                "mappable": "true",
                "geoprivacy": "open",
            }
        )
    elif query.fallback_level != 3:
        raise ValueError("iNaturalist reference fallback_level must be between 0 and 3")
    return params


def _normalise_observation(
    record: Mapping[str, object],
    *,
    query: ReferenceSourceQuery,
    registry_version: str,
    accepted_photo_licences: tuple[str, ...],
    retrieved_at: datetime,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    observation_id = str(_observation_id(record))
    reference_observation_id = make_reference_observation_id(
        "iNaturalist",
        observation_id,
    )
    taxon = record.get("taxon")
    taxon = taxon if isinstance(taxon, dict) else {}
    community_taxon = record.get("community_taxon")
    community_taxon = community_taxon if isinstance(community_taxon, dict) else {}
    compact_community_taxon_id = _optional_positive_id(
        record.get("community_taxon_id")
    )
    expanded_community_taxon_id = _optional_positive_id(community_taxon.get("id"))
    source_taxon_id = compact_community_taxon_id or expanded_community_taxon_id
    observation_taxon_id = _optional_positive_id(taxon.get("id"))
    expected_taxon_id = _source_taxon_id(query.source_taxon_id)
    taxon_rank = str(
        community_taxon.get("rank") or taxon.get("rank") or ""
    ).strip().casefold()
    community_identity_consistent = (
        source_taxon_id is not None
        and observation_taxon_id == source_taxon_id
        and (
            compact_community_taxon_id is None
            or expanded_community_taxon_id is None
            or compact_community_taxon_id == expanded_community_taxon_id
        )
    )
    exact_taxon = (
        source_taxon_id == expected_taxon_id
        and community_identity_consistent
        and taxon_rank == "species"
    )
    quality_grade = str(record.get("quality_grade") or "").strip().casefold()
    research_grade = quality_grade == "research"
    captive = _optional_bool(record.get("captive"))
    disagreement = _identification_disagreement(record)
    coordinates_obscured = _coordinates_obscured(record)
    latitude, longitude = _coordinates(record)
    regional_scope = query.fallback_level < 3
    geospatial_issue = regional_scope and (
        latitude is None or longitude is None or coordinates_obscured is not False
    )
    photos = record.get("photos") or []
    if not isinstance(photos, list):
        raise ValueError("iNaturalist observation photos must be an array")
    has_photos = any(isinstance(photo, dict) for photo in photos)
    basis_suitable = research_grade and captive is False and has_photos
    uncertain_taxon_match = not exact_taxon
    annotations = _annotations(record.get("annotations"))
    observer = record.get("user")
    observer = observer if isinstance(observer, dict) else {}
    observation = {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": reference_observation_id,
        "source": "iNaturalist",
        "source_observation_id": observation_id,
        "source_taxon_id": source_taxon_id,
        "supplied_scientific_name": _optional_text(
            community_taxon.get("name") or taxon.get("name")
        ),
        "accepted_taxon_key": query.accepted_taxon_key if exact_taxon else None,
        "reconciled_scientific_name": query.scientific_name if exact_taxon else None,
        "registry_version": registry_version,
        "taxon_reconciliation_status": (
            "accepted_key_exact"
            if exact_taxon
            else "conflict"
            if source_taxon_id is not None
            else "unresolved"
        ),
        "identification_quality": quality_grade or None,
        "community_taxon_status": taxon_rank or None,
        "identification_disagreement": disagreement,
        "captive_or_cultivated": captive,
        "observer_id": _optional_text(_value(observer, "id", "login")),
        "locality": _optional_text(_value(record, "place_guess", "locality")),
        "life_stage": _life_stage(annotations.get("life stage")),
        "sex": _optional_text(annotations.get("sex")),
        "observed_at": _observed_at(record),
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_uncertainty": _optional_nonnegative_float(
            _value(record, "public_positional_accuracy", "positional_accuracy")
        ),
        "coordinates_obscured": coordinates_obscured,
        "country": None,
        "country_code": None,
        "geo_cluster_id": query.geo_cluster_id,
        "distance_to_cluster_medoid_km": _distance_to_medoid(
            query,
            latitude=latitude,
            longitude=longitude,
        ),
        "source_dataset_key": None,
        "source_dataset_doi": None,
        "source_record_url": _observation_url(record, observation_id),
        "source_record_hash": _semantic_hash(record),
        "retrieved_at": retrieved_at,
        "source_snapshot_version": query.source_snapshot_version,
        "source_query_fingerprint": query.query_fingerprint,
        "fallback_level": query.fallback_level,
        "geospatial_issue": geospatial_issue,
        # The API does not expose authoritative specimen/fossil/absence flags.
        "preserved_specimen": None,
        "fossil": None,
        "occurrence_absent": None,
        "uncertain_taxon_match": uncertain_taxon_match,
        "basis_of_record_suitable": basis_suitable,
    }
    observation_exclusions = _observation_exclusions(
        observation,
        research_grade=research_grade,
        captive=captive,
        disagreement=disagreement,
        has_photos=has_photos,
        regional_scope=regional_scope,
        coordinates_obscured=coordinates_obscured,
    )
    occurrence_licence = _normalise_optional_licence(record.get("license_code"))
    observer_name = _optional_text(_value(observer, "name", "login"))
    media_rows: list[dict[str, object]] = []
    for default_position, photo in enumerate(photos):
        if not isinstance(photo, dict):
            continue
        photo_id = _optional_positive_id(photo.get("id"))
        identifier = _optional_text(
            _value(photo, "original_url", "large_url", "url")
        )
        if photo_id is None or identifier is None:
            continue
        photo_licence = _normalise_optional_licence(photo.get("license_code"))
        accepted_licence = photo_licence in accepted_photo_licences
        reasons = list(observation_exclusions)
        if photo_licence is None:
            reasons.append("missing_photo_licence")
        elif not accepted_licence:
            reasons.append(f"photo_licence_not_allowed:{photo_licence}")
        if observation_exclusions:
            download_status = "excluded"
        elif photo_licence is None:
            download_status = "quarantined"
        elif not accepted_licence:
            download_status = "excluded"
        else:
            download_status = "pending"
        dimensions = photo.get("original_dimensions")
        dimensions = dimensions if isinstance(dimensions, dict) else {}
        position = _optional_uint32(photo.get("position"))
        if position is None:
            position = default_position
        attribution = _optional_text(photo.get("attribution"))
        media_rows.append(
            {
                "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                "reference_media_id": make_reference_media_id(
                    "iNaturalist",
                    photo_id,
                    reference_observation_id,
                ),
                "reference_observation_id": reference_observation_id,
                "provider_media_id": photo_id,
                "source": "iNaturalist",
                "media_identifier": identifier,
                "media_type": "StillImage",
                "width": _optional_uint32(dimensions.get("width")),
                "height": _optional_uint32(dimensions.get("height")),
                "creator": observer_name,
                "rights_holder": observer_name,
                "licence": photo_licence,
                "licence_uri": _LICENCE_URIS.get(photo_licence or ""),
                "attribution": attribution,
                "occurrence_licence": occurrence_licence,
                "original_provider": "iNaturalist",
                "media_position": position,
                "source_checksum": None,
                "source_checksum_algorithm": None,
                "download_status": download_status,
                # Research Grade is community taxon evidence, not image verification.
                "verification_status": "unreviewed",
                "exclusion_reason": ";".join(reasons) or None,
                "licence_policy_status": (
                    "allowed"
                    if accepted_licence
                    else "quarantined"
                    if photo_licence is None
                    else "denied"
                ),
                "retrieved_at": retrieved_at,
                "source_snapshot_version": query.source_snapshot_version,
            }
        )
    return observation, media_rows


def _observation_exclusions(
    observation: Mapping[str, object],
    *,
    research_grade: bool,
    captive: bool | None,
    disagreement: bool,
    has_photos: bool,
    regional_scope: bool,
    coordinates_obscured: bool | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if bool(observation["uncertain_taxon_match"]):
        reasons.append("uncertain_taxon_match")
    if not research_grade:
        reasons.append("quality_grade_not_research")
    if captive is True:
        reasons.append("captive_or_cultivated")
    elif captive is None:
        reasons.append("captive_status_unknown")
    if disagreement:
        reasons.append("identification_disagreement")
    if not has_photos:
        reasons.append("missing_photos")
    if regional_scope and (observation["latitude"] is None or observation["longitude"] is None):
        reasons.append("regional_coordinates_missing")
    if regional_scope and coordinates_obscured is not False:
        reasons.append("regional_coordinates_obscured_or_unknown")
    return tuple(reasons)


def _inaturalist_checkpoint(
    checkpoint: ReferencePageCheckpoint,
) -> INaturalistReferenceCheckpoint:
    return INaturalistReferenceCheckpoint(
        query_fingerprint=checkpoint.query_fingerprint,
        source_snapshot_version=checkpoint.source_snapshot_version,
        next_cursor=checkpoint.next_cursor,
        complete=checkpoint.complete,
        page_count=checkpoint.page_count,
        observation_count=checkpoint.observation_count,
        media_candidate_count=checkpoint.media_candidate_count,
        checkpoint_directory=checkpoint.checkpoint_directory,
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUSES or exc.response.status_code >= 500
    return False


def _retry_delay(exc: BaseException, retry_index: int) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return min(60.0, 0.5 * (2**retry_index))


def _counter_deltas(
    before: tuple[int, int, int] | None,
    after: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    if before is None or after is None:
        return 1, 0, 0
    return tuple(max(0, end - start) for start, end in zip(before, after, strict=True))


def _source_taxon_id(value: object) -> str:
    text = str(value or "").strip().removeprefix("inaturalist:")
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("iNaturalist reference queries require a positive source_taxon_id")
    return str(int(text))


def _accepted_gbif_key(value: object) -> str:
    text = str(value or "").strip()
    if not text.startswith("gbif:"):
        raise ValueError("accepted_taxon_key must be source-qualified as gbif:<key>")
    bare = text.removeprefix("gbif:")
    if not bare.isdigit() or int(bare) <= 0:
        raise ValueError("accepted_taxon_key must contain a positive GBIF key")
    return str(int(bare))


def _bounding_box_wkt(bounds: tuple[float, float, float, float]) -> str:
    south, west, north, east = bounds
    return (
        f"POLYGON(({west:g} {south:g},{east:g} {south:g},"
        f"{east:g} {north:g},{west:g} {north:g},{west:g} {south:g}))"
    )


def _cursor_id(cursor: str | None) -> int | None:
    if cursor is None:
        return None
    text = str(cursor).strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("iNaturalist cursor must be a positive observation ID")
    return int(text)


def _observation_id(record: Mapping[str, object]) -> int:
    value = _optional_int(record.get("id"))
    if value is None or value <= 0:
        raise ValueError("iNaturalist observation is missing a positive ID")
    return value


def _coordinates(record: Mapping[str, object]) -> tuple[float | None, float | None]:
    geojson = record.get("geojson")
    if isinstance(geojson, dict):
        coordinates = geojson.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            longitude = _optional_float(coordinates[0])
            latitude = _optional_float(coordinates[1])
            if _valid_coordinate(latitude, longitude):
                return latitude, longitude
    location = _optional_text(record.get("location"))
    if location is not None:
        parts = [part.strip() for part in location.split(",", 1)]
        if len(parts) == 2:
            latitude = _optional_float(parts[0])
            longitude = _optional_float(parts[1])
            if _valid_coordinate(latitude, longitude):
                return latitude, longitude
    return None, None


def _coordinates_obscured(record: Mapping[str, object]) -> bool | None:
    explicit = _optional_bool(_value(record, "obscured", "coordinates_obscured"))
    if explicit is not None:
        return explicit
    privacy = {
        str(record.get("geoprivacy") or "").casefold(),
        str(record.get("taxon_geoprivacy") or "").casefold(),
    }
    if privacy & {"obscured", "private"}:
        return True
    if "open" in privacy:
        return False
    return None


def _identification_disagreement(record: Mapping[str, object]) -> bool:
    explicit = _optional_bool(record.get("identifications_most_disagree"))
    if explicit is True:
        return True
    disagreement_count = _optional_int(record.get("num_identification_disagreements"))
    if disagreement_count is not None and disagreement_count > 0:
        return True
    identifications = record.get("identifications") or []
    if not isinstance(identifications, list):
        return False
    return any(
        isinstance(identification, dict)
        and str(identification.get("category") or "").casefold() == "maverick"
        and _optional_bool(identification.get("current")) is not False
        for identification in identifications
    )


def _annotations(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    result: dict[str, str] = {}
    for annotation in value:
        if not isinstance(annotation, dict):
            continue
        attribute = annotation.get("controlled_attribute")
        controlled_value = annotation.get("controlled_value")
        if not isinstance(attribute, dict) or not isinstance(controlled_value, dict):
            continue
        label = str(attribute.get("label") or "").strip().casefold()
        selected = str(controlled_value.get("label") or "").strip()
        if label and selected and label not in result:
            result[label] = selected
    return result


def _life_stage(value: object) -> str:
    normalised = str(value or "").strip().casefold()
    if normalised in {"adult", "teneral", "imago"}:
        return "adult"
    if normalised in {"larva", "caterpillar"}:
        return "larva"
    if normalised in {"pupa", "chrysalis"}:
        return "pupa"
    if normalised == "egg":
        return "egg"
    return normalised or "unknown"


def _observed_at(record: Mapping[str, object]) -> datetime | None:
    for field in ("time_observed_at", "observed_on", "observed_on_string"):
        value = _optional_text(record.get(field))
        if value is None:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


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


def _observation_url(record: Mapping[str, object], observation_id: str) -> str:
    uri = _optional_text(record.get("uri"))
    if uri is not None and uri.casefold().startswith(("https://", "http://")):
        return uri
    return f"https://www.inaturalist.org/observations/{observation_id}"


def _media_identity_keys(row: Mapping[str, object]) -> set[str]:
    keys: set[str] = set()
    photo_id = _inaturalist_photo_id(row.get("media_identifier"))
    if photo_id is not None:
        keys.add(f"photo:{photo_id}")
    if row.get("source") == "iNaturalist":
        provider_id = _optional_positive_id(row.get("provider_media_id"))
        if provider_id is not None:
            keys.add(f"photo:{provider_id}")
    identifier = _optional_text(row.get("media_identifier"))
    if identifier is not None:
        keys.add(f"url:{identifier}")
    return keys


def _inaturalist_observation_id(value: object) -> str | None:
    match = _INAT_OBSERVATION_PATTERN.search(str(value or ""))
    return match.group(1) if match else None


def _inaturalist_photo_id(value: object) -> str | None:
    match = _INAT_PHOTO_PATTERN.search(str(value or ""))
    return match.group(1) if match else None


def _append_reason(value: object, reason: str) -> str:
    reasons = [item for item in str(value or "").split(";") if item]
    if reason not in reasons:
        reasons.append(reason)
    return ";".join(reasons)


def _normalise_optional_licence(value: object) -> str | None:
    text = str(value or "").strip()
    return _normalise_licence_code(text) if text else None


def _normalise_licence_code(value: object) -> str:
    return str(value or "").strip().casefold().replace("_", "-")


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


def _optional_positive_id(value: object) -> str | None:
    parsed = _optional_int(value)
    return str(parsed) if parsed is not None and parsed > 0 else None


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


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _valid_coordinate(latitude: float | None, longitude: float | None) -> bool:
    return (
        latitude is not None
        and longitude is not None
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


__all__ = [
    "DEFAULT_ACCEPTED_PHOTO_LICENCES",
    "INATURALIST_API_BASE_URL",
    "INATURALIST_API_PAGE_SIZE",
    "INATURALIST_API_RESULT_LIMIT",
    "INATURALIST_GBIF_DATASET_DOI",
    "INATURALIST_GBIF_DATASET_KEY",
    "INATURALIST_MIN_REQUEST_INTERVAL_SECONDS",
    "INATURALIST_OBSERVATION_EXPORT_URL",
    "INATURALIST_REFERENCE_SOURCE_VERSION",
    "INaturalistBulkAcquisitionRequired",
    "INaturalistHTTPGet",
    "INaturalistReferenceAdapter",
    "INaturalistReferenceCheckpoint",
    "build_inaturalist_bulk_acquisition_options",
    "load_inaturalist_reference_checkpoint",
    "load_inaturalist_reference_checkpoint_frames",
    "mark_inaturalist_gbif_media_duplicates",
    "write_inaturalist_reference_checkpoint",
]
