from __future__ import annotations

from collections.abc import Callable, MutableMapping
from copy import deepcopy
import logging
import os
from time import sleep as default_sleep
from time import monotonic as default_monotonic
import threading
from typing import Any

import httpx

from biominer.common.http import RetryingHTTPClient
from biominer.registry.enrichment import SpeciesContext


HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]
CacheKey = tuple[str, tuple[tuple[str, str], ...]]
USER_AGENT = "BioMiner/0.1 registry-enrichment"
DEFAULT_WIKIDATA_MIN_DELAY_SECONDS = 1.5
DEFAULT_WIKIDATA_MAX_DELAY_SECONDS = 120.0
DEFAULT_WIKIDATA_RATE_LIMIT_COOLDOWN_SECONDS = 45.0
logger = logging.getLogger(__name__)


class WikidataRateLimitTermError(RuntimeError):
    """Raised when one Wikidata search term is skipped after HTTP 429 cooldown."""


class WikidataRateLimiter:
    def __init__(
        self,
        *,
        min_delay_seconds: float = DEFAULT_WIKIDATA_MIN_DELAY_SECONDS,
        max_delay_seconds: float = DEFAULT_WIKIDATA_MAX_DELAY_SECONDS,
        sleep: Callable[[float], None] = default_sleep,
        monotonic: Callable[[], float] = default_monotonic,
    ) -> None:
        if min_delay_seconds < 0:
            raise ValueError("min_delay_seconds must be >= 0")
        if max_delay_seconds < min_delay_seconds:
            raise ValueError("max_delay_seconds must be >= min_delay_seconds")
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._current_delay_seconds = min_delay_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._last_request_completed_at: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._monotonic()
            if self._last_request_completed_at is not None:
                wait_seconds = self._current_delay_seconds - (now - self._last_request_completed_at)
                if wait_seconds > 0:
                    logger.info("wikidata.rate_limit_sleep seconds=%.3f", wait_seconds)
                    self._sleep(wait_seconds)

    def record_request_complete(self) -> None:
        with self._lock:
            self._last_request_completed_at = self._monotonic()

    def retry_sleep(self, seconds: float) -> None:
        with self._lock:
            previous_delay = self._current_delay_seconds
            self._current_delay_seconds = min(self.max_delay_seconds, max(self.min_delay_seconds, previous_delay * 2))
            logger.info(
                "wikidata.retry_backoff_sleep retry_after_seconds=%.3f min_delay_seconds=%.3f previous_delay_seconds=%.3f next_delay_seconds=%.3f",
                seconds,
                self.min_delay_seconds,
                previous_delay,
                self._current_delay_seconds,
            )
        self._sleep(seconds)

    def rate_limit_cooldown(self, seconds: float) -> None:
        logger.info("wikidata.rate_limit_cooldown seconds=%.3f", seconds)
        self._sleep(seconds)


def _default_wikidata_min_delay_seconds() -> float:
    return _float_env("BIOMINER_WIKIDATA_MIN_DELAY_SECONDS", DEFAULT_WIKIDATA_MIN_DELAY_SECONDS)


def _default_wikidata_max_delay_seconds() -> float:
    return max(
        _default_wikidata_min_delay_seconds(),
        _float_env("BIOMINER_WIKIDATA_MAX_DELAY_SECONDS", DEFAULT_WIKIDATA_MAX_DELAY_SECONDS),
    )


def _default_wikidata_rate_limit_cooldown_seconds() -> float:
    return _float_env("BIOMINER_WIKIDATA_RATE_LIMIT_COOLDOWN_SECONDS", DEFAULT_WIKIDATA_RATE_LIMIT_COOLDOWN_SECONDS)


def _float_env(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.info("wikidata.invalid_float_env name=%s value=%s fallback=%.3f", name, raw, fallback)
        return fallback


_WIKIDATA_RATE_LIMITER = WikidataRateLimiter(
    min_delay_seconds=_default_wikidata_min_delay_seconds(),
    max_delay_seconds=_default_wikidata_max_delay_seconds(),
)
_WIKIDATA_CACHE: dict[CacheKey, dict[str, Any]] = {}
_WIKIDATA_CACHE_LOCK = threading.Lock()


class CatalogueOfLifeClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get("https://api.checklistbank.org", max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        payload = self._http_get("/dataset/3/nameusage/search", {"q": context.accepted_scientific_name, "limit": 10})
        rows = _result_rows(payload)
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for row in rows:
            scientific_name = _first_string(row, "scientificName", "name", "canonicalName")
            if scientific_name and scientific_name != context.accepted_scientific_name:
                continue
            source_id = _first_string(row, "id", "key", "usageKey")
            if source_id:
                links.append(_external_link(context, source="CoL", source_taxon_id=source_id, match_method="scientific_name"))
            for vernacular in _vernacular_values(row):
                assertions.append(_name_assertion(context, vernacular, source="CoL", source_record_id=f"col:{source_id}:vernacular:{vernacular}", trust_tier="T2"))
            for synonym in _list_values(row, "synonyms"):
                name = _first_string(synonym, "scientificName", "name", "canonicalName") if isinstance(synonym, dict) else str(synonym or "")
                if name:
                    assertions.append(
                        _name_assertion(
                            context,
                            name,
                            source="CoL",
                            source_record_id=f"col:{source_id}:synonym:{name}",
                            name_class="scientific_synonym",
                            language="la",
                            script="Latn",
                            trust_tier="T1",
                            precision_tier="high",
                            confidence="medium",
                        )
                    )
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [_snapshot("CoL", "checklistbank-dataset-3")]}


class WikidataClient:
    def __init__(
        self,
        *,
        http_get: HTTPGet | None = None,
        max_retries: int = 5,
        rate_limiter: WikidataRateLimiter | None = None,
        rate_limit_cooldown_seconds: float | None = None,
        cache: MutableMapping[CacheKey, dict[str, Any]] | None = None,
    ) -> None:
        self._rate_limiter = rate_limiter or _WIKIDATA_RATE_LIMITER
        # Wikidata 429s skip the current term; retrying the same term keeps the job in a penalty loop.
        self._http_get = http_get or _json_get("https://www.wikidata.org", max_retries=0, sleep=self._rate_limiter.retry_sleep)
        self._rate_limit_cooldown_seconds = (
            _default_wikidata_rate_limit_cooldown_seconds() if rate_limit_cooldown_seconds is None else rate_limit_cooldown_seconds
        )
        self._cache = cache if cache is not None else _WIKIDATA_CACHE

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "limit": 10,
            "maxlag": 5,
            "search": context.accepted_scientific_name,
            "type": "item",
        }
        payload = self._cached_rate_limited_get("/w/api.php", params)
        rows = _result_rows(payload)
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for row in rows:
            qid = _first_string(row, "title", "id", "item")
            label = _first_string(row, "label", "title", "itemLabel")
            if not qid or not label:
                continue
            links.append(_external_link(context, source="Wikidata", source_taxon_id=qid, match_method="gbif_taxon_id"))
            if label != context.accepted_scientific_name:
                assertions.append(_name_assertion(context, label, source="Wikidata", source_record_id=f"wikidata:{qid}:label", trust_tier="T3"))
            for alias in _list_values(row, "aliases"):
                alias_text = _first_string(alias, "value", "label") if isinstance(alias, dict) else str(alias or "")
                if alias_text:
                    assertions.append(
                        _name_assertion(
                            context,
                            alias_text,
                            source="Wikidata",
                            source_record_id=f"wikidata:{qid}:alias:{alias_text}",
                            trust_tier="T3",
                            confidence="low",
                            enabled=False,
                            review_state="candidate",
                            disabled_reason="wikidata_alias_requires_corroboration",
                        )
                    )
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [_snapshot("Wikidata", "wikibase-api", query=f"wbsearchentities:{context.accepted_scientific_name}")]}

    def _cached_rate_limited_get(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        cache_key = _cache_key(path, params)
        with _WIKIDATA_CACHE_LOCK:
            cached_payload = self._cache.get(cache_key)
            if cached_payload is not None:
                logger.info("wikidata.cache_hit search=%s", params.get("search"))
                return deepcopy(cached_payload)
        self._rate_limiter.wait()
        try:
            payload = self._http_get(path, params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After", "")
                self._rate_limiter.rate_limit_cooldown(self._rate_limit_cooldown_seconds)
                raise WikidataRateLimitTermError(f"wikidata HTTP 429 retry_after={retry_after}") from exc
            raise
        finally:
            self._rate_limiter.record_request_complete()
        with _WIKIDATA_CACHE_LOCK:
            self._cache[cache_key] = deepcopy(payload)
        return payload


class ITISClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get("https://www.itis.gov", max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        payload = self._http_get("/ITISWebService/jsonservice/searchByScientificName", {"srchKey": context.accepted_scientific_name})
        rows = _result_rows(payload)
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for row in rows:
            name = _first_string(row, "combinedName", "sciName", "scientificName")
            if name and name != context.accepted_scientific_name:
                continue
            tsn = _first_string(row, "tsn")
            if not tsn:
                continue
            links.append(_external_link(context, source="ITIS", source_taxon_id=tsn, match_method="scientific_name"))
            common_payload = self._http_get("/ITISWebService/jsonservice/getCommonNamesFromTSN", {"tsn": tsn})
            for common in _result_rows(common_payload):
                common_name = _first_string(common, "commonName", "vernacularName")
                language = _first_string(common, "language", "commonNameLanguage") or "eng"
                if common_name:
                    assertions.append(
                        _name_assertion(
                            context,
                            common_name,
                            source="ITIS",
                            source_record_id=f"itis:{tsn}:common:{common_name}",
                            language=language,
                            script="Latn",
                            trust_tier="T2",
                        )
                    )
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [_snapshot("ITIS", "itis-jsonservice")]}


class INaturalistClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get("https://api.inaturalist.org", max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        params = {"q": context.accepted_scientific_name, "rank": "species", "is_active": True, "all_names": True, "per_page": 10}
        payload = self._http_get("/v1/taxa", params)
        match = _inaturalist_exact_species_match(payload, context)
        if match is None:
            payload = self._http_get("/v1/taxa/autocomplete", params)
            match = _inaturalist_exact_species_match(payload, context)
        snapshot = _snapshot("iNaturalist", "inaturalist-v1", query=f"taxa:q={context.accepted_scientific_name}")
        if match is None:
            return {"name_assertions": [], "external_links": [], "source_snapshots": [snapshot]}

        source_taxon_id = _first_string(match, "id")
        assertions = _inaturalist_name_assertions(context, match, source_taxon_id=source_taxon_id)
        links = []
        if source_taxon_id:
            links.append(
                {
                    **_external_link(context, source="iNaturalist", source_taxon_id=source_taxon_id, match_method="scientific_name"),
                    "lineage_check": "gbif_scientific_name_exact",
                }
            )
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [snapshot]}


def _json_get(
    base_url: str,
    *,
    max_retries: int = 5,
    sleep: Callable[[float], None] = default_sleep,
) -> HTTPGet:
    client = RetryingHTTPClient(
        base_url=base_url,
        max_retries=max_retries,
        timeout_seconds=30.0,
        sleep=sleep,
        headers={"User-Agent": USER_AGENT},
    )

    def get(path: str, params: dict[str, object]) -> dict[str, Any]:
        payload = client.get_json(path, params=params)
        if isinstance(payload, dict):
            return payload
        return {"results": payload}

    return get


def _cache_key(path: str, params: dict[str, object]) -> CacheKey:
    return (path, tuple(sorted((str(key), str(value)) for key, value in params.items())))


def _name_assertion(
    context: SpeciesContext,
    display_name: str,
    *,
    source: str,
    source_record_id: str,
    language: str = "eng",
    script: str = "Latn",
    name_class: str = "vernacular",
    trust_tier: str = "T2",
    precision_tier: str = "medium",
    confidence: str = "high",
    enabled: bool = True,
    review_state: str = "accepted",
    disabled_reason: str = "",
    source_taxon_id: str = "",
) -> dict[str, Any]:
    return {
        "accepted_taxon_key": context.accepted_taxon_key,
        "verbatim_name": display_name,
        "display_name": display_name,
        "language": language,
        "script": script,
        "region": "",
        "bbox": "",
        "name_class": name_class,
        "source": source,
        "source_record_id": source_record_id,
        "source_taxon_id": source_taxon_id,
        "trust_tier": trust_tier,
        "precision_tier": precision_tier,
        "confidence": confidence,
        "enabled": enabled,
        "review_state": review_state,
        "disabled_reason": disabled_reason,
    }


def _external_link(context: SpeciesContext, *, source: str, source_taxon_id: str, match_method: str) -> dict[str, Any]:
    return {
        "accepted_taxon_key": context.accepted_taxon_key,
        "source": source,
        "source_taxon_id": source_taxon_id,
        "match_method": match_method,
        "match_confidence": "high",
        "lineage_check": "accepted_taxon_key",
    }


def _snapshot(source: str, version: str, *, query: str = "") -> dict[str, str]:
    return {
        "source": source,
        "source_version": version,
        "retrieved_at": "",
        "source_path": query,
        "source_response_hash": "",
        "licence": "",
    }


def _result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "result", "search", "scientificNames", "commonNames"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    if isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return [payload]


def _first_string(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _list_values(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return value if isinstance(value, list) else []


def _vernacular_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("vernacularNames", "commonNames", "names"):
        for item in _list_values(row, key):
            if isinstance(item, dict):
                value = _first_string(item, "name", "vernacularName", "commonName")
            else:
                value = str(item or "")
            if value:
                values.append(value)
    return values


def _inaturalist_exact_species_match(payload: dict[str, Any], context: SpeciesContext) -> dict[str, Any] | None:
    for row in _result_rows(payload):
        if str(row.get("rank") or "").lower() != "species":
            continue
        if row.get("is_active") is False:
            continue
        if _first_string(row, "name") == context.accepted_scientific_name:
            return row
    return None


def _inaturalist_name_assertions(context: SpeciesContext, row: dict[str, Any], *, source_taxon_id: str) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_name(
        display_name: str,
        *,
        source_record_id: str,
        language: str = "en",
        trust_tier: str = "T4",
        confidence: str = "medium",
        enabled: bool = False,
        review_state: str = "candidate",
        disabled_reason: str = "inaturalist_name_requires_review",
    ) -> None:
        normalized = display_name.strip()
        if not normalized or normalized == context.accepted_scientific_name or normalized in seen:
            return
        seen.add(normalized)
        assertions.append(
            _name_assertion(
                context,
                normalized,
                source="iNaturalist",
                source_record_id=source_record_id,
                language=language,
                script="Latn",
                trust_tier=trust_tier,
                precision_tier="medium",
                confidence=confidence,
                enabled=enabled,
                review_state=review_state,
                disabled_reason=disabled_reason,
                source_taxon_id=source_taxon_id,
            )
        )

    preferred = _first_string(row, "preferred_common_name", "english_common_name")
    if preferred:
        append_name(
            preferred,
            source_record_id=f"inaturalist:{source_taxon_id}:preferred_common_name",
            language=_inaturalist_language(row, "en"),
            trust_tier="T2",
            confidence="high",
            enabled=True,
            review_state="accepted",
            disabled_reason="",
        )
    matched_term = _first_string(row, "matched_term")
    if matched_term:
        append_name(matched_term, source_record_id=f"inaturalist:{source_taxon_id}:matched_term:{matched_term}")

    for index, item in enumerate(_inaturalist_taxon_name_items(row)):
        display_name = _first_string(item, "name", "lexicon_name", "vernacularName", "commonName")
        source_record_fragment = _first_string(item, "id", "source_id") or str(index)
        append_name(
            display_name,
            source_record_id=f"inaturalist:{source_taxon_id}:taxon_name:{source_record_fragment}",
            language=_inaturalist_language(item, "en"),
        )
    return assertions


def _inaturalist_taxon_name_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("taxon_names", "names", "common_names", "all_names"):
        value = row.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _inaturalist_language(row: dict[str, Any], fallback: str) -> str:
    locale = _first_string(row, "locale", "language", "lang")
    if locale:
        return locale.split("-")[0].lower()
    lexicon = _first_string(row, "lexicon").lower()
    if lexicon == "english":
        return "en"
    if lexicon == "scientific names":
        return "la"
    return fallback
