from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from email.utils import parsedate_to_datetime
import logging
import random
from time import sleep as default_sleep
from typing import Any

import httpx

from biominer.registry.enrichment import SpeciesContext


HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]
logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
USER_AGENT = "BioMiner/0.1 registry-enrichment"


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
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get("https://www.wikidata.org", max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        query = (
            "SELECT ?item ?itemLabel ?gbif WHERE { "
            f"?item wdt:P846 \"{context.accepted_taxon_key.removeprefix('gbif:')}\". "
            "OPTIONAL { ?item wdt:P846 ?gbif. } "
            "SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en,es,fr,de,zh,ja\". } } LIMIT 10"
        )
        payload = self._http_get("/w/api.php", {"action": "query", "format": "json", "list": "search", "srsearch": context.accepted_scientific_name})
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
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [_snapshot("Wikidata", "wikibase-api", query=query)]}


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


def _json_get(
    base_url: str,
    *,
    max_retries: int = 5,
    sleep: Callable[[float], None] = default_sleep,
) -> HTTPGet:
    client = httpx.Client(base_url=base_url, timeout=30.0, headers={"User-Agent": USER_AGENT})

    def get(path: str, params: dict[str, object]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = client.get(path, params=params)
                status_code = int(getattr(response, "status_code", 200))
                if status_code in RETRYABLE_STATUS_CODES and attempt <= max_retries:
                    wait_seconds = _retry_after_seconds(response.headers.get("Retry-After")) or _backoff_seconds(attempt)
                    logger.info(
                        "registry.enrichment.http_retry base_url=%s path=%s status=%d attempt=%d wait_seconds=%.3f",
                        base_url,
                        path,
                        status_code,
                        attempt,
                        wait_seconds,
                    )
                    sleep(wait_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                return {"results": payload}
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt > max_retries:
                    raise
                wait_seconds = _backoff_seconds(attempt)
                logger.info(
                    "registry.enrichment.http_retry base_url=%s path=%s error=transport attempt=%d wait_seconds=%.3f",
                    base_url,
                    path,
                    attempt,
                    wait_seconds,
                )
                sleep(wait_seconds)

    return get


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        parsed = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    delta = (parsed - datetime.now(parsed.tzinfo)).total_seconds()
    return max(delta, 0.0)


def _backoff_seconds(attempt: int) -> float:
    return min(30.0, (0.5 * (2 ** max(attempt - 1, 0))) + random.uniform(0.0, 0.25))


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
