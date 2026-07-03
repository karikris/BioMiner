from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import random
from time import sleep as default_sleep
from typing import Any

import httpx

from biominer.registry.enrichment import SpeciesContext
from biominer.registry.normalize import normalize_name_key


HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]
GraphQLPost = Callable[[dict[str, Any]], dict[str, Any]]
logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
USER_AGENT = "BioMiner/0.1 registry-enrichment"
TMD_TAXONOMY_GRAPHQL_URL = "https://web.app.ufz.de/biome-taxonomy-api/graphql"
TMD_SCIENTIFIC_PROJECT_ID = "407"
TMD_GERMAN_PROJECT_ID = "410"
TMD_SOURCE_VERSION = "tmd-taxonomy-graphql-projects-407-410"
WIKIDATA_WDQS_URL = "https://query.wikidata.org"
WIKIDATA_SOURCE_VERSION = "wikidata-wdqs-p225-p846-p1843-labels-aliases"


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
        payload = self._http_get(
            "/v1/taxa",
            {
                "q": context.accepted_scientific_name,
                "rank": "species",
                "per_page": 10,
            },
        )
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for row in _result_rows(payload):
            scientific_name = _first_string(row, "name", "scientific_name")
            if scientific_name and scientific_name != context.accepted_scientific_name:
                continue
            taxon_id = _first_string(row, "id")
            if taxon_id:
                links.append(_external_link(context, source="iNaturalist", source_taxon_id=taxon_id, match_method="scientific_name"))
            common_name = _first_string(row, "preferred_common_name", "english_common_name")
            if common_name:
                assertions.append(
                    _name_assertion(
                        context,
                        common_name,
                        source="iNaturalist",
                        source_record_id=f"inaturalist:{taxon_id}:preferred_common_name",
                        language="eng",
                        script="Latn",
                        trust_tier="T2",
                        precision_tier="medium",
                        confidence="high",
                    )
                )
            for taxon_name in _list_values(row, "names"):
                if not isinstance(taxon_name, dict):
                    continue
                value = _first_string(taxon_name, "name")
                locale = _first_string(taxon_name, "locale", "lexicon") or "eng"
                if value:
                    assertions.append(
                        _name_assertion(
                            context,
                            value,
                            source="iNaturalist",
                            source_record_id=f"inaturalist:{taxon_id}:name:{locale}:{value}",
                            language=locale,
                            script="Latn",
                            trust_tier="T4",
                            precision_tier="medium",
                            confidence="medium",
                        )
                    )
        return {"name_assertions": assertions, "external_links": links, "source_snapshots": [_snapshot("iNaturalist", "inaturalist-v1-taxa")]}


class WikidataClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get(WIKIDATA_WDQS_URL, max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        query = _wikidata_taxon_query(context.accepted_scientific_name)
        payload = self._http_get("/sparql", {"query": query, "format": "json"})
        bindings = payload.get("results", {}).get("bindings", [])
        if not isinstance(bindings, list):
            bindings = []
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        linked_taxa: set[str] = set()
        seen_assertions: set[tuple[str, str, str]] = set()
        for binding in bindings:
            if not isinstance(binding, dict) or not _wikidata_link_is_confident(binding, context):
                continue
            qid = _wikidata_entity_id(_binding_value(binding, "taxon"))
            if not qid:
                continue
            if qid not in linked_taxa:
                links.append(_external_link(context, source="Wikidata", source_taxon_id=qid, match_method="P225+P846"))
                linked_taxa.add(qid)
            for name, language, field in _wikidata_name_values(binding):
                if normalize_name_key(name) == normalize_name_key(context.accepted_scientific_name):
                    continue
                key = (name.casefold(), language, field)
                if key in seen_assertions:
                    continue
                seen_assertions.add(key)
                assertions.append(
                    _name_assertion(
                        context,
                        name,
                        source="Wikidata",
                        source_record_id=f"wikidata:{qid}:{field}:{language}:{name}",
                        language=language,
                        script="",
                        name_class="vernacular" if field == "P1843" else "vernacular_alias",
                        trust_tier="T3",
                        precision_tier="medium",
                        confidence="high",
                        enabled=True,
                        review_state="accepted",
                    )
                )
        return {
            "name_assertions": assertions,
            "external_links": links,
            "source_snapshots": [_snapshot("Wikidata", WIKIDATA_SOURCE_VERSION, query=query)],
        }


class TMDGermanClient:
    def __init__(self, *, graphql_post: GraphQLPost | None = None, max_retries: int = 5) -> None:
        self._graphql_post = graphql_post or _graphql_post(TMD_TAXONOMY_GRAPHQL_URL, max_retries=max_retries)

    def enrich_registry(self, *, taxa_rows: list[dict[str, Any]], name_rows: list[dict[str, Any]]) -> dict[str, Any]:
        scientific_rows, scientific_request_count = self._fetch_project(TMD_SCIENTIFIC_PROJECT_ID)
        german_rows, german_request_count = self._fetch_project(TMD_GERMAN_PROJECT_ID)
        german_by_id = {str(row.get("ptnameId") or ""): row for row in german_rows}
        accepted_lookup = _accepted_species_lookup(taxa_rows)
        synonym_lookup = _unambiguous_synonym_lookup(name_rows)
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        coverage = {
            "scientific_rows_fetched": len(scientific_rows),
            "german_rows_fetched": len(german_rows),
            "joined_ptname_rows": 0,
            "mapped_accepted_name_rows": 0,
            "mapped_synonym_rows": 0,
            "unmapped_rows": 0,
            "out_of_scope_rows": 0,
            "skipped_complex_rows": 0,
            "skipped_parent_rank_rows": 0,
            "family_genus_german_labels": "not_available_from_tmd",
            "request_count": scientific_request_count + german_request_count,
        }
        snapshot_payload = {"scientific": scientific_rows, "german": german_rows}

        for scientific in scientific_rows:
            ptname_id = str(scientific.get("ptnameId") or "")
            german = german_by_id.get(ptname_id)
            if german is None:
                coverage["unmapped_rows"] += 1
                continue
            coverage["joined_ptname_rows"] += 1
            scientific_name = _tmd_scientific_name(scientific)
            if not scientific_name:
                coverage["skipped_parent_rank_rows"] += 1
                continue
            if _is_tmd_complex(scientific):
                coverage["skipped_complex_rows"] += 1
                continue
            german_name = str(german.get("uiLabel") or german.get("species") or "").strip()
            if not german_name:
                coverage["unmapped_rows"] += 1
                continue
            match_key = normalize_name_key(scientific_name)
            accepted_taxon_key = accepted_lookup.get(match_key)
            match_method = "scientific_name"
            confidence = "high"
            if accepted_taxon_key:
                coverage["mapped_accepted_name_rows"] += 1
            else:
                accepted_taxon_key = synonym_lookup.get(match_key)
                match_method = "scientific_synonym"
                confidence = "medium"
                if accepted_taxon_key:
                    coverage["mapped_synonym_rows"] += 1
            if not accepted_taxon_key:
                coverage["out_of_scope_rows"] += 1
                continue
            source_record_id = f"tmd:{TMD_GERMAN_PROJECT_ID}:{ptname_id}:{german_name}"
            assertions.append(
                {
                    "accepted_taxon_key": accepted_taxon_key,
                    "verbatim_name": german_name,
                    "display_name": german_name,
                    "language": "deu",
                    "script": "Latn",
                    "region": "DE",
                    "bbox": "",
                    "name_class": "vernacular",
                    "source": "TMD",
                    "source_record_id": source_record_id,
                    "source_taxon_id": ptname_id,
                    "trust_tier": "T2",
                    "precision_tier": "high",
                    "confidence": confidence,
                    "enabled": True,
                    "review_state": "accepted",
                    "disabled_reason": "",
                }
            )
            links.append(
                {
                    "accepted_taxon_key": accepted_taxon_key,
                    "source": "TMD",
                    "source_taxon_id": ptname_id,
                    "match_method": match_method,
                    "match_confidence": confidence,
                    "lineage_check": "accepted_taxon_key",
                }
            )

        return {
            "name_assertions": assertions,
            "external_links": links,
            "source_snapshots": [
                {
                    "source": "TMD",
                    "source_version": TMD_SOURCE_VERSION,
                    "retrieved_at": "",
                    "source_path": TMD_TAXONOMY_GRAPHQL_URL,
                    "source_response_hash": _payload_hash(snapshot_payload),
                    "licence": "",
                }
            ],
            "coverage": coverage,
        }

    def _fetch_project(self, project_id: str) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        after: str | None = None
        request_count = 0
        while True:
            payload = self._graphql_post(
                {
                    "query": _TMD_TAXON_ENTRIES_QUERY,
                    "variables": {"project": project_id, "first": 500, "after": after},
                }
            )
            request_count += 1
            if payload.get("errors"):
                raise ValueError("TMD GraphQL response contained errors")
            connection = payload.get("data", {}).get("taxonEntries", {})
            rows.extend(
                edge["node"]
                for edge in connection.get("edges", [])
                if isinstance(edge, dict) and isinstance(edge.get("node"), dict)
            )
            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            if not after:
                break
        return rows, request_count


_TMD_TAXON_ENTRIES_QUERY = """
query TMDTaxonEntries($project: String, $first: Int, $after: String) {
  taxonEntries(first: $first, after: $after, projectId_is: $project) {
    totalCount
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        ptnameId
        projectId
        ptnameIdParent
        uiLabel
        toplevelgroup
        family
        genus
        species
        author
      }
    }
  }
}
"""


def _wikidata_taxon_query(scientific_name: str) -> str:
    escaped = scientific_name.replace('"', '\\"')
    return f"""
SELECT ?taxon ?taxonLabel ?taxonLabelLang ?alias ?aliasLang ?commonName ?commonNameLang ?gbifId WHERE {{
  ?taxon wdt:P225 "{escaped}" .
  OPTIONAL {{ ?taxon wdt:P846 ?gbifId . }}
  OPTIONAL {{ ?taxon wdt:P1843 ?commonName . BIND(LANG(?commonName) AS ?commonNameLang) }}
  OPTIONAL {{ ?taxon rdfs:label ?taxonLabel . BIND(LANG(?taxonLabel) AS ?taxonLabelLang) }}
  OPTIONAL {{ ?taxon skos:altLabel ?alias . BIND(LANG(?alias) AS ?aliasLang) }}
  FILTER(!BOUND(?taxonLabelLang) || ?taxonLabelLang != "mul")
  FILTER(!BOUND(?aliasLang) || ?aliasLang != "mul")
}}
""".strip()


def _wikidata_link_is_confident(binding: dict[str, Any], context: SpeciesContext) -> bool:
    scientific_name = _binding_value(binding, "scientificName") or context.accepted_scientific_name
    if normalize_name_key(scientific_name) != normalize_name_key(context.accepted_scientific_name):
        return False
    accepted_key = str(context.accepted_taxon_key or "")
    if accepted_key.startswith("gbif:"):
        gbif_id = _binding_value(binding, "gbifId")
        return bool(gbif_id and f"gbif:{gbif_id}" == accepted_key)
    return bool(_wikidata_entity_id(_binding_value(binding, "taxon")))


def _wikidata_name_values(binding: dict[str, Any]) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for field, lang_field, source_property in (
        ("commonName", "commonNameLang", "P1843"),
        ("taxonLabel", "taxonLabelLang", "rdfs:label"),
        ("alias", "aliasLang", "skos:altLabel"),
    ):
        name = _binding_value(binding, field).strip()
        if not name:
            continue
        language = _binding_value(binding, lang_field) or _binding_language(binding, field) or "und"
        values.append((name, language, source_property))
    return values


def _binding_value(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key)
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def _binding_language(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key)
    if isinstance(value, dict):
        return str(value.get("xml:lang") or value.get("lang") or "")
    return ""


def _wikidata_entity_id(value: str) -> str:
    text = str(value or "").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


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
                try:
                    payload = response.json()
                except UnicodeDecodeError:
                    try:
                        payload = _decode_json_payload(response)
                    except json.JSONDecodeError:
                        if attempt > max_retries:
                            raise
                        wait_seconds = _backoff_seconds(attempt)
                        logger.info(
                            "registry.enrichment.http_retry base_url=%s path=%s error=json_decode attempt=%d wait_seconds=%.3f",
                            base_url,
                            path,
                            attempt,
                            wait_seconds,
                        )
                        sleep(wait_seconds)
                        continue
                except json.JSONDecodeError:
                    if attempt > max_retries:
                        raise
                    wait_seconds = _backoff_seconds(attempt)
                    logger.info(
                        "registry.enrichment.http_retry base_url=%s path=%s error=json_decode attempt=%d wait_seconds=%.3f",
                        base_url,
                        path,
                        attempt,
                        wait_seconds,
                    )
                    sleep(wait_seconds)
                    continue
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


def _graphql_post(
    url: str,
    *,
    max_retries: int = 5,
    sleep: Callable[[float], None] = default_sleep,
) -> GraphQLPost:
    client = httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = client.post(url, json=payload)
                status_code = int(getattr(response, "status_code", 200))
                if status_code in RETRYABLE_STATUS_CODES and attempt <= max_retries:
                    wait_seconds = _retry_after_seconds(response.headers.get("Retry-After")) or _backoff_seconds(attempt)
                    logger.info(
                        "registry.enrichment.http_retry url=%s status=%d attempt=%d wait_seconds=%.3f",
                        url,
                        status_code,
                        attempt,
                        wait_seconds,
                    )
                    sleep(wait_seconds)
                    continue
                response.raise_for_status()
                result = response.json()
                if isinstance(result, dict):
                    return result
                return {"data": result}
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError):
                if attempt > max_retries:
                    raise
                wait_seconds = _backoff_seconds(attempt)
                logger.info(
                    "registry.enrichment.http_retry url=%s error=graphql attempt=%d wait_seconds=%.3f",
                    url,
                    attempt,
                    wait_seconds,
                )
                sleep(wait_seconds)

    return post


def _decode_json_payload(response: Any) -> dict[str, Any] | list[Any]:
    content = getattr(response, "content", b"")
    if not content:
        raise json.JSONDecodeError("empty response content after decode failure", "", 0)
    text = bytes(content).decode("utf-8", errors="replace")
    return json.loads(text)


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


def _accepted_species_lookup(taxa_rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        normalize_name_key(row.get("scientific_name")): str(row.get("accepted_taxon_key") or "")
        for row in taxa_rows
        if str(row.get("rank") or "") == "SPECIES" and row.get("scientific_name") and row.get("accepted_taxon_key")
    }


def _unambiguous_synonym_lookup(name_rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, set[str]] = {}
    for row in name_rows:
        if str(row.get("name_class") or "") != "scientific_synonym":
            continue
        key = normalize_name_key(row.get("display_name") or row.get("verbatim_name"))
        accepted_key = str(row.get("accepted_taxon_key") or "")
        if key and accepted_key:
            grouped.setdefault(key, set()).add(accepted_key)
    return {key: next(iter(values)) for key, values in grouped.items() if len(values) == 1}


def _tmd_scientific_name(row: dict[str, Any]) -> str:
    genus = str(row.get("genus") or "").strip()
    species = str(row.get("species") or "").strip()
    if genus and species:
        return f"{genus} {species}".strip()
    return ""


def _is_tmd_complex(row: dict[str, Any]) -> bool:
    ui_label = str(row.get("uiLabel") or "")
    species = str(row.get("species") or "")
    author = str(row.get("author") or "")
    return ui_label.startswith("#") or "/" in species or "Komplex" in author


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


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
