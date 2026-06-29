from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep as default_sleep
from typing import Any, Protocol
import hashlib
import json
import logging
import random
import re
import unicodedata

import httpx
import polars as pl

from biominer.registry.normalize import normalize_language_code, normalize_name_key


logger = logging.getLogger(__name__)

USER_AGENT = "BioMiner/0.1 name-translation-harvester (+https://github.com/karikris/BioMiner)"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_SOURCES = ("gbif", "wikidata", "wikimedia", "inaturalist", "col")
DEFAULT_OUTPUT_FILE = "name_translation_assertions.parquet"
DEFAULT_LINKS_FILE = "name_translation_external_links.parquet"
DEFAULT_SNAPSHOTS_FILE = "name_translation_source_snapshots.parquet"
DEFAULT_ERRORS_FILE = "name_translation_errors.parquet"
DEFAULT_MANIFEST_FILE = "name_translation_manifest.json"
ENRICHMENT_SOURCE_ASSERTIONS_FILE = "source_name_assertions.parquet"
ENRICHMENT_EXTERNAL_LINKS_FILE = "external_taxon_links.parquet"
ENRICHMENT_SOURCE_SNAPSHOTS_FILE = "enrichment_source_snapshots.parquet"
ENRICHMENT_SOURCE_ERRORS_FILE = "source_error_records.parquet"


@dataclass(frozen=True)
class NameTranslationContext:
    """One accepted species to enrich with multilingual/local names."""

    accepted_scientific_name: str
    accepted_taxon_key: str = ""
    seed_common_names: tuple[str, ...] = ()
    gbif_usage_key: str = ""
    wikidata_qid: str = ""


@dataclass(frozen=True)
class SourceResult:
    name_assertions: tuple[dict[str, Any], ...] = ()
    external_links: tuple[dict[str, Any], ...] = ()
    source_snapshots: tuple[dict[str, Any], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    request_count: int = 0


class NameSource(Protocol):
    source_name: str

    def fetch(self, context: NameTranslationContext) -> SourceResult: ...


class JsonHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.request_count = 0
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        attempt = 0
        params = params or {}
        while True:
            attempt += 1
            self.request_count += 1
            try:
                response = self._client.get(path, params=params)
                if response.status_code in RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                    _sleep_for_retry(response, attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                if isinstance(payload, list):
                    return {"results": payload}
                return {"value": payload}
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError):
                if attempt > self.max_retries:
                    raise
                default_sleep(_backoff_seconds(attempt))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            self.request_count += 1
            try:
                response = self._client.post(path, json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                    _sleep_for_retry(response, attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {"value": data}
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError):
                if attempt > self.max_retries:
                    raise
                default_sleep(_backoff_seconds(attempt))


def harvest_species_name_translations(
    *,
    scientific_name: str,
    output_dir: str | Path,
    accepted_taxon_key: str = "",
    seed_common_names: Sequence[str] = (),
    sources: Sequence[str] = DEFAULT_SOURCES,
    target_locales: Sequence[str] = (),
    max_inaturalist_locale_probes: int = 0,
    libretranslate_url: str = "",
    libretranslate_api_key: str = "",
) -> dict[str, Any]:
    """Harvest multilingual names for one species and write parquet sidecar files.

    This is useful for testing with Papilio demoleus before running over the whole
    BioMiner registry.
    """

    context = NameTranslationContext(
        accepted_scientific_name=scientific_name.strip(),
        accepted_taxon_key=accepted_taxon_key.strip(),
        seed_common_names=tuple(_clean_text(name) for name in seed_common_names if _clean_text(name)),
    )
    adapters = build_name_sources(
        sources=sources,
        target_locales=target_locales,
        max_inaturalist_locale_probes=max_inaturalist_locale_probes,
        libretranslate_url=libretranslate_url,
        libretranslate_api_key=libretranslate_api_key,
    )
    results = [_safe_fetch(adapter, context) for adapter in adapters]
    return write_translation_results(output_dir, context=context, results=results)


def harvest_registry_name_translations(
    *,
    registry_dir: str | Path,
    output_dir: str | Path | None = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
    target_locales: Sequence[str] = (),
    limit: int = 0,
    max_inaturalist_locale_probes: int = 0,
    libretranslate_url: str = "",
    libretranslate_api_key: str = "",
) -> dict[str, Any]:
    """Harvest names for every SPECIES row in an existing BioMiner registry."""

    registry = Path(registry_dir)
    output = Path(output_dir) if output_dir is not None else registry
    taxa_path = registry / "taxa.parquet"
    names_path = registry / "names.parquet"
    if not taxa_path.exists():
        raise FileNotFoundError(f"Missing registry taxa file: {taxa_path}")

    taxa = pl.read_parquet(taxa_path)
    names_by_key: dict[str, list[str]] = {}
    if names_path.exists():
        for row in pl.read_parquet(names_path).to_dicts():
            key = str(row.get("accepted_taxon_key") or "")
            display = str(row.get("display_name") or row.get("verbatim_name") or "").strip()
            if key and display:
                names_by_key.setdefault(key, []).append(display)

    species_rows = taxa.filter(pl.col("rank") == "SPECIES").sort(["family", "genus", "scientific_name"]).to_dicts()
    if limit > 0:
        species_rows = species_rows[:limit]

    all_results: list[SourceResult] = []
    contexts: list[NameTranslationContext] = []
    adapters = build_name_sources(
        sources=sources,
        target_locales=target_locales,
        max_inaturalist_locale_probes=max_inaturalist_locale_probes,
        libretranslate_url=libretranslate_url,
        libretranslate_api_key=libretranslate_api_key,
    )
    for row in species_rows:
        key = str(row.get("accepted_taxon_key") or "")
        context = NameTranslationContext(
            accepted_scientific_name=str(row.get("scientific_name") or ""),
            accepted_taxon_key=key,
            seed_common_names=tuple(names_by_key.get(key, ())),
        )
        contexts.append(context)
        for adapter in adapters:
            all_results.append(_safe_fetch(adapter, context))

    return write_translation_results(output, contexts=contexts, results=all_results)


def build_name_sources(
    *,
    sources: Sequence[str],
    target_locales: Sequence[str] = (),
    max_inaturalist_locale_probes: int = 0,
    libretranslate_url: str = "",
    libretranslate_api_key: str = "",
) -> tuple[NameSource, ...]:
    registry: dict[str, NameSource] = {
        "gbif": GBIFVernacularNameSource(),
        "wikidata": WikidataTaxonNameSource(),
        "wikimedia": WikimediaLanglinksNameSource(),
        "inaturalist": INaturalistNameSource(
            target_locales=tuple(target_locales),
            max_locale_probes=max_inaturalist_locale_probes,
        ),
        "col": CatalogueOfLifeNameSource(),
    }
    if libretranslate_url:
        registry["libretranslate"] = LibreTranslateNameSource(
            target_locales=tuple(target_locales),
            base_url=libretranslate_url,
            api_key=libretranslate_api_key,
        )
    selected: list[NameSource] = []
    for source in sources:
        key = source.strip().casefold()
        if not key:
            continue
        if key not in registry:
            raise ValueError(f"Unknown name translation source {source!r}; known sources: {sorted(registry)}")
        selected.append(registry[key])
    return tuple(selected)


class GBIFVernacularNameSource:
    source_name = "GBIF"

    def __init__(self, *, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient("https://api.gbif.org")

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        accepted_key = _strip_prefix(context.gbif_usage_key or context.accepted_taxon_key, "gbif:")
        links: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        if not accepted_key:
            match = self.client.get(
                "/v1/species/match",
                {"name": context.accepted_scientific_name, "rank": "SPECIES", "strict": "false"},
            )
            accepted_key = str(match.get("acceptedUsageKey") or match.get("usageKey") or match.get("key") or "")
            if accepted_key:
                links.append(
                    _external_link(
                        context,
                        source=self.source_name,
                        source_taxon_id=accepted_key,
                        match_method="scientific_name",
                        match_confidence=str(match.get("confidence") or ""),
                    )
                )
                snapshots.append(_snapshot(self.source_name, "species/match", "/v1/species/match", match, licence="various"))
        if not accepted_key:
            return SourceResult(errors=(_source_error(context, self.source_name, "unresolved_taxon", endpoint="/v1/species/match"),))

        rows, payloads = _get_paginated(self.client, f"/v1/species/{accepted_key}/vernacularNames", {"limit": 1000}, limit=1000)
        assertions = []
        for index, row in enumerate(rows):
            name = _first_string(row, "vernacularName", "name", "commonName")
            if not name or _is_scientific_name_like(name, context.accepted_scientific_name):
                continue
            language = _first_string(row, "language", "languageCode", "lang")
            country = _first_string(row, "country", "countryCode")
            source_record_id = _first_string(row, "sourceTaxonKey", "taxonKey", "nubKey") or f"{accepted_key}:{index}"
            assertions.append(
                _name_assertion(
                    context,
                    name,
                    source=self.source_name,
                    source_record_id=f"gbif:{accepted_key}:vernacular:{source_record_id}:{name}",
                    source_taxon_id=accepted_key,
                    language=language,
                    region=country,
                    trust_tier="T2",
                    precision_tier="high" if language else "medium",
                    confidence="high",
                    licence="various-by-dataset",
                )
            )
        if payloads:
            snapshots.append(_snapshot(self.source_name, "species-vernacularNames", f"/v1/species/{accepted_key}/vernacularNames", payloads, licence="various-by-dataset"))
        return SourceResult(
            name_assertions=tuple(assertions),
            external_links=tuple(links),
            source_snapshots=tuple(snapshots),
            request_count=self.client.request_count - started,
        )


class CatalogueOfLifeNameSource:
    source_name = "CoL"

    def __init__(self, *, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient("https://api.checklistbank.org")

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        payload = self.client.get("/dataset/3/nameusage/search", {"q": context.accepted_scientific_name, "limit": 25})
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for row in _result_rows(payload):
            scientific = _first_string(row, "scientificName", "name", "canonicalName")
            if scientific and _normal_binomial(scientific) != _normal_binomial(context.accepted_scientific_name):
                continue
            source_taxon_id = _first_string(row, "id", "key", "usageKey")
            if source_taxon_id:
                links.append(_external_link(context, source=self.source_name, source_taxon_id=source_taxon_id, match_method="scientific_name"))
            for index, value in enumerate(_vernacular_values(row)):
                assertions.append(
                    _name_assertion(
                        context,
                        value,
                        source=self.source_name,
                        source_record_id=f"col:{source_taxon_id}:vernacular:{index}:{value}",
                        source_taxon_id=source_taxon_id,
                        language=_vernacular_language(row, value),
                        trust_tier="T2",
                        precision_tier="medium",
                        confidence="medium",
                        licence="various-by-dataset",
                    )
                )
        return SourceResult(
            name_assertions=tuple(assertions),
            external_links=tuple(links),
            source_snapshots=(_snapshot(self.source_name, "checklistbank-dataset-3", "/dataset/3/nameusage/search", payload, licence="various-by-dataset"),),
            request_count=self.client.request_count - started,
        )


class WikidataTaxonNameSource:
    source_name = "Wikidata"

    def __init__(self, *, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient("https://query.wikidata.org")

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        common_payload = self.client.get("/sparql", {"query": _wikidata_common_names_query(context.accepted_scientific_name), "format": "json"})
        label_payload = self.client.get("/sparql", {"query": _wikidata_labels_query(context.accepted_scientific_name), "format": "json"})
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        qids: set[str] = set()

        for binding in _sparql_bindings(common_payload):
            qid = _wikidata_qid(_binding_value(binding, "taxon"))
            if qid:
                qids.add(qid)
            value = _binding_value(binding, "name")
            language = _binding_value(binding, "lang") or _literal_lang(binding, "name")
            if not value or _is_scientific_name_like(value, context.accepted_scientific_name):
                continue
            assertions.append(
                _name_assertion(
                    context,
                    value,
                    source=self.source_name,
                    source_record_id=f"wikidata:{qid}:P1843:{language}:{value}",
                    source_taxon_id=qid,
                    language=language,
                    name_class="vernacular",
                    trust_tier="T2",
                    precision_tier="high" if language else "medium",
                    confidence="high",
                    licence="CC0",
                )
            )

        for binding in _sparql_bindings(label_payload):
            qid = _wikidata_qid(_binding_value(binding, "taxon"))
            if qid:
                qids.add(qid)
            value = _binding_value(binding, "name")
            language = _binding_value(binding, "lang") or _literal_lang(binding, "name")
            kind = _binding_value(binding, "kind") or "label"
            if not value or _is_scientific_name_like(value, context.accepted_scientific_name):
                continue
            assertions.append(
                _name_assertion(
                    context,
                    value,
                    source=self.source_name,
                    source_record_id=f"wikidata:{qid}:{kind}:{language}:{value}",
                    source_taxon_id=qid,
                    language=language,
                    name_class=f"wikidata_{kind}",
                    trust_tier="T3",
                    precision_tier="medium",
                    confidence="medium",
                    licence="CC0",
                )
            )

        for qid in sorted(qids):
            links.append(_external_link(context, source=self.source_name, source_taxon_id=qid, match_method="P225", match_confidence="high"))
        return SourceResult(
            name_assertions=tuple(assertions),
            external_links=tuple(links),
            source_snapshots=(
                _snapshot(self.source_name, "WDQS-P1843", "/sparql", common_payload, licence="CC0"),
                _snapshot(self.source_name, "WDQS-labels-aliases", "/sparql", label_payload, licence="CC0"),
            ),
            request_count=self.client.request_count - started,
        )


class WikimediaLanglinksNameSource:
    source_name = "Wikimedia"

    def __init__(self, *, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient("https://en.wikipedia.org")

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        title = _find_enwiki_title(self.client, context.accepted_scientific_name)
        if not title:
            return SourceResult(errors=(_source_error(context, self.source_name, "missing_enwiki_page", endpoint="/w/api.php"),))

        params: dict[str, object] = {
            "action": "query",
            "titles": title,
            "redirects": "1",
            "prop": "langlinks",
            "lllimit": "max",
            "llprop": "url|langname|autonym",
            "format": "json",
        }
        payloads: list[dict[str, Any]] = []
        langlinks: list[dict[str, Any]] = []
        while True:
            payload = self.client.get("/w/api.php", params)
            payloads.append(payload)
            langlinks.extend(_mediawiki_langlinks(payload))
            cont = payload.get("continue")
            if not isinstance(cont, dict) or not cont.get("llcontinue"):
                break
            params["llcontinue"] = str(cont["llcontinue"])
            params["continue"] = str(cont.get("continue") or "")

        assertions: list[dict[str, Any]] = []
        for item in langlinks:
            language = _first_string(item, "lang")
            value = _first_string(item, "*", "title")
            if not value or _is_scientific_name_like(value, context.accepted_scientific_name):
                continue
            assertions.append(
                _name_assertion(
                    context,
                    value,
                    source=self.source_name,
                    source_record_id=f"wikimedia:enwiki:{title}:langlink:{language}:{value}",
                    source_taxon_id=title,
                    language=language,
                    name_class="article_title",
                    trust_tier="T3",
                    precision_tier="medium",
                    confidence="medium",
                    licence="CC BY-SA / Wikimedia project terms",
                )
            )
        return SourceResult(
            name_assertions=tuple(assertions),
            external_links=(_external_link(context, source=self.source_name, source_taxon_id=title, match_method="enwiki_title", match_confidence="medium"),),
            source_snapshots=(_snapshot(self.source_name, "MediaWiki-langlinks", "/w/api.php?action=query&prop=langlinks", payloads, licence="CC BY-SA / Wikimedia project terms"),),
            request_count=self.client.request_count - started,
        )


class INaturalistNameSource:
    source_name = "iNaturalist"

    def __init__(
        self,
        *,
        client: JsonHttpClient | None = None,
        target_locales: tuple[str, ...] = (),
        max_locale_probes: int = 0,
    ) -> None:
        self.client = client or JsonHttpClient("https://api.inaturalist.org")
        self.target_locales = target_locales
        self.max_locale_probes = max_locale_probes

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        payload = self.client.get("/v1/taxa", {"q": context.accepted_scientific_name, "rank": "species", "per_page": 10})
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        taxon_id = ""
        for row in _result_rows(payload):
            scientific = _first_string(row, "name", "scientific_name")
            if scientific and _normal_binomial(scientific) != _normal_binomial(context.accepted_scientific_name):
                continue
            taxon_id = _first_string(row, "id")
            if taxon_id:
                links.append(_external_link(context, source=self.source_name, source_taxon_id=taxon_id, match_method="scientific_name"))
            assertions.extend(self._names_from_taxon_row(context, row, taxon_id=taxon_id, locale=""))
            break

        locale_payloads: list[dict[str, Any]] = []
        if taxon_id and self.target_locales and self.max_locale_probes:
            for locale in tuple(self.target_locales)[: self.max_locale_probes]:
                locale_payload = self.client.get(f"/v1/taxa/{taxon_id}", {"locale": locale})
                locale_payloads.append(locale_payload)
                for row in _result_rows(locale_payload) or ([locale_payload] if locale_payload else []):
                    assertions.extend(self._names_from_taxon_row(context, row, taxon_id=taxon_id, locale=locale))

        return SourceResult(
            name_assertions=tuple(assertions),
            external_links=tuple(links),
            source_snapshots=(
                _snapshot(self.source_name, "inaturalist-v1-taxa", "/v1/taxa", payload, licence="various"),
                *([] if not locale_payloads else [_snapshot(self.source_name, "inaturalist-v1-taxa-locale-probes", "/v1/taxa/{id}", locale_payloads, licence="various")]),
            ),
            request_count=self.client.request_count - started,
        )

    def _names_from_taxon_row(self, context: NameTranslationContext, row: dict[str, Any], *, taxon_id: str, locale: str) -> list[dict[str, Any]]:
        assertions: list[dict[str, Any]] = []
        preferred = _first_string(row, "preferred_common_name", "english_common_name")
        if preferred:
            assertions.append(
                _name_assertion(
                    context,
                    preferred,
                    source=self.source_name,
                    source_record_id=f"inaturalist:{taxon_id}:preferred:{locale or 'default'}:{preferred}",
                    source_taxon_id=taxon_id,
                    language=locale or "eng",
                    name_class="preferred_common_name",
                    trust_tier="T2",
                    precision_tier="medium",
                    confidence="high",
                    licence="various",
                )
            )
        for collection_key in ("names", "taxon_names", "common_names"):
            for index, item in enumerate(_list_values(row, collection_key)):
                if isinstance(item, dict):
                    value = _first_string(item, "name", "common_name", "lexicon")
                    language = _first_string(item, "locale", "language", "lexicon") or locale
                else:
                    value = str(item or "")
                    language = locale
                if not value or _is_scientific_name_like(value, context.accepted_scientific_name):
                    continue
                assertions.append(
                    _name_assertion(
                        context,
                        value,
                        source=self.source_name,
                        source_record_id=f"inaturalist:{taxon_id}:{collection_key}:{index}:{language}:{value}",
                        source_taxon_id=taxon_id,
                        language=language,
                        trust_tier="T4",
                        precision_tier="medium",
                        confidence="medium",
                        licence="various",
                    )
                )
        return assertions


class LibreTranslateNameSource:
    """Optional machine-translation fallback.

    These records are intentionally disabled and marked needs_review. Use this
    only to create broad search keywords after source-backed names run out.
    """

    source_name = "LibreTranslate"

    def __init__(self, *, target_locales: tuple[str, ...], base_url: str, api_key: str = "") -> None:
        self.client = JsonHttpClient(base_url)
        self.target_locales = target_locales
        self.api_key = api_key

    def fetch(self, context: NameTranslationContext) -> SourceResult:
        started = self.client.request_count
        if not context.seed_common_names or not self.target_locales:
            return SourceResult()
        assertions: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for seed in context.seed_common_names:
            for target in self.target_locales:
                payload: dict[str, Any] = {"q": seed, "source": "en", "target": _two_letter_lang(target), "format": "text"}
                if self.api_key:
                    payload["api_key"] = self.api_key
                try:
                    translated = self.client.post_json("/translate", payload)
                except Exception as exc:  # noqa: BLE001 - unsupported target languages should not stop the batch.
                    errors.append(_source_error(context, self.source_name, f"{type(exc).__name__}:{target}", endpoint="/translate"))
                    continue
                value = _first_string(translated, "translatedText")
                if not value or _is_scientific_name_like(value, context.accepted_scientific_name):
                    continue
                assertions.append(
                    _name_assertion(
                        context,
                        value,
                        source=self.source_name,
                        source_record_id=f"libretranslate:{seed}:{target}:{value}",
                        source_taxon_id="",
                        language=target,
                        name_class="machine_translation_candidate",
                        trust_tier="T5",
                        precision_tier="synthetic",
                        confidence="low",
                        enabled=False,
                        review_state="needs_review",
                        disabled_reason="machine_translation_not_source_backed",
                        licence="generated-candidate-review-required",
                    )
                )
        return SourceResult(
            name_assertions=tuple(assertions),
            source_snapshots=(_snapshot(self.source_name, "libretranslate", "/translate", {"targets": self.target_locales}, licence="generated-candidate-review-required"),),
            errors=tuple(errors),
            request_count=self.client.request_count - started,
        )


def write_translation_results(
    output_dir: str | Path,
    *,
    results: Sequence[SourceResult],
    context: NameTranslationContext | None = None,
    contexts: Sequence[NameTranslationContext] = (),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assertions = _dedupe_rows([row for result in results for row in result.name_assertions], key_fields=("accepted_taxon_key", "display_name", "language", "source", "source_record_id"))
    links = _dedupe_rows([row for result in results for row in result.external_links], key_fields=("accepted_taxon_key", "source", "source_taxon_id", "match_method"))
    snapshots = _dedupe_rows([row for result in results for row in result.source_snapshots], key_fields=("source", "source_version", "source_path", "source_response_hash"))
    errors = [row for result in results for row in result.errors]

    assertion_frame = _frame(assertions, _name_assertion_schema())
    links_frame = _frame(links, _external_link_schema())
    snapshots_frame = _frame(snapshots, _source_snapshot_schema())
    errors_frame = _frame(errors, _source_error_schema())
    assertion_frame.write_parquet(output / DEFAULT_OUTPUT_FILE)
    links_frame.write_parquet(output / DEFAULT_LINKS_FILE)
    snapshots_frame.write_parquet(output / DEFAULT_SNAPSHOTS_FILE)
    errors_frame.write_parquet(output / DEFAULT_ERRORS_FILE)

    source_counts = assertion_frame.group_by("source").len().sort("source").to_dicts() if assertion_frame.height else []
    language_counts = assertion_frame.group_by("language").len().sort("len", descending=True).to_dicts() if assertion_frame.height else []
    manifest = {
        "schema_version": "name-translation-harvest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_name": context.accepted_scientific_name if context else "",
        "species_count": len(contexts) if contexts else (1 if context else 0),
        "name_assertion_rows": assertion_frame.height,
        "external_link_rows": links_frame.height,
        "source_snapshot_rows": snapshots_frame.height,
        "source_error_rows": errors_frame.height,
        "request_count": sum(result.request_count for result in results),
        "source_counts": source_counts,
        "top_language_counts": language_counts[:50],
        "files": {
            "name_translation_assertions": DEFAULT_OUTPUT_FILE,
            "name_translation_external_links": DEFAULT_LINKS_FILE,
            "name_translation_source_snapshots": DEFAULT_SNAPSHOTS_FILE,
            "name_translation_errors": DEFAULT_ERRORS_FILE,
        },
    }
    (output / DEFAULT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def merge_name_translation_sidecar_into_enrichment(output_dir: str | Path) -> dict[str, Any]:
    """Append harvested translation sidecars into BioMiner's enrichment parquet files.

    Run this after harvest when you want compile-enriched to see these records.
    The sidecar files remain in place for auditability.
    """

    output = Path(output_dir)
    translations = _read_or_empty(output / DEFAULT_OUTPUT_FILE, _name_assertion_schema())
    translation_links = _read_or_empty(output / DEFAULT_LINKS_FILE, _external_link_schema())
    translation_snapshots = _read_or_empty(output / DEFAULT_SNAPSHOTS_FILE, _source_snapshot_schema())
    translation_errors = _read_or_empty(output / DEFAULT_ERRORS_FILE, _source_error_schema())

    merged_assertions = _merge_frames(
        _read_or_empty(output / ENRICHMENT_SOURCE_ASSERTIONS_FILE, _name_assertion_schema()),
        translations,
        subset=["assertion_id"],
    )
    merged_links = _merge_frames(
        _read_or_empty(output / ENRICHMENT_EXTERNAL_LINKS_FILE, _external_link_schema()),
        translation_links,
        subset=["accepted_taxon_key", "source", "source_taxon_id", "match_method"],
    )
    merged_snapshots = _merge_frames(
        _read_or_empty(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema()),
        translation_snapshots,
        subset=["source", "source_version", "source_path", "source_response_hash"],
    )
    merged_errors = _merge_frames(
        _read_or_empty(output / ENRICHMENT_SOURCE_ERRORS_FILE, _source_error_schema()),
        translation_errors,
        subset=["accepted_taxon_key", "source", "endpoint", "error_class"],
    )

    merged_assertions.write_parquet(output / ENRICHMENT_SOURCE_ASSERTIONS_FILE)
    merged_links.write_parquet(output / ENRICHMENT_EXTERNAL_LINKS_FILE)
    merged_snapshots.write_parquet(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
    merged_errors.write_parquet(output / ENRICHMENT_SOURCE_ERRORS_FILE)
    manifest = {
        "merged_at": datetime.now(UTC).isoformat(),
        "translation_assertion_rows": translations.height,
        "merged_source_name_assertion_rows": merged_assertions.height,
        "merged_external_link_rows": merged_links.height,
        "merged_source_snapshot_rows": merged_snapshots.height,
        "merged_source_error_rows": merged_errors.height,
        "files": {
            "source_name_assertions": ENRICHMENT_SOURCE_ASSERTIONS_FILE,
            "external_taxon_links": ENRICHMENT_EXTERNAL_LINKS_FILE,
            "enrichment_source_snapshots": ENRICHMENT_SOURCE_SNAPSHOTS_FILE,
            "source_error_records": ENRICHMENT_SOURCE_ERRORS_FILE,
        },
    }
    (output / "name_translation_merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _safe_fetch(adapter: NameSource, context: NameTranslationContext) -> SourceResult:
    try:
        return adapter.fetch(context)
    except Exception as exc:  # noqa: BLE001 - source harvesting should not fail the whole species batch.
        logger.info("name_translation.source_error source=%s species=%s error=%s", adapter.source_name, context.accepted_scientific_name, type(exc).__name__)
        return SourceResult(errors=(_source_error(context, adapter.source_name, type(exc).__name__),))


def _name_assertion(
    context: NameTranslationContext,
    name: str,
    *,
    source: str,
    source_record_id: str,
    source_taxon_id: str = "",
    language: str = "",
    region: str = "",
    name_class: str = "vernacular",
    trust_tier: str = "T3",
    precision_tier: str = "medium",
    confidence: str = "medium",
    enabled: bool = True,
    review_state: str = "accepted",
    disabled_reason: str = "",
    licence: str = "",
) -> dict[str, Any]:
    display_name = _clean_text(name)
    return {
        "assertion_id": _stable_id(context.accepted_taxon_key, context.accepted_scientific_name, source, source_record_id, display_name),
        "accepted_taxon_key": context.accepted_taxon_key,
        "verbatim_name": name,
        "display_name": display_name,
        "normalized_match_key": normalize_name_key(display_name),
        "language": normalize_language_code(language) or _language_from_text(display_name),
        "script": _guess_script(display_name),
        "region": region,
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
        "retrieved_at": datetime.now(UTC).isoformat(),
        "licence": licence,
    }


def _external_link(
    context: NameTranslationContext,
    *,
    source: str,
    source_taxon_id: str,
    match_method: str,
    match_confidence: str = "high",
) -> dict[str, Any]:
    return {
        "accepted_taxon_key": context.accepted_taxon_key,
        "source": source,
        "source_taxon_id": source_taxon_id,
        "match_method": match_method,
        "match_confidence": match_confidence,
        "lineage_check": "scientific_name",
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def _snapshot(source: str, source_version: str, source_path: str, payload: Any, *, licence: str = "") -> dict[str, Any]:
    return {
        "source": source,
        "source_version": source_version,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "source_response_hash": _payload_hash(payload),
        "licence": licence,
    }


def _source_error(context: NameTranslationContext, source: str, error_class: str, *, endpoint: str = "") -> dict[str, Any]:
    return {
        "accepted_taxon_key": context.accepted_taxon_key,
        "source": source,
        "endpoint": endpoint,
        "error_class": error_class,
        "attempts": 1,
        "retryable": "false",
        "disposition": "quarantined",
    }


def _wikidata_common_names_query(scientific_name: str) -> str:
    escaped = _sparql_string(scientific_name)
    return f"""
SELECT ?taxon ?name ?lang WHERE {{
  ?taxon wdt:P225 {escaped} .
  ?taxon wdt:P1843 ?name .
  BIND(LANG(?name) AS ?lang)
}}
"""


def _wikidata_labels_query(scientific_name: str) -> str:
    escaped = _sparql_string(scientific_name)
    return f"""
SELECT ?taxon ?kind ?name ?lang WHERE {{
  ?taxon wdt:P225 {escaped} .
  {{ ?taxon rdfs:label ?name . BIND("label" AS ?kind) }}
  UNION
  {{ ?taxon skos:altLabel ?name . BIND("alias" AS ?kind) }}
  BIND(LANG(?name) AS ?lang)
}}
"""


def _find_enwiki_title(client: JsonHttpClient, scientific_name: str) -> str:
    payload = client.get(
        "/w/api.php",
        {
            "action": "query",
            "titles": scientific_name,
            "redirects": "1",
            "format": "json",
        },
    )
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload.get("query"), dict) else {}
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "missing" not in page:
                return str(page.get("title") or scientific_name)

    search_payload = client.get(
        "/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": f'"{scientific_name}"',
            "srlimit": 5,
            "format": "json",
        },
    )
    for row in _list_values(search_payload.get("query", {}) if isinstance(search_payload.get("query"), dict) else {}, "search"):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        if title.casefold() == scientific_name.casefold():
            return title
    return ""


def _mediawiki_langlinks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload.get("query"), dict) else {}
    rows: list[dict[str, Any]] = []
    if not isinstance(pages, dict):
        return rows
    for page in pages.values():
        if isinstance(page, dict):
            links = page.get("langlinks", [])
            if isinstance(links, list):
                rows.extend(link for link in links if isinstance(link, dict))
    return rows


def _get_paginated(client: JsonHttpClient, path: str, params: dict[str, object], *, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_params = dict(params)
        if offset:
            page_params["offset"] = offset
        payload = client.get(path, page_params)
        payloads.append(payload)
        page_rows = _result_rows(payload)
        rows.extend(page_rows)
        if payload.get("endOfRecords") is True or not page_rows:
            break
        count = payload.get("count")
        if isinstance(count, int) and offset + len(page_rows) >= count:
            break
        if len(page_rows) < limit:
            break
        offset += len(page_rows)
    return rows, payloads


def _result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _sparql_bindings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return []
    bindings = results.get("bindings", [])
    return [row for row in bindings if isinstance(row, dict)] if isinstance(bindings, list) else []


def _binding_value(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key)
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return ""


def _literal_lang(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key)
    if isinstance(value, dict):
        return str(value.get("xml:lang") or "")
    return ""


def _wikidata_qid(uri: str) -> str:
    if not uri:
        return ""
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _vernacular_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("vernacularName", "vernacular", "commonName"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for key in ("vernacularNames", "vernaculars", "commonNames"):
        for item in _list_values(row, key):
            if isinstance(item, dict):
                value = _first_string(item, "name", "vernacularName", "commonName")
            else:
                value = str(item or "")
            if value.strip():
                values.append(value.strip())
    return sorted(set(values), key=values.index)


def _vernacular_language(row: dict[str, Any], value: str) -> str:
    for key in ("language", "lang", "languageCode"):
        language = _first_string(row, key)
        if language:
            return language
    for collection_key in ("vernacularNames", "vernaculars", "commonNames"):
        for item in _list_values(row, collection_key):
            if not isinstance(item, dict):
                continue
            candidate = _first_string(item, "name", "vernacularName", "commonName")
            if candidate == value:
                return _first_string(item, "language", "lang", "languageCode")
    return ""


def _first_string(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _list_values(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if isinstance(value, list):
        return value
    return []


def _strip_prefix(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    return text.removeprefix(prefix)


def _clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    return re.sub(r"\s+", " ", text)


def _normal_binomial(value: str) -> str:
    parts = re.findall(r"[A-Za-z][A-Za-z-]+", value)
    return " ".join(parts[:2]).casefold()


def _is_scientific_name_like(value: str, scientific_name: str) -> bool:
    cleaned = _clean_text(value).casefold()
    sci = _clean_text(scientific_name).casefold()
    if not cleaned:
        return True
    if cleaned == sci:
        return True
    if cleaned.replace("_", " ") == sci:
        return True
    return False


def _language_from_text(value: str) -> str:
    script = _guess_script(value)
    return {"Hani": "zho", "Jpan": "jpn", "Kore": "kor", "Thai": "tha", "Arab": "ara", "Cyrl": "rus"}.get(script, "")


def _guess_script(value: str) -> str:
    text = _clean_text(value)
    ranges = [
        ("Hani", 0x4E00, 0x9FFF),
        ("Jpan", 0x3040, 0x30FF),
        ("Kore", 0xAC00, 0xD7AF),
        ("Cyrl", 0x0400, 0x04FF),
        ("Arab", 0x0600, 0x06FF),
        ("Deva", 0x0900, 0x097F),
        ("Beng", 0x0980, 0x09FF),
        ("Guru", 0x0A00, 0x0A7F),
        ("Gujr", 0x0A80, 0x0AFF),
        ("Orya", 0x0B00, 0x0B7F),
        ("Taml", 0x0B80, 0x0BFF),
        ("Telu", 0x0C00, 0x0C7F),
        ("Knda", 0x0C80, 0x0CFF),
        ("Mlym", 0x0D00, 0x0D7F),
        ("Sinh", 0x0D80, 0x0DFF),
        ("Thai", 0x0E00, 0x0E7F),
        ("Laoo", 0x0E80, 0x0EFF),
        ("Mymr", 0x1000, 0x109F),
        ("Ethi", 0x1200, 0x137F),
        ("Khmr", 0x1780, 0x17FF),
        ("Grek", 0x0370, 0x03FF),
        ("Hebr", 0x0590, 0x05FF),
    ]
    for char in text:
        code = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z" or char in "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ":
            return "Latn"
        for script, start, end in ranges:
            if start <= code <= end:
                return script
    return "Zyyy"


def _dedupe_rows(rows: Iterable[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(normalize_name_key(row.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    normalized = [{column: row.get(column, _default_for_dtype(dtype)) for column, dtype in schema.items()} for row in rows]
    return pl.DataFrame(normalized, schema=schema) if normalized else pl.DataFrame(schema=schema)


def _default_for_dtype(dtype: pl.DataType) -> object:
    if dtype == pl.Boolean:
        return False
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return 0
    return ""


def _read_or_empty(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=schema)


def _merge_frames(left: pl.DataFrame, right: pl.DataFrame, *, subset: list[str]) -> pl.DataFrame:
    if left.height == 0:
        return right.unique(subset=subset, keep="last") if right.height else right
    if right.height == 0:
        return left.unique(subset=subset, keep="last") if left.height else left
    return pl.concat([left, right], how="diagonal_relaxed").unique(subset=subset, keep="last")


def _name_assertion_schema() -> dict[str, pl.DataType]:
    return {
        "assertion_id": pl.String,
        "accepted_taxon_key": pl.String,
        "verbatim_name": pl.String,
        "display_name": pl.String,
        "normalized_match_key": pl.String,
        "language": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "source_taxon_id": pl.String,
        "trust_tier": pl.String,
        "precision_tier": pl.String,
        "confidence": pl.String,
        "enabled": pl.Boolean,
        "review_state": pl.String,
        "disabled_reason": pl.String,
        "retrieved_at": pl.String,
        "licence": pl.String,
    }


def _external_link_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "source": pl.String,
        "source_taxon_id": pl.String,
        "match_method": pl.String,
        "match_confidence": pl.String,
        "lineage_check": pl.String,
        "retrieved_at": pl.String,
    }


def _source_snapshot_schema() -> dict[str, pl.DataType]:
    return {
        "source": pl.String,
        "source_version": pl.String,
        "retrieved_at": pl.String,
        "source_path": pl.String,
        "source_response_hash": pl.String,
        "licence": pl.String,
    }


def _source_error_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "source": pl.String,
        "endpoint": pl.String,
        "error_class": pl.String,
        "attempts": pl.Int64,
        "retryable": pl.String,
        "disposition": pl.String,
    }


def _stable_id(*parts: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps([str(part or "") for part in parts], ensure_ascii=False).encode("utf-8")).hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sparql_string(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _two_letter_lang(value: str) -> str:
    text = str(value or "").replace("_", "-").strip().casefold()
    return text.split("-", 1)[0]


def _sleep_for_retry(response: httpx.Response, attempt: int) -> None:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            default_sleep(float(retry_after))
            return
        except ValueError:
            pass
    default_sleep(_backoff_seconds(attempt))


def _backoff_seconds(attempt: int) -> float:
    return min(60.0, (2 ** max(0, attempt - 1)) + random.random())
