from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import polars as pl

from biominer.registry.enrichment import (
    ENRICHMENT_MANIFEST_FILE,
    ENRICHMENT_SOURCE_SNAPSHOTS_FILE,
    EXTERNAL_LINKS_FILE,
    SOURCE_ASSERTIONS_FILE,
    SOURCE_ERRORS_FILE,
    SOURCE_WORK_LEDGER_FILE,
    _deduplicate_dicts,
    _deduplicate_latest_dicts,
    _name_assertion_schema,
    _name_assertions_frame,
    _source_error_schema,
    _source_errors_frame,
    _source_snapshot_schema,
    _source_snapshots_frame,
    _source_work_frame,
    _source_work_schema,
)
from biominer.registry.enrichment_sources import HTTPGet, USER_AGENT, _json_get
from biominer.registry.normalize import parse_language_tag, normalize_language_code, normalize_name_key
from biominer.registry.query_eligibility import assess_name_query_eligibility
from biominer.registry.translation_sources import (
    DEFAULT_TRANSLATION_SOURCES,
    DEFAULT_TRANSLATION_TARGET_LOCALES_JSON,
    TRANSLATION_CANDIDATES_FILE,
    generated_translation_candidate,
    translation_candidate_schema,
    translation_candidates_frame,
    translation_source_display_names,
)


logger = logging.getLogger(__name__)

TRANSLATION_WORK_LEDGER_FILE = "translation_work_ledger.parquet"
DEFAULT_TRANSLATION_TARGET_LOCALES = (
    "de",
    "fr",
    "es",
    "pt",
    "it",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "ru",
    "ja",
    "zh",
    "ko",
    "id",
    "ms",
    "th",
    "vi",
    "hi",
)
MYMEMORY_SOURCE_VERSION = "mymemory-get-v1"
WIKIMEDIA_SOURCE_VERSION = "mediawiki-langlinks-pageprops-v2"
MYMEMORY_BASE_URL = "https://api.mymemory.translated.net"
WIKIPEDIA_API_BASE_URL = "https://en.wikipedia.org"
COMMON_NAME_CLASSES = {"vernacular", "vernacular_alias", "common_name", "common_name_alias"}
SCIENTIFIC_NAME_CLASSES = {"accepted_scientific", "scientific", "scientific_name", "scientific_synonym", "synonym"}
ENGLISH_LANGUAGE_CODES = {"eng", "en", "english"}
API_LANGUAGE_CODES = {
    "ara": "ar",
    "ces": "cs",
    "dan": "da",
    "deu": "de",
    "eng": "en",
    "fin": "fi",
    "fra": "fr",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "nld": "nl",
    "nor": "no",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "spa": "es",
    "swe": "sv",
    "zho": "zh",
}


@dataclass(frozen=True)
class TranslationSeed:
    accepted_taxon_key: str
    accepted_scientific_name: str
    source_name: str
    source_language: str
    source: str
    name_class: str
    trust_tier: str


@dataclass(frozen=True)
class SpeciesTranslationContext:
    accepted_taxon_key: str
    accepted_scientific_name: str


@dataclass(frozen=True)
class WikimediaLanglink:
    language: str
    title: str
    page_id: str
    page_title: str
    wikidata_item: str = ""


@dataclass(frozen=True)
class TranslationWorkRecord:
    source: str
    accepted_taxon_key: str
    accepted_scientific_name: str
    source_name: str
    source_language: str
    target_language: str
    work_key: str
    provider_config_hash: str
    status: str
    attempts: int
    started_at: str
    finished_at: str
    request_count: int
    error_class: str = ""
    retryable: str = "false"
    request_day: str = ""

    def to_row(self) -> dict[str, object]:
        row = self.__dict__.copy()
        row["request_day"] = self.request_day or datetime.now(UTC).date().isoformat()
        return row


class TranslationRequestBudget:
    def __init__(self, *, daily_limit: int, existing_work: list[dict[str, Any]]) -> None:
        self.daily_limit = daily_limit
        self.day = datetime.now(UTC).date().isoformat()
        self.used = sum(
            int(row.get("request_count") or 0)
            for row in existing_work
            if str(row.get("request_day") or "") == self.day
        )
        self.exhausted = False

    def reserve(self, count: int = 1) -> bool:
        if self.daily_limit <= 0:
            return True
        if self.used + count > self.daily_limit:
            self.exhausted = True
            return False
        self.used += count
        return True


class WikimediaLanglinksProvider:
    source_key = "wikimedia"
    source_name = "Wikimedia"

    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get(WIKIPEDIA_API_BASE_URL, max_retries=max_retries)

    def langlinks(self, title: str, *, target_locales: tuple[str, ...]) -> tuple[list[WikimediaLanglink], int, str]:
        request_count = 0
        params: dict[str, object] = {
            "action": "query",
            "format": "json",
            "prop": "langlinks|pageprops",
            "ppprop": "wikibase_item",
            "titles": title,
            "lllimit": "max",
            "redirects": "1",
        }
        links: list[WikimediaLanglink] = []
        page_title = title
        page_id = ""
        target_api_codes = {_api_language_code(locale) for locale in target_locales}
        while True:
            payload = self._http_get("/w/api.php", params)
            request_count += 1
            pages = payload.get("query", {}).get("pages", {})
            if isinstance(pages, dict):
                for page_key, page in pages.items():
                    if not isinstance(page, dict) or "missing" in page:
                        continue
                    page_id = str(page.get("pageid") or page_key or "")
                    page_title = str(page.get("title") or page_title)
                    pageprops = page.get("pageprops") if isinstance(page.get("pageprops"), dict) else {}
                    wikidata_item = str(pageprops.get("wikibase_item") or "") if isinstance(pageprops, dict) else ""
                    for item in page.get("langlinks") or []:
                        if not isinstance(item, dict):
                            continue
                        language = str(item.get("lang") or "")
                        linked_title = str(item.get("*") or item.get("title") or "")
                        if language in target_api_codes and linked_title:
                            links.append(
                                WikimediaLanglink(
                                    language=language,
                                    title=linked_title,
                                    page_id=page_id,
                                    page_title=page_title,
                                    wikidata_item=wikidata_item,
                                )
                            )
            continuation = payload.get("continue")
            if not isinstance(continuation, dict) or not continuation.get("llcontinue"):
                break
            params = {**params, **continuation}
        return links, request_count, page_title


class MyMemoryTranslationProvider:
    source_key = "mymemory"
    source_name = "MyMemory"

    def __init__(
        self,
        *,
        http_get: HTTPGet | None = None,
        max_retries: int = 5,
        email: str | None = None,
        api_key: str | None = None,
        allow_machine_translation: bool = False,
    ) -> None:
        self._http_get = http_get or _json_get(MYMEMORY_BASE_URL, max_retries=max_retries)
        self.email = email or os.environ.get("MYMEMORY_EMAIL", "")
        self.api_key = api_key or os.environ.get("MYMEMORY_API_KEY", "")
        self.allow_machine_translation = allow_machine_translation

    def translate(
        self,
        *,
        source_name: str,
        source_language: str,
        target_language: str,
        max_candidates: int,
    ) -> tuple[list[str], int]:
        source_api = _api_language_code(source_language)
        target_api = _api_language_code(target_language)
        params: dict[str, object] = {
            "q": source_name,
            "langpair": f"{source_api}|{target_api}",
            "mt": "1" if self.allow_machine_translation else "0",
        }
        if self.email:
            params["de"] = self.email
        if self.api_key:
            params["key"] = self.api_key
        payload = self._http_get("/get", params)
        candidates: list[str] = []
        response_data = payload.get("responseData")
        if isinstance(response_data, dict):
            _append_translation(candidates, response_data.get("translatedText"), source_name=source_name)
        matches = payload.get("matches")
        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, dict):
                    continue
                _append_translation(candidates, match.get("translation"), source_name=source_name)
                if max_candidates > 0 and len(candidates) >= max_candidates:
                    break
        return candidates[:max_candidates] if max_candidates > 0 else candidates, 1


def build_translation_candidates_from_registry(
    *,
    registry_dir: str | Path,
    enrichment_dir: str | Path | None = None,
    translation_sources: tuple[str, ...] = DEFAULT_TRANSLATION_SOURCES,
    target_locales_json: str | Path = DEFAULT_TRANSLATION_TARGET_LOCALES_JSON,
    providers: dict[str, Any] | None = None,
    max_retries: int = 5,
    daily_request_limit: int = 10000,
    max_candidates_per_name: int = 0,
    mymemory_email: str | None = None,
    mymemory_key: str | None = None,
    mymemory_allow_machine_translation: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    started = monotonic()
    registry = Path(registry_dir)
    output = Path(enrichment_dir) if enrichment_dir is not None else registry
    output.mkdir(parents=True, exist_ok=True)
    source_order = tuple(source for source in translation_sources if source)
    target_locales = load_translation_target_locales(target_locales_json)

    taxa = pl.read_parquet(registry / "taxa.parquet")
    species_rows = taxa.filter(pl.col("rank") == "SPECIES").sort(["family", "genus", "scientific_name"]).to_dicts()
    if limit:
        species_rows = species_rows[:limit]
    contexts = [
        SpeciesTranslationContext(
            accepted_taxon_key=str(row.get("accepted_taxon_key") or ""),
            accepted_scientific_name=str(row.get("scientific_name") or ""),
        )
        for row in species_rows
    ]

    existing_assertions = _read_or_empty(output / SOURCE_ASSERTIONS_FILE, _name_assertion_schema()).to_dicts()
    existing_candidates = _read_or_empty(output / TRANSLATION_CANDIDATES_FILE, translation_candidate_schema()).to_dicts()
    existing_snapshots = _read_or_empty(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema()).to_dicts()
    existing_errors = _read_or_empty(output / SOURCE_ERRORS_FILE, _source_error_schema()).to_dicts()
    existing_source_work = _read_or_empty(output / SOURCE_WORK_LEDGER_FILE, _source_work_schema()).to_dicts()
    existing_translation_work = _read_or_empty(output / TRANSLATION_WORK_LEDGER_FILE, _translation_work_schema()).to_dicts()

    names = pl.read_parquet(registry / "names.parquet")
    seeds_by_taxon = _seed_names_by_taxon(taxa=taxa, names=names, source_assertions=existing_assertions)
    wikidata_items_by_taxon = _wikidata_items_by_taxon(registry, output)
    completed_work = {
        str(row.get("work_key") or "")
        for row in existing_translation_work
        if str(row.get("status") or "") == "complete"
    }
    budget = TranslationRequestBudget(daily_limit=daily_request_limit, existing_work=existing_translation_work)
    provider_bundle = providers or _default_translation_providers(
        max_retries=max_retries,
        mymemory_email=mymemory_email,
        mymemory_key=mymemory_key,
        mymemory_allow_machine_translation=mymemory_allow_machine_translation,
    )

    new_assertions: list[dict[str, Any]] = []
    new_candidates: list[dict[str, Any]] = []
    new_snapshots: list[dict[str, Any]] = []
    new_errors: list[dict[str, Any]] = []
    new_source_work: list[dict[str, Any]] = []
    new_translation_work: list[dict[str, Any]] = []
    request_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()

    logger.info(
        "registry.translation.start registry=%s enrichment_dir=%s species=%d sources=%s target_locales=%d daily_request_limit=%d",
        registry,
        output,
        len(contexts),
        ",".join(source_order),
        len(target_locales),
        daily_request_limit,
    )
    for index, context in enumerate(contexts, start=1):
        seeds = seeds_by_taxon.get(context.accepted_taxon_key, [])
        if "wikimedia" in source_order:
            result = _harvest_wikimedia_context(
                context=context,
                seeds=_wikimedia_seeds(context, seeds),
                provider=provider_bundle.get("wikimedia"),
                target_locales=target_locales,
                expected_wikidata_items=wikidata_items_by_taxon.get(context.accepted_taxon_key, set()),
                completed_work=completed_work,
                config_hash=_translation_config_hash("wikimedia", target_locales, max_candidates_per_name=0, allow_machine_translation=False),
                budget=budget,
            )
            new_assertions.extend(result["name_assertions"])
            new_snapshots.extend(result["source_snapshots"])
            new_errors.extend(result["errors"])
            new_translation_work.extend(result["translation_work"])
            new_source_work.extend(result["source_work"])
            request_counts["wikimedia"] += result["request_count"]
            assertion_counts["wikimedia"] += len(result["name_assertions"])
        if "mymemory" in source_order:
            result = _harvest_mymemory_context(
                context=context,
                seeds=_mymemory_seeds(seeds),
                provider=provider_bundle.get("mymemory"),
                target_locales=target_locales,
                completed_work=completed_work,
                config_hash=_translation_config_hash(
                    "mymemory",
                    target_locales,
                    max_candidates_per_name=max_candidates_per_name,
                    allow_machine_translation=mymemory_allow_machine_translation,
                ),
                budget=budget,
                max_candidates_per_name=max_candidates_per_name,
            )
            new_candidates.extend(result["translation_candidates"])
            new_snapshots.extend(result["source_snapshots"])
            new_errors.extend(result["errors"])
            new_translation_work.extend(result["translation_work"])
            new_source_work.extend(result["source_work"])
            request_counts["mymemory"] += result["request_count"]
            candidate_counts["mymemory"] += len(result["translation_candidates"])
        if index % 100 == 0 or index == len(contexts):
            logger.info(
                "registry.translation.progress completed=%d/%d wikimedia_assertions=%d mymemory_candidates=%d requests=%d elapsed_seconds=%.1f",
                index,
                len(contexts),
                assertion_counts["wikimedia"],
                candidate_counts["mymemory"],
                sum(request_counts.values()),
                monotonic() - started,
            )
        if budget.exhausted:
            logger.info("registry.translation.budget_exhausted completed=%d/%d request_limit=%d", index, len(contexts), daily_request_limit)
            break

    _write_translation_outputs(
        output,
        existing_assertions=existing_assertions,
        new_assertions=new_assertions,
        existing_candidates=existing_candidates,
        new_candidates=new_candidates,
        existing_snapshots=existing_snapshots,
        new_snapshots=new_snapshots,
        existing_errors=existing_errors,
        new_errors=new_errors,
        existing_source_work=existing_source_work,
        new_source_work=new_source_work,
        existing_translation_work=existing_translation_work,
        new_translation_work=new_translation_work,
    )
    manifest = _update_translation_manifest(
        output,
        source_order=source_order,
        target_locales=target_locales,
        request_counts=request_counts,
        assertion_counts=assertion_counts,
        candidate_counts=candidate_counts,
        errors=new_errors,
        work_rows=new_translation_work,
        elapsed_seconds=monotonic() - started,
        status="budget_exhausted" if budget.exhausted else "complete",
        daily_request_limit=daily_request_limit,
        max_candidates_per_name=max_candidates_per_name,
        mymemory_allow_machine_translation=mymemory_allow_machine_translation,
    )
    logger.info("registry.translation.complete registry=%s enrichment_dir=%s status=%s", registry, output, manifest.get("translation_status"))
    return manifest


def load_translation_target_locales(path: str | Path) -> tuple[str, ...]:
    candidate = Path(path)
    if not candidate.exists():
        return DEFAULT_TRANSLATION_TARGET_LOCALES
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("translation target locales config must be a JSON list")
    locales: list[str] = []
    for item in payload:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = str(item.get("locale") or item.get("code") or item.get("language") or "")
        else:
            value = ""
        code = parse_language_tag(value).bcp47
        if code and code not in locales:
            locales.append(code)
    return tuple(locales)


def _harvest_wikimedia_context(
    *,
    context: SpeciesTranslationContext,
    seeds: list[TranslationSeed],
    provider: Any,
    target_locales: tuple[str, ...],
    expected_wikidata_items: set[str],
    completed_work: set[str],
    config_hash: str,
    budget: TranslationRequestBudget,
) -> dict[str, Any]:
    source_key = "wikimedia"
    display_source = "Wikimedia"
    assertions: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    translation_work: list[dict[str, Any]] = []
    seen_assertions: set[tuple[str, str]] = set()
    request_count_total = 0
    if provider is None:
        errors.append(_source_error(display_source, context, "missing_client", retryable=False))
        return _source_result(assertions, [], snapshots, errors, translation_work, [], request_count_total)
    for seed in seeds:
        work_key = _translation_work_key(source_key, context.accepted_taxon_key, seed.source_name, seed.source_language, "*", config_hash)
        if work_key in completed_work:
            continue
        started_at = datetime.now(UTC).isoformat()
        if not budget.reserve(1):
            translation_work.append(
                _translation_work_row(
                    source=source_key,
                    context=context,
                    seed=seed,
                    target_language="*",
                    work_key=work_key,
                    provider_config_hash=config_hash,
                    status="budget_exhausted",
                    started_at=started_at,
                    request_count=0,
                )
            )
            break
        try:
            links, request_count, page_title = provider.langlinks(seed.source_name, target_locales=target_locales)
        except Exception as exc:  # noqa: BLE001 - source errors are recorded and the registry build continues.
            error_class = type(exc).__name__
            errors.append(_source_error(display_source, context, error_class, retryable=_is_retryable_error(error_class)))
            translation_work.append(
                _translation_work_row(
                    source=source_key,
                    context=context,
                    seed=seed,
                    target_language="*",
                    work_key=work_key,
                    provider_config_hash=config_hash,
                    status="error",
                    started_at=started_at,
                    request_count=1,
                    error_class=error_class,
                    retryable=_is_retryable_error(error_class),
                )
            )
            continue
        request_count_total += request_count
        for link in links:
            title = " ".join(link.title.split())
            if not title or normalize_name_key(title) in {normalize_name_key(seed.source_name), normalize_name_key(context.accepted_scientific_name)}:
                continue
            language = parse_language_tag(link.language).bcp47
            assertion_key = (language, normalize_name_key(title))
            if assertion_key in seen_assertions:
                continue
            seen_assertions.add(assertion_key)
            disabled_reason = _wikimedia_binding_disabled_reason(link, expected_wikidata_items)
            bound_to_same_taxon = not disabled_reason
            assertions.append(
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "display_name": title,
                    "language": language,
                    "script": "",
                    "region": "",
                    "name_class": "vernacular_alias",
                    "source": display_source,
                    "source_record_id": f"wikimedia:{link.page_id}:{link.wikidata_item}:{link.language}:{title}",
                    "source_taxon_id": link.wikidata_item if bound_to_same_taxon else "",
                    "trust_tier": "T3" if bound_to_same_taxon else "T4",
                    "precision_tier": "medium",
                    "confidence": "high" if bound_to_same_taxon else "low",
                    "enabled": bound_to_same_taxon,
                    "review_state": "accepted" if bound_to_same_taxon else "candidate",
                    "disabled_reason": disabled_reason,
                }
            )
        snapshots.append(_source_snapshot(display_source, WIKIMEDIA_SOURCE_VERSION, source_path=f"enwiki:{page_title}", payload={"links": [link.__dict__ for link in links]}))
        translation_work.append(
            _translation_work_row(
                source=source_key,
                context=context,
                seed=seed,
                target_language="*",
                work_key=work_key,
                provider_config_hash=config_hash,
                status="complete",
                started_at=started_at,
                request_count=request_count,
            )
        )
    source_work = [_aggregate_source_work(source_key, context, request_count_total, status="complete")] if translation_work else []
    return _source_result(assertions, [], snapshots, errors, translation_work, source_work, request_count_total)


def _harvest_mymemory_context(
    *,
    context: SpeciesTranslationContext,
    seeds: list[TranslationSeed],
    provider: Any,
    target_locales: tuple[str, ...],
    completed_work: set[str],
    config_hash: str,
    budget: TranslationRequestBudget,
    max_candidates_per_name: int,
) -> dict[str, Any]:
    source_key = "mymemory"
    display_source = "MyMemory"
    candidates: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    translation_work: list[dict[str, Any]] = []
    request_count_total = 0
    if provider is None:
        errors.append(_source_error(display_source, context, "missing_client", retryable=False))
        return _source_result([], candidates, snapshots, errors, translation_work, [], request_count_total)
    for seed in seeds:
        source_language = _api_language_code(seed.source_language or "en")
        for target_language in target_locales:
            if normalize_language_code(target_language) == normalize_language_code(source_language):
                continue
            target_api_language = _api_language_code(target_language)
            work_key = _translation_work_key(source_key, context.accepted_taxon_key, seed.source_name, source_language, target_language, config_hash)
            if work_key in completed_work:
                continue
            started_at = datetime.now(UTC).isoformat()
            if not budget.reserve(1):
                translation_work.append(
                    _translation_work_row(
                        source=source_key,
                        context=context,
                        seed=seed,
                        target_language=target_language,
                        work_key=work_key,
                        provider_config_hash=config_hash,
                        status="budget_exhausted",
                        started_at=started_at,
                        request_count=0,
                    )
                )
                break
            try:
                translated_names, request_count = provider.translate(
                    source_name=seed.source_name,
                    source_language=source_language,
                    target_language=target_api_language,
                    max_candidates=max_candidates_per_name,
                )
            except Exception as exc:  # noqa: BLE001 - source errors are recorded and the registry build continues.
                error_class = type(exc).__name__
                errors.append(_source_error(display_source, context, error_class, retryable=_is_retryable_error(error_class)))
                translation_work.append(
                    _translation_work_row(
                        source=source_key,
                        context=context,
                        seed=seed,
                        target_language=target_language,
                        work_key=work_key,
                        provider_config_hash=config_hash,
                        status="error",
                        started_at=started_at,
                        request_count=1,
                        error_class=error_class,
                        retryable=_is_retryable_error(error_class),
                    )
                )
                continue
            request_count_total += request_count
            for translated_name in translated_names:
                candidates.append(
                    generated_translation_candidate(
                        source=display_source,
                        source_language=normalize_language_code(source_language),
                        target_language=parse_language_tag(target_language).bcp47,
                        source_name=seed.source_name,
                        translated_name=translated_name,
                        accepted_taxon_key=context.accepted_taxon_key,
                        source_record_id=_translation_record_id(display_source, context.accepted_taxon_key, source_language, target_language, seed.source_name, translated_name),
                        source_kind="dictionary",
                    ).to_row()
                )
            snapshots.append(
                _source_snapshot(
                    display_source,
                    MYMEMORY_SOURCE_VERSION,
                    source_path=f"{source_language}|{target_language}:{seed.source_name}",
                    payload={"translations": translated_names},
                )
            )
            translation_work.append(
                _translation_work_row(
                    source=source_key,
                    context=context,
                    seed=seed,
                    target_language=target_language,
                    work_key=work_key,
                    provider_config_hash=config_hash,
                    status="complete",
                    started_at=started_at,
                    request_count=request_count,
                )
            )
        if budget.exhausted:
            break
    source_work = [_aggregate_source_work(source_key, context, request_count_total, status="complete")] if translation_work else []
    return _source_result([], candidates, snapshots, errors, translation_work, source_work, request_count_total)


def _write_translation_outputs(
    output: Path,
    *,
    existing_assertions: list[dict[str, Any]],
    new_assertions: list[dict[str, Any]],
    existing_candidates: list[dict[str, Any]],
    new_candidates: list[dict[str, Any]],
    existing_snapshots: list[dict[str, Any]],
    new_snapshots: list[dict[str, Any]],
    existing_errors: list[dict[str, Any]],
    new_errors: list[dict[str, Any]],
    existing_source_work: list[dict[str, Any]],
    new_source_work: list[dict[str, Any]],
    existing_translation_work: list[dict[str, Any]],
    new_translation_work: list[dict[str, Any]],
) -> None:
    assertions = _name_assertions_frame(
        _deduplicate_dicts(
            [*existing_assertions, *new_assertions],
            keys=("accepted_taxon_key", "source", "source_record_id", "display_name"),
        )
    )
    candidates = translation_candidates_frame([*existing_candidates, *new_candidates])
    snapshots = _source_snapshots_frame(
        _deduplicate_dicts(
            [*existing_snapshots, *new_snapshots],
            keys=("source", "source_version", "source_path", "source_response_hash"),
        )
    )
    errors = _source_errors_frame([*existing_errors, *new_errors])
    source_work = _source_work_frame(_deduplicate_latest_dicts([*existing_source_work, *new_source_work], keys=("source", "accepted_taxon_key")))
    translation_work = _translation_work_frame(_deduplicate_latest_dicts([*existing_translation_work, *new_translation_work], keys=("work_key",)))
    assertions.write_parquet(output / SOURCE_ASSERTIONS_FILE)
    candidates.write_parquet(output / TRANSLATION_CANDIDATES_FILE)
    snapshots.write_parquet(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
    errors.write_parquet(output / SOURCE_ERRORS_FILE)
    source_work.write_parquet(output / SOURCE_WORK_LEDGER_FILE)
    translation_work.write_parquet(output / TRANSLATION_WORK_LEDGER_FILE)


def _update_translation_manifest(
    output: Path,
    *,
    source_order: tuple[str, ...],
    target_locales: tuple[str, ...],
    request_counts: Counter[str],
    assertion_counts: Counter[str],
    candidate_counts: Counter[str],
    errors: list[dict[str, Any]],
    work_rows: list[dict[str, Any]],
    elapsed_seconds: float,
    status: str,
    daily_request_limit: int,
    max_candidates_per_name: int,
    mymemory_allow_machine_translation: bool,
) -> dict[str, Any]:
    manifest_path = output / ENRICHMENT_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "translation_sources": list(source_order),
            "translation_source_display_names": list(translation_source_display_names(source_order)),
            "translation_target_locale_count": len(target_locales),
            "translation_target_locales": list(target_locales),
            "translation_status": status,
            "translation_daily_request_limit": daily_request_limit,
            "translation_max_candidates_per_name": max_candidates_per_name,
            "mymemory_allow_machine_translation": mymemory_allow_machine_translation,
            "wikimedia_assertion_rows": assertion_counts.get("wikimedia", 0),
            "mymemory_candidate_rows": candidate_counts.get("mymemory", 0),
            "translation_request_rows": sum(request_counts.values()),
            "translation_error_rows": len(errors),
            "translation_work_rows": len(work_rows),
            "translation_error_counts_by_source": dict(sorted(Counter(str(error.get("source") or "") for error in errors).items())),
            "translation_elapsed_seconds": round(elapsed_seconds, 6),
            "files": {
                **manifest.get("files", {}),
                "translation_candidates": TRANSLATION_CANDIDATES_FILE,
                "translation_work_ledger": TRANSLATION_WORK_LEDGER_FILE,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _seed_names_by_taxon(*, taxa: pl.DataFrame, names: pl.DataFrame, source_assertions: list[dict[str, Any]]) -> dict[str, list[TranslationSeed]]:
    species = {
        str(row.get("accepted_taxon_key") or ""): str(row.get("scientific_name") or "")
        for row in taxa.filter(pl.col("rank") == "SPECIES").to_dicts()
    }
    rows = [*names.to_dicts(), *source_assertions]
    seeds: dict[str, list[TranslationSeed]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        if accepted_taxon_key not in species:
            continue
        display_name = " ".join(str(row.get("display_name") or row.get("verbatim_name") or "").split())
        if not display_name or normalize_name_key(display_name) == normalize_name_key(species[accepted_taxon_key]):
            continue
        name_class = str(row.get("name_class") or "")
        if name_class in SCIENTIFIC_NAME_CLASSES or name_class not in COMMON_NAME_CLASSES:
            continue
        language = normalize_language_code(row.get("language") or "eng")
        if language not in ENGLISH_LANGUAGE_CODES and _api_language_code(language) != "en":
            continue
        if not _boolish(row.get("enabled", True)):
            continue
        trust_tier = str(row.get("trust_tier") or "")
        if trust_tier == "T5":
            continue
        if not _translation_seed_query_eligible(row):
            continue
        key = (accepted_taxon_key, normalize_name_key(display_name), language)
        if key in seen:
            continue
        seen.add(key)
        seeds.setdefault(accepted_taxon_key, []).append(
            TranslationSeed(
                accepted_taxon_key=accepted_taxon_key,
                accepted_scientific_name=species[accepted_taxon_key],
                source_name=display_name,
                source_language=language,
                source=str(row.get("source") or ""),
                name_class=name_class,
                trust_tier=trust_tier,
            )
        )
    return {taxon: sorted(values, key=lambda seed: (seed.source_name.casefold(), seed.source)) for taxon, values in seeds.items()}


def _wikimedia_seeds(context: SpeciesTranslationContext, common_seeds: list[TranslationSeed]) -> list[TranslationSeed]:
    scientific_seed = TranslationSeed(
        accepted_taxon_key=context.accepted_taxon_key,
        accepted_scientific_name=context.accepted_scientific_name,
        source_name=context.accepted_scientific_name,
        source_language="la",
        source="GBIF",
        name_class="accepted_scientific",
        trust_tier="T1",
    )
    return [scientific_seed, *common_seeds]


def _mymemory_seeds(common_seeds: list[TranslationSeed]) -> list[TranslationSeed]:
    return common_seeds


def _translation_seed_query_eligible(row: dict[str, Any]) -> bool:
    if "query_eligible" in row and row.get("query_eligible") is not None:
        return _boolish(row.get("query_eligible"))
    return assess_name_query_eligibility(row).query_eligible


def _default_translation_providers(
    *,
    max_retries: int,
    mymemory_email: str | None,
    mymemory_key: str | None,
    mymemory_allow_machine_translation: bool,
) -> dict[str, Any]:
    return {
        "wikimedia": WikimediaLanglinksProvider(max_retries=max_retries),
        "mymemory": MyMemoryTranslationProvider(
            max_retries=max_retries,
            email=mymemory_email,
            api_key=mymemory_key,
            allow_machine_translation=mymemory_allow_machine_translation,
        ),
    }


def _wikidata_items_by_taxon(*roots: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for root in dict.fromkeys(roots):
        path = root / EXTERNAL_LINKS_FILE
        if not path.exists():
            continue
        links = pl.read_parquet(path)
        if links.is_empty() or not {"accepted_taxon_key", "source", "source_taxon_id"}.issubset(links.columns):
            continue
        for row in links.to_dicts():
            if str(row.get("source") or "").casefold() != "wikidata":
                continue
            accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
            source_taxon_id = str(row.get("source_taxon_id") or "")
            if accepted_taxon_key and source_taxon_id:
                result.setdefault(accepted_taxon_key, set()).add(source_taxon_id)
    return result


def _wikimedia_binding_disabled_reason(link: WikimediaLanglink, expected_wikidata_items: set[str]) -> str:
    if not expected_wikidata_items:
        return "wikimedia_source_binding_missing"
    if not link.wikidata_item:
        return "wikimedia_page_missing_wikidata_item"
    if link.wikidata_item not in expected_wikidata_items:
        return "wikimedia_page_not_bound_to_accepted_taxon"
    return ""


def _translation_work_schema() -> dict[str, pl.DataType]:
    return {
        "source": pl.String,
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "source_name": pl.String,
        "source_language": pl.String,
        "target_language": pl.String,
        "work_key": pl.String,
        "provider_config_hash": pl.String,
        "status": pl.String,
        "attempts": pl.Int64,
        "started_at": pl.String,
        "finished_at": pl.String,
        "request_count": pl.Int64,
        "error_class": pl.String,
        "retryable": pl.String,
        "request_day": pl.String,
    }


def _translation_work_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "source": str(row.get("source") or ""),
            "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
            "accepted_scientific_name": str(row.get("accepted_scientific_name") or ""),
            "source_name": str(row.get("source_name") or ""),
            "source_language": str(row.get("source_language") or ""),
            "target_language": str(row.get("target_language") or ""),
            "work_key": str(row.get("work_key") or ""),
            "provider_config_hash": str(row.get("provider_config_hash") or ""),
            "status": str(row.get("status") or ""),
            "attempts": int(row.get("attempts") or 0),
            "started_at": str(row.get("started_at") or ""),
            "finished_at": str(row.get("finished_at") or ""),
            "request_count": int(row.get("request_count") or 0),
            "error_class": str(row.get("error_class") or ""),
            "retryable": str(row.get("retryable") or "false"),
            "request_day": str(row.get("request_day") or ""),
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=_translation_work_schema()) if normalized else pl.DataFrame(schema=_translation_work_schema())


def _translation_work_row(
    *,
    source: str,
    context: SpeciesTranslationContext,
    seed: TranslationSeed,
    target_language: str,
    work_key: str,
    provider_config_hash: str,
    status: str,
    started_at: str,
    request_count: int,
    error_class: str = "",
    retryable: bool = False,
) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    return TranslationWorkRecord(
        source=source,
        accepted_taxon_key=context.accepted_taxon_key,
        accepted_scientific_name=context.accepted_scientific_name,
        source_name=seed.source_name,
        source_language=seed.source_language,
        target_language=target_language,
        work_key=work_key,
        provider_config_hash=provider_config_hash,
        status=status,
        attempts=1,
        started_at=started_at,
        finished_at=now,
        request_count=request_count,
        error_class=error_class,
        retryable=str(retryable).casefold(),
    ).to_row()


def _aggregate_source_work(source: str, context: SpeciesTranslationContext, request_count: int, *, status: str, error_class: str = "") -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    return {
        "source": source,
        "accepted_taxon_key": context.accepted_taxon_key,
        "accepted_scientific_name": context.accepted_scientific_name,
        "status": status,
        "attempts": 1,
        "started_at": now,
        "finished_at": now,
        "request_count": request_count,
        "error_class": error_class,
        "request_day": datetime.now(UTC).date().isoformat(),
    }


def _source_error(source: str, context: SpeciesTranslationContext, error_class: str, *, retryable: bool) -> dict[str, object]:
    return {
        "source": source,
        "accepted_taxon_key": context.accepted_taxon_key,
        "endpoint": "",
        "error_class": error_class,
        "attempts": "1",
        "retryable": str(retryable).casefold(),
        "disposition": "quarantined",
    }


def _source_snapshot(source: str, version: str, *, source_path: str, payload: dict[str, Any]) -> dict[str, object]:
    return {
        "source": source,
        "source_version": version,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "source_response_hash": _payload_hash(payload),
        "licence": "",
    }


def _source_result(
    assertions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    translation_work: list[dict[str, Any]],
    source_work: list[dict[str, Any]],
    request_count: int,
) -> dict[str, Any]:
    return {
        "name_assertions": assertions,
        "translation_candidates": candidates,
        "source_snapshots": snapshots,
        "errors": errors,
        "translation_work": translation_work,
        "source_work": source_work,
        "request_count": request_count,
    }


def _read_or_empty(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=schema)


def _append_translation(candidates: list[str], value: object, *, source_name: str) -> None:
    text = " ".join(str(value or "").split())
    if not text:
        return
    if normalize_name_key(text) == normalize_name_key(source_name):
        return
    if normalize_name_key(text) in {normalize_name_key(existing) for existing in candidates}:
        return
    candidates.append(text)


def _api_language_code(value: object) -> str:
    tag = parse_language_tag(value)
    if not tag.api_language_code:
        return ""
    return API_LANGUAGE_CODES.get(tag.language, API_LANGUAGE_CODES.get(tag.api_language_code, tag.api_language_code))


def _translation_config_hash(source: str, target_locales: tuple[str, ...], *, max_candidates_per_name: int, allow_machine_translation: bool) -> str:
    return _stable_id("translation-config", source, ",".join(target_locales), max_candidates_per_name, allow_machine_translation, USER_AGENT)


def _translation_work_key(source: str, accepted_taxon_key: str, source_name: str, source_language: str, target_language: str, config_hash: str) -> str:
    return _stable_id("translation-work", source, accepted_taxon_key, source_name, source_language, target_language, config_hash)


def _translation_record_id(source: str, accepted_taxon_key: str, source_language: str, target_language: str, source_name: str, translated_name: str) -> str:
    return _stable_id("translation-record", source, accepted_taxon_key, source_language, target_language, source_name, translated_name)


def _stable_id(*parts: object) -> str:
    payload = json.dumps([str(part or "") for part in parts], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    return _stable_id("payload", json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "accepted", "enabled"}


def _is_retryable_error(error_class: str) -> bool:
    return error_class in {"TimeoutException", "TransportError", "HTTPStatusError", "ConnectError", "ReadTimeout"}


__all__ = [
    "DEFAULT_TRANSLATION_TARGET_LOCALES",
    "TRANSLATION_WORK_LEDGER_FILE",
    "MyMemoryTranslationProvider",
    "WikimediaLanglinksProvider",
    "build_translation_candidates_from_registry",
    "load_translation_target_locales",
]
