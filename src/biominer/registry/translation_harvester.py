from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
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
MYMEMORY_MONTHLY_REQUEST_LIMIT = 10_000
MYMEMORY_MONTHLY_INPUT_WORD_LIMIT = 10_000
MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT = 10_240
MYMEMORY_RESPONSE_BYTE_RESERVATION = 1_048_576
MYMEMORY_SOURCE_VERSION = "mymemory-get-v1"
WIKIMEDIA_SOURCE_VERSION = "mediawiki-langlinks-pageprops-wikispecies-v3"
MYMEMORY_BASE_URL = "https://api.mymemory.translated.net"
WIKIPEDIA_API_BASE_URL = "https://en.wikipedia.org"
WIKISPECIES_API_BASE_URL = "https://species.wikimedia.org"
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
    input_word_count: int = 0
    response_byte_count: int = 0
    bandwidth_reserved_byte_count: int = 0
    budget_exhausted_reason: str = ""

    def to_row(self) -> dict[str, object]:
        row = self.__dict__.copy()
        row["request_day"] = self.request_day or datetime.now(UTC).date().isoformat()
        return row


@dataclass(frozen=True)
class MyMemoryBudgetReservation:
    allowed: bool
    reason: str = ""
    request_count: int = 0
    input_word_count: int = 0
    bandwidth_reserved_byte_count: int = 0


class MyMemoryMonthlyBudget:
    def __init__(
        self,
        *,
        request_limit: int,
        input_word_limit: int,
        bandwidth_byte_limit: int,
        response_byte_reservation: int,
        existing_work: list[dict[str, Any]],
    ) -> None:
        self.month = datetime.now(UTC).strftime("%Y-%m")
        self.request_limit = max(0, int(request_limit))
        self.input_word_limit = max(0, int(input_word_limit))
        self.bandwidth_byte_limit = max(0, int(bandwidth_byte_limit))
        self.response_byte_reservation = max(0, int(response_byte_reservation))
        self.requests_used = 0
        self.input_words_used = 0
        self.bandwidth_reserved_bytes = 0
        self.response_bytes_observed = 0
        self.exhausted = False
        self.exhausted_reason = ""
        self._lock = threading.Lock()
        for row in existing_work:
            if str(row.get("source") or "").casefold() != "mymemory":
                continue
            if not str(row.get("request_day") or "").startswith(self.month):
                continue
            request_count = int(row.get("request_count") or 0)
            self.requests_used += request_count
            self.input_words_used += int(row.get("input_word_count") or 0) or request_count * _input_word_count(row.get("source_name"))
            self.bandwidth_reserved_bytes += int(row.get("bandwidth_reserved_byte_count") or 0) or request_count * self.response_byte_reservation
            self.response_bytes_observed += int(row.get("response_byte_count") or 0)

    def reserve(self, seed: TranslationSeed) -> MyMemoryBudgetReservation:
        input_words = _input_word_count(seed.source_name)
        with self._lock:
            checks = (
                (self.requests_used + 1 > self.request_limit, "mymemory_monthly_request_limit"),
                (self.input_words_used + input_words > self.input_word_limit, "mymemory_monthly_input_word_limit"),
                (
                    self.bandwidth_reserved_bytes + self.response_byte_reservation > self.bandwidth_byte_limit,
                    "mymemory_monthly_bandwidth_limit",
                ),
            )
            for failed, reason in checks:
                if failed:
                    self.exhausted = True
                    self.exhausted_reason = reason
                    return MyMemoryBudgetReservation(allowed=False, reason=reason)
            self.requests_used += 1
            self.input_words_used += input_words
            self.bandwidth_reserved_bytes += self.response_byte_reservation
            return MyMemoryBudgetReservation(
                allowed=True,
                request_count=1,
                input_word_count=input_words,
                bandwidth_reserved_byte_count=self.response_byte_reservation,
            )

    def release(self, reservation: MyMemoryBudgetReservation) -> None:
        if not reservation.allowed:
            return
        with self._lock:
            self.requests_used = max(0, self.requests_used - reservation.request_count)
            self.input_words_used = max(0, self.input_words_used - reservation.input_word_count)
            self.bandwidth_reserved_bytes = max(0, self.bandwidth_reserved_bytes - reservation.bandwidth_reserved_byte_count)

    def record_response(self, response_byte_count: int) -> None:
        with self._lock:
            self.response_bytes_observed += max(0, int(response_byte_count))
            if self.response_bytes_observed > self.bandwidth_byte_limit:
                self.exhausted = True
                self.exhausted_reason = "mymemory_monthly_observed_bandwidth_limit"

    def manifest_metrics(self) -> dict[str, Any]:
        return {
            "mymemory_monthly_budget_month": self.month,
            "mymemory_monthly_request_limit": self.request_limit,
            "mymemory_monthly_requests_used": self.requests_used,
            "mymemory_monthly_input_word_limit": self.input_word_limit,
            "mymemory_monthly_input_words_used": self.input_words_used,
            "mymemory_monthly_bandwidth_byte_limit": self.bandwidth_byte_limit,
            "mymemory_monthly_bandwidth_reserved_bytes": self.bandwidth_reserved_bytes,
            "mymemory_monthly_response_bytes_observed": self.response_bytes_observed,
            "mymemory_response_byte_reservation": self.response_byte_reservation,
            "mymemory_budget_exhausted": self.exhausted,
            "mymemory_budget_exhausted_reason": self.exhausted_reason,
        }


@dataclass(frozen=True)
class TranslationWorkUnit:
    source: str
    context: SpeciesTranslationContext
    seed: TranslationSeed
    target_language: str
    target_api_language: str
    work_key: str
    provider_config_hash: str


@dataclass(frozen=True)
class TranslationBatch:
    name_assertions: tuple[dict[str, Any], ...] = ()
    translation_candidates: tuple[dict[str, Any], ...] = ()
    source_snapshots: tuple[dict[str, Any], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    translation_work: tuple[dict[str, Any], ...] = ()
    source_work: tuple[dict[str, Any], ...] = ()
    request_count: int = 0


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
        self._lock = threading.Lock()

    def reserve(self, count: int = 1) -> bool:
        with self._lock:
            if self.daily_limit <= 0:
                return True
            if self.used + count > self.daily_limit:
                self.exhausted = True
                return False
            self.used += count
            return True


class TranslationCheckpointWriter:
    def __init__(
        self,
        output: Path,
        *,
        source_order: tuple[str, ...],
        target_locales: tuple[str, ...],
        daily_request_limit: int,
        max_candidates_per_name: int,
        mymemory_allow_machine_translation: bool,
        started_at: float,
        checkpoint_every: int,
        checkpoint_seconds: float,
    ) -> None:
        self.output = output
        self.source_order = source_order
        self.target_locales = target_locales
        self.daily_request_limit = daily_request_limit
        self.max_candidates_per_name = max_candidates_per_name
        self.mymemory_allow_machine_translation = mymemory_allow_machine_translation
        self.started_at = started_at
        self.checkpoint_every = max(1, checkpoint_every)
        self.checkpoint_seconds = checkpoint_seconds
        self.last_flush_at = monotonic()
        self.existing_assertions = _read_or_empty(output / SOURCE_ASSERTIONS_FILE, _name_assertion_schema()).to_dicts()
        self.existing_candidates = _read_or_empty(output / TRANSLATION_CANDIDATES_FILE, translation_candidate_schema()).to_dicts()
        self.existing_snapshots = _read_or_empty(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema()).to_dicts()
        self.existing_errors = _read_or_empty(output / SOURCE_ERRORS_FILE, _source_error_schema()).to_dicts()
        self.existing_source_work = _read_or_empty(output / SOURCE_WORK_LEDGER_FILE, _source_work_schema()).to_dicts()
        self.existing_translation_work = _read_or_empty(output / TRANSLATION_WORK_LEDGER_FILE, _translation_work_schema()).to_dicts()
        self.buffered_assertions: list[dict[str, Any]] = []
        self.buffered_candidates: list[dict[str, Any]] = []
        self.buffered_snapshots: list[dict[str, Any]] = []
        self.buffered_errors: list[dict[str, Any]] = []
        self.buffered_source_work: list[dict[str, Any]] = []
        self.buffered_translation_work: list[dict[str, Any]] = []
        self.buffered_work_count = 0
        self.request_counts: Counter[str] = Counter()
        self.candidate_counts: Counter[str] = Counter()
        self.assertion_counts: Counter[str] = Counter()
        self.manifest = _read_json_or_empty(output / ENRICHMENT_MANIFEST_FILE)
        self.mymemory_budget: MyMemoryMonthlyBudget | None = None

    def append(self, batch: TranslationBatch) -> None:
        self.buffered_assertions.extend(batch.name_assertions)
        self.buffered_candidates.extend(batch.translation_candidates)
        self.buffered_snapshots.extend(batch.source_snapshots)
        self.buffered_errors.extend(batch.errors)
        self.buffered_source_work.extend(batch.source_work)
        self.buffered_translation_work.extend(batch.translation_work)
        self.buffered_work_count += len(batch.translation_work)
        source_key = _batch_source_key(batch)
        if source_key and batch.request_count:
            self.request_counts[source_key] += batch.request_count
        for row in batch.translation_candidates:
            self.candidate_counts[_source_counter_key(row.get("source"))] += 1
        for row in batch.name_assertions:
            self.assertion_counts[_source_counter_key(row.get("source"))] += 1

    def should_flush(self) -> bool:
        if not self._has_buffered_rows():
            return False
        if self.buffered_work_count >= self.checkpoint_every:
            return True
        if self.checkpoint_seconds <= 0:
            return True
        return monotonic() - self.last_flush_at >= self.checkpoint_seconds

    def flush(self, *, status: str, force: bool = False) -> dict[str, Any]:
        if not force and not self._has_buffered_rows():
            return self.manifest
        frames = _translation_output_frames(
            existing_assertions=self.existing_assertions,
            new_assertions=self.buffered_assertions,
            existing_candidates=self.existing_candidates,
            new_candidates=self.buffered_candidates,
            existing_snapshots=self.existing_snapshots,
            new_snapshots=self.buffered_snapshots,
            existing_errors=self.existing_errors,
            new_errors=self.buffered_errors,
            existing_source_work=self.existing_source_work,
            new_source_work=self.buffered_source_work,
            existing_translation_work=self.existing_translation_work,
            new_translation_work=self.buffered_translation_work,
        )
        _write_parquet_atomic(frames["assertions"], self.output / SOURCE_ASSERTIONS_FILE)
        _write_parquet_atomic(frames["candidates"], self.output / TRANSLATION_CANDIDATES_FILE)
        _write_parquet_atomic(frames["snapshots"], self.output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
        _write_parquet_atomic(frames["errors"], self.output / SOURCE_ERRORS_FILE)
        _write_parquet_atomic(frames["source_work"], self.output / SOURCE_WORK_LEDGER_FILE)
        _write_parquet_atomic(frames["translation_work"], self.output / TRANSLATION_WORK_LEDGER_FILE)
        self.manifest = self._manifest_payload(status=status, frames=frames)
        _write_json_atomic(self.manifest, self.output / ENRICHMENT_MANIFEST_FILE)
        self.existing_assertions = frames["assertions"].to_dicts()
        self.existing_candidates = frames["candidates"].to_dicts()
        self.existing_snapshots = frames["snapshots"].to_dicts()
        self.existing_errors = frames["errors"].to_dicts()
        self.existing_source_work = frames["source_work"].to_dicts()
        self.existing_translation_work = frames["translation_work"].to_dicts()
        self._clear_buffers()
        self.last_flush_at = monotonic()
        return self.manifest

    def completion_status(self, *, budget_exhausted: bool) -> str:
        if budget_exhausted:
            return "budget_exhausted"
        work_rows = [*self.existing_translation_work, *self.buffered_translation_work]
        if self.existing_errors or self.buffered_errors or any(str(row.get("status") or "") == "error" for row in work_rows):
            return "complete_with_errors"
        return "complete"

    def _manifest_payload(self, *, status: str, frames: dict[str, pl.DataFrame]) -> dict[str, Any]:
        errors = frames["errors"].to_dicts()
        assertion_counts = _frame_source_counts(frames["assertions"], "source")
        candidate_counts = _frame_source_counts(frames["candidates"], "source")
        request_counts = _translation_request_counts_from_source_work_frame(frames["source_work"])
        manifest = dict(self.manifest)
        manifest.update(
            {
                "translation_sources": list(self.source_order),
                "translation_source_display_names": list(translation_source_display_names(self.source_order)),
                "translation_target_locale_count": len(self.target_locales),
                "translation_target_locales": list(self.target_locales),
                "translation_status": status,
                "translation_daily_request_limit": self.daily_request_limit,
                "translation_max_candidates_per_name": self.max_candidates_per_name,
                "mymemory_allow_machine_translation": self.mymemory_allow_machine_translation,
                "wikimedia_assertion_rows": assertion_counts.get("wikimedia", 0),
                "mymemory_candidate_rows": candidate_counts.get("mymemory", 0),
                "translation_request_rows": sum(request_counts.values()),
                "translation_request_counts_by_source": dict(sorted(request_counts.items())),
                "translation_current_run_request_rows": sum(self.request_counts.values()),
                "translation_current_run_request_counts_by_source": dict(sorted(self.request_counts.items())),
                "translation_assertion_counts_by_source": dict(sorted(assertion_counts.items())),
                "translation_candidate_counts_by_source": dict(sorted(candidate_counts.items())),
                "translation_error_rows": frames["errors"].height,
                "translation_work_rows": frames["translation_work"].height,
                "translation_error_counts_by_source": dict(
                    sorted(Counter(_source_counter_key(error.get("source")) for error in errors if _source_counter_key(error.get("source"))).items())
                ),
                "translation_elapsed_seconds": round(monotonic() - self.started_at, 6),
                "files": {
                    **manifest.get("files", {}),
                    "translation_candidates": TRANSLATION_CANDIDATES_FILE,
                    "translation_work_ledger": TRANSLATION_WORK_LEDGER_FILE,
                },
            }
        )
        if self.mymemory_budget is not None:
            manifest.update(self.mymemory_budget.manifest_metrics())
        return manifest

    def _has_buffered_rows(self) -> bool:
        return any(
            (
                self.buffered_assertions,
                self.buffered_candidates,
                self.buffered_snapshots,
                self.buffered_errors,
                self.buffered_source_work,
                self.buffered_translation_work,
            )
        )

    def _clear_buffers(self) -> None:
        self.buffered_assertions.clear()
        self.buffered_candidates.clear()
        self.buffered_snapshots.clear()
        self.buffered_errors.clear()
        self.buffered_source_work.clear()
        self.buffered_translation_work.clear()
        self.buffered_work_count = 0


class WikimediaLanglinksProvider:
    source_key = "wikimedia"
    source_name = "Wikimedia"

    def __init__(self, *, http_get: HTTPGet | None = None, wikispecies_http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get(WIKIPEDIA_API_BASE_URL, max_retries=max_retries)
        self._wikispecies_http_get = wikispecies_http_get or (http_get if http_get is not None else _json_get(WIKISPECIES_API_BASE_URL, max_retries=max_retries))

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

    def vernacular_names(self, title: str, *, target_locales: tuple[str, ...]) -> tuple[list[WikimediaLanglink], int, str]:
        params: dict[str, object] = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "wikitext",
            "redirects": "1",
        }
        payload = self._wikispecies_http_get("/w/api.php", params)
        parsed = payload.get("parse", {}) if isinstance(payload, dict) else {}
        if not isinstance(parsed, dict):
            return [], 1, title
        page_title = str(parsed.get("title") or title)
        page_id = str(parsed.get("pageid") or "")
        wikitext = parsed.get("wikitext", {})
        if isinstance(wikitext, dict):
            wikitext_value = str(wikitext.get("*") or "")
        else:
            wikitext_value = str(wikitext or "")
        target_api_codes = {_api_language_code(locale) for locale in target_locales}
        wikidata_item = _wikispecies_taxonbar_item(wikitext_value)
        links = _wikispecies_vernacular_links(
            wikitext_value,
            target_api_codes=target_api_codes,
            page_id=f"wikispecies:{page_id}" if page_id else "wikispecies",
            page_title=page_title,
            wikidata_item=wikidata_item,
        )
        return links, 1, page_title


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
        self.email = email
        self.api_key = api_key
        self.allow_machine_translation = allow_machine_translation

    def translate(
        self,
        *,
        source_name: str,
        source_language: str,
        target_language: str,
        max_candidates: int,
    ) -> tuple[list[str], int, int]:
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
        response_byte_count = len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return candidates[:max_candidates] if max_candidates > 0 else candidates, 1, response_byte_count


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
    mymemory_monthly_request_limit: int = MYMEMORY_MONTHLY_REQUEST_LIMIT,
    mymemory_monthly_input_word_limit: int = MYMEMORY_MONTHLY_INPUT_WORD_LIMIT,
    mymemory_monthly_bandwidth_mb_limit: int = MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT,
    mymemory_response_byte_reservation: int = MYMEMORY_RESPONSE_BYTE_RESERVATION,
    translation_checkpoint_every: int = 100,
    translation_checkpoint_seconds: float = 60.0,
    translation_workers: int = 1,
    translation_language_shards: int = 0,
    limit: int = 0,
) -> dict[str, Any]:
    started = monotonic()
    registry = Path(registry_dir)
    output = Path(enrichment_dir) if enrichment_dir is not None else registry
    output.mkdir(parents=True, exist_ok=True)
    source_order = tuple(source for source in translation_sources if source)
    target_locales = load_translation_target_locales(target_locales_json)
    translation_worker_count = max(1, translation_workers)
    language_shard_count = max(1, translation_language_shards or translation_worker_count)

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

    checkpoint_writer = TranslationCheckpointWriter(
        output,
        source_order=source_order,
        target_locales=target_locales,
        daily_request_limit=daily_request_limit,
        max_candidates_per_name=max_candidates_per_name,
        mymemory_allow_machine_translation=mymemory_allow_machine_translation,
        started_at=started,
        checkpoint_every=translation_checkpoint_every,
        checkpoint_seconds=translation_checkpoint_seconds,
    )
    mymemory_budget = MyMemoryMonthlyBudget(
        request_limit=mymemory_monthly_request_limit,
        input_word_limit=mymemory_monthly_input_word_limit,
        bandwidth_byte_limit=mymemory_monthly_bandwidth_mb_limit * 1024 * 1024,
        response_byte_reservation=mymemory_response_byte_reservation,
        existing_work=checkpoint_writer.existing_translation_work,
    )
    checkpoint_writer.mymemory_budget = mymemory_budget

    names = pl.read_parquet(registry / "names.parquet")
    seeds_by_taxon = _seed_names_by_taxon(taxa=taxa, names=names, source_assertions=checkpoint_writer.existing_assertions)
    wikidata_items_by_taxon = _wikidata_items_by_taxon(registry, output)
    completed_work = {
        str(row.get("work_key") or "")
        for row in checkpoint_writer.existing_translation_work
        if str(row.get("status") or "") == "complete"
    }
    budget = TranslationRequestBudget(daily_limit=daily_request_limit, existing_work=checkpoint_writer.existing_translation_work)
    provider_bundle = providers or _default_translation_providers(
        max_retries=max_retries,
        mymemory_email=mymemory_email,
        mymemory_key=mymemory_key,
        mymemory_allow_machine_translation=mymemory_allow_machine_translation,
    )
    mymemory_provider_spec = provider_bundle.get("mymemory") if "mymemory" in source_order else None
    mymemory_config_hash = _translation_config_hash(
        "mymemory",
        target_locales,
        max_candidates_per_name=max_candidates_per_name,
        allow_machine_translation=mymemory_allow_machine_translation,
    )
    mymemory_units: list[TranslationWorkUnit] = []

    logger.info(
        "registry.translation.start registry=%s enrichment_dir=%s species=%d sources=%s target_locales=%d daily_request_limit=%d checkpoint_every=%d checkpoint_seconds=%.1f translation_workers=%d translation_language_shards=%d",
        registry,
        output,
        len(contexts),
        ",".join(source_order),
        len(target_locales),
        daily_request_limit,
        translation_checkpoint_every,
        translation_checkpoint_seconds,
        translation_worker_count,
        language_shard_count,
    )
    for index, context in enumerate(contexts, start=1):
        seeds = seeds_by_taxon.get(context.accepted_taxon_key, [])
        if "wikimedia" in source_order:
            batch = _batch_from_source_result(
                _harvest_wikimedia_context(
                    context=context,
                    seeds=_wikimedia_seeds(context, seeds),
                    provider=provider_bundle.get("wikimedia"),
                    target_locales=target_locales,
                    expected_wikidata_items=wikidata_items_by_taxon.get(context.accepted_taxon_key, set()),
                    completed_work=completed_work,
                    config_hash=_translation_config_hash("wikimedia", target_locales, max_candidates_per_name=0, allow_machine_translation=False),
                    budget=budget,
                )
            )
            checkpoint_writer.append(batch)
            completed_work.update(str(row.get("work_key") or "") for row in batch.translation_work if str(row.get("status") or "") == "complete")
            if checkpoint_writer.should_flush():
                checkpoint_writer.flush(status="running")
        if "mymemory" in source_order:
            if mymemory_provider_spec is None:
                batch = _batch_from_source_result(
                    _harvest_mymemory_context(
                        context=context,
                        seeds=_mymemory_seeds(seeds),
                        provider=None,
                        target_locales=target_locales,
                        completed_work=completed_work,
                        config_hash=mymemory_config_hash,
                        budget=budget,
                        mymemory_budget=mymemory_budget,
                        max_candidates_per_name=max_candidates_per_name,
                    )
                )
                checkpoint_writer.append(batch)
                if checkpoint_writer.should_flush():
                    checkpoint_writer.flush(status="running")
            else:
                mymemory_units.extend(
                    _mymemory_work_units(
                        context,
                        _mymemory_seeds(seeds),
                        target_locales=target_locales,
                        completed_work=completed_work,
                        config_hash=mymemory_config_hash,
                    )
                )
        if index % 100 == 0 or index == len(contexts):
            logger.info(
                "registry.translation.progress completed=%d/%d wikimedia_assertions=%d mymemory_candidates=%d requests=%d elapsed_seconds=%.1f",
                index,
                len(contexts),
                checkpoint_writer.assertion_counts["wikimedia"],
                checkpoint_writer.candidate_counts["mymemory"],
                sum(checkpoint_writer.request_counts.values()),
                monotonic() - started,
            )
        if budget.exhausted or mymemory_budget.exhausted:
            logger.info("registry.translation.budget_exhausted completed=%d/%d request_limit=%d", index, len(contexts), daily_request_limit)
            break

    if mymemory_units and not budget.exhausted and not mymemory_budget.exhausted:
        logger.info(
            "registry.translation.mymemory.start units=%d translation_workers=%d language_shards=%d",
            len(mymemory_units),
            translation_worker_count,
            language_shard_count,
        )
        if translation_worker_count <= 1 or len(mymemory_units) <= 1:
            mymemory_provider = _new_mymemory_provider(mymemory_provider_spec)
            mymemory_batches = (
                _harvest_single_mymemory_unit(
                    unit,
                    provider=mymemory_provider,
                    budget=budget,
                    mymemory_budget=mymemory_budget,
                    max_candidates_per_name=max_candidates_per_name,
                )
                for unit in mymemory_units
            )
        else:
            mymemory_batches = _harvest_mymemory_units_parallel(
                tuple(mymemory_units),
                target_locales=target_locales,
                language_shards=language_shard_count,
                max_workers=translation_worker_count,
                provider_spec=mymemory_provider_spec,
                budget=budget,
                mymemory_budget=mymemory_budget,
                max_candidates_per_name=max_candidates_per_name,
            )
        for batch in mymemory_batches:
            checkpoint_writer.append(batch)
            completed_work.update(str(row.get("work_key") or "") for row in batch.translation_work if str(row.get("status") or "") == "complete")
            if checkpoint_writer.should_flush():
                checkpoint_writer.flush(status="running")
            if budget.exhausted or mymemory_budget.exhausted:
                break

    manifest = checkpoint_writer.flush(
        status=checkpoint_writer.completion_status(budget_exhausted=budget.exhausted or mymemory_budget.exhausted),
        force=True,
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
    wikispecies_seed = TranslationSeed(
        accepted_taxon_key=context.accepted_taxon_key,
        accepted_scientific_name=context.accepted_scientific_name,
        source_name=context.accepted_scientific_name,
        source_language="mul",
        source="Wikispecies",
        name_class="vernacular",
        trust_tier="T3",
    )
    wikispecies_work_key = _translation_work_key(source_key, context.accepted_taxon_key, context.accepted_scientific_name, "wikispecies", "*", config_hash)
    if wikispecies_work_key not in completed_work and hasattr(provider, "vernacular_names"):
        started_at = datetime.now(UTC).isoformat()
        if budget.reserve(1):
            try:
                links, request_count, page_title = provider.vernacular_names(context.accepted_scientific_name, target_locales=target_locales)
            except Exception as exc:  # noqa: BLE001 - source errors are recorded and the registry build continues.
                error_class = type(exc).__name__
                request_count_total += 1
                errors.append(_source_error(display_source, context, error_class, retryable=_is_retryable_error(error_class)))
                translation_work.append(
                    _translation_work_row(
                        source=source_key,
                        context=context,
                        seed=wikispecies_seed,
                        target_language="*",
                        work_key=wikispecies_work_key,
                        provider_config_hash=config_hash,
                        status="error",
                        started_at=started_at,
                        request_count=1,
                        error_class=error_class,
                        retryable=_is_retryable_error(error_class),
                    )
                )
            else:
                request_count_total += request_count
                _append_wikimedia_assertions(
                    assertions=assertions,
                    seen_assertions=seen_assertions,
                    links=links,
                    context=context,
                    expected_wikidata_items=expected_wikidata_items,
                    skip_names={context.accepted_scientific_name},
                )
                snapshots.append(_source_snapshot(display_source, WIKIMEDIA_SOURCE_VERSION, source_path=f"wikispecies:{page_title}", payload={"links": [link.__dict__ for link in links]}))
                translation_work.append(
                    _translation_work_row(
                        source=source_key,
                        context=context,
                        seed=wikispecies_seed,
                        target_language="*",
                        work_key=wikispecies_work_key,
                        provider_config_hash=config_hash,
                        status="complete",
                        started_at=started_at,
                        request_count=request_count,
                    )
                )
        else:
            translation_work.append(
                _translation_work_row(
                    source=source_key,
                    context=context,
                    seed=wikispecies_seed,
                    target_language="*",
                    work_key=wikispecies_work_key,
                    provider_config_hash=config_hash,
                    status="budget_exhausted",
                    started_at=started_at,
                    request_count=0,
                )
            )
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
            request_count_total += 1
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
        _append_wikimedia_assertions(
            assertions=assertions,
            seen_assertions=seen_assertions,
            links=links,
            context=context,
            expected_wikidata_items=expected_wikidata_items,
            skip_names={seed.source_name, context.accepted_scientific_name},
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
    mymemory_budget: MyMemoryMonthlyBudget,
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
            mymemory_reservation = mymemory_budget.reserve(seed)
            if not mymemory_reservation.allowed:
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
                        budget_exhausted_reason=mymemory_reservation.reason,
                    )
                )
                break
            if not budget.reserve(1):
                mymemory_budget.release(mymemory_reservation)
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
                        budget_exhausted_reason="translation_daily_request_limit",
                    )
                )
                break
            try:
                translated_names, request_count, response_byte_count = _call_mymemory_provider(
                    provider,
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
                        input_word_count=mymemory_reservation.input_word_count,
                        bandwidth_reserved_byte_count=mymemory_reservation.bandwidth_reserved_byte_count,
                    )
                )
                continue
            mymemory_budget.record_response(response_byte_count)
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
                    input_word_count=mymemory_reservation.input_word_count,
                    response_byte_count=response_byte_count,
                    bandwidth_reserved_byte_count=mymemory_reservation.bandwidth_reserved_byte_count,
                )
            )
        if budget.exhausted or mymemory_budget.exhausted:
            break
    source_work = [_aggregate_source_work(source_key, context, request_count_total, status="complete")] if translation_work else []
    return _source_result([], candidates, snapshots, errors, translation_work, source_work, request_count_total)


def _mymemory_work_units(
    context: SpeciesTranslationContext,
    seeds: list[TranslationSeed],
    target_locales: tuple[str, ...],
    completed_work: set[str],
    config_hash: str,
) -> tuple[TranslationWorkUnit, ...]:
    source_key = "mymemory"
    units: list[TranslationWorkUnit] = []
    for seed in seeds:
        source_language = _api_language_code(seed.source_language or "en")
        for target_language in target_locales:
            target_api_language = _api_language_code(target_language)
            if target_api_language == source_language:
                continue
            work_key = _translation_work_key(
                source_key,
                context.accepted_taxon_key,
                seed.source_name,
                source_language,
                target_language,
                config_hash,
            )
            if work_key in completed_work:
                continue
            units.append(
                TranslationWorkUnit(
                    source=source_key,
                    context=context,
                    seed=seed,
                    target_language=target_language,
                    target_api_language=target_api_language,
                    work_key=work_key,
                    provider_config_hash=config_hash,
                )
            )
    return tuple(units)


def _partition_languages(target_locales: tuple[str, ...], shard_count: int) -> tuple[tuple[str, ...], ...]:
    shards: list[list[str]] = [[] for _ in range(max(1, shard_count))]
    for index, locale in enumerate(target_locales):
        shards[index % len(shards)].append(locale)
    return tuple(tuple(shard) for shard in shards if shard)


def _new_mymemory_provider(provider_spec: Any) -> Any:
    if provider_spec is None:
        return None
    if callable(provider_spec) and not hasattr(provider_spec, "translate"):
        return provider_spec()
    return provider_spec


def _call_mymemory_provider(
    provider: Any,
    *,
    source_name: str,
    source_language: str,
    target_language: str,
    max_candidates: int,
) -> tuple[list[str], int, int]:
    result = provider.translate(
        source_name=source_name,
        source_language=source_language,
        target_language=target_language,
        max_candidates=max_candidates,
    )
    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise TypeError("mymemory provider translate() must return (translations, request_count[, response_byte_count])")
    translations = [str(item) for item in (result[0] or [])]
    request_count = int(result[1] or 0)
    response_byte_count = int(result[2] or 0) if len(result) == 3 else 0
    return translations, request_count, response_byte_count


def _harvest_mymemory_units(
    units: tuple[TranslationWorkUnit, ...],
    *,
    provider_spec: Any,
    budget: TranslationRequestBudget,
    mymemory_budget: MyMemoryMonthlyBudget,
    max_candidates_per_name: int,
) -> tuple[TranslationBatch, ...]:
    provider = _new_mymemory_provider(provider_spec)
    batches: list[TranslationBatch] = []
    for unit in units:
        batches.append(
            _harvest_single_mymemory_unit(
                unit,
                provider=provider,
                budget=budget,
                mymemory_budget=mymemory_budget,
                max_candidates_per_name=max_candidates_per_name,
            )
        )
        if budget.exhausted or mymemory_budget.exhausted:
            break
    return tuple(batches)


def _harvest_mymemory_units_parallel(
    units: tuple[TranslationWorkUnit, ...],
    *,
    target_locales: tuple[str, ...],
    language_shards: int,
    max_workers: int,
    provider_spec: Any,
    budget: TranslationRequestBudget,
    mymemory_budget: MyMemoryMonthlyBudget,
    max_candidates_per_name: int,
) -> tuple[TranslationBatch, ...]:
    shards = _partition_languages(target_locales, language_shards)
    unit_groups: list[tuple[TranslationWorkUnit, ...]] = []
    for shard in shards:
        shard_languages = set(shard)
        shard_units = tuple(unit for unit in units if unit.target_language in shard_languages)
        if shard_units:
            unit_groups.append(shard_units)
    if not unit_groups:
        return ()
    worker_count = min(max(1, max_workers), len(unit_groups))
    batches: list[TranslationBatch] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for shard_batches in executor.map(
            lambda shard_units: _harvest_mymemory_units(
                shard_units,
                provider_spec=provider_spec,
                budget=budget,
                mymemory_budget=mymemory_budget,
                max_candidates_per_name=max_candidates_per_name,
            ),
            unit_groups,
        ):
            batches.extend(shard_batches)
    return tuple(batches)


def _batch_from_source_result(result: dict[str, Any]) -> TranslationBatch:
    return TranslationBatch(
        name_assertions=tuple(result.get("name_assertions") or ()),
        translation_candidates=tuple(result.get("translation_candidates") or ()),
        source_snapshots=tuple(result.get("source_snapshots") or ()),
        errors=tuple(result.get("errors") or ()),
        translation_work=tuple(result.get("translation_work") or ()),
        source_work=tuple(result.get("source_work") or ()),
        request_count=int(result.get("request_count") or 0),
    )


def _harvest_single_mymemory_unit(
    unit: TranslationWorkUnit,
    *,
    provider: Any,
    budget: TranslationRequestBudget,
    mymemory_budget: MyMemoryMonthlyBudget,
    max_candidates_per_name: int,
) -> TranslationBatch:
    display_source = "MyMemory"
    candidates: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    translation_work: list[dict[str, Any]] = []
    request_count_total = 0
    if provider is None:
        errors.append(_source_error(display_source, unit.context, "missing_client", retryable=False))
        return TranslationBatch(errors=tuple(errors))
    started_at = datetime.now(UTC).isoformat()
    mymemory_reservation = mymemory_budget.reserve(unit.seed)
    if not mymemory_reservation.allowed:
        translation_work.append(
            _translation_work_row(
                source=unit.source,
                context=unit.context,
                seed=unit.seed,
                target_language=unit.target_language,
                work_key=unit.work_key,
                provider_config_hash=unit.provider_config_hash,
                status="budget_exhausted",
                started_at=started_at,
                request_count=0,
                budget_exhausted_reason=mymemory_reservation.reason,
            )
        )
        return TranslationBatch(
            translation_work=tuple(translation_work),
            source_work=(_aggregate_source_work(unit.source, unit.context, request_count_total, status="budget_exhausted"),),
        )
    if not budget.reserve(1):
        mymemory_budget.release(mymemory_reservation)
        translation_work.append(
            _translation_work_row(
                source=unit.source,
                context=unit.context,
                seed=unit.seed,
                target_language=unit.target_language,
                work_key=unit.work_key,
                provider_config_hash=unit.provider_config_hash,
                status="budget_exhausted",
                started_at=started_at,
                request_count=0,
                budget_exhausted_reason="translation_daily_request_limit",
            )
        )
        return TranslationBatch(
            translation_work=tuple(translation_work),
            source_work=(_aggregate_source_work(unit.source, unit.context, request_count_total, status="budget_exhausted"),),
        )
    source_language = _api_language_code(unit.seed.source_language or "en")
    try:
        translated_names, request_count, response_byte_count = _call_mymemory_provider(
            provider,
            source_name=unit.seed.source_name,
            source_language=source_language,
            target_language=unit.target_api_language,
            max_candidates=max_candidates_per_name,
        )
    except Exception as exc:  # noqa: BLE001 - source errors are recorded and the registry build continues.
        error_class = type(exc).__name__
        errors.append(_source_error(display_source, unit.context, error_class, retryable=_is_retryable_error(error_class)))
        translation_work.append(
            _translation_work_row(
                source=unit.source,
                context=unit.context,
                seed=unit.seed,
                target_language=unit.target_language,
                work_key=unit.work_key,
                provider_config_hash=unit.provider_config_hash,
                status="error",
                started_at=started_at,
                request_count=1,
                error_class=error_class,
                retryable=_is_retryable_error(error_class),
                input_word_count=mymemory_reservation.input_word_count,
                bandwidth_reserved_byte_count=mymemory_reservation.bandwidth_reserved_byte_count,
            )
        )
        return TranslationBatch(
            errors=tuple(errors),
            translation_work=tuple(translation_work),
            source_work=(_aggregate_source_work(unit.source, unit.context, 1, status="error", error_class=error_class),),
            request_count=1,
        )
    mymemory_budget.record_response(response_byte_count)
    request_count_total += request_count
    for translated_name in translated_names:
        candidates.append(
            generated_translation_candidate(
                source=display_source,
                source_language=normalize_language_code(source_language),
                target_language=parse_language_tag(unit.target_language).bcp47,
                source_name=unit.seed.source_name,
                translated_name=translated_name,
                accepted_taxon_key=unit.context.accepted_taxon_key,
                source_record_id=_translation_record_id(
                    display_source,
                    unit.context.accepted_taxon_key,
                    source_language,
                    unit.target_language,
                    unit.seed.source_name,
                    translated_name,
                ),
                source_kind="dictionary",
            ).to_row()
        )
    snapshots.append(
        _source_snapshot(
            display_source,
            MYMEMORY_SOURCE_VERSION,
            source_path=f"{source_language}|{unit.target_api_language};target={unit.target_language}:{unit.seed.source_name}",
            payload={"translations": translated_names},
        )
    )
    translation_work.append(
        _translation_work_row(
            source=unit.source,
            context=unit.context,
            seed=unit.seed,
            target_language=unit.target_language,
            work_key=unit.work_key,
            provider_config_hash=unit.provider_config_hash,
            status="complete",
            started_at=started_at,
            request_count=request_count,
            input_word_count=mymemory_reservation.input_word_count,
            response_byte_count=response_byte_count,
            bandwidth_reserved_byte_count=mymemory_reservation.bandwidth_reserved_byte_count,
        )
    )
    return TranslationBatch(
        translation_candidates=tuple(candidates),
        source_snapshots=tuple(snapshots),
        translation_work=tuple(translation_work),
        source_work=(_aggregate_source_work(unit.source, unit.context, request_count_total, status="complete"),),
        request_count=request_count_total,
    )


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.write_parquet(tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _translation_output_frames(
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
) -> dict[str, pl.DataFrame]:
    return {
        "assertions": _name_assertions_frame(
            _deduplicate_dicts(
                [*existing_assertions, *new_assertions],
                keys=("accepted_taxon_key", "source", "source_record_id", "display_name"),
            )
        ),
        "candidates": translation_candidates_frame([*existing_candidates, *new_candidates]),
        "snapshots": _source_snapshots_frame(
            _deduplicate_dicts(
                [*existing_snapshots, *new_snapshots],
                keys=("source", "source_version", "source_path", "source_response_hash"),
            )
        ),
        "errors": _source_errors_frame([*existing_errors, *new_errors]),
        "source_work": _source_work_frame(_aggregate_source_work_rows([*existing_source_work, *new_source_work])),
        "translation_work": _translation_work_frame(_deduplicate_latest_dicts([*existing_translation_work, *new_translation_work], keys=("work_key",))),
    }


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
    frames = _translation_output_frames(
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
    _write_parquet_atomic(frames["assertions"], output / SOURCE_ASSERTIONS_FILE)
    _write_parquet_atomic(frames["candidates"], output / TRANSLATION_CANDIDATES_FILE)
    _write_parquet_atomic(frames["snapshots"], output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
    _write_parquet_atomic(frames["errors"], output / SOURCE_ERRORS_FILE)
    _write_parquet_atomic(frames["source_work"], output / SOURCE_WORK_LEDGER_FILE)
    _write_parquet_atomic(frames["translation_work"], output / TRANSLATION_WORK_LEDGER_FILE)


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
    manifest = _read_json_or_empty(manifest_path)
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
    _write_json_atomic(manifest, manifest_path)
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
        "mymemory": lambda: MyMemoryTranslationProvider(
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


def _append_wikimedia_assertions(
    *,
    assertions: list[dict[str, Any]],
    seen_assertions: set[tuple[str, str]],
    links: list[WikimediaLanglink],
    context: SpeciesTranslationContext,
    expected_wikidata_items: set[str],
    skip_names: set[str],
) -> None:
    skip_name_keys = {normalize_name_key(name) for name in skip_names if name}
    for link in links:
        title = " ".join(link.title.split())
        if not title or normalize_name_key(title) in skip_name_keys:
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
                "source": "Wikimedia",
                "source_record_id": f"wikimedia:{link.page_id}:{link.wikidata_item}:{link.language}:{title}",
                "source_taxon_id": link.wikidata_item if bound_to_same_taxon else "",
                "lineage_check": "accepted_taxon_key" if bound_to_same_taxon else "",
                "trust_tier": "T3" if bound_to_same_taxon else "T4",
                "precision_tier": "medium",
                "confidence": "high" if bound_to_same_taxon else "low",
                "enabled": bound_to_same_taxon,
                "review_state": "accepted" if bound_to_same_taxon else "candidate",
                "disabled_reason": disabled_reason,
            }
        )


def _wikispecies_taxonbar_item(wikitext: str) -> str:
    match = re.search(r"\{\{\s*Taxonbar\b[^{}]*\|\s*from\s*=\s*(Q\d+)", wikitext, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _wikispecies_vernacular_links(
    wikitext: str,
    *,
    target_api_codes: set[str],
    page_id: str,
    page_title: str,
    wikidata_item: str,
) -> list[WikimediaLanglink]:
    match = re.search(r"\{\{\s*VN\b(?P<body>.*?)\n\s*\}\}", wikitext, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    links: list[WikimediaLanglink] = []
    seen: set[tuple[str, str]] = set()
    for line in match.group("body").splitlines():
        row = re.match(r"\s*\|\s*(?P<language>[A-Za-z][A-Za-z0-9-]*)\s*=\s*(?P<names>.+?)\s*$", line)
        if row is None:
            continue
        language = row.group("language").strip()
        if language not in target_api_codes:
            continue
        for raw_name in re.split(r"[,;]", row.group("names")):
            name = " ".join(raw_name.split())
            if not name:
                continue
            key = (language, normalize_name_key(name))
            if key in seen:
                continue
            seen.add(key)
            links.append(WikimediaLanglink(language=language, title=name, page_id=page_id, page_title=page_title, wikidata_item=wikidata_item))
    return links


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
        "input_word_count": pl.Int64,
        "response_byte_count": pl.Int64,
        "bandwidth_reserved_byte_count": pl.Int64,
        "budget_exhausted_reason": pl.String,
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
            "input_word_count": int(row.get("input_word_count") or 0),
            "response_byte_count": int(row.get("response_byte_count") or 0),
            "bandwidth_reserved_byte_count": int(row.get("bandwidth_reserved_byte_count") or 0),
            "budget_exhausted_reason": str(row.get("budget_exhausted_reason") or ""),
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
    input_word_count: int = 0,
    response_byte_count: int = 0,
    bandwidth_reserved_byte_count: int = 0,
    budget_exhausted_reason: str = "",
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
        input_word_count=input_word_count,
        response_byte_count=response_byte_count,
        bandwidth_reserved_byte_count=bandwidth_reserved_byte_count,
        budget_exhausted_reason=budget_exhausted_reason,
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


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_counter_key(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in {"wikimedia", "mymemory"}:
        return text
    if text == "my memory":
        return "mymemory"
    return {"wikimedia": "wikimedia", "mymemory": "mymemory", "my memory": "mymemory"}.get(text, text)


def _batch_source_key(batch: TranslationBatch) -> str:
    for rows in (batch.translation_work, batch.source_work, batch.translation_candidates, batch.name_assertions, batch.errors):
        for row in rows:
            source = _source_counter_key(row.get("source"))
            if source:
                return source
    return ""


def _frame_source_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts: Counter[str] = Counter()
    for value in frame.select(column).to_series().to_list():
        key = _source_counter_key(value)
        if key:
            counts[key] += 1
    return dict(sorted(counts.items()))


def _translation_request_counts_from_source_work_frame(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or not {"source", "request_count"}.issubset(frame.columns):
        return {}
    counts: Counter[str] = Counter()
    for row in frame.select(["source", "request_count"]).to_dicts():
        source = _source_counter_key(row.get("source"))
        if source:
            counts[source] += int(row.get("request_count") or 0)
    return dict(sorted(counts.items()))


def _aggregate_source_work_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        key = (source, accepted_taxon_key)
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["request_count"] = int(row.get("request_count") or 0)
            grouped[key]["attempts"] = int(row.get("attempts") or 0)
            continue
        current = grouped[key]
        current["request_count"] = int(current.get("request_count") or 0) + int(row.get("request_count") or 0)
        current["attempts"] = int(current.get("attempts") or 0) + int(row.get("attempts") or 0)
        if str(row.get("started_at") or "") and (
            not str(current.get("started_at") or "") or str(row.get("started_at") or "") < str(current.get("started_at") or "")
        ):
            current["started_at"] = row.get("started_at")
        if str(row.get("finished_at") or "") >= str(current.get("finished_at") or ""):
            for field in ("accepted_scientific_name", "status", "finished_at", "error_class", "request_day"):
                current[field] = row.get(field, current.get(field))
    return list(grouped.values())


def _append_translation(candidates: list[str], value: object, *, source_name: str) -> None:
    text = " ".join(str(value or "").split())
    if not text:
        return
    if normalize_name_key(text) == normalize_name_key(source_name):
        return
    if normalize_name_key(text) in {normalize_name_key(existing) for existing in candidates}:
        return
    candidates.append(text)


def _input_word_count(value: object) -> int:
    return len(str(value or "").split())


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
