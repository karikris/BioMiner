"""Source-bound names for the image-bearing species inventory.

These adapters collect provider assertions, never machine translations. Transport
is injectable so the run coordinator owns request budgets and rate limiting.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any, Mapping, Iterable
from urllib.parse import quote, urlencode

from biominer.registry.enrichment import SpeciesContext
from biominer.registry.enrichment_sources import HTTPGet, _external_link, _json_get, _name_assertion
from biominer.registry.normalize import normalize_name_key, parse_language_tag


GBIF_BASE_URL = "https://api.gbif.org/v1"
ALA_BASE_URL = "https://api.ala.org.au/species"
COMMONS_BASE_URL = "https://commons.wikimedia.org"
SOURCE_DOCUMENTATION = {
    "gbif_vernacular": "https://techdocs.gbif.org/en/openapi/v1/species",
    "ala": "https://docs.ala.org.au/openapi/specs/bie-index.json",
    "commons": "https://commons.wikimedia.org/wiki/Template:VN",
}


class ProviderTaxonMismatch(ValueError):
    """A provider response cannot be bound to the requested species."""


class ProviderPaginationLimit(ValueError):
    """A configured bound would silently truncate provider evidence."""


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _same_name(value: object, expected: str) -> bool:
    return bool(_text(value)) and normalize_name_key(value) == normalize_name_key(expected)


def _family_matches(value: object, context: SpeciesContext) -> bool:
    return not value or not context.family or _same_name(value, context.family)


def _snapshot(source: str, endpoint: str, payload: dict[str, Any]) -> dict[str, str]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source": source,
        "source_version": "image-species-names-v1",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_path": endpoint,
        "source_response_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "licence": "",
    }


def _result(source: str) -> dict[str, Any]:
    return {"source": source, "name_assertions": [], "external_links": [], "source_snapshots": [], "request_count": 0}


def _record(result: dict[str, Any], endpoint: str, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("provider response must be an object")
    if payload.get("error"):
        raise ValueError("provider returned an API error")
    result["request_count"] += 1
    result["source_snapshots"].append(_snapshot(result["source"], endpoint, payload))


def _assertion(context: SpeciesContext, *, source: str, source_id: str, row_id: str,
               name: str, language: str, region: str = "", scientific: bool = False,
               bound: bool = True) -> dict[str, Any]:
    tag = parse_language_tag(language)
    enabled = bound and bool(tag.language)
    reason = "" if enabled else ("missing_language" if bound else "commons_taxon_binding_unconfirmed")
    row = _name_assertion(
        context, name, source=source, source_record_id=row_id,
        source_taxon_id=source_id if bound else "", language=tag.bcp47,
        script=tag.script or ("Latn" if scientific else ""),
        name_class="scientific_synonym" if scientific else "vernacular",
        trust_tier="T2" if source == "ALA" else ("T1" if source == "GBIF" else "T3"),
        enabled=enabled, review_state="accepted" if enabled else "candidate",
        disabled_reason=reason, lineage_check="accepted_taxon_key" if bound else "",
    )
    row["region"] = region or tag.region
    row["source_binding_evidence"] = "exact_species_identity" if bound else ""
    return row


class GBIFImageNamesClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5,
                 page_size: int = 1000, max_pages: int = 100,
                 resolve_legacy_by_name: bool = False) -> None:
        if not 1 <= page_size <= 1000 or max_pages <= 0:
            raise ValueError("invalid GBIF pagination bounds")
        self._http_get = http_get or _json_get(GBIF_BASE_URL, max_retries=max_retries)
        self.page_size, self.max_pages = page_size, max_pages
        self.resolve_legacy_by_name = resolve_legacy_by_name

    def enrich_species(self, context: SpeciesContext) -> dict[str, Any]:
        match = re.fullmatch(r"gbif:([A-Za-z0-9][A-Za-z0-9_.-]*)", context.accepted_taxon_key)
        if not match:
            raise ProviderTaxonMismatch("a source-bound taxon key is required")
        source_key = match.group(1)
        result = _result("GBIF")
        resolution = "species_key+scientific_name"
        if source_key.isdigit() and (set(source_key) == {"0"} or (
                not self.resolve_legacy_by_name and not re.fullmatch(r"[1-9][0-9]*", source_key))):
            raise ProviderTaxonMismatch("a legacy numeric key must be canonical and source keys must be nonzero")
        if source_key.isdigit() and not self.resolve_legacy_by_name:
            key = source_key
        else:
            # Current GBIF occurrence exports can carry opaque CoL XR keys.
            # Even digit-only CoL keys are not legacy Backbone usage keys.
            # Cohort callers explicitly resolve every key through this bridge;
            # the default retains compatibility for known legacy-only callers.
            params = {"name": context.accepted_scientific_name, "rank": "SPECIES",
                      "family": context.family, "strict": "true"}
            matched = self._http_get("/species/match", params)
            _record(result, GBIF_BASE_URL + "/species/match?" + urlencode(params), matched)
            key = str(matched.get("usageKey") or "")
            accepted = matched.get("acceptedUsageKey")
            status = str(matched.get("status") or matched.get("taxonomicStatus") or "").upper()
            if (not re.fullmatch(r"[1-9][0-9]*", key)
                    or str(matched.get("matchType") or "").upper() != "EXACT"
                    or matched.get("confidence") not in (99, 100)
                    or str(matched.get("rank") or "").upper() != "SPECIES"
                    or not _same_name(matched.get("canonicalName"), context.accepted_scientific_name)
                    or not context.family or not _same_name(matched.get("family"), context.family)
                    or status != "ACCEPTED"
                    or (accepted is not None and str(accepted) != key)):
                raise ProviderTaxonMismatch("source key has no exact accepted numeric GBIF match")
            resolution = ("source_taxon_key" if source_key.isdigit() else "source_opaque_key") + "+exact_scientific_name+family+numeric_usage"
        path = f"/species/{key}"
        taxon = self._http_get(path, {})
        _record(result, GBIF_BASE_URL + path, taxon)
        if (str(taxon.get("key")) != key or str(taxon.get("rank") or "").upper() != "SPECIES"
                or (taxon.get("acceptedKey") is not None and str(taxon["acceptedKey"]) != key)
                or not _same_name(taxon.get("canonicalName") or taxon.get("scientificName"), context.accepted_scientific_name)
                or not _family_matches(taxon.get("family"), context)):
            raise ProviderTaxonMismatch("GBIF species identity differs from the inventory")
        result["external_links"].append(_external_link(context, source="GBIF", source_taxon_id=key, match_method=resolution))
        for kind in ("vernacularNames", "synonyms"):
            path = f"/species/{key}/{kind}"
            for page in range(self.max_pages):
                offset = page * self.page_size
                payload = self._http_get(path, {"limit": self.page_size, "offset": offset})
                _record(result, GBIF_BASE_URL + path + f"?offset={offset}&limit={self.page_size}", payload)
                rows = payload.get("results")
                if not isinstance(rows, list):
                    raise ValueError("GBIF page is missing results")
                for ordinal, row in enumerate(rows):
                    if not isinstance(row, dict):
                        raise ValueError("GBIF result row must be an object")
                    scientific = kind == "synonyms"
                    if scientific:
                        if str(row.get("acceptedKey") or "") != key:
                            continue
                        name = _text(row.get("canonicalName") or row.get("scientificName"))
                    else:
                        if row.get("taxonKey") is not None and str(row["taxonKey"]) != key:
                            continue
                        name = _text(row.get("vernacularName"))
                    if not name:
                        continue
                    # Keep provider row position and upstream source identity;
                    # identical spelling from two providers is two assertions.
                    identity = f"gbif:{key}:{kind}:{offset + ordinal}:{row.get('sourceTaxonKey', row.get('key', ''))}"
                    result["name_assertions"].append(_assertion(
                        context, source="GBIF", source_id=key, row_id=identity,
                        name=name, language="la" if scientific else _text(row.get("language")),
                        region=_text(row.get("country") or row.get("area")), scientific=scientific,
                    ))
                ended = payload.get("endOfRecords")
                if ended is True or (ended is None and len(rows) < self.page_size):
                    break
                if not rows:
                    raise ValueError("GBIF pagination made no progress")
            else:
                raise ProviderPaginationLimit("GBIF pagination bound reached")
        return result


class ALAImageNamesClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5,
                 page_size: int = 100, max_pages: int = 10) -> None:
        if page_size <= 0 or max_pages <= 0:
            raise ValueError("invalid ALA pagination bounds")
        self._http_get = http_get or _json_get(ALA_BASE_URL, max_retries=max_retries)
        self.page_size, self.max_pages = page_size, max_pages

    def enrich_species(self, context: SpeciesContext) -> dict[str, Any]:
        result = _result("ALA")
        matches: dict[str, dict[str, Any]] = {}
        for page in range(self.max_pages):
            start = page * self.page_size
            payload = self._http_get("/search", {"q": context.accepted_scientific_name, "start": start, "pageSize": self.page_size})
            _record(result, ALA_BASE_URL + "/search?" + urlencode({"q": context.accepted_scientific_name, "start": start, "pageSize": self.page_size}), payload)
            search = payload.get("searchResults")
            if not isinstance(search, dict) or not isinstance(search.get("results"), list):
                raise ValueError("ALA search response is missing searchResults.results")
            rows = search["results"]
            for row in rows:
                if (isinstance(row, dict) and _same_name(row.get("scientificName") or row.get("nameString"), context.accepted_scientific_name)
                        and str(row.get("rank") or row.get("rankString") or "").upper() == "SPECIES"
                        and _family_matches(row.get("family"), context)):
                    guid = _text(row.get("guid"))
                    if guid:
                        matches[guid] = row
            total = search.get("totalRecords")
            if (isinstance(total, int) and start + len(rows) >= total) or (total is None and len(rows) < self.page_size):
                break
            if not rows:
                raise ValueError("ALA pagination made no progress")
        else:
            raise ProviderPaginationLimit("ALA search pagination bound reached")
        if not matches:
            result["status"] = "no_match"
            return result
        if len(matches) != 1:
            raise ProviderTaxonMismatch("ALA has multiple exact species concepts")
        guid = next(iter(matches))
        path = "/species/" + quote(guid, safe="")
        payload = self._http_get(path, {})
        _record(result, ALA_BASE_URL + path, payload)
        taxon = payload.get("taxonConcept") or {}
        classification = payload.get("classification") or {}
        if (str(taxon.get("guid") or "") != guid
                or not _same_name(taxon.get("nameString") or taxon.get("scientificName"), context.accepted_scientific_name)
                or str(taxon.get("rankString") or taxon.get("rank") or "").upper() != "SPECIES"
                # Exact spelling alone does not distinguish homonyms. The
                # resolved profile must supply the inventory's family proof.
                or not context.family or not _same_name(classification.get("family"), context.family)):
            raise ProviderTaxonMismatch("ALA profile differs from the exact search match")
        result["external_links"].append(_external_link(context, source="ALA", source_taxon_id=guid, match_method="exact_scientific_name+profile"))
        for kind in ("commonNames", "synonyms"):
            for ordinal, row in enumerate(payload.get(kind) or []):
                if not isinstance(row, dict):
                    raise ValueError("ALA name must be an object")
                name = _text(row.get("nameString") or row.get("scientificName"))
                if name:
                    scientific = kind == "synonyms"
                    result["name_assertions"].append(_assertion(
                        context, source="ALA", source_id=guid,
                        row_id=f"ala:{guid}:{kind}:{ordinal}:{row.get('nameGuid') or row.get('infoSourceURL') or ''}",
                        name=name, language="la" if scientific else _text(row.get("language")),
                        region=_text(row.get("country") or row.get("region")), scientific=scientific,
                    ))
        return result


class CommonsImageNamesClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5,
                 expected_wikidata_items: Mapping[str, Iterable[str]] | None = None) -> None:
        self._http_get = http_get or _json_get(COMMONS_BASE_URL, max_retries=max_retries)
        self.expected_wikidata_items = expected_wikidata_items if expected_wikidata_items is not None else {}

    def enrich_species(self, context: SpeciesContext) -> dict[str, Any]:
        result = _result("Wikimedia Commons")
        titles = (context.accepted_scientific_name, "Category:" + context.accepted_scientific_name)
        payload = self._http_get("/w/api.php", {
            "action": "query", "format": "json", "prop": "pageprops|revisions",
            "ppprop": "wikibase_item", "rvprop": "ids|content", "rvslots": "main",
            "titles": "|".join(titles),
        })
        _record(result, COMMONS_BASE_URL + "/w/api.php?" + urlencode({"action": "query", "titles": "|".join(titles), "prop": "pageprops|revisions"}), payload)
        pages = payload.get("query", {}).get("pages")
        if not isinstance(pages, dict):
            raise ValueError("Commons response is missing pages")
        expected = set(self.expected_wikidata_items.get(context.accepted_taxon_key, ()))
        for page in pages.values():
            if not isinstance(page, dict) or "missing" in page:
                continue
            if _text(page.get("title")) not in titles or page.get("ns") not in (0, 14):
                continue
            qid = _text((page.get("pageprops") or {}).get("wikibase_item"))
            bound = bool(re.fullmatch(r"Q[1-9][0-9]*", qid)) and qid in expected
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            revision = revisions[0]
            text = (revision.get("slots", {}).get("main", {}).get("*") or revision.get("*") or "")
            if not isinstance(text, str):
                raise ValueError("Commons revision content must be text")
            for template_index, match in enumerate(re.finditer(r"\{\{\s*VN\s*\|([^{}]*)\}\}", text, re.IGNORECASE)):
                entries = [part.partition("=") for part in match.group(1).split("|")]
                fields = {_text(key): _text(value) for key, sep, value in entries if sep}
                if fields.get("sciname") and not _same_name(fields["sciname"], context.accepted_scientific_name):
                    continue
                # Explicitly overridden VN belongs to a different item.
                override = fields.get("useWikidata", "")
                template_bound = bound and (not override.startswith("Q") or override == qid)
                if override.startswith("Q") and override != qid:
                    continue
                for language, name in fields.items():
                    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language) or language in {"nocat"}:
                        continue
                    if not name or any(char in name for char in "[]{}<>|"):
                        continue
                    row = _assertion(
                        context, source="Wikimedia Commons", source_id=qid,
                        row_id=f"commons:{page.get('pageid')}:{revision.get('revid')}:{template_index}:{language}",
                        name=name, language=language, bound=template_bound,
                    )
                    row["observed_source_taxon_id"] = qid
                    row["source_binding_status"] = "direct_qitem_match" if template_bound else "unconfirmed"
                    row["source_binding_evidence"] = "wikibase_item_exact" if template_bound else ""
                    row["expected_source_taxon_ids"] = sorted(expected)
                    result["name_assertions"].append(row)
            if bound:
                result["external_links"].append(_external_link(context, source="Wikimedia Commons", source_taxon_id=qid, match_method="exact_title+wikibase_item", lineage_check="wikibase_item_exact"))
        return result
