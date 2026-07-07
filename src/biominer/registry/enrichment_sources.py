from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections.abc import Callable
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
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
GBIF_VERNACULAR_SOURCE_VERSION = "gbif-vernacular-from-registry-v1"
GBIF_VERNACULAR_SOURCE_PATH = "registry:names.parquet"
TAXREF_BASE_URL = "https://taxref.mnhn.fr/taxref-web"
TAXREF_TAXA_SEARCH_PATH = "/api/taxa/search"
TAXREF_SOURCE_PATH = f"{TAXREF_BASE_URL}{TAXREF_TAXA_SEARCH_PATH}"
TAXREF_SOURCE_VERSION = "taxref-web-api-taxa-search"
WIKIDATA_WDQS_URL = "https://query.wikidata.org"
WIKIDATA_SOURCE_VERSION = "wikidata-wdqs-p225-p846-p1843-labels-aliases"
COL_NAME_USAGE_SEARCH_LIMIT = 1000
INATURALIST_TAXA_PER_PAGE = 200
TAXREF_SEARCH_ROWS_PER_TAXON = 50
STATIC_VERNACULAR_REQUIRED_FIELDS = (
    "source_key",
    "source_name",
    "source_version",
    "country_code",
    "admin1_code",
    "scientific_name",
    "source_taxon_id",
    "accepted_name_usage",
    "vernacular_name",
    "language",
    "script",
    "region",
    "rank",
    "licence",
    "source_url",
    "citation",
)
STATIC_VERNACULAR_CONFIG_REQUIRED_FIELDS = (
    "source_key",
    "source_name",
    "source_version",
    "snapshot_path",
    "country_code",
    "language",
    "script",
    "region",
    "trust_tier",
    "precision_tier",
    "source_url",
    "citation",
    "licence",
)
TAXREF_TERRITORY_FIELDS = (
    ("fr", "FR"),
    ("gf", "GF"),
    ("mar", "MAR"),
    ("gua", "GUA"),
    ("sm", "SM"),
    ("sb", "SB"),
    ("spm", "SPM"),
    ("epa", "EPA"),
    ("may", "MAY"),
    ("reu", "REU"),
    ("sa", "SA"),
    ("ta", "TA"),
    ("nc", "NC"),
    ("wf", "WF"),
    ("pf", "PF"),
    ("cli", "CLI"),
)


class GBIFVernacularClient:
    def enrich_registry(self, *, taxa_rows: list[dict[str, Any]], name_rows: list[dict[str, Any]]) -> dict[str, Any]:
        species_keys = {
            _full_gbif_key(row.get("accepted_taxon_key"))
            for row in taxa_rows
            if str(row.get("rank") or "") == "SPECIES" and row.get("accepted_taxon_key")
        }
        synonym_lookup, ambiguous_synonym_keys = _gbif_synonym_source_lookup(name_rows)
        candidate_rows = [
            row
            for row in name_rows
            if _is_gbif_source(row) and str(row.get("name_class") or "") in {"vernacular", "vernacular_alias"}
        ]
        coverage = {
            "rows_inspected": len(candidate_rows),
            "names_extracted": 0,
            "names_with_language": 0,
            "names_without_language": 0,
            "accepted_matches": 0,
            "synonym_matches": 0,
            "ambiguous_matches": 0,
            "duplicate_names": 0,
            "out_of_scope_rows": 0,
            "disabled_names": 0,
            "request_count": 0,
        }
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        seen_assertions: set[tuple[str, str, str, str, str]] = set()
        seen_links: set[tuple[str, str, str]] = set()

        for row in candidate_rows:
            display_name = str(row.get("display_name") or row.get("verbatim_name") or "").strip()
            if not display_name:
                continue
            row_taxon_key = _full_gbif_key(row.get("accepted_taxon_key"))
            source_taxon_id = _gbif_source_taxon_id(row) or row_taxon_key
            accepted_taxon_key = ""
            match_method = ""
            match_confidence = ""
            lineage_check = ""
            if row_taxon_key in species_keys:
                accepted_taxon_key = row_taxon_key
                match_method = "accepted_taxon_key"
                match_confidence = "high"
                lineage_check = "accepted_taxon_key"
                coverage["accepted_matches"] += 1
            elif source_taxon_id in species_keys:
                accepted_taxon_key = source_taxon_id
                match_method = "accepted_taxon_key"
                match_confidence = "high"
                lineage_check = "accepted_taxon_key"
                coverage["accepted_matches"] += 1
            elif source_taxon_id in ambiguous_synonym_keys or row_taxon_key in ambiguous_synonym_keys:
                coverage["ambiguous_matches"] += 1
                continue
            elif source_taxon_id in synonym_lookup:
                accepted_taxon_key = synonym_lookup[source_taxon_id]
                match_method = "scientific_synonym"
                match_confidence = "medium"
                lineage_check = "scientific_synonym"
                coverage["synonym_matches"] += 1
            elif row_taxon_key in synonym_lookup:
                source_taxon_id = row_taxon_key
                accepted_taxon_key = synonym_lookup[row_taxon_key]
                match_method = "scientific_synonym"
                match_confidence = "medium"
                lineage_check = "scientific_synonym"
                coverage["synonym_matches"] += 1
            else:
                coverage["out_of_scope_rows"] += 1
                continue

            language = str(row.get("language") or "").strip()
            script = str(row.get("script") or "").strip()
            region = str(row.get("region") or row.get("country") or "").strip()
            name_class = "vernacular_alias" if match_method == "scientific_synonym" else str(row.get("name_class") or "vernacular")
            dedupe_key = (accepted_taxon_key, normalize_name_key(display_name), language, region, name_class)
            if dedupe_key in seen_assertions:
                coverage["duplicate_names"] += 1
                continue
            seen_assertions.add(dedupe_key)

            enabled = bool(language)
            disabled_reason = "" if enabled else "missing_language"
            if language:
                coverage["names_with_language"] += 1
            else:
                coverage["names_without_language"] += 1
                coverage["disabled_names"] += 1
            coverage["names_extracted"] += 1

            source_record_id = str(row.get("source_record_id") or "").strip()
            if not source_record_id:
                source_record_id = f"gbif:vernacular:{source_taxon_id}:{display_name}"
            assertions.append(
                {
                    "accepted_taxon_key": accepted_taxon_key,
                    "verbatim_name": str(row.get("verbatim_name") or display_name),
                    "display_name": display_name,
                    "language": language,
                    "script": script,
                    "region": region,
                    "bbox": str(row.get("bbox") or ""),
                    "name_class": name_class,
                    "source": "GBIF",
                    "source_record_id": f"gbif_vernacular:{source_record_id}",
                    "source_taxon_id": source_taxon_id,
                    "lineage_check": lineage_check,
                    "trust_tier": "T3",
                    "precision_tier": str(row.get("precision_tier") or "medium"),
                    "confidence": match_confidence,
                    "enabled": enabled,
                    "review_state": "accepted" if enabled else "candidate",
                    "disabled_reason": disabled_reason,
                    "licence": str(row.get("licence") or "GBIF.org"),
                }
            )
            link_key = (accepted_taxon_key, source_taxon_id, match_method)
            if link_key not in seen_links:
                seen_links.add(link_key)
                links.append(
                    {
                        "accepted_taxon_key": accepted_taxon_key,
                        "source": "GBIF",
                        "source_taxon_id": source_taxon_id,
                        "match_method": match_method,
                        "match_confidence": match_confidence,
                        "lineage_check": lineage_check,
                    }
                )

        assertions.sort(key=lambda item: (str(item["display_name"]).casefold(), str(item["accepted_taxon_key"]), str(item["source_record_id"])))
        links.sort(key=lambda item: (str(item["source_taxon_id"]), str(item["accepted_taxon_key"]), str(item["match_method"])))
        return {
            "name_assertions": assertions,
            "external_links": links,
            "source_snapshots": [
                {
                    "source": "GBIF",
                    "source_version": GBIF_VERNACULAR_SOURCE_VERSION,
                    "retrieved_at": "",
                    "source_path": GBIF_VERNACULAR_SOURCE_PATH,
                    "source_response_hash": _payload_hash({"vernacular_name_rows": candidate_rows}),
                    "licence": "GBIF.org",
                }
            ],
            "coverage": coverage,
        }


class TAXREFFrenchClient:
    def __init__(
        self,
        *,
        http_get: HTTPGet | None = None,
        max_retries: int = 5,
        taxref_rows: list[dict[str, Any]] | None = None,
        source_path: str = TAXREF_SOURCE_PATH,
        source_version: str = TAXREF_SOURCE_VERSION,
    ) -> None:
        self._http_get = http_get or _json_get(TAXREF_BASE_URL, max_retries=max_retries)
        self._taxref_rows = taxref_rows
        self._source_path = source_path
        self._source_version = source_version

    def enrich_registry(self, *, taxa_rows: list[dict[str, Any]], name_rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows, request_count = self._load_rows(taxa_rows)
        accepted_lookup = _accepted_species_lookup(taxa_rows)
        synonym_lookup, ambiguous_synonym_keys = _scientific_synonym_lookup_with_ambiguity(name_rows)
        source_id_lookup = _source_taxon_id_lookup(name_rows, source="TAXREF")
        coverage = {
            "rows_fetched": len(rows),
            "vernacular_names_extracted": 0,
            "mapped_source_id_rows": 0,
            "mapped_accepted_name_rows": 0,
            "mapped_synonym_rows": 0,
            "out_of_scope_rows": 0,
            "ambiguous_synonym_rows": 0,
            "disabled_candidate_rows": 0,
            "rows_without_vernacular": 0,
            "territory_rows": 0,
            "request_count": request_count,
        }
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        seen_assertions: set[tuple[str, str, str, str]] = set()
        seen_links: set[tuple[str, str, str]] = set()

        for row in rows:
            vernacular_names = _taxref_vernacular_values(row)
            if not vernacular_names:
                coverage["rows_without_vernacular"] += 1
                continue
            scientific_name = _first_string(row, "lbNom", "nomComplet", "scientificName", "name").strip()
            source_taxon_id = _taxref_source_taxon_id(row)
            source_ref_id = _taxref_source_ref_id(row)
            name_key = normalize_name_key(scientific_name)
            accepted_taxon_key = ""
            match_method = ""
            match_confidence = ""
            lineage_check = ""
            if source_taxon_id and source_taxon_id in source_id_lookup:
                accepted_taxon_key = source_id_lookup[source_taxon_id]
                match_method = "source_taxon_id"
                match_confidence = "high"
                lineage_check = "source_taxon_id"
                coverage["mapped_source_id_rows"] += 1
            elif source_ref_id and source_ref_id in source_id_lookup:
                accepted_taxon_key = source_id_lookup[source_ref_id]
                match_method = "source_taxon_id"
                match_confidence = "high"
                lineage_check = "source_taxon_id"
                coverage["mapped_source_id_rows"] += 1
            elif name_key in accepted_lookup:
                accepted_taxon_key = accepted_lookup[name_key]
                match_method = "scientific_name"
                match_confidence = "high"
                lineage_check = "accepted_scientific_name"
                coverage["mapped_accepted_name_rows"] += 1
            elif name_key in ambiguous_synonym_keys:
                coverage["ambiguous_synonym_rows"] += 1
                continue
            elif name_key in synonym_lookup:
                accepted_taxon_key = synonym_lookup[name_key]
                match_method = "scientific_synonym"
                match_confidence = "medium"
                lineage_check = "scientific_synonym"
                coverage["mapped_synonym_rows"] += 1
            else:
                coverage["out_of_scope_rows"] += 1
                continue

            region = _taxref_region(row)
            if region:
                coverage["territory_rows"] += 1
            enabled = bool(region)
            disabled_reason = "" if enabled else "missing_taxref_territory"
            for vernacular_name in vernacular_names:
                dedupe_key = (accepted_taxon_key, normalize_name_key(vernacular_name), region, source_taxon_id)
                if dedupe_key in seen_assertions:
                    continue
                seen_assertions.add(dedupe_key)
                if not enabled:
                    coverage["disabled_candidate_rows"] += 1
                coverage["vernacular_names_extracted"] += 1
                assertions.append(
                    {
                        "accepted_taxon_key": accepted_taxon_key,
                        "verbatim_name": vernacular_name,
                        "display_name": vernacular_name,
                        "language": "fra",
                        "script": "Latn",
                        "region": region,
                        "bbox": "",
                        "name_class": "vernacular",
                        "source": "TAXREF",
                        "source_record_id": f"taxref:{source_taxon_id}:vernacular:{vernacular_name}",
                        "source_taxon_id": source_taxon_id,
                        "lineage_check": lineage_check,
                        "trust_tier": "T2",
                        "precision_tier": "high",
                        "confidence": match_confidence,
                        "enabled": enabled,
                        "review_state": "accepted" if enabled else "candidate",
                        "disabled_reason": disabled_reason,
                    }
                )
            link_key = (accepted_taxon_key, source_taxon_id, match_method)
            if source_taxon_id and link_key not in seen_links:
                seen_links.add(link_key)
                links.append(
                    {
                        "accepted_taxon_key": accepted_taxon_key,
                        "source": "TAXREF",
                        "source_taxon_id": source_taxon_id,
                        "match_method": match_method,
                        "match_confidence": match_confidence,
                        "lineage_check": lineage_check,
                    }
                )

        assertions.sort(key=lambda item: (str(item["display_name"]).casefold(), str(item["accepted_taxon_key"]), str(item["source_taxon_id"])))
        links.sort(key=lambda item: (str(item["source_taxon_id"]), str(item["accepted_taxon_key"]), str(item["match_method"])))
        return {
            "name_assertions": assertions,
            "external_links": links,
            "source_snapshots": [
                {
                    "source": "TAXREF",
                    "source_version": self._source_version,
                    "retrieved_at": "",
                    "source_path": self._source_path,
                    "source_response_hash": _payload_hash({"taxref_rows": rows}),
                    "licence": "",
                }
            ],
            "coverage": coverage,
        }

    def _load_rows(self, taxa_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        if self._taxref_rows is not None:
            return [dict(row) for row in self._taxref_rows], 0
        rows: list[dict[str, Any]] = []
        request_count = 0
        seen_cd_nom: set[str] = set()
        species_names = sorted(
            {
                str(row.get("scientific_name") or "").strip()
                for row in taxa_rows
                if str(row.get("rank") or "") == "SPECIES" and row.get("scientific_name")
            }
        )
        for scientific_name in species_names:
            payload = self._http_get(
                TAXREF_TAXA_SEARCH_PATH,
                {"nomComplet": scientific_name, "rang.rang": "ES", "nbRows": TAXREF_SEARCH_ROWS_PER_TAXON},
            )
            request_count += 1
            payload_rows = _result_rows(payload) if isinstance(payload, dict) else [item for item in payload if isinstance(item, dict)]
            for row in payload_rows:
                cd_nom = _taxref_source_taxon_id(row)
                if cd_nom and cd_nom in seen_cd_nom:
                    continue
                if cd_nom:
                    seen_cd_nom.add(cd_nom)
                rows.append(row)
        return rows, request_count


class StaticVernacularSourceClient:
    def __init__(
        self,
        *,
        source_key: str,
        source_name: str,
        source_version: str,
        snapshot_path: str | Path,
        country_code: str = "",
        language: str = "",
        script: str = "",
        region: str = "",
        trust_tier: str = "T2",
        precision_tier: str = "high",
        source_url: str = "",
        citation: str = "",
        licence: str = "",
    ) -> None:
        self.source_key = source_key
        self.source_name = source_name
        self.source_version = source_version
        self.snapshot_path = Path(snapshot_path)
        self.country_code = country_code
        self.language = language
        self.script = script
        self.region = region
        self.trust_tier = trust_tier
        self.precision_tier = precision_tier
        self.source_url = source_url
        self.citation = citation
        self.licence = licence

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> StaticVernacularSourceClient:
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Static vernacular source config must be a JSON object: {path}")
        missing = [field for field in STATIC_VERNACULAR_CONFIG_REQUIRED_FIELDS if field not in config]
        if missing:
            raise ValueError(f"Static vernacular source config missing required fields: {', '.join(missing)}")
        snapshot_path = _resolve_static_snapshot_path(path, str(config.get("snapshot_path") or ""))
        return cls(
            source_key=str(config.get("source_key") or "").strip(),
            source_name=str(config.get("source_name") or "").strip(),
            source_version=str(config.get("source_version") or "").strip(),
            snapshot_path=snapshot_path,
            country_code=str(config.get("country_code") or "").strip(),
            language=str(config.get("language") or "").strip(),
            script=str(config.get("script") or "").strip(),
            region=str(config.get("region") or "").strip(),
            trust_tier=str(config.get("trust_tier") or "T2").strip(),
            precision_tier=str(config.get("precision_tier") or "high").strip(),
            source_url=str(config.get("source_url") or "").strip(),
            citation=str(config.get("citation") or "").strip(),
            licence=str(config.get("licence") or "").strip(),
        )

    def enrich_registry(self, *, taxa_rows: list[dict[str, Any]], name_rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = self._load_rows()
        accepted_lookup = _accepted_species_lookup(taxa_rows)
        synonym_lookup, ambiguous_synonym_keys = _scientific_synonym_lookup_with_ambiguity(name_rows)
        source_id_lookup = _source_taxon_id_lookup(name_rows, source=self.source_name)
        coverage = {
            "rows_read": len(rows),
            "rows_with_vernacular": 0,
            "name_assertions": 0,
            "mapped_source_id_rows": 0,
            "mapped_accepted_name_rows": 0,
            "mapped_synonym_rows": 0,
            "out_of_scope_rows": 0,
            "ambiguous_synonym_rows": 0,
            "duplicate_rows": 0,
            "rows_without_vernacular": 0,
            "missing_source_name_rows": 0,
            "request_count": 0,
        }
        assertions: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        seen_assertions: set[tuple[str, str, str, str, str]] = set()
        seen_links: set[tuple[str, str, str]] = set()
        taxa_with_assertions: set[str] = set()

        for row in rows:
            if _static_value(row, "source_key", self.source_key) != self.source_key:
                coverage["out_of_scope_rows"] += 1
                continue
            vernacular_name = _static_value(row, "vernacular_name")
            if not vernacular_name:
                coverage["rows_without_vernacular"] += 1
                continue
            coverage["rows_with_vernacular"] += 1

            source_taxon_id = _static_value(row, "source_taxon_id")
            scientific_name = _static_value(row, "accepted_name_usage") or _static_value(row, "scientific_name")
            match_key = normalize_name_key(scientific_name)
            accepted_taxon_key = ""
            match_method = ""
            match_confidence = ""
            lineage_check = ""
            if source_taxon_id and source_taxon_id in source_id_lookup:
                accepted_taxon_key = source_id_lookup[source_taxon_id]
                match_method = "source_taxon_id"
                match_confidence = "high"
                lineage_check = "source_taxon_id"
                coverage["mapped_source_id_rows"] += 1
            elif match_key in accepted_lookup:
                accepted_taxon_key = accepted_lookup[match_key]
                match_method = "scientific_name"
                match_confidence = "high"
                lineage_check = "accepted_scientific_name"
                coverage["mapped_accepted_name_rows"] += 1
            elif match_key in ambiguous_synonym_keys:
                coverage["ambiguous_synonym_rows"] += 1
                continue
            elif match_key in synonym_lookup:
                accepted_taxon_key = synonym_lookup[match_key]
                match_method = "scientific_synonym"
                match_confidence = "medium"
                lineage_check = "scientific_synonym"
                coverage["mapped_synonym_rows"] += 1
            else:
                coverage["out_of_scope_rows"] += 1
                continue

            language = _static_value(row, "language", self.language)
            script = _static_value(row, "script", self.script)
            region = _static_value(row, "region", self.region)
            name_class = _static_value(row, "name_class") or ("vernacular_alias" if match_method == "scientific_synonym" else "vernacular")
            dedupe_key = (accepted_taxon_key, normalize_name_key(vernacular_name), language, region, name_class)
            if dedupe_key in seen_assertions:
                coverage["duplicate_rows"] += 1
                continue
            seen_assertions.add(dedupe_key)

            disabled_reasons = []
            if not language:
                disabled_reasons.append("missing_language")
            if not region:
                disabled_reasons.append("missing_region")
            enabled = not disabled_reasons
            source_record_id = _static_source_record_id(self.source_key, source_taxon_id, vernacular_name)
            licence = _static_value(row, "licence", self.licence)
            assertions.append(
                {
                    "accepted_taxon_key": accepted_taxon_key,
                    "verbatim_name": vernacular_name,
                    "display_name": vernacular_name,
                    "language": language,
                    "script": script,
                    "region": region,
                    "bbox": "",
                    "name_class": name_class,
                    "source": self.source_name,
                    "source_record_id": source_record_id,
                    "source_taxon_id": source_taxon_id,
                    "lineage_check": lineage_check,
                    "trust_tier": self.trust_tier,
                    "precision_tier": self.precision_tier,
                    "confidence": match_confidence,
                    "enabled": enabled,
                    "review_state": "accepted" if enabled else "candidate",
                    "disabled_reason": ";".join(disabled_reasons),
                    "licence": licence,
                }
            )
            coverage["name_assertions"] += 1
            taxa_with_assertions.add(accepted_taxon_key)

            link_key = (accepted_taxon_key, source_taxon_id, match_method)
            if source_taxon_id and link_key not in seen_links:
                seen_links.add(link_key)
                links.append(
                    {
                        "accepted_taxon_key": accepted_taxon_key,
                        "source": self.source_name,
                        "source_taxon_id": source_taxon_id,
                        "match_method": match_method,
                        "match_confidence": match_confidence,
                        "lineage_check": lineage_check,
                    }
                )

        species_keys = {
            str(row.get("accepted_taxon_key") or "")
            for row in taxa_rows
            if str(row.get("rank") or "") == "SPECIES" and row.get("accepted_taxon_key")
        }
        coverage["missing_source_name_rows"] = len(species_keys - taxa_with_assertions)
        assertions.sort(key=lambda item: (str(item["display_name"]).casefold(), str(item["accepted_taxon_key"]), str(item["source_record_id"])))
        links.sort(key=lambda item: (str(item["source_taxon_id"]), str(item["accepted_taxon_key"]), str(item["match_method"])))
        return {
            "name_assertions": assertions,
            "external_links": links,
            "source_snapshots": [
                {
                    "source": self.source_name,
                    "source_version": self.source_version,
                    "retrieved_at": "",
                    "source_path": str(self.snapshot_path),
                    "source_response_hash": _payload_hash(
                        {
                            "source_key": self.source_key,
                            "source_name": self.source_name,
                            "source_version": self.source_version,
                            "source_url": self.source_url,
                            "citation": self.citation,
                            "rows": rows,
                        }
                    ),
                    "licence": self.licence,
                    "source_url": self.source_url,
                    "citation": self.citation,
                }
            ],
            "coverage": coverage,
        }

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Static vernacular source snapshot not found: {self.snapshot_path}")
        with self.snapshot_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = [field for field in STATIC_VERNACULAR_REQUIRED_FIELDS if field not in fieldnames]
            if missing:
                raise ValueError(f"Static vernacular source CSV missing required fields: {', '.join(missing)}")
            return [
                {str(key): str(value or "").strip() for key, value in row.items() if key is not None}
                for row in reader
            ]


class CatalogueOfLifeClient:
    def __init__(self, *, http_get: HTTPGet | None = None, max_retries: int = 5) -> None:
        self._http_get = http_get or _json_get("https://api.checklistbank.org", max_retries=max_retries)

    def enrich_species(self, context: SpeciesContext) -> dict[str, list[dict[str, Any]]]:
        payload = self._http_get("/dataset/3/nameusage/search", {"q": context.accepted_scientific_name, "limit": COL_NAME_USAGE_SEARCH_LIMIT})
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
                "per_page": INATURALIST_TAXA_PER_PAGE,
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
                        source_taxon_id=qid,
                        lineage_check="accepted_taxon_key",
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


def _scientific_synonym_lookup_with_ambiguity(name_rows: list[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
    grouped: dict[str, set[str]] = {}
    for row in name_rows:
        if str(row.get("name_class") or "") != "scientific_synonym":
            continue
        key = normalize_name_key(row.get("display_name") or row.get("verbatim_name"))
        accepted_key = str(row.get("accepted_taxon_key") or "")
        if key and accepted_key:
            grouped.setdefault(key, set()).add(accepted_key)
    unambiguous = {key: next(iter(values)) for key, values in grouped.items() if len(values) == 1}
    ambiguous = {key for key, values in grouped.items() if len(values) > 1}
    return unambiguous, ambiguous


def _source_taxon_id_lookup(name_rows: list[dict[str, Any]], *, source: str) -> dict[str, str]:
    grouped: dict[str, set[str]] = {}
    expected_source = source.casefold()
    for row in name_rows:
        if str(row.get("source") or "").casefold() != expected_source:
            continue
        accepted_key = str(row.get("accepted_taxon_key") or "")
        source_taxon_id = str(row.get("source_taxon_id") or "").strip()
        if not source_taxon_id:
            source_taxon_id = _source_taxon_id_from_record(str(row.get("source_record_id") or ""), source=source)
        if accepted_key and source_taxon_id:
            grouped.setdefault(source_taxon_id, set()).add(accepted_key)
    return {source_taxon_id: next(iter(values)) for source_taxon_id, values in grouped.items() if len(values) == 1}


def _source_taxon_id_from_record(source_record_id: str, *, source: str) -> str:
    prefix = f"{source.casefold()}:"
    if not source_record_id.casefold().startswith(prefix):
        return ""
    parts = source_record_id.split(":")
    return parts[1] if len(parts) > 1 else ""


def _gbif_synonym_source_lookup(name_rows: list[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
    grouped: dict[str, set[str]] = {}
    for row in name_rows:
        if not _is_gbif_source(row) or str(row.get("name_class") or "") != "scientific_synonym":
            continue
        source_taxon_id = _gbif_source_taxon_id(row)
        accepted_key = _full_gbif_key(row.get("accepted_taxon_key"))
        if source_taxon_id and accepted_key:
            grouped.setdefault(source_taxon_id, set()).add(accepted_key)
    unambiguous = {source_key: next(iter(accepted_keys)) for source_key, accepted_keys in grouped.items() if len(accepted_keys) == 1}
    ambiguous = {source_key for source_key, accepted_keys in grouped.items() if len(accepted_keys) > 1}
    return unambiguous, ambiguous


def _is_gbif_source(row: dict[str, Any]) -> bool:
    return str(row.get("source") or "").strip().casefold() == "gbif"


def _gbif_source_taxon_id(row: dict[str, Any]) -> str:
    explicit = _full_gbif_key(row.get("source_taxon_id"))
    if explicit:
        return explicit
    source_record_id = str(row.get("source_record_id") or "").strip()
    if not source_record_id.startswith("gbif:"):
        return ""
    parts = source_record_id.split(":")
    if len(parts) >= 4 and parts[1] == "vernacular" and parts[2] == "gbif":
        return _full_gbif_key(parts[3])
    if len(parts) >= 3 and parts[1] == "vernacular":
        return _full_gbif_key(parts[2])
    if len(parts) >= 2:
        return _full_gbif_key(parts[1])
    return ""


def _full_gbif_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("gbif:"):
        return text
    return f"gbif:{text}"


def _taxref_vernacular_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("nomVernFr", "nomVern", "nomVernaculaire", "vernacularName", "frenchVernacularName"):
        value = row.get(key)
        if isinstance(value, list):
            raw_items = value
        elif value is None:
            raw_items = []
        else:
            raw_items = [part for chunk in str(value).split("|") for part in chunk.split(";")]
        for item in raw_items:
            if isinstance(item, dict):
                name = _first_string(item, "name", "vernacularName", "nomVernFr", "nomVern")
            else:
                name = str(item or "")
            name = name.strip()
            if name and name not in values:
                values.append(name)
    return values


def _taxref_region(row: dict[str, Any]) -> str:
    explicit = str(row.get("region") or row.get("territory") or row.get("territoire") or "").strip().upper()
    if explicit:
        return explicit
    regions: list[str] = []
    for field, region in TAXREF_TERRITORY_FIELDS:
        status = str(row.get(field) or "").strip().upper()
        if status and status != "A":
            regions.append(region)
    return ";".join(regions)


def _taxref_source_taxon_id(row: dict[str, Any]) -> str:
    return str(row.get("cdNom") or row.get("cd_nom") or row.get("id") or "").strip()


def _taxref_source_ref_id(row: dict[str, Any]) -> str:
    return str(row.get("cdRef") or row.get("cd_ref") or "").strip()


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


def _resolve_static_snapshot_path(config_path: Path, snapshot_path: str) -> Path:
    raw = Path(snapshot_path)
    if raw.is_absolute():
        return raw
    cwd_path = Path.cwd() / raw
    if cwd_path.exists():
        return cwd_path
    config_relative = config_path.parent / raw
    if config_relative.exists():
        return config_relative
    repo_relative = Path(__file__).resolve().parents[3] / raw
    return repo_relative


def _static_value(row: dict[str, Any], field: str, default: str = "") -> str:
    value = str(row.get(field) or "").strip()
    return value if value else default


def _static_source_record_id(source_key: str, source_taxon_id: str, vernacular_name: str) -> str:
    if source_taxon_id:
        return f"{source_key}:{source_taxon_id}:vernacular:{vernacular_name}"
    return f"{source_key}:vernacular:{vernacular_name}"


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
    source_taxon_id: str = "",
    lineage_check: str = "",
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
        "lineage_check": lineage_check,
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
