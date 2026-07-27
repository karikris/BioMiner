from __future__ import annotations

import csv
import json
import logging
import threading
import time

import httpx
import polars as pl
import pytest

from biominer.cli import build_parser, run
from biominer.registry.compiler import compile_registry_fixture, compile_registry_frames
from biominer.registry.enrichment import (
    SOURCE_WORK_LEDGER_FILE,
    SpeciesContext,
    build_enrichment_sources_from_registry,
    compile_enriched_registry,
    default_enrichment_clients,
    write_enrichment_sources,
)
from biominer.registry.enrichment_sources import (
    CatalogueOfLifeClient,
    GBIFCOLXRLegacyResolver,
    GBIFVernacularClient,
    INaturalistClient,
    ITISClient,
    OpenTreeBulkClient,
    StaticVernacularSourceClient,
    TAXREFFrenchClient,
    TMDGermanClient,
    WikidataClient,
    _json_get,
)
from biominer.registry.translation_sources import generated_translation_candidate, write_translation_candidates


def _write_base_registry(tmp_path, species_names: tuple[str, ...] = ("Papilio demoleus",)):
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "scope_id": "test-scope",
                "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"},
                "included_families": ["Papilionidae"],
                "gbif_family_taxon_keys": {"Papilionidae": 10},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "source": "GBIF",
                "source_version": "gbif-species-api",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "taxa": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "scientific_name": "Papilionoidea",
                        "rank": "SUPERFAMILY",
                        "parent_key": "",
                        "family_key": "",
                        "family": "",
                        "genus_key": "",
                        "genus": "",
                        "species_key": "",
                        "species": "",
                    },
                    {
                        "accepted_taxon_key": "gbif:10",
                        "scientific_name": "Papilionidae",
                        "rank": "FAMILY",
                        "parent_key": "gbif:1",
                        "family_key": "gbif:10",
                        "family": "Papilionidae",
                        "genus_key": "",
                        "genus": "",
                        "species_key": "",
                        "species": "",
                    },
                    {
                        "accepted_taxon_key": "gbif:90",
                        "scientific_name": "Papilio",
                        "rank": "GENUS",
                        "parent_key": "gbif:10",
                        "family_key": "gbif:10",
                        "family": "Papilionidae",
                        "genus_key": "gbif:90",
                        "genus": "Papilio",
                        "species_key": "",
                        "species": "",
                    },
                    *[
                        {
                            "accepted_taxon_key": f"gbif:{100 + index}",
                            "scientific_name": species_name,
                            "rank": "SPECIES",
                            "parent_key": "gbif:90",
                            "family_key": "gbif:10",
                            "family": "Papilionidae",
                            "genus_key": "gbif:90",
                            "genus": "Papilio",
                            "species_key": f"gbif:{100 + index}",
                            "species": species_name,
                        }
                        for index, species_name in enumerate(species_names)
                    ],
                ],
                "names": [
                    {
                        "accepted_taxon_key": f"gbif:{100 + index}",
                        "verbatim_name": species_name,
                        "display_name": species_name,
                        "language": "la",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": f"gbif:{100 + index}",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    }
                    for index, species_name in enumerate(species_names)
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry"
    compile_registry_fixture(source, registry, registry_version="base", scope_path=scope)
    return registry, scope


def _species_context() -> SpeciesContext:
    return SpeciesContext(
        accepted_taxon_key="gbif:100",
        accepted_scientific_name="Papilio demoleus",
        family_key="gbif:10",
        family="Papilionidae",
        genus_key="gbif:90",
        genus="Papilio",
        current_names=("Papilio demoleus",),
    )


def _write_static_source(
    tmp_path,
    *,
    source_key: str = "boi_india_en",
    rows: list[dict[str, str]] | None = None,
    config_updates: dict[str, str] | None = None,
    fieldnames: list[str] | None = None,
):
    snapshot = tmp_path / f"{source_key}.csv"
    csv_fieldnames = fieldnames or [
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
        "name_class",
        "taxonomic_caution",
        "taxonomic_caution_reason",
    ]
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in rows or []:
            writer.writerow({field: row.get(field, "") for field in csv_fieldnames})
    config = {
        "source_key": source_key,
        "source_name": "Butterflies of India",
        "source_display_name": "Butterflies of India",
        "source_type": "static_csv",
        "source_version": "test-static-v1",
        "snapshot_version": "test-static-v1",
        "snapshot_path": str(snapshot),
        "country_code": "IN",
        "country_scope": ["IN"],
        "language": "eng",
        "language_scope": ["eng"],
        "script": "Latn",
        "region": "IN",
        "trust_tier": "T2",
        "source_reliability_tier": "T2",
        "precision_tier": "high",
        "source_url": "https://www.ifoundbutterflies.org/",
        "citation": "Fixture citation",
        "licence": "fixture licence",
    }
    if config_updates:
        config.update(config_updates)
    config_path = tmp_path / f"{source_key}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_compile_enriched_registry_adds_enabled_names_and_preserves_candidates(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    base_manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "US",
                "name_class": "vernacular",
                "source": "ITIS",
                "source_record_id": "itis:common:1",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "inaturalist:1:preferred_common_name",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Citrus butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular_alias",
                "source": "iNaturalist",
                "source_record_id": "inaturalist:1:name:en:Citrus butterfly",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "low",
                "enabled": False,
                "review_state": "candidate",
                "disabled_reason": "regional_name_requires_review",
            },
        ],
        external_links=[
            {
                "accepted_taxon_key": "gbif:100",
                "source": "iNaturalist",
                "source_taxon_id": "1",
                "match_method": "scientific_name",
                "match_confidence": "high",
                "lineage_check": "accepted_taxon_key",
            }
        ],
        source_snapshots=[
            {
                "source": "iNaturalist",
                "source_version": "inaturalist-v1-taxa",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "source_path": "memory://inaturalist",
                "source_response_hash": "sha256:test",
                "licence": "",
            }
        ],
    )
    staged_manifest = json.loads((registry / "enrichment_manifest.json").read_text(encoding="utf-8"))
    assert json.loads((registry / "manifest.json").read_text(encoding="utf-8"))["registry_version"] == base_manifest["registry_version"]
    assert staged_manifest["schema_version"] == "registry-enrichment-v1"
    assert pl.read_parquet(registry / "source_snapshots.parquet").select("source").to_series().to_list() == ["GBIF"]

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    output = registry
    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    candidates = pl.read_parquet(output / "name_candidates.parquet")
    evidence = pl.read_parquet(output / "name_evidence.parquet")
    links = pl.read_parquet(output / "external_taxon_links.parquet")
    snapshots = pl.read_parquet(output / "source_snapshots.parquet")

    assert manifest["registry_version"] == "enriched"
    assert manifest["enrichment_name_assertion_rows"] == 3
    assert manifest["enabled_enrichment_name_rows"] == 2
    assert "Lime Swallowtail" in names["display_name"].to_list()
    assert "Lime butterfly" in names["display_name"].to_list()
    assert "Citrus butterfly" not in names["display_name"].to_list()
    assert "Citrus butterfly" in candidates["display_name"].to_list()
    assert "Citrus butterfly" not in queries["source_term"].to_list()
    assert evidence.filter(pl.col("source") == "ITIS").height == 1
    assert evidence.filter(pl.col("source") == "iNaturalist").height == 2
    assert links.select("source_taxon_id").to_series().to_list() == ["1"]
    assert snapshots.select("source").to_series().to_list() == ["GBIF", "iNaturalist"]


def test_catalogue_of_life_source_uses_non_truncating_search_limit() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        requests.append((path, dict(params)))
        return {"result": []}

    client = CatalogueOfLifeClient(http_get=fake_get)
    client.enrich_species(_species_context())

    assert requests == [("/dataset/3/nameusage/search", {"q": "Papilio demoleus", "limit": 1000})]


def test_catalogue_of_life_source_rejects_drift_and_preserves_vernacular_languages() -> None:
    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        assert path == "/dataset/3/nameusage/search"
        assert params == {"q": "Papilio demoleus", "limit": 1000}
        return {
            "result": [
                {
                    "id": "demoleus",
                    "usage": {"name": {"scientificName": "Papilio demoleus"}},
                    "vernacularNames": [
                        {"name": "Lime Butterfly", "language": "eng"},
                        {"name": "Schwalbenschwanz", "language": "deu"},
                        {"name": "Ritariperhonen", "language": "fin"},
                        {"name": "Untyped name"},
                    ],
                },
                {
                    "id": "machaon",
                    "usage": {"name": {"scientificName": "Papilio machaon"}},
                    "vernacularNames": [
                        {"name": "Old World Swallowtail", "language": "eng"},
                        {"name": "Makaonfjäril", "language": "swe"},
                    ],
                },
                {
                    "id": "sculpin",
                    "usage": {"name": {"scientificName": "Hemilepidotus papilio"}},
                    "vernacularNames": [
                        {"name": "butterfly sculpin", "language": "eng"},
                        {"name": "クジャクカジカ", "language": "jpn"},
                        {"name": "Бычок-бабочка", "language": "rus"},
                    ],
                },
            ]
        }

    result = CatalogueOfLifeClient(http_get=fake_get).enrich_species(_species_context())

    assertions = sorted(result["name_assertions"], key=lambda row: row["display_name"])
    assert [(row["display_name"], row["language"], row["script"], row["enabled"], row["disabled_reason"]) for row in assertions] == [
        ("Lime Butterfly", "eng", "Latn", True, ""),
        ("Ritariperhonen", "fin", "Latn", True, ""),
        ("Schwalbenschwanz", "deu", "Latn", True, ""),
        ("Untyped name", "", "", False, "missing_language"),
    ]
    assert {row["source_record_id"] for row in assertions} == {
        "col:demoleus:vernacular:Lime Butterfly",
        "col:demoleus:vernacular:Ritariperhonen",
        "col:demoleus:vernacular:Schwalbenschwanz",
        "col:demoleus:vernacular:Untyped name",
    }
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "CoL",
            "source_taxon_id": "demoleus",
            "match_method": "scientific_name",
            "match_confidence": "high",
            "lineage_check": "accepted_scientific_name",
        }
    ]
    assert result["coverage"] == {
        "rows_fetched": 3,
        "accepted_name_rows": 1,
        "synonym_rows": 0,
        "rejected_unmatched_rows": 2,
        "vernacular_names_extracted": 4,
        "vernaculars_with_language": 3,
        "vernaculars_without_language": 1,
        "disabled_candidate_rows": 1,
    }


def test_catalogue_of_life_source_accepts_unambiguous_synonym_rows() -> None:
    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        assert path == "/dataset/3/nameusage/search"
        assert params == {"q": "Papilio demoleus", "limit": 1000}
        return {
            "result": [
                {
                    "id": "synonym-row",
                    "usage": {"name": {"scientificName": "Princeps demoleus"}},
                    "vernacularNames": [{"name": "Synonym Lime", "language": "eng"}],
                }
            ]
        }

    context = _species_context()
    context = SpeciesContext(
        accepted_taxon_key=context.accepted_taxon_key,
        accepted_scientific_name=context.accepted_scientific_name,
        family_key=context.family_key,
        family=context.family,
        genus_key=context.genus_key,
        genus=context.genus,
        current_names=("Papilio demoleus", "Princeps demoleus"),
    )

    result = CatalogueOfLifeClient(http_get=fake_get).enrich_species(context)

    assert [(row["display_name"], row["language"], row["lineage_check"], row["confidence"]) for row in result["name_assertions"]] == [
        ("Synonym Lime", "eng", "scientific_synonym", "medium")
    ]
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "CoL",
            "source_taxon_id": "synonym-row",
            "match_method": "scientific_synonym",
            "match_confidence": "medium",
            "lineage_check": "scientific_synonym",
        }
    ]
    assert result["coverage"]["synonym_rows"] == 1


def test_inaturalist_source_uses_non_truncating_taxa_page_size() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        requests.append((path, dict(params)))
        return {"results": []}

    client = INaturalistClient(http_get=fake_get)
    client.enrich_species(_species_context())

    assert requests == [("/v1/taxa", {"q": "Papilio demoleus", "rank": "species", "per_page": 200})]


def test_open_tree_bulk_source_keeps_exact_names_and_synonyms_at_t2() -> None:
    requests = []

    def fake_post(path, payload):  # noqa: ANN001, ANN202 - source test double.
        requests.append((path, payload))
        return {
            "results": [
                {
                    "name": "Papilio demoleus",
                    "matches": [
                        {
                            "matched_name": "Papilio demoleus",
                            "score": 1.0,
                            "taxon": {
                                "ott_id": 123,
                                "name": "Papilio demoleus",
                                "rank": "species",
                                "synonyms": ["Princeps demoleus"],
                                "is_suppressed": False,
                            },
                        }
                    ],
                }
            ]
        }

    result = OpenTreeBulkClient(json_post=fake_post).enrich_registry(
        taxa_rows=[
            {
                "accepted_taxon_key": "col:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family_key": "col:10",
                "family": "Papilionidae",
                "genus_key": "col:90",
                "genus": "Papilio",
            }
        ],
        name_rows=[],
    )

    assert requests[0][0] == "/tnrs/match_names"
    assert [(row["display_name"], row["name_class"], row["trust_tier"]) for row in result["name_assertions"]] == [
        ("Papilio demoleus", "accepted_scientific", "T2"),
        ("Princeps demoleus", "scientific_synonym", "T2"),
    ]
    assert result["external_links"][0]["source_taxon_id"] == "123"


def test_wikidata_source_name_assertions_preserve_same_taxon_binding() -> None:
    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        return {
            "results": {
                "bindings": [
                    {
                        "taxon": {"value": "http://www.wikidata.org/entity/Q123"},
                        "gbifId": {"value": "100"},
                        "commonName": {"value": "Zitronenfalter", "xml:lang": "de"},
                        "commonNameLang": {"value": "de"},
                    }
                ]
            }
        }

    result = WikidataClient(http_get=fake_get).enrich_species(_species_context())
    assertion = result["name_assertions"][0]

    assert assertion["source_taxon_id"] == "Q123"
    assert assertion["lineage_check"] == "accepted_taxon_key"


def test_wikidata_source_accepts_current_col_xr_p14607_identifier_without_crosswalk() -> None:
    class ResolverMustNotRun:
        def resolve(self, context, legacy_gbif_id):  # noqa: ANN001, ANN202 - test double.
            raise AssertionError("legacy resolver must not run for an exact P14607 binding")

    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        assert "P14607" in params["query"]
        assert "P846" in params["query"]
        return {
            "results": {
                "bindings": [
                    {
                        "taxon": {"value": "http://www.wikidata.org/entity/Q285314"},
                        "scientificName": {"value": "Papilio demoleus"},
                        "gbifTaxonId": {"value": "6TLXW"},
                        "commonName": {"value": "Lime butterfly", "xml:lang": "en"},
                        "commonNameLang": {"value": "en"},
                    }
                ]
            }
        }

    context = SpeciesContext(
        accepted_taxon_key="gbif:6TLXW",
        accepted_scientific_name="Papilio demoleus",
        family_key="gbif:6254D",
        family="Papilionidae",
        genus_key="gbif:84RZD",
        genus="Papilio",
        current_names=("Papilio demoleus",),
    )
    result = WikidataClient(
        http_get=fake_get,
        col_xr_resolver=ResolverMustNotRun(),
    ).enrich_species(context)

    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:6TLXW",
            "source": "Wikidata",
            "source_taxon_id": "Q285314",
            "match_method": "P225+P14607",
            "match_confidence": "high",
            "lineage_check": "accepted_taxon_key",
        }
    ]
    assert result["name_assertions"][0]["display_name"] == "Lime butterfly"
    assert result["request_count"] == 1


def test_wikidata_source_crosswalks_legacy_p846_through_gbif_col_xr_with_lineage() -> None:
    gbif_requests: list[tuple[str, dict[str, object]]] = []

    def fake_gbif_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        gbif_requests.append((path, params))
        if path == "/v1/species/1938069":
            return {
                "canonicalName": "Papilio demoleus",
                "species": "Papilio demoleus",
                "rank": "SPECIES",
                "family": "Papilionidae",
                "genus": "Papilio",
            }
        assert path == "/v2/species/match"
        return {
            "usage": {
                "key": "6TLXW",
                "canonicalName": "Papilio demoleus",
                "rank": "SPECIES",
            },
            "classification": [
                {"key": "6254D", "name": "Papilionidae", "rank": "FAMILY"},
                {"key": "84RZD", "name": "Papilio", "rank": "GENUS"},
            ],
            "diagnostics": {"matchType": "EXACT", "confidence": 100},
        }

    def fake_wikidata_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        return {
            "results": {
                "bindings": [
                    {
                        "taxon": {"value": "http://www.wikidata.org/entity/Q285314"},
                        "scientificName": {"value": "Papilio demoleus"},
                        "legacyGbifId": {"value": "1938069"},
                        "commonName": {"value": "Lime butterfly", "xml:lang": "en"},
                        "commonNameLang": {"value": "en"},
                    }
                ]
            }
        }

    context = SpeciesContext(
        accepted_taxon_key="gbif:6TLXW",
        accepted_scientific_name="Papilio demoleus",
        family_key="gbif:6254D",
        family="Papilionidae",
        genus_key="gbif:84RZD",
        genus="Papilio",
        current_names=("Papilio demoleus",),
    )
    result = WikidataClient(
        http_get=fake_wikidata_get,
        col_xr_resolver=GBIFCOLXRLegacyResolver(http_get=fake_gbif_get),
    ).enrich_species(context)

    assert result["external_links"][0]["match_method"] == "P225+P846+GBIF_COL_XR"
    assert result["external_links"][0]["lineage_check"] == "legacy_taxon+accepted_taxon_key+family+genus"
    assert result["request_count"] == 3
    assert gbif_requests[1][1] == {
        "scientificName": "Papilio demoleus",
        "rank": "SPECIES",
        "checklistKey": "7ddf754f-d193-4cc9-b351-99906754a03b",
        "family": "Papilionidae",
        "genus": "Papilio",
    }


def test_wikidata_source_rejects_legacy_crosswalk_with_wrong_col_xr_lineage() -> None:
    def fake_gbif_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        if path.startswith("/v1/species/"):
            return {
                "canonicalName": "Papilio demoleus",
                "rank": "SPECIES",
                "family": "Papilionidae",
                "genus": "Papilio",
            }
        return {
            "usage": {
                "key": "6TLXW",
                "canonicalName": "Papilio demoleus",
                "rank": "SPECIES",
            },
            "classification": [
                {"key": "WRONG", "name": "Nymphalidae", "rank": "FAMILY"},
                {"key": "84RZD", "name": "Papilio", "rank": "GENUS"},
            ],
            "diagnostics": {"matchType": "EXACT", "confidence": 100},
        }

    result = WikidataClient(
        http_get=lambda path, params: {
            "results": {
                "bindings": [
                    {
                        "taxon": {"value": "http://www.wikidata.org/entity/Q285314"},
                        "scientificName": {"value": "Papilio demoleus"},
                        "legacyGbifId": {"value": "1938069"},
                        "commonName": {"value": "Lime butterfly", "xml:lang": "en"},
                    }
                ]
            }
        },
        col_xr_resolver=GBIFCOLXRLegacyResolver(http_get=fake_gbif_get),
    ).enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:6TLXW",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:6254D",
            family="Papilionidae",
            genus_key="gbif:84RZD",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert result["external_links"] == []
    assert result["name_assertions"] == []


def test_english_language_variants_normalize_and_deduplicate(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Butterfly",
                "language": "en",
                "script": "Latn",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "inaturalist:1:name:en:Lime Butterfly",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Butterfly",
                "language": "English",
                "script": "Latn",
                "name_class": "vernacular",
                "source": "ITIS",
                "source_record_id": "itis:123:common:Lime Butterfly",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
        ],
    )

    staged = pl.read_parquet(registry / "source_name_assertions.parquet")
    assert staged.select("language").to_series().to_list() == ["eng", "eng"]

    compile_enriched_registry(registry_dir=registry, registry_version="enriched", scope_path=scope)

    names = pl.read_parquet(registry / "names.parquet")
    lime_names = names.filter(pl.col("display_name") == "Lime Butterfly")
    assert lime_names.height == 1
    assert lime_names.select("language").to_series().to_list() == ["eng"]

    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    lime_queries = queries.filter(pl.col("source_term") == "Lime Butterfly")
    assert lime_queries.height == 2
    assert set(lime_queries.select("language").to_series().to_list()) == {"eng"}

    evidence = pl.read_parquet(registry / "name_evidence.parquet")
    assert evidence.filter(pl.col("source").is_in(["iNaturalist", "ITIS"])).height == 2


def test_tmd_german_client_maps_species_names_and_skips_unusable_rows() -> None:
    calls = []

    def graphql_post(payload):
        calls.append(payload)
        project = str(payload["variables"]["project"])
        if project == "407":
            return {
                "data": {
                    "taxonEntries": {
                        "totalCount": 4,
                        "edges": [
                            {
                                "node": {
                                    "ptnameId": 1,
                                    "projectId": 407,
                                    "uiLabel": "Papilio demoleus",
                                    "family": "Papilionidae",
                                    "genus": "Papilio",
                                    "species": "demoleus",
                                    "author": "Linnaeus, 1758",
                                }
                            },
                            {
                                "node": {
                                    "ptnameId": 2,
                                    "projectId": 407,
                                    "uiLabel": "Papilio machaon",
                                    "family": "Papilionidae",
                                    "genus": "Papilio",
                                    "species": "machaon",
                                    "author": "Linnaeus, 1758",
                                }
                            },
                            {
                                "node": {
                                    "ptnameId": 3,
                                    "projectId": 407,
                                    "uiLabel": "#Zygaena minos/purpuralis(Komplex)",
                                    "family": "Zygaenidae",
                                    "genus": "Zygaena",
                                    "species": "minos/purpuralis",
                                    "author": "(Komplex)",
                                }
                            },
                            {
                                "node": {
                                    "ptnameId": 600003,
                                    "projectId": 407,
                                    "uiLabel": "Papilionidae",
                                    "family": "Papilionidae",
                                    "genus": None,
                                    "species": None,
                                    "author": None,
                                }
                            },
                        ],
                    }
                }
            }
        return {
            "data": {
                "taxonEntries": {
                    "totalCount": 4,
                    "edges": [
                        {
                            "node": {
                                "ptnameId": 1,
                                "projectId": 410,
                                "uiLabel": "Karierter Schwalbenschwanz",
                                "family": "Papilionidae",
                                "genus": "Papilio",
                                "species": "Karierter Schwalbenschwanz",
                            }
                        },
                        {
                            "node": {
                                "ptnameId": 2,
                                "projectId": 410,
                                "uiLabel": "Schwalbenschwanz",
                                "family": "Papilionidae",
                                "genus": "Papilio",
                                "species": "Schwalbenschwanz",
                            }
                        },
                        {
                            "node": {
                                "ptnameId": 3,
                                "projectId": 410,
                                "uiLabel": "Komplex",
                                "family": "Zygaenidae",
                                "genus": "Zygaena",
                                "species": "Komplex",
                            }
                        },
                        {
                            "node": {
                                "ptnameId": 600003,
                                "projectId": 407,
                                "uiLabel": "Papilionidae",
                                "family": "Papilionidae",
                                "genus": None,
                                "species": None,
                            }
                        },
                    ],
                }
            }
        }

    client = TMDGermanClient(graphql_post=graphql_post)
    result = client.enrich_registry(
        taxa_rows=[
            {"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"},
            {"accepted_taxon_key": "gbif:101", "rank": "SPECIES", "scientific_name": "Papilio machaon"},
            {"accepted_taxon_key": "gbif:10", "rank": "FAMILY", "scientific_name": "Papilionidae"},
        ],
        name_rows=[],
    )

    assertions = result["name_assertions"]
    assert [(row["accepted_taxon_key"], row["display_name"], row["language"], row["region"]) for row in assertions] == [
        ("gbif:100", "Karierter Schwalbenschwanz", "deu", "DE"),
        ("gbif:101", "Schwalbenschwanz", "deu", "DE"),
    ]
    assert {row["source"] for row in assertions} == {"TMD"}
    assert {row["match_method"] for row in result["external_links"]} == {"scientific_name"}
    assert result["coverage"]["family_genus_german_labels"] == "not_available_from_tmd"
    assert result["coverage"]["mapped_accepted_name_rows"] == 2
    assert result["coverage"]["skipped_complex_rows"] == 1
    assert result["coverage"]["skipped_parent_rank_rows"] == 1
    assert len(calls) == 2


def test_tmd_german_client_maps_unambiguous_synonyms_with_lower_confidence() -> None:
    def graphql_post(payload):
        project = str(payload["variables"]["project"])
        label = "Old papilio" if project == "407" else "Alter Schwalbenschwanz"
        return {
            "data": {
                "taxonEntries": {
                    "totalCount": 1,
                    "edges": [
                        {
                            "node": {
                                "ptnameId": 9,
                                "projectId": int(project),
                                "uiLabel": label,
                                "family": "Papilionidae",
                                "genus": "Old",
                                "species": "papilio" if project == "407" else "Alter Schwalbenschwanz",
                            }
                        }
                    ],
                }
            }
        }

    result = TMDGermanClient(graphql_post=graphql_post).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Old papilio",
                "name_class": "scientific_synonym",
            }
        ],
    )

    assert result["name_assertions"][0]["accepted_taxon_key"] == "gbif:100"
    assert result["name_assertions"][0]["confidence"] == "medium"
    assert result["external_links"][0]["match_method"] == "scientific_synonym"
    assert result["coverage"]["mapped_synonym_rows"] == 1


def test_gbif_vernacular_client_reuses_existing_names_for_accepted_and_synonym_matches() -> None:
    result = GBIFVernacularClient().enrich_registry(
        taxa_rows=[
            {"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"},
            {"accepted_taxon_key": "gbif:101", "rank": "SPECIES", "scientific_name": "Papilio machaon"},
        ],
        name_rows=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Papilio demoleus",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:100",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Princeps demoleus",
                "name_class": "scientific_synonym",
                "source": "GBIF",
                "source_record_id": "gbif:900",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Shared synonym",
                "name_class": "scientific_synonym",
                "source": "GBIF",
                "source_record_id": "gbif:901",
            },
            {
                "accepted_taxon_key": "gbif:101",
                "display_name": "Shared synonym",
                "name_class": "scientific_synonym",
                "source": "GBIF",
                "source_record_id": "gbif:901",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:100:Lime Butterfly",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:100:Lime Butterfly duplicate",
            },
            {
                "accepted_taxon_key": "gbif:900",
                "display_name": "Citrus Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:900:Citrus Swallowtail",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "No Language Butterfly",
                "language": "",
                "script": "",
                "region": "",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:100:No Language Butterfly",
            },
            {
                "accepted_taxon_key": "gbif:999",
                "display_name": "Out of Scope Butterfly",
                "language": "eng",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:999:Out of Scope Butterfly",
            },
            {
                "accepted_taxon_key": "gbif:901",
                "display_name": "Ambiguous Butterfly",
                "language": "eng",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:vernacular:901:Ambiguous Butterfly",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Non GBIF Name",
                "language": "eng",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:vernacular:1",
            },
        ],
    )

    assertions = result["name_assertions"]
    assert [(row["accepted_taxon_key"], row["display_name"], row["enabled"], row["disabled_reason"]) for row in assertions] == [
        ("gbif:100", "Citrus Swallowtail", True, ""),
        ("gbif:100", "Lime Butterfly", True, ""),
        ("gbif:100", "No Language Butterfly", False, "missing_language"),
    ]
    assert [row["name_class"] for row in assertions] == ["vernacular_alias", "vernacular", "vernacular"]
    assert {row["source"] for row in assertions} == {"GBIF"}
    assert {row["trust_tier"] for row in assertions} == {"T1"}
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "GBIF",
            "source_taxon_id": "gbif:100",
            "match_method": "accepted_taxon_key",
            "match_confidence": "high",
            "lineage_check": "accepted_taxon_key",
        },
        {
            "accepted_taxon_key": "gbif:100",
            "source": "GBIF",
            "source_taxon_id": "gbif:900",
            "match_method": "scientific_synonym",
            "match_confidence": "medium",
            "lineage_check": "scientific_synonym",
        },
    ]
    assert result["coverage"] == {
        "rows_inspected": 6,
        "names_extracted": 3,
        "names_with_language": 2,
        "names_without_language": 1,
        "accepted_matches": 3,
        "synonym_matches": 1,
        "ambiguous_matches": 1,
        "duplicate_names": 1,
        "out_of_scope_rows": 1,
        "disabled_names": 1,
        "request_count": 0,
    }
    assert result["source_snapshots"][0]["source"] == "GBIF"
    assert result["source_snapshots"][0]["source_version"] == "gbif-vernacular-from-registry-v1"
    assert result["source_snapshots"][0]["source_path"] == "registry:names.parquet"
    assert result["source_snapshots"][0]["source_response_hash"].startswith("sha256:")


def test_build_enrichment_sources_runs_gbif_vernacular_as_zero_request_bulk_source(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)
    names = pl.read_parquet(registry / "names.parquet")
    names = pl.concat(
        [
            names,
            pl.DataFrame(
                [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "Lime Butterfly",
                        "display_name": "Lime Butterfly",
                        "language": "eng",
                        "api_language_code": "en",
                        "script": "Latn",
                        "region": "",
                        "bcp47": "en-Latn",
                        "bbox": "",
                        "name_class": "vernacular",
                        "source": "GBIF",
                        "source_record_id": "gbif:vernacular:100:Lime Butterfly",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "medium",
                        "enabled": True,
                        "review_state": "accepted",
                        "corroborated": False,
                        "disabled_reason": "",
                        "query_eligible": True,
                        "query_eligibility_reason": "eligible",
                        "query_eligibility_score": 1.0,
                    }
                ],
                schema=names.schema,
            ),
        ],
        how="vertical",
    )
    names.write_parquet(registry / "names.parquet")

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("gbif_vernacular",),
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    work = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)
    coverage = json.loads((registry / "enrichment_coverage.json").read_text(encoding="utf-8"))

    assert manifest["source_order"] == ["gbif_vernacular"]
    assert assertions.select("source").to_series().to_list() == ["GBIF"]
    assert assertions.select("display_name").to_series().to_list() == ["Lime Butterfly"]
    assert work.select(["source", "accepted_taxon_key", "request_count"]).to_dicts() == [
        {"source": "gbif_vernacular", "accepted_taxon_key": "__registry__", "request_count": 0}
    ]
    assert coverage["bulk_sources"]["GBIF"]["request_count"] == 0


def test_taxref_french_client_maps_accepted_synonym_territory_and_rejects_ambiguous_rows() -> None:
    result = TAXREFFrenchClient(
        taxref_rows=[
            {
                "cdNom": 654792,
                "cdRef": 654792,
                "lbNom": "Papilio demoleus",
                "validite": "NR",
                "rang": {"rang": "ES"},
                "nomVernFr": "Papillon du citron",
                "fr": "P",
            },
            {
                "cdNom": 700001,
                "cdRef": 654792,
                "lbNom": "Princeps demoleus",
                "validite": "SY",
                "rang": {"rang": "ES"},
                "nomVernFr": "Grand papillon citron",
                "sm": "I",
                "sb": "I",
            },
            {
                "cdNom": 700002,
                "cdRef": 700002,
                "lbNom": "Out of scope",
                "validite": "NR",
                "rang": {"rang": "ES"},
                "nomVernFr": "Hors champ",
                "fr": "P",
            },
            {
                "cdNom": 700003,
                "cdRef": 700003,
                "lbNom": "Shared synonym",
                "validite": "SY",
                "rang": {"rang": "ES"},
                "nomVernFr": "Nom ambigu",
                "fr": "P",
            },
            {
                "cdNom": 700004,
                "cdRef": 700004,
                "lbNom": "Papilio demoleus",
                "validite": "NR",
                "rang": {"rang": "ES"},
                "nomVernFr": "Nom sans territoire",
            },
        ]
    ).enrich_registry(
        taxa_rows=[
            {"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"},
            {"accepted_taxon_key": "gbif:101", "rank": "SPECIES", "scientific_name": "Papilio machaon"},
        ],
        name_rows=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Princeps demoleus",
                "name_class": "scientific_synonym",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Shared synonym",
                "name_class": "scientific_synonym",
            },
            {
                "accepted_taxon_key": "gbif:101",
                "display_name": "Shared synonym",
                "name_class": "scientific_synonym",
            },
        ],
    )

    assertions = result["name_assertions"]
    assert [(row["accepted_taxon_key"], row["display_name"], row["region"], row["confidence"], row["enabled"], row["disabled_reason"]) for row in assertions] == [
        ("gbif:100", "Grand papillon citron", "SM;SB", "medium", True, ""),
        ("gbif:100", "Nom sans territoire", "", "high", False, "missing_taxref_territory"),
        ("gbif:100", "Papillon du citron", "FR", "high", True, ""),
    ]
    assert {row["source"] for row in assertions} == {"TAXREF"}
    assert {row["language"] for row in assertions} == {"fra"}
    assert {row["trust_tier"] for row in assertions} == {"T2"}
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "TAXREF",
            "source_taxon_id": "654792",
            "match_method": "scientific_name",
            "match_confidence": "high",
            "lineage_check": "accepted_scientific_name",
        },
        {
            "accepted_taxon_key": "gbif:100",
            "source": "TAXREF",
            "source_taxon_id": "700001",
            "match_method": "scientific_synonym",
            "match_confidence": "medium",
            "lineage_check": "scientific_synonym",
        },
        {
            "accepted_taxon_key": "gbif:100",
            "source": "TAXREF",
            "source_taxon_id": "700004",
            "match_method": "scientific_name",
            "match_confidence": "high",
            "lineage_check": "accepted_scientific_name",
        },
    ]
    assert result["coverage"] == {
        "rows_fetched": 5,
        "vernacular_names_extracted": 3,
        "mapped_source_id_rows": 0,
        "mapped_accepted_name_rows": 2,
        "mapped_synonym_rows": 1,
        "out_of_scope_rows": 1,
        "ambiguous_synonym_rows": 1,
        "disabled_candidate_rows": 1,
        "rows_without_vernacular": 0,
        "territory_rows": 2,
        "request_count": 0,
    }
    assert result["source_snapshots"][0]["source"] == "TAXREF"
    assert result["source_snapshots"][0]["source_version"] == "taxref-web-api-taxa-search"
    assert result["source_snapshots"][0]["source_path"] == "https://taxref.mnhn.fr/taxref-web/api/taxa/search"
    assert result["source_snapshots"][0]["source_response_hash"].startswith("sha256:")


def test_static_vernacular_source_client_ingests_csv_and_maps_names_with_metadata(tmp_path) -> None:
    config_path = _write_static_source(
        tmp_path,
        rows=[
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "boi:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
            },
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Princeps demoleus",
                "source_taxon_id": "boi:princeps-demoleus",
                "accepted_name_usage": "",
                "vernacular_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
                "name_class": "vernacular_alias",
            },
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "boi:duplicate",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
            },
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Shared synonym",
                "source_taxon_id": "boi:ambiguous",
                "accepted_name_usage": "",
                "vernacular_name": "Ambiguous Lime",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
            },
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio outscope",
                "source_taxon_id": "boi:outscope",
                "accepted_name_usage": "",
                "vernacular_name": "Outscope Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
            },
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "boi:blank",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "CC-BY fixture",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation, Butterflies of India",
            },
        ],
        config_updates={"licence": "CC-BY fixture", "citation": "Fixture citation, Butterflies of India"},
    )

    result = StaticVernacularSourceClient.from_config_path(config_path).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[
            {"accepted_taxon_key": "gbif:100", "display_name": "Princeps demoleus", "name_class": "scientific_synonym"},
            {"accepted_taxon_key": "gbif:100", "display_name": "Shared synonym", "name_class": "scientific_synonym"},
            {"accepted_taxon_key": "gbif:101", "display_name": "Shared synonym", "name_class": "scientific_synonym"},
        ],
    )

    assertions = result["name_assertions"]
    assert [
        (
            row["display_name"],
            row["source"],
            row["language"],
            row["script"],
            row["region"],
            row["name_class"],
            row["trust_tier"],
            row["confidence"],
            row["lineage_check"],
            row["licence"],
        )
        for row in assertions
    ] == [
        ("Lime Butterfly", "Butterflies of India", "eng", "Latn", "IN", "vernacular_alias", "T2", "medium", "scientific_synonym", "CC-BY fixture"),
        ("Lime Swallowtail", "Butterflies of India", "eng", "Latn", "IN", "vernacular", "T2", "high", "accepted_scientific_name", "CC-BY fixture"),
    ]
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "Butterflies of India",
            "source_taxon_id": "boi:papilio-demoleus",
            "match_method": "scientific_name",
            "match_confidence": "high",
            "lineage_check": "accepted_scientific_name",
        },
        {
            "accepted_taxon_key": "gbif:100",
            "source": "Butterflies of India",
            "source_taxon_id": "boi:princeps-demoleus",
            "match_method": "scientific_synonym",
            "match_confidence": "medium",
            "lineage_check": "scientific_synonym",
        },
    ]
    assert result["coverage"] == {
        "rows_read": 6,
        "rows_with_vernacular": 5,
        "name_assertions": 2,
        "mapped_source_id_rows": 0,
        "mapped_accepted_name_rows": 2,
        "mapped_synonym_rows": 1,
        "out_of_scope_rows": 1,
        "ambiguous_synonym_rows": 1,
        "duplicate_rows": 1,
        "rows_without_vernacular": 1,
        "missing_source_name_rows": 0,
        "rejected_rank_rows": 0,
        "taxonomic_caution_rows": 0,
        "disabled_candidate_rows": 0,
        "request_count": 0,
    }
    assert result["source_snapshots"][0]["source"] == "Butterflies of India"
    assert result["source_snapshots"][0]["source_version"] == "test-static-v1"
    assert result["source_snapshots"][0]["source_path"] == str(config_path.with_suffix(".csv"))
    assert result["source_snapshots"][0]["source_response_hash"].startswith("sha256:")
    assert result["source_snapshots"][0]["licence"] == "CC-BY fixture"
    assert result["source_snapshots"][0]["source_url"] == "https://www.ifoundbutterflies.org/"
    assert result["source_snapshots"][0]["citation"] == "Fixture citation, Butterflies of India"


def test_static_vernacular_source_client_validates_config_and_csv_headers(tmp_path) -> None:
    config_path = _write_static_source(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["source_key"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source_key"):
        StaticVernacularSourceClient.from_config_path(config_path)

    bad_header_config = _write_static_source(
        tmp_path,
        source_key="bad_header",
        fieldnames=[
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
        ],
    )
    with pytest.raises(ValueError, match="citation"):
        StaticVernacularSourceClient.from_config_path(bad_header_config).enrich_registry(taxa_rows=[], name_rows=[])


def test_static_vernacular_source_client_validates_reusable_config_metadata(tmp_path) -> None:
    config_path = _write_static_source(tmp_path, source_key="future_source")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["source_display_name"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source_display_name"):
        StaticVernacularSourceClient.from_config_path(config_path)

    config_path = _write_static_source(tmp_path, source_key="bad_type")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source_type"] = "web_scrape"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source_type"):
        StaticVernacularSourceClient.from_config_path(config_path)

    config_path = _write_static_source(tmp_path, source_key="bad_scope")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["country_scope"] = []
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="country_scope"):
        StaticVernacularSourceClient.from_config_path(config_path)


def test_static_vernacular_source_client_preserves_same_vernacular_across_regions_and_source_versions(tmp_path) -> None:
    v1_config = _write_static_source(
        tmp_path,
        source_key="regional_checklist_v1",
        rows=[
            {
                "source_key": "regional_checklist_v1",
                "source_name": "Regional Checklist",
                "source_version": "v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "regional:v1:papilio-demoleus-in",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "fixture licence v1",
                "source_url": "https://example.invalid/regional",
                "citation": "Regional Checklist v1",
            },
            {
                "source_key": "regional_checklist_v1",
                "source_name": "Regional Checklist",
                "source_version": "v1",
                "country_code": "LK",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "regional:v1:papilio-demoleus-lk",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "LK",
                "rank": "species",
                "licence": "fixture licence v1",
                "source_url": "https://example.invalid/regional",
                "citation": "Regional Checklist v1",
            },
        ],
        config_updates={
            "source_name": "Regional Checklist",
            "source_display_name": "Regional Checklist",
            "source_version": "v1",
            "snapshot_version": "v1",
            "country_scope": ["IN", "LK"],
            "language_scope": ["eng"],
            "licence": "fixture licence v1",
            "citation": "Regional Checklist v1",
        },
    )
    v2_config = _write_static_source(
        tmp_path,
        source_key="regional_checklist_v2",
        rows=[
            {
                "source_key": "regional_checklist_v2",
                "source_name": "Regional Checklist",
                "source_version": "v2",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "regional:v2:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "fixture licence v2",
                "source_url": "https://example.invalid/regional",
                "citation": "Regional Checklist v2",
            }
        ],
        config_updates={
            "source_name": "Regional Checklist",
            "source_display_name": "Regional Checklist",
            "source_version": "v2",
            "snapshot_version": "v2",
            "country_scope": ["IN"],
            "language_scope": ["eng"],
            "licence": "fixture licence v2",
            "citation": "Regional Checklist v2",
        },
    )

    taxa_rows = [{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}]
    v1_result = StaticVernacularSourceClient.from_config_path(v1_config).enrich_registry(taxa_rows=taxa_rows, name_rows=[])
    v2_result = StaticVernacularSourceClient.from_config_path(v2_config).enrich_registry(taxa_rows=taxa_rows, name_rows=[])

    assert [row["region"] for row in v1_result["name_assertions"]] == ["IN", "LK"]
    assert {row["source"] for row in [*v1_result["name_assertions"], *v2_result["name_assertions"]]} == {"Regional Checklist"}
    assert [row["source_version"] for row in [v1_result["source_snapshots"][0], v2_result["source_snapshots"][0]]] == ["v1", "v2"]
    assert {row["licence"] for row in v1_result["name_assertions"]} == {"fixture licence v1"}
    assert v1_result["name_assertions"][0]["source_record_id"] != v2_result["name_assertions"][0]["source_record_id"]


def test_static_vernacular_source_client_preserves_t5_review_metadata_for_queries(tmp_path) -> None:
    fieldnames = [
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
        "name_class",
        "review_state",
        "precision_tier",
        "confidence",
    ]
    config_path = _write_static_source(
        tmp_path,
        source_key="papilio_demoleus_multilingual_t5",
        fieldnames=fieldnames,
        rows=[
            {
                "source_key": "papilio_demoleus_multilingual_t5",
                "source_name": "Papilio demoleus multilingual T5",
                "source_version": "test-static-v1",
                "country_code": "001",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "curated:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Limettiperhonen",
                "language": "fin",
                "script": "Latn",
                "region": "001",
                "rank": "species",
                "licence": "not_configured",
                "source_url": "",
                "citation": "Fixture citation",
                "name_class": "generated_translation",
                "review_state": "reviewed",
                "precision_tier": "medium",
                "confidence": "medium",
            }
        ],
        config_updates={
            "source_name": "Papilio demoleus multilingual T5",
            "source_display_name": "Papilio demoleus multilingual T5",
            "country_code": "001",
            "country_scope": ["001"],
            "language": "eng",
            "language_scope": ["eng", "fin", "swe"],
            "script": "Latn",
            "region": "001",
            "trust_tier": "T5",
            "source_reliability_tier": "T5",
            "precision_tier": "low",
            "source_url": "",
            "licence": "not_configured",
        },
    )

    result = StaticVernacularSourceClient.from_config_path(config_path).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[],
    )

    assertion = result["name_assertions"][0]
    assert assertion["trust_tier"] == "T5"
    assert assertion["name_class"] == "generated_translation"
    assert assertion["review_state"] == "reviewed"
    assert assertion["precision_tier"] == "medium"
    assert assertion["confidence"] == "medium"


def test_static_vernacular_source_client_rejects_broad_rank_rows_and_disables_taxonomic_caution(tmp_path) -> None:
    config_path = _write_static_source(
        tmp_path,
        source_key="cautionary_static",
        rows=[
            {
                "source_key": "cautionary_static",
                "source_name": "Cautionary Checklist",
                "source_version": "v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "cautionary:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Butterfly",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "fixture licence",
                "source_url": "",
                "citation": "Cautionary Checklist",
            },
            {
                "source_key": "cautionary_static",
                "source_name": "Cautionary Checklist",
                "source_version": "v1",
                "country_code": "AU",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "cautionary:papilio-demoleus-au",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Caution Lime",
                "language": "eng",
                "script": "Latn",
                "region": "AU",
                "rank": "species",
                "licence": "fixture licence",
                "source_url": "",
                "citation": "Cautionary Checklist",
                "taxonomic_caution": "true",
                "taxonomic_caution_reason": "demoleus_sthenelus_unresolved",
            },
            {
                "source_key": "cautionary_static",
                "source_name": "Cautionary Checklist",
                "source_version": "v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio",
                "source_taxon_id": "cautionary:papilio",
                "accepted_name_usage": "",
                "vernacular_name": "Papilio broad name",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "genus",
                "licence": "fixture licence",
                "source_url": "",
                "citation": "Cautionary Checklist",
            },
            {
                "source_key": "cautionary_static",
                "source_name": "Cautionary Checklist",
                "source_version": "v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus complex",
                "source_taxon_id": "cautionary:papilio-demoleus-complex",
                "accepted_name_usage": "",
                "vernacular_name": "Complex Lime",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species_complex",
                "licence": "fixture licence",
                "source_url": "",
                "citation": "Cautionary Checklist",
            },
        ],
        config_updates={
            "source_name": "Cautionary Checklist",
            "source_display_name": "Cautionary Checklist",
            "country_scope": ["IN", "AU"],
            "language_scope": ["eng"],
        },
    )

    result = StaticVernacularSourceClient.from_config_path(config_path).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[],
    )

    assert [(row["display_name"], row["enabled"], row["review_state"], row["disabled_reason"]) for row in result["name_assertions"]] == [
        ("Caution Lime", False, "candidate", "taxonomic_caution:demoleus_sthenelus_unresolved"),
        ("Lime Butterfly", True, "accepted", ""),
    ]
    assert result["coverage"]["name_assertions"] == 2
    assert result["coverage"]["rejected_rank_rows"] == 2
    assert result["coverage"]["taxonomic_caution_rows"] == 1
    assert result["coverage"]["disabled_candidate_rows"] == 1


def test_static_vernacular_source_client_preserves_regional_language_script_and_reports_missing_names(tmp_path) -> None:
    kannada_config = _write_static_source(
        tmp_path,
        source_key="karnataka_chitte_kn",
        rows=[
            {
                "source_key": "karnataka_chitte_kn",
                "source_name": "Karnataka Chitte",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "KA",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "karnataka-chitte:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Fixture Kannada Source Name",
                "language": "kan",
                "script": "Knda",
                "region": "IN-KA",
                "rank": "species",
                "licence": "fixture licence",
                "source_url": "",
                "citation": "Fixture Kannada citation",
            }
        ],
        config_updates={
            "source_name": "Karnataka Chitte",
            "source_display_name": "Karnataka Chitte",
            "language": "kan",
            "language_scope": ["kan"],
            "script": "Knda",
            "region": "IN-KA",
        },
    )

    kannada_result = StaticVernacularSourceClient.from_config_path(kannada_config).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[],
    )

    assert kannada_result["name_assertions"][0]["source"] == "Karnataka Chitte"
    assert kannada_result["name_assertions"][0]["language"] == "kan"
    assert kannada_result["name_assertions"][0]["script"] == "Knda"
    assert kannada_result["name_assertions"][0]["region"] == "IN-KA"

    hindi_config = _write_static_source(
        tmp_path,
        source_key="bharat_ki_titliya_hi",
        rows=[],
        config_updates={
            "source_name": "Bharat Ki Titliya",
            "source_display_name": "Bharat Ki Titliya",
            "language": "hin",
            "language_scope": ["hin"],
            "script": "Deva",
            "region": "IN",
        },
    )
    hindi_result = StaticVernacularSourceClient.from_config_path(hindi_config).enrich_registry(
        taxa_rows=[{"accepted_taxon_key": "gbif:100", "rank": "SPECIES", "scientific_name": "Papilio demoleus"}],
        name_rows=[],
    )

    assert hindi_result["name_assertions"] == []
    assert hindi_result["coverage"]["missing_source_name_rows"] == 1
    assert hindi_result["coverage"]["name_assertions"] == 0


def test_build_enrichment_sources_runs_static_source_as_bulk_source(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)
    config_path = _write_static_source(
        tmp_path,
        rows=[
            {
                "source_key": "boi_india_en",
                "source_name": "Butterflies of India",
                "source_version": "test-static-v1",
                "country_code": "IN",
                "admin1_code": "",
                "scientific_name": "Papilio demoleus",
                "source_taxon_id": "boi:papilio-demoleus",
                "accepted_name_usage": "Papilio demoleus",
                "vernacular_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "rank": "species",
                "licence": "fixture licence",
                "source_url": "https://www.ifoundbutterflies.org/",
                "citation": "Fixture citation",
            }
        ],
    )

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("boi_india_en",),
        clients={"boi_india_en": StaticVernacularSourceClient.from_config_path(config_path)},
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    work = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)
    coverage = json.loads((registry / "enrichment_coverage.json").read_text(encoding="utf-8"))

    assert manifest["source_order"] == ["boi_india_en"]
    assert assertions.select(["source", "display_name", "language", "region"]).to_dicts() == [
        {"source": "Butterflies of India", "display_name": "Lime Swallowtail", "language": "eng", "region": "IN"}
    ]
    assert work.select(["source", "accepted_taxon_key", "request_count"]).to_dicts() == [
        {"source": "boi_india_en", "accepted_taxon_key": "__registry__", "request_count": 0}
    ]
    assert coverage["bulk_sources"]["Butterflies of India"]["name_assertions"] == 1


def test_build_enrichment_sources_runs_taxref_fr_as_bulk_source(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("taxref_fr",),
        clients={
            "taxref_fr": TAXREFFrenchClient(
                taxref_rows=[
                    {
                        "cdNom": 654792,
                        "cdRef": 654792,
                        "lbNom": "Papilio demoleus",
                        "validite": "NR",
                        "rang": {"rang": "ES"},
                        "nomVernFr": "Papillon du citron",
                        "fr": "P",
                    }
                ]
            )
        },
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    work = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)
    coverage = json.loads((registry / "enrichment_coverage.json").read_text(encoding="utf-8"))

    assert manifest["source_order"] == ["taxref_fr"]
    assert assertions.select(["source", "display_name", "language", "region"]).to_dicts() == [
        {"source": "TAXREF", "display_name": "Papillon du citron", "language": "fra", "region": "FR"}
    ]
    assert work.select(["source", "accepted_taxon_key", "request_count"]).to_dicts() == [
        {"source": "taxref_fr", "accepted_taxon_key": "__registry__", "request_count": 0}
    ]
    assert coverage["bulk_sources"]["TAXREF"]["vernacular_names_extracted"] == 1


def test_build_enrichment_sources_runs_tmd_de_as_bulk_source(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)

    class FakeTMDClient:
        def enrich_registry(self, *, taxa_rows, name_rows):  # noqa: ANN001 - fake client mirrors source API.
            assert any(row["scientific_name"] == "Papilio demoleus" for row in taxa_rows)
            assert any(row["display_name"] == "Papilio demoleus" for row in name_rows)
            return {
                "name_assertions": [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "display_name": "Karierter Schwalbenschwanz",
                        "language": "deu",
                        "script": "Latn",
                        "region": "DE",
                        "name_class": "vernacular",
                        "source": "TMD",
                        "source_record_id": "tmd:410:1:Karierter Schwalbenschwanz",
                        "source_taxon_id": "1",
                        "trust_tier": "T2",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                        "review_state": "accepted",
                    }
                ],
                "external_links": [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "source": "TMD",
                        "source_taxon_id": "1",
                        "match_method": "scientific_name",
                        "match_confidence": "high",
                        "lineage_check": "accepted_taxon_key",
                    }
                ],
                "source_snapshots": [
                    {
                        "source": "TMD",
                        "source_version": "tmd-taxonomy-graphql-projects-407-410",
                        "retrieved_at": "2026-06-20T00:00:00+00:00",
                        "source_path": "memory://tmd",
                        "source_response_hash": "sha256:tmd",
                        "licence": "",
                    }
                ],
                "coverage": {"family_genus_german_labels": "not_available_from_tmd"},
            }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("tmd_de",),
        clients={"tmd_de": FakeTMDClient()},
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    work = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)
    coverage = json.loads((registry / "enrichment_coverage.json").read_text(encoding="utf-8"))

    assert manifest["source_order"] == ["tmd_de"]
    assert assertions.select("source").to_series().to_list() == ["TMD"]
    assert assertions.select("display_name").to_series().to_list() == ["Karierter Schwalbenschwanz"]
    assert work.select("accepted_taxon_key").to_series().to_list() == ["__registry__"]
    assert coverage["bulk_sources"]["TMD"]["family_genus_german_labels"] == "not_available_from_tmd"


def test_compile_enriched_registry_keeps_conflicting_source_name_disabled(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:missing",
                "display_name": "Bad Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:name:bad",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "accepted",
            }
        ],
    )

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    names = pl.read_parquet(registry / "names.parquet")
    candidates = pl.read_parquet(registry / "name_candidates.parquet")
    qa = pl.read_parquet(registry / "qa_findings.parquet")

    assert manifest["qa_status"] == "passed"
    assert "Bad Lime" not in names["display_name"].to_list()
    assert candidates.select("disabled_reason").to_series().to_list() == ["unknown_accepted_taxon_key"]
    assert {"severity": "warning", "code": "enrichment_name_without_base_taxon", "subject": "col:name:bad"} in qa.to_dicts()


def test_compile_enriched_registry_keeps_unreviewed_t5_translations_as_audit_only_name_evidence(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Translated Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "generated_translation",
                "source": "Translation",
                "source_record_id": "translation:gbif:100:eng",
                "trust_tier": "T5",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
                "review_state": "candidate",
            }
        ],
    )

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
        requested_sources=("translation",),
    )

    names = pl.read_parquet(registry / "names.parquet")
    candidates = pl.read_parquet(registry / "name_candidates.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")

    assert "Translated Lime" in names["display_name"].to_list()
    assert "Translated Lime" not in candidates["display_name"].to_list()
    t5_names = names.filter(pl.col("normalized_match_key") == "translated lime")
    assert t5_names.select("enabled").to_series().to_list() == [True]
    assert t5_names.select("trust_tier").to_series().to_list() == ["T5"]
    assert t5_names.select("disabled_reason").to_series().to_list() == [""]
    assert t5_names.select("query_eligible").to_series().to_list() == [False]
    assert t5_names.select("query_disabled_reason").to_series().to_list() == ["generated_translation_requires_review_or_corroboration"]
    t5_queries = queries.filter(pl.col("normalized_match_key") == "translated lime").sort("search_field")
    assert t5_queries.height == 0
    assert manifest["enabled_t5_name_rows"] == 1
    assert manifest["t5_query_definition_rows"] == 0
    assert manifest["t5_retrieval_query_definition_rows"] == 0
    assert manifest["query_definition_rows"] == queries.height


def test_compile_enriched_registry_requires_wikimedia_binding_for_queries(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Zitronenfalter",
                "language": "de",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular_alias",
                "source": "Wikimedia",
                "source_record_id": "wikimedia:123:Q123:de:Zitronenfalter",
                "source_taxon_id": "Q123",
                "lineage_check": "accepted_taxon_key",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Unbound Butterfly",
                "language": "de",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular_alias",
                "source": "Wikimedia",
                "source_record_id": "wikimedia:999:Q999:de:Unbound_Butterfly",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
        ],
    )

    compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
        requested_sources=("wikimedia",),
    )

    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")

    assert "Zitronenfalter" in queries["source_term"].to_list()
    assert "Unbound Butterfly" not in queries["source_term"].to_list()


def test_compile_enriched_registry_keeps_translation_candidate_file_off_flickr_queries_by_default(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_translation_candidates(
        [
            generated_translation_candidate(
                source="LibreTranslate",
                source_language="eng",
                target_language="deu",
                source_name="Lime Butterfly",
                translated_name="Limettenfalter",
                accepted_taxon_key="gbif:100",
            )
        ],
        registry,
    )

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    t5_names = names.filter(pl.col("normalized_match_key") == "limettenfalter")
    t5_queries = queries.filter(pl.col("normalized_match_key") == "limettenfalter").sort("search_field")

    assert t5_names.height == 1
    assert t5_names.select("trust_tier").to_series().to_list() == ["T5"]
    assert t5_names.select("name_class").to_series().to_list() == ["generated_translation"]
    assert t5_names.select("query_eligible").to_series().to_list() == [False]
    assert t5_queries.is_empty()
    assert manifest["translation_candidate_rows"] == 1
    assert manifest["enabled_t5_name_rows"] == 1
    assert manifest["t5_query_definition_rows"] == 0


def test_compile_enriched_registry_preserves_translation_locale_script_and_region_in_queries(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_translation_candidates(
        [
            {
                "source": "MyMemory",
                "source_language": "eng",
                "target_language": "pt-BR",
                "source_name": "Lime Butterfly",
                "translated_name": "Borboleta lima",
                "accepted_taxon_key": "gbif:100",
                "review_state": "reviewed",
                "confidence": "medium",
                "precision_tier": "medium",
                "enabled": True,
            },
            {
                "source": "MyMemory",
                "source_language": "eng",
                "target_language": "zh-Hant",
                "source_name": "Lime Butterfly",
                "translated_name": "鳳蝶",
                "accepted_taxon_key": "gbif:100",
                "review_state": "reviewed",
                "confidence": "medium",
                "precision_tier": "medium",
                "enabled": True,
            },
        ],
        registry,
    )

    compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    pt_name = names.filter(pl.col("normalized_match_key") == "borboleta lima").row(0, named=True)
    zh_name = names.filter(pl.col("normalized_match_key") == "鳳蝶").row(0, named=True)
    pt_query = queries.filter((pl.col("normalized_match_key") == "borboleta lima") & (pl.col("search_field") == "tags")).row(0, named=True)

    assert (pt_name["language"], pt_name["api_language_code"], pt_name["script"], pt_name["region"], pt_name["bcp47"], pt_name["query_eligible"]) == (
        "por",
        "pt",
        "Latn",
        "BR",
        "pt-BR",
        True,
    )
    assert (zh_name["language"], zh_name["api_language_code"], zh_name["script"], zh_name["region"], zh_name["bcp47"], zh_name["query_eligible"]) == (
        "zho",
        "zh",
        "Hant",
        "",
        "zh-Hant",
        False,
    )
    assert zh_name["query_disabled_reason"] == "generic_single_token"
    assert (pt_query["language"], pt_query["api_language_code"], pt_query["script"], pt_query["region"], pt_query["bcp47"]) == ("por", "pt", "Latn", "BR", "pt-BR")
    assert queries.filter(pl.col("normalized_match_key") == "鳳蝶").is_empty()


def test_compile_enriched_registry_keeps_script_associations_but_queries_term_once(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_translation_candidates(
        [
            {
                "source": "MyMemory",
                "source_language": "eng",
                "target_language": "zh-Hans",
                "source_name": "Lime Butterfly",
                "translated_name": "青鳳蝶",
                "accepted_taxon_key": "gbif:100",
                "review_state": "reviewed",
                "confidence": "medium",
                "precision_tier": "medium",
                "enabled": True,
            },
            {
                "source": "MyMemory",
                "source_language": "eng",
                "target_language": "zh-Hant",
                "source_name": "Lime Butterfly",
                "translated_name": "青鳳蝶",
                "accepted_taxon_key": "gbif:100",
                "review_state": "reviewed",
                "confidence": "medium",
                "precision_tier": "medium",
                "enabled": True,
            },
        ],
        registry,
    )

    compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    script_names = names.filter(pl.col("normalized_match_key") == "青鳳蝶").sort("script")
    script_queries = queries.filter(pl.col("normalized_match_key") == "青鳳蝶").sort("search_field")

    assert script_names.height == 2
    assert script_names.select("script").to_series().to_list() == ["Hans", "Hant"]
    assert script_names.filter(pl.col("is_canonical_keyword")).height == 1
    assert script_names.filter(pl.col("suppressed_duplicate")).height == 1
    assert script_queries.height == 2
    assert set(script_queries["search_field"].to_list()) == {"tags", "text"}


def test_compile_enriched_registry_disables_unreviewed_cross_taxon_collisions(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon"))
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Shared Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "inat:100:shared-lime",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "candidate",
            },
            {
                "accepted_taxon_key": "gbif:101",
                "display_name": "Shared Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "inat:101:shared-lime",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "candidate",
            },
        ],
    )

    compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
        requested_sources=("inaturalist",),
    )

    names = pl.read_parquet(registry / "names.parquet")
    candidates = pl.read_parquet(registry / "name_candidates.parquet").sort("accepted_taxon_key")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")

    assert "Shared Lime" not in names["display_name"].to_list()
    assert candidates.select("accepted_taxon_key").to_series().to_list() == ["gbif:100", "gbif:101"]
    assert candidates.select("enabled").to_series().to_list() == [False, False]
    assert candidates.select("disabled_reason").to_series().to_list() == ["name_collision_requires_review", "name_collision_requires_review"]
    assert "shared lime" not in queries["normalized_query_term"].to_list()


def test_compile_enriched_registry_collision_ignores_name_class_differences(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon"))
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Shared Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "inat:100:shared-lime",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "candidate",
            },
            {
                "accepted_taxon_key": "gbif:101",
                "display_name": "Shared Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular_alias",
                "source": "iNaturalist",
                "source_record_id": "inat:101:shared-lime",
                "trust_tier": "T4",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "candidate",
            },
        ],
    )

    compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
        requested_sources=("inaturalist",),
    )

    names = pl.read_parquet(registry / "names.parquet")
    candidates = pl.read_parquet(registry / "name_candidates.parquet").sort("accepted_taxon_key")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")

    assert "Shared Lime" not in names["display_name"].to_list()
    assert candidates.select("accepted_taxon_key").to_series().to_list() == ["gbif:100", "gbif:101"]
    assert candidates.select("enabled").to_series().to_list() == [False, False]
    assert candidates.select("disabled_reason").to_series().to_list() == ["name_collision_requires_review", "name_collision_requires_review"]
    assert "shared lime" not in queries["normalized_query_term"].to_list()


def test_species_slice_keeps_full_registry_name_collision_associations_queryable(tmp_path) -> None:
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "scope_id": "test-scope",
                "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"},
                "included_families": ["Papilionidae"],
                "gbif_family_taxon_keys": {"Papilionidae": 10},
            }
        ),
        encoding="utf-8",
    )
    lineage = [
        {
            "accepted_taxon_key": "gbif:1",
            "scientific_name": "Papilionoidea",
            "rank": "SUPERFAMILY",
            "parent_key": "",
            "family_key": "",
            "family": "",
            "genus_key": "",
            "genus": "",
            "species_key": "",
            "species": "",
        },
        {
            "accepted_taxon_key": "gbif:10",
            "scientific_name": "Papilionidae",
            "rank": "FAMILY",
            "parent_key": "gbif:1",
            "family_key": "gbif:10",
            "family": "Papilionidae",
            "genus_key": "",
            "genus": "",
            "species_key": "",
            "species": "",
        },
        {
            "accepted_taxon_key": "gbif:90",
            "scientific_name": "Papilio",
            "rank": "GENUS",
            "parent_key": "gbif:10",
            "family_key": "gbif:10",
            "family": "Papilionidae",
            "genus_key": "gbif:90",
            "genus": "Papilio",
            "species_key": "",
            "species": "",
        },
    ]
    species = [
        ("gbif:100", "Papilio demoleus"),
        ("gbif:101", "Papilio machaon"),
    ]

    def species_taxa(selected: list[tuple[str, str]]) -> list[dict[str, object]]:
        return [
            {
                "accepted_taxon_key": key,
                "scientific_name": scientific_name,
                "rank": "SPECIES",
                "parent_key": "gbif:90",
                "family_key": "gbif:10",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": key,
                "species": scientific_name,
            }
            for key, scientific_name in selected
        ]

    def names(selected: list[tuple[str, str]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key, scientific_name in selected:
            rows.append(
                {
                    "accepted_taxon_key": key,
                    "verbatim_name": scientific_name,
                    "display_name": scientific_name,
                    "language": "la",
                    "script": "Latn",
                    "name_class": "accepted_scientific",
                    "source": "GBIF",
                    "source_record_id": key,
                    "trust_tier": "T1",
                    "precision_tier": "high",
                    "confidence": "high",
                    "enabled": True,
                }
            )
            for display_name in ("Dingy Swallowtail", "Small Citrus Butterfly"):
                rows.append(
                    {
                        "accepted_taxon_key": key,
                        "verbatim_name": display_name,
                        "display_name": display_name,
                        "language": "eng",
                        "script": "Latn",
                        "name_class": "vernacular",
                        "source": "GBIF",
                        "source_record_id": f"{key}:{display_name}",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "high",
                        "enabled": True,
                    }
                )
        return rows

    full_payload = {
        "source": "GBIF",
        "source_version": "fixture",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "taxa": [*lineage, *species_taxa(species)],
        "names": names(species),
    }
    full_frames, _ = compile_registry_frames(
        full_payload,
        source_ref="memory://full",
        output_ref="memory://full",
        registry_version="full",
        scope_path=scope,
    )
    slice_payload = {
        **full_payload,
        "taxa": [*lineage, *species_taxa([species[0]])],
        "names": names([species[0]]),
    }

    slice_frames, _ = compile_registry_frames(
        slice_payload,
        source_ref="memory://slice",
        output_ref="memory://slice",
        registry_version="slice",
        scope_path=scope,
        global_names_for_collision=full_frames["names.parquet"],
    )

    names_frame = slice_frames["names.parquet"]
    collision_names = names_frame.filter(pl.col("display_name").is_in(["Dingy Swallowtail", "Small Citrus Butterfly"])).sort("display_name")
    queries = slice_frames["flickr_query_definitions.parquet"]

    assert collision_names.select("query_eligible").to_series().to_list() == [True, True]
    assert collision_names.select("query_disabled_reason").to_series().to_list() == ["", ""]
    assert {"Dingy Swallowtail", "Small Citrus Butterfly"}.issubset(set(names_frame["display_name"].to_list()))
    for term in ("dingy swallowtail", "small citrus butterfly"):
        term_queries = queries.filter(pl.col("normalized_query_term") == term)
        assert term_queries.height == 2
        assert set(term_queries["search_field"].to_list()) == {"tags", "text"}


def test_registry_compile_enriched_cli_writes_expanded_outputs(tmp_path, capsys) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "CoL Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:name:1",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            }
        ],
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "registry",
            "compile-enriched",
            "--registry-dir",
            str(registry),
            "--registry-version",
            "enriched",
            "--scope-json",
            str(scope),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "enriched"
    assert payload["registry_dir"] == str(registry)
    assert (registry / "external_taxon_links.parquet").exists()
    assert (registry / "source_name_assertions.parquet").exists()


def test_compile_enriched_registry_filters_stale_sources_when_requested(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "CoL Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:name:1",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Old Wiki Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "Wikidata",
                "source_record_id": "wikidata:old",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "low",
                "enabled": True,
                "review_state": "accepted",
            },
        ],
    )

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
        requested_sources=("col", "inaturalist", "itis"),
    )

    names = pl.read_parquet(registry / "names.parquet")
    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")

    assert manifest["enrichment_sources"] == ["col", "inaturalist", "itis"]
    assert "CoL Lime" in names["display_name"].to_list()
    assert "Old Wiki Lime" not in names["display_name"].to_list()
    assert "Wikidata" not in assertions["source"].to_list()


class RecordingEnrichmentClient:
    def __init__(self, source: str, display_name: str, *, enabled: bool = True) -> None:
        self.source = source
        self.display_name = display_name
        self.enabled = enabled
        self.contexts = []

    def enrich_species(self, context):  # noqa: ANN001 - test double checks context shape.
        self.contexts.append(context)
        source_taxon_id = f"{self.source}:taxon:{context.accepted_taxon_key}"
        return {
            "name_assertions": [
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "display_name": self.display_name,
                    "language": "eng",
                    "script": "Latn",
                    "region": "",
                    "name_class": "vernacular",
                    "source": self.source,
                    "source_record_id": f"{self.source}:name:{context.accepted_taxon_key}",
                    "source_taxon_id": source_taxon_id if self.source == "Wikidata" else "",
                    "lineage_check": "accepted_taxon_key" if self.source == "Wikidata" else "",
                    "trust_tier": "T2",
                    "precision_tier": "medium",
                    "confidence": "high",
                    "enabled": self.enabled,
                    "review_state": "accepted" if self.enabled else "candidate",
                    "disabled_reason": "" if self.enabled else "source_name_requires_review",
                }
            ],
            "external_links": [
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "source": self.source,
                    "source_taxon_id": source_taxon_id,
                    "match_method": "scientific_name",
                    "match_confidence": "high",
                    "lineage_check": "accepted_taxon_key",
                }
            ],
            "source_snapshots": [
                {
                    "source": self.source,
                    "source_version": "fixture",
                    "retrieved_at": "2026-06-20T00:00:00+00:00",
                    "source_path": f"memory://{self.source}",
                    "source_response_hash": f"sha256:{self.source}",
                    "licence": "",
                }
            ],
        }


def test_build_enrichment_sources_feeds_species_context_to_priority_services(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)
    clients = {
        "col": RecordingEnrichmentClient("CoL", "CoL Lime"),
        "inaturalist": RecordingEnrichmentClient("iNaturalist", "iNat Lime"),
        "itis": RecordingEnrichmentClient("ITIS", "ITIS Lime"),
    }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col", "inaturalist", "itis"),
        clients=clients,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    links = pl.read_parquet(registry / "external_taxon_links.parquet")
    assert manifest["source_order"] == ["col", "inaturalist", "itis"]
    assert manifest["species_seen"] == 1
    assert assertions.select("source").to_series().to_list() == ["CoL", "iNaturalist", "ITIS"]
    assert links.height == 3
    assert [client.contexts[0].accepted_scientific_name for client in clients.values()] == ["Papilio demoleus"] * 3
    assert clients["col"].contexts[0].current_names == ("Papilio demoleus",)


def test_build_enrichment_sources_checkpoints_logs_and_reports(tmp_path, caplog) -> None:
    registry, _scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes"))
    caplog.set_level(logging.INFO, logger="biominer.registry.enrichment")

    def client_factory():
        return {"col": RecordingEnrichmentClient("CoL", "Source Lime")}

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col",),
        client_factory=client_factory,
        workers=2,
        progress_every=1,
        checkpoint_every=1,
        max_retries=0,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    staged_manifest = json.loads((registry / "enrichment_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "reports" / "registry_enrichment_base.json").read_text(encoding="utf-8"))
    log_text = "\n".join(record.getMessage() for record in caplog.records)

    assert manifest["workers"] == 2
    assert manifest["completed_species"] == 3
    assert manifest["name_assertion_rows"] == 3
    assert staged_manifest["completed_species"] == 3
    assert assertions.select("accepted_taxon_key").to_series().to_list() == ["gbif:100", "gbif:101", "gbif:102"]
    assert report["artifact_bytes"]["source_name_assertions.parquet"] > 0
    assert "registry.enrichment.checkpoint_write" in log_text
    assert "source_name_assertions.parquet" in log_text
    assert "name_assertion_rows=3" in log_text


class ConcurrencyTrackingClient:
    def __init__(self, source: str, *, delay_seconds: float) -> None:
        self.source = source
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def enrich_species(self, context):  # noqa: ANN001 - test double tracks concurrent calls.
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            return {
                "name_assertions": [
                    {
                        "accepted_taxon_key": context.accepted_taxon_key,
                        "display_name": f"{self.source} {context.accepted_scientific_name}",
                        "language": "eng",
                        "script": "Latn",
                        "region": "",
                        "name_class": "vernacular",
                        "source": self.source,
                        "source_record_id": f"{self.source}:{context.accepted_taxon_key}",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "high",
                        "enabled": True,
                        "review_state": "accepted",
                    }
                ],
                "external_links": [],
                "source_snapshots": [],
            }
        finally:
            with self.lock:
                self.active -= 1


def test_inaturalist_source_is_limited_to_one_concurrent_query(tmp_path) -> None:
    registry, _scope = _write_base_registry(
        tmp_path,
        species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes", "Papilio xuthus"),
    )
    trackers = {
        "col": ConcurrencyTrackingClient("CoL", delay_seconds=0.03),
        "inaturalist": ConcurrencyTrackingClient("iNaturalist", delay_seconds=0.03),
        "itis": ConcurrencyTrackingClient("ITIS", delay_seconds=0.06),
        "wikidata": ConcurrencyTrackingClient("Wikidata", delay_seconds=0.03),
    }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col", "inaturalist", "itis", "wikidata"),
        clients=trackers,
        workers=4,
        progress_every=4,
        checkpoint_every=4,
        report_dir=tmp_path / "reports",
    )

    assert trackers["col"].max_active > 1
    assert trackers["itis"].max_active > 1
    assert trackers["inaturalist"].max_active == 1
    assert trackers["wikidata"].max_active == 1
    assert manifest["source_worker_limits"] == {"col": 4, "inaturalist": 1, "itis": 4, "wikidata": 1}


def test_default_enrichment_clients_include_wikidata_gbif_vernacular_and_taxref() -> None:
    clients = default_enrichment_clients(max_retries=0)

    assert list(clients) == [
        "col",
        "ncbi",
        "open_tree",
        "inaturalist",
        "itis",
        "tmd_de",
        "wikidata",
        "gbif_vernacular",
        "taxref_fr",
        "boi_india_en",
        "bharat_ki_titliya_hi",
        "karnataka_chitte_kn",
        "papilio_demoleus_multilingual_t5",
    ]
    assert clients["ncbi"].__class__.__name__ == "NCBIBulkClient"
    assert clients["open_tree"].__class__.__name__ == "OpenTreeBulkClient"
    assert clients["gbif_vernacular"].__class__.__name__ == "GBIFVernacularClient"
    assert clients["taxref_fr"].__class__.__name__ == "TAXREFFrenchClient"
    assert clients["boi_india_en"].__class__.__name__ == "StaticVernacularSourceClient"
    assert clients["bharat_ki_titliya_hi"].__class__.__name__ == "StaticVernacularSourceClient"
    assert clients["karnataka_chitte_kn"].__class__.__name__ == "StaticVernacularSourceClient"
    assert clients["papilio_demoleus_multilingual_t5"].__class__.__name__ == "StaticVernacularSourceClient"
    assert clients["papilio_demoleus_multilingual_t5"].source_name == "Multilingual T5"
    assert clients["wikidata"].__class__.__name__ == "WikidataClient"


def test_inaturalist_daily_budget_writes_ledger_and_stops_cleanly(tmp_path) -> None:
    registry, _scope = _write_base_registry(
        tmp_path,
        species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes"),
    )
    client = RecordingEnrichmentClient("iNaturalist", "iNat Lime")

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("inaturalist",),
        clients={"inaturalist": client},
        inaturalist_daily_request_limit=2,
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    ledger = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)

    assert manifest["status"] == "budget_exhausted"
    assert manifest["source_budget_exhausted"] == ["inaturalist"]
    assert client.contexts[0].accepted_scientific_name == "Papilio demoleus"
    assert client.contexts[1].accepted_scientific_name == "Papilio machaon"
    assert len(client.contexts) == 2
    assert assertions.height == 2
    assert ledger.filter((pl.col("source") == "inaturalist") & (pl.col("status") == "complete")).height == 2
    assert ledger.select(pl.col("request_count").sum()).item() == 2


def test_enrichment_resume_skips_completed_source_work(tmp_path) -> None:
    registry, _scope = _write_base_registry(
        tmp_path,
        species_names=("Papilio demoleus", "Papilio machaon"),
    )
    first_client = RecordingEnrichmentClient("iNaturalist", "iNat Lime")
    build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("inaturalist",),
        clients={"inaturalist": first_client},
        inaturalist_daily_request_limit=1,
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )
    second_client = RecordingEnrichmentClient("iNaturalist", "iNat Lime")

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("inaturalist",),
        clients={"inaturalist": second_client},
        inaturalist_daily_request_limit=10,
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    ledger = pl.read_parquet(registry / SOURCE_WORK_LEDGER_FILE)

    assert manifest["status"] == "complete"
    assert [context.accepted_scientific_name for context in second_client.contexts] == ["Papilio machaon"]
    assert assertions.height == 2
    assert ledger.filter(pl.col("status") == "complete").height == 2


def test_registry_enrich_sources_cli_writes_sources_into_registry_dir(tmp_path, capsys, monkeypatch) -> None:
    registry, _scope = _write_base_registry(tmp_path)

    def fake_build(
        *,
        registry_dir,
        enrichment_dir,
        sources,
        workers,
        progress_every,
        checkpoint_every,
        max_retries,
        inaturalist_daily_request_limit,
        limit,
        report_dir,
    ):  # noqa: ANN001 - CLI wiring test.
        output_dir = enrichment_dir or registry_dir
        return write_enrichment_sources(
            output_dir,
            name_assertions=[
                {
                    "accepted_taxon_key": "gbif:100",
                    "display_name": "CLI Lime",
                    "language": "eng",
                    "script": "Latn",
                    "name_class": "vernacular",
                    "source": "CoL",
                    "source_record_id": "col:cli",
                    "trust_tier": "T2",
                    "precision_tier": "medium",
                    "confidence": "high",
                    "enabled": True,
                    "review_state": "accepted",
                }
            ],
        ) | {
            "registry_dir": str(registry_dir),
            "enrichment_dir": str(output_dir),
            "source_order": list(sources),
            "species_seen": 1,
            "errors": [],
            "workers": workers,
            "progress_every": progress_every,
            "checkpoint_every": checkpoint_every,
            "max_retries": max_retries,
            "inaturalist_daily_request_limit": inaturalist_daily_request_limit,
            "limit": limit,
            "report_dir": str(report_dir),
        }

    monkeypatch.setattr("biominer.cli.build_enrichment_sources_from_registry", fake_build)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "registry",
            "enrich-sources",
            "--registry-dir",
            str(registry),
            "--sources",
            "col,inaturalist,itis",
            "--workers",
            "3",
            "--progress-every",
            "4",
            "--checkpoint-every",
            "5",
            "--max-retries",
            "6",
            "--inaturalist-daily-request-limit",
            "8",
            "--limit",
            "7",
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_dir"] == str(registry)
    assert payload["source_order"] == ["col", "inaturalist", "itis"]
    assert payload["workers"] == 3
    assert payload["progress_every"] == 4
    assert payload["checkpoint_every"] == 5
    assert payload["max_retries"] == 6
    assert payload["inaturalist_daily_request_limit"] == 8
    assert payload["limit"] == 7
    assert payload["report_dir"] == str(tmp_path / "reports")
    assert (registry / "source_name_assertions.parquet").exists()


def test_source_clients_reuse_http_client_between_requests(monkeypatch) -> None:
    created_clients = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float, headers: dict[str, str] | None = None) -> None:
            self.base_url = base_url
            self.timeout = timeout
            self.headers = headers or {}
            self.paths = []
            created_clients.append(self)

        def get(self, path: str, params):
            self.paths.append(path)
            if path.endswith("searchByScientificName"):
                return FakeResponse({"scientificNames": [{"combinedName": "Papilio demoleus", "tsn": "123"}]})
            return FakeResponse({"commonNames": [{"commonName": "Lime Swallowtail", "language": "eng"}]})

    monkeypatch.setattr("biominer.registry.enrichment_sources.httpx.Client", FakeHTTPClient)

    client = ITISClient()
    result = client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert len(created_clients) == 1
    assert created_clients[0].paths == [
        "/ITISWebService/jsonservice/searchByScientificName",
        "/ITISWebService/jsonservice/getCommonNamesFromTSN",
    ]
    assert result["name_assertions"][0]["display_name"] == "Lime Swallowtail"


def test_json_get_sends_user_agent_and_retries_transient_errors(monkeypatch) -> None:
    attempts = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self.headers = {"Retry-After": "0"} if status_code == 503 else {}
            self._payload = payload
            self.request = httpx.Request("GET", "https://example.test/resource")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=self.request, response=self)

        def json(self):
            return self._payload

    class FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float, headers: dict[str, str]) -> None:
            self.base_url = base_url
            self.timeout = timeout
            self.headers = headers

        def get(self, path: str, params):
            attempts.append({"path": path, "params": params, "headers": self.headers})
            if len(attempts) == 1:
                return FakeResponse(503, {"error": "temporary"})
            return FakeResponse(200, {"results": [{"name": "ok"}]})

    monkeypatch.setattr("biominer.registry.enrichment_sources.httpx.Client", FakeHTTPClient)
    payload = _json_get("https://example.test", max_retries=2, sleep=lambda _seconds: None)("/resource", {"q": "Papilio"})

    assert payload == {"results": [{"name": "ok"}]}
    assert len(attempts) == 2
    assert attempts[0]["headers"]["User-Agent"].startswith("BioMiner/")


def test_json_get_does_not_retry_permanent_4xx(monkeypatch) -> None:
    attempts = []

    class FakeResponse:
        status_code = 403
        headers = {}
        request = httpx.Request("GET", "https://example.test/resource")

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("forbidden", request=self.request, response=self)

        def json(self):
            return {"error": "forbidden"}

    class FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float, headers: dict[str, str]) -> None:
            self.headers = headers

        def get(self, path: str, params):
            attempts.append((path, params))
            return FakeResponse()

    monkeypatch.setattr("biominer.registry.enrichment_sources.httpx.Client", FakeHTTPClient)

    try:
        _json_get("https://example.test", max_retries=2, sleep=lambda _seconds: None)("/resource", {})
    except httpx.HTTPStatusError:
        pass
    else:  # pragma: no cover - assertion path is clearer than pytest.raises import churn here.
        raise AssertionError("expected permanent HTTP error")

    assert len(attempts) == 1


def test_json_get_falls_back_to_replacement_decode_for_unicode_errors(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {}

        @property
        def content(self) -> bytes:
            return b'{"results":[{"commonName":"Lime \xff Swallowtail"}]}'

        def raise_for_status(self) -> None:
            return None

        def json(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    class FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float, headers: dict[str, str]) -> None:
            self.headers = headers

        def get(self, path: str, params):
            return FakeResponse()

    monkeypatch.setattr("biominer.registry.enrichment_sources.httpx.Client", FakeHTTPClient)

    payload = _json_get("https://example.test", max_retries=0, sleep=lambda _seconds: None)("/resource", {})

    assert payload == {"results": [{"commonName": "Lime � Swallowtail"}]}
