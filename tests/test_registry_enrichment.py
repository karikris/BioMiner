from __future__ import annotations

import json
import logging
import threading
import time

import httpx
import polars as pl

from biominer.cli import build_parser, run
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.enrichment import (
    SOURCE_WORK_LEDGER_FILE,
    SpeciesContext,
    build_enrichment_sources_from_registry,
    compile_enriched_registry,
    default_enrichment_clients,
    write_enrichment_sources,
)
from biominer.registry.enrichment_sources import CatalogueOfLifeClient, INaturalistClient, ITISClient, TMDGermanClient, _json_get
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
    assert "Citrus butterfly" not in queries["normalized_query_term"].to_list()
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


def test_inaturalist_source_uses_non_truncating_taxa_page_size() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - source test double.
        requests.append((path, dict(params)))
        return {"results": []}

    client = INaturalistClient(http_get=fake_get)
    client.enrich_species(_species_context())

    assert requests == [("/v1/taxa", {"q": "Papilio demoleus", "rank": "species", "per_page": 200})]


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
    zh_query = queries.filter((pl.col("normalized_match_key") == "鳳蝶") & (pl.col("search_field") == "tags")).row(0, named=True)

    assert (pt_name["language"], pt_name["script"], pt_name["region"], pt_name["query_eligible"]) == ("por", "Latn", "BR", True)
    assert (zh_name["language"], zh_name["script"], zh_name["region"], zh_name["query_eligible"]) == ("zho", "Hant", "", True)
    assert (pt_query["language"], pt_query["script"], pt_query["region"]) == ("por", "Latn", "BR")
    assert (zh_query["language"], zh_query["script"], zh_query["region"]) == ("zho", "Hant", "")


def test_compile_enriched_registry_keeps_same_label_script_variants_distinct(tmp_path) -> None:
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
    script_queries = queries.filter(pl.col("normalized_match_key") == "青鳳蝶").sort(["script", "search_field"])

    assert script_names.height == 2
    assert script_names.select("script").to_series().to_list() == ["Hans", "Hant"]
    assert script_queries.select(["script", "search_field"]).to_dicts() == [
        {"script": "Hans", "search_field": "tags"},
        {"script": "Hans", "search_field": "text"},
        {"script": "Hant", "search_field": "tags"},
        {"script": "Hant", "search_field": "text"},
    ]


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
                    "source_taxon_id": f"{self.source}:taxon:{context.accepted_taxon_key}",
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


def test_default_enrichment_clients_include_wikidata() -> None:
    clients = default_enrichment_clients(max_retries=0)

    assert list(clients) == ["col", "inaturalist", "itis", "tmd_de", "wikidata"]
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
        sources,
        workers,
        progress_every,
        checkpoint_every,
        max_retries,
        inaturalist_daily_request_limit,
        limit,
        report_dir,
    ):  # noqa: ANN001 - CLI wiring test.
        return write_enrichment_sources(
            registry_dir,
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
