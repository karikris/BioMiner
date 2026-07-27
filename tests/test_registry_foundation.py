from __future__ import annotations

import json

import polars as pl

from biominer.flickr_fetch.query_planner import load_registry_flickr_queries_from_frame
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.scope import load_scope


BUTTERFLY_FAMILIES = (
    "Hesperiidae",
    "Papilionidae",
    "Pieridae",
    "Lycaenidae",
    "Riodinidae",
    "Nymphalidae",
    "Hedylidae",
)


def _write_fixture(path) -> None:
    family_rows = [
        {
            "accepted_taxon_key": f"gbif:fam:{family}",
            "scientific_name": family,
            "rank": "FAMILY",
            "parent_key": "gbif:1",
            "family_key": f"gbif:fam:{family}",
            "family": family,
            "genus_key": "",
            "genus": "",
            "species_key": "",
            "species": "",
        }
        for family in BUTTERFLY_FAMILIES
    ]
    path.write_text(
        json.dumps(
            {
                "source": "fixture",
                "source_version": "2026-06-20",
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
                    *family_rows,
                    {
                        "accepted_taxon_key": "gbif:100",
                        "scientific_name": "Papilio demoleus",
                        "rank": "SPECIES",
                        "parent_key": "gbif:90",
                        "family_key": "gbif:10",
                        "family": "Papilionidae",
                        "genus_key": "gbif:90",
                        "genus": "Papilio",
                        "species_key": "gbif:100",
                        "species": "Papilio demoleus",
                    },
                ],
                "names": [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "Papilio demoleus",
                        "display_name": "Papilio demoleus",
                        "language": "la",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": "gbif:100",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    },
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "Lime Butterfly",
                        "display_name": "Lime Butterfly",
                        "language": "en",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "vernacular",
                        "source": "fixture",
                        "source_record_id": "fixture:name:1",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "medium",
                        "enabled": True,
                    },
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "butterfly",
                        "display_name": "butterfly",
                        "language": "en",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "vernacular_alias",
                        "source": "fixture",
                        "source_record_id": "fixture:name:2",
                        "trust_tier": "T7",
                        "precision_tier": "broad",
                        "confidence": "low",
                        "enabled": False,
                        "disabled_reason": "generic_word",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_scope_config_has_seven_butterfly_families() -> None:
    scope = load_scope()

    assert scope.root_scientific_name == "Papilionoidea"
    assert scope.root_rank == "SUPERFAMILY"
    assert scope.included_families == BUTTERFLY_FAMILIES


def test_compile_registry_fixture_writes_normalized_parquet_and_manifest(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    assert manifest["registry_version"] == "test-registry"
    assert manifest["taxa_rows"] == 9
    assert manifest["name_rows"] == 3
    assert manifest["query_definition_rows"] == 4
    assert manifest["qa_status"] == "passed"
    assert manifest["qa_fatal_count"] == 0
    assert manifest["qa_warning_count"] == 1
    assert (output / "taxa.parquet").exists()
    assert (output / "names.parquet").exists()
    assert (output / "name_collision_ledger.parquet").exists()
    assert (output / "name_evidence.parquet").exists()
    assert (output / "source_snapshots.parquet").exists()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["name_collision_ledger_rows"] == 0
    assert manifest["query_blocking_name_collision_rows"] == 0

    taxa = pl.read_parquet(output / "taxa.parquet")
    assert set(taxa["rank"]) <= {
        "KINGDOM",
        "PHYLUM",
        "CLASS",
        "ORDER",
        "SUPERFAMILY",
        "FAMILY",
        "GENUS",
        "SPECIES",
    }
    assert taxa.filter(pl.col("rank") == "SUPERFAMILY").select("scientific_name").item() == "Papilionoidea"

    paths = pl.read_parquet(output / "species_paths.parquet")
    assert paths.select("superfamily").item() == "Papilionoidea"
    assert paths.select("superfamily_node_id").item() == "gbif:1"

    names = pl.read_parquet(output / "names.parquet")
    assert names.select("normalized_match_key").to_series().to_list() == [
        "papilio demoleus",
        "lime butterfly",
        "butterfly",
    ]


def test_compile_registry_fixture_emits_atomic_flickr_queries_in_name_priority_order(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)

    compile_registry_fixture(source, output, registry_version="test-registry")

    queries = pl.read_parquet(output / "flickr_query_definitions.parquet").sort("search_priority")
    assert queries.select("search_field").to_series().to_list() == ["tags", "text", "tags", "text"]
    assert queries.select("source_term").to_series().to_list() == [
        "Papilio demoleus",
        "Papilio demoleus",
        "Lime Butterfly",
        "Lime Butterfly",
    ]
    assert queries.select("normalized_query_term").to_series().to_list() == [
        "papilio demoleus",
        "papilio demoleus",
        "lime butterfly",
        "lime butterfly",
    ]
    assert queries.select("query_definition_id").to_series().n_unique() == 4
    assert queries.select(["normalized_query_term", "search_field", "region"]).unique().height == 4


def test_compile_registry_fixture_keeps_broad_scientific_names_out_of_queries(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["taxa"].append(
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
        }
    )
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:fam:Papilionidae",
                "verbatim_name": "Papilionidae",
                "display_name": "Papilionidae",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:fam:Papilionidae",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:90",
                "verbatim_name": "Papilio",
                "display_name": "Papilio",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:90",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    broad_names = names.filter(pl.col("display_name").is_in(["Papilionidae", "Papilio"])).sort("display_name")

    assert broad_names.select("query_eligible").to_series().to_list() == [False, False]
    assert broad_names.select("query_disabled_reason").to_series().to_list() == ["broad_scientific_name", "broad_scientific_name"]
    assert not set(queries.select("source_term").to_series().to_list()) & {"Papilionidae", "Papilio"}


def test_compile_registry_fixture_stores_enabled_but_query_ineligible_weak_names(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Common Lime Butterfly",
                "display_name": "Common Lime Butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:name:phrase",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "lime",
                "display_name": "lime",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular_alias",
                "source": "fixture",
                "source_record_id": "fixture:name:lime",
                "trust_tier": "T2",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "swallowtails",
                "display_name": "swallowtails",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular_alias",
                "source": "fixture",
                "source_record_id": "fixture:name:swallowtails",
                "trust_tier": "T2",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Swallowtail Butterfly",
                "display_name": "Swallowtail Butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular_alias",
                "source": "fixture",
                "source_record_id": "fixture:name:swallowtail-butterfly",
                "trust_tier": "T2",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "papillon",
                "display_name": "papillon",
                "language": "fr",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "generated_translation",
                "source": "MyMemory",
                "source_record_id": "mymemory:papillon",
                "trust_tier": "T5",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
                "review_state": "candidate",
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    by_name = {row["normalized_match_key"]: row for row in names.to_dicts()}

    assert by_name["common lime butterfly"]["enabled"] is True
    assert by_name["common lime butterfly"]["query_eligible"] is True
    assert by_name["common lime butterfly"]["species_specificity_score"] > by_name["lime"]["species_specificity_score"]
    assert by_name["swallowtails"]["query_eligible"] is False
    assert by_name["swallowtails"]["query_disabled_reason"] == "plural_group_name"
    for weak_name in ("lime", "swallowtails", "swallowtail butterfly", "papillon"):
        assert by_name[weak_name]["enabled"] is True
        assert by_name[weak_name]["query_eligible"] is False
        assert by_name[weak_name]["query_disabled_reason"]
    assert "Common Lime Butterfly" in queries["source_term"].to_list()
    assert not {"lime", "swallowtails", "swallowtail butterfly", "papillon"} & set(queries["normalized_match_key"].to_list())
    assert manifest["query_eligible_name_rows"] == names.filter(pl.col("query_eligible")).height
    assert manifest["query_ineligible_name_rows"] == names.filter(pl.col("enabled") & ~pl.col("query_eligible")).height


def test_compile_registry_fixture_allows_reviewed_generated_translation_queries(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["names"].append(
        {
            "accepted_taxon_key": "gbif:100",
            "verbatim_name": "Limettenfalter",
            "display_name": "Limettenfalter",
            "language": "de",
            "script": "Latn",
            "region": "",
            "bbox": "",
            "name_class": "generated_translation",
            "source": "MyMemory",
            "source_record_id": "mymemory:reviewed",
            "trust_tier": "T5",
            "precision_tier": "medium",
            "confidence": "medium",
            "enabled": True,
            "review_state": "reviewed",
        }
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    reviewed = names.filter(pl.col("normalized_match_key") == "limettenfalter")

    assert reviewed.select("query_eligible").to_series().to_list() == [True]
    assert queries.filter(pl.col("normalized_match_key") == "limettenfalter").select("search_field").to_series().to_list() == ["tags", "text"]


def test_compile_registry_fixture_canonicalizes_cross_species_common_name_collisions(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["taxa"].append(
        {
            "accepted_taxon_key": "gbif:200",
            "scientific_name": "Danaus plexippus",
            "rank": "SPECIES",
            "parent_key": "gbif:190",
            "family_key": "gbif:20",
            "family": "Nymphalidae",
            "genus_key": "gbif:190",
            "genus": "Danaus",
            "species_key": "gbif:200",
            "species": "Danaus plexippus",
        }
    )
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Danaus plexippus",
                "display_name": "Danaus plexippus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:200",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Lime Butterfly",
                "display_name": "Lime Butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:name:danaus-lime",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    ledger = pl.read_parquet(output / "name_collision_ledger.parquet")
    lime_names = names.filter(pl.col("normalized_match_key") == "lime butterfly").sort("accepted_taxon_key")

    assert lime_names.select("query_eligible").to_series().to_list() == [True, True]
    assert lime_names.select("query_disabled_reason").to_series().to_list() == ["", ""]
    assert lime_names.filter(pl.col("is_canonical_keyword")).height == 1
    assert lime_names.filter(pl.col("suppressed_duplicate")).height == 1
    lime_queries = queries.filter(pl.col("normalized_match_key") == "lime butterfly")
    assert lime_queries.height == 2
    assert set(lime_queries["search_field"].to_list()) == {"tags", "text"}
    assert set(queries["normalized_match_key"].to_list()) == {
        "papilio demoleus",
        "danaus plexippus",
        "lime butterfly",
    }
    assert ledger.height == 1
    assert ledger.row(0, named=True)["normalized_match_key"] == "lime butterfly"
    assert ledger.row(0, named=True)["language"] == "eng"
    assert ledger.row(0, named=True)["accepted_taxon_keys"] == ["gbif:100", "gbif:200"]
    assert ledger.row(0, named=True)["collision_status"] == "query_blocking"
    assert manifest["name_collision_ledger_rows"] == 1
    assert manifest["query_blocking_name_collision_rows"] == 1


def test_compile_registry_fixture_does_not_warn_for_cross_language_same_text(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["taxa"].append(
        {
            "accepted_taxon_key": "gbif:200",
            "scientific_name": "Danaus plexippus",
            "rank": "SPECIES",
            "parent_key": "gbif:190",
            "family_key": "gbif:20",
            "family": "Nymphalidae",
            "genus_key": "gbif:190",
            "genus": "Danaus",
            "species_key": "gbif:200",
            "species": "Danaus plexippus",
        }
    )
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Danaus plexippus",
                "display_name": "Danaus plexippus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:200",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Lime Butterfly",
                "display_name": "Lime Butterfly",
                "language": "es",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:name:danaus-lime-es",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    ledger = pl.read_parquet(output / "name_collision_ledger.parquet")
    qa = pl.read_parquet(output / "qa_findings.parquet").to_dicts()
    lime_names = names.filter(pl.col("normalized_match_key") == "lime butterfly").sort("accepted_taxon_key")

    assert lime_names.select("language").to_series().to_list() == ["eng", "spa"]
    assert lime_names.select("query_eligible").to_series().to_list() == [True, True]
    assert ledger.is_empty()
    lime_queries = queries.filter(pl.col("normalized_match_key") == "lime butterfly")
    assert lime_queries.height == 2
    assert set(lime_queries["search_field"].to_list()) == {"tags", "text"}
    assert not any(row["code"] == "normalized_name_collision" and row["subject"] == "lime butterfly" for row in qa)
    assert manifest["name_collision_ledger_rows"] == 0
    assert manifest["query_blocking_name_collision_rows"] == 0


def test_compile_registry_fixture_canonicalizes_reviewed_collision_without_species_specific_signal(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    for name in payload["names"]:
        if name["display_name"] == "Lime Butterfly":
            name["review_state"] = "reviewed"
            name["precision_tier"] = "medium"
    payload["taxa"].append(
        {
            "accepted_taxon_key": "gbif:200",
            "scientific_name": "Danaus plexippus",
            "rank": "SPECIES",
            "parent_key": "gbif:190",
            "family_key": "gbif:20",
            "family": "Nymphalidae",
            "genus_key": "gbif:190",
            "genus": "Danaus",
            "species_key": "gbif:200",
            "species": "Danaus plexippus",
        }
    )
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Danaus plexippus",
                "display_name": "Danaus plexippus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:200",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Lime Butterfly",
                "display_name": "Lime Butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:name:danaus-reviewed-lime",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "reviewed",
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    lime_names = names.filter(pl.col("normalized_match_key") == "lime butterfly").sort("accepted_taxon_key")

    assert lime_names.select("query_eligible").to_series().to_list() == [True, True]
    assert lime_names.select("query_disabled_reason").to_series().to_list() == ["", ""]
    assert lime_names.filter(pl.col("is_canonical_keyword")).height == 1
    assert lime_names.filter(pl.col("suppressed_duplicate")).height == 1
    assert queries.filter(pl.col("normalized_match_key") == "lime butterfly").height == 2


def test_compile_registry_fixture_allows_query_approved_species_specific_collision(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    for name in payload["names"]:
        if name["display_name"] == "Lime Butterfly":
            name["review_state"] = "query_approved"
            name["precision_tier"] = "high"
    payload["taxa"].append(
        {
            "accepted_taxon_key": "gbif:200",
            "scientific_name": "Danaus plexippus",
            "rank": "SPECIES",
            "parent_key": "gbif:190",
            "family_key": "gbif:20",
            "family": "Nymphalidae",
            "genus_key": "gbif:190",
            "genus": "Danaus",
            "species_key": "gbif:200",
            "species": "Danaus plexippus",
        }
    )
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Danaus plexippus",
                "display_name": "Danaus plexippus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:200",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Lime Butterfly",
                "display_name": "Lime Butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:name:danaus-approved-lime",
                "trust_tier": "T2",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
                "review_state": "query_approved",
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    compile_registry_fixture(source, output, registry_version="test-registry")

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    lime_names = names.filter(pl.col("normalized_match_key") == "lime butterfly").sort("accepted_taxon_key")

    assert lime_names.select("query_eligible").to_series().to_list() == [True, True]
    assert lime_names.select("query_disabled_reason").to_series().to_list() == ["", ""]
    assert lime_names.filter(pl.col("is_canonical_keyword")).height == 1
    assert lime_names.filter(pl.col("suppressed_duplicate")).height == 1
    assert queries.filter(pl.col("normalized_match_key") == "lime butterfly").height == 2


def test_accepted_scientific_queries_schedule_before_synonyms(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["names"].append(
        {
            "accepted_taxon_key": "gbif:100",
            "verbatim_name": "Papilio annamiticus",
            "display_name": "Papilio annamiticus",
            "language": "la",
            "script": "Latn",
            "region": "",
            "bbox": "",
            "name_class": "scientific_synonym",
            "source": "GBIF",
            "source_record_id": "gbif:synonym:1",
            "trust_tier": "T1",
            "precision_tier": "high",
            "confidence": "high",
            "enabled": True,
        }
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    compile_registry_fixture(source, output, registry_version="test-registry")

    queries = load_registry_flickr_queries_from_frame(pl.read_parquet(output / "flickr_query_definitions.parquet"))
    assert [(query.term, query.search_field) for query in queries[:2]] == [
        ("Papilio demoleus", "tags"),
        ("Papilio demoleus", "text"),
    ]
    assert all(query.min_upload_date is None and query.max_upload_date is None for query in queries)


def test_compile_registry_fixture_marks_missing_configured_family_as_fatal(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["taxa"] = [row for row in payload["taxa"] if row["scientific_name"] != "Hedylidae"]
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    qa = pl.read_parquet(output / "qa_findings.parquet")
    assert manifest["qa_status"] == "failed"
    assert {"severity": "fatal", "code": "configured_family_not_in_source", "subject": "Hedylidae"} in qa.to_dicts()


def test_compile_registry_fixture_keeps_query_ids_unique_for_repeated_name_ids(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["taxa"].append(
        {
            "accepted_taxon_key": "gbif:200",
            "scientific_name": "Pieris brassicae",
            "rank": "SPECIES",
            "parent_key": "gbif:190",
            "family_key": "gbif:fam:Pieridae",
            "family": "Pieridae",
            "genus_key": "gbif:190",
            "genus": "Pieris",
            "species_key": "gbif:200",
            "species": "Pieris brassicae",
        }
    )
    payload["names"].extend(
        [
            dict(payload["names"][0]),
            {
                "accepted_taxon_key": "gbif:200",
                "verbatim_name": "Lime Butterfly",
                "display_name": "Lime Butterfly",
                "language": "",
                "script": "",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    qa = pl.read_parquet(output / "qa_findings.parquet").to_dicts()
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")

    assert manifest["qa_status"] == "passed"
    assert queries.select("query_definition_id").to_series().n_unique() == queries.height
    assert not any(row["code"] == "duplicate_query_definition_id" for row in qa)
    assert not any(row["code"] == "normalized_name_collision" and row["subject"] == "lime butterfly" for row in qa)
    assert {"severity": "warning", "code": "weak_language_or_script_metadata", "subject": "Lime Butterfly"} in qa
    assert {"severity": "warning", "code": "missing_name_source_evidence", "subject": "Lime Butterfly"} in qa


def test_compile_registry_fixture_warns_on_deterministic_language_script_mismatch(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["names"].extend(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "クジャクカジカ",
                "display_name": "クジャクカジカ",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:jpn-mistag",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Бычок-бабочка",
                "display_name": "Бычок-бабочка",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:rus-mistag",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
            {
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "正しい日本語",
                "display_name": "正しい日本語",
                "language": "jpn",
                "script": "Jpan",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "fixture",
                "source_record_id": "fixture:jpn-ok",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
            },
        ]
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = compile_registry_fixture(source, output, registry_version="test-registry")

    qa = pl.read_parquet(output / "qa_findings.parquet").to_dicts()

    assert manifest["qa_status"] == "passed"
    assert {"severity": "warning", "code": "language_script_mismatch", "subject": "クジャクカジカ"} in qa
    assert {"severity": "warning", "code": "language_script_mismatch", "subject": "Бычок-бабочка"} in qa
    assert {"severity": "warning", "code": "language_script_mismatch", "subject": "正しい日本語"} not in qa
