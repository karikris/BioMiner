from __future__ import annotations

import json

import polars as pl

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
    assert manifest["taxa_rows"] == 3
    assert manifest["name_rows"] == 3
    assert manifest["query_definition_rows"] == 4
    assert manifest["qa_status"] == "passed"
    assert (output / "taxa.parquet").exists()
    assert (output / "names.parquet").exists()
    assert (output / "name_evidence.parquet").exists()
    assert (output / "source_snapshots.parquet").exists()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest

    names = pl.read_parquet(output / "names.parquet")
    assert names.select("normalized_match_key").to_series().to_list() == [
        "papilio demoleus",
        "lime butterfly",
        "butterfly",
    ]


def test_compile_registry_fixture_emits_atomic_flickr_queries_with_tags_before_text(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    _write_fixture(source)

    compile_registry_fixture(source, output, registry_version="test-registry")

    queries = pl.read_parquet(output / "flickr_query_definitions.parquet").sort("search_priority")
    assert queries.select("search_field").to_series().to_list() == ["tags", "tags", "text", "text"]
    assert queries.select("normalized_query_term").to_series().to_list() == [
        "Papilio demoleus",
        "Lime Butterfly",
        "Papilio demoleus",
        "Lime Butterfly",
    ]
    assert queries.select("query_definition_id").to_series().n_unique() == 4
    assert queries.select(["normalized_query_term", "search_field", "region"]).unique().height == 4
