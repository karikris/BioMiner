from __future__ import annotations

import json

import polars as pl

from biominer.registry.compiler import compile_registry_fixture


def test_query_curation_disables_query_without_removing_name_evidence(tmp_path) -> None:
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
                "source_version": "fixture",
                "retrieved_at": "2026-07-07T00:00:00+00:00",
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
                        "verbatim_name": "Dingy Swallowtail",
                        "display_name": "Dingy Swallowtail",
                        "language": "eng",
                        "script": "Latn",
                        "name_class": "vernacular",
                        "source": "GBIF",
                        "source_record_id": "gbif:100:dingy",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "high",
                        "enabled": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    curation = tmp_path / "query_curation.json"
    curation.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "rules": [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "normalized_match_key": "dingy swallowtail",
                        "source": "GBIF",
                        "action": "disable_query",
                        "reason": "misapplied_common_name_conflicts_with_other_species",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"

    compile_registry_fixture(source, output, registry_version="curated", scope_path=scope, query_curation_json=curation)

    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    row = names.filter(pl.col("display_name") == "Dingy Swallowtail").to_dicts()[0]

    assert row["enabled"] is True
    assert row["query_eligible"] is False
    assert row["query_disabled_reason"] == "misapplied_common_name_conflicts_with_other_species"
    assert "dingy swallowtail" not in queries.select("normalized_query_term").to_series().to_list()
