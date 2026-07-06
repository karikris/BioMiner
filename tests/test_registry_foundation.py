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
    assert (output / "name_evidence.parquet").exists()
    assert (output / "source_snapshots.parquet").exists()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest

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
    assert queries.select("normalized_query_term").to_series().to_list() == [
        "Papilio demoleus",
        "Papilio demoleus",
        "Lime Butterfly",
        "Lime Butterfly",
    ]
    assert queries.select("query_definition_id").to_series().n_unique() == 4
    assert queries.select(["normalized_query_term", "search_field", "region"]).unique().height == 4


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
    for weak_name in ("lime", "swallowtails", "papillon"):
        assert by_name[weak_name]["enabled"] is True
        assert by_name[weak_name]["query_eligible"] is False
        assert by_name[weak_name]["query_disabled_reason"]
    assert "Common Lime Butterfly" in queries["normalized_query_term"].to_list()
    assert not {"lime", "swallowtails", "papillon"} & set(queries["normalized_match_key"].to_list())
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
    assert {"severity": "warning", "code": "normalized_name_collision", "subject": "lime butterfly"} in qa
    assert {"severity": "warning", "code": "weak_language_or_script_metadata", "subject": "Lime Butterfly"} in qa
    assert {"severity": "warning", "code": "missing_name_source_evidence", "subject": "Lime Butterfly"} in qa
