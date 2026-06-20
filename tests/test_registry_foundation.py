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
    assert {artifact["file_name"] for artifact in manifest["artifacts"]} == {
        "taxa.parquet",
        "taxon_relations.parquet",
        "names.parquet",
        "name_evidence.parquet",
        "source_snapshots.parquet",
        "flickr_query_definitions.parquet",
        "qa_findings.parquet",
    }
    taxa_artifact = next(artifact for artifact in manifest["artifacts"] if artifact["file_name"] == "taxa.parquet")
    assert taxa_artifact["rows"] == 9
    assert taxa_artifact["size_bytes"] > 0
    assert taxa_artifact["sha256"].startswith("sha256:")
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


def test_compile_registry_fixture_reports_duplicate_query_ids_and_name_warnings(tmp_path) -> None:
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
    assert manifest["qa_status"] == "failed"
    assert {"severity": "fatal", "code": "duplicate_query_definition_id", "subject": "4"} in qa
    assert {"severity": "warning", "code": "normalized_name_collision", "subject": "lime butterfly"} in qa
    assert {"severity": "warning", "code": "weak_language_or_script_metadata", "subject": "Lime Butterfly"} in qa
    assert {"severity": "warning", "code": "missing_name_source_evidence", "subject": "Lime Butterfly"} in qa
