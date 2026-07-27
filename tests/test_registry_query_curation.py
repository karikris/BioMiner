from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.registry.compiler import compile_registry_fixture, compile_registry_frames


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


def test_papilio_query_curation_updates_query_keyword_regression_counts(tmp_path) -> None:
    scope = _write_scope(tmp_path)
    eligible_terms = ("Papilio demoleus", *(f"Source-backed Lime Name {index:02d}" for index in range(1, 47)))
    curated_exclusions = ("Dingy Swallowtail", "Small Citrus Butterfly", "swallowtails")
    assert len((*eligible_terms, *curated_exclusions)) == 50

    full_frames, _ = compile_registry_frames(
        _papilio_payload([*eligible_terms, *curated_exclusions], include_conflicting_species=True),
        source_ref="memory://papilio-full",
        output_ref="memory://papilio-full",
        registry_version="papilio-full",
        scope_path=scope,
    )
    slice_frames, manifest = compile_registry_frames(
        _papilio_payload([*eligible_terms, *curated_exclusions], include_conflicting_species=False),
        source_ref="memory://papilio-slice",
        output_ref="memory://papilio-slice",
        registry_version="papilio-slice",
        scope_path=scope,
        global_names_for_collision=full_frames["names.parquet"],
        query_curation_json=Path("examples/species/papilio_demoleus/query_curation.json"),
    )

    names = slice_frames["names.parquet"]
    queries = slice_frames["flickr_query_definitions.parquet"]
    curated_rows = names.filter(pl.col("display_name").is_in(curated_exclusions)).sort("display_name")

    assert curated_rows.select("enabled").to_series().to_list() == [True, True, True]
    assert curated_rows.select("query_eligible").to_series().to_list() == [False, False, False]
    assert curated_rows.select("query_disabled_reason").to_series().to_list() == [
        "misapplied_common_name_conflicts_with_other_species",
        "misapplied_common_name_conflicts_with_other_species",
        "broad_group_name_not_species_specific",
    ]
    assert queries.height == 98
    assert queries.select("normalized_query_term").n_unique() == 49
    assert not {"dingy swallowtail", "small citrus butterfly", "swallowtails"} & set(
        queries.select("normalized_query_term").to_series().to_list()
    )
    assert manifest["query_definition_unique_term_count"] == 49
    assert manifest["query_definition_unique_term_counts_by_source"] == {
        "GBIF": 47,
        "butterfly_scope_policy": 2,
    }
    assert manifest["query_definition_rows_by_source"] == {
        "GBIF": 94,
        "butterfly_scope_policy": 4,
    }
    assert manifest["query_definition_unique_term_counts_by_source"].get("CoL", 0) == 0


def test_papilio_specific_query_curation_terms_are_data_only() -> None:
    hits: dict[str, list[str]] = {"dingy swallowtail": [], "small citrus butterfly": [], "swallowtails": []}
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        for term in hits:
            if term in text:
                hits[term].append(path.as_posix())

    assert hits["dingy swallowtail"] == []
    assert hits["small citrus butterfly"] == []
    assert set(hits["swallowtails"]) <= {"src/biominer/registry/query_eligibility.py"}


def _write_scope(tmp_path) -> Path:
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "scope_id": "test-scope",
                "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"},
                "included_families": ["Papilionidae"],
                "gbif_family_taxon_keys": {"Papilionidae": 9417},
            }
        ),
        encoding="utf-8",
    )
    return scope


def _papilio_payload(terms: list[str], *, include_conflicting_species: bool) -> dict[str, object]:
    lineage = [
        {
            "accepted_taxon_key": "gbif:1875",
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
            "accepted_taxon_key": "gbif:9417",
            "scientific_name": "Papilionidae",
            "rank": "FAMILY",
            "parent_key": "gbif:1875",
            "family_key": "gbif:9417",
            "family": "Papilionidae",
            "genus_key": "",
            "genus": "",
            "species_key": "",
            "species": "",
        },
        {
            "accepted_taxon_key": "gbif:1933",
            "scientific_name": "Papilio",
            "rank": "GENUS",
            "parent_key": "gbif:9417",
            "family_key": "gbif:9417",
            "family": "Papilionidae",
            "genus_key": "gbif:1933",
            "genus": "Papilio",
            "species_key": "",
            "species": "",
        },
    ]
    taxa = [
        *lineage,
        _species_taxon("gbif:1938069", "Papilio demoleus"),
    ]
    names = [_name_row("gbif:1938069", term, source_record_id=f"gbif:1938069:{index}") for index, term in enumerate(terms)]
    if include_conflicting_species:
        taxa.append(_species_taxon("gbif:777", "Papilio fixtureus"))
        names.extend(
            [
                _name_row("gbif:777", "Dingy Swallowtail", source_record_id="gbif:777:dingy"),
                _name_row("gbif:777", "Small Citrus Butterfly", source_record_id="gbif:777:small-citrus"),
            ]
        )
    return {
        "source": "GBIF",
        "source_version": "fixture",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "taxa": taxa,
        "names": names,
    }


def _species_taxon(accepted_taxon_key: str, scientific_name: str) -> dict[str, object]:
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "rank": "SPECIES",
        "parent_key": "gbif:1933",
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:1933",
        "genus": "Papilio",
        "species_key": accepted_taxon_key,
        "species": scientific_name,
    }


def _name_row(accepted_taxon_key: str, display_name: str, *, source_record_id: str) -> dict[str, object]:
    scientific = display_name == "Papilio demoleus"
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "verbatim_name": display_name,
        "display_name": display_name,
        "language": "la" if scientific else "eng",
        "script": "Latn",
        "name_class": "accepted_scientific" if scientific else "vernacular",
        "source": "GBIF",
        "source_record_id": source_record_id,
        "trust_tier": "T1" if scientific else "T2",
        "precision_tier": "high" if scientific else "medium",
        "confidence": "high",
        "enabled": True,
    }
