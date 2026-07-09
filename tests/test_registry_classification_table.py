from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.registry.classification_table import (
    BUTTERFLY_CLASSIFICATION_MANIFEST_FILE,
    BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE,
    BUTTERFLY_CLASSIFICATION_TAXA_FILE,
    BUTTERFLY_FAMILY_LABELS_FILE,
    BUTTERFLY_SPECIES_LABELS_FILE,
    CLASSIFICATION_TABLE_VERSION,
    CLASSIFICATION_TAXA_SCHEMA,
    FAMILY_LABEL_SCHEMA,
    PROMPT_VARIANT_VERSION,
    SPECIES_LABEL_SCHEMA,
    bare_gbif_key,
    build_classification_table_manifest,
    build_classification_tables_from_registry_dir,
    build_classification_taxa_frame,
    build_family_label_frame,
    build_species_label_frame,
    derive_species_epithet,
    validate_classification_tables,
)


def test_classification_table_schema_constants_are_explicit() -> None:
    assert CLASSIFICATION_TABLE_VERSION == "gbif-butterfly-classification-v1"
    assert PROMPT_VARIANT_VERSION == "butterfly-hierarchical-prompts-v1"
    assert list(CLASSIFICATION_TAXA_SCHEMA) == [
        "registry_version",
        "classification_table_version",
        "source",
        "source_version",
        "retrieved_at",
        "scope_id",
        "accepted_taxon_key",
        "gbif_species_key",
        "scientific_name",
        "canonical_name",
        "rank",
        "taxonomic_status",
        "family_key",
        "family",
        "genus_key",
        "genus",
        "species_key",
        "species",
        "species_epithet",
        "in_scope",
        "classification_enabled",
        "classification_disabled_reason",
    ]
    assert "label" in FAMILY_LABEL_SCHEMA
    assert "label" in SPECIES_LABEL_SCHEMA


def test_classification_taxa_frame_filters_species_dedupes_and_sorts() -> None:
    frame = build_classification_taxa_frame(
        _taxa_fixture(
            [
                _taxon("gbif:9417", "Papilionidae", "FAMILY", family_key="gbif:9417", family="Papilionidae"),
                _taxon("gbif:200", "Danaus plexippus", "SPECIES", family_key="gbif:7017", family="Nymphalidae", genus_key="gbif:190", genus="Danaus"),
                _taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
                _taxon("gbif:100", "Papilio demoleus duplicate", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
                _taxon("gbif:300", "Pieris rapae", "SPECIES", family_key="gbif:5481", family="Pieridae", genus_key="", genus="Pieris", in_scope=False),
            ]
        ),
        registry_manifest={"registry_version": "registry-v1", "scope_id": "scope-1"},
        source_snapshots=_source_snapshots(),
    )

    assert frame.schema == CLASSIFICATION_TAXA_SCHEMA
    assert frame.select("scientific_name").to_series().to_list() == ["Danaus plexippus", "Papilio demoleus"]
    assert frame.select("gbif_species_key").to_series().to_list() == ["200", "100"]
    assert frame.select("species_epithet").to_series().to_list() == ["plexippus", "demoleus"]
    assert frame.select("classification_enabled").to_series().to_list() == [True, True]
    assert frame.select("source").to_series().unique().to_list() == ["GBIF"]


def test_classification_taxa_missing_family_disables_row_without_crashing() -> None:
    frame = build_classification_taxa_frame(
        _taxa_fixture([_taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="", family="", genus_key="gbif:90", genus="Papilio")]),
        registry_manifest={"registry_version": "registry-v1"},
    )

    row = frame.to_dicts()[0]
    assert row["classification_enabled"] is False
    assert row["classification_disabled_reason"] == "missing_family_key,missing_family"


def test_gbif_key_and_epithet_helpers() -> None:
    assert bare_gbif_key("gbif:123") == "123"
    assert bare_gbif_key("123") == "123"
    assert bare_gbif_key("") == ""
    assert bare_gbif_key(None) == ""
    assert derive_species_epithet("Papilio demoleus") == "demoleus"
    assert derive_species_epithet("Papilio") == ""


def test_family_and_species_label_frames_are_prompt_based_and_stable() -> None:
    taxa = build_classification_taxa_frame(
        _taxa_fixture(
            [
                _taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
                _taxon("gbif:101", "Papilio machaon", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
                _taxon("gbif:200", "Danaus plexippus", "SPECIES", family_key="gbif:7017", family="Nymphalidae", genus_key="gbif:190", genus="Danaus"),
            ]
        ),
        registry_manifest={"registry_version": "registry-v1"},
    )

    family_labels = build_family_label_frame(taxa)
    species_labels = build_species_label_frame(taxa)

    assert family_labels.schema == FAMILY_LABEL_SCHEMA
    assert species_labels.schema == SPECIES_LABEL_SCHEMA
    assert family_labels.height == 6
    assert species_labels.height == 9
    assert family_labels.filter(pl.col("family") == "Papilionidae").select("label").to_series().to_list()[0] == (
        "a photo of a butterfly in the family Papilionidae"
    )
    assert species_labels.filter(pl.col("accepted_taxon_key") == "gbif:100").select("label").to_series().to_list()[0] == "a photo of Papilio demoleus"
    assert species_labels.select(["accepted_taxon_key", "label"]).unique().height == species_labels.height
    assert species_labels.select("sort_order").to_series().head(3).to_list() == [1, 2, 3]


def test_empty_label_frames_keep_schemas() -> None:
    empty_taxa = pl.DataFrame(schema=CLASSIFICATION_TAXA_SCHEMA)

    assert build_family_label_frame(empty_taxa).schema == FAMILY_LABEL_SCHEMA
    assert build_species_label_frame(empty_taxa).schema == SPECIES_LABEL_SCHEMA


def test_validation_and_manifest_counts() -> None:
    taxa = build_classification_taxa_frame(
        _taxa_fixture(
            [
                _taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="", genus=""),
                _taxon("gbif:101", "Papilio machaon", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
            ]
        ),
        registry_manifest={"registry_version": "registry-v1"},
    )
    family_labels = build_family_label_frame(taxa)
    species_labels = build_species_label_frame(taxa)

    findings = validate_classification_tables(taxa, family_labels, species_labels)
    manifest = build_classification_table_manifest(
        registry_manifest={"registry_version": "registry-v1", "qa_status": "passed"},
        classification_taxa=taxa,
        family_labels=family_labels,
        species_labels=species_labels,
        findings=findings,
    )

    assert not [finding for finding in findings if finding["severity"] == "fatal"]
    assert [finding["code"] for finding in findings if finding["severity"] == "warning"] == ["enabled_species_missing_genus"]
    assert manifest["species_count"] == 2
    assert manifest["family_count"] == 1
    assert manifest["family_label_count"] == 3
    assert manifest["species_label_count"] == 6
    assert manifest["source_registry_qa_status"] == "passed"


def test_validation_reports_fatal_missing_and_broken_references() -> None:
    taxa = build_classification_taxa_frame(
        _taxa_fixture([_taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio")]),
        registry_manifest={"registry_version": "registry-v1"},
    )
    family_labels = build_family_label_frame(taxa).with_columns(pl.lit("gbif:missing").alias("family_key"))
    species_labels = build_species_label_frame(taxa).with_columns(pl.lit("gbif:missing").alias("accepted_taxon_key"))

    codes = {finding["code"] for finding in validate_classification_tables(taxa, family_labels, species_labels) if finding["severity"] == "fatal"}

    assert "family_label_unknown_family_key" in codes
    assert "species_label_unknown_accepted_taxon_key" in codes


def test_build_classification_tables_from_registry_dir_writes_expected_files(tmp_path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa_fixture(
        [
            _taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
            _taxon("gbif:101", "Papilio machaon", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
        ]
    ).write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "registry-v1", "qa_status": "passed"}), encoding="utf-8")
    _source_snapshots().write_parquet(registry / "source_snapshots.parquet")

    summary = build_classification_tables_from_registry_dir(registry)

    assert (registry / BUTTERFLY_CLASSIFICATION_TAXA_FILE).exists()
    assert (registry / BUTTERFLY_FAMILY_LABELS_FILE).exists()
    assert (registry / BUTTERFLY_SPECIES_LABELS_FILE).exists()
    assert (registry / BUTTERFLY_CLASSIFICATION_MANIFEST_FILE).exists()
    assert (registry / BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE).exists()
    assert summary["species_count"] == 2
    assert summary["family_count"] == 1
    assert summary["family_label_rows"] == 3
    assert summary["species_label_rows"] == 6
    assert summary["artifact_file_sizes"]["classification_taxa"] > 0
    assert summary["local_file_sizes_mb"]["classification_taxa"] > 0


def test_build_classification_tables_requires_taxa_parquet(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="taxa.parquet"):
        build_classification_tables_from_registry_dir(tmp_path)


def test_butterfly_taxonomy_store_reads_and_filters_by_family(tmp_path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa_fixture(
        [
            _taxon("gbif:100", "Papilio demoleus", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
            _taxon("gbif:101", "Papilio machaon", "SPECIES", family_key="gbif:9417", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
            _taxon("gbif:200", "Danaus plexippus", "SPECIES", family_key="gbif:7017", family="Nymphalidae", genus_key="gbif:190", genus="Danaus"),
        ]
    ).write_parquet(registry / "taxa.parquet")
    build_classification_tables_from_registry_dir(registry)

    store = ButterflyTaxonomyStore.read(registry)

    assert store.family_candidates().select("family").to_series().to_list() == ["Nymphalidae", "Papilionidae"]
    assert store.species_for_family("gbif:9417").select("scientific_name").to_series().to_list() == ["Papilio demoleus", "Papilio machaon"]
    assert store.species_prompt_labels_for_family("gbif:9417")[0] == "a photo of Papilio demoleus"
    assert store.species_labels_for_taxa(["gbif:200"]).select("label").to_series().to_list()[0] == "a photo of Danaus plexippus"
    with pytest.raises(KeyError, match="unknown family_key"):
        store.species_for_family("gbif:missing")


def _taxa_fixture(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "registry_schema_version": pl.String,
            "scope_id": pl.String,
            "accepted_taxon_key": pl.String,
            "scientific_name": pl.String,
            "rank": pl.String,
            "parent_key": pl.String,
            "family_key": pl.String,
            "family": pl.String,
            "genus_key": pl.String,
            "genus": pl.String,
            "species_key": pl.String,
            "species": pl.String,
            "in_scope": pl.Boolean,
        },
    )


def _taxon(
    accepted_taxon_key: str,
    scientific_name: str,
    rank: str,
    *,
    family_key: str = "",
    family: str = "",
    genus_key: str = "",
    genus: str = "",
    species_key: str | None = None,
    species: str | None = None,
    in_scope: bool = True,
) -> dict[str, object]:
    return {
        "registry_schema_version": "registry-foundation-v1",
        "scope_id": "papilionoidea-seven-family-scope",
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "rank": rank,
        "parent_key": "",
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": species_key if species_key is not None else (accepted_taxon_key if rank == "SPECIES" else ""),
        "species": species if species is not None else (scientific_name if rank == "SPECIES" else ""),
        "in_scope": in_scope,
    }


def _source_snapshots() -> pl.DataFrame:
    return pl.DataFrame(
        [{"source": "GBIF", "source_version": "gbif-species-api", "retrieved_at": "2026-01-01T00:00:00Z"}],
        schema={"source": pl.String, "source_version": pl.String, "retrieved_at": pl.String},
    )
