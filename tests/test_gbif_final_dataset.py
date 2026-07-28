from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from biominer.gbif_final.dimensions import (
    build_species_enrichment_dimension,
)
from biominer.gbif_final.pipeline import build_species_enrichments


def test_species_enrichments_inherit_owned_queries_without_row_explosion(
    tmp_path: Path,
) -> None:
    taxa = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:s1",
                "scientific_name": "Papilio one",
                "rank": "SPECIES",
                "parent_key": "gbif:g1",
                "family_key": "gbif:f1",
                "family": "Papilionidae",
                "genus_key": "gbif:g1",
                "genus": "Papilio",
                "species_key": "gbif:s1",
                "species": "Papilio one",
            },
            {
                "accepted_taxon_key": "gbif:s2",
                "scientific_name": "Papilio two",
                "rank": "SPECIES",
                "parent_key": "gbif:g1",
                "family_key": "gbif:f1",
                "family": "Papilionidae",
                "genus_key": "gbif:g1",
                "genus": "Papilio",
                "species_key": "gbif:s2",
                "species": "Papilio two",
            },
            {
                "accepted_taxon_key": "gbif:g1",
                "scientific_name": "Papilio",
                "rank": "GENUS",
                "parent_key": "gbif:f1",
                "family_key": "gbif:f1",
                "family": "Papilionidae",
                "genus_key": "gbif:g1",
                "genus": "Papilio",
                "species_key": "",
                "species": "",
            },
            {
                "accepted_taxon_key": "gbif:f1",
                "scientific_name": "Papilionidae",
                "rank": "FAMILY",
                "parent_key": "",
                "family_key": "gbif:f1",
                "family": "Papilionidae",
                "genus_key": "",
                "genus": "",
                "species_key": "",
                "species": "",
            },
        ]
    )
    names = pl.DataFrame(
        [
            {
                "name_id": "n1",
                "registry_version": "fixture",
                "accepted_taxon_key": "gbif:s1",
                "verbatim_name": "Swallowtail",
                "display_name": "Swallowtail",
                "normalized_match_key": "swallowtail",
                "language": "en",
                "api_language_code": "en",
                "script": "Latn",
                "region": "",
                "bcp47": "en",
                "bbox": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "r1",
                "source_taxon_id": "i1",
                "lineage_check": "pass",
                "trust_tier": "T2",
                "precision_tier": "species",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
                "review_state": "unreviewed",
                "corroborated": False,
                "query_eligible": True,
                "query_disabled_reason": "",
                "species_specificity_score": 1.0,
                "keyword_id": "old1",
                "canonical_keyword_id": "oldc",
                "original_trust_tier": "T2",
                "effective_trust_tier": "T2",
                "is_canonical_keyword": True,
                "suppressed_duplicate": False,
            },
            {
                "name_id": "n2",
                "registry_version": "fixture",
                "accepted_taxon_key": "gbif:s2",
                "verbatim_name": "Swallowtail",
                "display_name": "Swallowtail",
                "normalized_match_key": "swallowtail",
                "language": "en",
                "api_language_code": "en",
                "script": "Latn",
                "region": "",
                "bcp47": "en",
                "bbox": "",
                "name_class": "vernacular",
                "source": "iNaturalist",
                "source_record_id": "r2",
                "source_taxon_id": "i2",
                "lineage_check": "pass",
                "trust_tier": "T2",
                "precision_tier": "species",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
                "review_state": "unreviewed",
                "corroborated": False,
                "query_eligible": True,
                "query_disabled_reason": "",
                "species_specificity_score": 1.0,
                "keyword_id": "old2",
                "canonical_keyword_id": "oldc",
                "original_trust_tier": "T2",
                "effective_trust_tier": "T2",
                "is_canonical_keyword": False,
                "suppressed_duplicate": True,
            },
        ]
    )
    paths = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:s1",
                "species_node_id": "gbif:s1",
                "genus_node_id": "gbif:g1",
                "family_node_id": "gbif:f1",
            },
            {
                "accepted_taxon_key": "gbif:s2",
                "species_node_id": "gbif:s2",
                "genus_node_id": "gbif:g1",
                "family_node_id": "gbif:f1",
            },
        ]
    )
    source = pl.DataFrame(
        {
            "speciesKey": ["s1", "s2", "missing", None, ""],
            "species": [
                "Papilio one",
                "Papilio two",
                "Unknown one",
                None,
                "",
            ],
        }
    )
    for name, frame in {
        "taxa.parquet": taxa,
        "names.parquet": names,
        "species_paths.parquet": paths,
    }.items():
        frame.write_parquet(tmp_path / name)
    source.write_parquet(tmp_path / "source.parquet")

    result = build_species_enrichments(
        source_parquet=tmp_path / "source.parquet",
        registry_dir=tmp_path,
        output_path=tmp_path / "species_enrichments.parquet",
        source_assertions_path=None,
    )

    assert result.height == 4
    by_key = {row["dataset_species_key"]: row for row in result.to_dicts()}
    inherited = by_key["s1"]["flickr_query_terms"]
    swallowtail = [
        term for term in inherited if term["normalized_query_term"] == "swallowtail"
    ]
    assert {term["search_field"] for term in swallowtail} == {"tags", "text"}
    assert {term["keyword_owner_rank"] for term in swallowtail} == {"GENUS"}
    assert "butterfly" in {term["normalized_query_term"] for term in inherited}
    assert by_key["missing"]["registry_match_status"] == "unmatched"

    sealed_path = tmp_path / "sealed-species.parquet"
    receipt = build_species_enrichment_dimension(
        source_parquet=tmp_path / "source.parquet",
        registry_dir=tmp_path,
        output_path=sealed_path,
        source_assertions_path=None,
        producer_git_sha="deadbeef",
        row_group_size=2,
    )
    first_mtime = sealed_path.stat().st_mtime_ns
    resumed = build_species_enrichment_dimension(
        source_parquet=tmp_path / "source.parquet",
        registry_dir=tmp_path,
        output_path=sealed_path,
        source_assertions_path=None,
        producer_git_sha="deadbeef",
        row_group_size=2,
    )

    assert receipt["artifact"]["row_count"] == 4
    assert resumed["part_id"] == receipt["part_id"]
    assert sealed_path.stat().st_mtime_ns == first_mtime
    sealed = pq.read_table(sealed_path)
    assert sealed["dataset_species_key"].to_pylist() == [
        "",
        "missing",
        "s1",
        "s2",
    ]
