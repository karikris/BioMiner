from __future__ import annotations

import json
import zipfile

import polars as pl
import pytest

from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.path_cascade_classifier import classify_path_cascade
from biominer.bioclip.path_cascade_output import path_cascade_result_to_output_row
from biominer.flickr_fetch.metadata_poller import MetadataPollState, _work_item_id
from biominer.flickr_fetch.query_planner import FlickrQuery, load_registry_flickr_queries_from_frame
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.col_xr import extract_col_xr_snapshot


FAMILIES = ("Hesperiidae", "Papilionidae", "Pieridae", "Lycaenidae", "Riodinidae", "Nymphalidae", "Hedylidae")


def test_col_xr_darwin_core_extracts_supported_lineage_synonyms_and_vernaculars(tmp_path) -> None:
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "scope_id": "pieridae",
                "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"},
                "included_families": ["Pieridae"],
                "gbif_family_taxon_keys": {"Pieridae": 1},
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "col.zip"
    taxon_header = "taxonID\tparentNameUsageID\tacceptedNameUsageID\tscientificName\ttaxonRank\ttaxonomicStatus\tfamily\tgenus\n"
    taxon_rows = [
        "k\t\t\tAnimalia\tkingdom\taccepted\t\t",
        "p\tk\t\tArthropoda\tphylum\taccepted\t\t",
        "c\tp\t\tInsecta\tclass\taccepted\t\t",
        "o\tc\t\tLepidoptera\torder\taccepted\t\t",
        "sf\to\t\tPapilionoidea\tsuperfamily\taccepted\t\t",
        "f\tsf\t\tPieridae\tfamily\taccepted\tPieridae\t",
        "g\tf\t\tPieris\tgenus\taccepted\tPieridae\tPieris",
        "s\tg\t\tPieris rapae\tspecies\taccepted\tPieridae\tPieris",
        "syn\t\ts\tArtogeia rapae\tspecies\tsynonym\tPieridae\tArtogeia",
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Taxon.tsv", taxon_header + "\n".join(taxon_rows))
        bundle.writestr(
            "VernacularName.tsv",
            "ID\ttaxonID\tvernacularName\tlanguage\nvern-1\ts\tSmall White\ten\n",
        )

    snapshot = extract_col_xr_snapshot(archive, scope_path=scope)

    assert snapshot["source_dataset_key"] == "315557"
    assert {row["rank"] for row in snapshot["taxa"]} >= {"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES"}
    assert {row["name_class"] for row in snapshot["names"]} >= {"accepted_scientific", "scientific_synonym", "vernacular"}


def _registry(tmp_path, names: list[dict[str, object]]):
    source = tmp_path / "source.json"
    output = tmp_path / "registry"
    taxa = [
        {
            "accepted_taxon_key": "col:Animalia",
            "scientific_name": "Animalia",
            "rank": "KINGDOM",
            "parent_key": "",
        },
        {
            "accepted_taxon_key": "col:Insecta",
            "scientific_name": "Insecta",
            "rank": "CLASS",
            "parent_key": "col:Animalia",
        },
        {
            "accepted_taxon_key": "col:Lepidoptera",
            "scientific_name": "Lepidoptera",
            "rank": "ORDER",
            "parent_key": "col:Insecta",
        },
        {
            "accepted_taxon_key": "col:root",
            "scientific_name": "Papilionoidea",
            "rank": "SUPERFAMILY",
            "parent_key": "col:Lepidoptera",
        },
        *[
            {
                "accepted_taxon_key": f"col:{family}",
                "scientific_name": family,
                "rank": "FAMILY",
                "parent_key": "col:root",
                "family_key": f"col:{family}",
                "family": family,
            }
            for family in FAMILIES
        ],
        {
            "accepted_taxon_key": "col:Pieris",
            "scientific_name": "Pieris",
            "rank": "GENUS",
            "parent_key": "col:Pieridae",
            "family_key": "col:Pieridae",
            "family": "Pieridae",
            "genus_key": "col:Pieris",
            "genus": "Pieris",
        },
        {
            "accepted_taxon_key": "col:rapae",
            "scientific_name": "Pieris rapae",
            "rank": "SPECIES",
            "parent_key": "col:Pieris",
            "family_key": "col:Pieridae",
            "family": "Pieridae",
            "genus_key": "col:Pieris",
            "genus": "Pieris",
            "species_key": "col:rapae",
            "species": "Pieris rapae",
        },
        {
            "accepted_taxon_key": "col:brassicae",
            "scientific_name": "Pieris brassicae",
            "rank": "SPECIES",
            "parent_key": "col:Pieris",
            "family_key": "col:Pieridae",
            "family": "Pieridae",
            "genus_key": "col:Pieris",
            "genus": "Pieris",
            "species_key": "col:brassicae",
            "species": "Pieris brassicae",
        },
    ]
    source.write_text(
        json.dumps(
            {
                "source": "CoL XR",
                "source_version": "COL26.6 XR",
                "retrieved_at": "2026-07-12T00:00:00Z",
                "taxa": taxa,
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    compile_registry_fixture(source, output, registry_version="v1")
    return output


def _name(taxon: str, term: str, tier: str, source: str, record: str) -> dict[str, object]:
    return {
        "accepted_taxon_key": taxon,
        "display_name": term,
        "language": "en",
        "script": "Latn",
        "name_class": "accepted_scientific" if tier == "T1" else "vernacular_alias",
        "source": source,
        "source_record_id": record,
        "trust_tier": tier,
        "precision_tier": "high",
        "confidence": "high",
        "enabled": True,
        "review_state": "reviewed",
        "corroborated": True,
    }


def test_same_cross_species_cross_tier_term_has_one_logical_query_per_field(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        [
            _name("col:rapae", "Cabbage White", "T1", "CoL XR", "n1"),
            _name("col:brassicae", "Cabbage\u00a0White", "T5", "translation", "n2"),
        ],
    )
    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")

    assert names["canonical_keyword_id"].n_unique() == 1
    assert names.filter(pl.col("is_canonical_keyword"))["accepted_taxon_key"].to_list() == ["col:rapae"]
    assert names["effective_trust_tier"].unique().to_list() == ["T1"]
    assert queries.height == 2
    assert set(queries["search_field"].to_list()) == {"tags", "text"}
    assert len(load_registry_flickr_queries_from_frame(queries)) == 2


def test_registry_upgrade_backfills_new_keyword_without_new_request_or_photo_queue(tmp_path) -> None:
    registry = _registry(tmp_path, [_name("col:rapae", "Cabbage White", "T1", "CoL XR", "n1")])
    state = MetadataPollState(tmp_path / "state.sqlite")
    first = state.register_registry(registry)
    queries = load_registry_flickr_queries_from_frame(pl.read_parquet(registry / "flickr_query_definitions.parquet"))
    query = next(item for item in queries if item.search_field == "tags")
    assert state.enqueue_work_item(query) == 1
    inserted = state.insert_source_records(
        [{"id": "photo-1", "title": "Cabbage White", "url_l": "https://example.test/1.jpg"}],
        source_query=query,
    )
    state.complete_work_item(_work_item_id(query))
    names = pl.read_parquet(registry / "names.parquet")
    extra = names.row(0, named=True) | {
        "keyword_id": "new-association",
        "accepted_taxon_key": "col:brassicae",
        "original_trust_tier": "T3",
        "is_canonical_keyword": False,
        "suppressed_duplicate": True,
        "registry_version": "v2",
    }
    second = state.register_query_definitions(
        pl.read_parquet(registry / "flickr_query_definitions.parquet"),
        keyword_associations=pl.DataFrame([extra], schema=names.schema),
    )

    assert first["logical_queries_inserted"] == 2
    assert inserted[:3] == (1, 0, 1)
    assert second["keyword_associations_inserted"] == 1
    assert state.enqueue_work_item(query) == 0
    evidence = state.photo_keyword_evidence_frame()
    assert set(evidence["accepted_taxon_id"].to_list()) == {"col:rapae", "col:brassicae"}


def test_active_tier_barrier_and_non_overlapping_physical_intervals(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
    lower = FlickrQuery(
        term="weak alias",
        normalized_term="weak alias",
        language="en",
        search_field="tags",
        lane="normal_page",
        effective_trust_tier="T5",
        min_upload_date="2020-01-01",
        max_upload_date="2020-01-31",
    )
    higher = FlickrQuery(
        term="Pieris rapae",
        normalized_term="pieris rapae",
        language="la",
        search_field="text",
        lane="normal_page",
        effective_trust_tier="T1",
        min_upload_date="2020-01-01",
        max_upload_date="2020-01-31",
    )
    state.enqueue_work_item(lower)
    state.enqueue_work_item(higher)
    claimed = state.claim_and_reserve_pending(limit=10, max_api_calls=10, endpoint="test")
    assert [query.effective_trust_tier for _, query in claimed] == ["T1"]
    with pytest.raises(ValueError, match="overlapping upload-date interval"):
        state.enqueue_work_item(
            FlickrQuery(
                **{
                    **lower.__dict__,
                    "min_upload_date": "2020-01-15",
                    "max_upload_date": "2020-02-15",
                }
            )
        )


def test_species_paths_use_semantic_parent_proxies_and_store_reads_registry_dir(tmp_path) -> None:
    registry = _registry(tmp_path, [_name("col:rapae", "Pieris rapae", "T1", "CoL XR", "n1")])
    paths = pl.read_parquet(registry / "species_paths.parquet")
    row = paths.filter(pl.col("accepted_taxon_key") == "col:rapae").row(0, named=True)

    assert row["phylum_candidate_kind"] == "carry_forward_proxy"
    assert row["phylum_semantic_rank"] == "KINGDOM"
    assert row["phylum_proxy_source_node_id"] == "col:Animalia"
    store = PathTaxonomyStore.read(registry)
    prompts = store.prompt_rows_for_nodes([row["phylum_node_id"]], "rank_screen")
    assert prompts.height == 1
    assert "kingdom Animalia" in prompts["label"][0]
    assert "phylum Animalia" not in prompts["label"][0]


def test_unified_store_drives_bioclip_supported_rank_order(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        [
            _name("col:rapae", "Pieris rapae", "T1", "CoL XR", "n1"),
            _name("col:brassicae", "Pieris brassicae", "T1", "CoL XR", "n2"),
        ],
    )
    store = PathTaxonomyStore.read(registry)

    class Scorer:
        model_id = "fake"
        model_checkpoint = "fake"

        def raw_similarities(self, item, labels):  # noqa: ANN001, ANN201
            del item
            return {label: 1.0 - index / 100 for index, label in enumerate(labels)}

    result = classify_path_cascade(item={}, scorer=Scorer(), taxonomy_store=store)

    assert tuple(step.rank for step in result.rank_steps) == ("FAMILY", "GENUS", "SPECIES")
    assert result.rank_steps[0].retained_count == 1
    assert result.rank_steps[1].shortlist_limit == 1
    assert result.rank_steps[1].retained_count == 1
    assert result.rank_steps[2].shortlist_limit == 20
    assert result.species_rerank_step.shortlist_limit == 5
    assert result.species_top1 == result.species_top5[0]
    output = path_cascade_result_to_output_row(result)
    assert output["workflow"] == "family_top1_genus_top20_top3_species_top20_top5_top1"
    assert output["genus_top20"]
    assert output["genus_top3"] == output["genus_top20"][:3]
    assert set(output["species_top5"]).issubset(output["species_top20"])
    assert output["species_top1"] == output["species_top5"][0]
    assert output["genus_routing_mode"] == "top1_above_90pct"

    class LowConfidenceGenusScorer(Scorer):
        def raw_similarities(self, item, labels):  # noqa: ANN001, ANN201
            del item
            return {
                label: (0.90 if "genus" in label else 0.95 - index / 100)
                for index, label in enumerate(labels)
            }

    broad = classify_path_cascade(item={}, scorer=LowConfidenceGenusScorer(), taxonomy_store=store)
    broad_output = path_cascade_result_to_output_row(broad)
    assert broad.rank_steps[1].shortlist_limit == 20
    assert broad_output["genus_routing_mode"] == "top20_then_top3"
