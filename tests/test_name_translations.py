from __future__ import annotations

import json

import polars as pl

from biominer.registry.name_translations import (
    NameTranslationContext,
    SourceResult,
    _external_link,
    _name_assertion,
    _snapshot,
    _source_error,
    merge_name_translation_sidecar_into_enrichment,
    write_translation_results,
)


def test_write_translation_results_dedupes_and_merges_enrichment_sidecars(tmp_path) -> None:
    context = NameTranslationContext(
        accepted_scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:1938069",
        seed_common_names=("Lime butterfly",),
    )
    assertion = _name_assertion(
        context,
        "Lime butterfly",
        source="Wikidata",
        source_record_id="wikidata:Q123:P1843:en:Lime butterfly",
        source_taxon_id="Q123",
        language="en",
        trust_tier="T2",
        precision_tier="high",
        confidence="high",
        licence="CC0",
    )
    result = SourceResult(
        name_assertions=(assertion, assertion),
        external_links=(
            _external_link(context, source="Wikidata", source_taxon_id="Q123", match_method="P225"),
        ),
        source_snapshots=(
            _snapshot("Wikidata", "WDQS-P1843", "/sparql", {"results": []}, licence="CC0"),
        ),
        errors=(
            _source_error(context, "Wikidata", "TimeoutException", endpoint="/sparql"),
        ),
        request_count=2,
    )

    manifest = write_translation_results(tmp_path, context=context, results=(result,))

    assertions = pl.read_parquet(tmp_path / "name_translation_assertions.parquet")
    errors = pl.read_parquet(tmp_path / "name_translation_errors.parquet")
    assert manifest["name_assertion_rows"] == 1
    assert manifest["source_error_rows"] == 1
    assert assertions.height == 1
    assert assertions.to_dicts()[0]["language"] == "eng"
    assert assertions.to_dicts()[0]["enabled"] is True
    assert errors.to_dicts()[0]["disposition"] == "quarantined"

    merge = merge_name_translation_sidecar_into_enrichment(tmp_path)
    merge_again = merge_name_translation_sidecar_into_enrichment(tmp_path)

    merged_assertions = pl.read_parquet(tmp_path / "source_name_assertions.parquet")
    merged_links = pl.read_parquet(tmp_path / "external_taxon_links.parquet")
    merged_manifest = json.loads((tmp_path / "name_translation_merge_manifest.json").read_text(encoding="utf-8"))
    assert merge["merged_source_name_assertion_rows"] == 1
    assert merge_again["merged_source_name_assertion_rows"] == 1
    assert merged_assertions.height == 1
    assert merged_links.height == 1
    assert merged_manifest["translation_assertion_rows"] == 1
