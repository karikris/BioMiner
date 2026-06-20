from __future__ import annotations

import json

import polars as pl

from biominer.registry.build import build_registry


def _gbif_snapshot() -> dict[str, object]:
    return {
        "source": "GBIF",
        "source_version": "gbif-species-api",
        "retrieved_at": "2026-06-20T00:00:00+00:00",
        "metrics": {"gbif_calls": 1, "gbif_retries": 0, "workers": 1},
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
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "source_record_id": "gbif:100",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
            }
        ],
    }


def _scope(path) -> None:
    path.write_text(
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


class RecordingSourceClient:
    def __init__(self, source: str, display_name: str, *, raise_error: bool = False) -> None:
        self.source = source
        self.display_name = display_name
        self.raise_error = raise_error
        self.contexts = []

    def enrich_species(self, context):  # noqa: ANN001 - test double.
        self.contexts.append(context)
        if self.raise_error:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return {
            "name_assertions": [
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "display_name": self.display_name,
                    "language": "eng",
                    "script": "Latn",
                    "region": "",
                    "name_class": "vernacular",
                    "source": self.source,
                    "source_record_id": f"{self.source}:name:{context.accepted_taxon_key}",
                    "trust_tier": "T2",
                    "precision_tier": "medium",
                    "confidence": "high",
                    "enabled": True,
                    "review_state": "accepted",
                }
            ],
            "external_links": [
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "source": self.source,
                    "source_taxon_id": f"{self.source}:taxon:{context.accepted_taxon_key}",
                    "match_method": "scientific_name",
                    "match_confidence": "high",
                    "lineage_check": "accepted_taxon_key",
                }
            ],
            "source_snapshots": [
                {
                    "source": self.source,
                    "source_version": "fixture",
                    "retrieved_at": "2026-06-20T00:00:00+00:00",
                    "source_path": f"memory://{self.source}",
                    "source_response_hash": f"sha256:{self.source}",
                    "licence": "",
                }
            ],
        }


def test_registry_build_outputs_one_canonical_enriched_register_by_default(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    clients = {
        "col": RecordingSourceClient("CoL", "Lime Swallowtail"),
        "itis": RecordingSourceClient("ITIS", "Lime Butterfly"),
        "inaturalist": RecordingSourceClient("iNaturalist", "Chequered Swallowtail"),
        "wikidata": RecordingSourceClient("Wikidata", "Stale Wikidata Name"),
    }
    monkeypatch.setattr("biominer.registry.enrichment.default_enrichment_clients", lambda max_retries=5: clients)

    result = build_registry(
        output_dir=tmp_path / "registry",
        registry_version="enriched",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir=tmp_path / "reports",
        workers=1,
    )

    registry = tmp_path / "registry"
    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    errors = pl.read_parquet(registry / "source_error_records.parquet")
    manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))

    assert result["manifest"]["qa_status"] == "passed"
    assert manifest["enrichment_sources"] == ["col", "itis", "inaturalist"]
    assert {"Lime Swallowtail", "Lime Butterfly", "Chequered Swallowtail"}.issubset(set(names["display_name"].to_list()))
    assert "Stale Wikidata Name" not in names["display_name"].to_list()
    assert "Stale Wikidata Name" not in assertions["display_name"].to_list()
    assert "Lime Swallowtail" in queries["normalized_query_term"].to_list()
    assert errors.is_empty()


def test_registry_build_quarantines_source_errors_without_siloing_successful_names(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    clients = {
        "col": RecordingSourceClient("CoL", "Lime Swallowtail"),
        "itis": RecordingSourceClient("ITIS", "Broken ITIS", raise_error=True),
        "inaturalist": RecordingSourceClient("iNaturalist", "Chequered Swallowtail"),
    }
    monkeypatch.setattr("biominer.registry.enrichment.default_enrichment_clients", lambda max_retries=5: clients)

    result = build_registry(
        output_dir=tmp_path / "registry",
        registry_version="enriched",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir=tmp_path / "reports",
        workers=1,
    )

    registry = tmp_path / "registry"
    names = pl.read_parquet(registry / "names.parquet")
    errors = pl.read_parquet(registry / "source_error_records.parquet")
    qa = pl.read_parquet(registry / "qa_findings.parquet")

    assert result["manifest"]["qa_status"] == "passed"
    assert {"Lime Swallowtail", "Chequered Swallowtail"}.issubset(set(names["display_name"].to_list()))
    assert "Broken ITIS" not in names["display_name"].to_list()
    assert errors.select("source").to_series().to_list() == ["itis"]
    assert errors.select("error_class").to_series().to_list() == ["UnicodeDecodeError"]
    assert {"severity": "warning", "code": "source_enrichment_error", "subject": "itis:gbif:100:UnicodeDecodeError"} in qa.to_dicts()
