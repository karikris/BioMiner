from __future__ import annotations

import json
import logging
import threading
import time

import httpx
import polars as pl

from biominer.cli import build_parser, run
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.enrichment import SpeciesContext, build_enrichment_sources_from_registry, compile_enriched_registry, write_enrichment_sources
from biominer.registry.enrichment_sources import (
    INaturalistClient,
    INaturalistRateLimitTermError,
    INaturalistRateLimiter,
    ITISClient,
    WikidataClient,
    WikidataRateLimitTermError,
    WikidataRateLimiter,
    _json_get,
)


def _write_base_registry(tmp_path, species_names: tuple[str, ...] = ("Papilio demoleus",)):
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
                "source_version": "gbif-species-api",
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
                    *[
                        {
                            "accepted_taxon_key": f"gbif:{100 + index}",
                            "scientific_name": species_name,
                            "rank": "SPECIES",
                            "parent_key": "gbif:90",
                            "family_key": "gbif:10",
                            "family": "Papilionidae",
                            "genus_key": "gbif:90",
                            "genus": "Papilio",
                            "species_key": f"gbif:{100 + index}",
                            "species": species_name,
                        }
                        for index, species_name in enumerate(species_names)
                    ],
                ],
                "names": [
                    {
                        "accepted_taxon_key": f"gbif:{100 + index}",
                        "verbatim_name": species_name,
                        "display_name": species_name,
                        "language": "la",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": f"gbif:{100 + index}",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    }
                    for index, species_name in enumerate(species_names)
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry"
    compile_registry_fixture(source, registry, registry_version="base", scope_path=scope)
    return registry, scope


def test_compile_enriched_registry_adds_enabled_names_and_preserves_candidates(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    base_manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "US",
                "name_class": "vernacular",
                "source": "ITIS",
                "source_record_id": "itis:common:1",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "Wikidata",
                "source_record_id": "wikidata:Q1:label:en",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Citrus butterfly",
                "language": "en",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular_alias",
                "source": "Wikidata",
                "source_record_id": "wikidata:Q1:alias:en:0",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "low",
                "enabled": False,
                "review_state": "candidate",
                "disabled_reason": "wikidata_alias_requires_corroboration",
            },
        ],
        external_links=[
            {
                "accepted_taxon_key": "gbif:100",
                "source": "Wikidata",
                "source_taxon_id": "Q1",
                "match_method": "gbif_taxon_id",
                "match_confidence": "high",
                "lineage_check": "accepted_taxon_key",
            }
        ],
        source_snapshots=[
            {
                "source": "Wikidata",
                "source_version": "wikidata-entities",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "source_path": "memory://wikidata",
                "source_response_hash": "sha256:test",
                "licence": "CC0-1.0",
            }
        ],
    )
    staged_manifest = json.loads((registry / "enrichment_manifest.json").read_text(encoding="utf-8"))
    assert json.loads((registry / "manifest.json").read_text(encoding="utf-8"))["registry_version"] == base_manifest["registry_version"]
    assert staged_manifest["schema_version"] == "registry-enrichment-v1"
    assert pl.read_parquet(registry / "source_snapshots.parquet").select("source").to_series().to_list() == ["GBIF"]

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    output = registry
    names = pl.read_parquet(output / "names.parquet")
    queries = pl.read_parquet(output / "flickr_query_definitions.parquet")
    candidates = pl.read_parquet(output / "name_candidates.parquet")
    evidence = pl.read_parquet(output / "name_evidence.parquet")
    links = pl.read_parquet(output / "external_taxon_links.parquet")
    snapshots = pl.read_parquet(output / "source_snapshots.parquet")

    assert manifest["registry_version"] == "enriched"
    assert manifest["enrichment_name_assertion_rows"] == 3
    assert manifest["enabled_enrichment_name_rows"] == 2
    assert "Lime Swallowtail" in names["display_name"].to_list()
    assert "Lime butterfly" in names["display_name"].to_list()
    assert "Citrus butterfly" not in names["display_name"].to_list()
    assert "Citrus butterfly" in candidates["display_name"].to_list()
    assert "Citrus butterfly" not in queries["normalized_query_term"].to_list()
    assert evidence.filter(pl.col("source") == "ITIS").height == 1
    assert evidence.filter(pl.col("source") == "Wikidata").height == 2
    assert links.select("source_taxon_id").to_series().to_list() == ["Q1"]
    assert snapshots.select("source").to_series().to_list() == ["GBIF", "Wikidata"]


def test_compile_enriched_registry_keeps_conflicting_source_name_disabled(tmp_path) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:missing",
                "display_name": "Bad Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:name:bad",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "medium",
                "enabled": True,
                "review_state": "accepted",
            }
        ],
    )

    manifest = compile_enriched_registry(
        registry_dir=registry,
        registry_version="enriched",
        scope_path=scope,
    )

    names = pl.read_parquet(registry / "names.parquet")
    candidates = pl.read_parquet(registry / "name_candidates.parquet")
    qa = pl.read_parquet(registry / "qa_findings.parquet")

    assert manifest["qa_status"] == "passed"
    assert "Bad Lime" not in names["display_name"].to_list()
    assert candidates.select("disabled_reason").to_series().to_list() == ["unknown_accepted_taxon_key"]
    assert {"severity": "warning", "code": "enrichment_name_without_base_taxon", "subject": "col:name:bad"} in qa.to_dicts()


def test_registry_compile_enriched_cli_writes_expanded_outputs(tmp_path, capsys) -> None:
    registry, scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "CoL Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:name:1",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            }
        ],
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "registry",
            "compile-enriched",
            "--registry-dir",
            str(registry),
            "--registry-version",
            "enriched",
            "--scope-json",
            str(scope),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "enriched"
    assert payload["registry_dir"] == str(registry)
    assert (registry / "external_taxon_links.parquet").exists()
    assert (registry / "source_name_assertions.parquet").exists()


class RecordingEnrichmentClient:
    def __init__(self, source: str, display_name: str, *, enabled: bool = True) -> None:
        self.source = source
        self.display_name = display_name
        self.enabled = enabled
        self.contexts = []

    def enrich_species(self, context):  # noqa: ANN001 - test double checks context shape.
        self.contexts.append(context)
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
                    "trust_tier": "T2" if self.source != "Wikidata" else "T3",
                    "precision_tier": "medium",
                    "confidence": "high",
                    "enabled": self.enabled,
                    "review_state": "accepted" if self.enabled else "candidate",
                    "disabled_reason": "" if self.enabled else "wikidata_alias_requires_corroboration",
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


def test_build_enrichment_sources_feeds_species_context_to_priority_services(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)
    clients = {
        "col": RecordingEnrichmentClient("CoL", "CoL Lime"),
        "wikidata": RecordingEnrichmentClient("Wikidata", "Wikidata Lime"),
        "itis": RecordingEnrichmentClient("ITIS", "ITIS Lime"),
    }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col", "wikidata", "itis"),
        clients=clients,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    links = pl.read_parquet(registry / "external_taxon_links.parquet")
    assert manifest["source_order"] == ["col", "wikidata", "itis"]
    assert manifest["species_seen"] == 1
    assert assertions.select("source").to_series().to_list() == ["CoL", "Wikidata", "ITIS"]
    assert links.height == 3
    assert [client.contexts[0].accepted_scientific_name for client in clients.values()] == ["Papilio demoleus"] * 3
    assert clients["col"].contexts[0].current_names == ("Papilio demoleus",)


def test_source_split_runs_merge_with_existing_staged_sources(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path)
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Existing CoL Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "CoL",
                "source_record_id": "col:existing",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
                "review_state": "accepted",
            }
        ],
    )

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("wikidata",),
        clients={"wikidata": RecordingEnrichmentClient("Wikidata", "Wikidata Lime")},
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    assert assertions.select("source").to_series().to_list() == ["CoL", "Wikidata"]
    assert assertions.select("display_name").to_series().to_list() == ["Existing CoL Lime", "Wikidata Lime"]
    assert manifest["source_order"] == ["wikidata"]
    assert manifest["name_assertion_rows"] == 2


def test_source_split_partial_run_replaces_only_completed_taxa(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon"))
    write_enrichment_sources(
        registry,
        name_assertions=[
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Old Wikidata Lime",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "Wikidata",
                "source_record_id": "wikidata:old:100",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "low",
                "enabled": False,
                "review_state": "candidate",
            },
            {
                "accepted_taxon_key": "gbif:101",
                "display_name": "Old Wikidata Machaon",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "name_class": "vernacular",
                "source": "Wikidata",
                "source_record_id": "wikidata:old:101",
                "trust_tier": "T3",
                "precision_tier": "medium",
                "confidence": "low",
                "enabled": False,
                "review_state": "candidate",
            },
        ],
    )

    class EmptyClient:
        def enrich_species(self, context):  # noqa: ANN001 - test double returns no new data for completed taxon.
            return {"name_assertions": [], "external_links": [], "source_snapshots": []}

    build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("wikidata",),
        clients={"wikidata": EmptyClient()},
        workers=1,
        limit=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    assert assertions.select("accepted_taxon_key").to_series().to_list() == ["gbif:101"]
    assert assertions.select("display_name").to_series().to_list() == ["Old Wikidata Machaon"]


def test_inaturalist_only_run_skips_species_with_existing_source_snapshot(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon"))
    write_enrichment_sources(
        registry,
        source_snapshots=[
            {
                "source": "iNaturalist",
                "source_version": "inaturalist-v1",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "source_path": "taxa:q=Papilio demoleus",
                "source_response_hash": "sha256:existing",
                "licence": "CC-BY-NC-4.0",
            }
        ],
    )
    client = RecordingEnrichmentClient("iNaturalist", "Machaon")

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("inaturalist",),
        clients={"inaturalist": client},
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assert [context.accepted_scientific_name for context in client.contexts] == ["Papilio machaon"]
    assert manifest["species_seen"] == 1
    assert manifest["completed_species"] == 1


def test_build_enrichment_sources_checkpoints_logs_and_reports(tmp_path, caplog) -> None:
    registry, _scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes"))
    caplog.set_level(logging.INFO, logger="biominer.registry.enrichment")

    def client_factory():
        return {"col": RecordingEnrichmentClient("CoL", "Source Lime")}

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col",),
        client_factory=client_factory,
        workers=2,
        progress_every=1,
        checkpoint_every=1,
        max_retries=0,
        report_dir=tmp_path / "reports",
    )

    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    staged_manifest = json.loads((registry / "enrichment_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "reports" / "registry_enrichment_base.json").read_text(encoding="utf-8"))
    log_text = "\n".join(record.getMessage() for record in caplog.records)

    assert manifest["workers"] == 2
    assert manifest["completed_species"] == 3
    assert manifest["name_assertion_rows"] == 3
    assert staged_manifest["completed_species"] == 3
    assert assertions.select("accepted_taxon_key").to_series().to_list() == ["gbif:100", "gbif:101", "gbif:102"]
    assert report["artifact_bytes"]["source_name_assertions.parquet"] > 0
    assertion_artifact = staged_manifest["artifacts"]["source_name_assertions.parquet"]
    assert assertion_artifact["rows"] == 3
    assert assertion_artifact["size_bytes"] > 0
    assert assertion_artifact["sha256"].startswith("sha256:")
    assert staged_manifest["artifacts"]["enrichment_manifest.json"]["sha256"].startswith("sha256:")
    assert "registry.enrichment.checkpoint_write" in log_text
    assert "source_name_assertions.parquet" in log_text
    assert "name_assertion_rows=3" in log_text


class ConcurrencyTrackingClient:
    def __init__(self, source: str, *, delay_seconds: float) -> None:
        self.source = source
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def enrich_species(self, context):  # noqa: ANN001 - test double tracks concurrent calls.
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            return {
                "name_assertions": [
                    {
                        "accepted_taxon_key": context.accepted_taxon_key,
                        "display_name": f"{self.source} {context.accepted_scientific_name}",
                        "language": "eng",
                        "script": "Latn",
                        "region": "",
                        "name_class": "vernacular",
                        "source": self.source,
                        "source_record_id": f"{self.source}:{context.accepted_taxon_key}",
                        "trust_tier": "T2",
                        "precision_tier": "medium",
                        "confidence": "high",
                        "enabled": True,
                        "review_state": "accepted",
                    }
                ],
                "external_links": [],
                "source_snapshots": [],
            }
        finally:
            with self.lock:
                self.active -= 1


def test_wikidata_source_is_limited_to_one_concurrent_query(tmp_path) -> None:
    registry, _scope = _write_base_registry(
        tmp_path,
        species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes", "Papilio xuthus"),
    )
    trackers = {
        "col": ConcurrencyTrackingClient("CoL", delay_seconds=0.03),
        "wikidata": ConcurrencyTrackingClient("Wikidata", delay_seconds=0.03),
        "itis": ConcurrencyTrackingClient("ITIS", delay_seconds=0.06),
    }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col", "wikidata", "itis"),
        clients=trackers,
        workers=4,
        progress_every=4,
        checkpoint_every=4,
        report_dir=tmp_path / "reports",
    )

    assert trackers["col"].max_active > 1
    assert trackers["itis"].max_active > 1
    assert trackers["wikidata"].max_active == 1
    assert manifest["source_worker_limits"] == {"col": 4, "wikidata": 1, "itis": 4}


def test_inaturalist_source_is_limited_to_one_concurrent_query(tmp_path) -> None:
    registry, _scope = _write_base_registry(
        tmp_path,
        species_names=("Papilio demoleus", "Papilio machaon", "Papilio polytes", "Papilio xuthus"),
    )
    trackers = {
        "col": ConcurrencyTrackingClient("CoL", delay_seconds=0.03),
        "inaturalist": ConcurrencyTrackingClient("iNaturalist", delay_seconds=0.03),
        "itis": ConcurrencyTrackingClient("ITIS", delay_seconds=0.06),
    }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("col", "inaturalist", "itis"),
        clients=trackers,
        workers=4,
        progress_every=4,
        checkpoint_every=4,
        report_dir=tmp_path / "reports",
    )

    assert trackers["col"].max_active > 1
    assert trackers["itis"].max_active > 1
    assert trackers["inaturalist"].max_active == 1
    assert manifest["source_worker_limits"] == {"col": 4, "inaturalist": 1, "itis": 4}


def test_registry_enrich_sources_cli_writes_sources_into_registry_dir(tmp_path, capsys, monkeypatch) -> None:
    registry, _scope = _write_base_registry(tmp_path)

    def fake_build(
        *,
        registry_dir,
        sources,
        workers,
        progress_every,
        checkpoint_every,
        max_retries,
        limit,
        report_dir,
    ):  # noqa: ANN001 - CLI wiring test.
        return write_enrichment_sources(
            registry_dir,
            name_assertions=[
                {
                    "accepted_taxon_key": "gbif:100",
                    "display_name": "CLI Lime",
                    "language": "eng",
                    "script": "Latn",
                    "name_class": "vernacular",
                    "source": "CoL",
                    "source_record_id": "col:cli",
                    "trust_tier": "T2",
                    "precision_tier": "medium",
                    "confidence": "high",
                    "enabled": True,
                    "review_state": "accepted",
                }
            ],
        ) | {
            "registry_dir": str(registry_dir),
            "source_order": list(sources),
            "species_seen": 1,
            "errors": [],
            "workers": workers,
            "progress_every": progress_every,
            "checkpoint_every": checkpoint_every,
            "max_retries": max_retries,
            "limit": limit,
            "report_dir": str(report_dir),
        }

    monkeypatch.setattr("biominer.cli.build_enrichment_sources_from_registry", fake_build)
    parser = build_parser()
    args = parser.parse_args(
        [
            "registry",
            "enrich-sources",
            "--registry-dir",
            str(registry),
            "--sources",
            "col,wikidata,itis",
            "--workers",
            "3",
            "--progress-every",
            "4",
            "--checkpoint-every",
            "5",
            "--max-retries",
            "6",
            "--limit",
            "7",
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_dir"] == str(registry)
    assert payload["source_order"] == ["col", "wikidata", "itis"]
    assert payload["workers"] == 3
    assert payload["progress_every"] == 4
    assert payload["checkpoint_every"] == 5
    assert payload["max_retries"] == 6
    assert payload["limit"] == 7
    assert payload["report_dir"] == str(tmp_path / "reports")
    assert (registry / "source_name_assertions.parquet").exists()


def test_source_clients_reuse_http_client_between_requests(monkeypatch) -> None:
    created_clients = []

    class FakeRetryingHTTPClient:
        def __init__(self, *, base_url: str, max_retries: int, timeout_seconds: float, sleep, headers: dict[str, str]) -> None:  # noqa: ANN001 - mirrors production seam.
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds
            self.headers = headers or {}
            self.paths = []
            created_clients.append(self)

        def get_json(self, path: str, *, params):
            self.paths.append(path)
            if path.endswith("searchByScientificName"):
                return {"scientificNames": [{"combinedName": "Papilio demoleus", "tsn": "123"}]}
            return {"commonNames": [{"commonName": "Lime Swallowtail", "language": "eng"}]}

    monkeypatch.setattr("biominer.registry.enrichment_sources.RetryingHTTPClient", FakeRetryingHTTPClient)

    client = ITISClient()
    result = client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert len(created_clients) == 1
    assert created_clients[0].paths == [
        "/ITISWebService/jsonservice/searchByScientificName",
        "/ITISWebService/jsonservice/getCommonNamesFromTSN",
    ]
    assert result["name_assertions"][0]["display_name"] == "Lime Swallowtail"


def test_json_get_uses_shared_retrying_http_client_with_user_agent(monkeypatch) -> None:
    created_clients = []
    calls = []

    class FakeRetryingHTTPClient:
        def __init__(self, *, base_url: str, max_retries: int, timeout_seconds: float, sleep, headers: dict[str, str]) -> None:  # noqa: ANN001 - mirrors production seam.
            self.base_url = base_url
            self.max_retries = max_retries
            self.timeout_seconds = timeout_seconds
            self.headers = headers
            created_clients.append(self)

        def get_json(self, path: str, *, params):
            calls.append({"path": path, "params": params})
            return [{"name": "ok"}]

    monkeypatch.setattr("biominer.registry.enrichment_sources.RetryingHTTPClient", FakeRetryingHTTPClient)
    payload = _json_get("https://example.test", max_retries=2, sleep=lambda _seconds: None)("/resource", {"q": "Papilio"})

    assert payload == {"results": [{"name": "ok"}]}
    assert calls == [{"path": "/resource", "params": {"q": "Papilio"}}]
    assert created_clients[0].base_url == "https://example.test"
    assert created_clients[0].max_retries == 2
    assert created_clients[0].headers["User-Agent"].startswith("BioMiner/")


def test_json_get_propagates_permanent_4xx_from_retrying_client(monkeypatch) -> None:
    attempts = []

    class FakeRetryingHTTPClient:
        def __init__(self, *, base_url: str, max_retries: int, timeout_seconds: float, sleep, headers: dict[str, str]) -> None:  # noqa: ANN001 - mirrors production seam.
            return None

        def get_json(self, path: str, *, params):
            attempts.append((path, params))
            request = httpx.Request("GET", "https://example.test/resource")
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    monkeypatch.setattr("biominer.registry.enrichment_sources.RetryingHTTPClient", FakeRetryingHTTPClient)

    try:
        _json_get("https://example.test", max_retries=2, sleep=lambda _seconds: None)("/resource", {})
    except httpx.HTTPStatusError:
        pass
    else:  # pragma: no cover - assertion path is clearer than pytest.raises import churn here.
        raise AssertionError("expected permanent HTTP error")

    assert len(attempts) == 1


def test_wikidata_client_uses_wbsearchentities_with_maxlag() -> None:
    calls = []

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        return {"search": [{"id": "Q1", "label": "Lime butterfly", "aliases": ["Citrus swallowtail"]}]}

    result = WikidataClient(http_get=fake_get).enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert calls == [
        {
            "path": "/w/api.php",
            "params": {
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "limit": 10,
                "maxlag": 5,
                "search": "Papilio demoleus",
                "type": "item",
            },
        }
    ]
    assert result["external_links"][0]["source_taxon_id"] == "Q1"
    assert [row["display_name"] for row in result["name_assertions"]] == ["Lime butterfly", "Citrus swallowtail"]


def test_inaturalist_client_uses_taxa_keyword_query_and_extracts_names() -> None:
    calls = []

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        return {
            "results": [
                {
                    "id": 52194,
                    "name": "Papilio demoleus",
                    "rank": "species",
                    "is_active": True,
                    "preferred_common_name": "Lime Swallowtail",
                    "matched_term": "Lime Butterfly",
                    "taxon_names": [
                        {"name": "Lime Swallowtail", "locale": "en", "lexicon": "English", "source_id": 1},
                        {"name": "Citrus Swallowtail", "locale": "en", "lexicon": "English", "source_id": 2},
                        {"name": "Papilio demoleus", "locale": "la", "lexicon": "Scientific Names", "source_id": 3},
                    ],
                }
            ]
        }

    result = INaturalistClient(http_get=fake_get, rate_limiter=INaturalistRateLimiter(min_delay_seconds=0.0)).enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert calls == [
        {
            "path": "/v1/taxa",
            "params": {"q": "Papilio demoleus", "rank": "species", "is_active": True, "all_names": True, "per_page": 10},
        }
    ]
    assert result["external_links"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "source": "iNaturalist",
            "source_taxon_id": "52194",
            "match_method": "scientific_name",
            "match_confidence": "high",
            "lineage_check": "gbif_scientific_name_exact",
        }
    ]
    assert [(row["display_name"], row["language"], row["trust_tier"], row["enabled"]) for row in result["name_assertions"]] == [
        ("Lime Swallowtail", "en", "T2", True),
        ("Lime Butterfly", "en", "T4", False),
        ("Citrus Swallowtail", "en", "T4", False),
    ]
    assert result["source_snapshots"][0]["source"] == "iNaturalist"
    assert result["source_snapshots"][0]["source_path"] == "taxa:q=Papilio demoleus"


def test_inaturalist_client_uses_autocomplete_fallback_when_taxa_has_no_exact_match() -> None:
    calls = []

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        if path == "/v1/taxa":
            return {"results": [{"id": 1, "name": "Papilio", "rank": "genus", "is_active": True}]}
        return {
            "results": [
                {
                    "id": 2,
                    "name": "Papilio demoleus",
                    "rank": "species",
                    "is_active": True,
                    "preferred_common_name": "Lime Swallowtail",
                }
            ]
        }

    result = INaturalistClient(http_get=fake_get, rate_limiter=INaturalistRateLimiter(min_delay_seconds=0.0)).enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert [call["path"] for call in calls] == ["/v1/taxa", "/v1/taxa/autocomplete"]
    assert result["external_links"][0]["source_taxon_id"] == "2"
    assert result["name_assertions"][0]["display_name"] == "Lime Swallowtail"


def test_inaturalist_client_rate_limits_taxa_and_autocomplete_requests() -> None:
    calls = []
    sleeps = []
    current_time = 100.0

    def monotonic() -> float:
        return current_time

    def sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        nonlocal current_time
        calls.append({"path": path, "params": params})
        current_time += 0.2
        if path == "/v1/taxa":
            return {"results": []}
        return {
            "results": [
                {
                    "id": 2,
                    "name": "Papilio demoleus",
                    "rank": "species",
                    "is_active": True,
                    "preferred_common_name": "Lime Swallowtail",
                }
            ]
        }

    client = INaturalistClient(
        http_get=fake_get,
        rate_limiter=INaturalistRateLimiter(min_delay_seconds=1.0, sleep=sleep, monotonic=monotonic),
    )

    client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert [call["path"] for call in calls] == ["/v1/taxa", "/v1/taxa/autocomplete"]
    assert sleeps == [1.0]


def test_inaturalist_429_waits_cooldown_and_does_not_retry_same_keyword() -> None:
    calls = []
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        request = httpx.Request("GET", "https://api.inaturalist.org/v1/taxa")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("normal_throttling", request=request, response=response)

    client = INaturalistClient(
        http_get=fake_get,
        rate_limiter=INaturalistRateLimiter(min_delay_seconds=1.0, sleep=sleep, monotonic=lambda: 100.0),
        rate_limit_cooldown_seconds=10.0,
    )

    try:
        client.enrich_species(
            SpeciesContext(
                accepted_taxon_key="gbif:100",
                accepted_scientific_name="Papilio demoleus",
                family_key="gbif:10",
                family="Papilionidae",
                genus_key="gbif:90",
                genus="Papilio",
                current_names=("Papilio demoleus",),
            )
        )
    except INaturalistRateLimitTermError:
        pass
    else:  # pragma: no cover - clearer assertion path than adding pytest import churn.
        raise AssertionError("expected iNaturalist term-level rate limit")

    assert len(calls) == 1
    assert sleeps == [10.0]


def test_inaturalist_client_rejects_non_exact_or_inactive_taxa_but_snapshots_query() -> None:
    def fake_get(_path: str, _params: dict[str, object]) -> dict[str, object]:
        return {
            "results": [
                {"id": 1, "name": "Papilio", "rank": "genus", "is_active": True, "preferred_common_name": "Swallowtails"},
                {"id": 2, "name": "Papilio demoleus", "rank": "species", "is_active": False, "preferred_common_name": "Lime Swallowtail"},
            ]
        }

    result = INaturalistClient(http_get=fake_get, rate_limiter=INaturalistRateLimiter(min_delay_seconds=0.0)).enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert result["external_links"] == []
    assert result["name_assertions"] == []
    assert result["source_snapshots"][0]["source_path"] == "taxa:q=Papilio demoleus"


def test_wikidata_client_rate_limits_uncached_requests_and_caches_no_results() -> None:
    calls = []
    sleeps = []
    current_time = 100.0

    def monotonic() -> float:
        return current_time

    def sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        return {"search": []}

    client = WikidataClient(
        http_get=fake_get,
        rate_limiter=WikidataRateLimiter(min_delay_seconds=1.5, sleep=sleep, monotonic=monotonic),
        cache={},
    )

    first = client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )
    second = client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:101",
            accepted_scientific_name="Papilio machaon",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio machaon",),
        )
    )
    cached = client.enrich_species(
        SpeciesContext(
            accepted_taxon_key="gbif:100",
            accepted_scientific_name="Papilio demoleus",
            family_key="gbif:10",
            family="Papilionidae",
            genus_key="gbif:90",
            genus="Papilio",
            current_names=("Papilio demoleus",),
        )
    )

    assert first["name_assertions"] == []
    assert second["name_assertions"] == []
    assert cached["name_assertions"] == []
    assert len(calls) == 2
    assert sleeps == [1.5]


def test_wikidata_rate_limit_waits_after_retry_penalty_completes() -> None:
    calls = []
    sleeps = []
    current_time = 100.0

    def monotonic() -> float:
        return current_time

    def sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        nonlocal current_time
        calls.append({"path": path, "params": params})
        current_time += 55.0
        return {"search": []}

    client = WikidataClient(
        http_get=fake_get,
        rate_limiter=WikidataRateLimiter(min_delay_seconds=2.0, sleep=sleep, monotonic=monotonic),
        cache={},
    )

    for taxon_key, scientific_name in (("gbif:100", "Papilio demoleus"), ("gbif:101", "Papilio machaon")):
        client.enrich_species(
            SpeciesContext(
                accepted_taxon_key=taxon_key,
                accepted_scientific_name=scientific_name,
                family_key="gbif:10",
                family="Papilionidae",
                genus_key="gbif:90",
                genus="Papilio",
                current_names=(scientific_name,),
            )
        )

    assert len(calls) == 2
    assert sleeps == [2.0]


def test_wikidata_retry_penalty_increases_following_request_delay() -> None:
    sleeps = []
    current_time = 100.0

    def monotonic() -> float:
        return current_time

    def sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    limiter = WikidataRateLimiter(min_delay_seconds=2.0, max_delay_seconds=6.0, sleep=sleep, monotonic=monotonic)
    limiter.wait()
    limiter.retry_sleep(55.0)
    limiter.record_request_complete()
    limiter.wait()
    limiter.retry_sleep(55.0)
    limiter.record_request_complete()
    limiter.wait()

    assert sleeps == [55.0, 4.0, 55.0, 6.0]


def test_wikidata_source_rate_limit_skips_term_and_continues(tmp_path) -> None:
    registry, _scope = _write_base_registry(tmp_path, species_names=("Papilio demoleus", "Papilio machaon"))
    calls = []

    class RateLimitedClient:
        def enrich_species(self, context):  # noqa: ANN001 - test double raises the fatal source error.
            calls.append(context.accepted_taxon_key)
            if context.accepted_taxon_key == "gbif:100":
                raise WikidataRateLimitTermError("wikidata rate limited")
            return {
                "name_assertions": [],
                "external_links": [],
                "source_snapshots": [{"source": "Wikidata", "source_version": "fixture", "source_path": "", "source_response_hash": "", "licence": ""}],
            }

    manifest = build_enrichment_sources_from_registry(
        registry_dir=registry,
        sources=("wikidata",),
        clients={"wikidata": RateLimitedClient()},
        workers=1,
        progress_every=1,
        checkpoint_every=1,
        report_dir=tmp_path / "reports",
    )

    assert calls == ["gbif:100", "gbif:101"]
    assert manifest["completed_species"] == 2
    assert manifest["error_counts_by_source"] == {"wikidata": 1}


def test_wikidata_429_waits_cooldown_and_does_not_retry_same_keyword() -> None:
    calls = []
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def fake_get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append({"path": path, "params": params})
        request = httpx.Request("GET", "https://www.wikidata.org/w/api.php")
        response = httpx.Response(429, request=request, headers={"Retry-After": "55"})
        raise httpx.HTTPStatusError("too many requests", request=request, response=response)

    client = WikidataClient(
        http_get=fake_get,
        rate_limiter=WikidataRateLimiter(min_delay_seconds=1.5, sleep=sleep, monotonic=lambda: 100.0),
        rate_limit_cooldown_seconds=45.0,
        cache={},
    )

    try:
        client.enrich_species(
            SpeciesContext(
                accepted_taxon_key="gbif:100",
                accepted_scientific_name="Papilio demoleus",
                family_key="gbif:10",
                family="Papilionidae",
                genus_key="gbif:90",
                genus="Papilio",
                current_names=("Papilio demoleus",),
            )
        )
    except WikidataRateLimitTermError:
        pass
    else:  # pragma: no cover - clearer assertion path than adding pytest import churn.
        raise AssertionError("expected Wikidata term-level rate limit")

    assert len(calls) == 1
    assert sleeps == [45.0]
