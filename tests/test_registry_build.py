from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.registry.build import build_registry
from biominer.registry.enrichment_sources import GBIFVernacularClient, TAXREFFrenchClient
from biominer.registry.range_discovery import OccurrenceCountryDetails, OccurrenceCountryFacet


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
        is_wikidata = self.source == "Wikidata"
        source_taxon_id = f"{self.source}:taxon:{context.accepted_taxon_key}"
        return {
            "name_assertions": [
                {
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "display_name": self.display_name,
                    "language": "eng",
                    "script": "Latn",
                    "region": "",
                    "name_class": "vernacular_alias" if is_wikidata else "vernacular",
                    "source": self.source,
                    "source_record_id": f"{self.source}:name:{context.accepted_taxon_key}",
                    "source_taxon_id": source_taxon_id if is_wikidata else "",
                    "lineage_check": "accepted_taxon_key" if is_wikidata else "",
                    "trust_tier": "T3" if is_wikidata else "T2",
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
                    "source_taxon_id": source_taxon_id,
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


class RecordingTMDClient:
    def enrich_registry(self, *, taxa_rows, name_rows):  # noqa: ANN001 - test double.
        assert any(row["scientific_name"] == "Papilio demoleus" for row in taxa_rows)
        assert any(row["display_name"] == "Papilio demoleus" for row in name_rows)
        return {
            "name_assertions": [
                {
                    "accepted_taxon_key": "gbif:100",
                    "display_name": "Zitronen-Schwalbenschwanz",
                    "language": "deu",
                    "script": "Latn",
                    "region": "DE",
                    "name_class": "vernacular",
                    "source": "TMD",
                    "source_record_id": "tmd:410:1:Zitronen-Schwalbenschwanz",
                    "source_taxon_id": "1",
                    "trust_tier": "T2",
                    "precision_tier": "high",
                    "confidence": "high",
                    "enabled": True,
                    "review_state": "accepted",
                }
            ],
            "external_links": [
                {
                    "accepted_taxon_key": "gbif:100",
                    "source": "TMD",
                    "source_taxon_id": "1",
                    "match_method": "scientific_name",
                    "match_confidence": "high",
                    "lineage_check": "accepted_taxon_key",
                }
            ],
            "source_snapshots": [
                {
                    "source": "TMD",
                    "source_version": "fixture",
                    "retrieved_at": "2026-06-20T00:00:00+00:00",
                    "source_path": "memory://tmd",
                    "source_response_hash": "sha256:tmd",
                    "licence": "",
                }
            ],
            "coverage": {"family_genus_german_labels": "not_available_from_tmd", "request_count": 1},
        }


class EmptyStaticClient:
    def __init__(self, source: str) -> None:
        self.source = source

    def enrich_registry(self, *, taxa_rows, name_rows):  # noqa: ANN001 - test double.
        return {
            "name_assertions": [],
            "external_links": [],
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
            "coverage": {"name_assertions": 0, "request_count": 0},
        }


class RecordingRangeClient:
    def __init__(self) -> None:
        self.facet_keys: list[str] = []
        self.detail_requests: list[tuple[str, str]] = []

    def country_facets(self, accepted_taxon_key: str, *, facet_limit: int = 300) -> tuple[OccurrenceCountryFacet, ...]:
        self.facet_keys.append(accepted_taxon_key)
        return (OccurrenceCountryFacet(country_code="IN", occurrence_count=4),)

    def country_details(self, accepted_taxon_key: str, country_code: str) -> OccurrenceCountryDetails:
        self.detail_requests.append((accepted_taxon_key, country_code))
        return OccurrenceCountryDetails(
            country_code=country_code,
            georeferenced_count=3,
            basis_of_record_counts={"HUMAN_OBSERVATION": 4},
            first_year=2010,
            last_year=2024,
        )


def test_registry_build_outputs_one_canonical_enriched_register_by_default(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    clients = {
        "col": RecordingSourceClient("CoL", "Lime Swallowtail"),
        "gbif_vernacular": GBIFVernacularClient(),
        "itis": RecordingSourceClient("ITIS", "Lime Butterfly"),
        "inaturalist": RecordingSourceClient("iNaturalist", "Chequered Swallowtail"),
        "taxref_fr": TAXREFFrenchClient(taxref_rows=[]),
        "tmd_de": RecordingTMDClient(),
        "wikidata": RecordingSourceClient("Wikidata", "Wikidata Lime"),
        "boi_india_en": EmptyStaticClient("Butterflies of India"),
        "bharat_ki_titliya_hi": EmptyStaticClient("Bharat Ki Titliya"),
        "karnataka_chitte_kn": EmptyStaticClient("Karnataka Chitte"),
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
        skip_translations=True,
    )

    registry = tmp_path / "registry"
    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    assertions = pl.read_parquet(registry / "source_name_assertions.parquet")
    errors = pl.read_parquet(registry / "source_error_records.parquet")
    manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))

    assert result["manifest"]["qa_status"] == "passed"
    assert manifest["enrichment_sources"] == [
        "col",
        "inaturalist",
        "itis",
        "tmd_de",
        "wikidata",
        "gbif_vernacular",
        "taxref_fr",
        "boi_india_en",
        "bharat_ki_titliya_hi",
        "karnataka_chitte_kn",
    ]
    assert {"Lime Swallowtail", "Lime Butterfly", "Chequered Swallowtail", "Zitronen-Schwalbenschwanz", "Wikidata Lime"}.issubset(
        set(names["display_name"].to_list())
    )
    wikidata_row = assertions.filter(pl.col("source") == "Wikidata").to_dicts()[0]
    assert wikidata_row["display_name"] == "Wikidata Lime"
    assert wikidata_row["trust_tier"] == "T3"
    assert "Lime Swallowtail" in queries["source_term"].to_list()
    assert "Wikidata Lime" in queries["source_term"].to_list()
    assert errors.is_empty()


def test_registry_build_writes_range_and_language_targets_when_configured(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    range_seed = tmp_path / "range_seed.json"
    range_seed.write_text(
        json.dumps(
            {
                "schema_version": "range-seed-v1",
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "regions": [
                    {
                        "region": "South Asia",
                        "range_status": "native_or_long_established",
                        "countries": [{"code": "IN", "name": "India"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    language_targets = tmp_path / "language_targets.json"
    language_targets.write_text(
        json.dumps(
            {
                "schema_version": "species-region-language-targets-v1",
                "source": "fixture_region_languages",
                "source_version": "fixture-v1",
                "regions": [
                    {
                        "region": "South Asia",
                        "languages": [
                            {"language": "eng", "language_name": "English", "priority": 10},
                            {"language": "hin", "language_name": "Hindi", "priority": 20},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    range_client = RecordingRangeClient()
    monkeypatch.setattr("biominer.registry.build.GBIFOccurrenceCountryClient", lambda: range_client)

    result = build_registry(
        output_dir=tmp_path / "registry",
        registry_version="regional",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir=tmp_path / "reports",
        range_seed_json=range_seed,
        language_targets_json=language_targets,
        skip_enrichment=True,
    )

    registry = tmp_path / "registry"
    range_rows = pl.read_parquet(registry / "range_countries.parquet").to_dicts()
    target_rows = pl.read_parquet(registry / "country_language_targets.parquet").sort("language_code").to_dicts()

    assert result["manifest"]["range_country_rows"] == 1
    assert result["manifest"]["language_target_rows"] == 2
    assert range_client.facet_keys == ["gbif:100"]
    assert range_client.detail_requests == [("gbif:100", "IN")]
    assert range_rows[0]["country_name"] == "India"
    assert range_rows[0]["region"] == "South Asia"
    assert [row["language_code"] for row in target_rows] == ["eng", "hin"]


def test_registry_build_skip_flags_disable_regional_outputs(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    range_seed = tmp_path / "range_seed.json"
    range_seed.write_text(
        json.dumps(
            {
                "schema_version": "range-seed-v1",
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "regions": [],
            }
        ),
        encoding="utf-8",
    )
    language_targets = tmp_path / "language_targets.json"
    language_targets.write_text(
        json.dumps({"schema_version": "species-region-language-targets-v1", "regions": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr("biominer.registry.build.GBIFOccurrenceCountryClient", lambda: pytest.fail("range discovery should be skipped"))

    result = build_registry(
        output_dir=tmp_path / "registry",
        registry_version="regional-skip",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir=tmp_path / "reports",
        range_seed_json=range_seed,
        language_targets_json=language_targets,
        skip_range_discovery=True,
        skip_language_targets=True,
        skip_enrichment=True,
    )

    registry = tmp_path / "registry"
    assert result["manifest"]["range_country_rows"] == 0
    assert result["manifest"]["language_target_rows"] == 0
    assert not (registry / "range_countries.parquet").exists()
    assert not (registry / "country_language_targets.parquet").exists()


def test_registry_build_requires_storage_backend_for_cloud_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")

    with pytest.raises(ValueError, match="storage_backend_required_for_cloud_registry"):
        build_registry(
            output_dir="s3://biominer/registry/version=test",
            registry_version="test",
            scope_path=scope,
            source_json=source,
            reuse_source_json=True,
            skip_enrichment=True,
        )

    assert not (tmp_path / "s3:").exists()


def test_cloud_registry_build_writes_canonical_artifacts_to_s3_version_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    storage = _FakeRegistryStorage()

    result = build_registry(
        output_dir="s3://biominer/biominer",
        registry_version="cloud-test",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir="s3://biominer/biominer/reports",
        skip_enrichment=True,
        storage=storage,
    )

    registry_prefix = "s3://biominer/biominer/registry/version=cloud-test"
    assert result["registry_prefix"] == registry_prefix
    for filename in (
        "taxa.parquet",
        "taxon_relations.parquet",
        "names.parquet",
        "name_evidence.parquet",
        "source_snapshots.parquet",
        "flickr_query_definitions.parquet",
        "qa_findings.parquet",
    ):
        assert f"{registry_prefix}/{filename}" in storage.parquet_payloads
    assert storage.json_payloads[f"{registry_prefix}/manifest.json"]["registry_version"] == "cloud-test"
    assert storage.json_payloads[f"{registry_prefix}/gbif_source_snapshot.json"]["source"] == "GBIF"
    assert storage.json_payloads["s3://biominer/biominer/reports/registry_build_cloud-test.json"]["status"] == "passed"
    current_pointer = storage.json_payloads["s3://biominer/biominer/registry/current/manifest.json"]
    assert current_pointer["registry_version"] == "cloud-test"
    assert current_pointer["registry_prefix"] == registry_prefix
    assert current_pointer["manifest_uri"] == f"{registry_prefix}/manifest.json"
    assert "promoted_at" in current_pointer
    assert "s3://biominer/biominer/registry/current/taxa.parquet" not in storage.parquet_payloads
    assert not (tmp_path / "s3:").exists()


def test_cloud_registry_build_resumes_from_existing_source_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scope = tmp_path / "scope.json"
    _scope(scope)
    storage = _FakeRegistryStorage()
    source_uri = "s3://biominer/biominer/registry/version=cloud-resume/gbif_source_snapshot.json"
    storage.json_payloads[source_uri] = _gbif_snapshot()

    def fail_gbif_client(*args, **kwargs):  # noqa: ANN001, ANN202 - should not be called during resume.
        raise AssertionError("cloud registry resume should reuse existing source snapshot")

    monkeypatch.setattr("biominer.registry.build.ProductionGBIFClient", fail_gbif_client)

    result = build_registry(
        output_dir="s3://biominer/biominer",
        registry_version="cloud-resume",
        scope_path=scope,
        report_dir="s3://biominer/biominer/reports",
        skip_enrichment=True,
        storage=storage,
    )

    registry_prefix = "s3://biominer/biominer/registry/version=cloud-resume"
    assert result["source_json"] == source_uri
    assert result["registry_prefix"] == registry_prefix
    assert storage.json_payloads[f"{registry_prefix}/manifest.json"]["qa_status"] == "passed"
    assert f"{registry_prefix}/taxa.parquet" in storage.parquet_payloads


def test_registry_build_quarantines_source_errors_without_siloing_successful_names(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    clients = {
        "col": RecordingSourceClient("CoL", "Lime Swallowtail"),
        "gbif_vernacular": GBIFVernacularClient(),
        "itis": RecordingSourceClient("ITIS", "Broken ITIS", raise_error=True),
        "inaturalist": RecordingSourceClient("iNaturalist", "Chequered Swallowtail"),
        "taxref_fr": TAXREFFrenchClient(taxref_rows=[]),
        "tmd_de": RecordingTMDClient(),
        "wikidata": RecordingSourceClient("Wikidata", "Wikidata Lime"),
        "boi_india_en": EmptyStaticClient("Butterflies of India"),
        "bharat_ki_titliya_hi": EmptyStaticClient("Bharat Ki Titliya"),
        "karnataka_chitte_kn": EmptyStaticClient("Karnataka Chitte"),
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
        skip_translations=True,
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


class _FakeRegistryStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}
        self.json_payloads: dict[str, dict[str, object]] = {}

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def read_json(self, uri: str) -> dict[str, object]:
        return self.json_payloads[uri]

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads or uri in self.json_payloads
