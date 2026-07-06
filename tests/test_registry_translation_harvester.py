from __future__ import annotations

import json

import polars as pl

from biominer.registry.build import build_registry
from biominer.registry.translation_harvester import (
    MyMemoryTranslationProvider,
    WikimediaLanglink,
    WikimediaLanglinksProvider,
    build_translation_candidates_from_registry,
    load_translation_target_locales,
)


def _write_registry(registry) -> None:  # noqa: ANN001 - pathlib fixture helper.
    registry.mkdir()
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family": "Papilionidae",
                "genus": "Papilio",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
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
                "verbatim_name": "Lime Swallowtail",
                "display_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular",
                "source": "GBIF",
                "source_record_id": "gbif:100:vernacular:Lime Swallowtail",
                "trust_tier": "T2",
                "precision_tier": "medium",
                "confidence": "high",
                "enabled": True,
            },
        ]
    ).write_parquet(registry / "names.parquet")


def _write_wikidata_link(registry, *, qid: str = "Q123") -> None:  # noqa: ANN001 - pathlib fixture helper.
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "source": "Wikidata",
                "source_taxon_id": qid,
                "match_method": "P225+P846",
                "match_confidence": "high",
                "lineage_check": "accepted_taxon_key",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
            }
        ]
    ).write_parquet(registry / "external_taxon_links.parquet")


def _scope(path) -> None:  # noqa: ANN001 - pathlib fixture helper.
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


class RecordingSourceClient:
    def __init__(self, source: str, display_name: str) -> None:
        self.source = source
        self.display_name = display_name

    def enrich_species(self, context):  # noqa: ANN001, ANN202 - test double.
        is_wikidata = self.source == "Wikidata"
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
                    "source": "Wikidata",
                    "source_taxon_id": "Q123",
                    "match_method": "P225+P846",
                    "match_confidence": "high",
                    "lineage_check": "accepted_taxon_key",
                    "retrieved_at": "2026-06-20T00:00:00+00:00",
                }
            ]
            if is_wikidata
            else [],
            "source_snapshots": [],
        }


class RecordingTMDClient:
    def enrich_registry(self, *, taxa_rows, name_rows):  # noqa: ANN001, ANN202 - test double.
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
            "external_links": [],
            "source_snapshots": [],
            "coverage": {"request_count": 1},
        }


class FakeWikimediaProvider:
    def __init__(self, *, wikidata_item: str = "Q123") -> None:
        self.titles: list[str] = []
        self.wikidata_item = wikidata_item

    def langlinks(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
        self.titles.append(title)
        assert target_locales == ("de",)
        return [WikimediaLanglink(language="de", title="Zitronenfalter", page_id="123", page_title=title, wikidata_item=self.wikidata_item)], 1, title


class LocaleVariantWikimediaProvider:
    def langlinks(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
        assert target_locales == ("pt-BR", "zh-Hant")
        return [
            WikimediaLanglink(language="pt-BR", title="Borboleta lima", page_id="123", page_title=title, wikidata_item="Q123"),
            WikimediaLanglink(language="zh-Hant", title="青鳳蝶", page_id="123", page_title=title, wikidata_item="Q123"),
        ], 1, title


class FakeMyMemoryProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        self.calls.append(
            {
                "source_name": source_name,
                "source_language": source_language,
                "target_language": target_language,
                "max_candidates": str(max_candidates),
            }
        )
        return ["Limettenfalter"], 1


def test_mymemory_provider_uses_translation_memory_mode_by_default() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        requests.append((path, params))
        return {
            "responseData": {"translatedText": "Limettenfalter"},
            "matches": [{"translation": "Limettenfalter"}],
        }

    provider = MyMemoryTranslationProvider(http_get=fake_get)

    translations, request_count = provider.translate(
        source_name="Lime Swallowtail",
        source_language="eng",
        target_language="deu",
        max_candidates=3,
    )

    assert translations == ["Limettenfalter"]
    assert request_count == 1
    assert requests == [
        (
            "/get",
            {
                "q": "Lime Swallowtail",
                "langpair": "en|de",
                "mt": "0",
            },
        )
    ]


def test_mymemory_provider_keeps_all_candidates_when_uncapped() -> None:
    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        return {
            "responseData": {"translatedText": "Limettenfalter"},
            "matches": [
                {"translation": "Limettenfalter"},
                {"translation": "Zitronenschwalbenschwanz"},
                {"translation": "Zitrusfalter"},
            ],
        }

    provider = MyMemoryTranslationProvider(http_get=fake_get)

    translations, request_count = provider.translate(
        source_name="Lime Swallowtail",
        source_language="eng",
        target_language="deu",
        max_candidates=0,
    )

    assert request_count == 1
    assert translations == ["Limettenfalter", "Zitronenschwalbenschwanz", "Zitrusfalter"]


def test_mymemory_provider_can_enable_machine_translation() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        requests.append((path, params))
        return {"responseData": {"translatedText": "Limettenfalter"}}

    provider = MyMemoryTranslationProvider(http_get=fake_get, allow_machine_translation=True)
    provider.translate(source_name="Lime Swallowtail", source_language="eng", target_language="deu", max_candidates=1)

    assert requests[0][1]["mt"] == "1"


def test_wikimedia_provider_follows_langlink_continuation() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        requests.append((path, dict(params)))
        if len(requests) == 1:
            return {
                "query": {
                    "pages": {
                        "123": {
                            "pageid": 123,
                            "title": "Papilio demoleus",
                            "pageprops": {"wikibase_item": "Q123"},
                            "langlinks": [{"lang": "de", "*": "Zitronenfalter"}],
                        }
                    }
                },
                "continue": {"llcontinue": "123|fr", "continue": "||"},
            }
        return {
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Papilio demoleus",
                        "pageprops": {"wikibase_item": "Q123"},
                        "langlinks": [{"lang": "fr", "*": "Papillon du citronnier"}],
                    }
                }
            },
        }

    provider = WikimediaLanglinksProvider(http_get=fake_get)

    links, request_count, page_title = provider.langlinks("Papilio demoleus", target_locales=("de", "fr"))

    assert request_count == 2
    assert page_title == "Papilio demoleus"
    assert [link.title for link in links] == ["Zitronenfalter", "Papillon du citronnier"]
    assert [link.wikidata_item for link in links] == ["Q123", "Q123"]
    assert "pageprops" in requests[0][1]["prop"]
    assert requests[0][1]["ppprop"] == "wikibase_item"
    assert requests[0][1]["lllimit"] == "max"
    assert requests[1][1]["llcontinue"] == "123|fr"


def test_load_translation_target_locales_preserves_bcp47_variants(tmp_path) -> None:
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["pt", "pt-BR", "zh", "zh-Hant"]), encoding="utf-8")

    assert load_translation_target_locales(locales) == ("pt", "pt-BR", "zh", "zh-Hant")


def test_translation_harvester_preserves_wikimedia_bcp47_variant_languages(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    _write_wikidata_link(registry, qid="Q123")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["pt-BR", "zh-Hant"]), encoding="utf-8")

    build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("wikimedia",),
        target_locales_json=locales,
        providers={"wikimedia": LocaleVariantWikimediaProvider()},
    )

    assertions = pl.read_parquet(registry / "enrichment" / "source_name_assertions.parquet").sort("language")

    assert assertions.select("language").to_series().to_list() == ["por", "zho"]
    assert assertions.select("script").to_series().to_list() == ["Latn", "Hant"]
    assert assertions.select("region").to_series().to_list() == ["BR", ""]
    assert assertions.select("display_name").to_series().to_list() == ["Borboleta lima", "青鳳蝶"]


def test_translation_harvester_writes_wikimedia_and_mymemory_outputs(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    _write_wikidata_link(registry, qid="Q123")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    wiki = FakeWikimediaProvider()
    mymemory = FakeMyMemoryProvider()

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        target_locales_json=locales,
        providers={"wikimedia": wiki, "mymemory": mymemory},
    )

    assertions = pl.read_parquet(registry / "enrichment" / "source_name_assertions.parquet")
    candidates = pl.read_parquet(registry / "enrichment" / "translation_candidates.parquet")
    work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet")

    assert manifest["wikimedia_assertion_rows"] == 1
    assert manifest["mymemory_candidate_rows"] == 1
    assert assertions.select("source").to_series().to_list() == ["Wikimedia"]
    assert assertions.select("display_name").to_series().to_list() == ["Zitronenfalter"]
    assert assertions.select("source_taxon_id").to_series().to_list() == ["Q123"]
    assert assertions.select("enabled").to_series().to_list() == [True]
    assert candidates.select("source").to_series().to_list() == ["MyMemory"]
    assert candidates.select("translated_name").to_series().to_list() == ["Limettenfalter"]
    assert wiki.titles == ["Papilio demoleus", "Lime Swallowtail"]
    assert mymemory.calls == [{"source_name": "Lime Swallowtail", "source_language": "en", "target_language": "de", "max_candidates": "0"}]
    assert set(work.select("source").to_series().to_list()) == {"wikimedia", "mymemory"}


def test_translation_harvester_disables_unbound_wikimedia_langlinks(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    _write_wikidata_link(registry, qid="Q123")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("wikimedia",),
        target_locales_json=locales,
        providers={"wikimedia": FakeWikimediaProvider(wikidata_item="Q999")},
    )

    assertions = pl.read_parquet(registry / "enrichment" / "source_name_assertions.parquet")

    assert manifest["wikimedia_assertion_rows"] == 1
    assert assertions.select("display_name").to_series().to_list() == ["Zitronenfalter"]
    assert assertions.select("enabled").to_series().to_list() == [False]
    assert assertions.select("review_state").to_series().to_list() == ["candidate"]
    assert assertions.select("source_taxon_id").to_series().to_list() == [""]
    assert assertions.select("disabled_reason").to_series().to_list() == ["wikimedia_page_not_bound_to_accepted_taxon"]


def test_harvester_does_not_translate_scientific_names_with_mymemory(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    names = pl.read_parquet(registry / "names.parquet")
    names.filter(pl.col("name_class") == "accepted_scientific").write_parquet(registry / "names.parquet")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")

    class FailingMyMemoryProvider:
        def translate(self, **kwargs):  # noqa: ANN003, ANN202 - should not be called.
            raise AssertionError("scientific names should not be translated")

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": FailingMyMemoryProvider()},
    )

    candidates = pl.read_parquet(registry / "enrichment" / "translation_candidates.parquet")
    assert manifest["mymemory_candidate_rows"] == 0
    assert candidates.is_empty()


def test_registry_build_promotes_translation_outputs(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "scope.json"
    _scope(scope)
    source = tmp_path / "gbif.json"
    source.write_text(json.dumps(_gbif_snapshot()), encoding="utf-8")
    clients = {
        "col": RecordingSourceClient("CoL", "Lime Swallowtail"),
        "itis": RecordingSourceClient("ITIS", "Lime Butterfly"),
        "inaturalist": RecordingSourceClient("iNaturalist", "Chequered Swallowtail"),
        "tmd_de": RecordingTMDClient(),
        "wikidata": RecordingSourceClient("Wikidata", "Wikidata Lime"),
    }
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    monkeypatch.setattr("biominer.registry.enrichment.default_enrichment_clients", lambda max_retries=5: clients)
    monkeypatch.setattr(
        "biominer.registry.translation_harvester._default_translation_providers",
        lambda **kwargs: {"wikimedia": FakeWikimediaProvider(), "mymemory": FakeMyMemoryProvider()},
    )

    result = build_registry(
        output_dir=tmp_path / "registry",
        registry_version="translated",
        scope_path=scope,
        source_json=source,
        reuse_source_json=True,
        report_dir=tmp_path / "reports",
        workers=1,
        translation_target_locales_json=locales,
    )

    registry = tmp_path / "registry"
    names = pl.read_parquet(registry / "names.parquet")
    queries = pl.read_parquet(registry / "flickr_query_definitions.parquet")
    manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))

    assert result["manifest"]["qa_status"] == "passed"
    assert {"Zitronenfalter", "Limettenfalter"}.issubset(set(names["display_name"].to_list()))
    assert "Zitronenfalter" in set(queries["source_term"].to_list())
    assert "Limettenfalter" not in set(queries["source_term"].to_list())
    assert names.filter(pl.col("normalized_match_key") == "limettenfalter").select("query_eligible").to_series().to_list() == [False]
    assert manifest["translation_sources"] == ["wikimedia", "mymemory"]
    assert manifest["enabled_t5_name_rows"] == 1
    assert manifest["t5_query_definition_rows"] == 0
    assert (registry / "translation_candidates.parquet").exists()
    assert (registry / "translation_work_ledger.parquet").exists()
