from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import polars as pl
import pytest

from biominer.registry import translation_harvester as harvester
from biominer.registry.build import build_registry
from biominer.registry.translation_harvester import (
    MyMemoryTranslationProvider,
    SpeciesTranslationContext,
    WikimediaLanglink,
    WikimediaLanglinksProvider,
    _seed_names_by_taxon,
    _translation_config_hash,
    _translation_work_key,
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
        self.vernacular_titles: list[str] = []
        self.wikidata_item = wikidata_item

    def langlinks(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
        self.titles.append(title)
        assert target_locales == ("de",)
        return [WikimediaLanglink(language="de", title="Zitronenfalter", page_id="123", page_title=title, wikidata_item=self.wikidata_item)], 1, title

    def vernacular_names(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
        self.vernacular_titles.append(title)
        assert target_locales == ("de",)
        return [], 0, title


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


class ByteCountingMyMemoryProvider:
    def __init__(self, *, response_bytes: int) -> None:
        self.response_bytes = response_bytes
        self.calls: list[str] = []

    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        self.calls.append(target_language)
        return [f"{target_language} translation"], 1, self.response_bytes


class InterruptingMyMemoryProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        self.calls.append(target_language)
        if target_language == "sv":
            raise KeyboardInterrupt
        return ["Limettenfalter"], 1


class FailingIfCalledMyMemoryProvider:
    def __init__(self) -> None:
        self.calls = 0

    def translate(self, **kwargs):  # noqa: ANN003, ANN202 - should not be called.
        self.calls += 1
        raise AssertionError("completed translation work should be skipped")


class MultiCandidateMyMemoryProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        self.calls.append(target_language)
        return ["Limettenfalter", "Zitrusfalter"], 1


class RuntimeErrorMyMemoryProvider:
    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        raise RuntimeError("translation backend unavailable")


class RecordingMyMemoryProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, *, source_name, source_language, target_language, max_candidates):  # noqa: ANN001, ANN202 - test double.
        self.calls.append(target_language)
        return [f"{target_language} translation"], 1


def test_mymemory_work_units_are_keyed_by_language_and_skip_completed(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    seeds = _seed_names_by_taxon(
        taxa_rows=taxa.iter_rows(named=True),
        name_rows=names.iter_rows(named=True),
        source_assertions=[],
    )["gbif:100"]
    context = SpeciesTranslationContext("gbif:100", "Papilio demoleus")
    config_hash = _translation_config_hash(
        "mymemory",
        ("de", "sv", "fi"),
        max_candidates_per_name=1,
        allow_machine_translation=False,
    )
    completed = {
        _translation_work_key(
            "mymemory",
            "gbif:100",
            "Lime Swallowtail",
            "en",
            "de",
            config_hash,
        )
    }

    units = harvester._mymemory_work_units(
        context,
        seeds,
        target_locales=("de", "sv", "fi"),
        completed_work=completed,
        config_hash=config_hash,
    )

    assert [(unit.seed.source_name, unit.target_language) for unit in units] == [
        ("Lime Swallowtail", "sv"),
        ("Lime Swallowtail", "fi"),
    ]


def test_mymemory_parallelism_partitions_languages_not_keywords() -> None:
    shards = harvester._partition_languages(("de", "sv", "fi", "fr", "es", "pt", "nl", "da"), 4)
    assert shards == (("de", "es"), ("sv", "pt"), ("fi", "nl"), ("fr", "da"))


def test_translation_request_budget_is_thread_safe() -> None:
    budget = harvester.TranslationRequestBudget(daily_limit=10, existing_work=[])

    with ThreadPoolExecutor(max_workers=8) as executor:
        reservations = list(executor.map(lambda _: budget.reserve(1), range(200)))

    assert sum(reservations) == 10
    assert budget.used == 10


def test_mymemory_monthly_input_word_limit_blocks_excess_before_request(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv", "fi"]), encoding="utf-8")
    mymemory = ByteCountingMyMemoryProvider(response_bytes=12)

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": mymemory},
        mymemory_monthly_input_word_limit=3,
        mymemory_monthly_request_limit=100,
        mymemory_monthly_bandwidth_mb_limit=100,
        mymemory_response_byte_reservation=1,
        translation_checkpoint_every=1,
    )

    work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet").sort("target_language")

    assert mymemory.calls == ["de"]
    assert manifest["translation_status"] == "budget_exhausted"
    assert manifest["mymemory_monthly_input_words_used"] == 2
    assert work.select(["target_language", "status", "request_count", "input_word_count"]).to_dicts() == [
        {"target_language": "de", "status": "complete", "request_count": 1, "input_word_count": 2},
        {"target_language": "sv", "status": "budget_exhausted", "request_count": 0, "input_word_count": 0},
    ]


def test_mymemory_monthly_bandwidth_reservation_blocks_request_before_provider_call(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    mymemory = ByteCountingMyMemoryProvider(response_bytes=12)

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": mymemory},
        mymemory_monthly_input_word_limit=100,
        mymemory_monthly_request_limit=100,
        mymemory_monthly_bandwidth_mb_limit=0,
        mymemory_response_byte_reservation=1,
        translation_checkpoint_every=1,
    )

    work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet")

    assert mymemory.calls == []
    assert manifest["translation_status"] == "budget_exhausted"
    assert manifest["mymemory_monthly_bandwidth_reserved_bytes"] == 0
    assert work.select(["target_language", "status", "request_count", "bandwidth_reserved_byte_count"]).to_dicts() == [
        {"target_language": "de", "status": "budget_exhausted", "request_count": 0, "bandwidth_reserved_byte_count": 0}
    ]


def test_mymemory_parallel_workers_create_provider_per_language_shard(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv", "fi", "fr"]), encoding="utf-8")
    providers: list[RecordingMyMemoryProvider] = []

    def provider_factory() -> RecordingMyMemoryProvider:
        provider = RecordingMyMemoryProvider()
        providers.append(provider)
        return provider

    build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": provider_factory},
        translation_workers=2,
        translation_language_shards=2,
    )

    assert len(providers) == 2
    assert sorted(tuple(provider.calls) for provider in providers) == [("de", "fi"), ("sv", "fr")]


def test_mymemory_parallel_work_is_drained_in_bounded_unit_batches(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv", "fi", "fr"]), encoding="utf-8")
    batch_sizes: list[int] = []
    original_harvest = harvester._harvest_mymemory_units_parallel

    def recording_harvest(units, **kwargs):  # noqa: ANN001, ANN003, ANN202 - wraps the production helper.
        batch_sizes.append(len(units))
        return original_harvest(units, **kwargs)

    monkeypatch.setattr(harvester, "_harvest_mymemory_units_parallel", recording_harvest)

    build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": RecordingMyMemoryProvider},
        translation_workers=2,
        translation_language_shards=2,
        translation_unit_batch_size=2,
    )

    assert batch_sizes == [2, 2]


def test_mymemory_work_units_preserve_bcp47_target_locales_and_api_codes(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    seeds = _seed_names_by_taxon(
        taxa_rows=taxa.iter_rows(named=True),
        name_rows=names.iter_rows(named=True),
        source_assertions=[],
    )["gbif:100"]
    context = SpeciesTranslationContext("gbif:100", "Papilio demoleus")
    target_locales = ("pt", "pt-BR", "zh-Hant")
    config_hash = _translation_config_hash(
        "mymemory",
        target_locales,
        max_candidates_per_name=1,
        allow_machine_translation=False,
    )

    units = harvester._mymemory_work_units(
        context,
        seeds,
        target_locales=target_locales,
        completed_work=set(),
        config_hash=config_hash,
    )

    assert [unit.target_language for unit in units] == ["pt", "pt-BR", "zh-Hant"]
    assert [unit.target_api_language for unit in units] == ["pt", "pt", "zh"]
    assert [unit.work_key for unit in units] == [
        _translation_work_key("mymemory", "gbif:100", "Lime Swallowtail", "en", target_language, config_hash)
        for target_language in target_locales
    ]


def test_mymemory_provider_uses_translation_memory_mode_by_default() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        requests.append((path, params))
        return {
            "responseData": {"translatedText": "Limettenfalter"},
            "matches": [{"translation": "Limettenfalter"}],
        }

    provider = MyMemoryTranslationProvider(http_get=fake_get)

    translations, request_count, response_byte_count = provider.translate(
        source_name="Lime Swallowtail",
        source_language="eng",
        target_language="deu",
        max_candidates=3,
    )

    assert translations == ["Limettenfalter"]
    assert request_count == 1
    assert response_byte_count > 0
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

    translations, request_count, response_byte_count = provider.translate(
        source_name="Lime Swallowtail",
        source_language="eng",
        target_language="deu",
        max_candidates=0,
    )

    assert request_count == 1
    assert response_byte_count > 0
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


def test_wikimedia_provider_extracts_wikispecies_vernacular_names() -> None:
    requests = []

    def fake_get(path, params):  # noqa: ANN001, ANN202 - provider test double.
        requests.append((path, dict(params)))
        return {
            "parse": {
                "pageid": 52929,
                "title": "Papilio demoleus",
                "wikitext": {
                    "*": """
=={{int:Vernacular names}}==
{{VN
|as=নেমুটেঙা পখিলা
|bn=সাত ডোরা, রুরু
|hu=Citruspillangó
|zh = 達摩鳳蝶
|en=Lime Butterfly
|ta=எலுமிச்சை அழகி
|te= నిమ్మ చిలుక
}}

{{Taxonbar|from=Q285314}}
"""
                },
            }
        }

    provider = WikimediaLanglinksProvider(http_get=fake_get)

    links, request_count, page_title = provider.vernacular_names(
        "Papilio demoleus",
        target_locales=("as", "bn", "en", "hu", "ta", "te", "zh"),
    )

    assert request_count == 1
    assert page_title == "Papilio demoleus"
    assert [(link.language, link.title, link.wikidata_item) for link in links] == [
        ("as", "নেমুটেঙা পখিলা", "Q285314"),
        ("bn", "সাত ডোরা", "Q285314"),
        ("bn", "রুরু", "Q285314"),
        ("hu", "Citruspillangó", "Q285314"),
        ("zh", "達摩鳳蝶", "Q285314"),
        ("en", "Lime Butterfly", "Q285314"),
        ("ta", "எலுமிச்சை அழகி", "Q285314"),
        ("te", "నిమ్మ చిలుక", "Q285314"),
    ]
    assert requests[0][0] == "/w/api.php"
    assert requests[0][1]["action"] == "parse"
    assert requests[0][1]["page"] == "Papilio demoleus"
    assert requests[0][1]["prop"] == "wikitext"


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


def test_translation_harvester_preserves_mymemory_bcp47_work_ledger(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["pt", "pt-BR", "zh-Hant"]), encoding="utf-8")
    mymemory = FakeMyMemoryProvider()

    build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": mymemory},
    )

    work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet").sort("target_language")
    candidates = pl.read_parquet(registry / "enrichment" / "translation_candidates.parquet").sort("target_language")

    assert work.select("target_language").to_series().to_list() == ["pt", "pt-BR", "zh-Hant"]
    assert work.select("work_key").to_series().n_unique() == 3
    assert [call["target_language"] for call in mymemory.calls] == ["pt", "pt", "zh"]
    assert candidates.select("target_language").to_series().to_list() == ["pt", "pt-BR", "zh-Hant"]


def test_translation_harvester_checkpoints_mymemory_unit_before_interrupt(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv"]), encoding="utf-8")
    mymemory = InterruptingMyMemoryProvider()

    with pytest.raises(KeyboardInterrupt):
        build_translation_candidates_from_registry(
            registry_dir=registry,
            enrichment_dir=registry / "enrichment",
            translation_sources=("mymemory",),
            target_locales_json=locales,
            providers={"mymemory": mymemory},
            translation_checkpoint_every=1,
        )

    candidates_path = registry / "enrichment" / "translation_candidates.parquet"
    work_path = registry / "enrichment" / "translation_work_ledger.parquet"
    assert candidates_path.exists()
    assert work_path.exists()

    candidates = pl.read_parquet(candidates_path)
    work = pl.read_parquet(work_path)
    assert "Limettenfalter" in candidates.select("translated_name").to_series().to_list()
    assert (
        work.filter((pl.col("target_language") == "de") & (pl.col("status") == "complete"))
        .select("work_key")
        .height
        == 1
    )


def test_translation_manifest_request_rows_remain_cumulative_after_resume(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    enrichment = registry / "enrichment"

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=enrichment,
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": FakeMyMemoryProvider()},
    )
    work = pl.read_parquet(enrichment / "translation_work_ledger.parquet")
    assert manifest["translation_request_rows"] == 1
    assert work.select(pl.col("request_count").sum()).item() == 1

    failing_provider = FailingIfCalledMyMemoryProvider()
    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=enrichment,
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": failing_provider},
    )
    work = pl.read_parquet(enrichment / "translation_work_ledger.parquet")

    assert failing_provider.calls == 0
    assert work.select(pl.col("request_count").sum()).item() == 1
    assert manifest["translation_request_rows"] == 1


def test_translation_checkpoint_every_batches_flushes_until_threshold(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv"]), encoding="utf-8")
    flush_calls: list[dict[str, object]] = []
    original_flush = harvester.TranslationCheckpointWriter.flush

    def counted_flush(self, *, status, force=False):  # noqa: ANN001, ANN202 - monkeypatch wrapper.
        flush_calls.append({"status": status, "force": force})
        return original_flush(self, status=status, force=force)

    monkeypatch.setattr(harvester.TranslationCheckpointWriter, "flush", counted_flush)

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": MultiCandidateMyMemoryProvider()},
        translation_checkpoint_every=2,
    )

    assert (registry / "enrichment" / "translation_candidates.parquet").exists()
    assert manifest["translation_status"] == "complete"
    assert manifest["mymemory_candidate_rows"] == 4
    assert flush_calls == [
        {"status": "running", "force": False},
        {"status": "complete", "force": True},
    ]


def test_translation_source_work_sums_mymemory_unit_request_counts(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de", "sv"]), encoding="utf-8")

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": FakeMyMemoryProvider()},
        translation_checkpoint_every=2,
    )

    translation_work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet")
    source_work = pl.read_parquet(registry / "enrichment" / "source_work_ledger.parquet")
    mymemory_source_work = source_work.filter(pl.col("source") == "mymemory")

    assert translation_work.select(pl.col("request_count").sum()).item() == 2
    assert mymemory_source_work.select("request_count").to_series().to_list() == [2]
    assert manifest["translation_request_rows"] == 2


def test_translation_source_work_counts_mymemory_error_requests(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": RuntimeErrorMyMemoryProvider()},
        translation_checkpoint_every=1,
    )

    translation_work = pl.read_parquet(registry / "enrichment" / "translation_work_ledger.parquet")
    source_work = pl.read_parquet(registry / "enrichment" / "source_work_ledger.parquet")
    mymemory_source_work = source_work.filter(pl.col("source") == "mymemory")

    assert translation_work.select("status").to_series().to_list() == ["error"]
    assert translation_work.select("request_count").to_series().to_list() == [1]
    assert mymemory_source_work.select("request_count").to_series().to_list() == [1]
    assert manifest["translation_request_rows"] == 1


def test_translation_manifest_reports_complete_with_errors_for_provider_failures(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": RuntimeErrorMyMemoryProvider()},
        translation_checkpoint_every=1,
    )

    assert manifest["translation_status"] == "complete_with_errors"
    assert manifest["translation_error_counts_by_source"] == {"mymemory": 1}
    assert manifest["translation_current_run_request_rows"] == 1


def test_translation_manifest_counts_error_then_success_resume_requests(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    enrichment = registry / "enrichment"

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=enrichment,
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": RuntimeErrorMyMemoryProvider()},
        translation_checkpoint_every=1,
    )
    source_work = pl.read_parquet(enrichment / "source_work_ledger.parquet")
    mymemory_source_work = source_work.filter(pl.col("source") == "mymemory")
    assert manifest["translation_request_rows"] == 1
    assert mymemory_source_work.select("request_count").to_series().to_list() == [1]

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=enrichment,
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": FakeMyMemoryProvider()},
        translation_checkpoint_every=1,
    )
    source_work = pl.read_parquet(enrichment / "source_work_ledger.parquet")
    translation_work = pl.read_parquet(enrichment / "translation_work_ledger.parquet")
    candidates = pl.read_parquet(enrichment / "translation_candidates.parquet")
    mymemory_source_work = source_work.filter(pl.col("source") == "mymemory")

    assert manifest["translation_request_rows"] == 2
    assert mymemory_source_work.select("request_count").to_series().to_list() == [2]
    assert translation_work.select("status").to_series().to_list() == ["complete"]
    assert "Limettenfalter" in candidates.select("translated_name").to_series().to_list()


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


def test_translation_harvester_writes_wikispecies_vernacular_assertions(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    _write_wikidata_link(registry, qid="Q285314")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["as", "bn", "en", "hu", "ta", "te", "zh"]), encoding="utf-8")

    class WikispeciesProvider:
        def langlinks(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
            return [], 0, title

        def vernacular_names(self, title, *, target_locales):  # noqa: ANN001, ANN202 - test double.
            assert title == "Papilio demoleus"
            assert target_locales == ("as", "bn", "en", "hu", "ta", "te", "zh")
            return [
                WikimediaLanglink(language="as", title="নেমুটেঙা পখিলা", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="bn", title="সাত ডোরা", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="bn", title="রুরু", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="hu", title="Citruspillangó", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="zh", title="達摩鳳蝶", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="en", title="Lime Butterfly", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="ta", title="எலுமிச்சை அழகி", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
                WikimediaLanglink(language="te", title="నిమ్మ చిలుక", page_id="wikispecies:52929", page_title=title, wikidata_item="Q285314"),
            ], 1, title

    manifest = build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("wikimedia",),
        target_locales_json=locales,
        providers={"wikimedia": WikispeciesProvider()},
    )

    assertions = pl.read_parquet(registry / "enrichment" / "source_name_assertions.parquet").sort(["language", "display_name"])

    assert manifest["wikimedia_assertion_rows"] == 8
    assert assertions.select("display_name").to_series().to_list() == [
        "নেমুটেঙা পখিলা",
        "রুরু",
        "সাত ডোরা",
        "Lime Butterfly",
        "Citruspillangó",
        "எலுமிச்சை அழகி",
        "నిమ్మ చిలుక",
        "達摩鳳蝶",
    ]
    assert assertions.select("source").to_series().to_list() == ["Wikimedia"] * 8
    assert assertions.select("source_taxon_id").to_series().to_list() == ["Q285314"] * 8
    assert assertions.select("enabled").to_series().to_list() == [True] * 8
    assert all("wikispecies:52929" in value for value in assertions.select("source_record_id").to_series().to_list())


def test_translation_harvester_skips_query_ineligible_phrase_fragments(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry(registry)
    names = pl.read_parquet(registry / "names.parquet")
    names = pl.concat(
        [
            names,
            pl.DataFrame(
                [
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "lime",
                        "display_name": "lime",
                        "language": "eng",
                        "script": "Latn",
                        "region": "",
                        "bbox": "",
                        "name_class": "vernacular",
                        "source": "fixture",
                        "source_record_id": "fixture:fragment:lime",
                        "trust_tier": "T2",
                        "precision_tier": "low",
                        "confidence": "medium",
                        "enabled": True,
                        "query_eligible": False,
                        "query_disabled_reason": "generic_single_token",
                        "species_specificity_score": 0.25,
                    }
                ],
                schema=names.schema | {"query_eligible": pl.Boolean, "query_disabled_reason": pl.String, "species_specificity_score": pl.Float64},
            ),
        ],
        how="diagonal_relaxed",
    )
    names.write_parquet(registry / "names.parquet")
    locales = tmp_path / "locales.json"
    locales.write_text(json.dumps(["de"]), encoding="utf-8")
    mymemory = FakeMyMemoryProvider()

    build_translation_candidates_from_registry(
        registry_dir=registry,
        enrichment_dir=registry / "enrichment",
        translation_sources=("mymemory",),
        target_locales_json=locales,
        providers={"mymemory": mymemory},
    )

    assert mymemory.calls == [{"source_name": "Lime Swallowtail", "source_language": "en", "target_language": "de", "max_candidates": "0"}]


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
