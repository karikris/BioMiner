from __future__ import annotations

import pytest

from biominer.registry.gbif import GBIFClient, resolve_family


class FakeGBIFHTTP:
    def __init__(self, responses: dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, params))
        key = (path, tuple(sorted(params.items())))
        return self.responses[key]


def test_gbif_client_keeps_match_usage_children_synonyms_and_vernacular_endpoints_separate() -> None:
    http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Papilionidae"), ("rank", "FAMILY"), ("strict", "false"))): {"usageKey": 10},
            ("/species/10", ()): {"key": 10, "rank": "FAMILY"},
            ("/species/10/children", (("limit", 1000), ("rank", "GENUS"))): {"results": [{"key": 90}]},
            ("/species/100/synonyms", (("limit", 1000),)): {"results": [{"key": 101}]},
            ("/species/100/vernacularNames", (("limit", 1000),)): {"results": [{"vernacularName": "Lime Butterfly"}]},
        }
    )
    client = GBIFClient(http_get=http)

    assert client.match_name("Papilionidae", rank="FAMILY")["usageKey"] == 10
    assert client.usage(10)["rank"] == "FAMILY"
    assert client.children(10, rank="GENUS") == [{"key": 90}]
    assert client.synonyms(100) == [{"key": 101}]
    assert client.vernacular_names(100) == [{"vernacularName": "Lime Butterfly"}]
    assert [call[0] for call in http.calls] == [
        "/species/match",
        "/species/10",
        "/species/10/children",
        "/species/100/synonyms",
        "/species/100/vernacularNames",
    ]


def test_gbif_client_paginates_list_endpoints_until_end_of_records() -> None:
    http = FakeGBIFHTTP(
        {
            ("/species/10/children", (("limit", 2), ("rank", "GENUS"))): {
                "offset": 0,
                "limit": 2,
                "count": 3,
                "endOfRecords": False,
                "results": [{"key": 90}, {"key": 91}],
            },
            ("/species/10/children", (("limit", 2), ("offset", 2), ("rank", "GENUS"))): {
                "offset": 2,
                "limit": 2,
                "count": 3,
                "endOfRecords": True,
                "results": [{"key": 92}],
            },
            ("/species/100/synonyms", (("limit", 1),)): {
                "offset": 0,
                "limit": 1,
                "count": 2,
                "endOfRecords": False,
                "results": [{"key": 101}],
            },
            ("/species/100/synonyms", (("limit", 1), ("offset", 1))): {
                "offset": 1,
                "limit": 1,
                "count": 2,
                "endOfRecords": True,
                "results": [{"key": 102}],
            },
            ("/species/100/vernacularNames", (("limit", 1),)): {
                "offset": 0,
                "limit": 1,
                "count": 2,
                "endOfRecords": False,
                "results": [{"vernacularName": "Lime Butterfly"}],
            },
            ("/species/100/vernacularNames", (("limit", 1), ("offset", 1))): {
                "offset": 1,
                "limit": 1,
                "count": 2,
                "endOfRecords": True,
                "results": [{"vernacularName": "Chequered Swallowtail"}],
            },
        }
    )
    client = GBIFClient(http_get=http)

    assert client.children(10, rank="GENUS", limit=2) == [{"key": 90}, {"key": 91}, {"key": 92}]
    assert client.synonyms(100, limit=1) == [{"key": 101}, {"key": 102}]
    assert client.vernacular_names(100, limit=1) == [
        {"vernacularName": "Lime Butterfly"},
        {"vernacularName": "Chequered Swallowtail"},
    ]


def test_resolve_family_accepts_family_with_papilionoidea_lineage() -> None:
    http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Papilionidae"), ("rank", "FAMILY"), ("strict", "false"))): {
                "usageKey": 10,
                "rank": "FAMILY",
                "status": "ACCEPTED",
                "matchType": "EXACT",
                "confidence": 99,
            },
            ("/species/10", ()): {
                "key": 10,
                "scientificName": "Papilionidae",
                "rank": "FAMILY",
                "taxonomicStatus": "ACCEPTED",
                "parents": [
                    {"key": 1, "scientificName": "Papilionoidea", "rank": "SUPERFAMILY"},
                    {"key": 2, "scientificName": "Lepidoptera", "rank": "ORDER"},
                ],
            },
        }
    )

    resolution = resolve_family(GBIFClient(http_get=http), "Papilionidae", root_name="Papilionoidea")

    assert resolution.accepted_taxon_key == "gbif:10"
    assert resolution.scientific_name == "Papilionidae"
    assert resolution.match_type == "EXACT"
    assert resolution.confidence == 99
    assert resolution.lineage_names == ("Papilionoidea", "Lepidoptera")


def test_resolve_family_follows_accepted_usage_for_synonym_match() -> None:
    http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Old Papilionidae"), ("rank", "FAMILY"), ("strict", "false"))): {
                "usageKey": 9,
                "acceptedUsageKey": 10,
                "rank": "FAMILY",
                "status": "SYNONYM",
                "matchType": "FUZZY",
                "confidence": 82,
            },
            ("/species/10", ()): {
                "key": 10,
                "scientificName": "Papilionidae",
                "rank": "FAMILY",
                "taxonomicStatus": "ACCEPTED",
                "parents": [{"key": 1, "scientificName": "Papilionoidea", "rank": "SUPERFAMILY"}],
            },
        }
    )

    resolution = resolve_family(GBIFClient(http_get=http), "Old Papilionidae", root_name="Papilionoidea")

    assert resolution.accepted_taxon_key == "gbif:10"
    assert resolution.matched_usage_key == "gbif:9"
    assert resolution.match_type == "FUZZY"


def test_resolve_family_rejects_wrong_rank_or_missing_root_lineage() -> None:
    wrong_rank_http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Papilio"), ("rank", "FAMILY"), ("strict", "false"))): {
                "usageKey": 90,
                "rank": "GENUS",
                "status": "ACCEPTED",
                "matchType": "EXACT",
                "confidence": 98,
            },
            ("/species/90", ()): {
                "key": 90,
                "scientificName": "Papilio",
                "rank": "GENUS",
                "taxonomicStatus": "ACCEPTED",
                "parents": [{"key": 1, "scientificName": "Papilionoidea", "rank": "SUPERFAMILY"}],
            },
        }
    )

    with pytest.raises(ValueError, match="rank FAMILY"):
        resolve_family(GBIFClient(http_get=wrong_rank_http), "Papilio", root_name="Papilionoidea")

    wrong_root_http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Noctuidae"), ("rank", "FAMILY"), ("strict", "false"))): {
                "usageKey": 30,
                "rank": "FAMILY",
                "status": "ACCEPTED",
                "matchType": "EXACT",
                "confidence": 99,
            },
            ("/species/30", ()): {
                "key": 30,
                "scientificName": "Noctuidae",
                "rank": "FAMILY",
                "taxonomicStatus": "ACCEPTED",
                "parents": [{"key": 2, "scientificName": "Noctuoidea", "rank": "SUPERFAMILY"}],
            },
        }
    )

    with pytest.raises(ValueError, match="Papilionoidea"):
        resolve_family(GBIFClient(http_get=wrong_root_http), "Noctuidae", root_name="Papilionoidea")
