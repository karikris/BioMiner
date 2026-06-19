from __future__ import annotations

from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import ButterflyScope


class FakeGBIFHTTP:
    def __init__(self, responses: dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, object]]) -> None:
        self.responses = responses

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        return self.responses[(path, tuple(sorted(params.items())))]


def test_build_gbif_source_snapshot_resolves_scope_and_traverses_species_names() -> None:
    scope = ButterflyScope(
        scope_id="test-scope",
        root_scientific_name="Papilionoidea",
        root_rank="SUPERFAMILY",
        included_families=("Papilionidae",),
    )
    http = FakeGBIFHTTP(
        {
            ("/species/match", (("name", "Papilionoidea"), ("rank", "SUPERFAMILY"), ("strict", "false"))): {
                "usageKey": 1,
                "rank": "SUPERFAMILY",
                "status": "ACCEPTED",
                "matchType": "EXACT",
                "confidence": 99,
            },
            ("/species/1", ()): {
                "key": 1,
                "scientificName": "Papilionoidea",
                "rank": "SUPERFAMILY",
                "taxonomicStatus": "ACCEPTED",
                "parents": [],
            },
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
                "parents": [{"key": 1, "scientificName": "Papilionoidea", "rank": "SUPERFAMILY"}],
            },
            ("/species/10/children", (("limit", 1000), ("rank", "GENUS"))): {
                "results": [
                    {
                        "key": 90,
                        "scientificName": "Papilio",
                        "rank": "GENUS",
                        "parentKey": 10,
                    }
                ]
            },
            ("/species/90/children", (("limit", 1000), ("rank", "SPECIES"))): {
                "results": [
                    {
                        "key": 100,
                        "scientificName": "Papilio demoleus",
                        "canonicalName": "Papilio demoleus",
                        "rank": "SPECIES",
                        "parentKey": 90,
                    }
                ]
            },
            ("/species/100/synonyms", (("limit", 1000),)): {
                "results": [
                    {
                        "key": 101,
                        "scientificName": "Papilio erithonius",
                        "canonicalName": "Papilio erithonius",
                    }
                ]
            },
            ("/species/100/vernacularNames", (("limit", 1000),)): {
                "results": [
                    {
                        "vernacularName": "Lime Butterfly",
                        "language": "eng",
                    }
                ]
            },
        }
    )

    snapshot = build_gbif_source_snapshot(
        GBIFClient(http_get=http),
        scope,
        retrieved_at="2026-06-20T00:00:00+00:00",
    )

    assert snapshot["source"] == "GBIF"
    assert snapshot["source_version"] == "gbif-species-api"
    assert [row["scientific_name"] for row in snapshot["taxa"]] == [
        "Papilionoidea",
        "Papilionidae",
        "Papilio",
        "Papilio demoleus",
    ]
    assert {(row["rank"], row["accepted_taxon_key"]) for row in snapshot["taxa"]} == {
        ("SUPERFAMILY", "gbif:1"),
        ("FAMILY", "gbif:10"),
        ("GENUS", "gbif:90"),
        ("SPECIES", "gbif:100"),
    }
    assert [(row["display_name"], row["name_class"], row["language"], row["trust_tier"]) for row in snapshot["names"]] == [
        ("Papilionoidea", "accepted_scientific", "la", "T1"),
        ("Papilionidae", "accepted_scientific", "la", "T1"),
        ("Papilio", "accepted_scientific", "la", "T1"),
        ("Papilio demoleus", "accepted_scientific", "la", "T1"),
        ("Papilio erithonius", "scientific_synonym", "la", "T1"),
        ("Lime Butterfly", "vernacular", "eng", "T2"),
    ]
    assert snapshot["source_assertions"][0]["configured_name"] == "Papilionidae"
    assert snapshot["source_assertions"][0]["match_type"] == "EXACT"
    assert snapshot["source_assertions"][0]["confidence"] == 99
