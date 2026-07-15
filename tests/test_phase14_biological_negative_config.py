from __future__ import annotations

import json
from pathlib import Path


LEDGER_PATH = Path("config/pilot/papilio_demoleus_biological_negative_taxa.json")
QUERY_PATH = Path(
    "config/pilot/papilio_demoleus_biological_negative_source_queries.json"
)
BASE_QUERY_PATH = Path("config/pilot/papilio_demoleus_reference_source_queries.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_biological_negative_ledger_covers_required_candidate_categories() -> None:
    ledger = _read(LEDGER_PATH)
    categories = {row["candidate_category"] for row in ledger["taxa"]}

    assert ledger["prototype_only"] is True
    assert ledger["candidate_semantics"] == (
        "metadata_source_candidate_not_verified_image_label"
    )
    assert categories == {
        "visual_neighbour_candidate",
        "moth_negative",
        "other_lepidoptera_negative",
        "other_insect_negative",
    }
    assert set(ledger["existing_candidate_sources"]) == {
        "regional_papilionidae",
        "same_genus_competitors",
        "false_winning_genera",
        "historical_false_winner_species",
    }


def test_every_new_taxon_has_exact_accepted_gbif_reconciliation() -> None:
    rows = _read(LEDGER_PATH)["taxa"]

    assert len(rows) == 11
    assert len({row["accepted_taxon_key"] for row in rows}) == len(rows)
    assert len({row["scientific_name"] for row in rows}) == len(rows)
    assert all(row["gbif_match_type"] == "EXACT" for row in rows)
    assert all(row["gbif_taxonomic_status"] == "ACCEPTED" for row in rows)
    assert all(row["gbif_match_confidence"] >= 99 for row in rows)


def test_papilionoidea_negatives_are_cross_linked_to_current_registry() -> None:
    rows = _read(LEDGER_PATH)["taxa"]
    scoped = [
        row
        for row in rows
        if row["candidate_category"]
        in {"visual_neighbour_candidate", "other_lepidoptera_negative"}
    ]

    assert len(scoped) == 5
    assert all(str(row["current_registry_key"]).startswith("col:") for row in scoped)
    assert all(row["current_registry_key"] is None for row in rows if row not in scoped)


def test_visual_neighbours_remain_candidates_until_embedding_confirmation() -> None:
    ledger = _read(LEDGER_PATH)
    policy = ledger["visual_neighbour_policy"]
    visual_rows = [
        row
        for row in ledger["taxa"]
        if row["candidate_category"] == "visual_neighbour_candidate"
    ]

    assert {row["scientific_name"] for row in visual_rows} == {
        "Papilio demodocus",
        "Papilio erithonioides",
    }
    assert policy["may_claim_visually_nearest"] is False
    assert policy["status"] == (
        "morphology_candidates_pending_frozen_embedding_confirmation"
    )
    assert policy["evidence_doi"] == "10.1017/S1477200008002703"


def test_negative_query_plan_exactly_covers_the_reconciled_ledger() -> None:
    ledger = _read(LEDGER_PATH)
    query_plan = _read(QUERY_PATH)
    queries = query_plan["queries"]

    assert query_plan["candidate_semantics"] == ledger["candidate_semantics"]
    assert {(row["accepted_taxon_key"], row["scientific_name"]) for row in queries} == {
        (row["accepted_taxon_key"], row["scientific_name"]) for row in ledger["taxa"]
    }
    assert all(row["source"] == "GBIF" for row in queries)
    assert all(row["fallback_level"] == 3 for row in queries)
    assert all(row["geo_cluster_id"] == "unassigned_geo" for row in queries)


def test_biological_negative_quota_is_at_least_one_hundred() -> None:
    quotas = _read(QUERY_PATH)["acquisition_quotas"]

    assert (
        sum(
            quotas[group]["minimum_total"]
            for group in (
                "moth_negatives",
                "other_lepidoptera_negatives",
                "other_insect_negatives",
            )
        )
        >= 100
    )
    assert quotas["visual_neighbour_candidates"]["status"] == (
        "pending_frozen_embedding_confirmation"
    )


def test_existing_papilionidae_and_false_winner_queries_remain_separate() -> None:
    base = _read(BASE_QUERY_PATH)["acquisition_quotas"]
    negative = _read(QUERY_PATH)

    assert {
        "selected_regional_competitors",
        "reviewed_false_winner_genera",
        "historical_false_winner_species",
        "broader_papilionidae",
    } <= set(base)
    assert set(negative["existing_candidate_groups"]) == {
        "regional_papilionidae",
        "same_genus_competitors",
        "false_winning_genera",
        "historical_false_winner_species",
    }
