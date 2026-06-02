from __future__ import annotations

from flickr_bio_occurrence.flickr.work_items import WorkItem, build_monthly_work_items
from flickr_bio_occurrence.taxonomy.species_mapper import get_seed_species


def test_work_item_ids_are_deterministic() -> None:
    item = WorkItem(
        species_name="Papilio demoleus",
        species_query_terms=["Papilio demoleus", "lime butterfly"],
        region_id="AU_QLD",
        region_name="Queensland",
        bbox="137.99,-29.18,153.55,-9.14",
        year=2024,
        month=1,
        min_taken_date="2024-01-01",
        max_taken_date="2024-01-31",
        page=1,
        query_variant="scientific_name",
    )

    assert item.work_item_id == "afd9a315d2b70db011dcb70fc6dc809f12819817e936ca5cd075993a1acc7ea7"


def test_partitioning_by_species_region_year_month() -> None:
    seed = get_seed_species("Papilio demoleus")
    items = build_monthly_work_items(
        species=seed,
        regions=[("AU_QLD", "Queensland", "137.99,-29.18,153.55,-9.14")],
        years=[2024],
        months=[1, 2],
    )

    assert len(items) == 8
    assert {item.year for item in items} == {2024}
    assert {item.month for item in items} == {1, 2}
    assert {item.region_id for item in items} == {"AU_QLD"}
    assert {item.species_name for item in items} == {"Papilio demoleus"}


def test_papilio_demoleus_seed_terms_present() -> None:
    seed = get_seed_species("Papilio demoleus")

    assert "Papilio demoleus" in seed.search_terms
    assert "lime butterfly" in seed.search_terms
    assert "chequered swallowtail" in seed.search_terms
    assert "citrus swallowtail" in seed.search_terms
    assert "swallowtail" in seed.search_terms
