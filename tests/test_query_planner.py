from __future__ import annotations

from pathlib import Path

import polars as pl
import biominer.flickr_fetch.query_planner as query_planner

from biominer.flickr_fetch.query_planner import (
    BBOX_PAGE_SIZE,
    FLICKR_SEARCH_RESULT_WINDOW,
    GEO_PAGE_SIZE,
    MAX_ACCESSIBLE_RESULTS_PER_QUERY,
    NORMAL_PAGE_SIZE,
    STABLE_RESULT_THRESHOLD,
    FlickrQuery,
    deduplicate_photo_records,
    fixed_upload_date_slices,
    flickr_search_params,
    load_registry_flickr_queries,
    load_registry_flickr_queries_from_frame,
    page_size_for_query,
    plan_fixed_upload_slice_pages,
    plan_queries_from_count,
    plan_pages_from_count,
    result_pages_for_total,
)


def test_query_planner_source_has_no_papilio_specific_production_hardcoding() -> None:
    source = Path(query_planner.__file__).read_text(encoding="utf-8")
    forbidden = (
        "Papilio demoleus",
        "PAPILIO_DEMOLEUS_ANCHOR_TERMS",
        "PAPILIO_DEMOLEUS_REGION_BBOXES",
        "load_papilio_demoleus_terms_from_json",
        "build_papilio_demoleus_count_probes_from_json",
        "papilio_demoleus_known_region_for_coordinate",
        "outside_known_papilio_demoleus_regions",
        "known_region_for_coordinate",
        "outside_known_regions",
    )

    assert [value for value in forbidden if value in source] == []


def test_registry_query_definitions_load_as_single_unsliced_page_one_work(tmp_path) -> None:
    registry_queries = tmp_path / "flickr_query_definitions.parquet"
    frame = pl.DataFrame(
        [
            {
                "query_definition_id": "q-text",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "api_language_code": "la",
                "bcp47": "la",
                "search_field": "text",
                "search_priority": 50,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
            },
            {
                "query_definition_id": "q-tags",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "api_language_code": "la",
                "bcp47": "la",
                "search_field": "tags",
                "search_priority": 10,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
            },
        ]
    )
    frame.write_parquet(registry_queries)

    queries = load_registry_flickr_queries(
        registry_queries,
        start_date="2026-01-01",
        end_date="2026-01-05",
    )
    frame_queries = load_registry_flickr_queries_from_frame(
        frame,
        start_date="2026-01-01",
        end_date="2026-01-05",
    )

    assert frame_queries == queries
    assert [query.search_field for query in queries] == ["tags", "text"]
    assert {query.lane for query in queries} == {"normal_page"}
    assert {query.page for query in queries} == {1}
    assert {query.per_page for query in queries} == {NORMAL_PAGE_SIZE}
    assert {query.has_geo for query in queries} == {0}
    assert queries[0].query_definition_id == "q-tags"
    assert queries[0].query_priority == 10
    assert queries[0].registry_version == "registry-v1"
    assert queries[0].accepted_taxon_key == "gbif:100"
    assert queries[0].accepted_scientific_name == "Papilio demoleus"
    assert queries[0].api_language_code == "la"
    assert queries[0].bcp47 == "la"
    assert queries[0].min_upload_date is None
    assert queries[0].max_upload_date is None


def test_query_hash_ignores_language_provenance_that_does_not_change_flickr_request() -> None:
    base = FlickrQuery(
        term="Borboleta lima",
        language="por",
        search_field="tags",
        lane="normal_page",
        query_definition_id="q-pt-br",
    )
    with_language_provenance = FlickrQuery(
        term="Borboleta lima",
        language="por",
        api_language_code="pt",
        bcp47="pt-BR",
        search_field="tags",
        lane="normal_page",
        query_definition_id="q-pt-br",
    )

    assert query_planner.query_hash(with_language_provenance) == query_planner.query_hash(base)


def test_explicitly_query_eligible_t5_definitions_become_flickr_api_search_params(tmp_path) -> None:
    registry_queries = tmp_path / "flickr_query_definitions.parquet"
    frame = pl.DataFrame(
        [
            {
                "query_definition_id": "q-t5-tags",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Translated Lime",
                "language": "en",
                "search_field": "tags",
                "search_priority": 5,
                "bbox": "",
                "region": "",
                "name_class": "generated_translation",
                "confidence": "low",
                "trust_tier": "T5",
                "enabled": True,
                "query_eligible": True,
            },
            {
                "query_definition_id": "q-t5-text",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Translated Lime",
                "language": "en",
                "search_field": "text",
                "search_priority": 5,
                "bbox": "",
                "region": "",
                "name_class": "generated_translation",
                "confidence": "low",
                "trust_tier": "T5",
                "enabled": True,
                "query_eligible": True,
            },
        ]
    )
    frame.write_parquet(registry_queries)

    queries = load_registry_flickr_queries(
        registry_queries,
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    by_field = {query.search_field: query for query in queries}

    assert set(by_field) == {"tags", "text"}
    assert by_field["tags"].trust_tier == "T5"
    assert by_field["tags"].term_type == "generated_translation"
    assert by_field["tags"].query_definition_id == "q-t5-tags"
    assert flickr_search_params(by_field["tags"])["tags"] == "Translated Lime"
    assert flickr_search_params(by_field["text"])["text"] == "Translated Lime"
    assert "min_upload_date" not in flickr_search_params(by_field["tags"])
    assert "max_upload_date" not in flickr_search_params(by_field["tags"])


def test_registry_query_loader_skips_generated_definitions_without_query_eligible_field(tmp_path) -> None:
    registry_queries = tmp_path / "flickr_query_definitions.parquet"
    frame = pl.DataFrame(
        [
            {
                "query_definition_id": "q-scientific",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "search_field": "tags",
                "search_priority": 10,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "trust_tier": "T1",
                "enabled": True,
            },
            {
                "query_definition_id": "q-generated",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Translated Lime",
                "language": "en",
                "search_field": "text",
                "search_priority": 40,
                "bbox": "",
                "region": "",
                "name_class": "generated_translation",
                "confidence": "low",
                "trust_tier": "T5",
                "enabled": True,
            },
        ]
    )
    frame.write_parquet(registry_queries)

    queries = load_registry_flickr_queries(registry_queries)

    assert [query.query_definition_id for query in queries] == ["q-scientific"]
    assert [query.term for query in queries] == ["Papilio demoleus"]


def test_registry_query_loader_skips_query_ineligible_definitions(tmp_path) -> None:
    registry_queries = tmp_path / "flickr_query_definitions.parquet"
    frame = pl.DataFrame(
        [
            {
                "query_definition_id": "q-good",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "search_field": "tags",
                "search_priority": 10,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
                "query_eligible": True,
            },
            {
                "query_definition_id": "q-weak",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "lime",
                "language": "en",
                "search_field": "text",
                "search_priority": 40,
                "bbox": "",
                "region": "",
                "name_class": "vernacular_alias",
                "confidence": "low",
                "enabled": True,
                "query_eligible": False,
                "query_disabled_reason": "generic_single_token",
            },
        ]
    )
    frame.write_parquet(registry_queries)

    queries = load_registry_flickr_queries(registry_queries)

    assert [query.query_definition_id for query in queries] == ["q-good"]
    assert [query.term for query in queries] == ["Papilio demoleus"]


def test_geo_pages_use_250_and_non_geo_pages_use_500_records() -> None:
    geo_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")
    non_geo_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)
    bbox_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", bbox="0,0,10,10")

    geo_pages = plan_pages_from_count(geo_probe, total=501)
    non_geo_pages = plan_pages_from_count(non_geo_probe, total=501)
    bbox_pages = plan_pages_from_count(bbox_probe, total=501)

    assert page_size_for_query(geo_probe) == GEO_PAGE_SIZE
    assert page_size_for_query(non_geo_probe) == NORMAL_PAGE_SIZE
    assert [page.per_page for page in geo_pages] == [GEO_PAGE_SIZE, GEO_PAGE_SIZE, GEO_PAGE_SIZE]
    assert [page.lane for page in geo_pages] == ["normal_page", "normal_page", "normal_page"]
    assert [page.per_page for page in non_geo_pages] == [NORMAL_PAGE_SIZE, NORMAL_PAGE_SIZE]
    assert [page.has_geo for page in non_geo_pages] == [0, 0]
    assert [page.per_page for page in bbox_pages] == [BBOX_PAGE_SIZE, BBOX_PAGE_SIZE, BBOX_PAGE_SIZE]
    assert [page.lane for page in bbox_pages] == ["bbox_page", "bbox_page", "bbox_page"]


def test_high_volume_queries_are_capped_to_first_accessible_result_window_without_date_slices() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(
        probe,
        total=(MAX_ACCESSIBLE_RESULTS_PER_QUERY + 1) * NORMAL_PAGE_SIZE,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
        upload_date_ranges=[("2024-01-01", "2024-12-31")],
        bboxes=["0,0,10,10"],
        narrower_terms=["swallowtail"],
    )

    assert len(pages) == 8
    assert {item.lane for item in pages} == {"normal_page"}
    assert {item.per_page for item in pages} == {NORMAL_PAGE_SIZE}
    assert not any(item.lane == "count_probe" for item in pages)
    assert not any(page.min_upload_date or page.max_upload_date for page in pages)
    assert [page.page for page in pages] == list(range(1, 9))


def test_stable_threshold_matches_flickr_result_window() -> None:
    assert STABLE_RESULT_THRESHOLD == 4000
    assert FLICKR_SEARCH_RESULT_WINDOW == 4000


def test_total_4000_creates_standard_pages_for_non_geo_leaf() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(probe, total=4000)

    expected_pages = result_pages_for_total(4000, per_page=NORMAL_PAGE_SIZE)
    assert [page.lane for page in pages] == ["normal_page"] * expected_pages
    assert [page.page for page in pages] == list(range(1, expected_pages + 1))
    assert {page.per_page for page in pages} == {NORMAL_PAGE_SIZE}


def test_text_butterfly_total_3300_creates_seven_standard_pages() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(probe, total=3300)

    assert [page.lane for page in pages] == ["normal_page"] * 7
    expected_pages = result_pages_for_total(3300, per_page=NORMAL_PAGE_SIZE)
    assert [page.page for page in pages] == list(range(1, expected_pages + 1))
    assert {page.per_page for page in pages} == {NORMAL_PAGE_SIZE}


def test_total_4000_creates_bbox_pages_for_geo_leaf() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", bbox="0,0,10,10")

    pages = plan_queries_from_count(probe, total=4000)

    assert [page.lane for page in pages] == ["bbox_page"] * 16
    assert [page.page for page in pages] == list(range(1, 17))
    assert {page.per_page for page in pages} == {BBOX_PAGE_SIZE}


def test_total_4001_returns_first_accessible_pages_only() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(
        probe,
        total=4001,
        upload_date_ranges=[("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31")],
        bboxes=["0,0,10,10"],
    )

    assert pages
    assert {query.lane for query in pages} == {"normal_page"}
    assert not any(query.lane == "count_probe" for query in pages)
    assert {query.per_page for query in pages} == {NORMAL_PAGE_SIZE}
    assert [query.page for query in pages] == list(range(1, 9))
    assert not any(query.min_upload_date or query.max_upload_date for query in pages)


def test_over_threshold_probe_does_not_carry_upload_dates_depth_or_slice_index() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(
        probe,
        total=4001,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
    )

    assert [(query.split_reason, query.split_depth, query.slice_index) for query in pages[:2]] == [
        (None, 0, None),
        (None, 0, None),
    ]
    assert pages[0].min_upload_date is None
    assert pages[0].max_upload_date is None


def test_over_threshold_probe_with_upload_bounds_discards_bounds_for_unsliced_fetch() -> None:
    probe = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="count_probe",
        has_geo=0,
        min_upload_date="2021-01-01",
        max_upload_date="2021-01-10",
    )

    pages = plan_queries_from_count(probe, total=4001)

    assert len(pages) == 8
    assert pages[0].min_upload_date is None
    assert pages[0].max_upload_date is None
    assert pages[-1].min_upload_date is None
    assert pages[-1].max_upload_date is None


def test_planned_work_is_sorted_deterministically() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(probe, total=4001)

    assert [(query.slice_index, query.min_upload_date, query.max_upload_date, query.page) for query in pages[:3]] == [
        (None, None, None, 1),
        (None, None, None, 2),
        (None, None, None, 3),
    ]


def test_query_inside_result_window_creates_250_record_geo_pages() -> None:
    probe = FlickrQuery(term="Papilio demoleus", language="la", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(probe, total=501)

    assert [page.page for page in pages] == [1, 2, 3]
    assert [page.per_page for page in pages] == [250, 250, 250]
    assert all(page.page < 4000 for page in pages)


def test_query_over_result_window_uses_accessible_pages_not_date_slices_or_count_probes() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(probe, total=MAX_ACCESSIBLE_RESULTS_PER_QUERY * GEO_PAGE_SIZE + 1)

    assert pages
    assert {query.lane for query in pages} == {"normal_page"}
    assert {query.split_reason for query in pages} == {None}
    assert not any(query.lane == "count_probe" for query in pages)
    assert all(query.per_page == GEO_PAGE_SIZE for query in pages)
    assert len(pages) == 16


def test_flickr_search_params_use_text_or_tags_and_url_l_url_m_only() -> None:
    query = FlickrQuery(term="mariposa", language="es", search_field="tags", lane="count_probe", bbox="0,0,10,10")

    params = flickr_search_params(query)

    assert params["tags"] == "mariposa"
    assert "text" not in params
    assert params["has_geo"] == 1
    assert params["bbox"] == "0,0,10,10"
    assert params["per_page"] == 1
    assert "geo" in str(params["extras"])
    assert "date_taken" in str(params["extras"])
    assert "url_l" in str(params["extras"])
    assert "url_m" in str(params["extras"])
    assert "url_o" not in str(params["extras"])


def test_fixed_upload_date_slices_create_five_day_periods() -> None:
    slices = fixed_upload_date_slices(start_date="2007-01-01", end_date="2007-12-31", slice_days=5)

    assert len(slices) == 73
    assert slices[0] == ("2007-01-01", "2007-01-05")
    assert slices[-1] == ("2007-12-27", "2007-12-31")


def test_fixed_upload_date_slices_cover_leap_year_and_full_range() -> None:
    leap = fixed_upload_date_slices(start_date="2008-01-01", end_date="2008-12-31", slice_days=5)
    full = fixed_upload_date_slices(start_date="2007-01-01", end_date="2026-12-31", slice_days=5)

    assert len(leap) == 74
    assert leap[-1] == ("2008-12-31", "2008-12-31")
    assert len(full) == 1461
    assert full[0] == ("2007-01-01", "2007-01-05")
    assert full[-1] == ("2026-12-27", "2026-12-31")


def test_fixed_upload_date_slices_support_coarse_then_fine_periods() -> None:
    slices = fixed_upload_date_slices(
        start_date="2004-02-10",
        end_date="2016-01-10",
        slice_days=5,
        coarse_end_date="2015-12-31",
        coarse_slice_days=10,
    )

    assert slices[0] == ("2004-02-10", "2004-02-19")
    assert ("2016-01-01", "2016-01-05") in slices
    assert slices[-1] == ("2016-01-06", "2016-01-10")
    assert all(start <= end for start, end in slices)


def test_plan_fixed_upload_slice_pages_creates_page_one_only_without_count_probes() -> None:
    pages = plan_fixed_upload_slice_pages(
        term="butterfly",
        search_field="text",
        start_date="2007-01-01",
        end_date="2007-01-10",
        slice_days=5,
        coarse_end_date=None,
        coarse_slice_days=None,
    )

    assert len(pages) == 2
    assert {page.lane for page in pages} == {"normal_page"}
    assert {page.per_page for page in pages} == {NORMAL_PAGE_SIZE}
    assert {page.page for page in pages} == {1}
    assert not any(page.lane == "count_probe" for page in pages)
    assert [(page.slice_index, page.min_upload_date, page.max_upload_date, page.page) for page in pages] == [
        (0, "2007-01-01", "2007-01-05", 1),
        (1, "2007-01-06", "2007-01-10", 1),
    ]


def test_deduplicates_by_photo_id() -> None:
    unique = deduplicate_photo_records(
        [
            {"id": "1", "url_l": "https://live.staticflickr.com/1_l.jpg"},
            {"id": "1", "url_l": "https://live.staticflickr.com/1_l.jpg"},
            {"id": "1", "url_l": "https://live.staticflickr.com/1_other.jpg"},
            {"id": "2", "url_m": "https://live.staticflickr.com/2_m.jpg"},
        ]
    )

    assert [row["id"] for row in unique] == ["1", "2"]
