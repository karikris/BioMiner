from __future__ import annotations

import json

import polars as pl
import biominer.flickr_fetch.query_planner as query_planner

from biominer.flickr_fetch.query_planner import (
    BBOX_PAGE_SIZE,
    COUNT_PROBE_PAGE_SIZE,
    FLICKR_SEARCH_RESULT_WINDOW,
    GEO_PAGE_SIZE,
    MAX_ACCESSIBLE_RESULTS_PER_QUERY,
    NORMAL_PAGE_SIZE,
    STABLE_RESULT_THRESHOLD,
    FlickrQuery,
    build_count_probes,
    build_species_count_probes_from_json,
    build_worldwide_discovery_plan,
    deduplicate_photo_records,
    fixed_upload_date_slices,
    flickr_search_params,
    load_registry_flickr_queries,
    load_species_terms_from_json,
    multilingual_seed_terms,
    outside_known_regions,
    page_size_for_query,
    plan_fixed_upload_slice_pages,
    plan_queries_from_count,
    plan_pages_from_count,
    result_pages_for_total,
    known_region_for_coordinate,
)


PAPILIO_DEMOLEUS_REGION_BBOXES = {
    "India": "68.11,6.55,97.40,35.67",
    "Florida": "-87.64,24.40,-79.97,31.00",
}


def test_species_keyword_helpers_are_generic_exports() -> None:
    assert hasattr(query_planner, "load_species_terms_from_json")
    assert hasattr(query_planner, "build_species_count_probes_from_json")
    assert hasattr(query_planner, "known_region_for_coordinate")
    assert hasattr(query_planner, "outside_known_regions")

    removed_exports = (
        "PAPILIO_DEMOLEUS_ANCHOR_TERMS",
        "PAPILIO_DEMOLEUS_REGION_BBOXES",
        "load_papilio_demoleus_terms_from_json",
        "build_papilio_demoleus_count_probes_from_json",
        "papilio_demoleus_known_region_for_coordinate",
        "outside_known_papilio_demoleus_regions",
    )
    assert [name for name in removed_exports if hasattr(query_planner, name)] == []


def test_multilingual_seed_terms_are_seeded_once_and_include_lifestages() -> None:
    terms = multilingual_seed_terms()
    values = [term.term for term in terms]

    assert len(values) == len({value.casefold() for value in values})
    for expected in ("butterfly", "caterpillar", "chrysalis", "pupa", "egg", "蝴蝶", "oruga", "فراشة", "kupu-kupu", "borboleta", "papillon", "蝶", "бабочка", "Schmetterling"):
        assert expected in values


def test_count_probes_are_recorded_for_text_and_tags() -> None:
    plan = build_worldwide_discovery_plan()

    assert plan.page_queries == ()
    assert plan.count_probes
    assert {probe.search_field for probe in plan.count_probes} == {"text", "tags"}
    assert {probe.per_page for probe in plan.count_probes} == {COUNT_PROBE_PAGE_SIZE}
    assert all(probe.lane == "count_probe" for probe in plan.count_probes)


def test_registry_query_definitions_load_as_page_one_upload_slice_work(tmp_path) -> None:
    registry_queries = tmp_path / "flickr_query_definitions.parquet"
    pl.DataFrame(
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
                "search_field": "tags",
                "search_priority": 10,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
            },
        ]
    ).write_parquet(registry_queries)

    queries = load_registry_flickr_queries(
        registry_queries,
        start_date="2026-01-01",
        end_date="2026-01-05",
    )

    assert [query.search_field for query in queries] == ["tags", "text"]
    assert {query.lane for query in queries} == {"normal_page"}
    assert {query.page for query in queries} == {1}
    assert {query.per_page for query in queries} == {NORMAL_PAGE_SIZE}
    assert {query.has_geo for query in queries} == {0}
    assert queries[0].query_definition_id == "q-tags"
    assert queries[0].registry_version == "registry-v1"
    assert queries[0].accepted_taxon_key == "gbif:100"
    assert queries[0].accepted_scientific_name == "Papilio demoleus"
    assert queries[0].min_upload_date == "2026-01-01"
    assert queries[0].max_upload_date == "2026-01-05"


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


def test_high_volume_queries_use_fixed_upload_slice_pages() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(
        probe,
        total=(MAX_ACCESSIBLE_RESULTS_PER_QUERY + 1) * NORMAL_PAGE_SIZE,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
        upload_date_ranges=[("2024-01-01", "2024-12-31")],
        bboxes=["0,0,10,10"],
        narrower_terms=["swallowtail"],
    )

    assert {item.lane for item in pages} == {"normal_page"}
    assert {item.per_page for item in pages} == {NORMAL_PAGE_SIZE}
    assert not any(item.lane == "count_probe" for item in pages)
    assert pages[0].min_upload_date == "2004-02-10"
    assert pages[0].max_upload_date == "2004-02-14"


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


def test_total_4001_returns_fixed_upload_slice_pages_only() -> None:
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


def test_fixed_slice_metadata_carries_upload_dates_depth_and_index() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(
        probe,
        total=4001,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
    )

    assert [(query.split_reason, query.split_depth, query.slice_index) for query in pages[:2]] == [
        ("upload_date", 1, 0),
        ("upload_date", 1, 1),
    ]
    assert pages[0].min_upload_date == "2004-02-10"
    assert pages[0].max_upload_date == "2004-02-14"


def test_over_threshold_probe_with_upload_bounds_uses_fixed_slices_within_bounds() -> None:
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

    assert len(pages) == 2
    assert pages[0].min_upload_date == "2021-01-01"
    assert pages[0].max_upload_date == "2021-01-05"
    assert pages[-1].max_upload_date == "2021-01-10"


def test_planned_work_is_sorted_deterministically() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0)

    pages = plan_queries_from_count(probe, total=4001)

    assert [(query.slice_index, query.min_upload_date, query.max_upload_date, query.page) for query in pages[:3]] == [
        (0, "2004-02-10", "2004-02-14", 1),
        (1, "2004-02-15", "2004-02-19", 1),
        (2, "2004-02-20", "2004-02-24", 1),
    ]


def test_query_inside_result_window_creates_250_record_geo_pages() -> None:
    probe = FlickrQuery(term="Papilio demoleus", language="la", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(probe, total=501)

    assert [page.page for page in pages] == [1, 2, 3]
    assert [page.per_page for page in pages] == [250, 250, 250]
    assert all(page.page < 4000 for page in pages)


def test_query_over_result_window_uses_fixed_slice_pages_not_count_probes() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(probe, total=MAX_ACCESSIBLE_RESULTS_PER_QUERY * GEO_PAGE_SIZE + 1)

    assert pages
    assert {query.lane for query in pages} == {"normal_page"}
    assert {query.split_reason for query in pages} == {"upload_date"}
    assert not any(query.lane == "count_probe" for query in pages)
    assert all(query.per_page == NORMAL_PAGE_SIZE for query in pages)


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


def test_loads_species_keyword_json_and_gates_broad_terms(tmp_path) -> None:
    path = tmp_path / "keywords.json"
    path.write_text(
        json.dumps(
            {
                "dictionary_groups": {
                    "scientific_taxonomic": [
                        {
                            "term": "Papilio demoleus",
                            "language": "la",
                            "term_type": "scientific_name",
                            "confidence": "high",
                            "use_for_flickr": True,
                            "precision_tier": "high",
                        }
                    ],
                    "english_common_names": [
                        {
                            "term": "lime butterfly",
                            "language": "en",
                            "term_type": "common_name",
                            "confidence": "high",
                            "use_for_flickr": True,
                            "precision_tier": "high",
                        }
                    ],
                    "multilingual_common_name_expansion": [
                        {
                            "term": "kupu-kupu",
                            "language": "id",
                            "term_type": "broad_butterfly",
                            "confidence": "medium",
                            "use_for_flickr": True,
                            "precision_tier": "low",
                        }
                    ],
                    "regional_terms": {
                        "India": [
                            {
                                "term": "Papilio demoleus India",
                                "language": "en",
                                "term_type": "regional_synonym",
                                "confidence": "medium",
                                "regions": ["India"],
                                "use_for_flickr": True,
                                "precision_tier": "high",
                            }
                        ]
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    terms = load_species_terms_from_json(
        path,
        scientific_name="Papilio demoleus",
        region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES,
    )

    assert ("Papilio demoleus", "la", None, "scientific_name", "high") in {
        (term.term, term.language, term.region, term.term_type, term.term_confidence) for term in terms
    }
    assert ("Papilio demoleus kupu-kupu", "id", None, "broad_butterfly", "broad") in {
        (term.term, term.language, term.region, term.term_type, term.term_confidence) for term in terms
    }
    assert any(term.region == "India" and term.bbox and term.term == "Papilio demoleus India" for term in terms)

    probes = build_species_count_probes_from_json(
        path,
        scientific_name="Papilio demoleus",
        region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES,
    )
    assert len(probes) == len(terms) * 2
    assert {probe.search_field for probe in probes} == {"text", "tags"}
    assert all(probe.per_page == COUNT_PROBE_PAGE_SIZE for probe in probes)
    assert any(probe.region == "India" and probe.bbox for probe in probes)


def test_global_high_confidence_terms_are_not_forced_into_known_region_bboxes(tmp_path) -> None:
    path = tmp_path / "keywords.json"
    path.write_text(
        json.dumps(
            {
                "dictionary_groups": {
                    "scientific_taxonomic": [
                        {
                            "term": "Papilio demoleus",
                            "language": "la",
                            "term_type": "scientific_name",
                            "confidence": "high",
                            "regions": [],
                            "use_for_flickr": True,
                            "precision_tier": "high",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    probes = build_species_count_probes_from_json(path, scientific_name="Papilio demoleus")

    assert probes
    assert all(probe.term == "Papilio demoleus" for probe in probes)
    assert all(probe.bbox is None for probe in probes)
    assert all(probe.region is None for probe in probes)


def test_outside_known_species_regions_can_be_flagged_for_discovery_review() -> None:
    assert known_region_for_coordinate(latitude=27.95, longitude=-82.46, region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES) == "Florida"
    assert outside_known_regions({"latitude": "27.95", "longitude": "-82.46"}, region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES) is False
    assert outside_known_regions({"latitude": "60.17", "longitude": "24.94"}, region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES) is True
    assert outside_known_regions({"latitude": "", "longitude": ""}, region_bboxes=PAPILIO_DEMOLEUS_REGION_BBOXES) is None
