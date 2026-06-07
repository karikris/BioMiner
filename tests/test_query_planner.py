from __future__ import annotations

import json

from flickr_bio_occurrence.flickr.query_planner import (
    BBOX_PAGE_SIZE,
    COUNT_PROBE_PAGE_SIZE,
    MAX_RESULT_PAGES_PER_QUERY,
    NORMAL_PAGE_SIZE,
    FlickrQuery,
    build_papilio_demoleus_count_probes_from_json,
    build_count_probes,
    build_worldwide_discovery_plan,
    deduplicate_photo_records,
    flickr_search_params,
    load_papilio_demoleus_terms_from_json,
    multilingual_seed_terms,
    outside_known_papilio_demoleus_regions,
    papilio_demoleus_known_region_for_coordinate,
    plan_queries_from_count,
    plan_pages_from_count,
)


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


def test_normal_and_bbox_pages_use_500_records() -> None:
    normal_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")
    bbox_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", bbox="0,0,10,10")

    normal_pages = plan_pages_from_count(normal_probe, total=501)
    bbox_pages = plan_pages_from_count(bbox_probe, total=501)

    assert [page.per_page for page in normal_pages] == [NORMAL_PAGE_SIZE, NORMAL_PAGE_SIZE]
    assert [page.lane for page in normal_pages] == ["normal_page", "normal_page"]
    assert [page.per_page for page in bbox_pages] == [BBOX_PAGE_SIZE, BBOX_PAGE_SIZE]
    assert [page.lane for page in bbox_pages] == ["bbox_page", "bbox_page"]


def test_high_volume_queries_split_before_pages() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    split = plan_queries_from_count(
        probe,
        total=(MAX_RESULT_PAGES_PER_QUERY + 1) * NORMAL_PAGE_SIZE,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
        upload_date_ranges=[("2024-01-01", "2024-12-31")],
        bboxes=["0,0,10,10"],
        narrower_terms=["swallowtail"],
    )

    assert [item.split_reason for item in split] == ["taken_date", "taken_date"]
    assert [item.per_page for item in split] == [COUNT_PROBE_PAGE_SIZE, COUNT_PROBE_PAGE_SIZE]
    assert split[0].min_taken_date == "2024-01-01"
    assert split[0].parent_total == (MAX_RESULT_PAGES_PER_QUERY + 1) * NORMAL_PAGE_SIZE


def test_query_under_page_limit_creates_500_record_pages() -> None:
    probe = FlickrQuery(term="Papilio demoleus", language="la", search_field="text", lane="count_probe")

    pages = plan_queries_from_count(probe, total=501)

    assert [page.page for page in pages] == [1, 2]
    assert [page.per_page for page in pages] == [500, 500]
    assert all(page.page < 4000 for page in pages)


def test_query_over_page_limit_splits_to_count_probes_not_oversized_pages() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    split = plan_queries_from_count(probe, total=MAX_RESULT_PAGES_PER_QUERY * NORMAL_PAGE_SIZE + 1)

    assert split
    assert {query.lane for query in split} == {"count_probe"}
    assert {query.split_reason for query in split} == {"bbox"}
    assert all(query.page == 1 for query in split)
    assert all(query.per_page == COUNT_PROBE_PAGE_SIZE for query in split)


def test_flickr_search_params_use_text_or_tags_and_url_l_url_m_only() -> None:
    query = FlickrQuery(term="mariposa", language="es", search_field="tags", lane="count_probe", bbox="0,0,10,10")

    params = flickr_search_params(query)

    assert params["tags"] == "mariposa"
    assert "text" not in params
    assert params["has_geo"] == 1
    assert params["bbox"] == "0,0,10,10"
    assert params["per_page"] == 1
    assert params["extras"] == "url_l,url_m"
    assert "url_o" not in str(params["extras"])


def test_deduplicates_by_photo_id_and_image_url() -> None:
    unique = deduplicate_photo_records(
        [
            {"id": "1", "url_l": "https://live.staticflickr.com/1_l.jpg"},
            {"id": "1", "url_l": "https://live.staticflickr.com/1_l.jpg"},
            {"id": "1", "url_l": "https://live.staticflickr.com/1_other.jpg"},
            {"id": "2", "url_m": "https://live.staticflickr.com/2_m.jpg"},
        ]
    )

    assert [row["id"] for row in unique] == ["1", "1", "2"]


def test_loads_papilio_demoleus_keyword_json_and_gates_broad_terms(tmp_path) -> None:
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

    terms = load_papilio_demoleus_terms_from_json(path)

    assert ("Papilio demoleus", "la", None, "scientific_name", "high") in {
        (term.term, term.language, term.region, term.term_type, term.term_confidence) for term in terms
    }
    assert ("Papilio demoleus kupu-kupu", "id", None, "broad_butterfly", "broad") in {
        (term.term, term.language, term.region, term.term_type, term.term_confidence) for term in terms
    }
    assert any(term.region == "India" and term.bbox and term.term == "Papilio demoleus India" for term in terms)

    probes = build_papilio_demoleus_count_probes_from_json(path)
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

    probes = build_papilio_demoleus_count_probes_from_json(path)

    assert probes
    assert all(probe.term == "Papilio demoleus" for probe in probes)
    assert all(probe.bbox is None for probe in probes)
    assert all(probe.region is None for probe in probes)


def test_outside_known_papilio_demoleus_regions_can_be_flagged_for_discovery_review() -> None:
    assert papilio_demoleus_known_region_for_coordinate(27.95, -82.46) == "Florida"
    assert outside_known_papilio_demoleus_regions({"latitude": "27.95", "longitude": "-82.46"}) is False
    assert outside_known_papilio_demoleus_regions({"latitude": "60.17", "longitude": "24.94"}) is True
    assert outside_known_papilio_demoleus_regions({"latitude": "", "longitude": ""}) is None
