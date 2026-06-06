from __future__ import annotations

from flickr_bio_occurrence.flickr.query_planner import (
    BBOX_PAGE_SIZE,
    COUNT_PROBE_PAGE_SIZE,
    NORMAL_PAGE_SIZE,
    FlickrQuery,
    build_count_probes,
    build_worldwide_discovery_plan,
    deduplicate_photo_records,
    flickr_search_params,
    multilingual_seed_terms,
    plan_pages_from_count,
    split_high_volume_query,
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


def test_normal_pages_use_500_and_bbox_pages_use_250() -> None:
    normal_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")
    bbox_probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", bbox="0,0,10,10")

    normal_pages = plan_pages_from_count(normal_probe, total=501)
    bbox_pages = plan_pages_from_count(bbox_probe, total=251)

    assert [page.per_page for page in normal_pages] == [NORMAL_PAGE_SIZE, NORMAL_PAGE_SIZE]
    assert [page.lane for page in normal_pages] == ["normal_page", "normal_page"]
    assert [page.per_page for page in bbox_pages] == [BBOX_PAGE_SIZE, BBOX_PAGE_SIZE]
    assert [page.lane for page in bbox_pages] == ["bbox_page", "bbox_page"]


def test_high_volume_queries_split_before_pages() -> None:
    probe = FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe")

    split = split_high_volume_query(
        probe,
        total=3501,
        taken_date_ranges=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")],
        upload_date_ranges=[("2024-01-01", "2024-12-31")],
        bboxes=["0,0,10,10"],
        narrower_terms=["swallowtail"],
    )

    assert [item.split_reason for item in split] == ["taken_date", "taken_date"]
    assert [item.per_page for item in split] == [COUNT_PROBE_PAGE_SIZE, COUNT_PROBE_PAGE_SIZE]
    assert split[0].min_taken_date == "2024-01-01"
    assert split[0].parent_total == 3501


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
