import pytest

from biominer.flickr_fetch.query_planner import validate_geotagged_records


@pytest.mark.parametrize("photo", [
    {}, {"latitude": "0", "longitude": "0", "accuracy": "0"},
    {"latitude": "nan", "longitude": "10", "accuracy": "16"},
    {"latitude": "91", "longitude": "10", "accuracy": "16"},
    {"latitude": "10", "longitude": "181", "accuracy": "16"},
    {"latitude": "10", "longitude": "10", "accuracy": "17"},
])
def test_rejects_missing_invalid_or_placeholder_geotags(photo):
    with pytest.raises(ValueError, match="geotag"):
        validate_geotagged_records([photo])


@pytest.mark.parametrize("latitude,longitude", [(0, 20), (20, 0), (0, 0), (-90, 180)])
def test_accepts_valid_geotags_including_zero_axis(latitude, longitude):
    validate_geotagged_records([{"latitude": latitude, "longitude": longitude, "accuracy": 16}])


@pytest.mark.parametrize("transport", ["httpx", "urllib"])
def test_live_request_path_rejects_ungeotagged_results_before_insertion(tmp_path, monkeypatch, transport):
    import httpx
    import io
    import json
    import urllib.request
    from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once, _http_fetcher
    from biominer.flickr_fetch.query_planner import FlickrQuery
    payload = {"stat": "ok", "photos": {"page": 1, "pages": 1, "perpage": 250, "total": "1",
               "photo": [{"id": "1", "url_l": "https://example.test/photo.jpg"}]}}
    def get(self, url, params):
        assert params["has_geo"] == 1
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx.Client, "get", get)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    db = tmp_path / "state.sqlite"
    state = MetadataPollState(db)
    state.enqueue_work_item(FlickrQuery(term="test", language="en", search_field="text",
                                       lane="normal_page", per_page=250, has_geo=1))
    result = poll_once(state_db=db, raw_root=tmp_path / "raw", evidence_output=tmp_path / "evidence.parquet",
                       api_key="test", max_api_calls=1, max_retries=0,
                       fetch_metadata=_http_fetcher(api_key="test", transport=transport))
    assert result.source_records_inserted == 0


def test_urllib_transport_retains_valid_geotags(tmp_path, monkeypatch):
    import io
    import json
    import urllib.request
    from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once, _http_fetcher
    from biominer.flickr_fetch.query_planner import FlickrQuery
    payload = {"stat": "ok", "photos": {"page": 1, "pages": 1, "perpage": 250, "total": "1",
               "photo": [{"id": "1", "url_l": "https://example.test/photo.jpg",
                          "latitude": "0", "longitude": "20", "accuracy": "16"}]}}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    db = tmp_path / "state.sqlite"
    state = MetadataPollState(db)
    state.enqueue_work_item(FlickrQuery(term="test", language="en", search_field="text",
                                       lane="normal_page", per_page=250, has_geo=1))
    result = poll_once(state_db=db, raw_root=tmp_path / "raw", evidence_output=tmp_path / "evidence.parquet",
                       api_key="test", max_api_calls=1,
                       fetch_metadata=_http_fetcher(api_key="test", transport="urllib"))
    assert result.source_records_inserted == 1
    frame = state.canonical_source_records_frame()
    assert frame["latitude"][0] == 0
    assert frame["longitude"][0] == 20
