"""Offline behavioural controls for the explicit local HTTP budget ledger."""

import json
import logging
import sqlite3

import httpx
import pytest

from biominer.registry.image_keyword_http import (
    BudgetDeferred,
    KeywordHTTP,
    KeywordHTTPError,
    MAX_BODY_BYTES,
    MYMEMORY_REQUEST_BYTES,
)


class Clock:
    def __init__(self):
        self.now = 1_792_627_200.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_http(tmp_path, handler, **kwargs):
    clock = kwargs.pop("clock", Clock())
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return KeywordHTTP(tmp_path / "ledger.sqlite", client=client, clock=clock,
                       sleep=clock.sleep, **kwargs), clock


def test_retry_after_reserves_each_physical_request_before_transport(tmp_path):
    seen = []

    def handler(request):
        observer = sqlite3.connect(tmp_path / "ledger.sqlite")
        seen.append(observer.execute("SELECT COUNT(*) FROM request_attempts").fetchone()[0])
        observer.close()
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"names": ["butterfly"]})

    http, clock = make_http(tmp_path, handler)
    with http.source("gbif"):
        assert http.get("https://api.gbif.org/v1")("species", {"name": "Pieris napi"}) == {"names": ["butterfly"]}
    assert seen == [1, 2]
    assert http.request_count == 2
    assert clock.sleeps == [7.0]
    assert http.db.execute("SELECT status FROM request_attempts ORDER BY id").fetchall() == [("429",), ("200",)]
    http.close()


def test_network_errors_and_retry_exhaustion_have_safe_messages(tmp_path):
    events = []

    def handler(request):
        raise httpx.ConnectError("private-token should never appear", request=request)

    http, _ = make_http(tmp_path, handler, max_retries=2, logger=events.append)
    with http.source("gbif"), pytest.raises(KeywordHTTPError, match="^network_retry_limit$"):
        http.get("https://api.gbif.org")("species", {"name": "private-query"})
    assert http.request_count == 3
    assert "private" not in json.dumps(events)
    assert http.db.execute("SELECT COUNT(*) FROM request_attempts WHERE status='network_error'").fetchone()[0] == 3
    http.close()


def test_invocation_limit_applies_to_retries(tmp_path):
    http, _ = make_http(tmp_path, lambda request: httpx.Response(503), max_requests=1)
    with http.source("gbif"), pytest.raises(BudgetDeferred, match="invocation_request_limit"):
        http.get("https://api.gbif.org")("species", {})
    assert http.request_count == 1
    http.close()


def test_persistent_inaturalist_limit_survives_reopen(tmp_path):
    handler = lambda request: httpx.Response(200, json={"results": []})
    http, clock = make_http(tmp_path, handler)
    period = __import__("datetime").datetime.fromtimestamp(clock(), __import__("datetime").timezone.utc).strftime("%Y-%m-%d")
    http.db.execute("INSERT INTO quota_usage VALUES ('inaturalist',?,9999,0,0)", (period,))
    with http.source("inaturalist"):
        http.get("https://api.inaturalist.org/v1")("taxa", {"q": "Pieris"})
    http.close()
    reopened, _ = make_http(tmp_path, handler, clock=clock)
    with reopened.source("inaturalist"), pytest.raises(BudgetDeferred, match="persistent_inaturalist_limit"):
        reopened.get("https://api.inaturalist.org/v1")("taxa", {"q": "Papilio"})
    assert reopened.request_count == 0
    reopened.close()


def test_cache_reuses_across_instances_and_excludes_credentials(tmp_path):
    calls, events = [], []

    def handler(request):
        calls.append(request)
        assert request.url.params["mt"] == "0"
        return httpx.Response(200, json={"responseData": {"translatedText": "butterfly"}})

    http, clock = make_http(tmp_path, handler, logger=events.append)
    with http.source("mymemory"):
        first = http.get("https://api.mymemory.translated.net")("get", {
            "q": "Pieris napi", "langpair": "la|en", "key": "secret-one", "de": "private@example.org",
        })
    assert len(http.job_receipts) == 1
    http.reset_job_receipts()
    assert http.job_receipts == []
    metadata = http.db.execute("SELECT source,host,path,request_sha256,status FROM request_attempts").fetchall()
    assert all(word not in json.dumps(metadata) + json.dumps(events) for word in ("secret-one", "private@example.org", "Pieris napi"))
    usage = http.db.execute("SELECT requests,words,bandwidth FROM quota_usage WHERE bucket='mymemory'").fetchone()
    assert usage == (1, 2, MYMEMORY_REQUEST_BYTES)
    http.close()
    http, _ = make_http(tmp_path, handler, clock=clock)
    with http.source("mymemory"):
        second = http.get("https://api.mymemory.translated.net")("get", {
            "q": "Pieris napi", "langpair": "la|en", "key": "secret-two", "de": "other@example.org",
        })
    assert first == second
    assert len(calls) == 1
    assert http.request_count == 0
    assert http.job_receipts[0]["cache_hit"] is True
    http.close()


@pytest.mark.parametrize("initial,parameter,expected", [
    ((0, 9999, 0), {"q": "Pieris napi"}, "persistent_mymemory_limit"),
    ((10000, 0, 0), {"q": "Pieris"}, "persistent_mymemory_limit"),
    ((0, 0, 10 * 1024**3), {"q": "Pieris"}, "persistent_mymemory_limit"),
])
def test_mymemory_monthly_budgets_are_reserved_atomically(tmp_path, initial, parameter, expected):
    http, clock = make_http(tmp_path, lambda request: pytest.fail("budget should prevent HTTP"))
    from datetime import datetime, timezone
    month = datetime.fromtimestamp(clock(), timezone.utc).strftime("%Y-%m")
    http.db.execute("INSERT INTO quota_usage VALUES ('mymemory',?,?,?,?)", (month, *initial))
    with http.source("mymemory"), pytest.raises(BudgetDeferred, match=expected):
        http.get("https://api.mymemory.translated.net")("get", parameter)
    assert http.db.execute("SELECT COUNT(*) FROM request_attempts").fetchone()[0] == 0
    assert http.db.execute("SELECT COUNT(*) FROM quota_usage").fetchone()[0] == 1
    http.close()


def test_wikimedia_and_mymemory_share_daily_budget(tmp_path):
    http, clock = make_http(tmp_path, lambda request: httpx.Response(200, json={}))
    from datetime import datetime, timezone
    day = datetime.fromtimestamp(clock(), timezone.utc).strftime("%Y-%m-%d")
    http.db.execute("INSERT INTO quota_usage VALUES ('wikimedia_mymemory',?,9999,0,0)", (day,))
    with http.source("commons"):
        http.get("https://commons.wikimedia.org")("w/api.php", {"action": "query"})
    with http.source("mymemory"), pytest.raises(BudgetDeferred, match="persistent_wikimedia_mymemory_limit"):
        http.get("https://api.mymemory.translated.net")("get", {"q": "Pieris"})
    http.close()


@pytest.mark.parametrize("params", [{"mt": 1}, {"mt": "1"}])
def test_machine_translation_is_rejected_before_any_reservation(tmp_path, params):
    http, _ = make_http(tmp_path, lambda request: pytest.fail("must not request"))
    with http.source("mymemory"), pytest.raises(ValueError, match="machine translation"):
        http.get("https://api.mymemory.translated.net")("get", params)
    assert http.request_count == 0
    http.close()


@pytest.mark.parametrize("status,body,error", [
    (404, b"private server body", "http_404"),
    (200, b"<html>not JSON</html>", "non_json_response"),
    (200, b'{"error":{"info":"private server info"}}', "provider_error_response"),
    (200, b'{"errors":[{"message":"private details"}]}', "provider_error_response"),
    (200, b'"scalar"', "unexpected_json_shape"),
    (200, b"x" * (MAX_BODY_BYTES + 1), "response_body_limit"),
])
def test_permanent_http_provider_and_body_failures_do_not_retry(tmp_path, status, body, error):
    http, _ = make_http(tmp_path, lambda request: httpx.Response(status, content=body))
    with http.source("commons"), pytest.raises(KeywordHTTPError, match="^" + error + "$"):
        http.get("https://commons.wikimedia.org")("w/api.php", {})
    assert http.request_count == 1
    assert http.db.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 0
    http.close()


def test_host_and_source_rate_limits_survive_reopen(tmp_path):
    handler = lambda request: httpx.Response(200, json={})
    http, clock = make_http(tmp_path, handler, min_interval=2)
    with http.source("gbif"):
        http.get("https://api.gbif.org")("species", {"q": "first"})
    http.close()
    http, _ = make_http(tmp_path, handler, clock=clock, min_interval=2)
    with http.source("gbif"):
        http.get("https://api.gbif.org")("species", {"q": "second"})
    assert clock.sleeps == [2]
    http.close()


def test_post_graphql_and_nested_source_restore(tmp_path):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"names": []}})

    http, _ = make_http(tmp_path, handler)
    with http.source("outer"):
        with http.source("inner"):
            http.graphql("https://example.org/graphql")("query Names($id: ID!) { names(id:$id) }", {"id": "1"})
        http.post("https://example.org/api")("names", {"name": "Pieris"})
    assert [receipt["source"] for receipt in http.job_receipts] == ["inner", "outer"]
    assert seen[0]["variables"] == {"id": "1"}
    with pytest.raises(ValueError, match="source context"):
        http.get("https://example.org")("names", {})
    http.close()


def test_httpx_info_urls_cannot_leak_credentials(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="httpx")
    http, _ = make_http(tmp_path, lambda request: httpx.Response(200, json={}))
    with http.source("mymemory"):
        http.get("https://api.mymemory.translated.net")("get", {"key": "private-token", "de": "private@example.org", "q": "Pieris napi"})
    assert not any("private-token" in record.getMessage() or "private@example.org" in record.getMessage()
                   for record in caplog.records)
    http.close()


def test_retry_after_persists_when_sleep_is_interrupted(tmp_path):
    clock = Clock()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={"Retry-After": "30"})))

    def interrupted_sleep(seconds):
        raise InterruptedError("simulated local process interruption")

    http = KeywordHTTP(tmp_path / "ledger.sqlite", client=client, clock=clock, sleep=interrupted_sleep)
    with http.source("commons"), pytest.raises(InterruptedError):
        http.get("https://commons.wikimedia.org")("w/api.php", {})
    assert http.request_count == 1
    http.close()
    http, _ = make_http(tmp_path, lambda request: httpx.Response(200, json={}), clock=clock)
    with http.source("commons"):
        http.get("https://commons.wikimedia.org")("w/api.php", {})
    assert clock.sleeps == [30]
    http.close()


@pytest.mark.parametrize("semantic_key", ["checklistKey", "taxonKey", "speciesKey"])
def test_public_taxonomy_keys_remain_in_semantic_fingerprint(tmp_path, semantic_key):
    seen = []

    def handler(request):
        seen.append(request.url.params[semantic_key])
        return httpx.Response(200, json={"key": request.url.params[semantic_key]})

    http, _ = make_http(tmp_path, handler)
    with http.source("gbif"):
        first = http.get("https://api.gbif.org/v2")("species/match", {semantic_key: "one", "api_key": "private-one"})
        second = http.get("https://api.gbif.org/v2")("species/match", {semantic_key: "two", "api_key": "private-one"})
        cached = http.get("https://api.gbif.org/v2")("species/match", {semantic_key: "two", "api_key": "private-two"})
    assert first == {"key": "one"}
    assert second == cached == {"key": "two"}
    assert seen == ["one", "two"]
    assert http.request_count == 2
    assert http.job_receipts[0]["request_sha256"] != http.job_receipts[1]["request_sha256"]
    assert http.job_receipts[1]["request_sha256"] == http.job_receipts[2]["request_sha256"]
    metadata = http.db.execute("SELECT source,host,path,request_sha256,status FROM request_attempts").fetchall()
    assert "private-" not in json.dumps(metadata)
    http.close()


def test_cached_repeated_cursor_is_bounded_and_next_job_resets_limit(tmp_path):
    http, _ = make_http(tmp_path, lambda request: httpx.Response(200, json={"cursor": "unchanged"}), max_job_calls=3)
    with http.source("checklistbank"):
        getter = http.get("https://api.checklistbank.org")
        for _ in range(3):
            assert getter("names", {"cursor": "unchanged"}) == {"cursor": "unchanged"}
        with pytest.raises(KeywordHTTPError, match="^job_call_limit$"):
            getter("names", {"cursor": "unchanged"})
        assert http.request_count == 1
        assert http.job_call_count == 3
        assert len(http.job_receipts) == 3
        http.reset_job_receipts()
        assert http.job_call_count == 0
        assert http.job_receipts == []
        assert getter("names", {"cursor": "unchanged"}) == {"cursor": "unchanged"}
        assert http.job_call_count == 1
        assert http.request_count == 1
    http.close()
