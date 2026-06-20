from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import polars as pl

from biominer.cli import build_parser
from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_production import RetryingHTTPGet
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import ButterflyScope


class FakeGBIFHTTP:
    def __init__(self, responses: dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object], int]] = []
        self.lock = threading.Lock()

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        with self.lock:
            self.calls.append((path, params, threading.get_ident()))
        return self.responses[(path, tuple(sorted(params.items())))]


def test_registry_build_cli_exposes_production_concurrency_defaults() -> None:
    args = build_parser().parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/test",
            "--registry-version",
            "test",
        ]
    )

    assert args.workers == 8
    assert args.progress_every == 100
    assert args.checkpoint_every == 500
    assert args.max_retries == 5


def test_retrying_http_get_retries_503_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    getter = RetryingHTTPGet(max_retries=2, sleep=sleeps.append, transport=httpx.MockTransport(handler))
    try:
        assert getter("/species/1", {}) == {"ok": True}
    finally:
        getter.close()

    assert calls == 2
    assert getter.retry_count == 1
    assert getter.attempt_count == 2
    assert sleeps == [0.5]


def test_retrying_http_get_applies_jitter_to_exponential_backoff() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    getter = RetryingHTTPGet(
        max_retries=2,
        sleep=sleeps.append,
        jitter=lambda attempt: 0.25 * (attempt + 1),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert getter("/species/1", {}) == {"ok": True}
    finally:
        getter.close()

    assert sleeps == [0.75]


def test_retrying_http_get_does_not_retry_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=request)

    getter = RetryingHTTPGet(max_retries=5, sleep=lambda _: None, transport=httpx.MockTransport(handler))
    try:
        try:
            getter("/species/missing", {})
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("Expected HTTPStatusError")
    finally:
        getter.close()

    assert calls == 1
    assert getter.retry_count == 0


def test_gbif_source_snapshot_checkpoints_and_resumes_species_enrichment(tmp_path) -> None:
    scope = ButterflyScope(
        scope_id="test-scope",
        root_scientific_name="Papilionoidea",
        root_rank="SUPERFAMILY",
        included_families=("Papilionidae",),
    )
    main_http = FakeGBIFHTTP(_family_source_responses())
    worker_http = FakeGBIFHTTP(_species_enrichment_responses())
    worker_clients: list[GBIFClient] = []

    def worker_factory() -> GBIFClient:
        client = GBIFClient(http_get=worker_http)
        worker_clients.append(client)
        return client

    first = build_gbif_source_snapshot(
        GBIFClient(http_get=main_http),
        scope,
        retrieved_at="2026-06-20T00:00:00+00:00",
        checkpoint_dir=tmp_path / "checkpoints",
        workers=2,
        progress_every=1,
        checkpoint_every=1,
        client_factory=worker_factory,
    )

    checkpoint_state = json.loads((tmp_path / "checkpoints" / "Papilionidae" / "state.json").read_text(encoding="utf-8"))
    checkpoint_names = pl.read_parquet(tmp_path / "checkpoints" / "Papilionidae" / "enrichment_names.parquet")
    assert checkpoint_state["status"] == "complete"
    assert checkpoint_state["completed_species_keys"] == ["100", "101"]
    assert checkpoint_names.select("display_name").to_series().to_list() == [
        "Papilio erithonius",
        "Lime Butterfly",
        "Papilio xuthus synonym",
        "Asian Swallowtail",
    ]
    assert first["metrics"]["workers"] == 2
    assert first["metrics"]["checkpoint_every"] == 1
    assert len(worker_clients) <= 2
    assert {call[2] for call in worker_http.calls}

    worker_http.calls.clear()
    second = build_gbif_source_snapshot(
        GBIFClient(http_get=FakeGBIFHTTP(_family_source_responses())),
        scope,
        retrieved_at="2026-06-20T00:00:00+00:00",
        checkpoint_dir=tmp_path / "checkpoints",
        workers=2,
        progress_every=1,
        checkpoint_every=1,
        client_factory=worker_factory,
    )

    assert worker_http.calls == []
    assert second["metrics"]["resumed_species"] == 2
    assert [row["display_name"] for row in second["names"] if row["name_class"] != "accepted_scientific"] == [
        "Papilio erithonius",
        "Lime Butterfly",
        "Papilio xuthus synonym",
        "Asian Swallowtail",
    ]


def test_gbif_species_enrichment_uses_bounded_ordered_map(monkeypatch, tmp_path) -> None:
    import biominer.registry.gbif_source as gbif_source

    calls: list[int] = []
    original = gbif_source.bounded_map_ordered

    def spy(executor, function, items, *, buffersize):  # noqa: ANN001 - mirrors helper signature.
        calls.append(buffersize)
        yield from original(executor, function, items, buffersize=buffersize)

    monkeypatch.setattr(gbif_source, "bounded_map_ordered", spy)
    build_gbif_source_snapshot(
        GBIFClient(http_get=FakeGBIFHTTP(_family_source_responses())),
        ButterflyScope(
            scope_id="test-scope",
            root_scientific_name="Papilionoidea",
            root_rank="SUPERFAMILY",
            included_families=("Papilionidae",),
        ),
        retrieved_at="2026-06-20T00:00:00+00:00",
        checkpoint_dir=tmp_path / "checkpoints",
        workers=2,
        progress_every=1,
        checkpoint_every=1,
        client_factory=lambda: GBIFClient(http_get=FakeGBIFHTTP(_species_enrichment_responses())),
    )

    assert calls == [4]


def _family_source_responses() -> dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, Any]]:
    return {
        ("/species/match", (("name", "Papilionoidea"), ("rank", "SUPERFAMILY"), ("strict", "false"))): {"usageKey": 1},
        ("/species/1", ()): {"key": 1, "scientificName": "Papilionoidea", "rank": "SUPERFAMILY", "parents": []},
        ("/species/match", (("name", "Papilionidae"), ("rank", "FAMILY"), ("strict", "false"))): {
            "usageKey": 10,
            "rank": "FAMILY",
            "matchType": "EXACT",
            "confidence": 99,
        },
        ("/species/10", ()): {
            "key": 10,
            "scientificName": "Papilionidae",
            "rank": "FAMILY",
            "taxonomicStatus": "ACCEPTED",
            "parents": [{"scientificName": "Papilionoidea"}],
        },
        ("/species/10/children", (("limit", 1000), ("rank", "GENUS"))): {
            "results": [{"key": 90, "scientificName": "Papilio", "rank": "GENUS"}],
        },
        ("/species/90/children", (("limit", 1000), ("rank", "SPECIES"))): {
            "results": [
                {"key": 100, "scientificName": "Papilio demoleus", "canonicalName": "Papilio demoleus", "rank": "SPECIES"},
                {"key": 101, "scientificName": "Papilio xuthus", "canonicalName": "Papilio xuthus", "rank": "SPECIES"},
            ],
        },
    }


def _species_enrichment_responses() -> dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, Any]]:
    return {
        ("/species/100/synonyms", (("limit", 1000),)): {
            "results": [{"key": 200, "scientificName": "Papilio erithonius", "canonicalName": "Papilio erithonius"}],
        },
        ("/species/100/vernacularNames", (("limit", 1000),)): {
            "results": [{"vernacularName": "Lime Butterfly", "language": "eng"}],
        },
        ("/species/101/synonyms", (("limit", 1000),)): {
            "results": [{"key": 201, "scientificName": "Papilio xuthus synonym", "canonicalName": "Papilio xuthus synonym"}],
        },
        ("/species/101/vernacularNames", (("limit", 1000),)): {
            "results": [{"vernacularName": "Asian Swallowtail", "language": "eng"}],
        },
    }
