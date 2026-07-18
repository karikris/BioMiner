from __future__ import annotations

import sqlite3

import polars as pl

from biominer.flickr_fetch import australia


def test_australia_presence_retries_429_and_preserves_completed_rows(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        {
            "accepted_taxon_key": ["col:complete", "col:retry"],
            "scientific_name": ["Complete species", "Retry species"],
            "rank": ["SPECIES", "SPECIES"],
            "taxonomic_status": ["ACCEPTED", "ACCEPTED"],
        }
    ).write_parquet(registry / "taxa.parquet")
    state = tmp_path / "presence.sqlite"
    with sqlite3.connect(state) as conn:
        conn.execute(
            """CREATE TABLE gbif_australia_presence (
            accepted_taxon_key TEXT PRIMARY KEY, scientific_name TEXT NOT NULL,
            occurrence_count INTEGER, status TEXT NOT NULL, error TEXT, retrieved_at TEXT)"""
        )
        conn.execute(
            "INSERT INTO gbif_australia_presence VALUES (?, ?, ?, ?, ?, ?)",
            ("col:complete", "Complete species", 7, "complete", None, "2026-01-01T00:00:00+00:00"),
        )
    calls = 0

    def fake_count(name: str) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise australia.GBIFRequestFailure("rate limited", http_status=429, retry_after_seconds=0)
        assert name == "Retry species"
        return 3

    monkeypatch.setattr(australia, "_gbif_australia_count", fake_count)
    result = australia.build_australia_presence(
        registry_dir=registry,
        state_db=state,
        output_path=tmp_path / "presence.parquet",
        request_interval_seconds=0,
        sleep=lambda _: None,
    )

    assert calls == 2
    rows = {row["accepted_taxon_key"]: row for row in result.to_dicts()}
    assert rows["col:complete"]["gbif_au_occurrence_count"] == 7
    assert rows["col:complete"]["status"] == "complete"
    assert rows["col:retry"]["gbif_au_occurrence_count"] == 3
    assert rows["col:retry"]["status"] == "complete"
    assert rows["col:retry"]["attempt_count"] == 1


def test_record_failure_marks_permanent_client_error_terminal(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "state.sqlite")
    connection.execute(
        """CREATE TABLE gbif_australia_presence (
        accepted_taxon_key TEXT PRIMARY KEY, scientific_name TEXT NOT NULL,
        occurrence_count INTEGER, status TEXT NOT NULL, error TEXT, retrieved_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
        last_http_status INTEGER, last_error_class TEXT)"""
    )
    connection.execute("INSERT INTO gbif_australia_presence(accepted_taxon_key, scientific_name, status) VALUES ('col:x', 'X', 'claimed')")
    australia._record_gbif_failure(
        connection,
        key="col:x",
        attempts=1,
        error=australia.GBIFRequestFailure("bad request", http_status=400),
        max_attempts=8,
    )
    row = connection.execute("SELECT status, last_error_class, last_http_status FROM gbif_australia_presence").fetchone()
    assert row == ("failed", "terminal", 400)
