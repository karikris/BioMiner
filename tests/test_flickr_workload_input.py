from __future__ import annotations

import hashlib
import json

import polars as pl
import pytest

from biominer.flickr_fetch.workload import (
    canonicalize_flickr_workload_hits,
    read_flickr_workload_input,
)


def _hit(
    photo_id: str,
    query_hash: str,
    *,
    fetched_at: str = "2026-07-14T00:00:00Z",
    raw_photo_json: str | None = None,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "fetched_at": fetched_at,
        "query_hash": query_hash,
        "query_field": "tags",
        "query_term": "Papilio demoleus",
        "query_term_confidence": "high",
        "query_term_type": "species_scientific",
        "raw_photo_json": raw_photo_json or f'{{"id":"{photo_id}"}}',
        "latitude": -27.47,
        "longitude": 153.02,
        "accuracy": 16,
    }


def test_canonicalizes_photos_without_discarding_query_hit_provenance() -> None:
    older = _hit("1", "query-b", fetched_at="2026-07-13T00:00:00Z")
    newest = _hit(
        "1",
        "query-a",
        fetched_at="2026-07-14T00:00:00Z",
        raw_photo_json='{"id":"1","version":2}',
    )
    second = _hit("2", "query-c")

    result = canonicalize_flickr_workload_hits(
        pl.DataFrame([second, older, newest])
    )

    assert result.input_row_count == 3
    assert result.canonical_photo_count == 2
    assert result.query_hit_count == 3
    assert result.canonical_photos["flickr_photo_id"].to_list() == ["1", "2"]
    first = result.canonical_photos.row(0, named=True)
    assert first["raw_photo_json"] == newest["raw_photo_json"]
    assert first["source_record_hash"] == "sha256:" + hashlib.sha256(
        str(newest["raw_photo_json"]).encode("utf-8")
    ).hexdigest()
    assert result.query_hits["query_hash"].to_list() == [
        "query-a",
        "query-b",
        "query-c",
    ]
    assert result.query_hits["query_tier"].to_list() == [
        "species_scientific:high:tags"
    ] * 3


def test_rejects_duplicate_query_hits_and_missing_identity() -> None:
    duplicate = _hit("1", "query-a")
    with pytest.raises(ValueError, match="duplicate source/photo/query hits"):
        canonicalize_flickr_workload_hits(pl.DataFrame([duplicate, duplicate]))
    with pytest.raises(ValueError, match="blank flickr_photo_id"):
        canonicalize_flickr_workload_hits(
            pl.DataFrame([{**duplicate, "flickr_photo_id": ""}])
        )
    with pytest.raises(ValueError, match="invalid fetched_at timestamps"):
        canonicalize_flickr_workload_hits(
            pl.DataFrame([{**duplicate, "fetched_at": "not-a-timestamp"}])
        )


def test_reads_ndjson_with_projection_and_default_source(tmp_path) -> None:
    path = tmp_path / "hits.ndjson"
    rows = [_hit("2", "query-b"), _hit("1", "query-a")]
    for row in rows:
        row.pop("source")
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = read_flickr_workload_input(path)

    assert result.canonical_photos["flickr_photo_id"].to_list() == ["1", "2"]
    assert set(result.canonical_photos["source"]) == {"flickr"}
