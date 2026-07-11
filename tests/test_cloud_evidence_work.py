from __future__ import annotations

import polars as pl

from biominer.evidence.cloud_work import _read_shards


class _ScanOnlyStorage:
    def __init__(self, payloads: dict[str, pl.DataFrame]) -> None:
        self.payloads = payloads
        self.scan_calls: list[str] = []

    def scan_parquet(self, uri: str) -> pl.LazyFrame:
        self.scan_calls.append(uri)
        return self.payloads[uri].lazy()

    def read_parquet(self, uri: str) -> pl.DataFrame:
        raise AssertionError(f"evidence shard was eagerly read: {uri}")


def test_cloud_evidence_shards_are_scanned_in_inventory_order() -> None:
    storage = _ScanOnlyStorage(
        {
            "s3://biominer/part-1.parquet": pl.DataFrame({"row_id": [1], "score": [0.7]}),
            "s3://biominer/part-2.parquet": pl.DataFrame({"row_id": [2], "label": ["butterfly"]}),
        }
    )
    shards = [
        {"uri": "s3://biominer/part-1.parquet"},
        {"uri": "s3://biominer/part-2.parquet"},
    ]

    frame = _read_shards(storage, shards)

    assert storage.scan_calls == ["s3://biominer/part-1.parquet", "s3://biominer/part-2.parquet"]
    assert frame.to_dicts() == [
        {"row_id": 1, "score": 0.7, "label": None},
        {"row_id": 2, "score": None, "label": "butterfly"},
    ]
