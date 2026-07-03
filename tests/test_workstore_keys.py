from __future__ import annotations

from biominer.workstore.keys import scoped_work_item_key, stable_work_key, uri_shard_id


def test_scoped_work_item_key_preserves_workstore_wire_format() -> None:
    payload = {"term": "butterfly", "page": 1}

    key = scoped_work_item_key("poll_once", "metadata", "registry-v1", payload)

    assert key == "poll_once:5df5ffbdc36fbad4300c05d0"


def test_existing_key_helpers_keep_expected_prefix_and_hash_shape() -> None:
    assert stable_work_key({"x": 1}, prefix="job").startswith("job:")
    assert len(uri_shard_id("s3://bucket/path.parquet")) == 64
