"""Public artifact contracts shared by the Polars/Arrow qualification matrix."""
from datetime import UTC, datetime

import polars as pl
import pytest

from biominer.storage.parquet import (
    ParquetRowSource,
    iter_parquet_batches,
    write_parquet,
    write_parquet_batches,
)


def test_typed_batches_preserve_nullable_lists_timestamps_and_source_order(tmp_path):
    schema = {
        "source_id": pl.String,
        "observed_at": pl.Datetime("us", "UTC"),
        "query_ids": pl.List(pl.String),
        "measurements": pl.List(pl.Float64),
        "optional": pl.String,
    }
    timestamp = datetime(2026, 10, 1, 12, 30, 0, 123456, tzinfo=UTC)
    first = pl.DataFrame(
        {"measurements": [[1, 2]], "query_ids": [["z", None, "z"]],
         "observed_at": [timestamp], "source_id": ["02"]},
    )
    second = pl.DataFrame(
        {"source_id": ["01", "01"], "observed_at": [None, timestamp],
         "query_ids": [[], None], "measurements": [None, []]},
        schema={key: value for key, value in schema.items() if key != "optional"},
    )
    path = tmp_path / "typed.parquet"

    write_parquet_batches([first, pl.DataFrame(schema=schema), second], path, schema=schema)

    restored = pl.concat(iter_parquet_batches(path, batch_size=1))
    assert restored.schema == schema
    expected = [
        {"source_id": "02", "observed_at": timestamp, "query_ids": ["z", None, "z"],
         "measurements": [1.0, 2.0], "optional": None},
        {"source_id": "01", "observed_at": None, "query_ids": [],
         "measurements": None, "optional": None},
        {"source_id": "01", "observed_at": timestamp, "query_ids": None,
         "measurements": [], "optional": None},
    ]
    assert restored.to_dicts() == expected
    assert list(ParquetRowSource(path, batch_size=2)) == expected


@pytest.mark.parametrize("batches", [[], [pl.DataFrame(schema={"id": pl.String})]])
def test_empty_batch_artifact_retains_explicit_schema(tmp_path, batches):
    schema = {"id": pl.String, "query_ids": pl.List(pl.String),
              "observed_at": pl.Datetime("us", "UTC")}
    path = tmp_path / "empty.parquet"

    write_parquet_batches(batches, path, schema=schema)

    restored = pl.read_parquet(path)
    assert restored.schema == schema
    assert restored.height == 0
    assert list(ParquetRowSource(path)) == []


@pytest.mark.parametrize("invalid", ["not-an-integer", "1.5", "9223372036854775808"])
def test_failed_schema_conversion_preserves_previous_artifact(tmp_path, invalid):
    path = tmp_path / "existing.parquet"
    write_parquet(pl.DataFrame({"id": [7]}), path)
    before = path.read_bytes()
    batches = [pl.DataFrame({"id": ["8"]}), pl.DataFrame({"id": [invalid]})]

    with pytest.raises(pl.exceptions.InvalidOperationError):
        write_parquet_batches(batches, path, schema={"id": pl.Int64})

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_batch_ingestion_never_publishes_partial_artifact(tmp_path):
    path = tmp_path / "new.parquet"

    def batches():
        yield pl.DataFrame({"id": ["source-1"]})
        raise ValueError("synthetic source read failed")

    with pytest.raises(ValueError, match="synthetic source read failed"):
        write_parquet_batches(batches(), path, schema={"id": pl.String})

    assert list(tmp_path.iterdir()) == []
