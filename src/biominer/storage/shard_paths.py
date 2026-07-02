from __future__ import annotations

from pathlib import Path

from biominer.storage.uri import join_uri


def build_parquet_shard_uri(
    base_prefix: str | Path,
    *,
    stage: str,
    run_id: str,
    worker_id: str,
    batch_id: int | str,
) -> str:
    batch = f"{batch_id:06d}" if isinstance(batch_id, int) else str(batch_id)
    if not batch.endswith(".parquet"):
        batch = f"{batch}.parquet"
    return join_uri(
        base_prefix,
        "evidence",
        f"stage={stage}",
        f"run_id={run_id}",
        f"worker={worker_id}",
        f"batch={batch}",
    )
