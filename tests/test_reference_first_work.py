from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
import sqlite3

import pytest

from biominer.run.constants import PRODUCTION_JOB_NAME
from biominer.run.reference_work import (
    REFERENCE_FIRST_WORK_KINDS,
    ReferenceFirstWorkItem,
    ReferenceFirstWorkKind,
    ReferenceFirstWorkPayloadError,
    WorkLeaseLostError,
    claim_reference_first_work,
    enqueue_reference_first_work,
)
from biominer.workstore.postgres import PostgresWorkStore
from biominer.workstore.sqlite import SQLiteWorkStore


EXPECTED_WORK_KINDS = (
    "reference_metadata_fetch",
    "reference_download",
    "reference_review_export",
    "reference_review_import",
    "reference_embedding",
    "prototype_build",
    "classifier_fit",
    "calibration",
    "flickr_embedding",
    "target_aware_scoring",
)


def test_reference_first_work_kinds_cover_every_resumable_queue() -> None:
    assert (
        tuple(kind.value for kind in REFERENCE_FIRST_WORK_KINDS) == EXPECTED_WORK_KINDS
    )


def test_reference_first_work_key_covers_all_semantic_inputs() -> None:
    item = _work_item(
        ReferenceFirstWorkKind.REFERENCE_EMBEDDING,
        input_uris={
            "review": "s3://bucket/review.parquet",
            "media": "s3://bucket/media/",
        },
        dependency_fingerprints={"model": "sha256:model", "review": "sha256:review"},
    )
    reordered = _work_item(
        ReferenceFirstWorkKind.REFERENCE_EMBEDDING,
        input_uris={
            "media": "s3://bucket/media/",
            "review": "s3://bucket/review.parquet",
        },
        dependency_fingerprints={"review": "sha256:review", "model": "sha256:model"},
    )

    assert item.work_key == reordered.work_key
    assert item.payload_fingerprint == reordered.payload_fingerprint
    assert item.work_key != _replace_item(item, partition_id="part-0002").work_key
    assert (
        item.work_key
        != _replace_item(item, config_fingerprint="sha256:config-v2").work_key
    )
    assert (
        item.work_key != _replace_item(item, reference_bank_version="bank-v2").work_key
    )
    assert (
        item.work_key
        != _replace_item(item, output_uri="s3://bucket/other.parquet").work_key
    )


def test_reference_first_queues_are_idempotent_and_round_trip_all_kinds(
    tmp_path,
) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    items = [
        _work_item(kind, partition_id=f"part-{index:04d}")
        for index, kind in enumerate(REFERENCE_FIRST_WORK_KINDS)
    ]

    first = enqueue_reference_first_work(store, items)
    second = enqueue_reference_first_work(store, list(reversed(items)))

    assert first.attempted_work_items == 10
    assert first.enqueued_work_items == 10
    assert first.duplicate_work_items == 0
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 10

    for expected in items:
        batch = claim_reference_first_work(
            store,
            kind=expected.kind,
            registry_version=expected.registry_version,
            worker_id="worker-1",
            limit=1,
            stale_after_seconds=300,
        )
        assert batch.stale_claims_requeued == 0
        assert batch.claimed_work_items == 1
        lease = batch.leases[0]
        assert lease.item == expected
        assert lease.attempt_count == 1
        assert lease.heartbeat()
        lease.complete(checksum=f"sha256:{expected.kind.value}", row_count=1)

        [stored] = store.list_work_items(
            job_name=PRODUCTION_JOB_NAME,
            stage=expected.kind.value,
            registry_version=expected.registry_version,
            statuses=["completed"],
        )
        assert stored["work_key"] == expected.work_key
        assert stored["output_uri"] == expected.output_uri
        assert stored["claimed_by"] is None
        assert stored["claimed_at"] is None


def test_reclaimed_reference_work_rejects_stale_worker_completion(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    item = _work_item(ReferenceFirstWorkKind.FLICKR_EMBEDDING)
    enqueue_reference_first_work(store, [item])
    first = claim_reference_first_work(
        store,
        kind=item.kind,
        registry_version=item.registry_version,
        worker_id="worker-a",
        limit=1,
        stale_after_seconds=60,
    ).leases[0]

    old_claim = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE biominer_work_items SET claimed_at = ? WHERE work_key = ?",
            (old_claim, item.work_key),
        )

    reclaimed_batch = claim_reference_first_work(
        store,
        kind=item.kind,
        registry_version=item.registry_version,
        worker_id="worker-b",
        limit=1,
        stale_after_seconds=60,
    )
    second = reclaimed_batch.leases[0]

    assert reclaimed_batch.stale_claims_requeued == 1
    assert first.attempt_count == 1
    assert second.attempt_count == 2
    assert not first.heartbeat()
    with pytest.raises(WorkLeaseLostError, match="no longer owned"):
        first.complete(checksum="sha256:stale", row_count=1)

    second.complete(checksum="sha256:current", row_count=2)
    [stored] = store.list_work_items(
        job_name=PRODUCTION_JOB_NAME,
        stage=item.kind.value,
        registry_version=item.registry_version,
        statuses=["completed"],
    )
    assert stored["attempt_count"] == 2
    assert stored["checksum"] == "sha256:current"
    assert stored["row_count"] == 2


def test_reference_work_lease_rejects_wrong_owner_generation_and_expiry(
    tmp_path,
) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    item = _work_item(ReferenceFirstWorkKind.CALIBRATION)
    enqueue_reference_first_work(store, [item])
    lease = claim_reference_first_work(
        store,
        kind=item.kind,
        registry_version=item.registry_version,
        worker_id="worker-1",
        limit=1,
        stale_after_seconds=60,
    ).leases[0]

    assert not store.renew_claim(
        item.work_key,
        worker_id="worker-2",
        attempt_count=lease.attempt_count,
        stale_after_seconds=60,
    )
    assert not store.renew_claim(
        item.work_key,
        worker_id="worker-1",
        attempt_count=lease.attempt_count + 1,
        stale_after_seconds=60,
    )

    old_claim = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE biominer_work_items SET claimed_at = ? WHERE work_key = ?",
            (old_claim, item.work_key),
        )
    assert not lease.heartbeat()
    with pytest.raises(WorkLeaseLostError, match="no longer owned"):
        lease.fail("expired")


def test_current_reference_work_lease_records_failure_once(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    item = _work_item(ReferenceFirstWorkKind.CLASSIFIER_FIT)
    enqueue_reference_first_work(store, [item])
    lease = claim_reference_first_work(
        store,
        kind=item.kind,
        registry_version=item.registry_version,
        worker_id="worker-1",
        limit=1,
        stale_after_seconds=60,
    ).leases[0]

    lease.fail("training data rejected")

    [stored] = store.list_work_items(
        job_name=PRODUCTION_JOB_NAME,
        stage=item.kind.value,
        registry_version=item.registry_version,
        statuses=["failed"],
    )
    assert stored["error"] == "training data rejected"
    assert stored["claimed_by"] is None
    assert stored["claimed_at"] is None
    with pytest.raises(WorkLeaseLostError, match="no longer owned"):
        lease.fail("duplicate failure")


def test_reference_work_payload_rejects_semantic_tampering(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    item = _work_item(ReferenceFirstWorkKind.PROTOTYPE_BUILD)
    enqueue_reference_first_work(store, [item])
    [row] = store.list_work_items(
        job_name=PRODUCTION_JOB_NAME,
        stage=item.kind.value,
        registry_version=item.registry_version,
    )
    row["payload"]["partition_id"] = "tampered"

    with pytest.raises(ReferenceFirstWorkPayloadError, match="key does not match"):
        ReferenceFirstWorkItem.from_workstore_item(row)


def test_postgres_claim_mutations_use_owner_generation_and_expiry_guards() -> None:
    connection = _RecordingPostgresConnection(rowcount=1)
    store = PostgresWorkStore(
        "postgresql://user:pass@example.test/db",
        connect=lambda: connection,
    )

    assert store.renew_claim(
        "work-1",
        worker_id="worker-1",
        attempt_count=2,
        stale_after_seconds=90,
    )
    assert store.complete_claim(
        "work-1",
        worker_id="worker-1",
        attempt_count=2,
        stale_after_seconds=90,
        output_uri="s3://bucket/output.parquet",
        checksum="sha256:output",
        row_count=3,
    )
    assert store.fail_claim(
        "work-1",
        worker_id="worker-1",
        attempt_count=2,
        stale_after_seconds=90,
        error="failed",
    )

    assert len(connection.calls) == 3
    for sql, _params in connection.calls:
        assert "status = %s" in sql
        assert "claimed_by = %s" in sql
        assert "attempt_count = %s" in sql
        assert "claimed_at IS NOT NULL" in sql
        assert "claimed_at >= now() - (%s * INTERVAL '1 second')" in sql
    assert connection.calls[0][1] == ("work-1", "claimed", "worker-1", 2, 90)
    assert connection.calls[1][1][-5:] == ("work-1", "claimed", "worker-1", 2, 90)
    assert connection.calls[2][1][-5:] == ("work-1", "claimed", "worker-1", 2, 90)


def _work_item(
    kind: ReferenceFirstWorkKind,
    *,
    partition_id: str = "part-0001",
    input_uris: dict[str, str] | None = None,
    dependency_fingerprints: dict[str, str] | None = None,
) -> ReferenceFirstWorkItem:
    return ReferenceFirstWorkItem.create(
        kind=kind,
        run_id="run-1",
        registry_version="registry-v1",
        partition_id=partition_id,
        output_uri=f"s3://bucket/{kind.value}/{partition_id}.parquet",
        config_fingerprint="sha256:config-v1",
        input_uris=input_uris or {"source": "s3://bucket/source.parquet"},
        dependency_fingerprints=dependency_fingerprints
        or {"registry": "sha256:registry-v1"},
        reference_bank_version="bank-v1",
    )


def _replace_item(
    item: ReferenceFirstWorkItem,
    *,
    partition_id: str | None = None,
    output_uri: str | None = None,
    config_fingerprint: str | None = None,
    reference_bank_version: str | None = None,
) -> ReferenceFirstWorkItem:
    return ReferenceFirstWorkItem.create(
        kind=item.kind,
        run_id=item.run_id,
        registry_version=item.registry_version,
        partition_id=partition_id or item.partition_id,
        output_uri=output_uri or item.output_uri,
        config_fingerprint=config_fingerprint or item.config_fingerprint,
        input_uris=dict(item.input_uris),
        dependency_fingerprints=dict(item.dependency_fingerprints),
        reference_bank_version=reference_bank_version or item.reference_bank_version,
    )


class _RecordingResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _RecordingPostgresConnection:
    def __init__(self, *, rowcount: int) -> None:
        self.rowcount = rowcount
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> _RecordingPostgresConnection:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> _RecordingResult:
        self.calls.append((" ".join(sql.split()), params))
        return _RecordingResult(self.rowcount)
