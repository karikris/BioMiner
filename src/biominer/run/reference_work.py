from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from biominer.common.status import CLAIMED
from biominer.run.constants import PRODUCTION_JOB_NAME
from biominer.workstore.base import WorkStore, validate_claim_lease
from biominer.workstore.keys import stable_work_key


REFERENCE_FIRST_WORK_SCHEMA_VERSION = 1


class ReferenceFirstWorkKind(StrEnum):
    REFERENCE_METADATA_FETCH = "reference_metadata_fetch"
    REFERENCE_DOWNLOAD = "reference_download"
    REFERENCE_REVIEW_EXPORT = "reference_review_export"
    REFERENCE_REVIEW_IMPORT = "reference_review_import"
    REFERENCE_EMBEDDING = "reference_embedding"
    PROTOTYPE_BUILD = "prototype_build"
    CLASSIFIER_FIT = "classifier_fit"
    CALIBRATION = "calibration"
    FLICKR_EMBEDDING = "flickr_embedding"
    TARGET_AWARE_SCORING = "target_aware_scoring"


REFERENCE_FIRST_WORK_KINDS: tuple[ReferenceFirstWorkKind, ...] = tuple(
    ReferenceFirstWorkKind
)


class ReferenceFirstWorkPayloadError(ValueError):
    pass


class WorkLeaseLostError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReferenceFirstWorkItem:
    kind: ReferenceFirstWorkKind
    run_id: str
    registry_version: str
    partition_id: str
    output_uri: str
    config_fingerprint: str
    input_uris: tuple[tuple[str, str], ...] = ()
    dependency_fingerprints: tuple[tuple[str, str], ...] = ()
    reference_bank_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReferenceFirstWorkKind):
            raise TypeError("kind must be a ReferenceFirstWorkKind")
        for field_name in (
            "run_id",
            "registry_version",
            "partition_id",
            "output_uri",
            "config_fingerprint",
        ):
            _required_text(getattr(self, field_name), field=field_name)
        if self.reference_bank_version is not None:
            _required_text(self.reference_bank_version, field="reference_bank_version")
        _validate_pairs(self.input_uris, field="input_uris")
        _validate_pairs(
            self.dependency_fingerprints,
            field="dependency_fingerprints",
        )

    @classmethod
    def create(
        cls,
        *,
        kind: ReferenceFirstWorkKind,
        run_id: str,
        registry_version: str,
        partition_id: str,
        output_uri: str,
        config_fingerprint: str,
        input_uris: Mapping[str, str] | None = None,
        dependency_fingerprints: Mapping[str, str] | None = None,
        reference_bank_version: str | None = None,
    ) -> ReferenceFirstWorkItem:
        return cls(
            kind=kind,
            run_id=run_id,
            registry_version=registry_version,
            partition_id=partition_id,
            output_uri=output_uri,
            config_fingerprint=config_fingerprint,
            input_uris=_sorted_pairs(input_uris or {}, field="input_uris"),
            dependency_fingerprints=_sorted_pairs(
                dependency_fingerprints or {},
                field="dependency_fingerprints",
            ),
            reference_bank_version=reference_bank_version,
        )

    @property
    def payload_fingerprint(self) -> str:
        return stable_work_key(self._semantic_payload())

    @property
    def work_key(self) -> str:
        return stable_work_key(
            self._semantic_payload(),
            prefix=f"reference_first:{self.kind.value}",
        )

    def to_workstore_payload(self) -> dict[str, Any]:
        return {
            "work_key": self.work_key,
            **self._semantic_payload(),
            "payload_fingerprint": self.payload_fingerprint,
        }

    @classmethod
    def from_workstore_item(cls, row: Mapping[str, Any]) -> ReferenceFirstWorkItem:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise ReferenceFirstWorkPayloadError("work item payload must be an object")
        expected_fields = {
            "schema_version",
            "kind",
            "run_id",
            "registry_version",
            "partition_id",
            "output_uri",
            "config_fingerprint",
            "input_uris",
            "dependency_fingerprints",
            "reference_bank_version",
            "payload_fingerprint",
        }
        if set(payload) != expected_fields:
            missing = sorted(expected_fields - set(payload))
            unexpected = sorted(set(payload) - expected_fields)
            raise ReferenceFirstWorkPayloadError(
                f"invalid work payload fields: missing={missing}, unexpected={unexpected}"
            )
        schema_version = payload.get("schema_version")
        if schema_version != REFERENCE_FIRST_WORK_SCHEMA_VERSION:
            raise ReferenceFirstWorkPayloadError(
                f"unsupported work payload schema_version: {schema_version!r}"
            )
        try:
            kind = ReferenceFirstWorkKind(
                _required_text(payload.get("kind"), field="kind")
            )
        except ValueError as exc:
            raise ReferenceFirstWorkPayloadError(str(exc)) from exc
        input_uris = _required_mapping(payload.get("input_uris"), field="input_uris")
        dependency_fingerprints = _required_mapping(
            payload.get("dependency_fingerprints"),
            field="dependency_fingerprints",
        )
        reference_bank_version = payload.get("reference_bank_version")
        if reference_bank_version is not None:
            reference_bank_version = _required_text(
                reference_bank_version,
                field="reference_bank_version",
            )
        item = cls.create(
            kind=kind,
            run_id=_required_text(payload.get("run_id"), field="run_id"),
            registry_version=_required_text(
                payload.get("registry_version"),
                field="registry_version",
            ),
            partition_id=_required_text(
                payload.get("partition_id"),
                field="partition_id",
            ),
            output_uri=_required_text(payload.get("output_uri"), field="output_uri"),
            config_fingerprint=_required_text(
                payload.get("config_fingerprint"),
                field="config_fingerprint",
            ),
            input_uris={str(key): str(value) for key, value in input_uris.items()},
            dependency_fingerprints={
                str(key): str(value) for key, value in dependency_fingerprints.items()
            },
            reference_bank_version=reference_bank_version,
        )
        _validate_workstore_envelope(row, item)
        fingerprint = _required_text(
            payload.get("payload_fingerprint"),
            field="payload_fingerprint",
        )
        if fingerprint != item.payload_fingerprint:
            raise ReferenceFirstWorkPayloadError(
                "work payload fingerprint does not match its semantic fields"
            )
        return item

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_FIRST_WORK_SCHEMA_VERSION,
            "kind": self.kind.value,
            "run_id": self.run_id,
            "registry_version": self.registry_version,
            "partition_id": self.partition_id,
            "output_uri": self.output_uri,
            "config_fingerprint": self.config_fingerprint,
            "input_uris": dict(self.input_uris),
            "dependency_fingerprints": dict(self.dependency_fingerprints),
            "reference_bank_version": self.reference_bank_version,
        }


@dataclass(frozen=True)
class ReferenceFirstEnqueueResult:
    attempted_work_items: int
    enqueued_work_items: int

    @property
    def duplicate_work_items(self) -> int:
        return self.attempted_work_items - self.enqueued_work_items


@dataclass(frozen=True)
class ReferenceFirstWorkLease:
    item: ReferenceFirstWorkItem
    worker_id: str
    attempt_count: int
    stale_after_seconds: int
    _workstore: WorkStore = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_claim_lease(
            work_key=self.item.work_key,
            worker_id=self.worker_id,
            attempt_count=self.attempt_count,
            stale_after_seconds=self.stale_after_seconds,
        )

    def heartbeat(self) -> bool:
        return self._workstore.renew_claim(
            self.item.work_key,
            worker_id=self.worker_id,
            attempt_count=self.attempt_count,
            stale_after_seconds=self.stale_after_seconds,
        )

    def complete(self, *, checksum: str | None, row_count: int | None) -> None:
        completed = self._workstore.complete_claim(
            self.item.work_key,
            worker_id=self.worker_id,
            attempt_count=self.attempt_count,
            stale_after_seconds=self.stale_after_seconds,
            output_uri=self.item.output_uri,
            checksum=checksum,
            row_count=row_count,
        )
        if not completed:
            raise WorkLeaseLostError(
                f"work lease is no longer owned: {self.item.work_key}"
            )

    def fail(self, error: str) -> None:
        _required_text(error, field="error")
        failed = self._workstore.fail_claim(
            self.item.work_key,
            worker_id=self.worker_id,
            attempt_count=self.attempt_count,
            stale_after_seconds=self.stale_after_seconds,
            error=error,
        )
        if not failed:
            raise WorkLeaseLostError(
                f"work lease is no longer owned: {self.item.work_key}"
            )


@dataclass(frozen=True)
class ReferenceFirstClaimBatch:
    kind: ReferenceFirstWorkKind
    stale_claims_requeued: int
    leases: tuple[ReferenceFirstWorkLease, ...]

    @property
    def claimed_work_items(self) -> int:
        return len(self.leases)


def enqueue_reference_first_work(
    workstore: WorkStore,
    items: Sequence[ReferenceFirstWorkItem],
) -> ReferenceFirstEnqueueResult:
    grouped: dict[tuple[str, ReferenceFirstWorkKind], list[ReferenceFirstWorkItem]] = {}
    for item in items:
        if not isinstance(item, ReferenceFirstWorkItem):
            raise TypeError("items must contain ReferenceFirstWorkItem values")
        grouped.setdefault((item.registry_version, item.kind), []).append(item)

    inserted = 0
    for registry_version, kind in sorted(
        grouped,
        key=lambda key: (key[0], key[1].value),
    ):
        batch = sorted(
            grouped[(registry_version, kind)], key=lambda item: item.work_key
        )
        inserted += workstore.enqueue_work(
            PRODUCTION_JOB_NAME,
            registry_version,
            [item.to_workstore_payload() for item in batch],
            stage=kind.value,
        )
    return ReferenceFirstEnqueueResult(
        attempted_work_items=len(items),
        enqueued_work_items=inserted,
    )


def claim_reference_first_work(
    workstore: WorkStore,
    *,
    kind: ReferenceFirstWorkKind,
    registry_version: str,
    worker_id: str,
    limit: int,
    stale_after_seconds: int,
) -> ReferenceFirstClaimBatch:
    if not isinstance(kind, ReferenceFirstWorkKind):
        raise TypeError("kind must be a ReferenceFirstWorkKind")
    _required_text(registry_version, field="registry_version")
    _required_text(worker_id, field="worker_id")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    validate_claim_lease(
        work_key="claim_batch",
        worker_id=worker_id,
        attempt_count=1,
        stale_after_seconds=stale_after_seconds,
    )
    stale_claims_requeued = workstore.requeue_stale_claims(
        job_name=PRODUCTION_JOB_NAME,
        stage=kind.value,
        registry_version=registry_version,
        stale_after_seconds=stale_after_seconds,
    )
    rows = workstore.claim_next_batch(
        worker_id,
        limit,
        job_name=PRODUCTION_JOB_NAME,
        stage=kind.value,
        registry_version=registry_version,
    )
    leases = tuple(
        _lease_from_claimed_row(
            workstore,
            row=row,
            expected_kind=kind,
            worker_id=worker_id,
            stale_after_seconds=stale_after_seconds,
        )
        for row in rows
    )
    return ReferenceFirstClaimBatch(
        kind=kind,
        stale_claims_requeued=stale_claims_requeued,
        leases=leases,
    )


def _lease_from_claimed_row(
    workstore: WorkStore,
    *,
    row: Mapping[str, Any],
    expected_kind: ReferenceFirstWorkKind,
    worker_id: str,
    stale_after_seconds: int,
) -> ReferenceFirstWorkLease:
    if row.get("status") != CLAIMED:
        raise ReferenceFirstWorkPayloadError(
            "claimed work row does not have claimed status"
        )
    if row.get("claimed_by") != worker_id:
        raise ReferenceFirstWorkPayloadError(
            "claimed work row has the wrong worker owner"
        )
    if row.get("claimed_at") is None:
        raise ReferenceFirstWorkPayloadError(
            "claimed work row has no claimed_at timestamp"
        )
    item = ReferenceFirstWorkItem.from_workstore_item(row)
    if item.kind != expected_kind:
        raise ReferenceFirstWorkPayloadError(
            f"claimed work kind mismatch: expected={expected_kind.value}, actual={item.kind.value}"
        )
    attempt_count = row.get("attempt_count")
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        raise ReferenceFirstWorkPayloadError(
            "claimed work row has an invalid attempt_count"
        )
    return ReferenceFirstWorkLease(
        item=item,
        worker_id=worker_id,
        attempt_count=attempt_count,
        stale_after_seconds=stale_after_seconds,
        _workstore=workstore,
    )


def _validate_workstore_envelope(
    row: Mapping[str, Any],
    item: ReferenceFirstWorkItem,
) -> None:
    if row.get("job_name") != PRODUCTION_JOB_NAME:
        raise ReferenceFirstWorkPayloadError("work item has the wrong job_name")
    if row.get("stage") != item.kind.value:
        raise ReferenceFirstWorkPayloadError("work item stage does not match its kind")
    if row.get("registry_version") != item.registry_version:
        raise ReferenceFirstWorkPayloadError(
            "work item registry_version does not match its payload"
        )
    if row.get("work_key") != item.work_key:
        raise ReferenceFirstWorkPayloadError(
            "work item key does not match its semantic payload"
        )


def _sorted_pairs(
    values: Mapping[str, str],
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(
        sorted(
            (
                _required_text(key, field=f"{field} key"),
                _required_text(value, field=f"{field}[{key}]"),
            )
            for key, value in values.items()
        )
    )
    _validate_pairs(pairs, field=field)
    return pairs


def _validate_pairs(values: tuple[tuple[str, str], ...], *, field: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    if values != tuple(sorted(values)):
        raise ValueError(f"{field} must be sorted by key")
    keys: set[str] = set()
    for pair in values:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{field} entries must be two-item tuples")
        key = _required_text(pair[0], field=f"{field} key")
        _required_text(pair[1], field=f"{field}[{key}]")
        if key in keys:
            raise ValueError(f"{field} contains duplicate key: {key}")
        keys.add(key)


def _required_mapping(value: Any, *, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ReferenceFirstWorkPayloadError(f"{field} must be an object")
    for key, item in value.items():
        _required_text(key, field=f"{field} key")
        _required_text(item, field=f"{field}[{key}]")
    return value


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceFirstWorkPayloadError(f"{field} must be a non-empty string")
    return value
