"""Deterministic memory-aware image batching with bounded retry semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import floor, isfinite

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage


MPS_MEMORY_SNAPSHOT_VERSION = "mps-memory-snapshot-v1"
MEMORY_AWARE_IMAGE_BATCH_POLICY_VERSION = "memory-aware-image-batch-policy-v1"
IMAGE_BATCH_ATTEMPT_VERSION = "image-batch-attempt-v1"
MEMORY_AWARE_IMAGE_BATCH_METRICS_VERSION = "memory-aware-image-batch-metrics-v1"


@dataclass(frozen=True, slots=True)
class MpsMemorySnapshot:
    """One allocator snapshot expressed in bytes."""

    current_allocated_bytes: int
    driver_allocated_bytes: int
    recommended_max_bytes: int
    snapshot_fingerprint: str | None = None

    def __post_init__(self) -> None:
        current = _nonnegative_int(
            self.current_allocated_bytes,
            field="current_allocated_bytes",
        )
        driver = _nonnegative_int(
            self.driver_allocated_bytes,
            field="driver_allocated_bytes",
        )
        recommended = _positive_int(
            self.recommended_max_bytes,
            field="recommended_max_bytes",
        )
        if current > driver:
            raise ValueError("MPS current allocation cannot exceed driver allocation")
        base = {
            "schema_version": MPS_MEMORY_SNAPSHOT_VERSION,
            "current_allocated_bytes": current,
            "driver_allocated_bytes": driver,
            "recommended_max_bytes": recommended,
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if self.snapshot_fingerprint not in {None, fingerprint}:
            raise ValueError("MPS memory snapshot fingerprint mismatch")
        object.__setattr__(self, "current_allocated_bytes", current)
        object.__setattr__(self, "driver_allocated_bytes", driver)
        object.__setattr__(self, "recommended_max_bytes", recommended)
        object.__setattr__(self, "snapshot_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class MemoryAwareImageBatchPolicy:
    """Configured image bounds plus an explicit MPS memory budget model."""

    initial_batch_size: int = 24
    minimum_batch_size: int = 1
    maximum_batch_size: int = 24
    target_driver_memory_fraction: float = 0.8
    estimated_incremental_bytes_per_image: int = 64 * 1024 * 1024
    retry_shrink_divisor: int = 2
    schema_version: str = MEMORY_AWARE_IMAGE_BATCH_POLICY_VERSION
    policy_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_AWARE_IMAGE_BATCH_POLICY_VERSION:
            raise ValueError("unsupported memory-aware image batch policy version")
        initial = _positive_int(self.initial_batch_size, field="initial_batch_size")
        minimum = _positive_int(self.minimum_batch_size, field="minimum_batch_size")
        maximum = _positive_int(self.maximum_batch_size, field="maximum_batch_size")
        if not minimum <= initial <= maximum:
            raise ValueError(
                "image batch sizes must satisfy minimum <= initial <= maximum"
            )
        fraction = _finite_float(
            self.target_driver_memory_fraction,
            field="target_driver_memory_fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("target_driver_memory_fraction must be in (0, 1]")
        estimated_bytes = _positive_int(
            self.estimated_incremental_bytes_per_image,
            field="estimated_incremental_bytes_per_image",
        )
        shrink_divisor = _positive_int(
            self.retry_shrink_divisor,
            field="retry_shrink_divisor",
        )
        if shrink_divisor < 2:
            raise ValueError("retry_shrink_divisor must be at least two")
        base = {
            "schema_version": MEMORY_AWARE_IMAGE_BATCH_POLICY_VERSION,
            "initial_batch_size": initial,
            "minimum_batch_size": minimum,
            "maximum_batch_size": maximum,
            "target_driver_memory_fraction": fraction,
            "estimated_incremental_bytes_per_image": estimated_bytes,
            "retry_shrink_divisor": shrink_divisor,
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if self.policy_fingerprint not in {None, fingerprint}:
            raise ValueError("memory-aware image batch policy fingerprint mismatch")
        object.__setattr__(self, "initial_batch_size", initial)
        object.__setattr__(self, "minimum_batch_size", minimum)
        object.__setattr__(self, "maximum_batch_size", maximum)
        object.__setattr__(self, "target_driver_memory_fraction", fraction)
        object.__setattr__(
            self,
            "estimated_incremental_bytes_per_image",
            estimated_bytes,
        )
        object.__setattr__(self, "retry_shrink_divisor", shrink_divisor)
        object.__setattr__(self, "policy_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class ImageBatchAttempt:
    """One successful or failed batch slice and the memory decision behind it."""

    schema_version: str
    attempt_number: int
    start_index: int
    requested_batch_size: int
    remaining_images_before: int
    decision_reason: str
    memory_capacity_images: int | None
    snapshot_fingerprint: str | None
    outcome: str
    attempt_fingerprint: str


@dataclass(frozen=True, slots=True)
class MemoryAwareImageBatchMetrics:
    """Complete attempts and high-water memory values for one encode call."""

    schema_version: str
    policy_fingerprint: str
    telemetry_status: str
    images_encoded: int
    successful_batches: int
    batch_attempts: int
    memory_retries: int
    memory_limited_attempts: int
    initial_batch_size: int
    final_successful_batch_size: int
    minimum_successful_batch_size: int
    maximum_successful_batch_size: int
    snapshots_observed: int
    peak_current_allocated_bytes: int | None
    peak_driver_allocated_bytes: int | None
    recommended_max_bytes_minimum: int | None
    attempts: tuple[ImageBatchAttempt, ...]
    metrics_fingerprint: str


@dataclass(frozen=True, slots=True)
class MemoryAwareImageEncodingResult:
    """Encoded vectors plus the decisions that shaped their batches."""

    vectors: tuple[tuple[float, ...], ...]
    metrics: MemoryAwareImageBatchMetrics


def encode_images_memory_aware(
    images: Sequence[DecodedImage],
    *,
    encode_batch: Callable[[Sequence[DecodedImage]], Sequence[Sequence[float]]],
    policy: MemoryAwareImageBatchPolicy,
    memory_probe: Callable[[], MpsMemorySnapshot] | None = None,
) -> MemoryAwareImageEncodingResult:
    """Encode each image once, shrinking only an uncommitted failed slice."""

    if isinstance(images, str | bytes) or not isinstance(images, Sequence):
        raise TypeError("images must be a sequence")
    items = tuple(images)
    if any(not isinstance(image, DecodedImage) for image in items):
        raise TypeError("images must contain DecodedImage values")
    if not callable(encode_batch):
        raise TypeError("encode_batch must be callable")
    if not isinstance(policy, MemoryAwareImageBatchPolicy):
        raise TypeError("policy must be MemoryAwareImageBatchPolicy")
    if memory_probe is not None and not callable(memory_probe):
        raise TypeError("memory_probe must be callable or null")

    attempts: list[ImageBatchAttempt] = []
    snapshots: list[MpsMemorySnapshot] = []
    vectors: list[tuple[float, ...]] = []
    successful_sizes: list[int] = []
    start = 0
    current_limit = policy.initial_batch_size
    memory_retries = 0
    memory_limited_attempts = 0
    while start < len(items):
        remaining = len(items) - start
        snapshot = memory_probe() if memory_probe is not None else None
        if snapshot is not None and not isinstance(snapshot, MpsMemorySnapshot):
            raise TypeError("memory_probe must return MpsMemorySnapshot")
        if snapshot is not None:
            snapshots.append(snapshot)
        requested, capacity, decision_reason = _requested_batch_size(
            remaining=remaining,
            current_limit=current_limit,
            policy=policy,
            snapshot=snapshot,
        )
        if decision_reason == "mps_memory_limited":
            memory_limited_attempts += 1
        batch = items[start : start + requested]
        try:
            raw_vectors = tuple(encode_batch(batch))
        except RuntimeError as exc:
            if requested <= policy.minimum_batch_size or not is_image_memory_error(exc):
                raise
            attempts.append(
                _attempt(
                    attempt_number=len(attempts) + 1,
                    start_index=start,
                    requested_batch_size=requested,
                    remaining_images_before=remaining,
                    decision_reason=decision_reason,
                    memory_capacity_images=capacity,
                    snapshot=snapshot,
                    outcome="memory_error_retry",
                )
            )
            memory_retries += 1
            current_limit = max(
                policy.minimum_batch_size,
                requested // policy.retry_shrink_divisor,
            )
            continue
        if len(raw_vectors) != len(batch):
            raise ValueError(
                "image encoder returned a vector count that differs from its batch"
            )
        normalized_vectors = tuple(
            tuple(_finite_float(value, field="encoded vector") for value in vector)
            for vector in raw_vectors
        )
        if any(not vector for vector in normalized_vectors):
            raise ValueError("image encoder returned an empty vector")
        attempts.append(
            _attempt(
                attempt_number=len(attempts) + 1,
                start_index=start,
                requested_batch_size=requested,
                remaining_images_before=remaining,
                decision_reason=decision_reason,
                memory_capacity_images=capacity,
                snapshot=snapshot,
                outcome="success",
            )
        )
        vectors.extend(normalized_vectors)
        successful_sizes.append(requested)
        start += requested
        if memory_probe is not None:
            after = memory_probe()
            if not isinstance(after, MpsMemorySnapshot):
                raise TypeError("memory_probe must return MpsMemorySnapshot")
            snapshots.append(after)

    metrics = _metrics(
        policy=policy,
        images_encoded=len(items),
        attempts=tuple(attempts),
        successful_sizes=tuple(successful_sizes),
        snapshots=tuple(snapshots),
        memory_retries=memory_retries,
        memory_limited_attempts=memory_limited_attempts,
        telemetry_status=(
            "not_observed_no_images"
            if not items and memory_probe is not None
            else "mps_memory_available"
            if memory_probe is not None
            else "not_available_fixed_batch"
        ),
    )
    return MemoryAwareImageEncodingResult(vectors=tuple(vectors), metrics=metrics)


def is_image_memory_error(exc: BaseException) -> bool:
    """Return true only for recognized accelerator allocation failures."""

    if not isinstance(exc, RuntimeError):
        return False
    message = " ".join(str(exc).casefold().split())
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda out of memory",
            "mps memory",
            "allocation failed",
        )
    )


def _requested_batch_size(
    *,
    remaining: int,
    current_limit: int,
    policy: MemoryAwareImageBatchPolicy,
    snapshot: MpsMemorySnapshot | None,
) -> tuple[int, int | None, str]:
    configured = min(remaining, current_limit, policy.maximum_batch_size)
    if snapshot is None:
        return configured, None, "configured_no_memory_probe"
    target_bytes = floor(
        snapshot.recommended_max_bytes * policy.target_driver_memory_fraction
    )
    headroom = max(0, target_bytes - snapshot.driver_allocated_bytes)
    capacity = headroom // policy.estimated_incremental_bytes_per_image
    safe_capacity = max(policy.minimum_batch_size, capacity)
    requested = min(configured, safe_capacity)
    return (
        requested,
        capacity,
        "mps_memory_limited"
        if requested < configured
        else "configured_within_mps_budget",
    )


def _attempt(
    *,
    attempt_number: int,
    start_index: int,
    requested_batch_size: int,
    remaining_images_before: int,
    decision_reason: str,
    memory_capacity_images: int | None,
    snapshot: MpsMemorySnapshot | None,
    outcome: str,
) -> ImageBatchAttempt:
    base = {
        "schema_version": IMAGE_BATCH_ATTEMPT_VERSION,
        "attempt_number": attempt_number,
        "start_index": start_index,
        "requested_batch_size": requested_batch_size,
        "remaining_images_before": remaining_images_before,
        "decision_reason": decision_reason,
        "memory_capacity_images": memory_capacity_images,
        "snapshot_fingerprint": (
            snapshot.snapshot_fingerprint if snapshot is not None else None
        ),
        "outcome": outcome,
    }
    return ImageBatchAttempt(
        schema_version=IMAGE_BATCH_ATTEMPT_VERSION,
        attempt_number=attempt_number,
        start_index=start_index,
        requested_batch_size=requested_batch_size,
        remaining_images_before=remaining_images_before,
        decision_reason=decision_reason,
        memory_capacity_images=memory_capacity_images,
        snapshot_fingerprint=(
            snapshot.snapshot_fingerprint if snapshot is not None else None
        ),
        outcome=outcome,
        attempt_fingerprint=canonical_semantic_fingerprint(base),
    )


def _metrics(
    *,
    policy: MemoryAwareImageBatchPolicy,
    images_encoded: int,
    attempts: tuple[ImageBatchAttempt, ...],
    successful_sizes: tuple[int, ...],
    snapshots: tuple[MpsMemorySnapshot, ...],
    memory_retries: int,
    memory_limited_attempts: int,
    telemetry_status: str,
) -> MemoryAwareImageBatchMetrics:
    peak_current = (
        max(snapshot.current_allocated_bytes for snapshot in snapshots)
        if snapshots
        else None
    )
    peak_driver = (
        max(snapshot.driver_allocated_bytes for snapshot in snapshots)
        if snapshots
        else None
    )
    recommended_minimum = (
        min(snapshot.recommended_max_bytes for snapshot in snapshots)
        if snapshots
        else None
    )
    final_size = successful_sizes[-1] if successful_sizes else 0
    minimum_size = min(successful_sizes) if successful_sizes else 0
    maximum_size = max(successful_sizes) if successful_sizes else 0
    base = {
        "schema_version": MEMORY_AWARE_IMAGE_BATCH_METRICS_VERSION,
        "policy_fingerprint": policy.policy_fingerprint,
        "telemetry_status": telemetry_status,
        "images_encoded": images_encoded,
        "successful_batches": len(successful_sizes),
        "batch_attempts": len(attempts),
        "memory_retries": memory_retries,
        "memory_limited_attempts": memory_limited_attempts,
        "initial_batch_size": policy.initial_batch_size,
        "final_successful_batch_size": final_size,
        "minimum_successful_batch_size": minimum_size,
        "maximum_successful_batch_size": maximum_size,
        "snapshots_observed": len(snapshots),
        "peak_current_allocated_bytes": peak_current,
        "peak_driver_allocated_bytes": peak_driver,
        "recommended_max_bytes_minimum": recommended_minimum,
        "attempt_fingerprints": [attempt.attempt_fingerprint for attempt in attempts],
    }
    return MemoryAwareImageBatchMetrics(
        schema_version=MEMORY_AWARE_IMAGE_BATCH_METRICS_VERSION,
        policy_fingerprint=str(policy.policy_fingerprint),
        telemetry_status=telemetry_status,
        images_encoded=images_encoded,
        successful_batches=len(successful_sizes),
        batch_attempts=len(attempts),
        memory_retries=memory_retries,
        memory_limited_attempts=memory_limited_attempts,
        initial_batch_size=policy.initial_batch_size,
        final_successful_batch_size=final_size,
        minimum_successful_batch_size=minimum_size,
        maximum_successful_batch_size=maximum_size,
        snapshots_observed=len(snapshots),
        peak_current_allocated_bytes=peak_current,
        peak_driver_allocated_bytes=peak_driver,
        recommended_max_bytes_minimum=recommended_minimum,
        attempts=attempts,
        metrics_fingerprint=canonical_semantic_fingerprint(base),
    )


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


__all__ = [
    "IMAGE_BATCH_ATTEMPT_VERSION",
    "MEMORY_AWARE_IMAGE_BATCH_METRICS_VERSION",
    "MEMORY_AWARE_IMAGE_BATCH_POLICY_VERSION",
    "MPS_MEMORY_SNAPSHOT_VERSION",
    "ImageBatchAttempt",
    "MemoryAwareImageBatchMetrics",
    "MemoryAwareImageBatchPolicy",
    "MemoryAwareImageEncodingResult",
    "MpsMemorySnapshot",
    "encode_images_memory_aware",
    "is_image_memory_error",
]
