"""Tests for deterministic MPS-aware image batching and bounded retries."""

from __future__ import annotations

import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.vision.memory_aware_batching import (
    MemoryAwareImageBatchPolicy,
    MpsMemorySnapshot,
    encode_images_memory_aware,
    is_image_memory_error,
)


def test_mps_headroom_limits_each_uncommitted_image_batch() -> None:
    encoded_batch_sizes: list[int] = []
    snapshot = MpsMemorySnapshot(
        current_allocated_bytes=400,
        driver_allocated_bytes=500,
        recommended_max_bytes=1000,
    )
    policy = MemoryAwareImageBatchPolicy(
        initial_batch_size=8,
        minimum_batch_size=1,
        maximum_batch_size=8,
        target_driver_memory_fraction=0.75,
        estimated_incremental_bytes_per_image=100,
    )

    result = encode_images_memory_aware(
        _images(5),
        encode_batch=lambda batch: _encode(batch, encoded_batch_sizes),
        policy=policy,
        memory_probe=lambda: snapshot,
    )

    assert encoded_batch_sizes == [2, 2, 1]
    assert [vector[0] for vector in result.vectors] == pytest.approx([1, 2, 3, 4, 5])
    metrics = result.metrics
    assert metrics.telemetry_status == "mps_memory_available"
    assert metrics.images_encoded == 5
    assert metrics.successful_batches == 3
    assert metrics.batch_attempts == 3
    assert metrics.memory_retries == 0
    assert metrics.memory_limited_attempts == 2
    assert metrics.initial_batch_size == 8
    assert metrics.final_successful_batch_size == 1
    assert metrics.minimum_successful_batch_size == 1
    assert metrics.maximum_successful_batch_size == 2
    assert metrics.snapshots_observed == 6
    assert metrics.peak_current_allocated_bytes == 400
    assert metrics.peak_driver_allocated_bytes == 500
    assert metrics.recommended_max_bytes_minimum == 1000
    assert all(
        attempt.attempt_fingerprint.startswith("sha256:")
        for attempt in metrics.attempts
    )
    assert metrics.metrics_fingerprint.startswith("sha256:")


def test_missing_memory_probe_uses_labeled_fixed_batches() -> None:
    encoded_batch_sizes: list[int] = []

    result = encode_images_memory_aware(
        _images(5),
        encode_batch=lambda batch: _encode(batch, encoded_batch_sizes),
        policy=MemoryAwareImageBatchPolicy(
            initial_batch_size=3,
            minimum_batch_size=1,
            maximum_batch_size=3,
        ),
    )

    assert encoded_batch_sizes == [3, 2]
    assert result.metrics.telemetry_status == "not_available_fixed_batch"
    assert result.metrics.snapshots_observed == 0
    assert result.metrics.peak_driver_allocated_bytes is None
    assert {attempt.decision_reason for attempt in result.metrics.attempts} == {
        "configured_no_memory_probe"
    }


def test_memory_error_retries_only_failed_slice_at_half_size() -> None:
    attempts: list[tuple[int, ...]] = []
    failed = False

    def encode(batch):  # noqa: ANN001, ANN202 - deterministic fake encoder.
        nonlocal failed
        values = tuple(image.data[0] for image in batch)
        attempts.append(values)
        if len(batch) == 4 and not failed:
            failed = True
            raise RuntimeError("MPS memory allocation failed")
        return [[float(value), 1.0] for value in values]

    result = encode_images_memory_aware(
        _images(5),
        encode_batch=encode,
        policy=MemoryAwareImageBatchPolicy(
            initial_batch_size=4,
            minimum_batch_size=1,
            maximum_batch_size=4,
        ),
    )

    assert attempts == [(1, 2, 3, 4), (1, 2), (3, 4), (5,)]
    assert [vector[0] for vector in result.vectors] == pytest.approx([1, 2, 3, 4, 5])
    assert result.metrics.memory_retries == 1
    assert result.metrics.batch_attempts == 4
    assert result.metrics.successful_batches == 3
    assert result.metrics.attempts[0].outcome == "memory_error_retry"
    assert [attempt.start_index for attempt in result.metrics.attempts] == [0, 0, 2, 4]


def test_minimum_batch_memory_error_and_nonmemory_error_propagate() -> None:
    def fail_memory(_batch):  # noqa: ANN001, ANN202 - fake encoder.
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        encode_images_memory_aware(
            _images(1),
            encode_batch=fail_memory,
            policy=MemoryAwareImageBatchPolicy(
                initial_batch_size=1,
                minimum_batch_size=1,
                maximum_batch_size=1,
            ),
        )

    def fail_other(_batch):  # noqa: ANN001, ANN202 - fake encoder.
        raise RuntimeError("decoder contract failed")

    with pytest.raises(RuntimeError, match="decoder contract failed"):
        encode_images_memory_aware(
            _images(2),
            encode_batch=fail_other,
            policy=MemoryAwareImageBatchPolicy(
                initial_batch_size=2,
                minimum_batch_size=1,
                maximum_batch_size=2,
            ),
        )
    assert is_image_memory_error(RuntimeError("CUDA out of memory")) is True
    assert is_image_memory_error(ValueError("out of memory")) is False


def test_memory_policy_and_snapshot_fail_closed() -> None:
    with pytest.raises(ValueError, match="minimum <= initial <= maximum"):
        MemoryAwareImageBatchPolicy(
            initial_batch_size=4,
            minimum_batch_size=5,
            maximum_batch_size=8,
        )
    with pytest.raises(ValueError, match="cannot exceed driver"):
        MpsMemorySnapshot(
            current_allocated_bytes=600,
            driver_allocated_bytes=500,
            recommended_max_bytes=1000,
        )
    policy = MemoryAwareImageBatchPolicy()
    with pytest.raises(ValueError, match="policy fingerprint mismatch"):
        MemoryAwareImageBatchPolicy(
            policy_fingerprint="sha256:" + "f" * 64,
        )
    assert policy.policy_fingerprint.startswith("sha256:")


def _images(count: int) -> tuple[DecodedImage, ...]:
    return tuple(
        DecodedImage(
            width=1,
            height=1,
            mode="RGB",
            data=bytes([index, index, index]),
            source_uri=f"memory://image-{index}",
        )
        for index in range(1, count + 1)
    )


def _encode(
    batch: tuple[DecodedImage, ...],
    observed_sizes: list[int],
) -> list[list[float]]:
    observed_sizes.append(len(batch))
    return [[float(image.data[0]), 1.0] for image in batch]
