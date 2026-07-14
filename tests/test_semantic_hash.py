from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
import struct

import pytest

from biominer.common.semantic_hash import (
    canonical_semantic_bytes,
    canonical_semantic_fingerprint,
)


def test_float64_preimage_has_exact_little_endian_golden_vector() -> None:
    expected = bytes.fromhex(
        "62696f6d696e65722d63616e6f6e6963616c2d73656d616e74696300763100"
        "46"
        "0000000000000008"
        "000000000000f03f"
    )

    assert canonical_semantic_bytes(1.0) == expected
    assert expected.endswith(struct.pack("<d", 1.0))
    assert canonical_semantic_fingerprint(1.0) == (
        "sha256:5bc8678513ea5f54f50ec4d5d4cf8484ae888123012c4292b6d4573067f556ea"
    )


def test_float64_signed_zero_is_preserved_in_preimage_and_fingerprint() -> None:
    positive = canonical_semantic_bytes(0.0)
    negative = canonical_semantic_bytes(-0.0)

    assert positive[-8:] == b"\x00\x00\x00\x00\x00\x00\x00\x00"
    assert negative[-8:] == b"\x00\x00\x00\x00\x00\x00\x00\x80"
    assert positive != negative
    assert canonical_semantic_fingerprint(0.0) != canonical_semantic_fingerprint(-0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_is_rejected_at_any_depth(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        canonical_semantic_bytes({"nested": [value]})


def test_mapping_and_sequence_canonicalization() -> None:
    assert canonical_semantic_bytes({"b": 2, "a": 1}) == canonical_semantic_bytes(
        {"a": 1, "b": 2}
    )
    assert canonical_semantic_bytes([1, 2]) != canonical_semantic_bytes([2, 1])
    assert canonical_semantic_bytes([1, 2]) == canonical_semantic_bytes((1, 2))


def test_type_tags_and_nested_length_frames_prevent_structural_collisions() -> None:
    scalar_and_empty_values = (None, False, 0, 0.0, "", [])

    assert len(
        {canonical_semantic_fingerprint(value) for value in scalar_and_empty_values}
    ) == len(scalar_and_empty_values)
    assert canonical_semantic_bytes(["ab", "c"]) != canonical_semantic_bytes(
        ["a", "bc"]
    )


def test_dates_and_aware_datetimes_have_canonical_typed_encodings() -> None:
    utc_value = datetime(2026, 7, 14, 1, 2, 3, 456, tzinfo=UTC)
    offset_value = utc_value.astimezone(timezone(timedelta(hours=10)))

    assert canonical_semantic_bytes(utc_value) == canonical_semantic_bytes(offset_value)
    assert canonical_semantic_bytes(utc_value) != canonical_semantic_bytes(
        date(2026, 7, 14)
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_semantic_bytes(utc_value.replace(tzinfo=None))


def test_unsupported_values_and_non_string_mapping_keys_fail_closed() -> None:
    with pytest.raises(TypeError, match="do not support bytes"):
        canonical_semantic_bytes(b"not-json-like")
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_semantic_bytes({1: "not-a-string-key"})
