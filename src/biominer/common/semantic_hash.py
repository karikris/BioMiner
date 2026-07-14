"""Canonical binary preimages for structured semantic fingerprints.

The encoding is deliberately smaller than a general serialization format. It
accepts the JSON-like values used by BioMiner identity contracts, plus aware
datetimes and dates, and gives every node an explicit type and byte length.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
import hashlib
import math
import struct


CANONICAL_SEMANTIC_ENCODING_VERSION = "biominer-canonical-semantic-v1"
_PREIMAGE_PREFIX = b"biominer-canonical-semantic\x00v1\x00"
_LENGTH_SIZE_BYTES = 8


def canonical_semantic_bytes(value: object) -> bytes:
    """Return the versioned canonical binary preimage for ``value``.

    Mapping keys must be strings. Mapping order is ignored. Lists and tuples
    both represent an ordered JSON array so their order is preserved while
    their container implementation is ignored. Python floats are finite
    IEEE-754 binary64 values encoded little-endian, including the sign bit of
    zero.
    """

    return _PREIMAGE_PREFIX + _encode(value)


def canonical_semantic_fingerprint(value: object) -> str:
    """Return the full lowercase SHA-256 of a canonical semantic preimage."""

    return "sha256:" + hashlib.sha256(canonical_semantic_bytes(value)).hexdigest()


def _encode(value: object) -> bytes:
    if value is None:
        return _frame(b"N", b"")
    if isinstance(value, bool):
        return _frame(b"B", b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return _frame(b"I", str(value).encode("ascii"))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical semantic floats must be finite")
        return _frame(b"F", struct.pack("<d", value))
    if isinstance(value, str):
        return _frame(b"S", value.encode("utf-8"))
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical semantic datetimes must be timezone-aware")
        normalized = value.astimezone(UTC).isoformat(timespec="microseconds")
        return _frame(b"Z", normalized.encode("ascii"))
    if isinstance(value, date):
        return _frame(b"D", value.isoformat().encode("ascii"))
    if isinstance(value, Mapping):
        return _encode_mapping(value)
    if isinstance(value, (list, tuple)):
        return _encode_sequence(value)
    raise TypeError(f"canonical semantic values do not support {type(value).__name__}")


def _encode_mapping(value: Mapping[object, object]) -> bytes:
    items: list[tuple[bytes, object]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical semantic mapping keys must be strings")
        items.append((_encode(key), item))
    items.sort(key=lambda pair: pair[0])
    payload = bytearray(_length(len(items)))
    for encoded_key, item in items:
        payload.extend(encoded_key)
        payload.extend(_encode(item))
    return _frame(b"M", bytes(payload))


def _encode_sequence(value: list[object] | tuple[object, ...]) -> bytes:
    payload = bytearray(_length(len(value)))
    for item in value:
        payload.extend(_encode(item))
    return _frame(b"L", bytes(payload))


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + _length(len(payload)) + payload


def _length(value: int) -> bytes:
    return value.to_bytes(_LENGTH_SIZE_BYTES, "big", signed=False)


__all__ = [
    "CANONICAL_SEMANTIC_ENCODING_VERSION",
    "canonical_semantic_bytes",
    "canonical_semantic_fingerprint",
]
