"""Deterministic, non-executable numeric array archives for ML artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import stat
from typing import Any
import zipfile


DEFAULT_MAX_NUMPY_HEADER_BYTES = 4_096


def deterministic_numeric_npz(arrays: Mapping[str, Any]) -> bytes:
    """Serialize a validated numeric array mapping as deterministic stored NPY files."""

    np = _load_numpy()
    normalized: dict[str, Any] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise ValueError("numeric array names must be non-empty identifiers")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "fiu":
            raise ValueError(f"numeric array {name} must contain only numeric values")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise ValueError(f"numeric array {name} contains non-finite values")
        normalized[name] = np.ascontiguousarray(array)

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(normalized):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                normalized[name],
                version=(2, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def numeric_array_manifest_entry(array: Any) -> dict[str, object]:
    """Return dtype, shape, size, and raw-byte identity for one numeric array."""

    np = _load_numpy()
    value = np.ascontiguousarray(array)
    if value.dtype.hasobject or value.dtype.kind not in "fiu":
        raise ValueError("array manifest entries require numeric arrays")
    if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
        raise ValueError("array manifest entries require finite arrays")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "size_bytes": int(value.nbytes),
        "raw_sha256": bytes_sha256(value.tobytes(order="C")),
    }


def load_numeric_npz(
    value: bytes,
    *,
    specs: Mapping[str, Mapping[str, object]],
    expected_dtypes: Mapping[str, str],
    expected_shapes: Mapping[str, tuple[int, ...]],
    max_uncompressed_bytes: int,
    artifact_label: str,
    max_header_bytes: int = DEFAULT_MAX_NUMPY_HEADER_BYTES,
) -> dict[str, Any]:
    """Load an exact, bounded numeric NPZ after validating its ZIP and NPY layers."""

    expected_names = set(specs)
    if set(expected_dtypes) != expected_names or set(expected_shapes) != expected_names:
        raise ValueError(
            f"{artifact_label} numeric array expectations are inconsistent"
        )
    expected_members = tuple(sorted(f"{name}.npy" for name in expected_names))
    try:
        with zipfile.ZipFile(io.BytesIO(value), mode="r") as archive:
            infos = archive.infolist()
            member_names = tuple(info.filename for info in infos)
            if member_names != expected_members or len(set(member_names)) != len(infos):
                raise ValueError(
                    f"{artifact_label} archive members do not match the manifest"
                )
            total_size = 0
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != info.compress_size
                ):
                    raise ValueError(
                        f"{artifact_label} archive members are not safe numeric files"
                    )
                total_size += info.file_size
            if total_size > max_uncompressed_bytes:
                raise ValueError(
                    f"{artifact_label} archive expands beyond the size limit"
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid {artifact_label} array archive") from exc

    np = _load_numpy()
    arrays: dict[str, Any] = {}
    try:
        with np.load(
            io.BytesIO(value),
            allow_pickle=False,
            max_header_size=max_header_bytes,
        ) as loaded:
            if set(loaded.files) != expected_names:
                raise ValueError(
                    f"{artifact_label} archive members do not match the manifest"
                )
            for name in sorted(expected_names):
                arrays[name] = np.array(loaded[name], copy=True, order="C")
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and "archive members" in str(exc):
            raise
        raise ValueError(f"invalid {artifact_label} array archive") from exc

    for name, array in arrays.items():
        spec = specs[name]
        if array.dtype.hasobject or array.dtype.kind not in "fiu":
            raise ValueError(f"{artifact_label} array {name} is not numeric")
        if array.dtype.str != expected_dtypes[name]:
            raise ValueError(f"{artifact_label} array {name} dtype is invalid")
        if tuple(array.shape) != expected_shapes[name]:
            raise ValueError(f"{artifact_label} array {name} shape is invalid")
        if array.dtype.str != spec.get("dtype") or list(array.shape) != spec.get(
            "shape"
        ):
            raise ValueError(
                f"{artifact_label} array {name} does not match its manifest"
            )
        if int(array.nbytes) != spec.get("size_bytes"):
            raise ValueError(f"{artifact_label} array {name} byte size is invalid")
        if bytes_sha256(array.tobytes(order="C")) != spec.get("raw_sha256"):
            raise ValueError(f"{artifact_label} array {name} checksum is invalid")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise ValueError(
                f"{artifact_label} array {name} contains non-finite values"
            )
    return arrays


def bytes_sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("NumPy is required for ML artifact persistence") from exc
    return np


__all__ = [
    "DEFAULT_MAX_NUMPY_HEADER_BYTES",
    "bytes_sha256",
    "deterministic_numeric_npz",
    "load_numeric_npz",
    "numeric_array_manifest_entry",
]
