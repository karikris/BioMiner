from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import gzip
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
import tarfile
from typing import Any
from urllib.parse import urlparse

from biominer.storage.cloud import CloudStorage
from biominer.storage.content_address import (
    normalize_sha256,
    sha256_file,
    sha256_stream,
)
from biominer.storage.uri import is_s3_uri, join_uri


HANDOFF_INVENTORY_SCHEMA_VERSION = "storage-handoff-inventory-v1.0.0"
HANDOFF_UPLOAD_RECEIPT_SCHEMA_VERSION = "storage-handoff-upload-receipt-v1.0.0"
HANDOFF_RECEIVE_RECEIPT_SCHEMA_VERSION = "storage-handoff-receive-receipt-v1.0.0"
INTERNAL_INVENTORY_PATH = ".biominer-handoff/inventory.json"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ARCHIVE_DIGEST_PATTERN = re.compile(
    r"\.sha256-([0-9a-f]{64})\.tar\.gz$"
)


@dataclass(frozen=True)
class HandoffFile:
    relative_path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class HandoffBundle:
    archive_path: Path
    sha256: str
    byte_count: int
    file_count: int
    source_byte_count: int
    source_git_sha: str
    source_roots: tuple[str, ...]


@dataclass(frozen=True)
class HandoffVerification:
    archive_path: Path
    sha256: str
    byte_count: int
    file_count: int
    source_byte_count: int
    source_git_sha: str
    source_roots: tuple[str, ...]
    files: tuple[HandoffFile, ...]


def build_handoff_bundle(
    *,
    root: str | Path,
    sources: tuple[str | Path, ...] | list[str | Path],
    output_dir: str | Path,
    name: str,
    source_git_sha: str,
) -> HandoffBundle:
    normalized_name = _validate_name(name)
    normalized_git_sha = _validate_git_sha(source_git_sha)
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    output_path = Path(output_dir).resolve()
    resolved_sources, source_roots = _resolve_sources(
        root=root_path,
        sources=sources,
    )
    for source in resolved_sources:
        if source.is_dir() and _is_relative_to(output_path, source):
            raise ValueError("handoff output directory must be outside source trees")
    files = _collect_files(root=root_path, sources=resolved_sources)
    if not files:
        raise ValueError("handoff sources contain no regular files")
    inventory_files = tuple(
        HandoffFile(
            relative_path=path.relative_to(root_path).as_posix(),
            byte_count=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    )
    inventory = _inventory_payload(
        name=normalized_name,
        source_git_sha=normalized_git_sha,
        source_roots=source_roots,
        files=inventory_files,
    )
    inventory_bytes = _canonical_json(inventory)
    output_path.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output_path,
            prefix=f".{normalized_name}.",
            suffix=".tar.gz.tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        _write_deterministic_archive(
            output=temporary_path,
            root=root_path,
            files=files,
            inventory_bytes=inventory_bytes,
        )
        digest = sha256_file(temporary_path)
        archive_path = output_path / (
            f"{normalized_name}.sha256-{digest.removeprefix('sha256:')}.tar.gz"
        )
        if archive_path.exists():
            if sha256_file(archive_path) != digest:
                raise FileExistsError(
                    f"content-addressed handoff path contains different bytes: {archive_path}"
                )
        else:
            temporary_path.replace(archive_path)
        temporary_path.unlink(missing_ok=True)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    verification = verify_handoff_bundle(archive_path, expected_sha256=digest)
    return HandoffBundle(
        archive_path=archive_path,
        sha256=verification.sha256,
        byte_count=verification.byte_count,
        file_count=verification.file_count,
        source_byte_count=verification.source_byte_count,
        source_git_sha=verification.source_git_sha,
        source_roots=verification.source_roots,
    )


def verify_handoff_bundle(
    archive: str | Path,
    *,
    expected_sha256: str | None = None,
) -> HandoffVerification:
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    actual_sha256 = sha256_file(archive_path)
    if expected_sha256 is not None:
        normalized_expected = normalize_sha256(expected_sha256)
        if actual_sha256 != normalized_expected:
            raise ValueError("handoff archive SHA-256 does not match expected value")
    filename_match = _ARCHIVE_DIGEST_PATTERN.search(archive_path.name)
    if filename_match is None:
        raise ValueError("handoff archive filename is not content-addressed")
    if actual_sha256.removeprefix("sha256:") != filename_match.group(1):
        raise ValueError("handoff archive SHA-256 does not match its filename")

    with tarfile.open(archive_path, mode="r:gz") as archive_file:
        members = archive_file.getmembers()
        by_name: dict[str, tarfile.TarInfo] = {}
        for member in members:
            _validate_archive_member(member)
            if member.name in by_name:
                raise ValueError(f"duplicate handoff archive member: {member.name}")
            by_name[member.name] = member
        inventory_member = by_name.pop(INTERNAL_INVENTORY_PATH, None)
        if inventory_member is None:
            raise ValueError("handoff archive is missing its embedded inventory")
        inventory_stream = archive_file.extractfile(inventory_member)
        if inventory_stream is None:
            raise ValueError("handoff inventory is not a regular file")
        inventory = json.loads(inventory_stream.read().decode("utf-8"))
        files, source_git_sha, source_roots = _validate_inventory(inventory)
        if set(by_name) != {item.relative_path for item in files}:
            raise ValueError("handoff inventory file set differs from archive members")
        for item in files:
            member = by_name[item.relative_path]
            if member.size != item.byte_count:
                raise ValueError(
                    f"handoff member byte count differs from inventory: {item.relative_path}"
                )
            stream = archive_file.extractfile(member)
            if stream is None or sha256_stream(stream) != item.sha256:
                raise ValueError(
                    f"handoff member SHA-256 differs from inventory: {item.relative_path}"
                )
    source_byte_count = sum(item.byte_count for item in files)
    if int(inventory["source_byte_count"]) != source_byte_count:
        raise ValueError("handoff inventory source byte count is inconsistent")
    return HandoffVerification(
        archive_path=archive_path,
        sha256=actual_sha256,
        byte_count=archive_path.stat().st_size,
        file_count=len(files),
        source_byte_count=source_byte_count,
        source_git_sha=source_git_sha,
        source_roots=source_roots,
        files=files,
    )


def extract_handoff_bundle(
    archive: str | Path,
    *,
    destination: str | Path,
    expected_sha256: str,
) -> dict[str, object]:
    verification = verify_handoff_bundle(
        archive,
        expected_sha256=expected_sha256,
    )
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    extracted = 0
    verified_existing = 0
    file_map = {item.relative_path: item for item in verification.files}
    with tarfile.open(verification.archive_path, mode="r:gz") as archive_file:
        for member in archive_file.getmembers():
            if member.name == INTERNAL_INVENTORY_PATH:
                continue
            item = file_map[member.name]
            target = destination_path.joinpath(*PurePosixPath(member.name).parts)
            _require_destination_containment(target, destination_path)
            if target.exists():
                if (
                    target.is_file()
                    and target.stat().st_size == item.byte_count
                    and sha256_file(target) == item.sha256
                ):
                    verified_existing += 1
                    continue
                raise FileExistsError(
                    f"existing destination differs from handoff: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive_file.extractfile(member)
            if stream is None:
                raise ValueError(f"handoff member is not a regular file: {member.name}")
            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    copyfileobj(stream, temporary)
                if (
                    temporary_path.stat().st_size != item.byte_count
                    or sha256_file(temporary_path) != item.sha256
                ):
                    raise OSError(
                        f"extracted handoff member failed local verification: {member.name}"
                    )
                try:
                    os.link(temporary_path, target)
                except FileExistsError as exc:
                    raise FileExistsError(target) from exc
                extracted += 1
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    return {
        "status": "locally_verified_and_extracted",
        "archive_path": str(verification.archive_path),
        "archive_sha256": verification.sha256,
        "file_count": verification.file_count,
        "source_byte_count": verification.source_byte_count,
        "extracted_file_count": extracted,
        "verified_existing_file_count": verified_existing,
        "destination": str(destination_path),
    }


def upload_handoff_bundle(
    *,
    storage: CloudStorage,
    archive: str | Path,
    expected_sha256: str,
    destination_prefix: str,
    receipt_path: str | Path | None = None,
) -> dict[str, object]:
    if not is_s3_uri(destination_prefix):
        raise ValueError("handoff upload destination must be an s3:// URI")
    verification = verify_handoff_bundle(
        archive,
        expected_sha256=expected_sha256,
    )
    uri = join_uri(destination_prefix, verification.archive_path.name)
    storage.write_content_addressed_file(
        uri,
        verification.archive_path,
        expected_sha256=verification.sha256,
        content_type="application/gzip",
    )
    receipt: dict[str, object] = {
        "schema_version": HANDOFF_UPLOAD_RECEIPT_SCHEMA_VERSION,
        "status": "remote_write_acknowledged",
        "created_at": datetime.now(UTC).isoformat(),
        "source_git_sha": verification.source_git_sha,
        "source_roots": list(verification.source_roots),
        "file_count": verification.file_count,
        "source_byte_count": verification.source_byte_count,
        "archive_byte_count": verification.byte_count,
        "archive_sha256": verification.sha256,
        "uri": uri,
        "local_integrity": "verified_before_upload",
        "remote_integrity": "not_read_back",
        "receiver_requirement": "verify_archive_sha256_and_embedded_inventory",
        "remote_operation_contract": {
            "content_addressed_object_streams_opened": 1,
            "explicit_head_requests": 0,
            "explicit_list_requests": 0,
            "remote_readback_requests": 0,
            "remote_completion_marker_writes": 0,
        },
    }
    if receipt_path is not None:
        _write_json_atomic(Path(receipt_path), receipt)
    return receipt


def receive_handoff_bundle(
    *,
    storage: CloudStorage,
    uri: str,
    expected_sha256: str,
    cache_dir: str | Path,
    destination: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, object]:
    if not is_s3_uri(uri):
        raise ValueError("handoff receive URI must use s3://")
    normalized_sha256 = normalize_sha256(expected_sha256)
    archive_name = Path(urlparse(uri).path).name
    if not archive_name:
        raise ValueError("handoff receive URI must identify an archive object")
    cache_path = Path(cache_dir) / archive_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not cache_path.is_file() or sha256_file(cache_path) != normalized_sha256:
            raise FileExistsError(
                f"cached handoff differs from expected SHA-256: {cache_path}"
            )
        cache_status = "verified_existing"
        remote_reads = 0
    else:
        storage.materialize_content_addressed_file(
            uri,
            cache_path,
            expected_sha256=normalized_sha256,
            overwrite=False,
        )
        cache_status = "downloaded_and_verified"
        remote_reads = 1
    extraction = extract_handoff_bundle(
        cache_path,
        destination=destination,
        expected_sha256=normalized_sha256,
    )
    receipt: dict[str, object] = {
        "schema_version": HANDOFF_RECEIVE_RECEIPT_SCHEMA_VERSION,
        "status": "received_and_locally_verified",
        "created_at": datetime.now(UTC).isoformat(),
        "uri": uri,
        "archive_sha256": normalized_sha256,
        "cache_path": str(cache_path),
        "cache_status": cache_status,
        "remote_read_streams": remote_reads,
        "explicit_head_requests": 0,
        "explicit_list_requests": 0,
        "extraction": extraction,
    }
    if receipt_path is not None:
        _write_json_atomic(Path(receipt_path), receipt)
    return receipt


def _resolve_sources(
    *,
    root: Path,
    sources: tuple[str | Path, ...] | list[str | Path],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    if not sources:
        raise ValueError("at least one handoff source is required")
    resolved: dict[str, Path] = {}
    for value in sources:
        candidate = Path(value)
        unresolved = candidate if candidate.is_absolute() else root / candidate
        if unresolved.is_symlink():
            raise ValueError(f"handoff source must not be a symlink: {value}")
        selected = unresolved.resolve()
        if not _is_relative_to(selected, root):
            raise ValueError(f"handoff source is outside root: {value}")
        if not selected.exists():
            raise FileNotFoundError(selected)
        relative = selected.relative_to(root).as_posix()
        resolved[relative] = selected
    roots = tuple(sorted(resolved))
    return tuple(resolved[item] for item in roots), roots


def _collect_files(*, root: Path, sources: tuple[Path, ...]) -> tuple[Path, ...]:
    collected: dict[str, Path] = {}
    for source in sources:
        candidates = (source,) if source.is_file() else tuple(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(
                    f"handoff source tree must not contain symlinks: {candidate}"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(
                    f"handoff source tree contains a non-regular file: {candidate}"
                )
            relative = candidate.relative_to(root).as_posix()
            _validate_member_name(relative)
            collected[relative] = candidate
    return tuple(collected[item] for item in sorted(collected))


def _inventory_payload(
    *,
    name: str,
    source_git_sha: str,
    source_roots: tuple[str, ...],
    files: tuple[HandoffFile, ...],
) -> dict[str, object]:
    return {
        "schema_version": HANDOFF_INVENTORY_SCHEMA_VERSION,
        "name": name,
        "source_git_sha": source_git_sha,
        "source_roots": list(source_roots),
        "file_count": len(files),
        "source_byte_count": sum(item.byte_count for item in files),
        "files": [
            {
                "relative_path": item.relative_path,
                "byte_count": item.byte_count,
                "sha256": item.sha256,
            }
            for item in files
        ],
    }


def _write_deterministic_archive(
    *,
    output: Path,
    root: Path,
    files: tuple[Path, ...],
    inventory_bytes: bytes,
) -> None:
    with output.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_stream,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as compressed_stream:
            with tarfile.open(
                fileobj=compressed_stream,
                mode="w|",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    info = _normalized_tar_info(relative, path.stat().st_size)
                    with path.open("rb") as source_stream:
                        archive.addfile(info, source_stream)
                inventory_info = _normalized_tar_info(
                    INTERNAL_INVENTORY_PATH,
                    len(inventory_bytes),
                )
                archive.addfile(inventory_info, BytesIO(inventory_bytes))


def _normalized_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    return info


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    _validate_member_name(member.name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"unsupported handoff archive member: {member.name}")
    if member.size < 0:
        raise ValueError(f"invalid handoff archive member size: {member.name}")


def _validate_member_name(name: str) -> None:
    if "\\" in name:
        raise ValueError(f"unsafe handoff archive path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"unsafe handoff archive path: {name}")


def _validate_inventory(
    payload: Any,
) -> tuple[tuple[HandoffFile, ...], str, tuple[str, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("handoff inventory must be a JSON object")
    if payload.get("schema_version") != HANDOFF_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported handoff inventory schema version")
    source_git_sha = _validate_git_sha(str(payload.get("source_git_sha") or ""))
    raw_roots = payload.get("source_roots")
    if not isinstance(raw_roots, list) or not all(
        isinstance(item, str) for item in raw_roots
    ):
        raise ValueError("handoff inventory source_roots must be strings")
    source_roots = tuple(raw_roots)
    if source_roots != tuple(sorted(set(source_roots))):
        raise ValueError("handoff inventory source_roots are not deterministic")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("handoff inventory files must be a list")
    files: list[HandoffFile] = []
    for row in raw_files:
        if not isinstance(row, dict):
            raise ValueError("handoff inventory file rows must be objects")
        relative_path = str(row.get("relative_path") or "")
        _validate_member_name(relative_path)
        byte_count = int(row.get("byte_count", -1))
        if byte_count < 0:
            raise ValueError("handoff inventory file byte_count must be non-negative")
        files.append(
            HandoffFile(
                relative_path=relative_path,
                byte_count=byte_count,
                sha256=normalize_sha256(str(row.get("sha256") or "")),
            )
        )
    paths = tuple(item.relative_path for item in files)
    if paths != tuple(sorted(set(paths))):
        raise ValueError("handoff inventory files are not unique and deterministic")
    if int(payload.get("file_count", -1)) != len(files):
        raise ValueError("handoff inventory file_count is inconsistent")
    return tuple(files), source_git_sha, source_roots


def _canonical_json(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(_canonical_json(payload))
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_name(name: str) -> str:
    normalized = str(name).strip()
    if _NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "handoff name must contain only letters, digits, dot, underscore, or dash"
        )
    return normalized


def _validate_git_sha(value: str) -> str:
    normalized = str(value).strip().casefold()
    if _GIT_SHA_PATTERN.fullmatch(normalized) is None:
        raise ValueError("source_git_sha must be a full 40-character Git SHA")
    return normalized


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_destination_containment(target: Path, destination: Path) -> None:
    resolved_target = target.resolve(strict=False)
    if not _is_relative_to(resolved_target, destination):
        raise ValueError(f"handoff extraction escaped destination: {target}")


__all__ = [
    "HANDOFF_INVENTORY_SCHEMA_VERSION",
    "HANDOFF_RECEIVE_RECEIPT_SCHEMA_VERSION",
    "HANDOFF_UPLOAD_RECEIPT_SCHEMA_VERSION",
    "HandoffBundle",
    "HandoffFile",
    "HandoffVerification",
    "build_handoff_bundle",
    "extract_handoff_bundle",
    "receive_handoff_bundle",
    "upload_handoff_bundle",
    "verify_handoff_bundle",
]
