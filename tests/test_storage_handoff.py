from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from biominer.storage.handoff import (
    build_handoff_bundle,
    extract_handoff_bundle,
    receive_handoff_bundle,
    upload_handoff_bundle,
    verify_handoff_bundle,
)


GIT_SHA = "0561906d994d6b9e56e0b6405fdb68272759595f"


def test_handoff_bundle_is_deterministic_and_locally_verified(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "runs/pilot/checkpoints").mkdir(parents=True)
    (root / "runs/pilot/checkpoints/state.json").write_text(
        '{"complete":true}\n', encoding="utf-8"
    )
    (root / "config").mkdir()
    (root / "config/pilot.json").write_text('{"phase":14}\n', encoding="utf-8")

    first = build_handoff_bundle(
        root=root,
        sources=("runs/pilot", "config/pilot.json"),
        output_dir=tmp_path / "first",
        name="papilio-phase14",
        source_git_sha=GIT_SHA,
    )
    os.utime(root / "config/pilot.json", (2_000_000_000, 2_000_000_000))
    second = build_handoff_bundle(
        root=root,
        sources=("config/pilot.json", "runs/pilot"),
        output_dir=tmp_path / "second",
        name="papilio-phase14",
        source_git_sha=GIT_SHA,
    )

    assert first.sha256 == second.sha256
    assert first.archive_path.name == second.archive_path.name
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.file_count == 2
    assert first.source_byte_count > 0
    verification = verify_handoff_bundle(
        first.archive_path, expected_sha256=first.sha256
    )
    assert verification.file_count == 2
    assert verification.sha256 == first.sha256
    assert verification.source_git_sha == GIT_SHA


def test_handoff_bundle_rejects_symlinks_and_sources_outside_root(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        build_handoff_bundle(
            root=root,
            sources=("link.txt",),
            output_dir=tmp_path / "output",
            name="handoff",
            source_git_sha=GIT_SHA,
        )
    with pytest.raises(ValueError, match="outside root"):
        build_handoff_bundle(
            root=root,
            sources=(outside,),
            output_dir=tmp_path / "output",
            name="handoff",
            source_git_sha=GIT_SHA,
        )


def test_handoff_verification_rejects_wrong_content_address(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "artifact.txt").write_text("payload\n", encoding="utf-8")
    bundle = build_handoff_bundle(
        root=root,
        sources=("artifact.txt",),
        output_dir=tmp_path / "output",
        name="handoff",
        source_git_sha=GIT_SHA,
    )

    wrong = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify_handoff_bundle(bundle.archive_path, expected_sha256=wrong)


def test_handoff_extracts_atomically_and_verifies_existing_files(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "runs/pilot").mkdir(parents=True)
    (root / "runs/pilot/a.txt").write_text("a\n", encoding="utf-8")
    (root / "runs/pilot/b.txt").write_text("b\n", encoding="utf-8")
    bundle = build_handoff_bundle(
        root=root,
        sources=("runs/pilot",),
        output_dir=tmp_path / "output",
        name="handoff",
        source_git_sha=GIT_SHA,
    )
    destination = tmp_path / "received"
    (destination / "runs/pilot").mkdir(parents=True)
    (destination / "runs/pilot/a.txt").write_text("a\n", encoding="utf-8")

    result = extract_handoff_bundle(
        bundle.archive_path,
        destination=destination,
        expected_sha256=bundle.sha256,
    )

    assert result["verified_existing_file_count"] == 1
    assert result["extracted_file_count"] == 1
    assert (destination / "runs/pilot/b.txt").read_text(encoding="utf-8") == "b\n"
    assert not list(destination.rglob("*.tmp"))
    (destination / "runs/pilot/a.txt").write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs from handoff"):
        extract_handoff_bundle(
            bundle.archive_path,
            destination=destination,
            expected_sha256=bundle.sha256,
        )


def test_handoff_upload_uses_one_content_addressed_write_and_local_receipt(
    tmp_path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"payload")
    bundle = build_handoff_bundle(
        root=root,
        sources=("artifact.bin",),
        output_dir=tmp_path / "output",
        name="handoff",
        source_git_sha=GIT_SHA,
    )
    storage = _UploadOnlyStorage()
    receipt_path = tmp_path / "receipt.json"

    receipt = upload_handoff_bundle(
        storage=storage,
        archive=bundle.archive_path,
        expected_sha256=bundle.sha256,
        destination_prefix="s3://biominer/biominer/handoffs",
        receipt_path=receipt_path,
    )

    assert len(storage.calls) == 1
    uri, source, expected_sha256 = storage.calls[0]
    assert uri.endswith(bundle.archive_path.name)
    assert source == bundle.archive_path
    assert expected_sha256 == bundle.sha256
    assert receipt["status"] == "remote_write_acknowledged"
    assert receipt["remote_integrity"] == "not_read_back"
    assert receipt["remote_operation_contract"] == {
        "content_addressed_object_streams_opened": 1,
        "explicit_head_requests": 0,
        "explicit_list_requests": 0,
        "remote_readback_requests": 0,
        "remote_completion_marker_writes": 0,
    }
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_handoff_receive_downloads_once_then_reuses_verified_cache(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"payload")
    bundle = build_handoff_bundle(
        root=root,
        sources=("artifact.bin",),
        output_dir=tmp_path / "output",
        name="handoff",
        source_git_sha=GIT_SHA,
    )
    uri = f"s3://biominer/biominer/handoffs/{bundle.archive_path.name}"
    storage = _DownloadOnlyStorage(bundle.archive_path)
    cache_dir = tmp_path / "cache"
    destination = tmp_path / "received"

    first = receive_handoff_bundle(
        storage=storage,
        uri=uri,
        expected_sha256=bundle.sha256,
        cache_dir=cache_dir,
        destination=destination,
    )
    second = receive_handoff_bundle(
        storage=storage,
        uri=uri,
        expected_sha256=bundle.sha256,
        cache_dir=cache_dir,
        destination=destination,
    )

    assert storage.calls == 1
    assert first["remote_read_streams"] == 1
    assert second["remote_read_streams"] == 0
    assert second["cache_status"] == "verified_existing"
    assert (destination / "artifact.bin").read_bytes() == b"payload"


class _UploadOnlyStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str]] = []

    def write_content_addressed_file(
        self,
        uri: str,
        source: str | Path,
        *,
        expected_sha256: str,
        content_type: str | None = None,
    ) -> str:
        assert content_type == "application/gzip"
        self.calls.append((uri, Path(source), expected_sha256))
        return uri


class _DownloadOnlyStorage:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls = 0

    def materialize_content_addressed_file(
        self,
        uri: str,
        destination: str | Path,
        *,
        expected_sha256: str,
        overwrite: bool = False,
    ) -> str:
        _ = uri, expected_sha256, overwrite
        self.calls += 1
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, output)
        return str(output)
