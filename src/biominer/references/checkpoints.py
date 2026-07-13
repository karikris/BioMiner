from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import polars as pl

from biominer.references.schemas import (
    reference_media_candidates_frame,
    reference_observations_frame,
)
from biominer.references.source_base import ReferenceMetadataPage, ReferenceSourceQuery
from biominer.storage.parquet import write_parquet


REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION = "reference-page-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class ReferencePageCheckpoint:
    source: str
    source_version: str
    query_fingerprint: str
    source_snapshot_version: str
    next_cursor: str | None
    complete: bool
    page_count: int
    observation_count: int
    media_candidate_count: int
    checkpoint_directory: Path


def write_reference_page_checkpoint(
    query: ReferenceSourceQuery,
    page: ReferenceMetadataPage,
    output: str | Path,
) -> ReferencePageCheckpoint:
    if page.query_fingerprint != query.query_fingerprint:
        raise ValueError("reference checkpoint page does not match the source query")
    directory = _checkpoint_directory(page.source, query, output)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    state = _read_checkpoint_state(
        query,
        source=page.source,
        source_version=page.source_version,
        directory=directory,
        allow_missing=True,
    )
    if state is None:
        if any(directory.iterdir()):
            raise ValueError("reference checkpoint directory is partial: state.json is missing")
        pages: list[dict[str, object]] = []
        expected_cursor = page.page_cursor
    else:
        if bool(state["complete"]):
            raise ValueError("reference checkpoint is already complete")
        pages = [dict(value) for value in _page_entries(state)]
        expected_cursor = str(state["next_cursor"])
    if page.page_cursor != expected_cursor:
        raise ValueError(
            f"reference checkpoint expected page cursor {expected_cursor}, "
            f"got {page.page_cursor}"
        )
    page_index = len(pages)
    prefix = f"page-{page_index:09d}"
    observations_name = f"{prefix}-observations.parquet"
    media_name = f"{prefix}-media.parquet"
    observations_path = write_parquet(page.observations, directory / observations_name)
    media_path = write_parquet(page.media_candidates, directory / media_name)
    entry = {
        "page_cursor": page.page_cursor,
        "next_cursor": page.next_cursor,
        "complete": page.complete,
        "observation_file": observations_name,
        "observation_rows": page.observations.height,
        "observation_sha256": _file_sha256(observations_path),
        "media_file": media_name,
        "media_rows": page.media_candidates.height,
        "media_sha256": _file_sha256(media_path),
        "request_count": page.request_count,
        "retry_count": page.retry_count,
        "rate_limit_count": page.rate_limit_count,
    }
    pages.append(entry)
    new_state = {
        "schema_version": REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION,
        "source": page.source,
        "source_version": page.source_version,
        "query_fingerprint": query.query_fingerprint,
        "source_snapshot_version": query.source_snapshot_version,
        "next_cursor": page.next_cursor,
        "complete": page.complete,
        "pages": pages,
    }
    _write_json_atomic(new_state, state_path)
    checkpoint = load_reference_page_checkpoint(
        query,
        source=page.source,
        source_version=page.source_version,
        output=output,
    )
    assert checkpoint is not None
    return checkpoint


def load_reference_page_checkpoint(
    query: ReferenceSourceQuery,
    *,
    source: str,
    source_version: str,
    output: str | Path,
) -> ReferencePageCheckpoint | None:
    directory = _checkpoint_directory(source, query, output)
    if not directory.exists():
        return None
    state = _read_checkpoint_state(
        query,
        source=source,
        source_version=source_version,
        directory=directory,
        allow_missing=False,
    )
    assert state is not None
    pages = _page_entries(state)
    expected_cursor = pages[0].get("page_cursor") if pages else None
    observation_count = 0
    media_count = 0
    for index, entry in enumerate(pages):
        cursor = entry.get("page_cursor")
        if cursor != expected_cursor:
            raise ValueError(f"reference checkpoint page {index} breaks the cursor chain")
        observation_path = _checkpoint_file(directory, entry, "observation_file")
        media_path = _checkpoint_file(directory, entry, "media_file")
        _verify_checkpoint_file(
            observation_path,
            expected_hash=entry.get("observation_sha256"),
        )
        _verify_checkpoint_file(media_path, expected_hash=entry.get("media_sha256"))
        observation_rows = _nonnegative_int(
            entry.get("observation_rows"),
            field="observation_rows",
        )
        media_rows = _nonnegative_int(entry.get("media_rows"), field="media_rows")
        if _parquet_row_count(observation_path) != observation_rows:
            raise ValueError("reference checkpoint observation row count mismatch")
        if _parquet_row_count(media_path) != media_rows:
            raise ValueError("reference checkpoint media row count mismatch")
        observation_count += observation_rows
        media_count += media_rows
        complete = bool(entry.get("complete"))
        next_cursor = entry.get("next_cursor")
        if complete:
            if next_cursor is not None or index != len(pages) - 1:
                raise ValueError("reference checkpoint has a non-terminal complete page")
            expected_cursor = None
        else:
            expected_cursor = str(next_cursor or "")
            if not expected_cursor:
                raise ValueError("reference checkpoint incomplete page lacks a next cursor")
    complete = bool(state.get("complete"))
    state_next_cursor = state.get("next_cursor")
    if complete:
        if not pages or state_next_cursor is not None or expected_cursor is not None:
            raise ValueError("reference checkpoint complete state is inconsistent")
        resume_cursor = None
    else:
        resume_cursor = str(state_next_cursor or "")
        if not pages or resume_cursor != expected_cursor:
            raise ValueError("reference checkpoint state cursor does not match its final page")
    return ReferencePageCheckpoint(
        source=source,
        source_version=source_version,
        query_fingerprint=query.query_fingerprint,
        source_snapshot_version=query.source_snapshot_version,
        next_cursor=resume_cursor,
        complete=complete,
        page_count=len(pages),
        observation_count=observation_count,
        media_candidate_count=media_count,
        checkpoint_directory=directory,
    )


def load_reference_page_checkpoint_frames(
    query: ReferenceSourceQuery,
    *,
    source: str,
    source_version: str,
    output: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    checkpoint = load_reference_page_checkpoint(
        query,
        source=source,
        source_version=source_version,
        output=output,
    )
    if checkpoint is None:
        return reference_observations_frame([]), reference_media_candidates_frame([])
    state = _read_checkpoint_state(
        query,
        source=source,
        source_version=source_version,
        directory=checkpoint.checkpoint_directory,
    )
    assert state is not None
    observation_rows: list[dict[str, object]] = []
    media_rows: list[dict[str, object]] = []
    for entry in _page_entries(state):
        observation_rows.extend(
            pl.read_parquet(
                _checkpoint_file(
                    checkpoint.checkpoint_directory,
                    entry,
                    "observation_file",
                )
            ).iter_rows(named=True)
        )
        media_rows.extend(
            pl.read_parquet(
                _checkpoint_file(checkpoint.checkpoint_directory, entry, "media_file")
            ).iter_rows(named=True)
        )
    return (
        reference_observations_frame(observation_rows),
        reference_media_candidates_frame(media_rows),
    )


def _checkpoint_directory(
    source: str,
    query: ReferenceSourceQuery,
    output: str | Path,
) -> Path:
    source_name = "".join(
        character if character.isalnum() else "-" for character in source.casefold()
    ).strip("-")
    if not source_name:
        raise ValueError("reference checkpoint source must be nonblank")
    digest = query.query_fingerprint.removeprefix("sha256:")
    return Path(output) / f"{source_name}-{digest}"


def _read_checkpoint_state(
    query: ReferenceSourceQuery,
    *,
    source: str,
    source_version: str,
    directory: Path,
    allow_missing: bool = False,
) -> dict[str, object] | None:
    state_path = directory / "state.json"
    if not state_path.exists():
        if allow_missing:
            return None
        raise ValueError("reference checkpoint directory is partial: state.json is missing")
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("reference checkpoint state is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("reference checkpoint state must be a JSON object")
    expected = {
        "schema_version": REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION,
        "source": source,
        "source_version": source_version,
        "query_fingerprint": query.query_fingerprint,
        "source_snapshot_version": query.source_snapshot_version,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"reference checkpoint {field} is incompatible")
    _page_entries(payload)
    return payload


def _page_entries(state: Mapping[str, object]) -> list[Mapping[str, object]]:
    pages = state.get("pages")
    if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
        raise ValueError("reference checkpoint pages must be an array of objects")
    return pages


def _checkpoint_file(
    directory: Path,
    entry: Mapping[str, object],
    field: str,
) -> Path:
    name = _required_text(entry.get(field), field=field)
    path = directory / name
    if path.parent != directory or path.name != name:
        raise ValueError("reference checkpoint contains an unsafe artifact path")
    return path


def _verify_checkpoint_file(path: Path, *, expected_hash: object) -> None:
    if not path.is_file():
        raise ValueError(f"reference checkpoint artifact is missing: {path.name}")
    if _file_sha256(path) != _required_sha256(expected_hash, field=f"{path.name} sha256"):
        raise ValueError(f"reference checkpoint artifact checksum mismatch: {path.name}")


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    try:
        return int(parquet_file.metadata.num_rows)
    finally:
        parquet_file.close()


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _optional_int(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: object, *, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return parsed


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _required_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a full sha256 digest")
    return text


__all__ = [
    "REFERENCE_PAGE_CHECKPOINT_SCHEMA_VERSION",
    "ReferencePageCheckpoint",
    "load_reference_page_checkpoint",
    "load_reference_page_checkpoint_frames",
    "write_reference_page_checkpoint",
]
