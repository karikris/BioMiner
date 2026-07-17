"""Content-addressed work planning after a reference-bank revision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


INCREMENTAL_FEATURE_REUSE_PLAN_FILE = "incremental_feature_reuse_plan.parquet"
FEATURE_REUSE_REQUEST_SCHEMA_VERSION = "feature-reuse-request-v1.0.0"
FEATURE_CACHE_ENTRY_SCHEMA_VERSION = "feature-cache-entry-v1.0.0"
INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION = (
    "incremental-feature-reuse-plan-v1.0.0"
)

DETECTOR_SCOPE = "yoloe_detection"
FLICKR_EMBEDDING_SCOPE = "flickr_embedding"
REFERENCE_EMBEDDING_SCOPE = "reference_embedding"
FEATURE_SCOPES = frozenset(
    {DETECTOR_SCOPE, FLICKR_EMBEDDING_SCOPE, REFERENCE_EMBEDDING_SCOPE}
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


FEATURE_REUSE_REQUEST_SCHEMA = {
    "schema_version": pl.String,
    "feature_scope": pl.String,
    "item_id": pl.String,
    "input_content_fingerprint": pl.String,
    "producer_fingerprint": pl.String,
    "preprocessing_fingerprint": pl.String,
    "required": pl.Boolean,
    "newly_admitted": pl.Boolean,
    "request_fingerprint": pl.String,
}

FEATURE_CACHE_ENTRY_SCHEMA = {
    "schema_version": pl.String,
    "cache_entry_id": pl.String,
    "feature_scope": pl.String,
    "input_content_fingerprint": pl.String,
    "producer_fingerprint": pl.String,
    "preprocessing_fingerprint": pl.String,
    "artifact_id": pl.String,
    "artifact_fingerprint": pl.String,
    "cache_entry_fingerprint": pl.String,
}


def incremental_feature_reuse_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "feature_scope": pl.String,
        "item_id": pl.String,
        "input_content_fingerprint": pl.String,
        "producer_fingerprint": pl.String,
        "preprocessing_fingerprint": pl.String,
        "required": pl.Boolean,
        "newly_admitted": pl.Boolean,
        "request_fingerprint": pl.String,
        "cache_hit": pl.Boolean,
        "cache_entry_id": pl.String,
        "reusable_artifact_id": pl.String,
        "reusable_artifact_fingerprint": pl.String,
        "action": pl.String,
        "reason": pl.String,
        "plan_fingerprint": pl.String,
    }


def feature_reuse_request_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", FEATURE_REUSE_REQUEST_SCHEMA_VERSION)
        row.setdefault("newly_admitted", False)
        row["request_fingerprint"] = ""
        payload = dict(row)
        payload.pop("request_fingerprint")
        row["request_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=FEATURE_REUSE_REQUEST_SCHEMA,
        orient="row",
        strict=True,
    ).sort("feature_scope", "item_id")
    validate_feature_reuse_requests(frame)
    return frame


def feature_cache_entry_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", FEATURE_CACHE_ENTRY_SCHEMA_VERSION)
        row["cache_entry_fingerprint"] = ""
        payload = dict(row)
        payload.pop("cache_entry_fingerprint")
        row["cache_entry_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=FEATURE_CACHE_ENTRY_SCHEMA,
        orient="row",
        strict=True,
    ).sort("feature_scope", "cache_entry_id")
    validate_feature_cache_entries(frame)
    return frame


def calculate_incremental_feature_reuse(
    requests: pl.DataFrame,
    cache_entries: pl.DataFrame,
) -> pl.DataFrame:
    """Choose reuse, filtering, or compute from exact content identities."""

    validate_feature_reuse_requests(requests)
    validate_feature_cache_entries(cache_entries)
    cache_by_key = {
        _content_key(row): row for row in cache_entries.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for request in requests.iter_rows(named=True):
        cached = cache_by_key.get(_content_key(request))
        action, reason = _reuse_action(request, cache_hit=cached is not None)
        row = {
            "schema_version": INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION,
            "feature_scope": request["feature_scope"],
            "item_id": request["item_id"],
            "input_content_fingerprint": request[
                "input_content_fingerprint"
            ],
            "producer_fingerprint": request["producer_fingerprint"],
            "preprocessing_fingerprint": request[
                "preprocessing_fingerprint"
            ],
            "required": request["required"],
            "newly_admitted": request["newly_admitted"],
            "request_fingerprint": request["request_fingerprint"],
            "cache_hit": cached is not None,
            "cache_entry_id": cached["cache_entry_id"] if cached else None,
            "reusable_artifact_id": cached["artifact_id"] if cached else None,
            "reusable_artifact_fingerprint": (
                cached["artifact_fingerprint"] if cached else None
            ),
            "action": action,
            "reason": reason,
            "plan_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("plan_fingerprint")
        row["plan_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    plan = pl.DataFrame(
        rows,
        schema=incremental_feature_reuse_plan_schema(),
        orient="row",
        strict=True,
    ).sort("feature_scope", "item_id")
    validate_incremental_feature_reuse_plan(plan)
    return plan


def incremental_feature_reuse_metrics(plan: pl.DataFrame) -> pl.DataFrame:
    """Return deterministic per-scope/action work and reuse counts."""

    validate_incremental_feature_reuse_plan(plan)
    return (
        plan.group_by("feature_scope", "action")
        .agg(
            pl.len().cast(pl.UInt64).alias("item_count"),
            pl.col("cache_hit").sum().cast(pl.UInt64).alias("cache_hit_count"),
        )
        .with_columns(
            (pl.col("item_count") - pl.col("cache_hit_count")).alias(
                "cache_miss_count"
            )
        )
        .sort("feature_scope", "action")
    )


def feature_item_ids_to_compute(
    plan: pl.DataFrame,
    *,
    feature_scope: str,
) -> tuple[str, ...]:
    """Return the exact item IDs whose producer must execute."""

    validate_incremental_feature_reuse_plan(plan)
    scope = _feature_scope(feature_scope)
    compute_actions = {
        DETECTOR_SCOPE: {"run_yoloe"},
        FLICKR_EMBEDDING_SCOPE: {"embed_flickr_image"},
        REFERENCE_EMBEDDING_SCOPE: {
            "embed_new_reference_image",
            "reembed_changed_reference_image",
        },
    }[scope]
    return tuple(
        plan.filter(
            (pl.col("feature_scope") == scope)
            & pl.col("action").is_in(sorted(compute_actions))
        )["item_id"].to_list()
    )


def validate_feature_reuse_requests(frame: pl.DataFrame) -> None:
    if frame.schema != FEATURE_REUSE_REQUEST_SCHEMA:
        raise ValueError("feature reuse request schema mismatch")
    _reject_nulls(frame, tuple(FEATURE_REUSE_REQUEST_SCHEMA))
    if not frame.equals(frame.sort("feature_scope", "item_id")):
        raise ValueError("feature reuse requests are not deterministically sorted")
    if frame.select("feature_scope", "item_id").unique().height != frame.height:
        raise ValueError("feature reuse requests repeat an item")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FEATURE_REUSE_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported feature reuse request version")
        scope = _feature_scope(row["feature_scope"])
        _required_text(row["item_id"], field="item_id")
        _content_key(row)
        if not row["required"] and scope != REFERENCE_EMBEDDING_SCOPE:
            raise ValueError("only reference embeddings may be filtered")
        if row["newly_admitted"] and (
            scope != REFERENCE_EMBEDDING_SCOPE or not row["required"]
        ):
            raise ValueError(
                "newly admitted items must be required reference embeddings"
            )
        _validate_row_fingerprint(
            row,
            fingerprint_field="request_fingerprint",
            label="feature reuse request",
        )


def validate_feature_cache_entries(frame: pl.DataFrame) -> None:
    if frame.schema != FEATURE_CACHE_ENTRY_SCHEMA:
        raise ValueError("feature cache entry schema mismatch")
    _reject_nulls(frame, tuple(FEATURE_CACHE_ENTRY_SCHEMA))
    if not frame.equals(frame.sort("feature_scope", "cache_entry_id")):
        raise ValueError("feature cache entries are not deterministically sorted")
    if frame["cache_entry_id"].n_unique() != frame.height:
        raise ValueError("feature cache entries repeat a cache entry ID")
    keys: set[tuple[str, str, str, str]] = set()
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FEATURE_CACHE_ENTRY_SCHEMA_VERSION:
            raise ValueError("unsupported feature cache entry version")
        _required_text(row["cache_entry_id"], field="cache_entry_id")
        key = _content_key(row)
        if key in keys:
            raise ValueError("feature cache repeats a content identity")
        keys.add(key)
        _required_text(row["artifact_id"], field="artifact_id")
        _sha256(row["artifact_fingerprint"], field="artifact_fingerprint")
        _validate_row_fingerprint(
            row,
            fingerprint_field="cache_entry_fingerprint",
            label="feature cache entry",
        )


def validate_incremental_feature_reuse_plan(frame: pl.DataFrame) -> None:
    if frame.schema != incremental_feature_reuse_plan_schema():
        raise ValueError("incremental feature reuse plan schema mismatch")
    _reject_nulls(
        frame,
        tuple(
            field
            for field in incremental_feature_reuse_plan_schema()
            if field
            not in {
                "cache_entry_id",
                "reusable_artifact_id",
                "reusable_artifact_fingerprint",
            }
        ),
    )
    if not frame.equals(frame.sort("feature_scope", "item_id")):
        raise ValueError("incremental feature reuse plan is not sorted")
    if frame.select("feature_scope", "item_id").unique().height != frame.height:
        raise ValueError("incremental feature reuse plan repeats an item")
    for row in frame.iter_rows(named=True):
        scope = _feature_scope(row["feature_scope"])
        if row["schema_version"] != INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported incremental feature reuse plan version")
        _content_key(row)
        expected_action, expected_reason = _reuse_action(
            row,
            cache_hit=bool(row["cache_hit"]),
        )
        if (row["action"], row["reason"]) != (
            expected_action,
            expected_reason,
        ):
            raise ValueError("incremental feature reuse action mismatch")
        if not row["required"] and scope != REFERENCE_EMBEDDING_SCOPE:
            raise ValueError("only reference embeddings may be filtered")
        cache_fields = (
            "cache_entry_id",
            "reusable_artifact_id",
            "reusable_artifact_fingerprint",
        )
        if row["cache_hit"] != all(row[field] is not None for field in cache_fields):
            raise ValueError("incremental feature reuse cache evidence mismatch")
        if row["cache_hit"]:
            _required_text(row["cache_entry_id"], field="cache_entry_id")
            _required_text(
                row["reusable_artifact_id"],
                field="reusable_artifact_id",
            )
            _sha256(
                row["reusable_artifact_fingerprint"],
                field="reusable_artifact_fingerprint",
            )
        _validate_row_fingerprint(
            row,
            fingerprint_field="plan_fingerprint",
            label="incremental feature reuse plan",
        )


def write_incremental_feature_reuse_plan(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_incremental_feature_reuse_plan(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= INCREMENTAL_FEATURE_REUSE_PLAN_FILE
    return write_parquet(frame, destination)


def _reuse_action(
    row: Mapping[str, object],
    *,
    cache_hit: bool,
) -> tuple[str, str]:
    scope = _feature_scope(row["feature_scope"])
    if scope == REFERENCE_EMBEDDING_SCOPE and not bool(row["required"]):
        return (
            ("filter_excluded_reference", "excluded_reference_cache_filtered")
            if cache_hit
            else ("skip_excluded_reference", "excluded_reference_not_required")
        )
    if cache_hit:
        return {
            DETECTOR_SCOPE: (
                "reuse_yoloe_detection",
                "detector_and_input_identity_unchanged",
            ),
            FLICKR_EMBEDDING_SCOPE: (
                "reuse_flickr_embedding",
                "image_model_and_preprocessing_unchanged",
            ),
            REFERENCE_EMBEDDING_SCOPE: (
                "reuse_reference_embedding",
                "reference_image_identity_unchanged",
            ),
        }[scope]
    if scope == DETECTOR_SCOPE:
        return "run_yoloe", "detector_or_input_cache_miss"
    if scope == FLICKR_EMBEDDING_SCOPE:
        return "embed_flickr_image", "image_model_or_preprocessing_cache_miss"
    if bool(row["newly_admitted"]):
        return "embed_new_reference_image", "newly_admitted_reference"
    return "reembed_changed_reference_image", "reference_embedding_cache_miss"


def _content_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        _feature_scope(row["feature_scope"]),
        _sha256(
            row["input_content_fingerprint"],
            field="input_content_fingerprint",
        ),
        _sha256(row["producer_fingerprint"], field="producer_fingerprint"),
        _sha256(
            row["preprocessing_fingerprint"],
            field="preprocessing_fingerprint",
        ),
    )


def _feature_scope(value: object) -> str:
    scope = _required_text(value, field="feature_scope")
    if scope not in FEATURE_SCOPES:
        raise ValueError(f"unsupported feature scope: {scope!r}")
    return scope


def _validate_row_fingerprint(
    row: Mapping[str, object],
    *,
    fingerprint_field: str,
    label: str,
) -> None:
    payload = dict(row)
    fingerprint = payload.pop(fingerprint_field)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError(f"{label} fingerprint mismatch")


def _reject_nulls(frame: pl.DataFrame, fields: tuple[str, ...]) -> None:
    if any(frame[field].null_count() for field in fields):
        raise ValueError("feature reuse artifacts contain null required fields")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "DETECTOR_SCOPE",
    "FEATURE_CACHE_ENTRY_SCHEMA",
    "FEATURE_CACHE_ENTRY_SCHEMA_VERSION",
    "FEATURE_REUSE_REQUEST_SCHEMA",
    "FEATURE_REUSE_REQUEST_SCHEMA_VERSION",
    "FEATURE_SCOPES",
    "FLICKR_EMBEDDING_SCOPE",
    "INCREMENTAL_FEATURE_REUSE_PLAN_FILE",
    "INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION",
    "REFERENCE_EMBEDDING_SCOPE",
    "calculate_incremental_feature_reuse",
    "feature_cache_entry_frame",
    "feature_item_ids_to_compute",
    "feature_reuse_request_frame",
    "incremental_feature_reuse_metrics",
    "incremental_feature_reuse_plan_schema",
    "validate_feature_cache_entries",
    "validate_feature_reuse_requests",
    "validate_incremental_feature_reuse_plan",
    "write_incremental_feature_reuse_plan",
]
