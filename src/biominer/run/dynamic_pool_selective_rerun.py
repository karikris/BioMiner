"""Verified evidence reuse and bounded execution after dynamic-pool revisions."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_revision_impact import DynamicReferenceRevision
from biominer.run.incremental_feature_reuse import (
    INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION,
    REFERENCE_EMBEDDING_SCOPE,
    calculate_incremental_feature_reuse,
    incremental_feature_reuse_plan_schema,
    validate_feature_cache_entries,
    validate_feature_reuse_requests,
    validate_incremental_feature_reuse_plan,
)


DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION = "dynamic-reference-embedding-reuse-v1.0.0"


def dynamic_reference_embedding_reuse_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "decision_fingerprint": pl.String,
        "reference_media_id": pl.String,
        "revision_change_type": pl.String,
        "reference_change_fingerprint": pl.String,
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
        "feature_reuse_plan_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class DynamicReferenceEmbeddingReuseProjection:
    """Complete reference-vector decisions for one bank revision."""

    table: pl.DataFrame
    projection_fingerprint: str
    reused_reference_media_ids: tuple[str, ...]
    reference_media_ids_to_embed: tuple[str, ...]
    excluded_reference_media_ids: tuple[str, ...]


def plan_dynamic_reference_embedding_reuse(
    revision: DynamicReferenceRevision,
    requests: pl.DataFrame,
    cache_entries: pl.DataFrame,
) -> DynamicReferenceEmbeddingReuseProjection:
    """Bind content-addressed reference-vector reuse to a complete revision."""

    if not isinstance(revision, DynamicReferenceRevision):
        raise TypeError("revision must be a DynamicReferenceRevision")
    validate_feature_reuse_requests(requests)
    validate_feature_cache_entries(cache_entries)
    if requests.is_empty() or set(requests["feature_scope"]) != {
        REFERENCE_EMBEDDING_SCOPE
    }:
        raise ValueError("dynamic reference reuse requires reference-only requests")

    changes = {change.reference_media_id: change for change in revision.changes}
    request_ids = set(requests["item_id"])
    missing = sorted(set(changes) - request_ids)
    if missing:
        raise ValueError(
            "dynamic reference reuse is missing revision changes: " + ", ".join(missing)
        )

    generic_plan = calculate_incremental_feature_reuse(requests, cache_entries)
    rows: list[dict[str, object]] = []
    for plan_row in generic_plan.iter_rows(named=True):
        reference_media_id = str(plan_row["item_id"])
        change = changes.get(reference_media_id)
        change_type = change.change_type if change is not None else "unchanged"
        _validate_reference_requirement_flags(
            change_type,
            required=bool(plan_row["required"]),
            newly_admitted=bool(plan_row["newly_admitted"]),
        )
        base = {
            "revision_fingerprint": revision.fingerprint,
            "reference_media_id": reference_media_id,
            "revision_change_type": change_type,
            "reference_change_fingerprint": (
                change.fingerprint if change is not None else None
            ),
            "input_content_fingerprint": plan_row["input_content_fingerprint"],
            "producer_fingerprint": plan_row["producer_fingerprint"],
            "preprocessing_fingerprint": plan_row["preprocessing_fingerprint"],
            "required": plan_row["required"],
            "newly_admitted": plan_row["newly_admitted"],
            "request_fingerprint": plan_row["request_fingerprint"],
            "cache_hit": plan_row["cache_hit"],
            "cache_entry_id": plan_row["cache_entry_id"],
            "reusable_artifact_id": plan_row["reusable_artifact_id"],
            "reusable_artifact_fingerprint": plan_row["reusable_artifact_fingerprint"],
            "action": plan_row["action"],
            "reason": plan_row["reason"],
            "feature_reuse_plan_fingerprint": plan_row["plan_fingerprint"],
        }
        rows.append(
            {
                "schema_version": DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION,
                "decision_fingerprint": canonical_semantic_fingerprint(base),
                **base,
            }
        )

    table = pl.DataFrame(
        rows,
        schema=dynamic_reference_embedding_reuse_schema(),
        orient="row",
        strict=True,
    ).sort("reference_media_id")
    validate_dynamic_reference_embedding_reuse(table, revision=revision)
    reused = tuple(
        table.filter(pl.col("action") == "reuse_reference_embedding")[
            "reference_media_id"
        ]
    )
    to_embed = tuple(
        table.filter(
            pl.col("action").is_in(
                ["embed_new_reference_image", "reembed_changed_reference_image"]
            )
        )["reference_media_id"]
    )
    excluded = tuple(
        table.filter(
            pl.col("action").is_in(
                ["filter_excluded_reference", "skip_excluded_reference"]
            )
        )["reference_media_id"]
    )
    projection_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION,
            "revision_fingerprint": revision.fingerprint,
            "decision_fingerprints": table["decision_fingerprint"].to_list(),
            "reused_reference_media_ids": reused,
            "reference_media_ids_to_embed": to_embed,
            "excluded_reference_media_ids": excluded,
        }
    )
    return DynamicReferenceEmbeddingReuseProjection(
        table=table,
        projection_fingerprint=projection_fingerprint,
        reused_reference_media_ids=reused,
        reference_media_ids_to_embed=to_embed,
        excluded_reference_media_ids=excluded,
    )


def validate_dynamic_reference_embedding_reuse(
    table: pl.DataFrame,
    *,
    revision: DynamicReferenceRevision,
) -> None:
    if table.schema != dynamic_reference_embedding_reuse_schema():
        raise ValueError("dynamic reference embedding reuse schema mismatch")
    if table.is_empty() or table["reference_media_id"].n_unique() != table.height:
        raise ValueError("dynamic reference embedding reuse identities are invalid")
    if not table.equals(table.sort("reference_media_id")):
        raise ValueError("dynamic reference embedding reuse is not sorted")
    if table["revision_fingerprint"].unique().to_list() != [revision.fingerprint]:
        raise ValueError("dynamic reference embedding reuse revision mismatch")

    generic = table.select(
        pl.lit(INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION).alias("schema_version"),
        pl.lit(REFERENCE_EMBEDDING_SCOPE).alias("feature_scope"),
        pl.col("reference_media_id").alias("item_id"),
        "input_content_fingerprint",
        "producer_fingerprint",
        "preprocessing_fingerprint",
        "required",
        "newly_admitted",
        "request_fingerprint",
        "cache_hit",
        "cache_entry_id",
        "reusable_artifact_id",
        "reusable_artifact_fingerprint",
        "action",
        "reason",
        pl.col("feature_reuse_plan_fingerprint").alias("plan_fingerprint"),
    )
    generic = generic.cast(incremental_feature_reuse_plan_schema())
    validate_incremental_feature_reuse_plan(generic)

    changes = {change.reference_media_id: change for change in revision.changes}
    if not set(changes).issubset(set(table["reference_media_id"])):
        raise ValueError("dynamic reference embedding reuse omits a revision change")
    for row in table.iter_rows(named=True):
        change = changes.get(str(row["reference_media_id"]))
        expected_type = change.change_type if change is not None else "unchanged"
        expected_fingerprint = change.fingerprint if change is not None else None
        if (
            row["schema_version"] != DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION
            or row["revision_change_type"] != expected_type
            or row["reference_change_fingerprint"] != expected_fingerprint
        ):
            raise ValueError("dynamic reference embedding revision evidence mismatch")
        _validate_reference_requirement_flags(
            expected_type,
            required=bool(row["required"]),
            newly_admitted=bool(row["newly_admitted"]),
        )
        base = {
            field: row[field]
            for field in dynamic_reference_embedding_reuse_schema()
            if field not in {"schema_version", "decision_fingerprint"}
        }
        if row["decision_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError(
                "dynamic reference embedding decision fingerprint mismatch"
            )


def _validate_reference_requirement_flags(
    change_type: str,
    *,
    required: bool,
    newly_admitted: bool,
) -> None:
    expected = {
        "unchanged": (True, False),
        "added": (True, True),
        "modified": (True, False),
        "removed": (False, False),
    }
    if (
        change_type not in expected
        or (required, newly_admitted) != expected[change_type]
    ):
        raise ValueError(
            "reference reuse requirement flags do not match revision change type"
        )


__all__ = [
    "DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION",
    "DynamicReferenceEmbeddingReuseProjection",
    "dynamic_reference_embedding_reuse_schema",
    "plan_dynamic_reference_embedding_reuse",
    "validate_dynamic_reference_embedding_reuse",
]
