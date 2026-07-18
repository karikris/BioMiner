"""Verified evidence reuse and bounded execution after dynamic-pool revisions."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_revision_impact import (
    DynamicReferenceRevision,
    validate_dynamic_scoring_record_impact,
)
from biominer.run.incremental_feature_reuse import (
    FLICKR_EMBEDDING_SCOPE,
    INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION,
    REFERENCE_EMBEDDING_SCOPE,
    calculate_incremental_feature_reuse,
    incremental_feature_reuse_plan_schema,
    validate_feature_cache_entries,
    validate_feature_reuse_requests,
    validate_incremental_feature_reuse_plan,
)


DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION = "dynamic-reference-embedding-reuse-v1.0.0"
DYNAMIC_FLICKR_EMBEDDING_REUSE_VERSION = "dynamic-flickr-embedding-reuse-v1.0.0"


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


def dynamic_flickr_embedding_reuse_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "decision_fingerprint": pl.String,
        "flickr_embedding_fingerprint": pl.String,
        "source_image_sha256": pl.String,
        "scoring_record_ids": pl.List(pl.String),
        "affected_scoring_record_ids": pl.List(pl.String),
        "reusable_scoring_record_ids": pl.List(pl.String),
        "input_content_fingerprint": pl.String,
        "producer_fingerprint": pl.String,
        "preprocessing_fingerprint": pl.String,
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
class DynamicFlickrEmbeddingReuseProjection:
    """Unique Flickr-vector decisions with complete score-row bindings."""

    table: pl.DataFrame
    projection_fingerprint: str
    reused_embedding_fingerprints: tuple[str, ...]
    embedding_fingerprints_to_materialize: tuple[str, ...]


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


def plan_dynamic_flickr_embedding_reuse(
    revision: DynamicReferenceRevision,
    scoring_impacts: pl.DataFrame,
    requests: pl.DataFrame,
    cache_entries: pl.DataFrame,
) -> DynamicFlickrEmbeddingReuseProjection:
    """Verify one reusable Flickr vector per exact scoring dependency identity."""

    if not isinstance(revision, DynamicReferenceRevision):
        raise TypeError("revision must be a DynamicReferenceRevision")
    validate_dynamic_scoring_record_impact(scoring_impacts)
    if scoring_impacts["revision_fingerprint"].unique().to_list() != [
        revision.fingerprint
    ]:
        raise ValueError("Flickr reuse received impacts from another revision")
    validate_feature_reuse_requests(requests)
    validate_feature_cache_entries(cache_entries)
    if requests.is_empty() or set(requests["feature_scope"]) != {
        FLICKR_EMBEDDING_SCOPE
    }:
        raise ValueError("dynamic Flickr reuse requires Flickr-only requests")

    impacts_by_embedding = _scoring_impacts_by_embedding(scoring_impacts)
    expected_fingerprints = set(impacts_by_embedding)
    requested_fingerprints = set(requests["item_id"])
    if requested_fingerprints != expected_fingerprints:
        missing = sorted(expected_fingerprints - requested_fingerprints)
        extra = sorted(requested_fingerprints - expected_fingerprints)
        raise ValueError(
            "dynamic Flickr reuse request inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )

    generic_plan = calculate_incremental_feature_reuse(requests, cache_entries)
    rows: list[dict[str, object]] = []
    for plan_row in generic_plan.iter_rows(named=True):
        embedding_fingerprint = str(plan_row["item_id"])
        impacts = impacts_by_embedding[embedding_fingerprint]
        source_hashes = {str(row["source_image_sha256"]) for row in impacts}
        if len(source_hashes) != 1:
            raise ValueError("one Flickr embedding is bound to several source images")
        source_image_sha256 = source_hashes.pop()
        if plan_row["input_content_fingerprint"] != source_image_sha256:
            raise ValueError("Flickr embedding request source identity mismatch")
        if plan_row["cache_hit"] and (
            plan_row["reusable_artifact_fingerprint"] != embedding_fingerprint
        ):
            raise ValueError("Flickr cache artifact fingerprint mismatch")
        scoring_record_ids = sorted(str(row["scoring_record_id"]) for row in impacts)
        affected_ids = sorted(
            str(row["scoring_record_id"])
            for row in impacts
            if row["impact_status"] == "affected"
        )
        reusable_ids = sorted(set(scoring_record_ids) - set(affected_ids))
        base = {
            "revision_fingerprint": revision.fingerprint,
            "flickr_embedding_fingerprint": embedding_fingerprint,
            "source_image_sha256": source_image_sha256,
            "scoring_record_ids": scoring_record_ids,
            "affected_scoring_record_ids": affected_ids,
            "reusable_scoring_record_ids": reusable_ids,
            "input_content_fingerprint": plan_row["input_content_fingerprint"],
            "producer_fingerprint": plan_row["producer_fingerprint"],
            "preprocessing_fingerprint": plan_row["preprocessing_fingerprint"],
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
                "schema_version": DYNAMIC_FLICKR_EMBEDDING_REUSE_VERSION,
                "decision_fingerprint": canonical_semantic_fingerprint(base),
                **base,
            }
        )
    table = pl.DataFrame(
        rows,
        schema=dynamic_flickr_embedding_reuse_schema(),
        orient="row",
        strict=True,
    ).sort("flickr_embedding_fingerprint")
    validate_dynamic_flickr_embedding_reuse(
        table,
        revision=revision,
        scoring_impacts=scoring_impacts,
    )
    reused = tuple(
        table.filter(pl.col("action") == "reuse_flickr_embedding")[
            "flickr_embedding_fingerprint"
        ]
    )
    materialize = tuple(
        table.filter(pl.col("action") == "embed_flickr_image")[
            "flickr_embedding_fingerprint"
        ]
    )
    projection_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_FLICKR_EMBEDDING_REUSE_VERSION,
            "revision_fingerprint": revision.fingerprint,
            "decision_fingerprints": table["decision_fingerprint"].to_list(),
            "reused_embedding_fingerprints": reused,
            "embedding_fingerprints_to_materialize": materialize,
        }
    )
    return DynamicFlickrEmbeddingReuseProjection(
        table=table,
        projection_fingerprint=projection_fingerprint,
        reused_embedding_fingerprints=reused,
        embedding_fingerprints_to_materialize=materialize,
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


def validate_dynamic_flickr_embedding_reuse(
    table: pl.DataFrame,
    *,
    revision: DynamicReferenceRevision,
    scoring_impacts: pl.DataFrame,
) -> None:
    if table.schema != dynamic_flickr_embedding_reuse_schema():
        raise ValueError("dynamic Flickr embedding reuse schema mismatch")
    if (
        table.is_empty()
        or table["flickr_embedding_fingerprint"].n_unique() != table.height
    ):
        raise ValueError("dynamic Flickr embedding reuse identities are invalid")
    if not table.equals(table.sort("flickr_embedding_fingerprint")):
        raise ValueError("dynamic Flickr embedding reuse is not sorted")
    if table["revision_fingerprint"].unique().to_list() != [revision.fingerprint]:
        raise ValueError("dynamic Flickr embedding reuse revision mismatch")
    validate_dynamic_scoring_record_impact(scoring_impacts)
    impacts_by_embedding = _scoring_impacts_by_embedding(scoring_impacts)
    if set(table["flickr_embedding_fingerprint"]) != set(impacts_by_embedding):
        raise ValueError("dynamic Flickr embedding reuse omits scoring evidence")

    generic = table.select(
        pl.lit(INCREMENTAL_FEATURE_REUSE_PLAN_SCHEMA_VERSION).alias("schema_version"),
        pl.lit(FLICKR_EMBEDDING_SCOPE).alias("feature_scope"),
        pl.col("flickr_embedding_fingerprint").alias("item_id"),
        "input_content_fingerprint",
        "producer_fingerprint",
        "preprocessing_fingerprint",
        pl.lit(True).alias("required"),
        pl.lit(False).alias("newly_admitted"),
        "request_fingerprint",
        "cache_hit",
        "cache_entry_id",
        "reusable_artifact_id",
        "reusable_artifact_fingerprint",
        "action",
        "reason",
        pl.col("feature_reuse_plan_fingerprint").alias("plan_fingerprint"),
    ).cast(incremental_feature_reuse_plan_schema())
    validate_incremental_feature_reuse_plan(generic)

    for row in table.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_FLICKR_EMBEDDING_REUSE_VERSION:
            raise ValueError("unsupported dynamic Flickr embedding reuse version")
        embedding_fingerprint = str(row["flickr_embedding_fingerprint"])
        impacts = impacts_by_embedding[embedding_fingerprint]
        expected_records = sorted(str(item["scoring_record_id"]) for item in impacts)
        expected_affected = sorted(
            str(item["scoring_record_id"])
            for item in impacts
            if item["impact_status"] == "affected"
        )
        expected_reusable = sorted(set(expected_records) - set(expected_affected))
        expected_sources = {str(item["source_image_sha256"]) for item in impacts}
        if len(expected_sources) != 1:
            raise ValueError("one Flickr embedding is bound to several source images")
        if (
            row["scoring_record_ids"] != expected_records
            or row["affected_scoring_record_ids"] != expected_affected
            or row["reusable_scoring_record_ids"] != expected_reusable
            or row["source_image_sha256"] != expected_sources.pop()
            or row["input_content_fingerprint"] != row["source_image_sha256"]
        ):
            raise ValueError("dynamic Flickr embedding scoring evidence mismatch")
        if row["cache_hit"] and (
            row["reusable_artifact_fingerprint"] != embedding_fingerprint
        ):
            raise ValueError("Flickr cache artifact fingerprint mismatch")
        base = {
            field: row[field]
            for field in dynamic_flickr_embedding_reuse_schema()
            if field not in {"schema_version", "decision_fingerprint"}
        }
        if row["decision_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("dynamic Flickr embedding decision fingerprint mismatch")


def _scoring_impacts_by_embedding(
    scoring_impacts: pl.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in scoring_impacts.iter_rows(named=True):
        grouped.setdefault(str(row["flickr_embedding_fingerprint"]), []).append(row)
    return grouped


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
    "DYNAMIC_FLICKR_EMBEDDING_REUSE_VERSION",
    "DYNAMIC_REFERENCE_EMBEDDING_REUSE_VERSION",
    "DynamicFlickrEmbeddingReuseProjection",
    "DynamicReferenceEmbeddingReuseProjection",
    "dynamic_flickr_embedding_reuse_schema",
    "dynamic_reference_embedding_reuse_schema",
    "plan_dynamic_flickr_embedding_reuse",
    "plan_dynamic_reference_embedding_reuse",
    "validate_dynamic_flickr_embedding_reuse",
    "validate_dynamic_reference_embedding_reuse",
]
