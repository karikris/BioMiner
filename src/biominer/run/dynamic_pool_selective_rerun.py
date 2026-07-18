"""Verified evidence reuse and bounded execution after dynamic-pool revisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_revision_impact import (
    DynamicReferenceRevision,
    validate_dynamic_matrix_revision_impact,
    validate_dynamic_pool_revision_impact,
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
DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION = "dynamic-selective-rerun-plan-v1.0.0"
DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION = "dynamic-selective-rerun-receipt-v1.0.0"

_EXECUTION_ACTIONS = frozenset(
    {
        "embed_new_reference_image",
        "reembed_changed_reference_image",
        "embed_flickr_image",
        "rebuild_reference_pool",
        "rebuild_matrix",
        "rescore_record_from_reused_flickr_embedding",
    }
)
_REUSE_ACTIONS = frozenset(
    {
        "reuse_reference_embedding",
        "reuse_flickr_embedding",
        "reuse_pool_without_rebuild",
        "reuse_matrix_without_materialization",
        "reuse_scoring_record_without_recomputation",
    }
)
_EXCLUSION_ACTIONS = frozenset({"filter_excluded_reference", "skip_excluded_reference"})
_ACTIONS_BY_KIND = {
    "reference_embedding": frozenset(
        {
            "reuse_reference_embedding",
            "embed_new_reference_image",
            "reembed_changed_reference_image",
            "filter_excluded_reference",
            "skip_excluded_reference",
        }
    ),
    "flickr_embedding": frozenset({"reuse_flickr_embedding", "embed_flickr_image"}),
    "reference_pool": frozenset(
        {"rebuild_reference_pool", "reuse_pool_without_rebuild"}
    ),
    "scoring_matrix": frozenset(
        {"rebuild_matrix", "reuse_matrix_without_materialization"}
    ),
    "scoring_record": frozenset(
        {
            "rescore_record_from_reused_flickr_embedding",
            "reuse_scoring_record_without_recomputation",
        }
    ),
}
_STAGE_BY_KIND = {
    "reference_embedding": 0,
    "flickr_embedding": 0,
    "reference_pool": 1,
    "scoring_matrix": 2,
    "scoring_record": 3,
}


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


def dynamic_selective_rerun_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "operation_fingerprint": pl.String,
        "operation_id": pl.String,
        "artifact_kind": pl.String,
        "artifact_id": pl.String,
        "action": pl.String,
        "execution_stage": pl.UInt8,
        "execution_required": pl.Boolean,
        "dependency_operation_ids": pl.List(pl.String),
        "evidence_fingerprints": pl.List(pl.String),
    }


def dynamic_selective_rerun_receipt_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "plan_fingerprint": pl.String,
        "receipt_fingerprint": pl.String,
        "operation_id": pl.String,
        "artifact_kind": pl.String,
        "artifact_id": pl.String,
        "action": pl.String,
        "execution_stage": pl.UInt8,
        "execution_required": pl.Boolean,
        "status": pl.String,
        "output_artifact_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class DynamicSelectiveRerunPlan:
    """Complete topological plan over reused and recomputed artifacts."""

    table: pl.DataFrame
    plan_fingerprint: str
    operation_ids_to_execute: tuple[str, ...]
    operation_ids_reused: tuple[str, ...]
    operation_ids_excluded: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DynamicSelectiveRerunReceipt:
    """Measured execution outcomes for one immutable selective plan."""

    table: pl.DataFrame
    receipt_fingerprint: str
    executed_operation_ids: tuple[str, ...]


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


def build_dynamic_selective_rerun_plan(
    revision: DynamicReferenceRevision,
    pool_impacts: pl.DataFrame,
    matrix_impacts: pl.DataFrame,
    scoring_impacts: pl.DataFrame,
    reference_reuse: DynamicReferenceEmbeddingReuseProjection,
    flickr_reuse: DynamicFlickrEmbeddingReuseProjection,
) -> DynamicSelectiveRerunPlan:
    """Join exact impacts and cache evidence into one bounded execution DAG."""

    validate_dynamic_pool_revision_impact(pool_impacts)
    validate_dynamic_matrix_revision_impact(matrix_impacts)
    validate_dynamic_scoring_record_impact(scoring_impacts)
    for frame in (pool_impacts, matrix_impacts, scoring_impacts):
        if frame["revision_fingerprint"].unique().to_list() != [revision.fingerprint]:
            raise ValueError("selective rerun received evidence from another revision")
    if not isinstance(
        reference_reuse, DynamicReferenceEmbeddingReuseProjection
    ) or not isinstance(flickr_reuse, DynamicFlickrEmbeddingReuseProjection):
        raise TypeError("selective rerun requires typed embedding reuse projections")
    validate_dynamic_reference_embedding_reuse(
        reference_reuse.table,
        revision=revision,
    )
    validate_dynamic_flickr_embedding_reuse(
        flickr_reuse.table,
        revision=revision,
        scoring_impacts=scoring_impacts,
    )

    reference_ids = set(reference_reuse.table["reference_media_id"])
    flickr_ids = set(flickr_reuse.table["flickr_embedding_fingerprint"])
    pool_ids = set(pool_impacts["plan_id"])
    matrix_ids = set(matrix_impacts["matrix_id"])
    rows: list[dict[str, object]] = []
    for row in reference_reuse.table.iter_rows(named=True):
        rows.append(
            _selective_operation_row(
                revision_fingerprint=revision.fingerprint,
                artifact_kind="reference_embedding",
                artifact_id=str(row["reference_media_id"]),
                action=str(row["action"]),
                dependency_operation_ids=(),
                evidence_fingerprints=(str(row["decision_fingerprint"]),),
            )
        )
    for row in flickr_reuse.table.iter_rows(named=True):
        rows.append(
            _selective_operation_row(
                revision_fingerprint=revision.fingerprint,
                artifact_kind="flickr_embedding",
                artifact_id=str(row["flickr_embedding_fingerprint"]),
                action=str(row["action"]),
                dependency_operation_ids=(),
                evidence_fingerprints=(str(row["decision_fingerprint"]),),
            )
        )
    for row in pool_impacts.iter_rows(named=True):
        affected_references = set(row["affected_reference_media_ids"])
        unknown_references = sorted(affected_references - reference_ids)
        if unknown_references:
            raise ValueError(
                "pool rebuild lacks reference reuse decisions: "
                + ", ".join(unknown_references)
            )
        dependencies = tuple(
            _selective_operation_id("reference_embedding", reference_id)
            for reference_id in sorted(affected_references)
        )
        rows.append(
            _selective_operation_row(
                revision_fingerprint=revision.fingerprint,
                artifact_kind="reference_pool",
                artifact_id=str(row["plan_id"]),
                action=str(row["expected_action"]),
                dependency_operation_ids=dependencies,
                evidence_fingerprints=(str(row["impact_fingerprint"]),),
            )
        )
    for row in matrix_impacts.iter_rows(named=True):
        affected_references = set(row["affected_reference_media_ids"])
        affected_pools = set(row["affected_plan_ids"])
        if affected_references - reference_ids or affected_pools - pool_ids:
            raise ValueError("matrix rebuild lacks declared upstream decisions")
        dependencies = tuple(
            sorted(
                [
                    *(
                        _selective_operation_id("reference_embedding", item)
                        for item in affected_references
                    ),
                    *(
                        _selective_operation_id("reference_pool", item)
                        for item in affected_pools
                    ),
                ]
            )
        )
        rows.append(
            _selective_operation_row(
                revision_fingerprint=revision.fingerprint,
                artifact_kind="scoring_matrix",
                artifact_id=str(row["matrix_id"]),
                action=str(row["expected_action"]),
                dependency_operation_ids=dependencies,
                evidence_fingerprints=(str(row["impact_fingerprint"]),),
            )
        )
    for row in scoring_impacts.iter_rows(named=True):
        embedding_fingerprint = str(row["flickr_embedding_fingerprint"])
        affected_pools = set(row["affected_plan_ids"])
        affected_matrices = set(row["affected_matrix_ids"])
        if (
            embedding_fingerprint not in flickr_ids
            or affected_pools - pool_ids
            or affected_matrices - matrix_ids
        ):
            raise ValueError("score rerun lacks declared upstream decisions")
        dependencies: tuple[str, ...] = ()
        if row["impact_status"] == "affected":
            dependencies = tuple(
                sorted(
                    [
                        _selective_operation_id(
                            "flickr_embedding", embedding_fingerprint
                        ),
                        *(
                            _selective_operation_id("reference_pool", item)
                            for item in affected_pools
                        ),
                        *(
                            _selective_operation_id("scoring_matrix", item)
                            for item in affected_matrices
                        ),
                    ]
                )
            )
        rows.append(
            _selective_operation_row(
                revision_fingerprint=revision.fingerprint,
                artifact_kind="scoring_record",
                artifact_id=str(row["scoring_record_id"]),
                action=str(row["expected_action"]),
                dependency_operation_ids=dependencies,
                evidence_fingerprints=(str(row["impact_fingerprint"]),),
            )
        )

    table = pl.DataFrame(
        rows,
        schema=dynamic_selective_rerun_plan_schema(),
        orient="row",
        strict=True,
    ).sort("execution_stage", "operation_id")
    validate_dynamic_selective_rerun_plan(table, revision=revision)
    plan_fingerprint = _selective_plan_fingerprint(table, revision=revision)
    execute_ids = tuple(table.filter(pl.col("execution_required"))["operation_id"])
    reuse_ids = tuple(
        table.filter(pl.col("action").is_in(sorted(_REUSE_ACTIONS)))["operation_id"]
    )
    excluded_ids = tuple(
        table.filter(pl.col("action").is_in(sorted(_EXCLUSION_ACTIONS)))["operation_id"]
    )
    return DynamicSelectiveRerunPlan(
        table=table,
        plan_fingerprint=plan_fingerprint,
        operation_ids_to_execute=execute_ids,
        operation_ids_reused=reuse_ids,
        operation_ids_excluded=excluded_ids,
    )


def validate_dynamic_selective_rerun_plan(
    table: pl.DataFrame,
    *,
    revision: DynamicReferenceRevision,
) -> None:
    if not isinstance(revision, DynamicReferenceRevision):
        raise TypeError("revision must be a DynamicReferenceRevision")
    _validate_dynamic_selective_rerun_plan_table(
        table,
        revision_fingerprint=revision.fingerprint,
    )


def _validate_dynamic_selective_rerun_plan_table(
    table: pl.DataFrame,
    *,
    revision_fingerprint: str,
) -> None:
    if table.schema != dynamic_selective_rerun_plan_schema():
        raise ValueError("dynamic selective rerun plan schema mismatch")
    if table.is_empty() or table["operation_id"].n_unique() != table.height:
        raise ValueError("dynamic selective rerun operation identities are invalid")
    if not table.equals(table.sort("execution_stage", "operation_id")):
        raise ValueError("dynamic selective rerun plan is not topologically sorted")
    if table["revision_fingerprint"].unique().to_list() != [revision_fingerprint]:
        raise ValueError("dynamic selective rerun plan revision mismatch")
    operations = {str(row["operation_id"]): row for row in table.iter_rows(named=True)}
    for row in operations.values():
        kind = str(row["artifact_kind"])
        action = str(row["action"])
        if kind not in _ACTIONS_BY_KIND or action not in _ACTIONS_BY_KIND[kind]:
            raise ValueError("dynamic selective rerun action/kind mismatch")
        if row["operation_id"] != _selective_operation_id(
            kind, str(row["artifact_id"])
        ):
            raise ValueError("dynamic selective rerun operation ID mismatch")
        if int(row["execution_stage"]) != _STAGE_BY_KIND[kind]:
            raise ValueError("dynamic selective rerun execution stage mismatch")
        if bool(row["execution_required"]) != (action in _EXECUTION_ACTIONS):
            raise ValueError("dynamic selective rerun execution decision mismatch")
        dependencies = list(row["dependency_operation_ids"])
        if dependencies != sorted(set(dependencies)):
            raise ValueError("dynamic selective rerun dependencies are not canonical")
        for dependency_id in dependencies:
            dependency = operations.get(str(dependency_id))
            if dependency is None:
                raise ValueError("dynamic selective rerun dependency is unknown")
            if int(dependency["execution_stage"]) >= int(row["execution_stage"]):
                raise ValueError("dynamic selective rerun dependency order is invalid")
        if not row["evidence_fingerprints"]:
            raise ValueError("dynamic selective rerun operation lacks evidence")
        base = {
            field: row[field]
            for field in dynamic_selective_rerun_plan_schema()
            if field not in {"schema_version", "operation_fingerprint"}
        }
        if row["operation_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("dynamic selective rerun operation fingerprint mismatch")


def execute_dynamic_selective_rerun(
    plan: DynamicSelectiveRerunPlan,
    *,
    executors: Mapping[
        str,
        Callable[[Mapping[str, object]], str],
    ],
) -> DynamicSelectiveRerunReceipt:
    """Execute only required operations after preflighting the complete plan."""

    if not isinstance(plan, DynamicSelectiveRerunPlan):
        raise TypeError("plan must be a DynamicSelectiveRerunPlan")
    if (
        plan.table.schema != dynamic_selective_rerun_plan_schema()
        or plan.table.is_empty()
    ):
        raise ValueError("dynamic selective rerun plan schema or rows are invalid")
    revision_fingerprint = str(plan.table.item(0, "revision_fingerprint"))
    _validate_dynamic_selective_rerun_plan_table(
        plan.table,
        revision_fingerprint=revision_fingerprint,
    )
    expected_plan_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION,
            "revision_fingerprint": revision_fingerprint,
            "operation_fingerprints": plan.table["operation_fingerprint"].to_list(),
        }
    )
    if plan.plan_fingerprint != expected_plan_fingerprint:
        raise ValueError("dynamic selective rerun plan fingerprint mismatch")
    if (
        plan.operation_ids_to_execute
        != tuple(plan.table.filter(pl.col("execution_required"))["operation_id"])
        or plan.operation_ids_reused
        != tuple(
            plan.table.filter(pl.col("action").is_in(sorted(_REUSE_ACTIONS)))[
                "operation_id"
            ]
        )
        or plan.operation_ids_excluded
        != tuple(
            plan.table.filter(pl.col("action").is_in(sorted(_EXCLUSION_ACTIONS)))[
                "operation_id"
            ]
        )
    ):
        raise ValueError("dynamic selective rerun projection summary mismatch")
    required_actions = set(plan.table.filter(pl.col("execution_required"))["action"])
    missing = sorted(
        action
        for action in required_actions
        if action not in executors or not callable(executors[action])
    )
    if missing:
        raise ValueError(
            "dynamic selective rerun executors are missing: " + ", ".join(missing)
        )

    outputs: dict[str, str] = {}
    for row in plan.table.filter(pl.col("execution_required")).iter_rows(named=True):
        output = executors[str(row["action"])](row)
        outputs[str(row["operation_id"])] = _sha256(
            output,
            field="selective rerun output artifact fingerprint",
        )

    rows: list[dict[str, object]] = []
    for operation in plan.table.iter_rows(named=True):
        operation_id = str(operation["operation_id"])
        action = str(operation["action"])
        if operation["execution_required"]:
            status = "materialized"
            output = outputs[operation_id]
        elif action in _REUSE_ACTIONS:
            status = "reused_without_execution"
            output = None
        else:
            status = "excluded_without_execution"
            output = None
        base = {
            "revision_fingerprint": revision_fingerprint,
            "plan_fingerprint": plan.plan_fingerprint,
            "operation_id": operation_id,
            "artifact_kind": operation["artifact_kind"],
            "artifact_id": operation["artifact_id"],
            "action": action,
            "execution_stage": operation["execution_stage"],
            "execution_required": operation["execution_required"],
            "status": status,
            "output_artifact_fingerprint": output,
        }
        rows.append(
            {
                "schema_version": DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION,
                "receipt_fingerprint": canonical_semantic_fingerprint(base),
                **base,
            }
        )
    table = pl.DataFrame(
        rows,
        schema=dynamic_selective_rerun_receipt_schema(),
        orient="row",
        strict=True,
    ).sort("execution_stage", "operation_id")
    receipt_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION,
            "revision_fingerprint": revision_fingerprint,
            "plan_fingerprint": plan.plan_fingerprint,
            "receipt_fingerprints": table["receipt_fingerprint"].to_list(),
        }
    )
    executed = tuple(table.filter(pl.col("execution_required"))["operation_id"])
    receipt = DynamicSelectiveRerunReceipt(
        table=table,
        receipt_fingerprint=receipt_fingerprint,
        executed_operation_ids=executed,
    )
    validate_dynamic_selective_rerun_receipt(receipt, plan=plan)
    return receipt


def validate_dynamic_selective_rerun_receipt(
    receipt: DynamicSelectiveRerunReceipt,
    *,
    plan: DynamicSelectiveRerunPlan,
) -> None:
    if not isinstance(receipt, DynamicSelectiveRerunReceipt):
        raise TypeError("receipt must be a DynamicSelectiveRerunReceipt")
    if not isinstance(plan, DynamicSelectiveRerunPlan):
        raise TypeError("plan must be a DynamicSelectiveRerunPlan")
    if (
        plan.table.schema != dynamic_selective_rerun_plan_schema()
        or plan.table.is_empty()
    ):
        raise ValueError("dynamic selective rerun plan schema or rows are invalid")
    revision_fingerprint = str(plan.table.item(0, "revision_fingerprint"))
    _validate_dynamic_selective_rerun_plan_table(
        plan.table,
        revision_fingerprint=revision_fingerprint,
    )
    expected_plan_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION,
            "revision_fingerprint": revision_fingerprint,
            "operation_fingerprints": plan.table["operation_fingerprint"].to_list(),
        }
    )
    if plan.plan_fingerprint != expected_plan_fingerprint:
        raise ValueError("dynamic selective rerun plan fingerprint mismatch")
    table = receipt.table
    if table.schema != dynamic_selective_rerun_receipt_schema():
        raise ValueError("dynamic selective rerun receipt schema mismatch")
    if table.height != plan.table.height or not table.equals(
        table.sort("execution_stage", "operation_id")
    ):
        raise ValueError("dynamic selective rerun receipt rows are invalid")
    operations = {
        str(row["operation_id"]): row for row in plan.table.iter_rows(named=True)
    }
    if set(table["operation_id"]) != set(operations):
        raise ValueError("dynamic selective rerun receipt operation mismatch")
    for row in table.iter_rows(named=True):
        operation = operations[str(row["operation_id"])]
        expected_status = (
            "materialized"
            if operation["execution_required"]
            else (
                "reused_without_execution"
                if operation["action"] in _REUSE_ACTIONS
                else "excluded_without_execution"
            )
        )
        if (
            row["schema_version"] != DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION
            or row["revision_fingerprint"] != operation["revision_fingerprint"]
            or row["plan_fingerprint"] != plan.plan_fingerprint
            or row["artifact_kind"] != operation["artifact_kind"]
            or row["artifact_id"] != operation["artifact_id"]
            or row["action"] != operation["action"]
            or row["execution_stage"] != operation["execution_stage"]
            or row["execution_required"] != operation["execution_required"]
            or row["status"] != expected_status
        ):
            raise ValueError("dynamic selective rerun receipt evidence mismatch")
        output = row["output_artifact_fingerprint"]
        if operation["execution_required"]:
            _sha256(output, field="selective rerun output artifact fingerprint")
        elif output is not None:
            raise ValueError("non-executed selective operation has an output artifact")
        base = {
            field: row[field]
            for field in dynamic_selective_rerun_receipt_schema()
            if field not in {"schema_version", "receipt_fingerprint"}
        }
        if row["receipt_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("dynamic selective rerun receipt fingerprint mismatch")
    expected_receipt_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION,
            "revision_fingerprint": table.item(0, "revision_fingerprint"),
            "plan_fingerprint": plan.plan_fingerprint,
            "receipt_fingerprints": table["receipt_fingerprint"].to_list(),
        }
    )
    if (
        receipt.receipt_fingerprint != expected_receipt_fingerprint
        or receipt.executed_operation_ids
        != tuple(table.filter(pl.col("execution_required"))["operation_id"])
    ):
        raise ValueError("dynamic selective rerun receipt summary mismatch")


def dynamic_selective_rerun_metrics(plan: DynamicSelectiveRerunPlan) -> pl.DataFrame:
    """Report exact planned work counts; runtime savings remain unestimated."""

    if not isinstance(plan, DynamicSelectiveRerunPlan):
        raise TypeError("plan must be a DynamicSelectiveRerunPlan")
    return (
        plan.table.group_by("artifact_kind", "action")
        .agg(
            pl.len().cast(pl.UInt64).alias("operation_count"),
            pl.col("execution_required")
            .sum()
            .cast(pl.UInt64)
            .alias("execution_required_count"),
        )
        .with_columns(
            (pl.col("operation_count") - pl.col("execution_required_count")).alias(
                "reuse_or_exclusion_count"
            ),
            pl.lit(None, dtype=pl.Float64).alias("estimated_runtime_savings_seconds"),
            pl.lit("not_instrumented").alias("runtime_savings_status"),
        )
        .sort("artifact_kind", "action")
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


def _selective_operation_row(
    *,
    revision_fingerprint: str,
    artifact_kind: str,
    artifact_id: str,
    action: str,
    dependency_operation_ids: tuple[str, ...],
    evidence_fingerprints: tuple[str, ...],
) -> dict[str, object]:
    base = {
        "revision_fingerprint": revision_fingerprint,
        "operation_id": _selective_operation_id(artifact_kind, artifact_id),
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "action": action,
        "execution_stage": _STAGE_BY_KIND[artifact_kind],
        "execution_required": action in _EXECUTION_ACTIONS,
        "dependency_operation_ids": sorted(set(dependency_operation_ids)),
        "evidence_fingerprints": sorted(set(evidence_fingerprints)),
    }
    return {
        "schema_version": DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION,
        "operation_fingerprint": canonical_semantic_fingerprint(base),
        **base,
    }


def _selective_operation_id(artifact_kind: str, artifact_id: str) -> str:
    fingerprint = canonical_semantic_fingerprint(
        {"artifact_kind": artifact_kind, "artifact_id": artifact_id}
    )
    return f"dynamic-rerun:{fingerprint.removeprefix('sha256:')}"


def _selective_plan_fingerprint(
    table: pl.DataFrame,
    *,
    revision: DynamicReferenceRevision,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION,
            "revision_fingerprint": revision.fingerprint,
            "operation_fingerprints": table["operation_fingerprint"].to_list(),
        }
    )


def _sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip().casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


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
    "DYNAMIC_SELECTIVE_RERUN_PLAN_VERSION",
    "DYNAMIC_SELECTIVE_RERUN_RECEIPT_VERSION",
    "DynamicFlickrEmbeddingReuseProjection",
    "DynamicReferenceEmbeddingReuseProjection",
    "DynamicSelectiveRerunPlan",
    "DynamicSelectiveRerunReceipt",
    "build_dynamic_selective_rerun_plan",
    "dynamic_flickr_embedding_reuse_schema",
    "dynamic_reference_embedding_reuse_schema",
    "dynamic_selective_rerun_metrics",
    "dynamic_selective_rerun_plan_schema",
    "dynamic_selective_rerun_receipt_schema",
    "execute_dynamic_selective_rerun",
    "plan_dynamic_flickr_embedding_reuse",
    "plan_dynamic_reference_embedding_reuse",
    "validate_dynamic_flickr_embedding_reuse",
    "validate_dynamic_reference_embedding_reuse",
    "validate_dynamic_selective_rerun_plan",
    "validate_dynamic_selective_rerun_receipt",
]
